"""Fresh-adapter training trajectories, immutable state lineage and model exports."""
import gc
from pathlib import Path

from .protocol import ROOT, append, config, digest, file_hash, lines, locked, model_identity, read, write, resource_lock
from . import runtime


def run_dir(workspace, seed, state=None, mode=None):
    return Path(workspace) / ('pre_rl' if state is None else 'runs') / (f'seed{seed}' if state is None else f'{state}_seed{seed}_{mode}')


def state_file(workspace, seed, state):
    return Path(workspace) / 'states' / f'seed{seed}' / f'{state}.json'


def ensure_base(workspace, seed, cfg):
    with resource_lock(workspace, 'base', digest(cfg['base_model'])):
        return _ensure_base(workspace, seed, cfg)


def _ensure_base(workspace, seed, cfg):
    from huggingface_hub import snapshot_download
    destination = state_file(workspace, seed, 'S0')
    if destination.exists():
        record = read(destination)
        if model_identity(record['path'])['sha256'] != record['model_sha256']:
            raise ValueError('S0 content changed')
        return record
    source = cfg['base_model']
    path = snapshot_download(repo_id=source['repo_id'], revision=source['revision'], cache_dir=str(ROOT / '.cache/huggingface'))
    record = {'state': 'S0', 'path': path, 'model_sha256': model_identity(path)['sha256'],
              'source': source, 'pre_rl_trajectory': None, 'pre_rl_step': 0, 'seed': seed}
    write(destination, record)
    return record


def load_state(workspace, seed, state):
    record = read(state_file(workspace, seed, state))
    if model_identity(record['path'])['sha256'] != record['model_sha256']:
        raise ValueError(f'{state} weights changed')
    return record


def fresh_components(base_path, cfg, seed, device='cuda'):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if (Path(base_path) / 'adapter_config.json').exists():
        raise ValueError('Starting state must be a merged full model, not a previous adapter')
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
    if getattr(base, 'peft_config', None):
        raise ValueError('Starting state unexpectedly loaded a prior adapter')
    t = cfg['training']
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=t['lora_r'], lora_alpha=t['lora_alpha'],
                                          lora_dropout=0.0, bias='none', target_modules=t['lora_target_modules']))
    model.to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0
    if device == 'cuda':
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    from train import adapter_digest
    fresh = {'new_lora': True, 'new_optimizer': len(optimizer.state) == 0,
             'scheduler_initial_last_epoch': scheduler.last_epoch, 'new_training_state': True,
             'dropout_disabled': True, 'gradient_checkpointing': device == 'cuda',
             'initial_adapter_sha256': adapter_digest(model), 'optimizer_state_entries_at_start': len(optimizer.state)}
    return model, tokenizer, optimizer, scheduler, fresh


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, step, lineage):
    import random
    import torch
    import numpy as np
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path / 'training_state.pt')
    write(path / 'checkpoint.json', {'step': step, 'lineage': lineage,
                                    'adapter_sha256': file_hash(path / 'adapter_model.safetensors')})


def train_run(workspace, seed, state=None, mode=None, dry_run=False):
    if dry_run:
        return _train_run(workspace, seed, state, mode, True)
    with resource_lock(workspace, 'run', str(run_dir(workspace, seed, state, mode).resolve()), blocking=False):
        return _train_run(workspace, seed, state, mode, False)


def _train_run(workspace, seed, state=None, mode=None, dry_run=False):
    workspace = Path(workspace).resolve()
    cfg, receipt = locked(workspace)
    if seed not in cfg['seeds']:
        raise ValueError('Seed is outside the finalized protocol')
    if state is not None and (state not in ('S0', 'S1', 'S2') or mode not in ('random', 'curriculum')):
        raise ValueError('Specify S0/S1/S2 and random/curriculum')
    pre = state is None
    stages = cfg['pre_rl_checkpoints'] if pre else cfg['main_checkpoints'][1:]
    data_name = f'pre_rl_seed{seed}.jsonl' if pre else f'main_seed{seed}_{mode}.jsonl'
    rows = lines(workspace / 'data' / data_name)[:max(stages)]
    plan = {'kind': 'pre_rl' if pre else 'main', 'seed': seed, 'state': state or 'S0', 'mode': mode or 'random',
            'steps': stages, 'problems': len(rows), 'group_size': cfg['training']['num_generations'],
            'protocol_sha256': digest(receipt), 'data_sha256': file_hash(workspace / 'data' / data_name),
            'training': cfg['training']}
    if dry_run:
        return plan
    destination = run_dir(workspace, seed, state, mode)
    if destination.exists():
        raise FileExistsError(f'Will not overwrite or resume another training state: {destination}')
    import torch
    from train import Grader, check_versions, adapter_digest
    check_versions()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError('Run on one BF16 CUDA GPU; no CPU fallback for the full experiment')
    if pre or state == 'S0':
        source = ensure_base(workspace, seed, cfg)
    else:
        source = load_state(workspace, seed, state)
    model, tokenizer, optimizer, scheduler, fresh = fresh_components(source['path'], cfg, seed)
    audit = runtime.prompt_audit(tokenizer, rows)
    if max(r['tokens'] for r in audit['rows']) > cfg['training']['max_prompt_length']:
        raise ValueError('Prompt exceeds limit; revise the common protocol, never silently truncate')
    trajectory = digest({'plan': plan, 'source': source['model_sha256'], 'fresh_adapter': fresh['initial_adapter_sha256']})
    plan.update(status='training', source=source, fresh_initialization=fresh, trajectory_id=trajectory,
                state_dependence={'pre_rl_trajectory': trajectory if pre else source['pre_rl_trajectory'],
                                  'same_trajectory_S1_S2': True, 'independent_seed_trajectory': seed},
                loss_audit=runtime.loss_audit(), checkpoints={}, optimizer_steps=0)
    from importlib.metadata import version
    plan['package_versions'] = {name: version(name) for name in ('torch', 'trl', 'transformers', 'peft', 'accelerate', 'datasets')}
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'run.json', plan)
    write(destination / 'prompt_audit.json', audit)
    grader = Grader(cfg['training']['grading_timeout_seconds'])
    try:
        for index, row in enumerate(rows, start=1):
            if row['position'] != index-1:
                raise ValueError('Unexpected training order')
            log = runtime.train_update(model, tokenizer, row, cfg['training'], optimizer, scheduler, grader)
            if len(log['rewards']) != cfg['training']['num_generations'] or log['optimizer_steps'] != 1:
                raise ValueError('Broken G-completions/one-update correspondence')
            log['step'] = index
            append(destination / 'training.jsonl', log)
            plan['optimizer_steps'] += 1
            if index in stages:
                checkpoint = destination / f'checkpoint-{index}'
                save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, index, trajectory)
                plan['checkpoints'][str(index)] = {'path': str(checkpoint), 'sha256': model_identity(checkpoint)['sha256']}
                write(destination / 'run.json', plan)
        plan.update(status='trained', final_adapter_sha256=adapter_digest(model))
        plan['adapter_changed'] = plan['final_adapter_sha256'] != fresh['initial_adapter_sha256']
        write(destination / 'run.json', plan)
    except Exception as error:
        plan.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(destination / 'run.json', plan)
        raise
    finally:
        grader.close()
    del model, tokenizer, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    if pre:
        for state_name, step in zip(('S1', 'S2'), stages):
            exported = export_checkpoint(workspace, destination, step)
            write(state_file(workspace, seed, state_name), {
                'state': state_name, 'path': exported['path'], 'model_sha256': exported['model_sha256'],
                'source': source['source'], 'pre_rl_trajectory': trajectory, 'pre_rl_step': step,
                'seed': seed, 'paired_states': ['S1', 'S2'], 'not_independent_starting_states': True,
                'adapter_checkpoint': plan['checkpoints'][str(step)]})
    return plan


def export_checkpoint(workspace, run, step):
    with resource_lock(workspace, 'run', str(Path(run).resolve())):
        return _export_checkpoint(workspace, run, step)


def _export_checkpoint(workspace, run, step):
    workspace, run = Path(workspace).resolve(), Path(run).resolve()
    cfg, _ = locked(workspace)
    plan = read(run / 'run.json')
    if str(step) not in plan['checkpoints']:
        raise ValueError('Checkpoint not saved')
    source = plan['source']
    checkpoint = Path(plan['checkpoints'][str(step)]['path'])
    if model_identity(checkpoint)['sha256'] != plan['checkpoints'][str(step)]['sha256']:
        raise ValueError('Adapter checkpoint changed')
    if model_identity(source['path'])['sha256'] != source['model_sha256']:
        raise ValueError('Starting state changed')
    name = f"qwen2-{plan['kind']}-{run.name}-step{step}"
    destination = workspace / 'merged' / name
    identity = {'owner': 'yj_research', 'run': str(run), 'step': step,
                'base_sha256': source['model_sha256'], 'adapter_sha256': plan['checkpoints'][str(step)]['sha256'],
                'kind': plan['kind']}
    marker = destination / 'model_manifest.json'
    if marker.exists():
        saved = read(marker)
        if saved['source'] != identity or model_identity(destination)['sha256'] != saved['model_sha256']:
            raise ValueError('Export changed')
        return saved
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(source['path'], torch_dtype=torch.bfloat16)
    # PEFT otherwise auto-selects CUDA even though the merge model is on CPU.
    adapted = PeftModel.from_pretrained(base, checkpoint, torch_device='cpu')
    merged = adapted.merge_and_unload(safe_merge=True)
    destination.mkdir(parents=True, exist_ok=True)
    merged.config.use_cache = True
    merged.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(source['path']).save_pretrained(destination)
    record = {'path': str(destination), 'model_sha256': model_identity(destination)['sha256'], 'source': identity}
    write(marker, record)
    del base, adapted, merged
    gc.collect()
    return record


def target_for_run(workspace, run, step, direct_lora=False):
    run = Path(run).resolve()
    manifest = read(run / 'run.json')
    source = manifest['source']
    if step == 0:
        return {'path': source['path'], 'model_sha256': source['model_sha256']}
    if direct_lora:
        checkpoint = manifest['checkpoints'][str(step)]
        if model_identity(checkpoint['path'])['sha256'] != checkpoint['sha256']:
            raise ValueError('Adapter checkpoint changed')
        return {'path': source['path'], 'adapter_path': checkpoint['path'],
                'model_sha256': digest({'base': source['model_sha256'], 'adapter': checkpoint['sha256']})}
    return export_checkpoint(workspace, run, step)
