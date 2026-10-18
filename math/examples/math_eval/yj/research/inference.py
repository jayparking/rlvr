"""No-update probing, evaluation and question-prefix diagnostics with content caches."""
import gc
import json
from pathlib import Path
import subprocess
import sys

from . import runtime
from .metrics import reconstruction_scores, reconstruction_split, summarize_samples
from .protocol import A100, EVAL_ROOT, append, digest, file_hash, lines, locked, model_identity, read, write, write_lines, resource_lock
from .workflow import load_state, run_dir, target_for_run


def validate_target(target):
    base = model_identity(target['path'])['sha256']
    actual = digest({'base': base, 'adapter': model_identity(target['adapter_path'])['sha256']}) if target.get('adapter_path') else base
    if actual != target['model_sha256']:
        raise ValueError('Inference model content differs from manifest')


def inference_settings(cfg, kind):
    if kind in ('probe', 'validation'):
        t = cfg['training']
        settings = {k: t[k] for k in ('temperature', 'top_p', 'top_k', 'max_completion_length', 'max_prompt_length')}
        settings.update(k=8, seed=cfg['probing']['sampling_seed'])
    elif kind in ('eval_avg8', 'eval_greedy'):
        settings = {k: cfg['evaluation'][k] for k in ('temperature', 'top_p', 'top_k', 'max_completion_length')}
        settings.update(k=8, seed=cfg['evaluation']['sampling_seed'], max_prompt_length=8192-settings['max_completion_length'])
        if kind == 'eval_greedy':
            settings.update(k=1, temperature=0.0, top_p=1.0)
    elif kind == 'contamination':
        settings = dict(cfg['contamination'])
        settings.update(seed=settings.pop('sampling_seed'), max_prompt_length=8192-settings['max_completion_length'], use_math_stops=False)
    else:
        raise ValueError(kind)
    return settings


def engine_identity(backend, python):
    if backend in ('a100', 'vllm', 'vllm-lora'):
        script = "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ['torch','transformers','vllm']}))"
        versions = json.loads(subprocess.check_output([str(python), '-c', script], text=True))
    else:
        from importlib.metadata import version
        versions = {p: version(p) for p in (('torch', 'transformers') if backend == 'hf' else ('torch', 'transformers', 'vllm'))}
    identity = {'backend': backend, 'versions': versions}
    if backend == 'hf':
        import torch
        cuda = torch.cuda.is_available()
        identity['execution'] = {'device': 'cuda' if cuda else 'cpu',
                                 'dtype': 'bfloat16' if cuda else 'float32',
                                 'device_name': torch.cuda.get_device_name() if cuda else 'cpu',
                                 'cuda_runtime': torch.version.cuda if cuda else None}
    return identity


def cache_request(target, rows, settings, engine, kind, contract):
    return {'model_sha256': target['model_sha256'], 'dataset_sha256': digest(rows),
            'settings': settings, 'engine': engine, 'kind': kind,
            'contract_sha256': digest(contract), 'no_optimizer': True, 'schema': 1}


def load_cache(directory, request):
    directory = Path(directory)
    marker = directory / 'complete.json'
    if not marker.exists():
        return None
    receipt = read(marker)
    if receipt['request_sha256'] != digest(request) or file_hash(directory / 'result.json') != receipt['result_sha256']:
        raise ValueError('Cache request/result was modified')
    result = read(directory / 'result.json')
    if result['cache_key'] != digest(request):
        raise ValueError('Cache key mismatch')
    return result


def a100_command(target, directory, settings, python):
    if target.get('adapter_path'):
        raise ValueError('Unchanged A100 evaluator requires merged weights; use vllm-lora for adapters')
    # Existing evaluator selects Qwen stop IDs by path name. Our exports retain
    # qwen2 in their name; prevent a hidden EOS change after renaming/moving.
    if 'qwen2' not in target['path'].lower():
        raise ValueError('A100 Qwen EOS contract requires qwen2 in the model path; use the generated export path')
    return [str(python), str(A100), '--model_name_or_path', target['path'],
            '--data_names', 'math', '--data_dir', str(directory / 'data'), '--split', 'test',
            '--num_test_sample', '-1', '--prompt_type', 'cot', '--num_shots', '0',
            '--seed', str(settings['seed']), '--temperature', str(settings['temperature']),
            '--top_p', str(settings['top_p']), '--n_sampling', str(settings['k']),
            '--max_tokens_per_call', str(settings['max_completion_length']),
            '--use_vllm', '--save_outputs', '--overwrite', '--output_dir', str(directory / 'raw')]


def a100_inference(target, rows, settings, directory, python, audit):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target['path'])
    if runtime.eos_ids(tokenizer) != [151643, 151645]:
        raise ValueError('Tokenizer EOS differs from the protected A100 Qwen stop-token IDs')
    # Independently audit default tokenization in the actual evaluation environment.
    payload = directory / 'tokenization_input.json'
    texts = [runtime.prompt(r)[0] for r in rows]
    write(payload, texts)
    script = "import sys,json; from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(sys.argv[1]); print(json.dumps([t.encode(p,add_special_tokens=True) for p in json.load(open(sys.argv[2]))]))"
    actual = json.loads(subprocess.check_output([str(python), '-c', script, target['path'], str(payload)], text=True))
    expected = [runtime.encode_prompt(tokenizer, p) for p in texts]
    if actual != expected:
        raise ValueError('Training/evaluation environment tokenization differs')
    prepared = []
    for index, r in enumerate(rows):
        row = dict(r)
        # The protected loader preserves/sorts incoming idx values. Local
        # consecutive indices keep arbitrary auxiliary source IDs paired.
        row['idx'] = index
        if not row.get('solution'):
            row['solution'] = '\\boxed{' + str(row['answer']) + '}'
        prepared.append(row)
    write_lines(directory / 'data/math/test.jsonl', prepared)
    command = a100_command(target, directory, settings, python)
    import os
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if not env['CUDA_VISIBLE_DEVICES']:
        raise ValueError('No GPU selected')
    write(directory / 'command.json', command)
    subprocess.run(command, cwd=EVAL_ROOT, env=env, check=True)
    files = list((directory / 'raw/math').glob('*.jsonl'))
    if len(files) != 1:
        raise ValueError('Expected one evaluation output')
    results = lines(files[0])
    if len(results) != len(rows) or {r['idx'] for r in results} != set(range(len(rows))):
        raise ValueError('Incomplete or duplicate evaluation problems')
    parse_question, _, _ = runtime.imports()
    items = []
    for result in sorted(results, key=lambda r: r['idx']):
        row = prepared[result['idx']]
        if result['question'] != parse_question(row, 'math'):
            raise ValueError('Evaluator question identity changed')
        fields = [result[x] for x in ('code', 'score', 'pred', 'finish_reason')]
        if any(len(f) != settings['k'] for f in fields):
            raise ValueError('Incorrect completion count')
        samples = []
        for text, score, prediction, reason in zip(*fields):
            if type(score) is not bool or reason not in ('stop', 'length'):
                raise ValueError('Missing trustworthy completion score/termination reason')
            samples.append({'text': text, 'correct': score, 'prediction': prediction,
                            'finish_reason': reason, 'truncated': reason == 'length',
                            'grading_error': None})
        items.append({'id': row['id'], 'difficulty': row.get('difficulty'), 'samples': samples})
    return items


def parameter_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hf_inference(target, rows, settings, kind, directory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from train import Grader
    set_seed(settings['seed'])
    tokenizer = AutoTokenizer.from_pretrained(target['path'], padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(target['path'], torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
    if target.get('adapter_path'):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, target['adapter_path'], is_trainable=False)
    model.to(device)
    model.requires_grad_(False)
    before = parameter_digest(model)
    texts = [reconstruction_split(r['problem'])[0] if kind == 'contamination' else runtime.prompt(r)[0] for r in rows]
    grader = Grader(3)
    items = []
    try:
        for row, text in zip(rows, texts):
            samples = runtime.generate(model, tokenizer, [text], settings, settings['k'])[0]
            if kind == 'contamination':
                _, suffix = reconstruction_split(row['problem'])
                for sample in samples:
                    sample.update(reconstruction_scores(sample['text'], suffix))
            else:
                _, answer = runtime.prompt(row)
                for sample in samples:
                    value, prediction, error = grader.score(sample['text'], answer)
                    sample.update(correct=bool(value), prediction=prediction, grading_error=error)
            items.append({'id': row['id'], 'difficulty': row.get('difficulty'), 'samples': samples})
        after = parameter_digest(model)
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise AssertionError('No-update inference changed parameters or accumulated gradients')
        write(directory / 'no_update_audit.json', {'parameter_sha256_before': before, 'parameter_sha256_after': after,
                                                  'gradients_absent': True, 'optimizer_created': False})
    finally:
        grader.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return items


def vllm_inference(target, rows, settings, kind, directory):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from train import Grader
    llm = LLM(model=target['path'], enable_lora=bool(target.get('adapter_path')), max_lora_rank=64,
              tensor_parallel_size=1, seed=settings['seed'], max_model_len=8192, max_num_seqs=16,
              gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()
    texts = [reconstruction_split(r['problem'])[0] if kind == 'contamination' else runtime.prompt(r)[0] for r in rows]
    ids = [runtime.encode_prompt(tokenizer, p) for p in texts]
    params = SamplingParams(n=settings['k'], temperature=settings['temperature'], top_p=settings['top_p'],
                            top_k=-1, max_tokens=settings['max_completion_length'],
                            stop=[] if kind == 'contamination' else runtime.stop_strings(),
                            stop_token_ids=runtime.eos_ids(tokenizer))
    request = LoRARequest('research_adapter', 1, target['adapter_path']) if target.get('adapter_path') else None
    generated = llm.generate([{'prompt_token_ids': x} for x in ids], params, lora_request=request)
    if len(generated) != len(rows):
        raise ValueError('Incomplete vLLM output')
    grader = Grader(3)
    items = []
    try:
        for row, output in zip(rows, generated):
            samples = []
            if len(output.outputs) != settings['k']:
                raise ValueError('Incorrect vLLM sample count')
            for sample in output.outputs:
                record = {'text': sample.text.strip(), 'finish_reason': sample.finish_reason,
                          'truncated': sample.finish_reason == 'length', 'token_ids': list(sample.token_ids)}
                if kind == 'contamination':
                    record.update(reconstruction_scores(record['text'], reconstruction_split(row['problem'])[1]))
                else:
                    value, prediction, error = grader.score(record['text'], runtime.prompt(row)[1])
                    record.update(correct=bool(value), prediction=prediction, grading_error=error)
                samples.append(record)
            items.append({'id': row['id'], 'difficulty': row.get('difficulty'), 'samples': samples})
    finally:
        grader.close()
    return items


def vllm_subprocess(target, rows, settings, kind, directory, python, audit):
    """Keep vLLM's dependencies and GPU lifetime separate from the training process."""
    payload = directory / 'vllm_input.json'
    output = directory / 'vllm_output.json'
    write(payload, {'target': target, 'rows': rows, 'settings': settings, 'kind': kind,
                    'directory': str(directory), 'audit': audit})
    subprocess.run([str(python), str(EVAL_ROOT / 'yj/train.py'), '_research-vllm-worker',
                    str(payload), str(output)], cwd=EVAL_ROOT, check=True)
    return read(output)


def vllm_worker(input_path, output_path):
    from transformers import AutoTokenizer
    payload = read(input_path)
    target, rows = payload['target'], payload['rows']
    validate_target(target)
    tokenizer = AutoTokenizer.from_pretrained(target['path'])
    if payload['kind'] == 'contamination':
        actual = [digest(runtime.encode_prompt(tokenizer, reconstruction_split(r['problem'])[0])) for r in rows]
        if actual != payload['audit']['token_hashes'] or runtime.eos_ids(tokenizer) != payload['audit']['eos_ids']:
            raise ValueError('vLLM environment prefix tokenization/EOS differs')
    else:
        actual = runtime.prompt_audit(tokenizer, rows)
        actual.pop('template_source', None)
        if actual != payload['audit']:
            raise ValueError('vLLM environment prompt/tokenization/EOS differs')
    write(output_path, vllm_inference(target, rows, payload['settings'], payload['kind'], Path(payload['directory'])))


def sample(workspace, target, rows, cfg, kind, backend='a100', python=sys.executable):
    key = digest({'model': target['model_sha256'], 'rows': rows, 'kind': kind,
                  'settings': inference_settings(cfg, kind), 'backend': backend})
    with resource_lock(workspace, 'sample', key):
        return _sample(workspace, target, rows, cfg, kind, backend, python)


def _sample(workspace, target, rows, cfg, kind, backend='a100', python=sys.executable):
    workspace = Path(workspace).resolve()
    from transformers import AutoTokenizer
    validate_target(target)
    tokenizer = AutoTokenizer.from_pretrained(target['path'], padding_side='left')
    settings = inference_settings(cfg, kind)
    if kind == 'contamination':
        texts = [reconstruction_split(r['problem'])[0] for r in rows]
        if any(len(runtime.encode_prompt(tokenizer, text)) > settings['max_prompt_length'] for text in texts):
            raise ValueError('Question prefix exceeds inference context limit')
        audit = {'raw_question_prefix': True, 'split': cfg['contamination'], 'no_cot_prompt': True,
                 'token_hashes': [digest(runtime.encode_prompt(tokenizer, p)) for p in texts],
                 'stop_strings': [], 'eos_ids': runtime.eos_ids(tokenizer)}
        if backend == 'a100':
            raise ValueError('A100 math evaluator wraps prompts: use --backend hf or vllm for raw-prefix diagnosis')
    else:
        audit = runtime.prompt_audit(tokenizer, rows)
        if any(r['tokens'] > settings['max_prompt_length'] for r in audit['rows']):
            raise ValueError('Prompt exceeds inference context limit')
    # Exclude paths/step names so identical baseline checkpoints share results.
    audit.pop('template_source', None)
    request = cache_request(target, rows, settings, engine_identity(backend, python), kind, audit)
    key = digest(request)
    directory = workspace / 'cache' / key
    cached = load_cache(directory, request)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'request.json', request)
    write(directory / 'prompt_audit.json', audit)
    if backend == 'a100':
        items = a100_inference(target, rows, settings, directory, python, audit)
    elif backend == 'hf':
        items = hf_inference(target, rows, settings, kind, directory)
    elif backend in ('vllm', 'vllm-lora'):
        items = vllm_subprocess(target, rows, settings, kind, directory, python, audit)
    else:
        raise ValueError(backend)
    if [i['id'] for i in items] != [r['id'] for r in rows] or any(len(i['samples']) != settings['k'] for i in items):
        raise ValueError('Incomplete inference result; refusing cache completion')
    if kind == 'contamination':
        total = len(items)*settings['k']
        summary = {m: sum(s[m] for i in items for s in i['samples'])/total for m in ('exact_match', 'whitespace_normalized_exact_match', 'rouge_l_f1')}
        summary['interpretation'] = 'Question reconstruction is a diagnostic, not proof or exclusion of training-data contamination.'
    else:
        summary = summarize_samples(items, cfg['training']['num_generations'])
    result = {'cache_key': key, 'request': request, 'items': items, 'diagnostics': summary,
              'no_update': True, 'optimizer_created': False,
              'a100_grading_note': 'Original evaluator does not expose per-answer grading timeouts; raw aggregate metrics retained.' if backend == 'a100' else None}
    write(directory / 'result.json', result)
    write(directory / 'complete.json', {'request_sha256': key, 'result_sha256': file_hash(directory / 'result.json')})
    return result


def run_inference(workspace, seed, state, mode, kind, backend, python, steps=None):
    with resource_lock(workspace, 'run', str(run_dir(workspace, seed, state, mode).resolve())):
        return _run_inference(workspace, seed, state, mode, kind, backend, python, steps)


def _run_inference(workspace, seed, state, mode, kind, backend, python, steps=None):
    workspace = Path(workspace).resolve()
    cfg, receipt = locked(workspace)
    backend = backend or cfg['evaluation']['backend']
    run = run_dir(workspace, seed, state, mode)
    manifest = read(run / 'run.json')
    if manifest['status'] != 'trained':
        raise ValueError('Finish training before checkpoint probing/evaluation')
    stages = cfg['main_checkpoints'] if steps is None else steps
    if any(s not in cfg['main_checkpoints'] for s in stages):
        raise ValueError('Checkpoint outside finalized protocol')
    datasets = ['main'] if kind == 'probe' else ['math500'] + [s['name'] for s in cfg['evaluation']['auxiliary_sets']]
    kinds = ['probe'] if kind == 'probe' else ['eval_avg8', 'eval_greedy']
    references = []
    for stage in stages:
        target = target_for_run(workspace, run, stage, direct_lora=backend == 'vllm-lora')
        for dataset in datasets:
            rows = lines(workspace / 'data' / f'{dataset}.jsonl')
            for inference_kind in kinds:
                result = sample(workspace, target, rows, cfg, inference_kind, backend, python)
                ref = {'cache_key': result['cache_key'], 'run': run.name, 'seed': seed, 'state': state,
                       'mode': mode, 'step': stage, 'dataset': dataset, 'kind': inference_kind,
                       'model_sha256': target['model_sha256'], 'protocol_sha256': digest(receipt)}
                write(run / 'results' / f'{inference_kind}_{dataset}_step{stage}.json', ref)
                references.append(ref)
    return references


def diagnose_states(workspace, seed, kind, backend, python):
    workspace = Path(workspace).resolve()
    cfg, receipt = locked(workspace)
    backend = backend or cfg['evaluation']['backend']
    rows = lines(workspace / 'data' / ('math500.jsonl' if kind == 'contamination' else 'validation.jsonl'))
    report = {'seed': seed, 'kind': kind, 'states': {}, 'selection_rule': 'fixed candidates; do not select by significance',
              'protocol_sha256': digest(receipt)}
    for state in ('S0', 'S1', 'S2'):
        target = load_state(workspace, seed, state)
        result = sample(workspace, target, rows, cfg, kind, backend, python)
        entry = {'cache_key': result['cache_key'], 'diagnostics': result['diagnostics'],
                 'pre_rl_trajectory': target['pre_rl_trajectory'], 'pre_rl_step': target['pre_rl_step']}
        if kind == 'validation' and state != 'S0':
            logs = lines(run_dir(workspace, seed) / 'training.jsonl')[:target['pre_rl_step']]
            entry['observed_training_group_fractions'] = {
                'all_wrong': sum(x['all_wrong'] for x in logs)/len(logs),
                'all_correct': sum(x['all_correct'] for x in logs)/len(logs),
                'mixed': sum(not x['zero_reward_variance'] for x in logs)/len(logs),
                'truncation_rate': sum(x['truncated_count'] for x in logs)/(len(logs)*cfg['training']['num_generations'])}
        report['states'][state] = entry
    report['interpretation'] = ('Differences are descriptive; no automatic checkpoint replacement or significance filter.' if kind == 'validation'
                                else 'Prefix reconstruction alone neither confirms nor rules out data contamination; compare S0/S1/S2 on the same prefixes.')
    write(workspace / 'diagnostics' / f'{kind}_seed{seed}.json', report)
    return report


def compare_backends(workspace, run, step, python, limit=10):
    with resource_lock(workspace, 'run', str(Path(run).resolve())):
        return _compare_backends(workspace, run, step, python, limit)


def _compare_backends(workspace, run, step, python, limit=10):
    workspace, run = Path(workspace).resolve(), Path(run).resolve()
    cfg, _ = locked(workspace)
    if not 1 <= limit <= 500:
        raise ValueError('Set an explicit compatibility sample size 1..500')
    rows = lines(workspace / 'data/math500.jsonl')[:limit]
    merged = target_for_run(workspace, run, step)
    adapter = target_for_run(workspace, run, step, direct_lora=True)
    a = sample(workspace, merged, rows, cfg, 'eval_greedy', 'a100', python)
    b = sample(workspace, adapter, rows, cfg, 'eval_greedy', 'vllm-lora', python)
    report = {'problems': limit, 'merged_cache': a['cache_key'], 'adapter_cache': b['cache_key'],
              'greedy_text_match_fraction': sum(x['samples'][0]['text'] == y['samples'][0]['text'] for x, y in zip(a['items'], b['items']))/limit,
              'correctness_match_fraction': sum(x['samples'][0]['correct'] == y['samples'][0]['correct'] for x, y in zip(a['items'], b['items']))/limit,
              'note': 'Identical prompt/token/stop contracts checked. Numerical rounding can change greedy outputs. This is an empirical compatibility check, not a guarantee of bitwise equivalence.'}
    write(run / 'compatibility' / f'step{step}.json', report)
    return report


def cleanup_merged(workspace, run, step, delete=False):
    with resource_lock(workspace, 'run', str(Path(run).resolve())):
        return _cleanup_merged(workspace, run, step, delete)


def _cleanup_merged(workspace, run, step, delete=False):
    """Only remove owned, reproducible MAIN merged weights after completed diagnostics.

    Never touches adapters, optimizer checkpoints, raw results, caches, data, logs,
    base weights, or pre-RL S1/S2 starting states. Default is a dry run.
    """
    import shutil
    workspace, run = Path(workspace).resolve(), Path(run).resolve()
    cfg, receipt = locked(workspace)
    plan = read(run / 'run.json')
    if plan['kind'] != 'main' or step == 0 or str(step) not in plan['checkpoints']:
        raise ValueError('Only derived main-experiment merged checkpoints can be cleaned')
    datasets = ['math500'] + [s['name'] for s in cfg['evaluation']['auxiliary_sets']]
    required = [('probe', 'main')] + [(k, d) for d in datasets for k in ('eval_avg8', 'eval_greedy')]
    references = []
    for kind, dataset in required:
        ref = read(run / 'results' / f'{kind}_{dataset}_step{step}.json')
        cache_dir = workspace / 'cache' / ref['cache_key']
        request = read(cache_dir / 'request.json')
        if load_cache(cache_dir, request) is None:
            raise ValueError('Evaluation/probing not complete; refusing deletion')
        if (ref['run'] != run.name or ref['step'] != step or ref['kind'] != kind or ref['dataset'] != dataset
                or ref['protocol_sha256'] != digest(receipt) or ref['cache_key'] != digest(request)
                or request['model_sha256'] != ref['model_sha256']
                or request['kind'] != kind or request['dataset_sha256'] != digest(lines(workspace / 'data' / f'{dataset}.jsonl'))):
            raise ValueError('Completed results belong to another model/protocol/dataset')
        references.append(ref)
    directory = workspace / 'merged' / f'qwen2-main-{run.name}-step{step}'
    if (workspace / 'merged').is_symlink() or directory.is_symlink() or directory.resolve().parent != workspace / 'merged':
        raise ValueError('Unsafe cleanup path')
    record = read(directory / 'model_manifest.json')
    if record['source']['owner'] != 'yj_research' or record['source']['run'] != str(run) or record['source']['kind'] != 'main' or record['source']['step'] != step:
        raise ValueError('Not an owned derived model')
    if model_identity(directory)['sha256'] != record['model_sha256']:
        raise ValueError('Merged weights changed; refusing deletion')
    checkpoint = plan['checkpoints'][str(step)]
    if model_identity(checkpoint['path'])['sha256'] != checkpoint['sha256']:
        raise ValueError('Original adapter missing/changed; cannot safely reproduce export')
    if model_identity(plan['source']['path'])['sha256'] != plan['source']['model_sha256']:
        raise ValueError('Starting model missing/changed; cannot safely reproduce export')
    if record['source']['base_sha256'] != plan['source']['model_sha256'] or record['source']['adapter_sha256'] != checkpoint['sha256']:
        raise ValueError('Export lineage differs from original checkpoints')
    direct_id = digest({'base': plan['source']['model_sha256'], 'adapter': checkpoint['sha256']})
    if any(ref['model_sha256'] not in (record['model_sha256'], direct_id) for ref in references):
        raise ValueError('Completed results do not evaluate this checkpoint')
    result = {'path': str(directory), 'delete_requested': delete, 'protected_artifacts_untouched': True}
    if delete:
        write(run / 'cleanup' / f'step{step}.json', dict(result, export_manifest=record))
        shutil.rmtree(directory)
    return result
