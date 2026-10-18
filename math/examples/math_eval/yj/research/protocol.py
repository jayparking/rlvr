"""Configuration, disjoint data, immutable preregistration and cost accounting."""
import hashlib
import json
from pathlib import Path
import random
import re
import os
import tempfile
import threading
from contextlib import contextmanager

_held_locks = threading.local()


@contextmanager
def resource_lock(workspace, namespace, key, blocking=True):
    """POSIX process lock, reentrant within one thread; never unlink lock files.

    Use one local filesystem on one host. The OS releases locks on process exit.
    Keeping the inode avoids splitting waiters across old/new lock files.
    """
    import fcntl
    path = Path(workspace).resolve() / '.locks' / f'{namespace}-{digest(str(key))}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    held = getattr(_held_locks, 'keys', set())
    identity = (os.getpid(), str(path))
    if identity in held:
        yield
        return
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f'Resource already running: {namespace} {key}') from error
        held.add(identity)
        _held_locks.keys = held
        try:
            yield
        finally:
            held.remove(identity)
            fcntl.flock(handle, fcntl.LOCK_UN)

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT.parent
A100 = EVAL_ROOT / 'math_eval_a100.py'
A100_SHA = '291dfe3bbfd1732614c4a122d308259db950a1b1e3e373eafe2bb684fd6da4ce'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def fingerprint(text):
    return hashlib.sha256(re.sub(r'\s+', '', text).encode()).hexdigest()


def config(path=ROOT / 'research.json'):
    cfg = read(path)
    digest(cfg)  # Reject NaN/Infinity before any cost calculation or GPU work.
    t = cfg['training']
    integer_fields = {'main_per_group': cfg['main_per_group'], 'validation_per_group': cfg['validation_per_group'],
                      'pre_rl_pool_per_group': cfg['pre_rl_pool_per_group']}
    integer_fields.update({f'training.{k}': t[k] for k in ('num_generations', 'max_prompt_length',
                           'max_completion_length', 'lora_r', 'lora_alpha')})
    for section in ('evaluation', 'contamination'):
        integer_fields[f'{section}.max_completion_length'] = cfg[section]['max_completion_length']
        integer_fields[f'{section}.k'] = cfg[section]['k']
    for name, value in integer_fields.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(type(s) is not int or s < 0 for s in cfg['seeds'] + cfg['pre_rl_checkpoints']):
        raise ValueError('Seeds/checkpoints must be nonnegative integers')
    if not 0 <= t['epsilon_low'] < 1 or t['epsilon_high'] < 0:
        raise ValueError('Invalid GRPO clipping bounds')
    if t['weight_decay'] < 0 or t['max_grad_norm'] <= 0 or t['grading_timeout_seconds'] <= 0:
        raise ValueError('Invalid optimizer or grading settings')
    if not t['lora_target_modules'] or any(not isinstance(n,str) or not n for n in t['lora_target_modules']):
        raise ValueError('Specify LoRA target modules')
    if t['max_prompt_length'] + t['max_completion_length'] > 8192 or any(
            cfg[s]['max_completion_length'] >= 8192 for s in ('evaluation','contamination')):
        raise ValueError('Generation lengths exceed the common 8192-token inference context')
    if t['beta'] != 0.0 or t['num_iterations'] != 1:
        raise ValueError('This protocol requires beta=0 and num_iterations=1')
    if t['loss_type'] != 'bnpo' or t['scale_rewards'] is not True:
        raise ValueError('Fixed audited combination: bnpo + scale_rewards=True')
    if t['advantage_population'] != 'all_G_including_truncated':
        raise ValueError('Unsupported advantage population')
    if type(t['mask_truncated_completions']) is not bool:
        raise ValueError('Specify mask_truncated_completions explicitly')
    if cfg['main_checkpoints'] != [0, 100, 200, 300] or cfg['main_per_group'] != 100:
        raise ValueError('Main experiment requires 300 problems and checkpoints 0/100/200/300')
    if cfg['pre_rl_pool_per_group'] * 3 < 1000:
        raise ValueError('Pre-RL pool must contain at least 1000 distinct problems')
    steps = cfg['pre_rl_checkpoints']
    if len(steps) != 2 or steps != sorted(set(steps)) or steps[0] <= 0 or steps[-1] > 3*cfg['pre_rl_pool_per_group']:
        raise ValueError('S1/S2 require two increasing checkpoints within the pre-RL pool')
    if cfg['groups'] != {'easy': [1, 2], 'medium': [3], 'hard': [4, 5]}:
        raise ValueError('Unexpected difficulty grouping')
    if len(set(cfg['seeds'])) != len(cfg['seeds']) or len(cfg['seeds']) < 2:
        raise ValueError('Specify distinct training seeds')
    if cfg['probing']['k'] != 8 or cfg['evaluation']['k'] != 8:
        raise ValueError('Probing/evaluation use k=8')
    if cfg['evaluation']['primary'] not in ('math500.avg@8', 'math500.greedy_accuracy'):
        raise ValueError('Choose the primary metric before finalizing')
    if cfg['evaluation']['backend'] not in ('a100', 'hf', 'vllm', 'vllm-lora') or cfg['evaluation']['greedy_auxiliary'] is not True:
        raise ValueError('Specify evaluation backend and retain separate greedy evaluation')
    auxiliary_names = [s['name'] for s in cfg['evaluation']['auxiliary_sets']]
    if len(set(auxiliary_names)) != len(auxiliary_names) or set(auxiliary_names) & {'main', 'validation', 'pre_rl', 'math500'}:
        raise ValueError('Auxiliary names must be unique and must not replace built-in datasets')
    for settings in (t, cfg['evaluation']):
        if settings['temperature'] <= 0 or not 0 < settings['top_p'] <= 1:
            raise ValueError('Invalid sampling temperature/top_p')
        if settings['top_k'] != 0:
            raise ValueError('Use top_k=0 (disabled), matching the A100 vLLM path')
    if t['num_generations'] < 2 or t['learning_rate'] <= 0:
        raise ValueError('Invalid G/LR')
    diagnostic = cfg['contamination']
    if diagnostic['prefix_fraction'] != 0.6 or diagnostic['split_unit'] != 'whitespace_tokens':
        raise ValueError('This diagnostic uses a 60% whitespace-token question prefix')
    if diagnostic['top_k'] != 0 or diagnostic['temperature'] < 0 or not 0 < diagnostic['top_p'] <= 1:
        raise ValueError('Invalid diagnostic sampling settings; top_k must be disabled')
    if diagnostic['temperature'] == 0 and (diagnostic['k'] != 1 or diagnostic['top_p'] != 1):
        raise ValueError('Greedy reconstruction requires k=1 and top_p=1')
    stats = cfg['statistics']
    if type(stats['bootstrap_replicates']) is not int or stats['bootstrap_replicates'] < 2 or not 0 < stats['confidence'] < 1:
        raise ValueError('Invalid bootstrap settings')
    if file_hash(A100) != A100_SHA:
        raise ValueError('Protected math_eval_a100.py changed; re-audit explicitly')
    return cfg


def code_identity():
    files = list((ROOT / 'research').glob('*.py')) + [ROOT / 'train.py', ROOT / 'prepare_data.py']
    files += [EVAL_ROOT / n for n in ('math_eval_a100.py', 'utils.py', 'parser.py', 'grader.py',
                                     'evaluate.py', 'data_loader.py', 'python_executor.py', 'trajectory.py', 'model_utils.py')]
    return {str(p.relative_to(EVAL_ROOT)): file_hash(p) for p in sorted(files)}


def model_identity(path):
    """Hash weights + tokenizer/config content, not a mutable directory name."""
    path = Path(path).resolve()
    entries = {}
    for p in sorted(path.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt') and p.name not in ('model_manifest.json',):
            entries[str(p.relative_to(path))] = file_hash(p)
    if not any(n.endswith(('.safetensors', '.bin')) for n in entries):
        raise ValueError(f'No model/adapter weights: {path}')
    return {'sha256': digest(entries), 'files': entries}


def normalize(rows, source):
    result = []
    for i, row in enumerate(rows):
        problem = row.get('problem', row.get('question', ''))
        if not problem.strip() or not (row.get('solution') or 'answer' in row):
            raise ValueError(f'Missing question/reference in {source}:{i}')
        fp = fingerprint(problem)
        level = row.get('level')
        if isinstance(level, str):
            match = re.fullmatch(r'(?:Level )?([1-5])', level)
            level = int(match[1]) if match else None
        group = 'easy' if level in (1, 2) else 'medium' if level == 3 else 'hard' if level in (4, 5) else None
        result.append(dict(row, id=f'{source}:{fp}', problem=problem, question=problem,
                           fingerprint=fp, level=level, difficulty=group, source=source))
    if len({r['id'] for r in result}) != len(result):
        raise ValueError(f'Duplicate questions in {source}')
    return result


def schedule(rows, seed, mode):
    ordered = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(ordered)
    if mode == 'curriculum':
        rank = {'easy': 0, 'medium': 1, 'hard': 2}
        ordered.sort(key=lambda r: rank[r['difficulty']])
    elif mode != 'random':
        raise ValueError(mode)
    return [dict(row, position=i) for i, row in enumerate(ordered)]


def partition(cfg, train, excluded):
    mapping = {level: group for group, levels in cfg['groups'].items() for level in levels}
    buckets = {g: [] for g in cfg['groups']}
    seen = set(excluded)
    skipped = {'overlap_or_duplicate': 0, 'unknown_level': 0}
    for row in sorted(train, key=lambda r: r['id']):
        fp = fingerprint(row['problem'])
        if fp in seen:
            skipped['overlap_or_duplicate'] += 1
            continue
        seen.add(fp)
        if row['level'] not in mapping:
            skipped['unknown_level'] += 1
            continue
        buckets[mapping[row['level']]].append(dict(row, fingerprint=fp, difficulty=mapping[row['level']]))
    pools = {'main': [], 'validation': [], 'pre_rl': []}
    sizes = [cfg['main_per_group'], cfg['validation_per_group'], cfg['pre_rl_pool_per_group']]
    for group, rows in buckets.items():
        random.Random(f"{cfg['selection_seed']}:{group}:pool").shuffle(rows)
        if len(rows) < sum(sizes):
            raise ValueError(f'Insufficient distinct {group} questions')
        offset = 0
        for name, size in zip(pools, sizes):
            pools[name].extend(rows[offset:offset+size])
            offset += size
    return pools, skipped


def prepare(cfg_path, workspace, offline=False):
    from prepare_data import load_sources
    cfg = config(cfg_path)
    workspace = Path(workspace).resolve()
    destination = workspace / 'data'
    if destination.exists():
        raise FileExistsError(f'Will not overwrite prepared data: {destination}')
    math_path = (Path(cfg_path).resolve().parent / cfg['evaluation']['math500_file']).resolve()
    if file_hash(math_path) != cfg['evaluation']['math500_sha256']:
        raise ValueError('MATH500 checksum changed')
    exams = {'math500': normalize(lines(math_path), 'math500')}
    if len(exams['math500']) != 500:
        raise ValueError('Expected MATH500 = 500 problems')
    provenance = {'math500': {'path': str(math_path), 'sha256': file_hash(math_path)}}
    for spec in cfg['evaluation']['auxiliary_sets']:
        if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_-]*', spec['name']) or spec['name'] in exams:
            raise ValueError('Invalid/duplicate auxiliary dataset name')
        if not spec.get('provenance') or not spec.get('contamination_risk_note'):
            raise ValueError('Auxiliary datasets need provenance and contamination_risk_note')
        path = (Path(cfg_path).resolve().parent / spec['path']).resolve()
        if file_hash(path) != spec['sha256']:
            raise ValueError('Auxiliary checksum mismatch')
        exams[spec['name']] = normalize(lines(path), spec['name'])
        provenance[spec['name']] = spec
    splits, sources = load_sources(cfg, ROOT / '.cache/huggingface', offline)
    excluded = {fingerprint(r['problem']) for r in splits['test']}
    excluded.update(r['fingerprint'] for rows in exams.values() for r in rows)
    pools, skipped = partition(cfg, splits['train'], excluded)
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in {**pools, **exams}.items():
        write_lines(destination / f'{name}.jsonl', rows)
    for seed in cfg['seeds']:
        for mode in ('random', 'curriculum'):
            write_lines(destination / f'main_seed{seed}_{mode}.jsonl', schedule(pools['main'], seed, mode))
        write_lines(destination / f'pre_rl_seed{seed}.jsonl', schedule(pools['pre_rl'], seed, 'random'))
    manifest = {'config_sha256': digest(cfg), 'counts': {k: len(v) for k, v in {**pools, **exams}.items()},
                'excluded': skipped, 'sources': sources, 'exam_provenance': provenance,
                'selection_seed': cfg['selection_seed'], 'same_main_questions_across_seeds': True,
                'files': {p.name: file_hash(p) for p in sorted(destination.glob('*.jsonl'))}}
    write(destination / 'manifest.json', manifest)
    write(workspace / 'research.json', cfg)
    return manifest


def verify_data(workspace, cfg):
    workspace = Path(workspace)
    manifest = read(workspace / 'data/manifest.json')
    if manifest['config_sha256'] != digest(cfg):
        raise ValueError('Prepared config differs; prepare a new workspace for the new protocol')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Data manifest contains an unsafe path')
        if file_hash(workspace / 'data' / name) != expected:
            raise ValueError(f'Data changed: {name}')
    sizes = {'main': 300, 'validation': cfg['validation_per_group']*3,
             'pre_rl': cfg['pre_rl_pool_per_group']*3, 'math500': 500}
    names = list(sizes) + [s['name'] for s in cfg['evaluation']['auxiliary_sets']]
    pools = {}
    for name in names:
        if f'{name}.jsonl' not in manifest['files']:
            raise ValueError(f'Missing dataset in manifest: {name}')
        rows = lines(workspace / 'data' / f'{name}.jsonl')
        if not rows or len(rows) != manifest['counts'][name] or (name in sizes and len(rows) != sizes[name]):
            raise ValueError(f'Wrong dataset size: {name}')
        if len({r['id'] for r in rows}) != len(rows) or len({fingerprint(r['problem']) for r in rows}) != len(rows):
            raise ValueError(f'Duplicate dataset questions: {name}')
        pools[name] = rows
    training_names = ('main', 'validation', 'pre_rl')
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if name in training_names or other in training_names:
                if {fingerprint(r['problem']) for r in pools[name]} & {fingerprint(r['problem']) for r in pools[other]}:
                    raise ValueError(f'Overlapping datasets: {name}/{other}')
    for seed in cfg['seeds']:
        orders = [(f'main_seed{seed}_{mode}.jsonl', 'main', mode) for mode in ('random', 'curriculum')]
        orders.append((f'pre_rl_seed{seed}.jsonl', 'pre_rl', 'random'))
        for name, pool, mode in orders:
            if name not in manifest['files'] or lines(workspace / 'data' / name) != schedule(pools[pool], seed, mode):
                raise ValueError(f'Training schedule differs from paired seed protocol: {name}')
    return manifest


def finalize(workspace, primary, acceptance):
    if not acceptance:
        raise ValueError('Review conflicts/defaults, then pass --accept-design explicitly')
    workspace = Path(workspace)
    cfg = config(workspace / 'research.json')
    if primary != cfg['evaluation']['primary']:
        raise ValueError('Metric differs from prepared config; edit source config and prepare a new workspace')
    manifest = verify_data(workspace, cfg)
    path = workspace / 'protocol.lock.json'
    if path.exists():
        raise FileExistsError('Protocol already finalized; do not alter it after seeing results')
    receipt = {'config_sha256': digest(cfg), 'data_sha256': digest(manifest), 'code': code_identity(),
               'primary': primary, 'candidate_steps': cfg['pre_rl_checkpoints'],
               'checkpoint_selection': 'fixed_before_results_no_significance_selection',
               'accepted_design': True}
    write(path, receipt)
    return receipt


def locked(workspace):
    workspace = Path(workspace)
    cfg = config(workspace / 'research.json')
    manifest = verify_data(workspace, cfg)
    path = workspace / 'protocol.lock.json'
    if not path.exists():
        raise ValueError('Research design/primary metric not finalized. Review then run finalize.')
    receipt = read(path)
    if (receipt['config_sha256'] != digest(cfg) or receipt['data_sha256'] != digest(manifest)
            or receipt['code'] != code_identity()):
        raise ValueError('Finalized protocol/data/code changed; use a new workspace')
    return cfg, receipt


def costs(cfg, counts=None):
    nseed = len(cfg['seeds'])
    main_runs = nseed * 3 * 2
    pre_updates = nseed * max(cfg['pre_rl_checkpoints'])
    main_updates = main_runs * 300
    tasks = main_runs * 4
    # Common sampling seed makes S0's baseline identical across seeds, and each
    # S1/S2 baseline is shared by its random/curriculum arms.
    distinct = 1 + 2*nseed + main_runs*3
    exam_size = (counts or {}).get('math500', 500)
    aux_size = sum(v for k, v in (counts or {}).items() if k not in ('main', 'validation', 'pre_rl', 'math500'))
    weight = cfg['cost_estimates']['model_parameters'] * cfg['cost_estimates']['weight_bytes']
    adapter = cfg['cost_estimates']['lora_parameters'] * cfg['cost_estimates']['adapter_bytes']
    checkpoints = nseed*2 + main_runs*3
    return {'training_updates': {'pre_rl': pre_updates, 'main': main_updates, 'total': pre_updates+main_updates},
            'training_completions': (pre_updates+main_updates)*cfg['training']['num_generations'],
            'main_evaluation_requests_before_cache': tasks,
            'distinct_main_model_states_estimate': distinct,
            'probe_completions_before_cache': tasks*300*8,
            'probe_completions_after_baseline_dedup_estimate': distinct*300*8,
            'evaluation_completions_before_cache': tasks*(exam_size+aux_size)*9,
            'evaluation_completions_after_baseline_dedup_estimate': distinct*(exam_size+aux_size)*9,
            'pre_rl_validation_completions_estimate': (1+2*nseed)*cfg['validation_per_group']*3*8,
            'contamination_completions_estimate': (1+2*nseed)*500,
            'generated_token_upper_bounds_after_dedup': {
                'training': (pre_updates+main_updates)*cfg['training']['num_generations']*cfg['training']['max_completion_length'],
                'probe': distinct*300*8*cfg['training']['max_completion_length'],
                'evaluation': distinct*(exam_size+aux_size)*9*cfg['evaluation']['max_completion_length']},
            'all_merged_weights_GB_estimate': round((1+checkpoints)*weight/1e9, 2),
            'adapter_and_adam_states_GB_estimate': round(checkpoints*adapter*3/1e9, 2),
            'notes': ['Estimates exclude downloads, tokenizer files, logs and temporary merge memory.',
                      'Actual cache reuse requires identical model content, data, settings and inference seed.',
                      'No wall-time estimate without measured hardware throughput; no GPU work is launched.']}
