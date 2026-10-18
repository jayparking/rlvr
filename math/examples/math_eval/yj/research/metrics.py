"""Group diagnostics, reconstruction metrics and paired multi-seed inference."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import random
import re
from statistics import mean

from .protocol import read, write


def summarize_samples(items, g=4):
    rows = []
    for item in items:
        samples = item['samples']
        if not samples:
            raise ValueError('Empty sample group')
        if any(type(x['correct']) is not bool for x in samples):
            raise ValueError('Scores must be Boolean')
        k, correct = len(samples), sum(x['correct'] for x in samples)
        p = correct/k
        row = {'id': item['id'], 'k': k, 'correct': correct, 'p': p,
               'all_wrong': correct == 0, 'mixed': 0 < correct < k, 'all_correct': correct == k,
               'expected_effective_group_probability': 1-p**g-(1-p)**g,
               'truncated': sum(x['truncated'] for x in samples),
               'grading_errors': sum(x.get('grading_error') is not None for x in samples)}
        if k >= g:
            choose = lambda n: math.comb(n, g) if n >= g else 0
            row['effective_group_probability_u_statistic'] = 1-(choose(correct)+choose(k-correct))/math.comb(k, g)
        rows.append(row)
    if not rows:
        raise ValueError('Empty dataset')
    ks = {r['k'] for r in rows}
    if len(ks) != 1:
        raise ValueError('All problems must have the same k')
    k = next(iter(ks))
    return {'training_group_size': g, 'problems': rows, 'summary': {
        f'avg@{k}': mean(r['p'] for r in rows),
        f'pass@{k}': mean(not r['all_wrong'] for r in rows),
        'greedy_accuracy': mean(r['p'] for r in rows) if k == 1 else None,
        'all_wrong_fraction': mean(r['all_wrong'] for r in rows),
        'mixed_fraction': mean(r['mixed'] for r in rows),
        'all_correct_fraction': mean(r['all_correct'] for r in rows),
        'expected_effective_group_probability_mean': mean(r['expected_effective_group_probability'] for r in rows),
        'truncation_rate': sum(r['truncated'] for r in rows)/(len(rows)*k),
        'grading_error_rate': sum(r['grading_errors'] for r in rows)/(len(rows)*k)},
        'probability_note': 'p=c/k; 1-p^G-(1-p)^G assumes iid reward samples. Plug-in estimate is noisy at k=8 and excludes truncation masking effects. U-statistic uses all size-G subsets.'}


def reconstruction_split(question, fraction=0.6):
    spans = list(re.finditer(r'\S+', question))
    if len(spans) < 2:
        raise ValueError('Question too short for a prefix diagnostic')
    count = max(1, min(len(spans)-1, int(len(spans)*fraction)))
    boundary = spans[count].start()
    return question[:boundary], question[boundary:]


def rouge_l(prediction, reference):
    a, b = prediction.split(), reference.split()
    if not a or not b:
        return float(a == b)
    previous = [0]*(len(b)+1)
    for word in a:
        current = [0]
        for j, target in enumerate(b):
            current.append(previous[j]+1 if word == target else max(previous[j+1], current[-1]))
        previous = current
    return 2*previous[-1]/(len(a)+len(b))


def reconstruction_scores(prediction, reference):
    return {'exact_match': prediction.strip() == reference.strip(),
            'whitespace_normalized_exact_match': ' '.join(prediction.split()) == ' '.join(reference.split()),
            'rouge_l_f1': rouge_l(prediction, reference)}


def quantile(values, p):
    values = sorted(values)
    at = (len(values)-1)*p
    lo = int(at)
    hi = min(lo+1, len(values)-1)
    return values[lo]*(hi-at) + values[hi]*(at-lo) if hi != lo else values[lo]


def paired_bootstrap(records, repeats=5000, confidence=0.95, seed=20261018, primary='avg@8'):
    """Seed clusters sampled jointly across states/arms, then shared problem indices.

    Questions are crossed with seeds (the same MATH500 questions for everyone),
    so each replicate shares its problem draw across all sampled seed clusters.
    """
    if repeats < 2 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap parameters')
    cells = {}
    for r in records:
        key = (r['seed'], r['state'], r['mode'])
        if key in cells:
            raise ValueError(f'Duplicate cell: {key}')
        if r['metric'] != primary:
            raise ValueError('Do not mix avg@8, pass@8 or greedy results')
        pairs = [(x['id'], x['value']) for x in r['problems']]
        if len({x[0] for x in pairs}) != len(pairs) or any(not 0 <= x[1] <= 1 for x in pairs):
            raise ValueError('Invalid problem scores')
        cells[key] = dict(pairs)
    seeds = sorted({s for s, _, _ in cells})
    states = ['S0', 'S1', 'S2']
    if len(seeds) < 2:
        raise ValueError('Need multiple independent seed trajectories')
    required = {(s, state, mode) for s in seeds for state in states for mode in ('random', 'curriculum')}
    if set(cells) != required:
        raise ValueError('Need all S0/S1/S2 x Random/Curriculum cells for every seed')
    ids = sorted(next(iter(cells.values())))
    if not ids or any(set(c) != set(ids) for c in cells.values()):
        raise ValueError('All cells must pair on exactly the same problem IDs')
    delta = {state: [[cells[s, state, 'curriculum'][i]-cells[s, state, 'random'][i] for i in ids] for s in seeds] for state in states}
    names = states + ['S1-S0', 'S2-S0', 'S2-S1']
    distributions = {n: [] for n in names}
    rng = random.Random(seed)
    for _ in range(repeats):
        seed_indices = rng.choices(range(len(seeds)), k=len(seeds))
        question_indices = rng.choices(range(len(ids)), k=len(ids))
        draws = {state: sum(delta[state][s][q] for s in seed_indices for q in question_indices)/(len(seeds)*len(ids)) for state in states}
        for pair in names[3:]:
            a, b = pair.split('-')
            draws[pair] = draws[a]-draws[b]
        for name in names:
            distributions[name].append(draws[name])
    point = {s: mean(mean(v) for v in delta[s]) for s in states}
    for pair in names[3:]:
        a, b = pair.split('-')
        point[pair] = point[a]-point[b]
    alpha = (1-confidence)/2
    raw = []
    for s in seeds:
        for state in states:
            random_score = mean(cells[s, state, 'random'].values())
            curriculum_score = mean(cells[s, state, 'curriculum'].values())
            raw.append({'seed': s, 'state': state, 'random': random_score, 'curriculum': curriculum_score,
                        'curriculum_minus_random': curriculum_score-random_score})
    return {'metric': primary, 'seeds': seeds, 'problems': len(ids), 'bootstrap_replicates': repeats,
            'confidence': confidence, 'bootstrap_seed': seed, 'raw_by_seed': raw,
            'effects': {n: {'estimate_percentage_points': 100*point[n],
                           'ci_percentage_points': [100*quantile(distributions[n], alpha), 100*quantile(distributions[n], 1-alpha)]} for n in names},
            'method': 'paired seed-cluster bootstrap with a common paired question draw per replicate; S1/S2 retained within trajectory clusters',
            'limitations': 'Three seeds give only three training-trajectory clusters. Intervals are exploratory and may be unstable or under-cover; more question resamples do not create more independent training runs. No significance-based checkpoint selection.'}


def statistics_from_workspace(workspace, step=300):
    from .protocol import locked, digest, lines
    from .inference import load_cache, inference_settings
    workspace = Path(workspace)
    cfg, receipt = locked(workspace)
    if step not in cfg['main_checkpoints']:
        raise ValueError('Checkpoint outside finalized protocol')
    metric = 'avg@8' if cfg['evaluation']['primary'].endswith('avg@8') else 'greedy_accuracy'
    kind = 'eval_avg8' if metric == 'avg@8' else 'eval_greedy'
    exam = lines(workspace / 'data/math500.jsonl')
    records = []
    for seed in cfg['seeds']:
        for state in ('S0', 'S1', 'S2'):
            for mode in ('random', 'curriculum'):
                ref = read(workspace / 'runs' / f'{state}_seed{seed}_{mode}' / 'results' / f'{kind}_math500_step{step}.json')
                directory = workspace / 'cache' / ref['cache_key']
                request = read(directory / 'request.json')
                result = load_cache(directory, request)
                if (result is None or digest(request) != ref['cache_key']
                        or ref['protocol_sha256'] != digest(receipt) or ref['seed'] != seed
                        or ref['state'] != state or ref['mode'] != mode or ref['step'] != step
                        or request['kind'] != kind or request['model_sha256'] != ref['model_sha256']
                        or request['dataset_sha256'] != digest(exam)
                        or request['settings'] != inference_settings(cfg, kind)
                        or [i['id'] for i in result['items']] != [r['id'] for r in exam]):
                    raise ValueError('Unpaired, incomplete or incompatible evaluation cache')
                summary = summarize_samples(result['items'])
                if any(len(item['samples']) != request['settings']['k'] for item in result['items']):
                    raise ValueError('Incorrect completion count for finalized primary metric')
                records.append({'seed': seed, 'state': state, 'mode': mode, 'metric': metric,
                                'backend': request['engine']['backend'], 'engine': request['engine'],
                                'problems': [{'id': r['id'], 'value': r['p']} for r in summary['problems']]})
    if len({digest(r['engine']) for r in records}) != 1:
        raise ValueError('Do not mix inference backends/versions in the primary comparison')
    if records[0]['backend'] != cfg['evaluation']['backend']:
        raise ValueError('Primary results must use the finalized evaluation backend')
    opts = cfg['statistics']
    report = paired_bootstrap(records, opts['bootstrap_replicates'], opts['confidence'], opts['seed'], metric)
    destination = workspace / 'statistics' / f'step{step}'
    write(destination / 'bootstrap.json', report)
    write(destination / 'raw_problem_scores.json', records)
    with (destination / 'seed_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['raw_by_seed'][0]))
        writer.writeheader()
        writer.writerows(report['raw_by_seed'])
    return report
