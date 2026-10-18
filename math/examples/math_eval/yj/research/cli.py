"""Single user-facing entry: python yj/train.py research COMMAND."""
import argparse
import json
from pathlib import Path
import sys

from . import protocol as p


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, default=p.ROOT / 'studies/grpo_v1')
    sub = ap.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='No GPU: proposed design, conflicts and cost estimates')
    plan.add_argument('--config', type=Path, default=p.ROOT / 'research.json')
    prep = sub.add_parser('prepare', help='Prepare disjoint main/pre-RL/validation/evaluation data')
    prep.add_argument('--config', type=Path, default=p.ROOT / 'research.json')
    prep.add_argument('--offline', action='store_true')
    freeze = sub.add_parser('finalize', help='Explicitly freeze design and primary metric before full training')
    freeze.add_argument('--primary', required=True, choices=['math500.avg@8', 'math500.greedy_accuracy'])
    freeze.add_argument('--accept-design', action='store_true')
    sub.add_parser('inspect-loss', help='Audit installed TRL loss source, no GPU')
    server = sub.add_parser('server-check', help='A100 80GB hardware/environment/tokenizer audit; no training or generation')
    server.add_argument('--gpus', nargs='+', default=[str(i) for i in range(8)])
    server.add_argument('--python', required=True, help='Existing A100 evaluation environment Python')
    server.add_argument('--download-base', action='store_true', help='Download pinned Base and audit every prompt in both environments')
    launch = sub.add_parser('launch', help='One independent experiment per GPU; default is a plan only')
    launch.add_argument('--phase', required=True, choices=['pretrain','diagnose','main','contamination'])
    launch.add_argument('--gpus', nargs='+', default=[str(i) for i in range(8)])
    launch.add_argument('--python', required=True, help='Existing A100 evaluation environment Python')
    launch.add_argument('--execute', action='store_true', help='Actually launch GPU processes')
    launch.add_argument('--resume-completed', action='store_true', help='Verify/skip completed training, reuse evaluation caches; never resume a failed trainer')
    launch.add_argument('--limit', type=int, help='Run only the first N jobs, useful for the first server run')
    for name in ('pretrain', 'train'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--seed', type=int, required=True)
        if name == 'train':
            cmd.add_argument('--state', choices=['S0', 'S1', 'S2'], required=True)
            cmd.add_argument('--mode', choices=['random', 'curriculum'], required=True)
        cmd.add_argument('--dry-run', action='store_true')
    export = sub.add_parser('export', help='Merge one checkpoint; registers S1/S2 for pre-RL exports')
    export.add_argument('--run-dir', type=Path, required=True)
    export.add_argument('--step', type=int, required=True)
    for name in ('probe', 'evaluate', 'diagnose', 'contamination'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--seed', type=int, required=True)
        if name in ('probe', 'evaluate'):
            cmd.add_argument('--state', choices=['S0', 'S1', 'S2'], required=True)
            cmd.add_argument('--mode', choices=['random', 'curriculum'], required=True)
            cmd.add_argument('--steps', type=int, nargs='+')
        cmd.add_argument('--backend', choices=['a100', 'hf', 'vllm', 'vllm-lora'], default='hf' if name == 'contamination' else None,
                         help='Default: finalized evaluation.backend; raw-prefix contamination defaults to hf')
        cmd.add_argument('--python', default=sys.executable, help='Separate A100/vLLM inference Python executable')
    stats = sub.add_parser('statistics')
    stats.add_argument('--step', type=int, default=300)
    compare = sub.add_parser('compare-backends', help='Explicit GPU compatibility experiment; never run automatically')
    compare.add_argument('--run-dir', type=Path, required=True)
    compare.add_argument('--step', type=int, required=True)
    compare.add_argument('--limit', type=int, default=10)
    compare.add_argument('--python', default=sys.executable)
    clean = sub.add_parser('cleanup', help='Default dry run; only main merged models after completed eval AND probe')
    clean.add_argument('--run-dir', type=Path, required=True)
    clean.add_argument('--step', type=int, required=True)
    clean.add_argument('--delete', action='store_true', help='Explicitly delete only the reviewed derived merged directory')
    return ap


CONFLICTS = [
    {'old': 'V1 + external SimpleRL-Zoo V2', 'proposal': 'S0=Qwen2.5-1.5B; S1/S2 from one pre-RL trajectory per seed; external V2 is not a matched state'},
    {'old': 'seed 42 only', 'proposal': 'seeds 42/43/44; same fixed main pool; seed-specific within-stage and global shuffles'},
    {'old': '90-question greedy final/staged evaluation', 'proposal': 'primary candidate MATH500 avg@8 at T=0.8, top_p=0.95; greedy auxiliary; explicit finalize required'},
    {'old': 'beta=0.04 and implicit TRL options', 'proposal': 'beta=0 fixed; audited token-mean BNPO + scale_rewards=True; explicit truncation mask; custom yj_grpo_v1 loop'},
    {'old': 'TRL 0.16.1 has no loss_type or mask_truncated_completions fields', 'proposal': 'retain audited formula in explicit tested local loss; do not pass unsupported flags or silently upgrade TRL'},
    {'old': 'training prompt skipped parse_question; generation used EOS only', 'proposal': 'execute protected evaluator prompt statements; compare exact text/token IDs; common EOS and stop strings'},
    {'old': 'A100 evaluator uses model-path-dependent Qwen stop IDs', 'proposal': 'Qwen exports retain qwen2 in path; reject ambiguous paths; tokenizer and stops audited'},
    {'old': 'evaluation-only script has no LoRA or raw-prefix mode', 'proposal': 'keep A100 file unchanged; optional isolated vLLM adapter/raw-prefix inference and empirical compatibility report'},
    {'old': 'one Python training entry', 'proposal': 'keep train.py as entry; research support modules separated for testability; old artifacts retained'},
]


def main(argv=None):
    args = parser().parse_args(argv)
    workspace = args.workspace.resolve()
    if args.command == 'plan':
        cfg = p.config(args.config)
        counts = {s['name']: len(p.lines(args.config.resolve().parent / s['path'])) for s in cfg['evaluation']['auxiliary_sets']}
        result = {'status': 'proposed; not finalized', 'conflicts': CONFLICTS, 'config': cfg, 'costs': p.costs(cfg, counts),
                  'no_gpu_work': True, 'workspace': str(workspace)}
    elif args.command == 'prepare':
        result = p.prepare(args.config, workspace, args.offline)
    elif args.command == 'finalize':
        result = p.finalize(workspace, args.primary, args.accept_design)
    elif args.command == 'inspect-loss':
        from .runtime import loss_audit
        result = loss_audit()
    elif args.command == 'server-check':
        from .parallel import server_check
        result = server_check(workspace,args.gpus,args.python,args.download_base)
    elif args.command == 'launch':
        from .parallel import launch
        result = launch(workspace,args.phase,args.gpus,args.python,args.execute,args.resume_completed,args.limit)
    elif args.command in ('pretrain', 'train'):
        from .workflow import train_run
        result = train_run(workspace, args.seed, getattr(args, 'state', None), getattr(args, 'mode', None), args.dry_run)
    elif args.command == 'export':
        from .workflow import export_checkpoint, state_file
        cfg, _ = p.locked(workspace)
        result = export_checkpoint(workspace, args.run_dir, args.step)
        run = p.read(args.run_dir / 'run.json')
        if run['kind'] == 'pre_rl':
            if run['status'] != 'trained' or args.step not in cfg['pre_rl_checkpoints']:
                raise ValueError('Only completed preregistered pre-RL checkpoints can become S1/S2')
            state = ('S1', 'S2')[cfg['pre_rl_checkpoints'].index(args.step)]
            p.write(state_file(workspace, run['seed'], state), {
                'state': state, 'path': result['path'], 'model_sha256': result['model_sha256'],
                'source': run['source']['source'], 'pre_rl_trajectory': run['trajectory_id'],
                'pre_rl_step': args.step, 'seed': run['seed'], 'paired_states': ['S1', 'S2'],
                'not_independent_starting_states': True, 'adapter_checkpoint': run['checkpoints'][str(args.step)]})
    elif args.command in ('probe', 'evaluate'):
        from .inference import run_inference
        result = run_inference(workspace, args.seed, args.state, args.mode, args.command, args.backend, args.python, args.steps)
    elif args.command in ('diagnose', 'contamination'):
        from .inference import diagnose_states
        result = diagnose_states(workspace, args.seed, 'validation' if args.command == 'diagnose' else 'contamination', args.backend, args.python)
    elif args.command == 'statistics':
        from .metrics import statistics_from_workspace
        result = statistics_from_workspace(workspace, args.step)
    elif args.command == 'compare-backends':
        from .inference import compare_backends
        result = compare_backends(workspace, args.run_dir, args.step, args.python, args.limit)
    else:
        from .inference import cleanup_merged
        result = cleanup_merged(workspace, args.run_dir, args.step, args.delete)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
