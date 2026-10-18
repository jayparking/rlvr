"""One host, one independent experiment per GPU; never distributed GRPO."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import io
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import uuid

from . import protocol as p
from .workflow import run_dir, load_state


def gpu_inventory(gpus, require_idle=False):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Specify distinct GPU indices or UUIDs')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = []
    for row in csv.reader(io.StringIO(raw)):
        index, identity, name, memory = [x.strip() for x in row]
        devices.append({'index': index, 'uuid': identity, 'name': name, 'memory_mib': int(memory)})
    lookup = {key: d for d in devices for key in (d['index'], d['uuid'])}
    if any(g not in lookup for g in gpus):
        raise ValueError('Requested GPU not found in nvidia-smi inventory')
    selected = [lookup[g] for g in gpus]
    if len({d['uuid'] for d in selected}) != len(selected):
        raise ValueError('GPU index/UUID aliases refer to the same device')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is not None:
        allowed = {lookup[x.strip()]['uuid'] for x in inherited.split(',') if x.strip() in lookup}
        if any(d['uuid'] not in allowed for d in selected):
            raise ValueError('Requested GPUs exceed inherited CUDA_VISIBLE_DEVICES allocation')
    if any('A100' not in d['name'] or d['memory_mib'] < 75000 for d in selected):
        raise ValueError('This launch profile requires full A100 80GB GPUs (not 40GB/MIG slices)')
    if require_idle:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader,nounits'], text=True)
        occupied = {r[0].strip() for r in csv.reader(io.StringIO(active)) if r}
        if occupied & {d['uuid'] for d in selected}:
            raise ValueError('Selected GPU already has a compute process; inspect nvidia-smi before launching')
    return selected


def verify_completed_run(workspace, run, cfg, receipt, seed, state=None, mode=None):
    record = p.read(run / 'run.json')
    stages = cfg['pre_rl_checkpoints'] if state is None else cfg['main_checkpoints'][1:]
    expected = {'kind': 'pre_rl' if state is None else 'main', 'seed': seed,
                'state': state or 'S0', 'mode': mode or 'random'}
    if (record.get('status') != 'trained' or record.get('protocol_sha256') != p.digest(receipt)
            or any(record.get(k) != v for k, v in expected.items())
            or record.get('optimizer_steps') != max(stages)
            or set(record['checkpoints']) != {str(s) for s in stages}):
        raise ValueError(f'Cannot resume incomplete/incompatible training: {run}')
    if p.model_identity(record['source']['path'])['sha256'] != record['source']['model_sha256']:
        raise ValueError(f'Starting weights changed: {run}')
    for checkpoint in record['checkpoints'].values():
        if p.model_identity(checkpoint['path'])['sha256'] != checkpoint['sha256']:
            raise ValueError(f'Original adapter changed: {run}')


def build_jobs(workspace, cfg, phase, eval_python, train_python=sys.executable, resume_completed=False,
               receipt=None, execute=False, limit=None):
    workspace = Path(workspace).resolve()
    prefix = [str(train_python), str(p.ROOT / 'train.py'), 'research', '--workspace', str(workspace)]
    jobs = []
    cells = [(s, None, None) for s in cfg['seeds']] if phase != 'main' else [
        (s, state, mode) for s in cfg['seeds'] for state in ('S0','S1','S2') for mode in ('random','curriculum')]
    if limit is not None:
        if limit < 1: raise ValueError('limit must be positive')
        cells = cells[:limit]
    for seed, state, mode in cells:
        run = run_dir(workspace, seed, state, mode)
        skipped = False
        if phase in ('pretrain', 'main') and run.exists():
            if not resume_completed:
                if execute: raise FileExistsError(f'{run} exists; use --resume-completed for verified completed runs')
            else:
                if receipt is None: raise ValueError('Resume requires a finalized protocol')
                verify_completed_run(workspace, run, cfg, receipt, seed, state, mode)
                skipped = True
        commands = []
        if phase == 'pretrain':
            if not skipped: commands.append(prefix + ['pretrain','--seed',str(seed)])
            # Also repair a completed pre-RL run whose merge/registration was interrupted.
            commands += [prefix + ['export','--run-dir',str(run),'--step',str(step)] for step in cfg['pre_rl_checkpoints']]
        elif phase == 'main':
            if execute:
                load_state(workspace, seed, state)
            flags = ['--seed',str(seed),'--state',state,'--mode',mode]
            if not skipped: commands.append(prefix + ['train'] + flags)
            commands += [prefix + [name] + flags + ['--python',str(eval_python)] for name in ('probe','evaluate')]
        else:
            if execute:
                for name in ('S0','S1','S2'): load_state(workspace, seed, name)
            commands = [prefix + [phase,'--seed',str(seed),'--python',str(eval_python)]]
        jobs.append({'id': f'{phase}_seed{seed}' if state is None else f'{state}_seed{seed}_{mode}',
                     'commands': commands, 'training_already_complete': skipped})
    return jobs


def run_queue(jobs, devices, directory, base_env=None, plan=None):
    """Testable process scheduler. A failed command stops new dispatch, never later steps."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    p.write(directory/'plan.json', plan if plan is not None else {'jobs':jobs,'devices':devices})
    pending = queue.Queue()
    for job in jobs: pending.put(job)
    stop = threading.Event()
    mutex = threading.Lock()
    running = {}
    outcomes = {}
    env_template = dict(os.environ if base_env is None else base_env)
    env_template.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                        VLLM_WORKER_MULTIPROC_METHOD='spawn', CUDA_DEVICE_ORDER='PCI_BUS_ID')

    def save():
        p.write(directory / 'status.json', {'jobs': outcomes, 'stop_new_dispatch': stop.is_set()})

    def worker(device):
        while True:
            with mutex:
                if stop.is_set(): return
                try: job = pending.get_nowait()
                except queue.Empty: return
                outcomes[job['id']] = {'status': 'running', 'gpu': device['uuid'], 'commands_completed': 0}
                save()
            env = dict(env_template, CUDA_VISIBLE_DEVICES=device['uuid'])
            try:
                for number, command in enumerate(job['commands']):
                    with mutex:
                        if stop.is_set():
                            outcomes[job['id']]['status'] = 'interrupted_between_commands'; save(); return
                    with (directory / f'{job["id"]}_{number}.log').open('w') as log:
                        process = subprocess.Popen(command, cwd=p.EVAL_ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        with mutex: running[job['id']] = process
                        code = process.wait()
                        with mutex:
                            running.pop(job['id'], None)
                            outcomes[job['id']]['last_exit_code'] = code
                            if code != 0:
                                outcomes[job['id']]['status'] = 'failed'; stop.set(); save(); return
                            outcomes[job['id']]['commands_completed'] += 1
                            save()
                with mutex: outcomes[job['id']]['status'] = 'complete'; save()
            except Exception as error:
                with mutex:
                    outcomes[job['id']].update(status='failed', error=f'{type(error).__name__}: {error}')
                    stop.set(); save()
                return

    def terminate_children():
        with mutex:
            stop.set()
            processes = list(running.values())
        for process in processes:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        for process in processes:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass

    executor = ThreadPoolExecutor(max_workers=len(devices))
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures: future.result()
    except BaseException:
        terminate_children()
        raise
    finally:
        executor.shutdown(wait=True)
        if previous_handler is not None: signal.signal(signal.SIGTERM, previous_handler)
        with mutex:
            for job in jobs:
                outcomes.setdefault(job['id'], {'status': 'not_started'})
            save()
    if any(r['status'] != 'complete' for r in outcomes.values()):
        raise RuntimeError(f'Queue stopped with incomplete jobs. See {directory / "status.json"}')
    return outcomes


def launch(workspace, phase, gpus, eval_python, execute=False, resume_completed=False, limit=None):
    workspace = Path(workspace).resolve()
    cfg = p.config(workspace / 'research.json')
    receipt = p.locked(workspace)[1] if execute or resume_completed else None
    jobs = build_jobs(workspace,cfg,phase,eval_python,resume_completed=resume_completed,
                      receipt=receipt,execute=execute,limit=limit)
    plan = {'phase':phase,'gpu_slots':gpus,'jobs':jobs,'execution_requested':execute,
            'distribution':'one independent run per GPU; no DDP/torchrun; G and optimizer steps unchanged',
            'requires':'one host, shared local filesystem; sequential phase barriers',
            'no_automatic_cleanup':True}
    if not gpus or len(set(gpus)) != len(gpus): raise ValueError('Specify distinct GPU slots')
    if not execute: return plan
    check = p.read(workspace/'verification/server-check.json')
    if (check.get('config_sha256') != p.digest(cfg) or check.get('code') != p.code_identity()
            or check.get('eval_python') != os.path.abspath(eval_python) or check.get('eval_tokenizer_parity') is not True
            or check.get('train_python') != os.path.abspath(sys.executable)):
        raise ValueError('Run server-check --download-base with the current config/code and Python environments first')
    with ExitStack() as stack:
        stack.enter_context(p.resource_lock(workspace,'launcher','all-phases',blocking=False))
        devices = gpu_inventory(gpus,require_idle=True)
        for device in sorted(devices,key=lambda x:x['uuid']):
            stack.enter_context(p.resource_lock(p.ROOT,'gpu',socket.gethostname()+':'+device['uuid'],blocking=False))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        directory = workspace / 'launches' / f'{phase}-{stamp}'
        outcomes = run_queue(jobs,devices,directory,plan=dict(plan,devices=devices))
    return {'phase':phase,'directory':str(directory),'jobs':outcomes}


def server_check(workspace, gpus, eval_python, download_base=False):
    """Read GPUs/import environments; optionally download weights and audit all tokenizer inputs. No training."""
    import json
    from train import check_versions
    from . import runtime
    from .workflow import ensure_base
    cfg = p.config(Path(workspace)/'research.json')
    if not Path(eval_python).is_file() or not os.access(eval_python,os.X_OK):
        raise ValueError('Pass the executable path of the evaluation virtualenv Python')
    p.verify_data(workspace,cfg)
    check_versions()
    report = {'gpus':gpu_inventory(gpus), 'loss':runtime.loss_audit(), 'training_or_generation':False,
              'config_sha256':p.digest(cfg),'code':p.code_identity(),
              # Preserve virtualenv symlink paths: resolve() can collapse two
              # different environments to the same system Python binary.
              'eval_python':os.path.abspath(eval_python),'train_python':os.path.abspath(sys.executable)}
    import shutil
    report['disk_free_bytes'] = shutil.disk_usage(workspace).free
    report['cost_estimates'] = p.costs(cfg,p.read(Path(workspace)/'data/manifest.json')['counts'])
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        report['host_memory_kib'] = {line.split(':')[0]:int(line.split()[1]) for line in meminfo.read_text().splitlines()
                                     if line.startswith(('MemTotal:','MemAvailable:'))}
    report['cuda_checks'] = []
    for device in report['gpus']:
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
        script = ('import json,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; '
                  'assert torch.cuda.is_bf16_supported(); '
                  'print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"name":torch.cuda.get_device_name(),'
                  '"memory_bytes":torch.cuda.get_device_properties(0).total_memory}))')
        report['cuda_checks'].append(json.loads(subprocess.check_output([sys.executable,'-c',script],env=env,text=True)))
    script = 'import json,torch,importlib.metadata as m; import math_eval_a100; print(json.dumps({k:m.version(k) for k in ["torch","transformers","vllm"]}))'
    result = subprocess.check_output([str(eval_python),'-c',script],cwd=p.EVAL_ROOT,text=True)
    report['eval_environment'] = json.loads(result.strip().splitlines()[-1])
    if download_base:
        target = ensure_base(workspace,cfg['seeds'][0],cfg)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(target['path'])
        if runtime.eos_ids(tokenizer) != [151643,151645]: raise ValueError('Unexpected Qwen EOS IDs')
        names = ['main','pre_rl','validation','math500'] + [s['name'] for s in cfg['evaluation']['auxiliary_sets']]
        audits = {}
        for name in names:
            rows = p.lines(Path(workspace)/'data'/f'{name}.jsonl')
            audit = runtime.prompt_audit(tokenizer,rows)
            limit = cfg['training']['max_prompt_length'] if name in ('main','pre_rl','validation') else 8192-cfg['evaluation']['max_completion_length']
            if max(r['tokens'] for r in audit['rows']) > limit: raise ValueError(f'{name} exceeds finalized prompt length')
            audits[name] = audit
        report['tokenizer_audits'] = audits
        payload = Path(workspace)/'verification/server-tokenizer-input.json'
        p.write(payload,{name:p.lines(Path(workspace)/'data'/f'{name}.jsonl') for name in names})
        script = ('import sys,json; sys.path.insert(0,sys.argv[1]); from research.runtime import prompt_audit; '
                  'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(sys.argv[2]); '
                  'print(json.dumps({n:prompt_audit(t,r) for n,r in json.load(open(sys.argv[3])).items()}))')
        actual = subprocess.check_output([str(eval_python),'-c',script,str(p.ROOT),target['path'],str(payload)],text=True)
        if json.loads(actual.strip().splitlines()[-1]) != audits: raise ValueError('Training/evaluation tokenization differs')
        report['eval_tokenizer_parity'] = True
    p.write(Path(workspace)/'verification/server-check.json',report)
    return report
