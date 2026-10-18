import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from research import protocol as p, parallel
from test_research import fixture_workspace, cfg


class ParallelTests(unittest.TestCase):
    def test_eight_processes_share_one_producer_and_atomic_writes(self):
        script = '''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from research import protocol as p
root=Path(sys.argv[2]); number=int(sys.argv[3])
with p.resource_lock(root,'sample','same-model-and-settings'):
    with p.resource_lock(root,'sample','same-model-and-settings'):
        if not (root/'result.json').exists():
            with (root/'producers.txt').open('a') as f: f.write(str(number)+'\\n')
            time.sleep(.05)
            p.write(root/'result.json',{'producer':number})
for i in range(20): p.write(root/'shared.json',{'writer':number,'payload':'x'*10000,'i':i})
print(p.read(root/'result.json')['producer'])
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            children=[subprocess.Popen([sys.executable,'-c',script,str(p.ROOT),tmp,str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(8)]
            values=[]
            for child in children:
                out,err=child.communicate(timeout=20)
                self.assertEqual(child.returncode,0,err); values.append(out.strip())
            self.assertEqual(len(set(values)),1)
            self.assertEqual(len((root/'producers.txt').read_text().splitlines()),1)
            self.assertEqual(len(p.read(root/'shared.json')['payload']),10000)
            self.assertFalse(list(root.glob('*.tmp')))

    def test_lock_releases_after_process_termination(self):
        script='''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from research.protocol import resource_lock
root=Path(sys.argv[2])
with resource_lock(root,'run','test'):
    (root/'ready').write_text('ready')
    time.sleep(30)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=subprocess.Popen([sys.executable,'-c',script,str(p.ROOT),tmp])
            try:
                deadline=time.monotonic()+5
                while not (root/'ready').exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue((root/'ready').exists())
                with self.assertRaisesRegex(RuntimeError,'already running'):
                    with p.resource_lock(root,'run','test',blocking=False): pass
            finally:
                child.terminate(); child.wait(timeout=5)
            with p.resource_lock(root,'run','test',blocking=False): pass

    def test_eighteen_jobs_use_eight_exclusive_gpu_slots(self):
        script='''import sys,time,json,os
from pathlib import Path
path=Path(sys.argv[1]); start=time.monotonic(); time.sleep(.04)
path.write_text(json.dumps({'start':start,'end':time.monotonic(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'threads':os.environ['OMP_NUM_THREADS']}))
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=[{'id':f'job{i}','commands':[[sys.executable,'-c',script,str(root/f'event{i}-{j}.json')] for j in range(2)]} for i in range(18)]
            devices=[{'uuid':f'GPU-test-{i}'} for i in range(8)]
            result=parallel.run_queue(jobs,devices,root/'queue')
            self.assertEqual(len(result),18)
            self.assertTrue(all(r['status']=='complete' and r['commands_completed']==2 for r in result.values()))
            events=[p.read(f) for f in root.glob('event*.json')]
            self.assertEqual(len(events),36)
            self.assertEqual({e['gpu'] for e in events},{d['uuid'] for d in devices})
            for device in devices:
                ordered=sorted([e for e in events if e['gpu']==device['uuid']],key=lambda e:e['start'])
                self.assertTrue(all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])))
            points=sorted([(e['start'],1) for e in events]+[(e['end'],-1) for e in events])
            active=maximum=0
            for _,delta in points: active+=delta; maximum=max(maximum,active)
            self.assertGreater(maximum,1); self.assertLessEqual(maximum,8)
            self.assertEqual({e['threads'] for e in events},{'2'})

    def test_failure_stops_following_commands_and_new_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write=[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).touch()',str(root/'should-not-exist')]
            jobs=[{'id':'fails','commands':[[sys.executable,'-c','raise SystemExit(7)'],write]},
                  {'id':'not_started','commands':[write]}]
            with self.assertRaisesRegex(RuntimeError,'incomplete jobs'):
                parallel.run_queue(jobs,[{'uuid':'GPU-test'}],root/'queue')
            self.assertFalse((root/'should-not-exist').exists())
            status=p.read(root/'queue/status.json')['jobs']
            self.assertEqual(status['fails']['last_exit_code'],7)
            self.assertEqual(status['not_started']['status'],'not_started')

    def test_dry_launch_has_all_paired_cells_and_never_touches_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c,_=fixture_workspace(root)
            with patch.object(parallel,'gpu_inventory',side_effect=AssertionError('No GPU access')), \
                 patch.object(parallel,'run_queue',side_effect=AssertionError('No child launch')):
                plan=parallel.launch(root,'main',[str(i) for i in range(8)],'/eval/python')
            self.assertEqual(len(plan['jobs']),18)
            self.assertEqual(len({j['id'] for j in plan['jobs']}),18)
            for job in plan['jobs']:
                self.assertEqual([command[5] for command in job['commands']],['train','probe','evaluate'])
                self.assertEqual(job['commands'][0][0],sys.executable)
                self.assertEqual(job['commands'][1][-1],'/eval/python')
            self.assertEqual(p.read(root/'research.json'),c)
            self.assertFalse((root/'protocol.lock.json').exists())

    def test_inventory_rejects_aliases_wrong_memory_and_busy_devices(self):
        inventory='0, GPU-zero, NVIDIA A100-SXM4-80GB, 81920\n1, GPU-one, NVIDIA A100-SXM4-80GB, 81920\n'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1'}):
            with patch.object(parallel.subprocess,'check_output',return_value=inventory):
                self.assertEqual(parallel.gpu_inventory(['0','1'])[1]['uuid'],'GPU-one')
                with self.assertRaises(ValueError): parallel.gpu_inventory(['0','GPU-zero'])
            with patch.object(parallel.subprocess,'check_output',return_value=inventory.replace('81920','40960')):
                with self.assertRaisesRegex(ValueError,'80GB'): parallel.gpu_inventory(['0'])
            with patch.object(parallel.subprocess,'check_output',side_effect=[inventory,'GPU-zero, 1234\n']):
                with self.assertRaisesRegex(ValueError,'compute process'): parallel.gpu_inventory(['0'],True)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}), patch.object(parallel.subprocess,'check_output',return_value=inventory):
            with self.assertRaisesRegex(ValueError,'allocation'): parallel.gpu_inventory(['1'])

    def test_execute_requires_server_audit_before_gpu_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c,_=fixture_workspace(root)
            p.finalize(root,c['evaluation']['primary'],True)
            with patch.object(parallel,'gpu_inventory',side_effect=AssertionError('Do not access GPU yet')), \
                 patch.object(parallel,'run_queue',side_effect=AssertionError('Do not launch')):
                with self.assertRaises(FileNotFoundError):
                    parallel.launch(root,'pretrain',['0'],'/eval/python',execute=True)

    def test_resume_only_accepts_verified_completed_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c,_=fixture_workspace(root)
            receipt=p.finalize(root,c['evaluation']['primary'],True)
            run=root/'runs/S0_seed42_random'
            source=root/'base'; source.mkdir(); (source/'model.safetensors').write_bytes(b'base')
            checkpoints={}
            for step in (100,200,300):
                path=run/f'checkpoint-{step}'; path.mkdir(parents=True)
                (path/'adapter_model.safetensors').write_bytes(str(step).encode())
                checkpoints[str(step)]={'path':str(path),'sha256':p.model_identity(path)['sha256']}
            manifest={'status':'trained','kind':'main','seed':42,'state':'S0','mode':'random','optimizer_steps':300,
                      'protocol_sha256':p.digest(receipt),'checkpoints':checkpoints,
                      'source':{'path':str(source),'model_sha256':p.model_identity(source)['sha256']}}
            p.write(run/'run.json',manifest)
            jobs=parallel.build_jobs(root,c,'main','/eval/python',resume_completed=True,receipt=receipt,limit=1)
            self.assertTrue(jobs[0]['training_already_complete'])
            self.assertEqual(jobs[0]['commands'][0][5],'probe')
            manifest['status']='failed'; p.write(run/'run.json',manifest)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                parallel.build_jobs(root,c,'main','/eval/python',resume_completed=True,receipt=receipt,limit=1)


if __name__=='__main__': unittest.main()
