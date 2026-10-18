import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import protocol as p, metrics, runtime, inference, workflow

metrics_bootstrap = metrics.paired_bootstrap


def cfg():
    return p.config()


def fixture_workspace(root):
    c = cfg()
    p.write(root / 'research.json', c)
    pools = {}
    for name, n in [('main', 300), ('validation', 300), ('pre_rl', 1200), ('math500', 500)]:
        pools[name] = [dict(id=f'{name}/{i}', problem=f'{name} question {i}', question=f'{name} question {i}',
                            solution='\\boxed{2}', level=[1, 3, 5][i % 3], difficulty=['easy','medium','hard'][i % 3],
                            fingerprint=p.fingerprint(f'{name} question {i}')) for i in range(n)]
        p.write_lines(root / 'data' / f'{name}.jsonl', pools[name])
    for seed in c['seeds']:
        for mode in ('random', 'curriculum'):
            p.write_lines(root / 'data' / f'main_seed{seed}_{mode}.jsonl', p.schedule(pools['main'], seed, mode))
        p.write_lines(root / 'data' / f'pre_rl_seed{seed}.jsonl', p.schedule(pools['pre_rl'], seed, 'random'))
    manifest = {'config_sha256': p.digest(c), 'counts': {k: len(v) for k,v in pools.items()},
                'files': {f.name: p.file_hash(f) for f in (root / 'data').glob('*.jsonl')}}
    p.write(root / 'data/manifest.json', manifest)
    return c, pools


class ResearchTests(unittest.TestCase):
    def test_invalid_numeric_settings_are_rejected_before_training(self):
        cases = [('training','epsilon_low',-0.2), ('training','max_completion_length',0),
                 ('training','num_generations',2.5), ('training','temperature',float('nan')),
                 ('contamination','top_k',50), ('contamination','split_unit','characters')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            for section, key, value in cases:
                with self.subTest(section=section,key=key):
                    c=cfg(); c[section][key]=value
                    path.write_text(json.dumps(c))
                    with self.assertRaises(ValueError): p.config(path)

    def test_evaluator_dependencies_are_in_protocol_identity(self):
        identity = p.code_identity()
        for name in ('evaluate.py','data_loader.py','python_executor.py','trajectory.py','model_utils.py'):
            self.assertIn(name,identity)

    def test_disjoint_pools_and_seed_paired_schedules(self):
        c = cfg()
        train = [dict(id=f'train/{level}/{i}', problem=f'problem {level} {i}', level=level, solution='2')
                 for level in range(1, 6) for i in range(700)]
        excluded = {p.fingerprint(train[0]['problem'])}
        pools, skipped = p.partition(c, train, excluded)
        self.assertEqual({k: len(v) for k,v in pools.items()}, {'main':300, 'validation':300, 'pre_rl':1200})
        sets = [{r['fingerprint'] for r in rows} for rows in pools.values()]
        self.assertEqual(len(set.union(*sets)), 1800)
        self.assertFalse(set.union(*sets) & excluded)
        previous = None
        for seed in c['seeds']:
            r, q = p.schedule(pools['main'], seed, 'random'), p.schedule(pools['main'], seed, 'curriculum')
            self.assertEqual({x['id'] for x in r}, {x['id'] for x in q})
            self.assertEqual([x['difficulty'] for x in q], ['easy']*100+['medium']*100+['hard']*100)
            if previous is not None:
                self.assertNotEqual([x['id'] for x in q[:100]], previous)
            previous = [x['id'] for x in q[:100]]

    def test_metric_avg_pass_effective_probability_and_truncation(self):
        items = [{'id': str(i), 'samples': [{'correct': x, 'truncated': j == 0} for j,x in enumerate(group)]}
                 for i,group in enumerate([[False]*8, [False]*4+[True]*4, [True]*8])]
        result = metrics.summarize_samples(items)
        self.assertEqual(result['summary']['avg@8'], .5)
        self.assertAlmostEqual(result['summary']['pass@8'], 2/3)
        self.assertEqual(result['problems'][1]['expected_effective_group_probability'], .875)
        self.assertAlmostEqual(result['summary']['truncation_rate'], 1/8)
        self.assertEqual(result['summary']['mixed_fraction'], 1/3)

    def test_paired_bootstrap_preserves_seed_and_question_covariance(self):
        records = []
        for seed, offset in zip((42,43,44), (-.05,0,.05)):
            for state, effect in zip(('S0','S1','S2'), (0,.1,.2)):
                for mode in ('random','curriculum'):
                    rows = [{'id': str(i), 'value': .4 + (offset+effect+i*.01 if mode == 'curriculum' else 0)} for i in range(8)]
                    records.append(dict(seed=seed,state=state,mode=mode,metric='avg@8',problems=rows))
        report = metrics.paired_bootstrap(records, repeats=100)
        self.assertAlmostEqual(report['effects']['S1-S0']['estimate_percentage_points'], 10)
        for bound in report['effects']['S1-S0']['ci_percentage_points']:
            self.assertAlmostEqual(bound, 10)
        lo,hi = report['effects']['S0']['ci_percentage_points']
        self.assertGreater(hi-lo, 1)
        self.assertEqual(len(report['raw_by_seed']), 9)
        self.assertEqual(report, metrics.paired_bootstrap(records, repeats=100))
        with self.assertRaises(ValueError):
            metrics.paired_bootstrap(records[:-1], repeats=10)
        records[0]['problems'][0]['id'] = 'unpaired'
        with self.assertRaisesRegex(ValueError, 'pair'):
            metrics.paired_bootstrap(records, repeats=10)

    def test_prefix_reconstruction_metrics(self):
        prefix, suffix = metrics.reconstruction_split('one two three four five six seven eight nine ten')
        self.assertEqual(prefix, 'one two three four five six ')
        self.assertEqual(suffix, 'seven eight nine ten')
        self.assertEqual(metrics.reconstruction_scores(suffix, suffix)['rouge_l_f1'], 1)
        self.assertFalse(metrics.reconstruction_scores('eight nine', suffix)['exact_match'])
        self.assertAlmostEqual(metrics.rouge_l('eight nine', suffix), 2/3)

    def test_explicit_finalization_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c,_ = fixture_workspace(root)
            with self.assertRaisesRegex(ValueError, 'not finalized'):
                p.locked(root)
            with self.assertRaises(ValueError):
                p.finalize(root, c['evaluation']['primary'], False)
            p.finalize(root, c['evaluation']['primary'], True)
            self.assertEqual(p.locked(root)[0], c)
            c['training']['learning_rate'] *= 2
            p.write(root / 'research.json', c)
            with self.assertRaises(ValueError):
                p.locked(root)

    def test_cache_keys_and_corruption(self):
        target = {'model_sha256':'weights'}
        req = inference.cache_request(target, [{'id':'1'}], {'seed':1,'k':8}, {'backend':'a100'}, 'probe', {'tokens':[1]})
        other = inference.cache_request(target, [{'id':'1'}], {'seed':2,'k':8}, {'backend':'a100'}, 'probe', {'tokens':[1]})
        self.assertNotEqual(p.digest(req), p.digest(other))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(inference.load_cache(root, req))
            p.write(root / 'result.json', {'cache_key':p.digest(req)})
            p.write(root / 'complete.json', {'request_sha256':p.digest(req), 'result_sha256':p.file_hash(root / 'result.json')})
            self.assertEqual(inference.load_cache(root,req)['cache_key'],p.digest(req))
            p.write(root / 'result.json', {'cache_key':'corrupt'})
            with self.assertRaises(ValueError):
                inference.load_cache(root,req)

    def test_costs_and_beta_fixed(self):
        result = p.costs(cfg())
        self.assertEqual(result['training_updates']['total'], 6300)
        self.assertEqual(result['training_completions'], 25200)
        self.assertEqual(result['distinct_main_model_states_estimate'], 61)
        self.assertEqual(result['probe_completions_before_cache'], 172800)
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg(); c['training']['beta'] = .04
            path = Path(tmp) / 'cfg.json'; p.write(path,c)
            with self.assertRaises(ValueError): p.config(path)
            c = cfg(); c['evaluation']['auxiliary_sets'] = [{'name':'main'}]
            p.write(path,c)
            with self.assertRaisesRegex(ValueError,'replace built-in'): p.config(path)

    def test_cleanup_refuses_incomplete_and_never_deletes_source_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c,_ = fixture_workspace(root)
            p.finalize(root,c['evaluation']['primary'],True)
            run = root / 'runs/S0_seed42_random'
            p.write(run / 'run.json', {'kind':'main','checkpoints':{'100':{'path':'not touched'}}})
            with self.assertRaises(FileNotFoundError):
                inference.cleanup_merged(root,run,100,True)
            p.write(run / 'run.json', {'kind':'pre_rl','checkpoints':{'100':{}}})
            with self.assertRaisesRegex(ValueError,'Only derived'):
                inference.cleanup_merged(root,run,100,True)

    def test_cleanup_only_deletes_completed_derived_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); c,pools = fixture_workspace(root)
            receipt = p.finalize(root,c['evaluation']['primary'],True)
            run = root / 'runs/S0_seed42_random'
            base, adapter = root / 'base', run / 'checkpoint-100'
            merged = root / 'merged' / f'qwen2-main-{run.name}-step100'
            identities = []
            for folder, payload in ((base,b'base'),(adapter,b'adapter'),(merged,b'merged')):
                folder.mkdir(parents=True)
                (folder / 'model.safetensors').write_bytes(payload)
                identities.append(p.model_identity(folder)['sha256'])
            base_id, adapter_id, merged_id = identities
            p.write(run / 'run.json', {'kind':'main','source':{'path':str(base),'model_sha256':base_id},
                                     'checkpoints':{'100':{'path':str(adapter),'sha256':adapter_id}}})
            p.write(merged / 'model_manifest.json', {'model_sha256':merged_id,'source':{
                'owner':'yj_research','run':str(run),'kind':'main','step':100,
                'base_sha256':base_id,'adapter_sha256':adapter_id}})
            protected = p.file_hash(root / 'data/main.jsonl')
            for kind, dataset in [('probe','main'),('eval_avg8','math500'),('eval_greedy','math500')]:
                req = inference.cache_request({'model_sha256':merged_id},pools[dataset],{}, {},kind,{})
                key = p.digest(req); cache = root / 'cache' / key
                p.write(cache / 'request.json',req)
                p.write(cache / 'result.json',{'cache_key':key})
                p.write(cache / 'complete.json',{'request_sha256':key,'result_sha256':p.file_hash(cache / 'result.json')})
                p.write(run / 'results' / f'{kind}_{dataset}_step100.json',{
                    'run':run.name,'step':100,'kind':kind,'dataset':dataset,'cache_key':key,
                    'model_sha256':merged_id,'protocol_sha256':p.digest(receipt)})
            self.assertFalse(inference.cleanup_merged(root,run,100)['delete_requested'])
            self.assertTrue(merged.exists())
            outside=root/'outside-merged'
            (root/'merged').rename(outside)
            (root/'merged').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Unsafe cleanup'):
                inference.cleanup_merged(root,run,100,True)
            self.assertTrue((outside/merged.name).exists())
            (root/'merged').unlink()
            outside.rename(root/'merged')
            inference.cleanup_merged(root,run,100,True)
            self.assertFalse(merged.exists())
            self.assertTrue(base.exists() and adapter.exists() and cache.exists())
            self.assertEqual(protected,p.file_hash(root / 'data/main.jsonl'))
            self.assertTrue((run / 'cleanup/step100.json').exists())

    def test_complete_statistics_pipeline_and_cache_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c,pools = fixture_workspace(root)
            receipt = p.finalize(root,c['evaluation']['primary'],True)
            for seed in c['seeds']:
                for state in ('S0','S1','S2'):
                    for mode in ('random','curriculum'):
                        correct = 2 + (1 if mode == 'curriculum' else 0)
                        items = [{'id':r['id'],'samples':[{'correct':j<correct,'truncated':False} for j in range(8)]}
                                 for r in pools['math500']]
                        identity = f'{state}/{mode}/{seed}'
                        req = inference.cache_request({'model_sha256':identity}, pools['math500'],
                              inference.inference_settings(c,'eval_avg8'),{'backend':'a100','versions':{}},'eval_avg8',{})
                        key = p.digest(req); cache = root / 'cache' / key
                        p.write(cache / 'request.json',req)
                        p.write(cache / 'result.json',{'cache_key':key,'items':items})
                        p.write(cache / 'complete.json',{'request_sha256':key,'result_sha256':p.file_hash(cache / 'result.json')})
                        p.write(root / 'runs' / f'{state}_seed{seed}_{mode}' / 'results/eval_avg8_math500_step300.json',{
                            'seed':seed,'state':state,'mode':mode,'step':300,'cache_key':key,
                            'model_sha256':identity,'protocol_sha256':p.digest(receipt)})
            with patch.object(metrics,'paired_bootstrap',wraps=lambda records,*args: metrics_bootstrap(records,repeats=10)):
                report = metrics.statistics_from_workspace(root)
            self.assertEqual(report['effects']['S0']['estimate_percentage_points'],12.5)
            self.assertEqual(report['effects']['S2-S0']['estimate_percentage_points'],0.)
            self.assertTrue((root / 'statistics/step300/seed_results.csv').exists())
            incomplete = p.read(cache / 'result.json')
            for item in incomplete['items']:
                item['samples'] = item['samples'][:7]
            p.write(cache / 'result.json',incomplete)
            p.write(cache / 'complete.json',{'request_sha256':key,'result_sha256':p.file_hash(cache / 'result.json')})
            with self.assertRaisesRegex(ValueError,'completion count'):
                metrics.statistics_from_workspace(root)
            p.write(cache / 'result.json',{'cache_key':key,'items':[]})
            with self.assertRaisesRegex(ValueError,'modified'):
                metrics.statistics_from_workspace(root)


@unittest.skipUnless(os.environ.get('YJ_RESEARCH_INTEGRATION') == '1', 'Set YJ_RESEARCH_INTEGRATION=1 in training environment; CPU only')
class ResearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
        set_seed(int(os.environ.get('YJ_TEST_MODEL_SEED','42')))
        tokens = ['<pad>','<eos>','<unk>','Question','Answer',':','0','1','2','3','4','5','yes','no']
        engine = Tokenizer(WordLevel({s:i for i,s in enumerate(tokens)},unk_token='<unk>'))
        engine.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>',padding_side='left')
        self.model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokens),hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                                                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
                                                eos_token_id=1,pad_token_id=0,tie_word_embeddings=True))
        self.c = cfg()
        self.c['training'].update(lora_r=2,lora_alpha=4,learning_rate=.05,max_prompt_length=64,max_completion_length=4)

    def test_hf_cache_identity_separates_cpu_and_cuda_precision(self):
        import torch
        with patch.object(torch.cuda,'is_available',return_value=False):
            cpu = inference.engine_identity('hf',None)
        with patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'get_device_name',return_value='test GPU'):
            gpu = inference.engine_identity('hf',None)
        self.assertNotEqual(cpu,gpu)

    def test_a100_adapter_preserves_pairing_with_external_indices(self):
        from transformers import AutoTokenizer
        from data_loader import load_data
        rows=[{'id':'a','idx':91,'problem':'2','solution':'\\boxed{2}'},
              {'id':'b','idx':12,'problem':'3','solution':'\\boxed{3}'}]
        settings=inference.inference_settings(self.c,'eval_avg8')
        texts=[runtime.prompt(r)[0] for r in rows]
        def fake_evaluator(command,**kwargs):
            root=Path(command[command.index('--data_dir')+1]).parent
            inputs=load_data('math','test',str(root/'data'))
            results=[{'idx':r['idx'],'question':runtime.imports()[0](r,'math'),
                      'code':['2']*8,'score':[True]*8,'pred':['2']*8,'finish_reason':['stop']*8}
                     for r in inputs]
            p.write_lines(root/'raw/math/test.jsonl',results)
        with tempfile.TemporaryDirectory() as tmp, patch.object(AutoTokenizer,'from_pretrained',return_value=self.tokenizer), \
             patch.object(runtime,'eos_ids',return_value=[151643,151645]), \
             patch.object(inference.subprocess,'check_output',return_value=json.dumps([self.tokenizer.encode(t) for t in texts])), \
             patch.object(inference.subprocess,'run',side_effect=fake_evaluator):
            items=inference.a100_inference({'path':'qwen2-tiny'},rows,settings,Path(tmp),'python',{})
            self.assertEqual([r['id'] for r in items],['a','b'])

    def test_exact_prompt_parity_whitespace_boolean_and_special_tokens(self):
        from types import SimpleNamespace
        from tokenizers.processors import TemplateProcessing
        parse_question, parse_ground_truth, construct_prompt = runtime.imports()
        for problem, answer in [('  2 + 2?\n ', '4'), (' Is it true? ', 'true'), ('한글 $x^2$\n\n test', 'yes')]:
            row = {'id':problem,'problem':problem,'solution':'\\boxed{'+answer+'}'}
            example = dict(row)
            example['question'] = parse_question(example,'math')
            _, example['gt_ans'] = parse_ground_truth(example,'math')
            expected = construct_prompt(example,'math',SimpleNamespace(prompt_type='cot',num_shots=0,adapt_few_shot=False))
            actual,_ = runtime.prompt(row)
            self.assertEqual(actual.encode(),expected.encode())
            self.assertEqual(runtime.encode_prompt(self.tokenizer,actual),self.tokenizer.encode(expected,add_special_tokens=True))
        self.assertEqual(runtime.stop_strings(),['</s>','<|im_end|>','<|endoftext|>','\n\nQuestion:'])
        self.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single='<eos> $A',special_tokens=[('<eos>',1)])
        with self.assertRaisesRegex(ValueError,'special tokens'):
            runtime.encode_prompt(self.tokenizer,'Question')

    def test_eos_stop_strings_and_length_have_distinct_masks(self):
        eos = runtime.completion_info(self.tokenizer,[8,1,0])
        length = runtime.completion_info(self.tokenizer,[8,9])
        stop = runtime.completion_info(self.tokenizer,[8,9,10],stops=['2 3'],eos=[1])
        self.assertFalse(eos['truncated']); self.assertEqual(eos['length'],2)
        self.assertTrue(length['truncated'])
        self.assertFalse(stop['truncated']); self.assertEqual(stop['finish_reason'],'stop_string')

    def test_loss_matches_installed_trl_and_zero_uniform_reward_gradients(self):
        import torch
        from collections import defaultdict
        from types import SimpleNamespace
        from trl import GRPOTrainer
        settings = self.c['training']
        settings['mask_truncated_completions'] = False
        logps = torch.tensor([[-.2,-.3],[-.5,-.6],[-.4,-.2],[-.7,-.8]],requires_grad=True)
        mask = torch.tensor([[1,0],[1,1],[1,1],[1,0]])
        adv = runtime.advantages([0,1,0,1])
        native,_ = runtime.grpo_loss(logps,logps.detach(),adv,mask,torch.zeros(4,dtype=torch.bool),settings)
        fake = SimpleNamespace(beta=0.,epsilon_low=.2,epsilon_high=.2,num_iterations=1,
                               args=SimpleNamespace(report_to=[]),
                               _get_per_token_logps=lambda *a: logps,
                               control=SimpleNamespace(should_evaluate=False),
                               _metrics={'train':defaultdict(list)},accelerator=SimpleNamespace(gather_for_metrics=lambda t:t))
        inputs = {'prompt_ids':torch.ones((4,1),dtype=torch.long),'completion_ids':torch.ones((4,2),dtype=torch.long),
                  'prompt_mask':torch.ones((4,1)), 'completion_mask':mask,'advantages':adv,'old_per_token_logps':None}
        reference = GRPOTrainer.compute_loss(fake,None,inputs)
        self.assertTrue(torch.allclose(native,reference))
        self.assertTrue(torch.allclose(torch.autograd.grad(native,logps,retain_graph=True)[0],torch.autograd.grad(reference,logps)[0]))
        for rewards in ([0]*4,[1]*4):
            lp = logps.detach().clone().requires_grad_(True)
            loss,_ = runtime.grpo_loss(lp,lp.detach(),runtime.advantages(rewards),mask,torch.zeros(4,dtype=torch.bool),settings)
            loss.backward()
            self.assertEqual(float(loss),0.)
            self.assertEqual(int(torch.count_nonzero(lp.grad)),0)
        settings['mask_truncated_completions'] = True
        lp = logps.detach().clone().requires_grad_(True)
        loss,actual = runtime.grpo_loss(lp,lp.detach(),adv,mask,torch.tensor([True,False,False,False]),settings)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(lp.grad[0])),0)
        self.assertEqual(int(actual[0].sum()),0)
        self.assertGreater(int(torch.count_nonzero(lp.grad[1:])),0)
        lp = logps.detach().clone().requires_grad_(True)
        loss,_ = runtime.grpo_loss(lp,lp.detach(),adv,mask,torch.ones(4,dtype=torch.bool),settings)
        loss.backward(); self.assertEqual(float(loss),0.); self.assertTrue(torch.isfinite(lp.grad).all())

    def test_no_update_generation_and_probe_cache(self):
        import torch
        before = inference.parameter_digest(self.model)
        s = inference.inference_settings(self.c,'probe')
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer allowed')):
            results = runtime.generate(self.model,self.tokenizer,['Question 2 Answer :'],s,8)
        self.assertEqual(len(results[0]),8)
        self.assertEqual(before,inference.parameter_digest(self.model))
        self.assertTrue(all(x.grad is None for x in self.model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'qwen2-tiny'; self.model.save_pretrained(base); self.tokenizer.save_pretrained(base)
            target = {'path':str(base),'model_sha256':p.model_identity(base)['sha256']}
            rows = [{'id':'q','problem':'2','solution':'\\boxed{2}'}]
            result = inference.sample(root,target,rows,self.c,'probe','hf')
            with patch.object(inference,'hf_inference',side_effect=AssertionError('Cache must avoid inference')):
                cached = inference.sample(root,target,rows,self.c,'probe','hf')
            self.assertEqual(result['cache_key'],cached['cache_key'])
            self.assertTrue(result['no_update'])
            self.assertEqual(len(result['items'][0]['samples']),8)
            audit = p.read(root / 'cache' / result['cache_key'] / 'no_update_audit.json')
            self.assertEqual(audit['parameter_sha256_before'],audit['parameter_sha256_after'])

    def test_eight_probe_callers_generate_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=root/'qwen2-tiny'; self.model.save_pretrained(base); self.tokenizer.save_pretrained(base)
            target={'path':str(base),'model_sha256':p.model_identity(base)['sha256']}
            rows=[{'id':'q','problem':'2','solution':'\\boxed{2}'}]
            actual=inference.hf_inference
            with patch.object(inference,'hf_inference',wraps=actual) as generator:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures=[pool.submit(inference.sample,root,target,rows,self.c,'probe','hf') for _ in range(8)]
                    results=[future.result(timeout=30) for future in futures]
                self.assertEqual(generator.call_count,1)
            self.assertEqual(len({r['cache_key'] for r in results}),1)
            self.assertTrue(all(len(r['items'][0]['samples'])==8 for r in results))

    def test_fresh_lora_optimizer_and_actual_group_updates(self):
        from train import adapter_digest
        import torch
        class AlternatingGrader:
            count = 0
            def score(self,text,answer):
                self.count += 1
                return float(self.count%2), text, None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base'; self.model.save_pretrained(base); self.tokenizer.save_pretrained(base)
            c = copy.deepcopy(self.c)
            c['training']['mask_truncated_completions'] = False
            model,tok,opt,sch,fresh = workflow.fresh_components(base,c,42,'cpu')
            start = adapter_digest(model)
            run = root / 'pre_rl/seed42'; run.mkdir(parents=True)
            checkpoints = {}
            for step in (1,2):
                with patch.object(opt,'step',wraps=opt.step) as update:
                    log = runtime.train_update(model,tok,{'id':str(step),'problem':'2','solution':'\\boxed{2}',
                                                         'difficulty':'easy','position':step-1},c['training'],opt,sch,AlternatingGrader())
                self.assertEqual(update.call_count,1)
                self.assertEqual(len(log['rewards']),4)
                checkpoint = run / f'checkpoint-{step}'
                workflow.save_checkpoint(checkpoint,model,tok,opt,sch,step,'shared-trajectory')
                checkpoints[str(step)] = {'path':str(checkpoint),'sha256':p.model_identity(checkpoint)['sha256']}
            self.assertNotEqual(start,adapter_digest(model))
            self.assertEqual(sch.last_epoch,2)
            p.write(run / 'run.json',{'kind':'pre_rl','status':'trained','seed':42,'source':{'path':str(base),
                                    'model_sha256':p.model_identity(base)['sha256']},'checkpoints':checkpoints})
            initial_hashes = [start]
            exported_hashes = []
            with patch.object(workflow,'locked',return_value=(c,{})):
                for step in (1,2):
                    exported = workflow.export_checkpoint(root,run,step)
                    exported_hashes.append(exported['model_sha256'])
                    new,t,optimizer,scheduler,init = workflow.fresh_components(exported['path'],c,42,'cpu')
                    self.assertEqual(len(optimizer.state),0)
                    self.assertEqual(scheduler.last_epoch,0)
                    self.assertTrue(init['new_lora'])
                    initial_hashes.append(init['initial_adapter_sha256'])
                    self.assertFalse((Path(exported['path'])/'adapter_config.json').exists())
            self.assertEqual(len(set(initial_hashes)),1)
            self.assertEqual(len(set(exported_hashes)),2)

    def test_pre_rl_to_new_main_run_orchestration_on_cpu(self):
        from contextlib import ExitStack
        import torch
        import train
        class SyntheticGrader:
            def __init__(self,*args): self.count=0
            def close(self): pass
            def score(self,text,answer):
                self.count+=1
                return float(self.count%2),text,None
        real_fresh=workflow.fresh_components
        def cpu_components(path,c,seed):
            result=real_fresh(path,c,seed,'cpu')
            self.assertEqual(next(result[0].parameters()).device.type,'cpu')
            return result
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp).resolve()
            _,pools=fixture_workspace(root)
            c=copy.deepcopy(self.c)
            c.update(pre_rl_checkpoints=[1,3],main_checkpoints=[0,1,2,3])
            c['training']['mask_truncated_completions']=False
            base=root/'qwen2-base'; self.model.save_pretrained(base); self.tokenizer.save_pretrained(base)
            source={'path':str(base),'model_sha256':p.model_identity(base)['sha256'],
                    'source':c['base_model'],'pre_rl_trajectory':None,'pre_rl_step':0,'state':'S0','seed':42}
            stack.enter_context(patch.object(workflow,'locked',return_value=(c,{})))
            stack.enter_context(patch.object(workflow,'ensure_base',return_value=source))
            stack.enter_context(patch.object(workflow,'fresh_components',side_effect=cpu_components))
            stack.enter_context(patch.object(train,'Grader',SyntheticGrader))
            # Only the full-run CUDA admission check is mocked. Every model and
            # optimizer operates on CPU; small checkpoint steps are test-only.
            stack.enter_context(patch.object(torch.cuda,'is_available',return_value=True))
            stack.enter_context(patch.object(torch.cuda,'device_count',return_value=1))
            stack.enter_context(patch.object(torch.cuda,'is_bf16_supported',return_value=True))
            stack.enter_context(patch.object(torch.cuda,'get_rng_state_all',return_value=[]))
            pre=workflow.train_run(root,42)
            self.assertEqual(pre['optimizer_steps'],3)
            s1=workflow.load_state(root,42,'S1'); s2=workflow.load_state(root,42,'S2')
            self.assertEqual(s1['pre_rl_trajectory'],s2['pre_rl_trajectory'])
            self.assertEqual((s1['pre_rl_step'],s2['pre_rl_step']),(1,3))
            for mode in ('random','curriculum'):
                result=workflow.train_run(root,42,'S1',mode)
                self.assertEqual(result['status'],'trained')
                self.assertEqual(result['optimizer_steps'],3)
                self.assertEqual(set(result['checkpoints']),{'1','2','3'})
                self.assertEqual(result['source']['pre_rl_trajectory'],pre['trajectory_id'])
                self.assertEqual(result['fresh_initialization']['optimizer_state_entries_at_start'],0)
                self.assertTrue(result['adapter_changed'])
                with self.assertRaises(FileExistsError): workflow.train_run(root,42,'S1',mode)

    def test_actual_update_preserves_truncated_correct_reward(self):
        import torch
        from train import adapter_digest
        class Grader:
            values=iter([0.,1.,1.,0.])
            def score(self,text,answer): return next(self.values),text,None
        group=[runtime.completion_info(self.tokenizer,ids) for ids in ([8,1],[9,1],[8,9],[10,1])]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); self.model.save_pretrained(base); self.tokenizer.save_pretrained(base)
            model,tok,opt,sch,_=workflow.fresh_components(base,self.c,42,'cpu')
            before=adapter_digest(model)
            with patch.object(runtime,'generate',return_value=[group]):
                log=runtime.train_update(model,tok,{'id':'p','problem':'2','solution':'\\boxed{2}',
                                        'difficulty':'easy','position':0},self.c['training'],opt,sch,Grader())
            self.assertEqual(log['valid_loss_tokens'],6)
            self.assertEqual(log['truncated_count'],1)
            self.assertEqual(log['rewards'][2],1.)
            self.assertEqual(log['optimizer_steps'],1)
            self.assertEqual(sch.last_epoch,1)
            self.assertNotEqual(before,adapter_digest(model))


if __name__ == '__main__':
    unittest.main()
