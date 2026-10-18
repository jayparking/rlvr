"""Shared evaluator contract, no-gradient generation, and explicit audited GRPO loss."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from .protocol import A100, EVAL_ROOT, append, digest, file_hash

IMPLEMENTATION = 'yj_grpo_v1_bnpo'


def imports():
    import sys
    for p in (EVAL_ROOT, EVAL_ROOT / 'latex2sympy'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from parser import parse_question, parse_ground_truth
    from utils import construct_prompt
    return parse_question, parse_ground_truth, construct_prompt


def reference_nodes():
    """Read the protected evaluator's actual prompt-building statements, not a copy."""
    tree = ast.parse(A100.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'example')
    assignments = [n for n in loop.body if isinstance(n, ast.Assign)]
    selected = []
    for n in assignments:
        target = ast.unparse(n.targets[0])
        if target in ("example['question']", '(gt_cot, gt_ans)', "example['gt_ans']", 'full_prompt'):
            selected.append(copy.deepcopy(n))
    if len(selected) != 4:
        raise ValueError('Evaluator prompt construction changed; audit required')
    return selected


def prompt(row):
    parse_question, parse_ground_truth, construct_prompt = imports()
    example = dict(row)
    # Auxiliary datasets can supply answer-only records while using the same math template.
    if not example.get('solution'):
        example['solution'] = '\\boxed{' + str(example['answer']) + '}'
    scope = {'example': example, 'data_name': 'math', 'parse_question': parse_question,
             'parse_ground_truth': parse_ground_truth, 'construct_prompt': construct_prompt,
             'args': SimpleNamespace(prompt_type='cot', num_shots=0, adapt_few_shot=False)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=reference_nodes(), type_ignores=[])), str(A100), 'exec'), scope)
    return scope['full_prompt'], scope['gt_ans']


def stop_strings():
    main = next(n for n in ast.parse(A100.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'stop_words' for t in n.targets))
    nodes = copy.deepcopy(main.body[index:index+2])
    scope = {'args': SimpleNamespace(prompt_type='cot')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(A100), 'exec'), scope)
    stops = scope['stop_words']
    if stops != ['</s>', '<|im_end|>', '<|endoftext|>', '\n\nQuestion:']:
        raise ValueError('Unexpected evaluator CoT stop strings')
    return stops


def eos_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    vocab = tokenizer.get_vocab()
    ids.update(vocab[s] for s in stop_strings()[:3] if s in vocab)
    ids.discard(None)
    if not ids:
        raise ValueError('Tokenizer needs an EOS token')
    return sorted(ids)


def encode_prompt(tokenizer, text):
    raw = tokenizer.encode(text, add_special_tokens=False)
    # vLLM 0.6.3's text path defaults to add_special_tokens=True. For Qwen these
    # are identical; fail explicitly for a tokenizer that would add BOS/EOS.
    if raw != tokenizer.encode(text, add_special_tokens=True):
        raise ValueError('Tokenizer adds special tokens: incompatible with unchanged A100 text path')
    return raw


def prompt_audit(tokenizer, rows):
    rows_out = []
    for row in rows:
        text, _ = prompt(row)
        ids = encode_prompt(tokenizer, text)
        rows_out.append({'id': row['id'], 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'token_ids_sha256': digest(ids), 'tokens': len(ids)})
    return {'template_source': str(A100), 'a100_sha256': file_hash(A100), 'zero_shot': True,
            'apply_chat_template': False, 'add_special_tokens': False,
            'default_encoding_identical': True, 'stop_strings': stop_strings(),
            'eos_ids': eos_ids(tokenizer), 'padding_side': 'left', 'rows': rows_out}


def completion_info(tokenizer, token_ids, stops=None, eos=None):
    stops = stop_strings() if stops is None else stops
    eos = set(eos_ids(tokenizer) if eos is None else eos)
    token_ids = list(map(int, token_ids))
    terminal = None
    # Prefix decoding also handles a stop string split across multiple tokens.
    for end in range(1, len(token_ids)+1):
        text = tokenizer.decode(token_ids[:end], skip_special_tokens=False)
        if token_ids[end-1] in eos:
            terminal = (end, 'eos')
            break
        if any(s in text for s in stops):
            terminal = (end, 'stop_string')
            break
    length, reason = terminal if terminal else (len(token_ids), 'length')
    visible = tokenizer.decode(token_ids[:length], skip_special_tokens=True)
    for stop in stops:
        visible = visible.split(stop)[0]
    return {'text': visible.strip(), 'token_ids': token_ids[:length], 'length': length,
            'finish_reason': reason, 'truncated': terminal is None}


def generate(model, tokenizer, texts, settings, k):
    """Only generation. No optimizer is accepted/created; no parameter update is possible."""
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    outputs = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for text in texts:
                ids = encode_prompt(tokenizer, text)
                if len(ids) > settings.get('max_prompt_length', 8192):
                    raise ValueError('Input would be truncated; change the preregistered limit')
                prompt_length = len(ids)
                stop = stop_strings() if settings.get('use_math_stops', True) else []
                eos = eos_ids(tokenizer)

                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        values = []
                        for seq in input_ids[:, prompt_length:]:
                            text = tokenizer.decode(seq.tolist(), skip_special_tokens=False)
                            values.append(any(s in text for s in stop))
                        return torch.tensor(values, dtype=torch.bool, device=input_ids.device)

                inputs = torch.tensor([ids]*k, device=next(model.parameters()).device)
                sampled = settings['temperature'] > 0
                kwargs = dict(max_new_tokens=settings['max_completion_length'], do_sample=sampled,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                              repetition_penalty=1.0, num_beams=1, use_cache=True,
                              top_k=0)
                if sampled:
                    kwargs.update(temperature=settings['temperature'], top_p=settings['top_p'], top_k=settings['top_k'])
                sequences = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           generation_config=GenerationConfig(**kwargs),
                                           stopping_criteria=StoppingCriteriaList([Stop()]))
                outputs.append([completion_info(tokenizer, seq[prompt_length:].tolist(), stop, eos) for seq in sequences])
    finally:
        model.train(was_training)
    return outputs


def advantages(rewards, scale=True):
    import torch
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    centered = rewards - rewards.mean()
    return centered / (rewards.std(unbiased=True) + 1e-4) if scale else centered


def grpo_loss(logps, old_logps, adv, token_mask, truncated, settings):
    """TRL 0.16.1 actual token-mean objective + explicit whole-completion truncation mask.

    A_i=(r_i-mean(r))/(sample_std(r)+1e-4)
    L=-sum(m_it * min(exp(lp-old)*A_i, clip(exp(lp-old),1-epsL,1+epsH)*A_i))/sum(m)
    beta=0. For num_iterations=1 old=detach(lp). All-G rewards determine A,
    including truncated samples; only their direct token loss is masked out.
    """
    import torch
    if settings['beta'] != 0 or settings['num_iterations'] != 1 or settings['loss_type'] != 'bnpo':
        raise ValueError('Unsupported loss protocol')
    mask = token_mask.to(logps.dtype)
    if settings['mask_truncated_completions']:
        mask = mask * (~truncated.bool()).to(mask.dtype).unsqueeze(1)
    ratio = (logps - old_logps).exp()
    clipped = ratio.clamp(1-settings['epsilon_low'], 1+settings['epsilon_high'])
    per_token = -torch.minimum(ratio*adv[:, None], clipped*adv[:, None])
    # If the whole batch is truncated: differentiable zero, still one optimizer step.
    return (per_token * mask).sum() / mask.sum().clamp_min(1), mask


def loss_audit():
    from importlib.metadata import version
    from trl import GRPOTrainer
    source = inspect.getsource(GRPOTrainer.compute_loss)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if version('trl') != '0.16.1' or source_hash != '3131ded0d77083282e4eb18f1ad19778d8cadf658e9666c32ed8ee2aedebb549':
        raise ValueError('TRL reference implementation changed; audit required')
    return {'implementation': IMPLEMENTATION, 'reference_trl_version': version('trl'),
            'reference_compute_loss_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'reference_reduction': 'sum_token_loss / sum_valid_tokens',
            'custom_compute_loss_sha256': hashlib.sha256(inspect.getsource(grpo_loss).encode()).hexdigest(),
            'std_correction': 1, 'advantage_epsilon': 1e-4,
            'clipping_inactive_at_ratio_1_for_single_iteration': True,
            'truncation': 'whole completion masked; raw reward preserved; all G rewards enter normalization'}


def train_update(model, tokenizer, row, settings, optimizer, scheduler, grader):
    import torch
    text, answer = prompt(row)
    group = generate(model, tokenizer, [text], settings, settings['num_generations'])[0]
    scores = [grader.score(x['text'], answer) for x in group]
    rewards = [x[0] for x in scores]
    prefix = encode_prompt(tokenizer, text)
    longest = max(len(x['token_ids']) for x in group)
    if longest == 0:
        raise ValueError('Generation produced no tokens')
    data, masks = [], []
    for item in group:
        n = len(item['token_ids'])
        data.append(prefix + item['token_ids'] + [tokenizer.pad_token_id]*(longest-n))
        masks.append([1]*n + [0]*(longest-n))
    device = next(model.parameters()).device
    ids = torch.tensor(data, device=device)
    mask = torch.tensor(masks, device=device)
    attention = torch.cat([torch.ones((len(group), len(prefix)), device=device, dtype=mask.dtype), mask], dim=1)
    # Training mode activates gradient checkpointing. Dropout is disabled at
    # initialization so generation and policy likelihoods use the same policy.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, len(prefix)-1:-1, :]
    logits = logits.float() / settings['temperature']
    logps = logits.log_softmax(-1).gather(-1, ids[:, len(prefix):, None]).squeeze(-1)
    adv = advantages(rewards, settings['scale_rewards']).to(device)
    truncated = torch.tensor([x['truncated'] for x in group], device=device)
    loss, actual_mask = grpo_loss(logps, logps.detach(), adv, mask, truncated, settings)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {'problem_id': row['id'], 'difficulty': row['difficulty'], 'position': row['position'],
            'rewards': rewards, 'predictions': [s[1] for s in scores], 'grading_errors': [s[2] for s in scores],
            'answer': answer, 'completions': group, 'advantages': adv.detach().cpu().tolist(),
            'loss': float(loss.detach()), 'grad_norm': float(grad_norm),
            'valid_loss_tokens': int(actual_mask.sum()), 'truncated_count': int(truncated.sum()),
            'all_wrong': all(r == 0 for r in rewards), 'all_correct': all(r == 1 for r in rewards),
            'zero_reward_variance': len(set(rewards)) == 1, 'optimizer_steps': 1}
