"""Reuse the evaluator's prompt, extraction and mathematical equivalence logic."""
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

from data_pipeline import EVAL_ROOT

sys.path.insert(0, str(EVAL_ROOT))
from parser import extract_answer, parse_ground_truth
from grader import math_equal
from utils import construct_prompt


def make_example(row, prompt_cfg):
    _, answer = parse_ground_truth({"solution": row["solution"]}, "math")
    if not answer:
        raise ValueError(f"Could not parse reference answer: {row['id']}")
    args = SimpleNamespace(prompt_type=prompt_cfg["type"], num_shots=0, adapt_few_shot=False)
    prompt = construct_prompt({"question": row["question"], "gt_ans": answer}, "math", args)
    return dict(row, prompt=prompt, answer=answer)


def _timeout(signum, frame):
    raise TimeoutError("Answer grading exceeded time limit")


def score(completion, answer, timeout_seconds=3):
    # Bound the entire parser + symbolic checker on the Linux main thread.
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        prediction = extract_answer(completion, "math")
        return float(bool(prediction) and math_equal(prediction, answer)), prediction, None
    except TimeoutError:
        return 0.0, "", "grading_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def make_reward(log_path):
    def correctness_reward(completions, answer, id, difficulty, position, **kwargs):
        rewards = []
        with Path(log_path).open("a", encoding="utf-8") as handle:
            for completion, target, uid, group, pos in zip(completions, answer, id, difficulty, position):
                if isinstance(completion, list):
                    completion = completion[-1]["content"]
                value, prediction, error = score(completion, target)
                rewards.append(value)
                handle.write(json.dumps({"position": pos, "id": uid, "difficulty": group,
                    "reward": value, "prediction": prediction, "answer": target,
                    "grading_error": error}, ensure_ascii=False) + "\n")
        return rewards
    return correctness_reward
