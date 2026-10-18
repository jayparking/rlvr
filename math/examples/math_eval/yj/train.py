"""Ordered GRPO training and staged evaluation using the unchanged A100 evaluator.

Heavy dependencies are imported only for training/export. Evaluation orchestration,
summaries, dry runs and unit tests use the Python standard library.
"""
import argparse
from collections import Counter
import csv
import gc
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from prepare_data import ROOT, build_schedule, digest_file, fingerprint, load_config

EVAL_ROOT = ROOT.parent
EVALUATOR = EVAL_ROOT / "math_eval_a100.py"
RUNS = tuple(f"{v}_{m}" for v in ("v1", "v2") for m in ("random", "curriculum"))
PINNED = {"trl": "0.16.1", "transformers": "4.51.3", "peft": "0.15.2",
          "accelerate": "1.6.0", "datasets": "3.5.0"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def check_versions():
    for package, expected in PINNED.items():
        actual = version(package)
        if actual != expected:
            raise ValueError(f"{package}=={expected} required; found {actual}. Use requirements-train.txt.")


def load_experiment(artifacts, run):
    """Check provenance, paired schedules, group counts and held-out separation."""
    artifacts = Path(artifacts).resolve()
    cfg = load_config(artifacts / "config.json")
    t = cfg.get("training")
    if not t:
        raise ValueError("Prepared config has no training settings. Regenerate artifacts with current experiment.json.")
    for key in ("num_generations", "max_prompt_length", "max_completion_length", "lora_r", "lora_alpha"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"Invalid training.{key}")
    if t["num_generations"] < 2 or t["learning_rate"] <= 0 or t["beta"] < 0:
        raise ValueError("Invalid GRPO generations/learning_rate/beta")
    if t["temperature"] <= 0 or not 0 < t["top_p"] <= 1 or t["grading_timeout_seconds"] <= 0:
        raise ValueError("Invalid sampling or grading settings")
    if not t["lora_target_modules"]:
        raise ValueError("Specify LoRA target modules")
    manifest = read_json(artifacts / "manifest.json")
    names = ["train_selected.jsonl", "test_selected.jsonl", "train_random.jsonl",
             "train_curriculum.jsonl", "eval_data/math/test.jsonl"]
    for name in names:
        if digest_file(artifacts / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Artifact changed: {name}")
    train = read_jsonl(artifacts / "train_selected.jsonl")
    test = read_jsonl(artifacts / "test_selected.jsonl")
    for split, rows in (("train", train), ("test", test)):
        expected = {g: cfg[f"{split}_per_group"] for g in cfg["curriculum_order"]}
        if Counter(r["difficulty"] for r in rows) != expected:
            raise ValueError(f"Unexpected {split} group counts")
        if len({r["id"] for r in rows}) != len(rows) or len({r["fingerprint"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {split} problems")
        for row in rows:
            if row["split"] != split or row["fingerprint"] != fingerprint(row["problem"]):
                raise ValueError(f"Invalid {split} provenance: {row['id']}")
            if row["level"] not in cfg["groups"][row["difficulty"]]:
                raise ValueError(f"Invalid difficulty: {row['id']}")
    if {r["fingerprint"] for r in train} & {r["fingerprint"] for r in test}:
        raise ValueError("Train/test overlap")
    if read_jsonl(artifacts / "eval_data/math/test.jsonl") != test:
        raise ValueError("Evaluator data differs from the common held-out set")
    schedules = {}
    for mode in ("random", "curriculum"):
        schedules[mode] = read_jsonl(artifacts / f"train_{mode}.jsonl")
        if schedules[mode] != build_schedule(train, cfg, mode):
            raise ValueError(f"Unexpected {mode} schedule")
    model_version, mode = run.split("_", 1)
    return cfg, schedules[mode], test, cfg["models"][model_version]


class OrderedGroupSampler:
    """One prompt repeated G times per batch; never reshuffle prepared rows."""
    def __init__(self, size, generations):
        self.size, self.generations = size, generations

    def __iter__(self):
        for index in range(self.size):
            yield from [index] * self.generations

    def __len__(self):
        return self.size * self.generations


def make_trainer_class(base):
    class OrderedGRPOTrainer(base):
        def compute_loss(self, model, inputs, **kwargs):
            loss = super().compute_loss(model, inputs, **kwargs)
            if not loss.isfinite().all().item():
                raise FloatingPointError("Non-finite GRPO loss; training stopped")
            return loss

        def _get_train_sampler(self):
            if (self.accelerator.num_processes != 1 or self.args.n_gpu > 1
                    or self.args.gradient_accumulation_steps != 1
                    or self.args.per_device_train_batch_size != self.num_generations
                    or self.num_iterations != 1):
                raise ValueError("Ordered training requires one GPU, batch=G, accumulation=1, iterations=1")
            return OrderedGroupSampler(len(self.train_dataset), self.num_generations)
    return OrderedGRPOTrainer


def evaluator_imports():
    # Reuse existing code without editing it. The second path also resolves
    # grader.py's top-level latex2sympy2 import without an editable install.
    for directory in (EVAL_ROOT, EVAL_ROOT / "latex2sympy"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from parser import extract_answer, parse_ground_truth
    from grader import math_equal
    from utils import construct_prompt
    return extract_answer, parse_ground_truth, math_equal, construct_prompt


def grade_worker_init():
    global _extract, _equal
    _extract, _, _equal, _ = evaluator_imports()


def grade_worker(completion, answer):
    # Same stop-text cleanup used by the existing evaluator's CoT path.
    for stop in ("</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:"):
        completion = completion.split(stop)[0]
    prediction = _extract(completion.strip(), "math")
    return float(bool(prediction) and _equal(prediction, answer)), prediction


class Grader:
    """A killable CPU worker: symbolic grading can swallow signal exceptions."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.pool = None

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def score(self, completion, answer):
        if self.pool is None:
            self.pool = multiprocessing.get_context("spawn").Pool(1, initializer=grade_worker_init)
            # Allow imports to finish before starting the per-answer timeout.
            self.pool.apply_async(str, ("ready",)).get(timeout=120)
        try:
            value, prediction = self.pool.apply_async(grade_worker, (completion, answer)).get(self.timeout)
            return value, prediction, None
        except multiprocessing.TimeoutError:
            self.close()
            return 0.0, "", "grading_timeout"


def make_reward(rows, generations, grader, log_path):
    cursor = 0

    def correctness_reward(completions, answer, problem_id, position, difficulty, **kwargs):
        nonlocal cursor
        if cursor >= len(rows):
            raise ValueError("More training batches than planned")
        row = rows[cursor]
        columns = (completions, answer, problem_id, position, difficulty)
        if any(len(col) != generations for col in columns):
            raise ValueError("Expected exactly G completions for one problem")
        if (any(uid != row["id"] for uid in problem_id)
                or any(pos != cursor for pos in position)
                or any(g != row["difficulty"] for g in difficulty)):
            raise ValueError("Trainer changed the planned problem order/group")
        scores = [grader.score(text, target) for text, target in zip(completions, answer)]
        rewards = [s[0] for s in scores]
        append_jsonl(log_path, {
            "step": cursor + 1, "problem_id": row["id"], "position": cursor,
            "difficulty": row["difficulty"], "completions": completions,
            "answer": answer[0], "predictions": [s[1] for s in scores],
            "rewards": rewards, "grading_errors": [s[2] for s in scores],
            "zero_reward_variance": len(set(rewards)) == 1,
        })
        cursor += 1
        return rewards

    return correctness_reward


def training_kwargs(t, total, stage_size, seed, output, cpu=False):
    return dict(
        output_dir=str(output), seed=seed, data_seed=seed,
        per_device_train_batch_size=t["num_generations"], gradient_accumulation_steps=1,
        num_generations=t["num_generations"], num_iterations=1,
        num_train_epochs=1, max_steps=total, learning_rate=t["learning_rate"],
        lr_scheduler_type="constant", warmup_steps=0, beta=t["beta"],
        temperature=t["temperature"], top_p=t["top_p"], top_k=None,
        max_prompt_length=t["max_prompt_length"], max_completion_length=t["max_completion_length"],
        use_vllm=False, bf16=not cpu, fp16=False, use_cpu=cpu,
        gradient_checkpointing=not cpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=0,
        save_strategy="steps", save_steps=stage_size, save_total_limit=None,
        save_only_model=False, eval_strategy="no", logging_steps=1,
        logging_nan_inf_filter=False, report_to="none", optim="adamw_torch",
    )


def prepare_examples(rows, tokenizer, max_prompt_length):
    _, parse_ground_truth, _, construct_prompt = evaluator_imports()
    prompt_args = SimpleNamespace(prompt_type="cot", num_shots=0, adapt_few_shot=False)
    examples = []
    for row in rows:
        _, answer = parse_ground_truth({"solution": row["solution"]}, "math")
        if not answer:
            raise ValueError(f"Missing reference answer: {row['id']}")
        prompt = construct_prompt({"question": row["problem"], "gt_ans": answer}, "math", prompt_args)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_length:
            raise ValueError(f"Prompt would be truncated: {row['id']}; increase max_prompt_length for ALL runs")
        examples.append({"prompt": prompt, "answer": answer, "problem_id": row["id"],
                         "position": row["position"], "difficulty": row["difficulty"]})
    return examples


def adapter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_training(args):
    cfg, rows, test, source = load_experiment(args.artifacts, args.run)
    stage_size = 1 if args.smoke else cfg["train_per_group"]
    if args.smoke:
        rows = rows[:3]
    stages = [0, stage_size, stage_size * 2, stage_size * 3]
    output = (args.output or ROOT / "runs" / (args.run + ("_smoke" if args.smoke else ""))).resolve()
    plan = {"run": args.run, "purpose": "smoke" if args.smoke else "experiment",
            "initial_model": source, "seed": cfg["seed"], "training": cfg["training"],
            "artifacts": str(args.artifacts.resolve()), "output_dir": str(output),
            "stages": stages, "train_problems": len(rows), "eval_problems": len(test),
            "evaluator_sha256": digest_file(EVALUATOR)}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Choose a NEW --output directory: {output}")
    check_versions()
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or int(os.environ.get("WORLD_SIZE", "1")) != 1):
        raise ValueError("Use one CUDA GPU: CUDA_VISIBLE_DEVICES=0 python yj/train.py train ...")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This training configuration requires BF16 support")
    set_seed(cfg["seed"])
    base_path = snapshot_download(repo_id=source["repo_id"], revision=source["revision"],
                                  cache_dir=str(ROOT / ".cache" / "huggingface"))
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    t = cfg["training"]
    examples = prepare_examples(rows, tokenizer, t["max_prompt_length"])
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    output.mkdir(parents=True, exist_ok=False)
    # Copy the common exam so every stage uses exactly the same immutable input.
    eval_file = output / "eval_data" / "math" / "test.jsonl"
    eval_file.parent.mkdir(parents=True)
    eval_file.write_bytes((args.artifacts / "eval_data/math/test.jsonl").read_bytes())
    plan.update(status="training", base_model_path=base_path,
                eval_sha256=digest_file(eval_file), config=cfg,
                versions={p: version(p) for p in (*PINNED, "torch")})
    write_json(output / "run.json", plan)
    grader = Grader(t["grading_timeout_seconds"])
    trainer = None
    try:
        trainer = make_trainer_class(GRPOTrainer)(
            model=model, processing_class=tokenizer,
            args=GRPOConfig(**training_kwargs(t, len(rows), stage_size, cfg["seed"], output)),
            train_dataset=Dataset.from_list(examples),
            reward_funcs=make_reward(rows, t["num_generations"], grader, output / "rewards.jsonl"),
            peft_config=LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"],
                                   target_modules=t["lora_target_modules"], lora_dropout=0,
                                   bias="none", task_type="CAUSAL_LM"),
        )
        before = adapter_digest(trainer.model)
        trainer.train()
        trainer.save_state()
        after = adapter_digest(trainer.model)
        if trainer.state.global_step != len(rows) or len(read_jsonl(output / "rewards.jsonl")) != len(rows):
            raise ValueError("Training did not consume exactly one update per problem")
        for stage in stages[1:]:
            state = read_json(output / f"checkpoint-{stage}" / "trainer_state.json")
            if state["global_step"] != stage:
                raise ValueError(f"Incorrect checkpoint step: {stage}")
        plan.update(status="trained", adapter_sha256_before=before, adapter_sha256_after=after,
                    adapter_changed=before != after, completed_steps=trainer.state.global_step)
        write_json(output / "run.json", plan)
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", plan)
        raise
    finally:
        grader.close()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    if not args.no_export:
        export_stages(output)
    print(f"Training finished: {output}. GPU evaluation is a separate command.")


def export_stages(output):
    output = Path(output).resolve()
    plan = read_json(output / "run.json")
    if plan["status"] not in ("trained", "exported"):
        raise ValueError("Training must finish before stage export")
    check_versions()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["base_model_path"])
    for stage in plan["stages"][1:]:
        destination = output / "models" / f"step-{stage:03d}"
        marker = destination / "stage.json"
        checkpoint = output / f"checkpoint-{stage}"
        identity = {"step": stage, "adapter_sha256": digest_file(checkpoint / "adapter_model.safetensors"),
                    "initial_model": plan["initial_model"]}
        if marker.exists():
            if read_json(marker) != identity:
                raise ValueError(f"Export source changed: {destination}")
            continue
        # Re-exporting an interrupted export is safe; a stage is ready only with its marker.
        destination.mkdir(parents=True, exist_ok=True)
        base = AutoModelForCausalLM.from_pretrained(plan["base_model_path"], torch_dtype=torch.bfloat16)
        adapted = PeftModel.from_pretrained(base, checkpoint)
        merged = adapted.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True
        merged.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        write_json(marker, identity)
        del base, adapted, merged
        gc.collect()
    plan["status"] = "exported"
    write_json(output / "run.json", plan)


def evaluation_command(output, stage, python):
    output = Path(output).resolve()
    plan = read_json(output / "run.json")
    model = Path(plan["base_model_path"]) if stage == 0 else output / "models" / f"step-{stage:03d}"
    return [str(python), str(EVALUATOR), "--model_name_or_path", str(model),
            "--data_names", "math", "--data_dir", str(output / "eval_data"),
            "--split", "test", "--num_test_sample", "-1", "--prompt_type", "cot",
            "--seed", str(plan["seed"]), "--temperature", "0", "--n_sampling", "1",
            "--max_tokens_per_call", "2048", "--use_vllm", "--save_outputs",
            "--output_dir", str(output / "eval" / f"step-{stage:03d}")]


def summarize_stage(output, stage):
    output = Path(output)
    plan = read_json(output / "run.json")
    exam = read_jsonl(output / "eval_data/math/test.jsonl")
    path = output / "eval" / f"step-{stage:03d}" / "math"
    predictions = list(path.glob("*.jsonl"))
    if len(predictions) != 1:
        raise ValueError(f"Expected exactly one prediction file in {path}")
    samples = read_jsonl(predictions[0])
    if len(samples) != len(exam) or {r["idx"] for r in samples} != set(range(len(exam))):
        raise ValueError(f"Incomplete/duplicate evaluation at step {stage}")
    counts, correct = Counter(), Counter()
    for sample in samples:
        reference = exam[sample["idx"]]
        if (fingerprint(sample["question"]) != reference["fingerprint"]
                or sample.get("level") != reference["level"]):
            raise ValueError("Evaluation problem/level differs from common exam")
        scores = sample.get("score", [])
        if len(scores) != 1 or type(scores[0]) is not bool:
            raise ValueError("Expected one Boolean score per problem")
        for group in ("overall", reference["difficulty"]):
            counts[group] += 1
            correct[group] += int(scores[0])
    return {"run": plan["run"], "purpose": plan["purpose"], "step": stage,
            "metrics": {g: {"correct": correct[g], "total": counts[g],
                            "accuracy": round(100 * correct[g] / counts[g], 4)}
                        for g in ("overall", "easy", "medium", "hard")}}


def summarize(output):
    output = Path(output).resolve()
    plan = read_json(output / "run.json")
    rows = [summarize_stage(output, stage) for stage in plan["stages"]]
    write_json(output / "stage_metrics.json", rows)
    with (output / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "purpose", "step", "group", "correct", "total", "accuracy_percent"])
        for row in rows:
            for group, metric in row["metrics"].items():
                writer.writerow([row["run"], row["purpose"], row["step"], group,
                                 metric["correct"], metric["total"], metric["accuracy"]])
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def evaluate_stages(args):
    output = args.run_dir.resolve()
    plan = read_json(output / "run.json")
    if digest_file(EVALUATOR) != plan["evaluator_sha256"]:
        raise ValueError("math_eval_a100.py changed since training was configured")
    if digest_file(output / "eval_data/math/test.jsonl") != plan["eval_sha256"]:
        raise ValueError("Common evaluation data changed")
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not env["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to an available GPU")
    if not args.dry_run and plan["status"] != "exported":
        raise ValueError("Run the export command before evaluation")
    for stage in plan["stages"]:
        command = evaluation_command(output, stage, args.python)
        print(f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        model = Path(command[command.index("--model_name_or_path") + 1])
        if not (model / "config.json").is_file() or (stage and not (model / "stage.json").is_file()):
            raise ValueError(f"Missing exported model: {model}")
        eval_dir = output / "eval" / f"step-{stage:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        request = {"command": command, "eval_sha256": plan["eval_sha256"],
                   "evaluator_sha256": plan["evaluator_sha256"],
                   "model": plan["initial_model"] if stage == 0 else read_json(model / "stage.json")}
        receipt = eval_dir / "request.json"
        if receipt.exists() and read_json(receipt) != request:
            raise ValueError(f"Evaluation request changed: {eval_dir}")
        write_json(receipt, request)
        # A failed/partial evaluation must be rerun in full, not silently skipped by metrics files.
        if not (eval_dir / "complete.json").exists():
            subprocess.run(command + ["--overwrite"], cwd=EVAL_ROOT, env=env, check=True)
            result = summarize_stage(output, stage)
            write_json(eval_dir / "complete.json", result)
        else:
            summarize_stage(output, stage)
    if not args.dry_run:
        summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train", help="GRPO training, checkpointing and model export")
    train.add_argument("--run", choices=RUNS, required=True)
    train.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/seed42")
    train.add_argument("--output", type=Path)
    train.add_argument("--smoke", action="store_true", help="Only 3 problems, stages 0/1/2/3, separate output")
    train.add_argument("--dry-run", action="store_true", help="Validate data and print plan without loading models")
    train.add_argument("--no-export", action="store_true", help="Export later using the export command")
    export = sub.add_parser("export", help="Merge stage adapters into models for the unchanged evaluator")
    export.add_argument("--run-dir", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate all saved stages with math_eval_a100.py")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--python", default=sys.executable, help="Python executable in the existing A100 eval environment")
    evaluate.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize", help="Aggregate overall/easy/medium/hard accuracy for all stages")
    summary.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "train":
        run_training(args)
    elif args.action == "export":
        export_stages(args.run_dir)
    elif args.action == "evaluate":
        evaluate_stages(args)
    else:
        summarize(args.run_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_research-vllm-worker":
        from research.inference import vllm_worker
        vllm_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "research":
        from research.cli import main as research_main
        research_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        raise SystemExit("Legacy V1/V2 training is archived. Review: python yj/train.py research plan; then use research train. Existing artifacts are preserved.")
    else:
        main()
