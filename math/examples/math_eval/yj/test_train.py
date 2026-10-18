import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train
from prepare_data import fingerprint, load_config, save_artifacts


class FakeGrader:
    """Synthetic mixed rewards exercise the optimizer, not model math ability."""
    def __init__(self):
        self.calls = 0

    def score(self, completion, answer):
        self.calls += 1
        return float(self.calls % 2), completion, None


def exam_rows():
    return [{"id": f"test/{i}", "problem": f"exam {i}", "question": f"exam {i}",
             "fingerprint": fingerprint(f"exam {i}"), "level": level, "difficulty": group}
            for i, (group, level) in enumerate((("easy", 1), ("medium", 3), ("hard", 5)))]


def evaluation_fixture(output):
    exam = output / "eval_data/math/test.jsonl"
    exam.parent.mkdir(parents=True)
    exam.write_text("".join(json.dumps(r) + "\n" for r in exam_rows()))
    base = output / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    plan = {"run": "v1_curriculum", "purpose": "experiment", "seed": 42,
            "status": "exported", "stages": [0, 100, 200, 300], "base_model_path": str(base),
            "evaluator_sha256": train.digest_file(train.EVALUATOR),
            "eval_sha256": train.digest_file(exam), "initial_model": {"repo_id": "test", "revision": "a" * 40}}
    train.write_json(output / "run.json", plan)
    for stage in plan["stages"][1:]:
        folder = output / "models" / f"step-{stage:03d}"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
        train.write_json(folder / "stage.json", {"step": stage})
    return plan


def write_predictions(output, stage, corrupt=False):
    target = output / "eval" / f"step-{stage:03d}" / "math"
    target.mkdir(parents=True, exist_ok=True)
    rows = [{"idx": i, "question": r["question"], "level": r["level"], "score": [i != 1]}
            for i, r in enumerate(exam_rows())]
    if corrupt:
        rows[-1]["idx"] = 0
    (target / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class TrainingUnitTests(unittest.TestCase):
    def test_batch_grouping_and_runtime_order_guard(self):
        self.assertEqual(list(train.OrderedGroupSampler(3, 4)), [0]*4 + [1]*4 + [2]*4)
        with tempfile.TemporaryDirectory() as tmp:
            reward = train.make_reward([{"id": "a", "difficulty": "easy"}], 4,
                                       FakeGrader(), Path(tmp) / "rewards.jsonl")
            with self.assertRaisesRegex(ValueError, "order"):
                reward(["x"]*4, ["2"]*4, ["b"]*4, [0]*4, ["easy"]*4)
            self.assertEqual(reward(["x"]*4, ["2"]*4, ["a"]*4, [0]*4, ["easy"]*4), [1, 0, 1, 0])
            logged = train.read_jsonl(Path(tmp) / "rewards.jsonl")
            self.assertEqual(logged[0]["step"], 1)
            self.assertFalse(logged[0]["zero_reward_variance"])
            with self.assertRaisesRegex(ValueError, "More training"):
                reward(["x"]*4, ["2"]*4, ["a"]*4, [0]*4, ["easy"]*4)

    def test_prepared_data_validation_and_tamper_detection(self):
        cfg = load_config(train.ROOT / "experiment.json")
        splits = {}
        for split in ("train", "test"):
            splits[split] = [{"id": f"{split}/{level}/{i}", "split": split,
                              "level": level, "problem": f"{split} {level} {i}",
                              "solution": "Answer: \\boxed{2}",
                              "fingerprint": fingerprint(f"{split} {level} {i}")}
                             for level in range(1, 6) for i in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifacts"
            save_artifacts(output, cfg, splits, [])
            _, random, exam, _ = train.load_experiment(output, "v1_random")
            _, curriculum, _, _ = train.load_experiment(output, "v1_curriculum")
            self.assertEqual(len(exam), 90)
            self.assertEqual({r["id"] for r in random}, {r["id"] for r in curriculum})
            self.assertEqual([r["difficulty"] for r in curriculum], ["easy"]*100 + ["medium"]*100 + ["hard"]*100)
            with (output / "train_curriculum.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                train.load_experiment(output, "v1_random")

    def test_all_stages_use_existing_evaluator_and_same_exam(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            plan = evaluation_fixture(output)
            calls = []

            def fake_run(command, **kwargs):
                self.assertEqual(Path(command[1]), train.EVALUATOR)
                self.assertEqual(command[command.index("--data_dir")+1], str(output.resolve() / "eval_data"))
                self.assertIn("--overwrite", command)
                self.assertNotIn("--shuffle", command)
                self.assertEqual(kwargs["cwd"], train.EVAL_ROOT)
                self.assertTrue(kwargs["check"])
                folder = Path(command[command.index("--output_dir")+1])
                stage = int(folder.name.split("-")[1])
                calls.append(stage)
                write_predictions(output, stage)

            args = SimpleNamespace(run_dir=output, python="/test/python", dry_run=False)
            with patch.object(train.subprocess, "run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                train.evaluate_stages(args)
                # Completed, unchanged evaluations are reused, avoiding unnecessary GPU runs.
                train.evaluate_stages(args)
            self.assertEqual(calls, plan["stages"])
            results = train.read_json(output / "stage_metrics.json")
            self.assertEqual(results[-1]["metrics"]["overall"]["correct"], 2)
            self.assertEqual(results[-1]["metrics"]["medium"]["accuracy"], 0)
            self.assertEqual(results[-1]["metrics"]["hard"]["accuracy"], 100)
            self.assertEqual(len((output / "stage_metrics.csv").read_text().splitlines()), 17)
            write_predictions(output, 200, corrupt=True)
            with self.assertRaisesRegex(ValueError, "Incomplete/duplicate"):
                train.summarize_stage(output, 200)

    def test_dry_run_never_launches_and_changed_exam_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            evaluation_fixture(output)
            args = SimpleNamespace(run_dir=output, python="python", dry_run=True)
            with patch.object(train.subprocess, "run") as launch, contextlib.redirect_stdout(io.StringIO()):
                train.evaluate_stages(args)
            launch.assert_not_called()
            self.assertFalse((output / "eval").exists())
            with (output / "eval_data/math/test.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "data changed"):
                train.evaluate_stages(args)


@unittest.skipUnless(os.environ.get("YJ_TRAINING_INTEGRATION") == "1",
                     "Set YJ_TRAINING_INTEGRATION=1 in the training environment for real CPU GRPO tests")
class TrainingIntegrationTests(unittest.TestCase):
    def test_existing_grader_math_and_bounded_worker(self):
        grader = train.Grader(10)
        try:
            self.assertEqual(grader.score("The answer is \\boxed{1/2}", "0.5")[0], 1)
            self.assertEqual(grader.score("The answer is \\boxed{3}", "2")[0], 0)
        finally:
            grader.close()

    def test_real_grpo_updates_checkpoints_and_exports(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
        from trl import GRPOConfig, GRPOTrainer
        train.check_versions()
        torch.set_num_threads(1)
        set_seed(42)
        tokens = ["<pad>", "<eos>", "<unk>", "Question", "0", "1", "2", "Answer", ":", "3", "4", "5"]
        backend = Tokenizer(WordLevel({s: i for i, s in enumerate(tokens)}, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
                                            unk_token="<unk>", padding_side="left")
        cfg = Qwen2Config(vocab_size=len(tokens), hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, eos_token_id=1, pad_token_id=0,
                          tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg)
        t = load_config(train.ROOT / "experiment.json")["training"]
        t = dict(t, max_prompt_length=32, max_completion_length=4, learning_rate=0.01)
        rows = [{"id": str(i), "difficulty": g} for i, g in enumerate(("easy", "medium", "hard"))]
        examples = [{"prompt": f"Question {i} Answer :", "answer": "2", "problem_id": r["id"],
                     "position": i, "difficulty": r["difficulty"]} for i, r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            base = output / "initial"
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            reward_path = output / "rewards.jsonl"
            trainer = train.make_trainer_class(GRPOTrainer)(
                model=model, processing_class=tokenizer,
                args=GRPOConfig(**train.training_kwargs(t, 3, 1, 42, output, cpu=True)),
                train_dataset=Dataset.from_list(examples),
                reward_funcs=train.make_reward(rows, 4, FakeGrader(), reward_path),
                peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                       task_type="CAUSAL_LM"),
            )
            before = train.adapter_digest(trainer.model)
            trainer.train()
            self.assertNotEqual(before, train.adapter_digest(trainer.model))
            self.assertEqual(trainer.state.global_step, 3)
            logs = train.read_jsonl(reward_path)
            self.assertEqual([r["problem_id"] for r in logs], ["0", "1", "2"])
            self.assertEqual([len(r["rewards"]) for r in logs], [4, 4, 4])
            for step in (1, 2, 3):
                checkpoint = output / f"checkpoint-{step}"
                self.assertEqual(train.read_json(checkpoint / "trainer_state.json")["global_step"], step)
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
                    self.assertTrue((checkpoint / name).exists(), name)
            plan = {"status": "trained", "base_model_path": str(base), "stages": [0, 1, 2, 3],
                    "initial_model": {"repo_id": "tiny-local-test", "revision": "a"*40}}
            train.write_json(output / "run.json", plan)
            train.export_stages(output)
            self.assertEqual(train.read_json(output / "run.json")["status"], "exported")
            hashes = set()
            for step in (1, 2, 3):
                exported = output / "models" / f"step-{step:03d}"
                reloaded = AutoModelForCausalLM.from_pretrained(exported)
                self.assertFalse(hasattr(reloaded, "peft_config"))
                self.assertTrue((exported / "stage.json").exists())
                with torch.no_grad():
                    logits = reloaded(**tokenizer("Question 0", return_tensors="pt")).logits
                self.assertTrue(torch.isfinite(logits).all())
                hashes.add(train.digest_file(exported / "model.safetensors"))
            self.assertEqual(len(hashes), 3)


if __name__ == "__main__":
    unittest.main()
