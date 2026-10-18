import json
from pathlib import Path
import tempfile
import unittest

from prepare_data import ROOT, build_schedule, fingerprint, load_config, prepare, save_artifacts


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "experiment.json")
        self.splits = {}
        for split in ("train", "test"):
            self.splits[split] = [
                {"id": f"{split}/{level}/{i}", "split": split, "level": level,
                 "problem": f"{split} problem {level} {i}",
                 "fingerprint": fingerprint(f"{split} problem {level} {i}")}
                for level in range(1, 6) for i in range(120)
            ]

    def test_sizes_order_and_reproducibility(self):
        train, test, _ = prepare(self.cfg, self.splits)
        self.assertEqual((len(train), len(test)), (300, 90))
        reverse = {k: list(reversed(v)) for k, v in self.splits.items()}
        self.assertEqual(prepare(self.cfg, self.splits), prepare(self.cfg, reverse))
        random = build_schedule(train, self.cfg, "random")
        curriculum = build_schedule(train, self.cfg, "curriculum")
        self.assertEqual({r["id"] for r in random}, {r["id"] for r in curriculum})
        self.assertEqual([r["difficulty"] for r in curriculum],
                         [g for g in ("easy", "medium", "hard") for _ in range(100)])
        self.assertNotEqual([r["id"] for r in random], [r["id"] for r in curriculum])
        changed = dict(self.cfg, seed=43)
        self.assertNotEqual(train, prepare(changed, self.splits)[0])

    def test_leakage_and_insufficient_group(self):
        contaminated = dict(self.splits["test"][0], id="train/overlap", split="train")
        self.splits["train"].append(contaminated)
        train, _, stats = prepare(self.cfg, self.splits)
        self.assertNotIn(contaminated["fingerprint"], {r["fingerprint"] for r in train})
        self.assertEqual(stats["train"]["removed_duplicates_or_overlap"], 1)
        self.splits["train"] = [r for r in self.splits["train"] if r["level"] != 3]
        with self.assertRaises(ValueError):
            prepare(self.cfg, self.splits)

    def test_independent_runs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "experiment"
            save_artifacts(output, self.cfg, self.splits, [])
            runs = json.loads((output / "runs.json").read_text())
            self.assertEqual(len(runs), 4)
            for name, run in runs.items():
                self.assertEqual(run["initial_model"], self.cfg["models"][name.split("_")[0]])
                self.assertIsNone(run["resume_from_checkpoint"])
                self.assertFalse(run["shuffle"])
            with self.assertRaises(FileExistsError):
                save_artifacts(output, self.cfg, self.splits, [])


if __name__ == "__main__":
    unittest.main()
