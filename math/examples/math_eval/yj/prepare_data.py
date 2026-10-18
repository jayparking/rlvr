"""Prepare paired experiments from pinned official MATH train/test splits.

No model weights are downloaded and no GPU training is started.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(problem):
    return hashlib.sha256(re.sub(r"\s+", "", problem).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if cfg["groups"] != {"easy": [1, 2], "medium": [3], "hard": [4, 5]}:
        raise ValueError("Expected Level 1–2 / 3 / 4–5 groups")
    if cfg["curriculum_order"] != ["easy", "medium", "hard"]:
        raise ValueError("Expected easy -> medium -> hard")
    for key in ("train_per_group", "test_per_group"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for source in [cfg["dataset"], *cfg["models"].values()]:
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("All sources must use immutable 40-character commit SHAs")
    return cfg


def load_sources(cfg, cache_dir, offline=False):
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    source = cfg["dataset"]
    splits = {"train": [], "test": []}
    provenance = []
    for split in splits:
        for subject in source["subjects"]:
            filename = f"{subject}/{split}-00000-of-00001.parquet"
            path = hf_hub_download(
                repo_id=source["repo_id"], repo_type="dataset",
                revision=source["revision"], filename=filename,
                cache_dir=str(cache_dir), local_files_only=offline,
            )
            rows = pq.read_table(path).to_pylist()
            provenance.append({"file": filename, "sha256": digest_file(path), "rows": len(rows)})
            for index, row in enumerate(rows):
                problem, solution = row["problem"], row["solution"]
                if not problem.strip() or not solution.strip():
                    raise ValueError(f"Empty problem/solution in {filename}:{index}")
                match = re.fullmatch(r"Level ([1-5])", row["level"])
                if not match:
                    raise ValueError(f"Invalid level: {row['level']!r}")
                # This mirror omits original MATH filenames. Use pinned parquet row IDs.
                uid = f"{split}/{subject}/{index:05d}"
                splits[split].append({
                    "id": uid, "unique_id": uid, "split": split,
                    "source_file": filename, "source_row": index,
                    "problem": problem, "question": problem, "solution": solution,
                    "subject": subject, "level": int(match[1]),
                    "fingerprint": fingerprint(problem),
                })
    if {k: len(v) for k, v in splits.items()} != {"train": 7500, "test": 5000}:
        raise ValueError("Expected complete official MATH splits: train=7500, test=5000")
    return splits, provenance


def select(rows, cfg, split, excluded_hashes=()):
    mapping = {level: group for group, levels in cfg["groups"].items() for level in levels}
    buckets = {group: [] for group in cfg["curriculum_order"]}
    seen, ids = set(excluded_hashes), set()
    removed = 0
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["split"] != split or not row["id"].startswith(split + "/"):
            raise ValueError("Source split/ID mismatch")
        if row["id"] in ids:
            raise ValueError(f"Duplicate source ID: {row['id']}")
        ids.add(row["id"])
        if row["fingerprint"] in seen:
            removed += 1
            continue
        seen.add(row["fingerprint"])
        group = mapping[row["level"]]
        buckets[group].append(dict(row, difficulty=group))
    selected = []
    count = cfg[f"{split}_per_group"]
    for group in cfg["curriculum_order"]:
        candidates = buckets[group]
        if len(candidates) < count:
            raise ValueError(f"{split}/{group}: need {count}, have {len(candidates)}")
        # Independent random streams keep selection stable across groups and splits.
        rng = random.Random(f"{cfg['seed']}:{split}:{group}:selection")
        selected.extend(rng.sample(candidates, count))
    return selected, {"removed_duplicates_or_overlap": removed,
                      "available": {k: len(v) for k, v in buckets.items()}}


def build_schedule(rows, cfg, mode):
    if mode not in ("random", "curriculum"):
        raise ValueError(f"Unknown mode: {mode}")
    ordered = sorted(rows, key=lambda r: r["id"])
    random.Random(f"{cfg['seed']}:training_order").shuffle(ordered)
    if mode == "curriculum":
        ranks = {group: i for i, group in enumerate(cfg["curriculum_order"])}
        ordered.sort(key=lambda r: ranks[r["difficulty"]])
    return [dict(row, position=i) for i, row in enumerate(ordered)]


def prepare(cfg, splits):
    # Exclude all official test questions from train, not just the selected 90.
    test_hashes = {row["fingerprint"] for row in splits["test"]}
    train, train_stats = select(splits["train"], cfg, "train", test_hashes)
    test, test_stats = select(splits["test"], cfg, "test")
    return train, test, {"train": train_stats, "test": test_stats}


def save_artifacts(output, cfg, splits, provenance):
    train, test, stats = prepare(cfg, splits)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    write_jsonl(output / "train_selected.jsonl", train)
    write_jsonl(output / "test_selected.jsonl", test)
    # Existing math_eval_a100.py accepts this through --data_dir without edits.
    write_jsonl(output / "eval_data" / "math" / "test.jsonl", test)
    for mode in ("random", "curriculum"):
        write_jsonl(output / f"train_{mode}.jsonl", build_schedule(train, cfg, mode))
    artifacts = {str(p.relative_to(output)): digest_file(p)
                 for p in sorted(output.rglob("*.jsonl"))}
    runs = {}
    for version, model in cfg["models"].items():
        for mode in ("random", "curriculum"):
            name = f"{version}_{mode}"
            runs[name] = {
                "initial_model": model, "tokenizer": model,
                "resume_from_checkpoint": None, "seed": cfg["seed"],
                "train_file": f"train_{mode}.jsonl", "eval_file": "test_selected.jsonl",
                "shuffle": False, "output_dir": f"runs/{name}",
            }
    write_json(output / "runs.json", runs)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg["seed"], "dataset": cfg["dataset"], "models": cfg["models"],
        "counts": {"train": dict(Counter(r["difficulty"] for r in train)),
                   "test": dict(Counter(r["difficulty"] for r in test))},
        "selection_stats": stats, "source_files": provenance,
        "artifact_sha256": artifacts,
    }
    write_json(output / "manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "seed42")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "huggingface")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.preview:
        parser.error(f"Output already exists; choose a new directory: {args.output}")
    cfg = load_config(args.config)
    splits, provenance = load_sources(cfg, args.cache_dir, args.offline)
    if args.preview:
        train, test, stats = prepare(cfg, splits)
        print(json.dumps({"train": len(train), "test": len(test), "stats": stats}, indent=2))
    else:
        summary = save_artifacts(args.output, cfg, splits, provenance)
        print(json.dumps({"output": str(args.output), "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
