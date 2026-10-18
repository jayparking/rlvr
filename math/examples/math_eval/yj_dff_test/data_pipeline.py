"""Deterministic, shared problem selection; no GPU or trainer imports."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT.parent


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_dir"] = str(path.parent)
    levels = cfg["data"]["levels"]
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("data.levels must be nonempty and unique")
    if any(type(n) is not int or n not in range(1, 6) for n in levels):
        raise ValueError("MATH levels must be integers 1..5")
    assigned = []
    for name, group in cfg["groups"].items():
        if not isinstance(name, str) or not group:
            raise ValueError("Each named group must contain levels")
        assigned.extend(group)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(levels):
        raise ValueError("groups must cover selected levels exactly once")
    order = cfg["curriculum"]["order"]
    if len(order) != len(set(order)) or set(order) != set(cfg["groups"]):
        raise ValueError("curriculum.order must contain each group exactly once")
    for key in ("repeats_per_problem", "num_generations", "max_prompt_length",
                "max_completion_length", "save_steps", "logging_steps"):
        if type(cfg["training"][key]) is not int or cfg["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if cfg["training"]["num_generations"] < 2:
        raise ValueError("GRPO requires num_generations >= 2")
    if not 0 < cfg["training"]["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if cfg["training"]["temperature"] <= 0:
        raise ValueError("temperature must be positive for RL sampling")
    if cfg["training"]["learning_rate"] <= 0 or cfg["training"]["beta"] < 0:
        raise ValueError("learning_rate must be positive and beta nonnegative")
    limit = cfg["data"].get("max_per_group")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_per_group must be null or a positive integer")
    if not cfg["data"].get("exclude_files"):
        raise ValueError("Specify held-out evaluation data in exclude_files")
    return cfg


def resolve(cfg, path):
    return (Path(cfg["_config_dir"]) / path).resolve()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question(row):
    text = row.get("problem", row.get("question"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing problem/question text")
    return text


def fingerprint(row):
    # Whitespace-normalized exact duplicate detection, not semantic deduplication.
    normalized = re.sub(r"\s+", "", question(row))
    return hashlib.sha256(normalized.encode()).hexdigest()


def level_number(value):
    match = re.fullmatch(r"(?:Level\s+)?([1-5])", str(value).strip())
    if not match:
        raise ValueError(f"Invalid/missing MATH level: {value!r}")
    return int(match[1])


def prepare(cfg):
    held_ids, held_hashes = set(), set()
    for path in cfg["data"]["exclude_files"]:
        for row in read_jsonl(resolve(cfg, path)):
            if row.get("unique_id"):
                held_ids.add(str(row["unique_id"]))
            held_hashes.add(fingerprint(row))
    mapping = {level: group for group, levels in cfg["groups"].items() for level in levels}
    buckets = {name: [] for name in cfg["groups"]}
    seen_ids, seen_hashes = set(), set()
    stats = Counter()
    for row in read_jsonl(resolve(cfg, cfg["data"]["train_file"])):
        stats["source_rows"] += 1
        level = level_number(row.get("level"))
        digest = fingerprint(row)
        uid = str(row.get("unique_id") or digest)
        if uid in held_ids or digest in held_hashes:
            stats["excluded_eval_overlap"] += 1
            continue
        if level not in mapping:
            stats["excluded_level"] += 1
            continue
        if uid in seen_ids or digest in seen_hashes:
            stats["excluded_duplicates"] += 1
            continue
        if not row.get("solution"):
            raise ValueError(f"Missing solution for existing math answer parser: {uid}")
        seen_ids.add(uid)
        seen_hashes.add(digest)
        buckets[mapping[level]].append({
            "id": uid, "question": question(row), "solution": row["solution"],
            "level": level, "difficulty": mapping[level], "fingerprint": digest,
        })
    rng = random.Random(cfg["training"]["seed"])
    selected = []
    for name in cfg["curriculum"]["order"]:
        rows = sorted(buckets[name], key=lambda r: r["id"])
        rng.shuffle(rows)
        limit = cfg["data"].get("max_per_group")
        if limit is not None:
            stats["excluded_group_cap"] += max(0, len(rows) - limit)
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"Group {name!r} has no problems after filtering")
        selected.extend(rows)
    stats["selected"] = len(selected)
    summary = dict(stats)
    summary["groups"] = dict(Counter(row["difficulty"] for row in selected))
    summary["levels"] = dict(Counter(row["level"] for row in selected))
    summary["updates"] = len(selected) * cfg["training"]["repeats_per_problem"]
    summary["selection_sha256"] = hashlib.sha256(
        json.dumps(sorted((r["id"], r["fingerprint"]) for r in selected)).encode()
    ).hexdigest()
    return selected, summary


def build_schedule(rows, cfg, mode):
    if mode not in ("random", "curriculum"):
        raise ValueError("order must be random or curriculum")
    rank = {name: i for i, name in enumerate(cfg["curriculum"]["order"])}
    schedule = []
    for epoch in range(cfg["training"]["repeats_per_problem"]):
        ordered = list(rows)
        random.Random(cfg["training"]["seed"] + epoch).shuffle(ordered)
        if mode == "curriculum":
            ordered.sort(key=lambda row: rank[row["difficulty"]])
        for row in ordered:
            schedule.append(dict(row, repeat=epoch + 1, position=len(schedule)))
    return schedule


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
