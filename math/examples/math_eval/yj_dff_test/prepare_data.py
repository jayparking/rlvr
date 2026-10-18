import argparse
import json
from pathlib import Path

from data_pipeline import ROOT, build_schedule, load_config, prepare, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Preview groups and paired training orders without loading a model")
    parser.add_argument("--config", default=str(ROOT / "experiment.yaml"))
    parser.add_argument("--preview", action="store_true", help="Print only; do not write artifacts")
    parser.add_argument("--output", type=Path, help="New output directory for both schedules")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary = prepare(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    schedules = {mode: build_schedule(rows, cfg, mode) for mode in ("random", "curriculum")}
    for mode, schedule in schedules.items():
        print(f"\n{mode}: first 12 (position, id, level, group)")
        for row in schedule[:12]:
            print(row["position"], row["id"], row["level"], row["difficulty"])
    if args.output and not args.preview:
        args.output.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output / "selected.jsonl", rows)
        for mode, schedule in schedules.items():
            write_jsonl(args.output / f"{mode}.jsonl", schedule)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
