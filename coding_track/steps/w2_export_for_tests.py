"""Week 2, Day 6: pack every saved answer into one file for the test notebook.

Runs on: main Colab notebook, Drive mounted. CPU is enough.
The file holds only what the tests need (run name, task, extracted code); no
keys, no Drive paths. Download it and upload it to the test notebook.

Output: week2/answers_for_tests.jsonl (or --out)

Usage:  python -m coding_track.steps.w2_export_for_tests
        python -m coding_track.steps.w2_export_for_tests --only llama3.1-8b__clean_rag --out pilot.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from coding_track import config
from coding_track.common.io_utils import atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answers for the test runtime.")
    parser.add_argument("--only", nargs="+", help="run names (file stems in week2/generations)")
    parser.add_argument("--out", type=Path, default=config.TEST_EXPORT_FILE)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else config.WEEK2_DIR / args.out

    rows = []
    for path in sorted(config.GENERATIONS_DIR.glob("*.jsonl")):
        if args.only and path.stem not in args.only:
            continue
        for r in read_jsonl(path):
            rows.append({"run": path.stem, "task_id": r["task_id"], "dataset": r["dataset"],
                         "native_id": r["native_id"], "code": r["code"], "all_code": r["all_code"]})
    if not rows:
        raise SystemExit(f"No answers found in {config.GENERATIONS_DIR}")
    atomic_write_jsonl(out, rows)
    for run, n in sorted(Counter(r["run"] for r in rows).items()):
        print(f"  {run}: {n} answers")
    print(f"wrote {len(rows)} answers -> {out}")


if __name__ == "__main__":
    main()
