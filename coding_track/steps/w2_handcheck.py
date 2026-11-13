"""Week 2, Day 6: draw 50 random scored answers for a hand check of extraction and pass/fail.

Runs on: main Colab notebook, Drive mounted. CPU is enough.
Writes week2/handcheck_50.md: for each answer the question, the raw answer, the
extracted code and the test result, with two lines for you to fill in:
  extraction correct? (yes/no)    pass/fail looks right? (yes/no)
Count the "no"s yourself to get the error rate; nothing is filled in for you.

Usage:  python -m coding_track.steps.w2_handcheck
"""

from __future__ import annotations

import random

from coding_track import config
from coding_track.common.io_utils import atomic_write_text, read_jsonl
from coding_track.steps.w2_score import load_test_results


def main() -> None:
    tests, _, _ = load_test_results()
    pool = []
    for path in sorted(config.GENERATIONS_DIR.glob("*.jsonl")):
        for r in read_jsonl(path):
            if "scored" in r["question_sets"] and (path.stem, r["task_id"]) in tests:
                pool.append((path.stem, r))
    if len(pool) < config.HANDCHECK_N:
        raise SystemExit(f"Only {len(pool)} tested answers; run the tests first.")
    sample = random.Random(config.SEED).sample(pool, config.HANDCHECK_N)

    parts = [f"# Hand check of {config.HANDCHECK_N} random answers (seed {config.SEED})", "",
             "Fill in both lines for every answer, then count the 'no's.", ""]
    for i, (run, r) in enumerate(sample, 1):
        t = tests[(run, r["task_id"])]
        parts += [f"## {i}. {run} / {r['task_id']}",
                  f"test result: **{t['status']}** {t.get('error', '')[:300]}",
                  "", "**Raw answer:**", "````", r["raw_answer"], "````",
                  "", "**Extracted code (sent to the tests):**", "````python", r["code"] or "(none)", "````",
                  "", "- extraction correct? ____", "- pass/fail looks right? ____", ""]
    atomic_write_text(config.HANDCHECK_FILE, "\n".join(parts))
    print(f"wrote {config.HANDCHECK_FILE}")


if __name__ == "__main__":
    main()
