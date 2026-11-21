"""Week 2 diagnosis: why does one run score what it scores?

Runs on: main Colab notebook, Drive mounted. CPU is enough. Reads only (no model calls,
no tests): week2/generations/<run>.jsonl joined with week2/test_results/test_results_*.jsonl.

For one run and one dataset it prints, over every question that has an answer:
  * how many pass / fail / time out / have no code
  * the failures grouped by the exception that ended the test program
  * pass rate for answers that define a function vs. not, fenced vs. unfenced
  * 8 failing answers (task id, error, first lines of the code) to read by eye
It also writes the same text to week2/diagnose_<run>_<dataset>.txt.

Default: the anchor check (Llama 3.1 8B, no retrieval, DS-1000).

Usage:  python -m coding_track.steps.w2_diagnose
        python -m coding_track.steps.w2_diagnose --run llama3.1-8b__no_retrieval --dataset odex
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from coding_track import config
from coding_track.common.io_utils import atomic_write_text, read_jsonl

SAMPLE_N = 8


def error_kind(error: str) -> str:
    """'AssertionError: x' -> 'AssertionError'; unknown text -> its first 40 characters."""
    m = re.match(r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt))\b", error or "")
    return m.group(1) if m else (error or "(empty)")[:40]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="llama3.1-8b__no_retrieval")
    parser.add_argument("--dataset", default="ds1000", choices=["ds1000", "odex", "bigcodebench"])
    args = parser.parse_args()

    gen_path = config.GENERATIONS_DIR / f"{args.run}.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"Missing {gen_path}")
    gens = {r["task_id"]: r for r in read_jsonl(gen_path) if r["dataset"] == args.dataset}

    tests = {}
    for path in sorted(config.TEST_RESULTS_DIR.glob("test_results_*.jsonl"),
                       key=lambda p: p.stat().st_mtime):
        for r in read_jsonl(path):
            if r["run"] == args.run and r["dataset"] == args.dataset:
                tests[r["task_id"]] = r
    if not tests:
        raise SystemExit(f"No test results for {args.run} / {args.dataset} in {config.TEST_RESULTS_DIR}")

    rows = [(gens[t], tests[t]) for t in tests if t in gens]
    out = [f"Diagnosis: {args.run} / {args.dataset}: {len(rows)} answers with a test result "
           f"({len(gens)} answers, {len(tests)} test rows)", ""]

    status = Counter(t["status"] for _, t in rows)
    out.append("Status: " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    out.append(f"pass@1 over all of them: {pct(status['pass'], len(rows))}")
    out.append("")

    fails = [(g, t) for g, t in rows if t["status"] == "fail"]
    out.append(f"Failures by exception ({len(fails)}):")
    for kind, n in Counter(error_kind(t.get("error", "")) for _, t in fails).most_common(12):
        out.append(f"  {n:4d}  {kind}")
    out.append("")

    def split(label: str, key) -> None:
        out.append(f"Pass rate by {label}:")
        groups: dict[str, list[bool]] = {}
        for g, t in rows:
            groups.setdefault(str(key(g)), []).append(t["status"] == "pass")
        for k, v in sorted(groups.items()):
            out.append(f"  {k:<10} {pct(sum(v), len(v))}")
        out.append("")

    split("code defines a function", lambda g: bool(re.search(r"^def \w+\(", g.get("code") or "", re.M)))
    split("code unfenced", lambda g: bool(g.get("code_unfenced")))
    split("hit length limit", lambda g: g.get("done_reason") == "length")

    out.append(f"{SAMPLE_N} failing answers (every {max(1, len(fails) // SAMPLE_N)}th):")
    step = max(1, len(fails) // SAMPLE_N)
    for g, t in fails[::step][:SAMPLE_N]:
        head = "\n".join((g.get("code") or "").splitlines()[:6])
        out.append(f"--- {g['task_id']}  error: {t.get('error', '')[:160]}")
        out.append(head)
    text = "\n".join(out)
    print(text)
    dest = config.WEEK2_DIR / f"diagnose_{args.run}_{args.dataset}.txt"
    atomic_write_text(dest, text)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
