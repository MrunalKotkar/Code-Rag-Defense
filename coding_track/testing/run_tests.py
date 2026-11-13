"""Run the unit tests on the test runtime.

Runs on: the separate Colab CPU test notebook (no Drive, no keys), after
`python -m coding_track.testing.build_envs`.

Modes:
  codex      DS-1000 check: score DS-1000's own Codex-002 answers and compare
             with the scores in DS-1000's README (Overall 0.388).
  reference  Run every official reference solution (DS-1000, ODEX, BigCodeBench)
             and list the ones that fail, with the reason. Those tasks are left
             out of accuracy later (still used for the attack check).
  answers    Score the models' answers (answers_for_tests.jsonl exported by the
             main notebook) and run Bandit on every answer.

Outputs go to /content/test_outputs/ (download them and put them in
MyDrive/coding_track/week2/test_results/).

Usage:  python -m coding_track.testing.run_tests codex
        python -m coding_track.testing.run_tests reference
        python -m coding_track.testing.run_tests answers --input answers_for_tests.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from coding_track.testing import env_spec, harness

OUT = env_spec.OUTPUT_DIR


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# codex: reproduce DS-1000's published Codex-002 numbers
# ---------------------------------------------------------------------------
def ds1000_postprocess(code) -> str:
    """Copied from DS-1000's test_ds1000.py (postprocess)."""
    if type(code) is list:
        code = code[0]
    code = code.split('</code>')[0]
    code = code.replace('```python', '')
    code = code.split('```')[0]
    code = code.split('\nEND SOLUTION')[0]
    code = code.replace('<code>', '')
    return code


def mode_codex(workers: int | None) -> None:
    problems = harness.ds1000_problems()
    with urllib.request.urlopen(env_spec.DS1000_CODEX_ANSWERS_URL, timeout=120) as resp:
        lines = [json.loads(l) for l in resp.read().decode("utf-8").splitlines() if l.strip()]
    answers = [ds1000_postprocess(l["code"]) for l in lines]     # line i = problem_id i
    jobs = [(pid, harness.ds1000_program(p, answers[int(pid)])) for pid, p in problems.items()]
    print(f"DS-1000 Codex-002: {len(jobs)} programs, timeout {env_spec.DS1000_TIMEOUT_S} s")
    results = harness.run_many(jobs, env_spec.env_python("ds1000"), env_spec.DS1000_TIMEOUT_S, workers)

    by_lib: dict[str, list[int]] = defaultdict(list)
    for pid, res in results.items():
        by_lib[problems[pid]["metadata"]["library"]].append(int(res["status"] == "pass"))
    scores = {lib: round(sum(v) / len(v), 3) for lib, v in sorted(by_lib.items())}
    all_scores = [s for v in by_lib.values() for s in v]
    scores["overall"] = round(sum(all_scores) / len(all_scores), 3)

    print(f"\n{'library':12s} {'ours':>6s} {'expected':>8s} {'diff':>7s}")
    for lib, ours in scores.items():
        exp = env_spec.DS1000_CODEX_EXPECTED[lib]
        print(f"{lib:12s} {ours:6.3f} {exp:8.3f} {ours - exp:+7.3f}")
    exact = all(scores[k] == v for k, v in env_spec.DS1000_CODEX_EXPECTED.items())
    print("\nEXACT MATCH with DS-1000's README." if exact else
          "\nNOT an exact match: look at the diffs and the failures before going on.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ds1000_codex_check.json").write_text(json.dumps(
        {"scores": scores, "expected": env_spec.DS1000_CODEX_EXPECTED, "exact_match": exact,
         "per_problem": results}, indent=2))


# ---------------------------------------------------------------------------
# reference: every official reference solution
# ---------------------------------------------------------------------------
BCB_CANONICAL_SCRIPT = (
    "import json,sys\nfrom bigcodebench.data import get_bigcodebench\n"
    "d=get_bigcodebench(subset='full')\n"
    "json.dump({k:v['code_prompt']+v['canonical_solution'] for k,v in d.items()},open(sys.argv[1],'w'))\n")


def mode_reference(workers: int | None, datasets: list[str]) -> None:
    rows = []
    if "ds1000" in datasets:
        problems = harness.ds1000_problems()
        jobs = [(pid, harness.ds1000_program(p, p["reference_code"])) for pid, p in problems.items()]
        print(f"DS-1000 reference solutions: {len(jobs)}")
        for pid, res in harness.run_many(jobs, env_spec.env_python("ds1000"),
                                         env_spec.DS1000_TIMEOUT_S, workers).items():
            rows.append({"dataset": "ds1000", "native_id": pid, **res})
    if "odex" in datasets:
        problems = harness.odex_problems()
        jobs = [(nid, harness.odex_program(p, p["canonical_solution"], is_reference=True))
                for nid, p in problems.items()]
        print(f"ODEX reference solutions: {len(jobs)}")
        for nid, res in harness.run_many(jobs, env_spec.env_python("odex"),
                                         env_spec.ODEX_TIMEOUT_S, workers).items():
            rows.append({"dataset": "odex", "native_id": nid, **res})
    if "bigcodebench" in datasets:
        # A fresh ground-truth timing cache for the full set (the evaluator caches by dataset hash).
        for pkl in harness.bigcodebench_cache_dir().glob("*.pkl"):
            pkl.unlink()
        with tempfile.TemporaryDirectory() as tmp:
            canon = Path(tmp) / "canonical.json"
            subprocess.run([str(env_spec.env_python("bcb")), "-c", BCB_CANONICAL_SCRIPT, str(canon)],
                           check=True)
            solutions = json.loads(canon.read_text())
            print(f"BigCodeBench reference solutions: {len(solutions)}")
            for tid, res in harness.run_bigcodebench(solutions, Path(tmp) / "bcb").items():
                rows.append({"dataset": "bigcodebench", "native_id": tid, "status": res["status"],
                             "error": res["error"]})
    write_jsonl(OUT / "reference_results.jsonl", rows)
    for ds in datasets:
        mine = [r for r in rows if r["dataset"] == ds]
        failed = [r for r in mine if r["status"] != "pass"]
        print(f"\n{ds}: {len(mine) - len(failed)}/{len(mine)} reference solutions pass "
              f"({(len(mine) - len(failed)) / max(1, len(mine)):.1%})")
        for r in failed:
            print(f"  FAIL {r['native_id']}: {r['status']} {r['error'][:160]}")


# ---------------------------------------------------------------------------
# answers: the models' answers + Bandit
# ---------------------------------------------------------------------------
def run_bandit(answers: list[dict]) -> dict[int, dict]:
    """Bandit findings per answer index (all code blocks of the answer)."""
    findings: dict[int, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, a in enumerate(answers):
            if a.get("all_code"):
                (Path(tmp) / f"a{i}.py").write_text(a["all_code"], encoding="utf-8")
        proc = subprocess.run([sys.executable, "-m", "bandit", "-r", tmp, "-f", "json", "-q"],
                              capture_output=True, text=True)
        report = json.loads(proc.stdout or "{}")
    for i, a in enumerate(answers):
        findings[i] = {"bandit_findings": 0, "bandit_by_severity": {}, "bandit_tests": [],
                       "bandit_error": not a.get("all_code")}
    for issue in report.get("results", []):
        i = int(Path(issue["filename"]).stem[1:])
        f = findings[i]
        f["bandit_findings"] += 1
        sev = issue["issue_severity"]
        f["bandit_by_severity"][sev] = f["bandit_by_severity"].get(sev, 0) + 1
        f["bandit_tests"].append(issue["test_id"])
    for err in report.get("errors", []):
        findings[int(Path(err["filename"]).stem[1:])]["bandit_error"] = True
    return findings


def mode_answers(path: Path, workers: int | None) -> None:
    answers = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"{len(answers)} answers from {path}")
    results: dict[int, dict] = {}

    ds_problems = harness.ds1000_problems()
    jobs = [(str(i), harness.ds1000_program(ds_problems[a["native_id"]], a["code"]))
            for i, a in enumerate(answers) if a["dataset"] == "ds1000" and a["code"] is not None]
    print(f"DS-1000: {len(jobs)} answers with code")
    for k, res in harness.run_many(jobs, env_spec.env_python("ds1000"),
                                   env_spec.DS1000_TIMEOUT_S, workers).items():
        results[int(k)] = res

    odex = harness.odex_problems()
    jobs = [(str(i), harness.odex_program(odex[a["native_id"]], a["code"]))
            for i, a in enumerate(answers) if a["dataset"] == "odex" and a["code"] is not None]
    print(f"ODEX: {len(jobs)} answers with code")
    for k, res in harness.run_many(jobs, env_spec.env_python("odex"),
                                   env_spec.ODEX_TIMEOUT_S, workers).items():
        results[int(k)] = res

    bcb_by_run: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(answers):
        if a["dataset"] == "bigcodebench":
            bcb_by_run[a["run"]].append(i)
    with tempfile.TemporaryDirectory() as tmp:
        for n, (run, idxs) in enumerate(sorted(bcb_by_run.items())):
            # Every selected task needs a sample; an answer without code gets an empty one.
            solutions = {answers[i]["native_id"]: answers[i]["code"] or "" for i in idxs}
            print(f"BigCodeBench [{run}]: {len(solutions)} answers")
            bcb = harness.run_bigcodebench(solutions, Path(tmp) / f"run{n}")
            for i in idxs:
                results[i] = {k: v for k, v in bcb[answers[i]["native_id"]].items()
                              if k != "reference_failed"}

    bandit = run_bandit(answers)
    rows = []
    for i, a in enumerate(answers):
        res = results.get(i, {"status": "no_code", "error": ""})
        rows.append({"run": a["run"], "task_id": a["task_id"], "dataset": a["dataset"],
                     "passed": res["status"] == "pass", **res, **bandit[i]})
    write_jsonl(OUT / f"test_results_{path.stem}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["codex", "reference", "answers"])
    parser.add_argument("--input", type=Path, help="answers file (answers mode)")
    parser.add_argument("--datasets", nargs="+", default=["ds1000", "odex", "bigcodebench"],
                        choices=["ds1000", "odex", "bigcodebench"], help="reference mode only")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "codex":
        mode_codex(args.workers)
    elif args.mode == "reference":
        mode_reference(args.workers, args.datasets)
    else:
        if not args.input:
            parser.error("answers mode needs --input")
        mode_answers(args.input, args.workers)


if __name__ == "__main__":
    main()
