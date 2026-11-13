"""Build and run the test programs for DS-1000, ODEX and BigCodeBench.

Runs on: the test runtime (no Drive, no keys). Problem data is downloaded from
Hugging Face at the same pinned revisions as the main pipeline.

  DS-1000       program = code_context + `code = <answer>` + test_execution(code)
                (+ test_string(code) when the problem has it), exactly as
                DS-1000's test_ds1000.py builds it. Passes only if all pass.
  ODEX          program = solution + test_start + all test cases + check(entry_point),
                as in ODEX's verify.py, with two documented choices:
                  * all test cases are run (ODEX's script samples one at random);
                  * a chat model usually writes the whole function, so if the
                    answer defines the entry-point function it is used as is;
                    otherwise it is treated as a completion (prompt + answer + suffix),
                    which is also how reference solutions are run.
  BigCodeBench  scored by the official evaluator (bigcodebench.evaluate, local
                execution, Instruct split, calibrated), restricted to our task ids.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coding_track.common.question_ids import odex_task_ids
from coding_track.testing import env_spec

EXEC_SCRIPT = Path(__file__).with_name("exec_program.py")
TEST_ENV_VARS = {"MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_jsonl(repo: str, revision: str, filename: str) -> list[dict]:
    url = env_spec.hf_file_url(repo, revision, filename)
    with urllib.request.urlopen(url, timeout=120) as resp:
        return [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]


def ds1000_problems() -> dict[str, dict]:
    return {str(p["metadata"]["problem_id"]): p for p in download_jsonl(*env_spec.DS1000_HF)}


def odex_problems() -> dict[str, dict]:
    rows = download_jsonl(*env_spec.ODEX_HF)
    return dict(zip(odex_task_ids(r["task_id"] for r in rows), rows, strict=True))


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------
def ds1000_program(problem: dict, solution: str) -> str:
    context = problem["code_context"]
    return (context + "\n" + f"code = {solution!r}\n" + "test_execution(code)\n"
            + ("test_string(code)\n" if "test_string(" in context else "\n"))


def defines_function(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"def {name}(" in code
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
               for n in ast.walk(tree))


def odex_program(problem: dict, solution: str, is_reference: bool = False) -> str:
    entry = problem["entry_point"]
    if not is_reference and defines_function(solution.replace("\t", "    "), entry):
        body = solution
    else:
        body = f"{problem['prompt']}{solution}{problem['suffix']}"
    check = "\n".join([problem["test_start"], "".join(problem["test"]), "", f"check({entry})"])
    return body.replace("\t", " " * 4) + "\n" + check


# ---------------------------------------------------------------------------
# Execution (one process per program)
# ---------------------------------------------------------------------------
def run_program(python: Path, program: str, timeout_s: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"program": program}, f)
        job = f.name
    try:
        proc = subprocess.run([str(python), str(EXEC_SCRIPT), job], capture_output=True,
                              text=True, timeout=timeout_s, env={**os.environ, **TEST_ENV_VARS})
        if proc.returncode == 0:
            return {"status": "pass", "error": ""}
        lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
        return {"status": "fail", "error": (lines[-1] if lines else f"exit {proc.returncode}")[:500]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout_s} s"}
    finally:
        os.unlink(job)


def run_many(jobs: list[tuple[str, str]], python: Path, timeout_s: int,
             workers: int | None = None) -> dict[str, dict]:
    """jobs: (key, program). Returns {key: result}."""
    workers = workers or os.cpu_count() or 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda job: run_program(python, job[1], timeout_s), jobs)
        out = {}
        for i, ((key, _), result) in enumerate(zip(jobs, results, strict=True), 1):
            out[key] = result
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} done")
        return out


# ---------------------------------------------------------------------------
# BigCodeBench via the official evaluator
# ---------------------------------------------------------------------------
def bigcodebench_cache_dir() -> Path:
    return Path.home() / ".cache" / "bigcodebench"


def run_bigcodebench(solutions: dict[str, str], workdir: Path) -> dict[str, dict]:
    """solutions: {BigCodeBench task id: full solution code}. Official local evaluation."""
    workdir.mkdir(parents=True, exist_ok=True)
    samples = workdir / "samples.jsonl"
    with open(samples, "w", encoding="utf-8") as f:
        for task_id, code in solutions.items():
            f.write(json.dumps({"task_id": task_id, "solution": code}) + "\n")
    python = env_spec.env_python("bcb")
    cmd = [str(python), "-m", "bigcodebench.evaluate", "--execution", "local",
           "--split", env_spec.BCB_SPLIT, "--subset", env_spec.BCB_SUBSET,
           "--samples", str(samples), "--selective_evaluate", ",".join(solutions)]
    print("$", " ".join(cmd[:12]), "... (", len(solutions), "tasks )")
    subprocess.run(cmd, check=True, cwd=workdir, env={**os.environ, **TEST_ENV_VARS},
                   stdin=subprocess.DEVNULL)
    with open(workdir / "samples_eval_results.json", encoding="utf-8") as f:
        evals = json.load(f)["eval"]
    with open(workdir / "samples_pass_at_k.json", encoding="utf-8") as f:
        failed_gt = set(json.load(f).get("failed_tasks", []))
    out = {}
    for task_id in solutions:
        status = evals[task_id][0]["status"] if task_id in evals else "missing"
        out[task_id] = {"status": "pass" if status == "pass" else status,
                        "error": "" if status == "pass" else json.dumps(evals.get(task_id, [{}])[0].get("details", ""))[:500],
                        "reference_failed": task_id in failed_gt}
    return out
