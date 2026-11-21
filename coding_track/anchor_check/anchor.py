"""Anchor check with CodeRAG-Bench's own DS-1000 prompt (separate from the Week 2 pipeline).

Why: the Week 2 anchor (Llama 3.1 8B, no retrieval, ImportSnare chat prompt) scored DS-1000
15.7% against CodeRAG-Bench Table 5's 28.9%. CodeRAG-Bench does not use a chat prompt for
DS-1000: its task (generation/eval/tasks/ds1000.py, "ds1000-all-completion", checked
2026-11-20) sends DS-1000's own `prompt` field unchanged, lets the model continue it, stops
at "</code>" or "# SOLUTION END", and keeps the text after "BEGIN SOLUTION\\n<code>" /
"# SOLUTION START" / "\\n<code>" (and the contents of a ```python fence if there is one).
This script does exactly that and scores the answers with the Week 2 test harness, unchanged.

Nothing here reads or writes Week 2 files. Everything goes to
<DRIVE_ROOT>/anchor_check/ (and a local copy in <LOCAL_ROOT>/anchor_check/).

Subcommands (run in this order, all inside anchor_check_colab.ipynb):
  generate  Ollama, raw completion mode (no chat template), one answer per DS-1000 problem.
            Needs the GPU, Ollama running, Drive mounted.     --limit N for a short trial
  score     Run every answer in the DS-1000 test environment, print pass@1.
            Needs the ds1000 environment (testing.build_envs). Run with Drive UNMOUNTED:
            this executes model-written code.

Usage:  python -m coding_track.anchor_check.anchor generate --limit 20
        python -m coding_track.anchor_check.anchor generate
        python -m coding_track.anchor_check.anchor score
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from coding_track import config
from coding_track.testing import env_spec, harness

ANCHOR_DIR = config.DRIVE_ROOT / "anchor_check"            # persistent (Drive)
LOCAL_DIR = config.LOCAL_ROOT / "anchor_check"             # scratch (survives a Drive unmount)

# CodeRAG-Bench Table 5, Llama3-8B without retrieval (already verified in Week 2).
PUBLISHED_DS1000 = 28.9
# Earlier result of the same model with the ImportSnare chat prompt (Week 2 anchor run).
WEEK2_CHAT_PROMPT_RESULT = "15.7% (157/1000)"

STOP_WORDS = ["</code>", "# SOLUTION END"]                  # CodeRAG-Bench ds1000.py stop_words
START_MARKERS = ["BEGIN SOLUTION\n<code>", "# SOLUTION START", "\n<code>"]
OPTIONS = {"temperature": config.OLLAMA_TEMPERATURE, "seed": config.OLLAMA_SEED,
           "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.GEN_MAX_TOKENS,
           "stop": STOP_WORDS}
COPY_EVERY = 50


# ---------------------------------------------------------------------------
# CodeRAG-Bench's post-processing of one generation (DS-1000, Completion mode)
# ---------------------------------------------------------------------------
def extract_fenced(text: str, prefix: str) -> str:
    """First fenced block after `prefix`, as CodeRAG-Bench's extract_code_pieces returns it.

    Deviation: their helper starts reading 10 characters after the fence marker, which is right
    for "```python\\n" but skips 6 characters of code after a bare "```\\n"; here the start is
    len(prefix) + 1 (the marker and its newline) for both.
    """
    start = text.index(prefix) + len(prefix) + 1
    end = text.index("```", start) if "```" in text[start:] else len(text)
    return text[start:end].strip()


def postprocess(generation: str) -> str:
    for start in START_MARKERS:
        generation = generation.split(start, 1)[-1]
    for stop in STOP_WORDS:
        generation = generation.split(stop)[0]
    if "```python\n" in generation:
        generation = extract_fenced(generation, "```python")
    elif "```\n" in generation:
        generation = extract_fenced(generation, "```")
    return generation.strip()


# ---------------------------------------------------------------------------
# small I/O helpers (kept local so this folder depends only on config + the test harness)
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def names(model: str, limit: int | None) -> tuple[Path, Path]:
    stem = f"{model.replace(':', '-')}__ds1000_coderag_prompt" + ("__trial" if limit else "")
    return LOCAL_DIR / f"{stem}.jsonl", ANCHOR_DIR / f"{stem}.jsonl"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def cmd_generate(model: str, limit: int | None) -> None:
    import ollama

    problems = harness.ds1000_problems()                   # keyed by str(problem_id)
    ids = sorted(problems, key=int)[: limit or None]
    local, drive = names(model, limit)
    local.parent.mkdir(parents=True, exist_ok=True)
    if drive.exists() and (not local.exists() or drive.stat().st_size > local.stat().st_size):
        copy_atomic(drive, local)                          # resume after a dropped session
    done = {r["native_id"] for r in read_jsonl(local)}
    todo = [i for i in ids if i not in done]
    print(f"{model}: {len(ids)} DS-1000 problems, {len(ids) - len(todo)} already done, {len(todo)} to go")

    client = ollama.Client(host=config.OLLAMA_HOST)
    digest = {m.model: m.digest for m in client.list().models}.get(model)
    if digest is None:
        raise SystemExit(f"{model} is not pulled. Run: ollama pull {model}")
    for running in client.ps().models:
        if running.model != model:
            client.generate(model=running.model, prompt="", keep_alive=0)
    with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/version", timeout=10) as r:
        version = json.load(r)["version"]

    for n, pid in enumerate(todo, 1):
        problem = problems[pid]
        start = time.time()
        # raw=True: the prompt goes to the model as is, with no chat template.
        resp = client.generate(model=model, prompt=problem["prompt"], raw=True, options=OPTIONS)
        record = {
            "task_id": f"ds1000:{pid}", "dataset": "ds1000", "native_id": pid,
            "library": problem["metadata"]["library"], "model": model, "model_id": digest,
            "ollama_version": version, "raw_answer": resp.response, "code": postprocess(resp.response),
            "done_reason": resp.done_reason, "prompt_eval_count": resp.prompt_eval_count,
            "eval_count": resp.eval_count, "seconds": round(time.time() - start, 2),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(local, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if n % COPY_EVERY == 0:
            copy_atomic(local, drive)
            print(f"  {n}/{len(todo)} done; copied to Drive")
    if todo:
        copy_atomic(local, drive)
    client.generate(model=model, prompt="", keep_alive=0)

    rows = [r for r in read_jsonl(local) if r["native_id"] in set(ids)]
    print(f"\nanswers: {len(rows)}")
    print(f"empty code after post-processing: {sum(not r['code'] for r in rows)}")
    print(f"hit the {config.GEN_MAX_TOKENS}-token limit: {sum(r['done_reason'] == 'length' for r in rows)}")
    print(f"mean seconds per answer: {sum(r['seconds'] for r in rows) / max(1, len(rows)):.2f}")
    print(f"max prompt tokens: {max(r['prompt_eval_count'] for r in rows)}")
    print(f"Ollama {version}, model digest {digest}")
    if limit and rows:
        first = rows[0]
        print("\n--- first answer: raw ---\n" + first["raw_answer"][:600])
        print("--- first answer: code that will be tested ---\n" + first["code"][:600])


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def error_kind(error: str) -> str:
    return (error or "(none)").split(":")[0][:40]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def cmd_score(model: str, limit: int | None) -> None:
    local, drive = names(model, limit)
    path = local if local.exists() else drive
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"No answers at {local} or {drive}. Run `generate` first.")
    print(f"{len(rows)} answers from {path}")

    problems = harness.ds1000_problems()
    jobs = [(r["native_id"], harness.ds1000_program(problems[r["native_id"]], r["code"]))
            for r in rows if r["code"]]
    print(f"{len(jobs)} answers with code to run, {len(rows) - len(jobs)} with no code")
    results = harness.run_many(jobs, env_spec.env_python("ds1000"), env_spec.DS1000_TIMEOUT_S)

    out_rows = []
    for r in rows:
        res = results.get(r["native_id"], {"status": "no_code", "error": ""})
        out_rows.append({"task_id": r["task_id"], "library": r["library"],
                         "passed": res["status"] == "pass", **res})
    out_path = LOCAL_DIR / (path.stem + "__test_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    passed = sum(r["passed"] for r in out_rows)
    by_lib: dict[str, list[bool]] = defaultdict(list)
    for r in out_rows:
        by_lib[r["library"]].append(r["passed"])
    status = Counter(r["status"] for r in out_rows)
    errors = Counter(error_kind(r.get("error", "")) for r in out_rows if r["status"] == "fail")

    lines = [f"DS-1000 anchor, {model}, CodeRAG-Bench prompt (raw DS-1000 prompt, no chat template)",
             f"  pass@1: {pct(passed, len(out_rows))}",
             f"  published (CodeRAG-Bench Table 5, Llama3-8B, no retrieval): {PUBLISHED_DS1000}%",
             f"  earlier run, ImportSnare chat prompt: {WEEK2_CHAT_PROMPT_RESULT}",
             "  Reference solutions: 1000/1000 pass in this environment (Week 2), so nothing is excluded.",
             "  status: " + ", ".join(f"{k} {v}" for k, v in status.most_common()),
             "  by library: " + ", ".join(f"{k} {pct(sum(v), len(v))}" for k, v in sorted(by_lib.items())),
             "  failures by exception:"]
    lines += [f"    {n:4d}  {k}" for k, n in errors.most_common(8)]
    text = "\n".join(lines)
    print("\n" + text)
    (LOCAL_DIR / (path.stem + "__summary.txt")).write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["generate", "score"])
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--limit", type=int, help="only the first N problems (a trial; own file)")
    args = parser.parse_args()
    {"generate": cmd_generate, "score": cmd_score}[args.command](args.model, args.limit)


if __name__ == "__main__":
    main()
