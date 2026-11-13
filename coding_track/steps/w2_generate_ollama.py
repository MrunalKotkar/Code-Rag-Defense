"""Week 2, Days 4-6: run one local model through Ollama on the frozen prompts.

Runs on: main Colab notebook, L4 GPU, Drive mounted, after the setup cell
started `ollama serve` (OLLAMA_CONTEXT_LENGTH=16384) and pulled the models.

* One model at a time: any other model loaded in Ollama is unloaded first, and
  this model is unloaded at the end, so Ollama never swaps models mid-run.
* Settings for every call: temperature 0, seed 100, num_ctx 16384,
  num_predict GEN_MAX_TOKENS (answers that hit it are marked done_reason=length).
* One line per question is appended locally and copied to Drive every
  COPY_TO_DRIVE_EVERY questions; finished questions are skipped on restart.
* The model's Ollama digest (`ollama list` ID) and Ollama version are recorded.

Usage:
  python -m coding_track.steps.w2_generate_ollama --model llama3.1:8b --setting clean_rag --pilot
  python -m coding_track.steps.w2_generate_ollama --model llama3.1:8b --setting no_retrieval
  python -m coding_track.steps.w2_generate_ollama --model llama3.1:8b --setting no_retrieval --question-set anchor
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

from coding_track import config
from coding_track.common.generation import append, load_done, make_record
from coding_track.common.io_utils import atomic_copy, read_jsonl, require, save_stats
from coding_track.common.question_sets import fake_names, load_targets, pilot_ids, question_ids


def ollama_version() -> str:
    with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/version", timeout=10) as r:
        return json.load(r)["version"]


def isolate_model(client, model: str) -> str:
    """Unload every other model; return this model's digest."""
    digests = {m.model: m.digest for m in client.list().models}
    if model not in digests:
        raise SystemExit(f"{model} is not pulled. Run: ollama pull {model}")
    for running in client.ps().models:
        if running.model != model:
            print(f"unloading {running.model}")
            client.generate(model=running.model, prompt="", keep_alive=0)
    return digests[model]


def summarize(records: list[dict], n_total: int) -> dict:
    secs = [r["seconds"] for r in records]
    counts = [r["prompt_eval_count"] for r in records]
    ratios = sorted(r["prompt_chars"] / max(1, r["prompt_eval_count"]) for r in records)
    median_ratio = ratios[len(ratios) // 2]
    return {
        "questions": len(records),
        "max_prompt_eval_count": max(counts),
        "prompts_near_context_limit": sum(c >= config.OLLAMA_NUM_CTX - config.GEN_MAX_TOKENS for c in counts),
        # A cut prompt would read far fewer tokens than its length implies.
        "prompts_with_unusual_chars_per_token": sum(x > 2 * median_ratio for x in ratios),
        "code_found": sum(not r["abstained"] for r in records),
        "hit_length_limit": sum(r["done_reason"] == "length" for r in records),
        "fake_name_answers": sum(bool(r["fake_import"] or r["fake_install"]) for r in records),
        "mean_seconds": round(statistics.mean(secs), 2),
        "estimated_hours_for_question_set": round(statistics.mean(secs) * n_total / 3600, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Ollama model on the frozen prompts.")
    parser.add_argument("--model", required=True, choices=config.LOCAL_MODELS)
    parser.add_argument("--setting", required=True, choices=config.SETTINGS)
    parser.add_argument("--question-set", default="scored", choices=config.QUESTION_SETS)
    parser.add_argument("--pilot", action="store_true", help=f"only {config.PILOT_N} questions")
    args = parser.parse_args()
    if args.question_set == "anchor" and args.setting != "no_retrieval":
        parser.error("the anchor set is only used with --setting no_retrieval")

    import ollama

    prompts_file = config.PROMPTS_DIR / f"prompts_{args.setting}.jsonl"
    prompts = {p["task_id"]: p for p in read_jsonl(require(prompts_file, "coding_track.steps.w2_build_prompts"))}
    ids = [t for t in question_ids(args.question_set) if t in prompts]
    all_ids = ids
    if args.pilot:
        ids = pilot_ids(ids)
    fakes = fake_names(load_targets())

    client = ollama.Client(host=config.OLLAMA_HOST)
    digest = isolate_model(client, args.model)
    version = ollama_version()
    local, drive, done = load_done(args.model, args.setting)
    todo = [t for t in ids if t not in done]
    print(f"{args.model} / {args.setting} / {args.question_set}{' / pilot' if args.pilot else ''}: "
          f"{len(ids)} questions, {len(ids) - len(todo)} already done, {len(todo)} to go")

    options = {"temperature": config.OLLAMA_TEMPERATURE, "seed": config.OLLAMA_SEED,
               "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.GEN_MAX_TOKENS}
    new: list[dict] = []
    for n, task_id in enumerate(todo, 1):
        prompt = prompts[task_id]
        start = time.time()
        resp = client.chat(model=args.model, messages=[{"role": "user", "content": prompt["prompt"]}],
                           options=options)
        seconds = time.time() - start
        record = make_record(
            prompt, model=args.model, model_id=digest, setting=args.setting,
            raw_answer=resp.message.content, fakes=fakes,
            prompt_eval_count=resp.prompt_eval_count, eval_count=resp.eval_count,
            done_reason=resp.done_reason, seconds=round(seconds, 2), num_ctx=config.OLLAMA_NUM_CTX,
            ollama_version=version)
        append(local, record)
        new.append(record)
        if n % config.COPY_TO_DRIVE_EVERY == 0:
            atomic_copy(local, drive)
            print(f"  {n}/{len(todo)} done; copied to Drive")
    if new:
        atomic_copy(local, drive)

    loaded = [m for m in client.ps().models if m.model == args.model]
    context_length = getattr(loaded[0], "context_length", None) if loaded else None
    client.generate(model=args.model, prompt="", keep_alive=0)          # free the GPU

    finished = [r for r in read_jsonl(local) if r["task_id"] in set(ids)]
    summary = summarize(finished, len(all_ids)) if finished else {}
    summary.update({"model": args.model, "model_digest": digest, "ollama_version": version,
                    "ollama_ps_context_length": context_length, "setting": args.setting,
                    "question_set": args.question_set, "pilot": args.pilot})
    for k, v in summary.items():
        print(f"{k}: {v}")
    tag = f"w2_generate__{args.model.replace(':', '-')}__{args.setting}__{args.question_set}"
    save_stats(tag + ("__pilot" if args.pilot else ""), summary)


if __name__ == "__main__":
    main()
