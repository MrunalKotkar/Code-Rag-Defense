"""Week 2, Days 5-6: GPT-5.4 mini on the same frozen prompts, through the OpenAI Batch API.

Runs on: main Colab notebook, Drive mounted (no GPU). The API key comes from
Colab Secrets via the OPENAI_API_KEY environment variable; it is never written
to a file. Never run this in the test notebook.

Subcommands:
  probe     One normal (non-batch) request with the first scored prompt, to find
            out which settings the model accepts. It first sends temperature 0,
            seed 100 and reasoning_effort "none" (the model's default and lowest
            setting); every parameter the API rejects is dropped and recorded.
            The accepted parameters are saved to openai_batch/params.json.
  submit    Build one JSONL of requests from prompts_<setting>.jsonl with the
            probed parameters, upload it and start a batch (24 h window).
  status    Show the batch's status and request counts.
  collect   Download the results and save them in the same format as the
            Ollama runs (generations/gpt-5.4-mini__<setting>.jsonl).

Usage:  python -m coding_track.steps.w2_openai_batch probe
        python -m coding_track.steps.w2_openai_batch submit --setting clean_rag
        python -m coding_track.steps.w2_openai_batch status --setting clean_rag
        python -m coding_track.steps.w2_openai_batch collect --setting clean_rag
"""

from __future__ import annotations

import argparse
import json

from coding_track import config
from coding_track.common.generation import generation_paths, make_record
from coding_track.common.io_utils import (
    atomic_copy, atomic_write_json, atomic_write_jsonl, now_iso, read_json, read_jsonl, require,
    save_stats)
from coding_track.common.question_sets import fake_names, load_targets, question_ids

PARAMS_FILE = config.OPENAI_BATCH_DIR / "params.json"
WANTED = {"temperature": config.OLLAMA_TEMPERATURE, "seed": config.OLLAMA_SEED,
          "reasoning_effort": config.OPENAI_REASONING_EFFORT}


def client():
    from openai import OpenAI
    return OpenAI()            # reads OPENAI_API_KEY from the environment


def load_prompts(setting: str) -> list[dict]:
    path = require(config.PROMPTS_DIR / f"prompts_{setting}.jsonl", "coding_track.steps.w2_build_prompts")
    scored = set(question_ids("scored"))
    return [p for p in read_jsonl(path) if p["task_id"] in scored]


def request_body(prompt: str, params: dict) -> dict:
    return {"model": config.OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": config.GEN_MAX_TOKENS, **params}


def cmd_probe() -> None:
    import openai

    prompt = load_prompts("clean_rag")[0]["prompt"]
    params = dict(WANTED)
    rejected = {}
    while True:
        try:
            resp = client().chat.completions.create(**request_body(prompt, params))
            break
        except openai.BadRequestError as e:
            culprit = next((p for p in params if p in str(e)), None)
            if culprit is None:
                raise
            rejected[culprit] = str(e)[:300]
            print(f"rejected {culprit}: {rejected[culprit]}")
            params.pop(culprit)
    record = {"model": config.OPENAI_MODEL, "returned_model": resp.model, "accepted": params,
              "rejected": rejected, "probe_finish_reason": resp.choices[0].finish_reason,
              "probe_usage": resp.usage.model_dump() if resp.usage else None, "probed_at": now_iso()}
    atomic_write_json(PARAMS_FILE, record)
    print(json.dumps(record, indent=2))


def batch_info_path(setting: str):
    return config.OPENAI_BATCH_DIR / f"batch_{setting}.json"


def cmd_submit(setting: str) -> None:
    probe = read_json(require(PARAMS_FILE, "w2_openai_batch probe"))
    if batch_info_path(setting).exists():
        raise SystemExit(f"A batch for {setting} was already submitted: {batch_info_path(setting)}")
    prompts = load_prompts(setting)
    requests_path = config.OPENAI_BATCH_DIR / f"requests_{setting}.jsonl"
    atomic_write_jsonl(requests_path, (
        {"custom_id": p["task_id"], "method": "POST", "url": config.OPENAI_ENDPOINT,
         "body": request_body(p["prompt"], probe["accepted"])} for p in prompts))
    c = client()
    with open(requests_path, "rb") as f:
        uploaded = c.files.create(file=f, purpose="batch")
    batch = c.batches.create(input_file_id=uploaded.id, endpoint=config.OPENAI_ENDPOINT,
                             completion_window="24h")
    info = {"setting": setting, "batch_id": batch.id, "input_file_id": uploaded.id,
            "requests": len(prompts), "params": probe["accepted"], "submitted_at": now_iso()}
    atomic_write_json(batch_info_path(setting), info)
    print(json.dumps(info, indent=2))


def cmd_status(setting: str) -> None:
    info = read_json(require(batch_info_path(setting), "w2_openai_batch submit"))
    batch = client().batches.retrieve(info["batch_id"])
    print(f"{batch.id}: {batch.status}; counts {batch.request_counts}")


def cmd_collect(setting: str) -> None:
    info = read_json(require(batch_info_path(setting), "w2_openai_batch submit"))
    c = client()
    batch = c.batches.retrieve(info["batch_id"])
    if batch.status != "completed":
        raise SystemExit(f"Batch is {batch.status}; collect when it is completed.")
    lines = c.files.content(batch.output_file_id).text.splitlines()
    errors = c.files.content(batch.error_file_id).text.splitlines() if batch.error_file_id else []
    prompts = {p["task_id"]: p for p in load_prompts(setting)}
    fakes = fake_names(load_targets())
    records, failed = [], []
    for line in lines:
        item = json.loads(line)
        resp = item.get("response") or {}
        if resp.get("status_code") != 200:
            failed.append({"task_id": item["custom_id"], "error": item.get("error") or resp})
            continue
        body = resp["body"]
        choice = body["choices"][0]
        records.append(make_record(
            prompts[item["custom_id"]], model=config.OPENAI_MODEL, model_id=body["model"],
            setting=setting, raw_answer=choice["message"]["content"] or "", fakes=fakes,
            prompt_eval_count=body["usage"]["prompt_tokens"],
            eval_count=body["usage"]["completion_tokens"], done_reason=choice["finish_reason"],
            seconds=None, params=info["params"], batch_id=batch.id))
    failed += [json.loads(e) for e in errors]
    local, drive = generation_paths(config.OPENAI_MODEL, setting)
    atomic_write_jsonl(local, records)
    atomic_copy(local, drive)
    print(f"collected {len(records)} answers, {len(failed)} failed requests -> {drive}")
    save_stats(f"w2_generate__{config.OPENAI_MODEL}__{setting}__scored", {
        "model": config.OPENAI_MODEL, "returned_models": sorted({r["model_id"] for r in records}),
        "setting": setting, "answers": len(records), "failed_requests": len(failed),
        "failures": failed[:20], "params": info["params"],
        "hit_length_limit": sum(r["done_reason"] == "length" for r in records),
        "code_found": sum(not r["abstained"] for r in records),
        "prompt_tokens": sum(r["prompt_eval_count"] for r in records),
        "completion_tokens": sum(r["eval_count"] for r in records)})


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-5.4 mini through the OpenAI Batch API.")
    parser.add_argument("command", choices=["probe", "submit", "status", "collect"])
    parser.add_argument("--setting", choices=config.SETTINGS)
    args = parser.parse_args()
    if args.command != "probe" and not args.setting:
        parser.error(f"{args.command} needs --setting")
    config.OPENAI_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    {"probe": lambda: cmd_probe(), "submit": lambda: cmd_submit(args.setting),
     "status": lambda: cmd_status(args.setting), "collect": lambda: cmd_collect(args.setting)}[args.command]()


if __name__ == "__main__":
    main()
