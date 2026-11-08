"""Day 1: prove that num_ctx=16384 lets Ollama read a ~10,000-token prompt in full.

Runs on: Colab, any runtime (High-RAM CPU is fine), after the setup cell has
started `ollama serve` with OLLAMA_CONTEXT_LENGTH=16384 and pulled the model.

Ollama's default context is 4k tokens on GPUs under 24 GiB and it cuts longer
prompts with no error. Two checks:
  1. prompt_eval_count (tokens the model actually read) is close to the full
     prompt length, well above 4096 and below num_ctx;
  2. the model can recall a word placed on the very first line, which is the
     part that would be cut off if the prompt were truncated.

Usage:  python -m coding_track.steps.day1_ollama_check [--model llama3.2:3b]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from coding_track import config
from coding_track.common.io_utils import save_stats

OLLAMA_HOST = "http://localhost:11434"
SECRET_WORD = "pelican"
# One filler line is about 20 Llama-3 tokens; the real count is reported below.
FILLER = "Line {i}: this sentence is filler text for the context-length test and can be ignored."
TOKENS_PER_LINE_ESTIMATE = 20


def build_prompt(target_tokens: int) -> str:
    n_lines = target_tokens // TOKENS_PER_LINE_ESTIMATE
    lines = [f"Line 0: remember this secret word: {SECRET_WORD}."]
    lines += [FILLER.format(i=i) for i in range(1, n_lines)]
    lines.append("Question: what is the secret word given on line 0? Reply with that one word only.")
    return "\n".join(lines)


def ollama_version() -> str:
    with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=10) as r:
        return json.load(r)["version"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=config.OLLAMA_CHECK_MODEL)
    args = parser.parse_args()

    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    digests = {m.model: m.digest for m in client.list().models}
    if args.model not in digests:
        raise SystemExit(f"Model {args.model} is not pulled. Run: ollama pull {args.model}")

    prompt = build_prompt(config.OLLAMA_CHECK_PROMPT_TOKENS)
    options = {"temperature": config.OLLAMA_TEMPERATURE, "seed": config.OLLAMA_SEED,
               "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 8}
    print(f"Sending a ~{config.OLLAMA_CHECK_PROMPT_TOKENS:,}-token prompt to {args.model} "
          f"with num_ctx={config.OLLAMA_NUM_CTX} ...")
    start = time.time()
    resp = client.chat(model=args.model, messages=[{"role": "user", "content": prompt}],
                       options=options)
    seconds = time.time() - start

    read_tokens = resp["prompt_eval_count"]
    answer = resp["message"]["content"].strip()
    recalled = SECRET_WORD in answer.lower()
    full_read = 4096 < read_tokens < config.OLLAMA_NUM_CTX
    passed = full_read and recalled

    print(f"prompt_eval_count = {read_tokens:,} tokens   (must be > 4,096 and < {config.OLLAMA_NUM_CTX:,})")
    print(f"model answer      = {answer!r}   (expected '{SECRET_WORD}')")
    print(f"took {seconds:.0f} s")
    print("PASS: the full prompt was read." if passed else
          "FAIL: the prompt was cut or not understood. Check OLLAMA_CONTEXT_LENGTH and num_ctx.")

    save_stats("day1_ollama_check", {
        "model": args.model, "model_digest": digests[args.model],
        "ollama_version": ollama_version(), "num_ctx": config.OLLAMA_NUM_CTX,
        "prompt_chars": len(prompt), "prompt_eval_count": read_tokens,
        "answer": answer, "secret_recalled": recalled, "seconds": round(seconds, 1),
        "passed": passed,
    })
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
