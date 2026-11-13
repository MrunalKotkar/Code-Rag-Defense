"""Week 2, Day 3/4: build and freeze every prompt, once, for both settings.

Runs on: main Colab notebook, Drive mounted. CPU is enough.
Needs: Week 1 tasks/splits/targets, the raw question sets, and (for clean RAG)
w2_freeze_retrieval's files.

  no_retrieval  ImportSnare template with an empty context, for the scored set
                plus the anchor set (all DS-1000 + ODEX).
  clean_rag     the same template with the frozen top-10 chunks, scored set only.

Every model (Ollama and GPT-5.4 mini) reads these files, so all models get the
identical prompt for a question and setting.

Outputs (Drive, week2/prompts/): prompts_no_retrieval.jsonl, prompts_clean_rag.jsonl

Usage:  python -m coding_track.steps.w2_build_prompts
"""

from __future__ import annotations

from collections import Counter

from coding_track import config
from coding_track.common.hf_data import load_raw
from coding_track.common.io_utils import atomic_write_jsonl, read_jsonl, require, save_stats
from coding_track.common.prompt import build_prompt, odex_question
from coding_track.common.question_ids import odex_task_ids
from coding_track.common.question_sets import load_tasks, question_ids

FREEZE = "coding_track.steps.w2_freeze_retrieval"


def question_texts(tasks: dict[str, dict]) -> dict[str, str]:
    """[question] for every task, per question set (see common/prompt.py)."""
    texts = {}
    for row in load_raw("ds1000"):
        texts[f"ds1000:{row['metadata']['problem_id']}"] = row["prompt"]
    odex = load_raw("odex")
    for native_id, row in zip(odex_task_ids(odex["task_id"]), odex, strict=True):
        texts[f"odex:{native_id}"] = odex_question(row["prompt"], row["intent"])
    for row in load_raw("bigcodebench"):
        texts[f"bigcodebench:{row['task_id']}"] = row["instruct_prompt"]
    missing = set(tasks) - set(texts)
    if missing:
        raise SystemExit(f"{len(missing)} tasks have no question text, e.g. {sorted(missing)[:3]}")
    return texts


def prompt_record(task: dict, setting: str, sets: list[str], prompt: str, retrieved: list[str]) -> dict:
    return {"task_id": task["task_id"], "dataset": task["dataset"], "native_id": task["native_id"],
            "packages": task["packages"], "question_sets": sets, "setting": setting,
            "retrieved_ids": retrieved, "prompt": prompt}


def main() -> None:
    tasks = load_tasks()
    texts = question_texts(tasks)
    scored = question_ids("scored", tasks)
    anchor = question_ids("anchor", tasks)
    sets_of = {tid: [s for s, ids in (("scored", scored), ("anchor", anchor)) if tid in ids]
               for tid in set(scored) | set(anchor)}

    rankings = {r["task_id"]: r["chunk_ids"][:config.TOP_K]
                for r in read_jsonl(require(config.RETRIEVAL_FILE, FREEZE))}
    chunk_text = {c["chunk_id"]: c["text"] for c in read_jsonl(require(config.RETRIEVAL_CHUNKS_FILE, FREEZE))}
    if set(scored) - set(rankings):
        raise SystemExit("Some scored questions have no frozen retrieval; re-run w2_freeze_retrieval.")

    no_ret = [prompt_record(tasks[t], "no_retrieval", sets_of[t], build_prompt(texts[t], []), [])
              for t in sorted(sets_of)]
    rag = [prompt_record(tasks[t], "clean_rag", sets_of[t],
                         build_prompt(texts[t], [chunk_text[c] for c in rankings[t]]), rankings[t])
           for t in scored]
    config.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(config.PROMPTS_DIR / "prompts_no_retrieval.jsonl", no_ret)
    atomic_write_jsonl(config.PROMPTS_DIR / "prompts_clean_rag.jsonl", rag)

    def chars(rows):
        lengths = sorted(len(r["prompt"]) for r in rows)
        return {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]}

    stats = {
        "no_retrieval": {"prompts": len(no_ret), "chars": chars(no_ret),
                         "by_dataset": dict(Counter(r["dataset"] for r in no_ret))},
        "clean_rag": {"prompts": len(rag), "chars": chars(rag),
                      "by_dataset": dict(Counter(r["dataset"] for r in rag))},
        "scored_questions": len(scored), "anchor_questions": len(anchor),
    }
    for k, v in stats.items():
        print(f"{k}: {v}")
    save_stats("w2_build_prompts", stats)


if __name__ == "__main__":
    main()
