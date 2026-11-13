"""Week 2, Day 4 (part 1): retrieve once for every scored question and freeze the result.

Runs on: main Colab notebook, Drive mounted, L4 GPU (query embedding; CPU also works).
Needs the Week 1 index (unzipped from Drive automatically if not on /content).

For each scored question the top RETRIEVAL_POOL (40) chunks are fetched from the
clean ChromaDB index with mGTE, encoding the question exactly as in the Week 1
check (same query text, same model and settings). Every model later reads the
top 10 from this file, so all models see exactly the same documents; Week 4's
defense uses the full pool of 40.

Outputs (Drive, week2/):
  retrieval_top40.jsonl    {task_id, chunk_ids[40], scores[40]} per question
  retrieval_chunks.jsonl   text + metadata of every chunk listed above, so
                           later steps never need the 19 GB index again

Usage:  python -m coding_track.steps.w2_freeze_retrieval
"""

from __future__ import annotations

import time
from collections import Counter

from coding_track import config
from coding_track.common.io_utils import atomic_write_jsonl, save_stats
from coding_track.common.question_sets import load_tasks, question_ids
from coding_track.common.retrieval import embed, load_retriever, open_collection, search
from coding_track.steps.day7_check_index import ensure_local_index


def main() -> None:
    tasks = load_tasks()
    ids = question_ids("scored", tasks)
    print(f"scored questions: {len(ids)} {dict(Counter(tasks[t]['dataset'] for t in ids))}")

    ensure_local_index()
    collection = open_collection()
    model = load_retriever()
    start = time.time()
    vectors = embed(model, [tasks[t]["query_text"] for t in ids], show_progress=True)
    results = search(collection, vectors, k=config.RETRIEVAL_POOL)
    seconds = time.time() - start

    rankings, chunks = [], {}
    for task_id, hits in zip(ids, results, strict=True):
        if len(hits) != config.RETRIEVAL_POOL:
            raise SystemExit(f"{task_id}: got {len(hits)} hits, expected {config.RETRIEVAL_POOL}")
        rankings.append({"task_id": task_id, "chunk_ids": [h["chunk_id"] for h in hits],
                         "scores": [round(h["score"], 6) for h in hits]})
        for h in hits:
            chunks[h["chunk_id"]] = {"chunk_id": h["chunk_id"], "text": h["text"], **h["metadata"]}
    atomic_write_jsonl(config.RETRIEVAL_FILE, rankings)
    atomic_write_jsonl(config.RETRIEVAL_CHUNKS_FILE, chunks.values())

    top10_sources = Counter(chunks[c]["source"] for r in rankings for c in r["chunk_ids"][:10])
    print(f"froze top-{config.RETRIEVAL_POOL} for {len(rankings)} questions "
          f"({len(chunks):,} distinct chunks) in {seconds:.0f} s")
    print("top-10 sources:", dict(top10_sources.most_common()))
    save_stats("w2_freeze_retrieval", {
        "questions": len(rankings), "pool": config.RETRIEVAL_POOL, "distinct_chunks": len(chunks),
        "seconds": round(seconds, 1), "top10_sources": dict(top10_sources.most_common()),
        "retriever": config.RETRIEVER_MODEL, "retriever_revision": config.RETRIEVER_REVISION,
    })


if __name__ == "__main__":
    main()
