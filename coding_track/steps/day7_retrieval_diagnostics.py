"""Day 7 (extra): explain the NDCG@10 number from day7_check_index.

Runs on: Colab, Drive mounted. Same session as Day 7 is best (the local
ChromaDB is reused; otherwise it is unzipped from Drive). L4 or High-RAM CPU.

CodeRAG-Bench labels the correct docs for DS-1000/ODEX only among the
library-documentation pages, which are ~2% of our index. This script measures,
for the labelled tasks:
  1. Source mix of the full-index top 10: what share of retrieved chunks comes
     from library docs, Stack Overflow, tutorials, ... and how many are posts
     flagged on Day 4 as near-copies of a task.
  2. NDCG@10 when searching the library-documentation chunks only (exact
     search over their saved embeddings). If this is clearly above the
     full-index number, the scoring works and the labelled docs are simply
     out-ranked by other sources.

Usage:  python -m coding_track.steps.day7_retrieval_diagnostics
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

import numpy as np

from coding_track import config
from coding_track.common.io_utils import read_json, read_jsonl, require, save_stats
from coding_track.common.metrics import ndcg_at_k
from coding_track.common.retrieval import embed, load_retriever, open_collection, search
from coding_track.steps.day7_check_index import ensure_local_index, labelled_docs, load_chunk_table
from coding_track.steps.week1_notes import write_notes

LIBDOC = "library-documentation"


def libdoc_embeddings(chunks) -> tuple[np.ndarray, list[str]]:
    """Saved vectors and orig_ids of every library-documentation chunk."""
    sources = chunks.column("source").to_pylist()
    orig_ids = chunks.column("orig_id").to_pylist()
    chunk_ids = chunks.column("chunk_id").to_pylist()
    rows = [i for i, s in enumerate(sources) if s == LIBDOC]
    by_shard: dict[int, list[int]] = defaultdict(list)
    for i in rows:
        by_shard[i // config.EMBED_SHARD_SIZE].append(i)

    vectors, ids = [], []
    for shard, members in sorted(by_shard.items()):
        with np.load(config.EMBED_DIR / f"shard_{shard:05d}.npz", allow_pickle=False) as data:
            offsets = [i - shard * config.EMBED_SHARD_SIZE for i in members]
            if [str(x) for x in data["ids"][offsets]] != [chunk_ids[i] for i in members]:
                raise SystemExit(f"Shard {shard} ids do not match chunks.parquet.")
            vectors.append(data["embeddings"][offsets])
        ids.extend(orig_ids[i] for i in members)
    return np.concatenate(vectors), ids


def flagged_urls() -> set[str]:
    with open(require(config.SO_OVERLAP_FILE, "coding_track.steps.day4_filter_stackoverflow"),
              encoding="utf-8") as f:
        return {row["so_url"] for row in csv.DictReader(f)}


def main() -> None:
    ensure_local_index()
    collection = open_collection()
    chunks = load_chunk_table()
    tasks = {t["task_id"]: t for t in read_jsonl(config.TASKS_FILE)}
    split = read_json(config.SPLITS_FILE)["split"]
    labels = labelled_docs()
    task_ids = sorted(labels)

    model = load_retriever()
    queries = embed(model, [tasks[t]["query_text"] for t in task_ids], show_progress=True)

    # 1. What the full index returns
    flagged = flagged_urls()
    mix: dict[str, Counter] = defaultdict(Counter)
    near_copies: Counter = Counter()
    for task_id, hits in zip(task_ids, search(collection, queries), strict=True):
        dataset = tasks[task_id]["dataset"]
        mix[dataset].update(h["metadata"]["source"] for h in hits)
        near_copies[dataset] += sum(h["metadata"]["url"] in flagged for h in hits)

    # 2. NDCG@10 over library-documentation chunks only (exact search)
    doc_vectors, doc_ids = libdoc_embeddings(chunks)
    chunks_per_doc = Counter(doc_ids)
    top = np.argsort(-(queries @ doc_vectors.T), axis=1)[:, :config.TOP_K]
    scores: dict[str, list[float]] = defaultdict(list)
    for task_id, row in zip(task_ids, top, strict=True):
        relevant = labels[task_id]
        rel = [int(doc_ids[j] in relevant) for j in row]
        score = ndcg_at_k(rel, sum(chunks_per_doc[d] for d in relevant), config.TOP_K)
        dataset = tasks[task_id]["dataset"]
        scores[dataset].append(score)
        if split[task_id] == "test":
            scores[f"{dataset}_test"].append(score)

    results = {
        "library_doc_chunks": len(doc_ids),
        "top10_source_share": {
            ds: {src: round(n / sum(c.values()), 3) for src, n in c.most_common()}
            for ds, c in mix.items()},
        "top10_flagged_near_copy_posts_per_query": {
            ds: round(near_copies[ds] / (sum(mix[ds].values()) / config.TOP_K), 2) for ds in mix},
        "ndcg@10_library_docs_only": {
            name: {"ndcg@10": round(sum(v) / len(v), 4), "tasks": len(v)}
            for name, v in scores.items()},
    }
    for key, value in results.items():
        print(f"{key}: {value}")
    save_stats("day7_retrieval_diagnostics", results)
    write_notes()


if __name__ == "__main__":
    main()
