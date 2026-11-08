"""Day 7: check the index - count, 20 eyeballed questions, NDCG@10 - and write the notes.

Runs on: Colab, Drive mounted. Best right after Day 6 on the same L4 session
(query embedding is quick on GPU); a CPU runtime also works, just slower.
If the local ChromaDB folder is gone (new session), it is unzipped from Drive.

1. Count: chunks in ChromaDB must equal chunks in chunks.parquet, and every
   manifest document must have chunks. On a mismatch the script lists which
   embedding shards have ids missing from ChromaDB and stops.
2. Eyeball: 20 random test questions (seed 42), each with its top 10 results
   and their library -> derived/eyeball_top10.txt.
3. NDCG@10 on DS-1000 and ODEX with CodeRAG-Bench's labelled correct docs. A
   chunk counts as correct when its parent library-documentation doc is
   labelled for that task (see common/metrics.py for the exact formula).
4. Writes notes/week1_numbers.md from all the stats files.

Usage:  python -m coding_track.steps.day7_check_index
"""

from __future__ import annotations

import math
import random
import zipfile
from collections import Counter, defaultdict

from coding_track import config
from coding_track.common.hf_data import load_raw
from coding_track.common.io_utils import (
    atomic_copy, atomic_write_text, read_json, read_jsonl, require, save_stats)
from coding_track.common.metrics import ndcg_at_k
from coding_track.common.question_ids import odex_task_ids
from coding_track.common.retrieval import embed, load_retriever, open_collection, search
from coding_track.steps.week1_notes import write_notes

PREVIEW_CHARS = 160


def ensure_local_index() -> None:
    if (config.LOCAL_CHROMA_DIR / "chroma.sqlite3").exists():
        return
    zip_path = require(config.CHROMA_ZIP_FILE, "coding_track.steps.day6_build_chroma")
    print(f"unzipping {zip_path} -> {config.LOCAL_CHROMA_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.LOCAL_CHROMA_DIR)


def load_chunk_table():
    import pyarrow.parquet as pq

    src = require(config.CHUNKS_FILE, "coding_track.steps.day5_chunk")
    local = config.LOCAL_TMP_DIR / src.name
    if not local.exists() or local.stat().st_size != src.stat().st_size:
        atomic_copy(src, local)
    return pq.read_table(local, columns=["chunk_id", "doc_id", "orig_id", "source"])


# ---------------------------------------------------------------------------
# 1. Count
# ---------------------------------------------------------------------------
def missing_by_shard(collection, chunk_ids: list[str]) -> dict[int, int]:
    missing = {}
    step = 1000
    for shard in range(math.ceil(len(chunk_ids) / config.EMBED_SHARD_SIZE)):
        ids = chunk_ids[shard * config.EMBED_SHARD_SIZE:(shard + 1) * config.EMBED_SHARD_SIZE]
        found = sum(len(collection.get(ids=ids[s:s + step], include=[])["ids"])
                    for s in range(0, len(ids), step))
        if found != len(ids):
            missing[shard] = len(ids) - found
    return missing


def check_counts(collection, chunks) -> dict:
    n_chunks = chunks.num_rows
    count = collection.count()
    manifest_ids = {rec["doc_id"] for rec in read_jsonl(config.MANIFEST_FILE)}
    chunk_doc_ids = set(chunks.column("doc_id").to_pylist())
    result = {"chroma_count": count, "chunks_in_parquet": n_chunks,
              "manifest_docs": len(manifest_ids),
              "manifest_docs_without_chunks": len(manifest_ids - chunk_doc_ids),
              "chunk_docs_not_in_manifest": len(chunk_doc_ids - manifest_ids)}
    print(f"ChromaDB {count:,} | chunks {n_chunks:,} | manifest docs {len(manifest_ids):,}")
    if count != n_chunks:
        missing = missing_by_shard(collection, chunks.column("chunk_id").to_pylist())
        print(f"Shards with ids missing from ChromaDB: {missing}")
        result["missing_by_shard"] = missing
    result["passed"] = (count == n_chunks and not result["manifest_docs_without_chunks"]
                        and not result["chunk_docs_not_in_manifest"])
    return result


# ---------------------------------------------------------------------------
# 2. Eyeball
# ---------------------------------------------------------------------------
def eyeball(model, collection, tasks: list[dict], split: dict[str, str]) -> dict:
    test_ids = sorted(t for t, s in split.items() if s == "test")
    chosen = random.Random(config.SEED).sample(test_ids, config.EYEBALL_N_QUESTIONS)
    by_id = {t["task_id"]: t for t in tasks}
    picked = [by_id[t] for t in chosen]
    results = search(collection, embed(model, [t["query_text"] for t in picked]))

    lines, matches = [], []
    for task, hits in zip(picked, results, strict=True):
        wanted = set(task["packages"])
        n_match = sum(bool(wanted & set(h["metadata"]["library"].split(","))) for h in hits)
        matches.append(n_match)
        lines.append(f"=== {task['task_id']}  libraries={sorted(wanted) or ['(stdlib only)']}  "
                     f"same-library hits: {n_match}/{len(hits)}")
        lines.append("Q: " + " ".join(task["query_text"].split())[:PREVIEW_CHARS * 2])
        for rank, h in enumerate(hits, 1):
            md = h["metadata"]
            preview = " ".join(h["text"].split())[:PREVIEW_CHARS]
            lines.append(f"  {rank:2d}. {h['score']:.3f}  [{md['library']}] {md['source']}  "
                         f"{md['doc_id']}\n      {preview}")
        lines.append("")
    atomic_write_text(config.EYEBALL_FILE, "\n".join(lines))
    print("\n".join(lines))
    with_libs = [m for t, m in zip(picked, matches, strict=True) if t["packages"]]
    return {"questions": len(picked), "eyeball_file": str(config.EYEBALL_FILE),
            "mean_same_library_hits_of_10": round(sum(with_libs) / max(1, len(with_libs)), 2),
            "questions_with_majority_same_library": sum(m > len(r) / 2
                                                        for m, r in zip(matches, results, strict=True))}


# ---------------------------------------------------------------------------
# 3. NDCG@10
# ---------------------------------------------------------------------------
def labelled_docs() -> dict[str, set[str]]:
    """{task_id: ids of its labelled library docs}, only for tasks that have labels."""
    labels: dict[str, set[str]] = {}
    for row in load_raw("ds1000_coderag"):
        labels[f"ds1000:{row['metadata']['problem_id']}"] = {d["title"] for d in row["docs"] or []}
    odex = load_raw("odex_coderag")
    for native_id, row in zip(odex_task_ids(odex["task_id"]), odex, strict=True):
        labels[f"odex:{native_id}"] = {d["title"] for d in row["docs"] or []}
    return {t: docs for t, docs in labels.items() if docs}


def check_label_alignment(tasks: dict[str, dict]) -> dict[str, int]:
    """The CodeRAG-Bench copies must describe the same tasks as our question sets."""
    mismatches = Counter()
    for row in load_raw("ds1000_coderag"):
        mismatches["ds1000"] += tasks[f"ds1000:{row['metadata']['problem_id']}"]["query_text"] != row["prompt"]
    odex = load_raw("odex_coderag")
    for native_id, row in zip(odex_task_ids(odex["task_id"]), odex, strict=True):
        mismatches["odex"] += tasks[f"odex:{native_id}"]["query_text"] != row["intent"]
    return dict(mismatches)


def ndcg(model, collection, tasks: dict[str, dict], split: dict[str, str], chunks) -> dict:
    labels = labelled_docs()
    libdoc_chunks = Counter(orig for orig, src in zip(chunks.column("orig_id").to_pylist(),
                                                      chunks.column("source").to_pylist(), strict=True)
                            if src == "library-documentation")
    task_ids = sorted(labels)
    results = search(collection, embed(model, [tasks[t]["query_text"] for t in task_ids],
                                       show_progress=True))
    scores: dict[str, list[float]] = defaultdict(list)
    for task_id, hits in zip(task_ids, results, strict=True):
        relevant = labels[task_id]
        rel = [int(h["metadata"]["source"] == "library-documentation"
                   and h["metadata"]["orig_id"] in relevant) for h in hits]
        score = ndcg_at_k(rel, sum(libdoc_chunks[d] for d in relevant), config.TOP_K)
        dataset = tasks[task_id]["dataset"]
        scores[dataset].append(score)
        if split[task_id] == "test":
            scores[f"{dataset}_test"].append(score)
    out = {name: {"ndcg@10": round(sum(v) / len(v), 4), "tasks": len(v)} for name, v in scores.items()}
    out["label_alignment_mismatches"] = check_label_alignment(tasks)
    for name, v in out.items():
        print(f"NDCG@10 {name}: {v}")
    return out


def main() -> None:
    ensure_local_index()
    collection = open_collection()
    chunks = load_chunk_table()
    counts = check_counts(collection, chunks)
    if not counts["passed"]:
        save_stats("day7_check_index", {"count_check": counts})
        raise SystemExit("Count check FAILED. Fix the missing shards before going on.")
    print("Count check passed.\n")

    day3 = "coding_track.steps.day3_tasks_splits_targets"
    tasks = {t["task_id"]: t for t in read_jsonl(require(config.TASKS_FILE, day3))}
    split = read_json(require(config.SPLITS_FILE, day3))["split"]
    model = load_retriever()
    eyeball_stats = eyeball(model, collection, list(tasks.values()), split)
    ndcg_stats = ndcg(model, collection, tasks, split, chunks)
    save_stats("day7_check_index", {"count_check": counts, "eyeball": eyeball_stats,
                                    "ndcg": ndcg_stats})
    write_notes()


if __name__ == "__main__":
    main()
