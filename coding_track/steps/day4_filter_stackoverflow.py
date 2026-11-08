"""Day 4 (part 2): filter the ~2M Stack Overflow posts and flag overlap with the questions.

Runs on: Colab High-RAM CPU runtime, Drive mounted (needs Day 2 + Day 3 outputs).

1. Keep every post that mentions a target or rare library (rules in
   common/so_filter.py).
2. Add the same number of random other Python posts (seed 42) as background.
   Both passes use Dataset.filter(), which never loads 2M posts into a list.
3. Flag overlap. A kept post is flagged when its question part nearly copies a
   task's question text (MinHash Jaccard estimate >= OVERLAP_JACCARD_THRESHOLD),
   or when it IS the source question of an ODEX task (ODEX task_id is the Stack
   Overflow question id). Nothing is deleted; Prof. Stamp decides.

Outputs (Drive, derived/):
  stackoverflow_filtered/      kept posts + columns `libraries` and `so_role`
  stackoverflow_overlap.csv    one row per (post, task) overlap, with the task's split

Usage:  python -m coding_track.steps.day4_filter_stackoverflow
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict

from coding_track import config
from coding_track.common.hf_data import copy_dir_atomic, load_raw
from coding_track.common.io_utils import (
    atomic_write_text, load_stats, read_json, read_jsonl, require, save_stats)
from coding_track.common.minhash_utils import new_lsh, signatures, to_minhash
from coding_track.common.so_filter import LibraryMatcher, question_id
from coding_track.common.text_clean import stackoverflow_question

DAY3 = "coding_track.steps.day3_tasks_splits_targets"
LIBRARY_MATCH, BACKGROUND = "library_match", "background"


def _mentions_batch(batch: dict, matcher: LibraryMatcher) -> list[bool]:
    return [bool(matcher.libraries(t)) for t in batch["text"]]


def _unmatched_python_batch(batch: dict, matcher: LibraryMatcher) -> list[bool]:
    return [matcher.is_python(t) and not matcher.libraries(t) for t in batch["text"]]


def _tag_batch(batch: dict, matcher: LibraryMatcher, role: str) -> dict:
    return {"libraries": [matcher.libraries(t) for t in batch["text"]],
            "so_role": [role] * len(batch["text"])}


def filter_posts(so, matcher: LibraryMatcher):
    from datasets import concatenate_datasets

    kwargs = dict(batched=True, num_proc=config.NUM_PROC, fn_kwargs={"matcher": matcher})
    matched = so.filter(_mentions_batch, desc="posts that mention a library", **kwargs)
    pool = so.filter(_unmatched_python_batch, desc="other Python posts", **kwargs)
    n_background = min(len(pool), round(len(matched) * config.SO_BACKGROUND_RATIO))
    if n_background < len(matched) * config.SO_BACKGROUND_RATIO:
        print(f"WARNING: only {len(pool):,} unmatched Python posts available for background")
    background = pool.shuffle(seed=config.SEED).select(range(n_background))

    matched = matched.map(_tag_batch, batched=True, num_proc=config.NUM_PROC,
                          fn_kwargs={"matcher": matcher, "role": LIBRARY_MATCH})
    background = background.map(_tag_batch, batched=True, num_proc=config.NUM_PROC,
                                fn_kwargs={"matcher": matcher, "role": BACKGROUND})
    return concatenate_datasets([matched, background]), len(pool)


def find_overlap(kept, tasks: list[dict], split: dict[str, str]) -> list[dict]:
    task_by_id = {t["task_id"]: t for t in tasks}
    lsh = new_lsh(config.OVERLAP_JACCARD_THRESHOLD, config.MINHASH_NUM_PERM)
    task_sigs = {}
    for t, sig in zip(tasks, signatures((t["question_text"] for t in tasks),
                                        config.OVERLAP_SHINGLE_WORDS, config.MINHASH_NUM_PERM,
                                        config.NUM_PROC), strict=True):
        if sig is not None:
            task_sigs[t["task_id"]] = to_minhash(sig)
            lsh.insert(t["task_id"], task_sigs[t["task_id"]])
    tasks_by_question: dict[str, list[str]] = defaultdict(list)    # ODEX: SO id -> its rows
    for t in tasks:
        if t["so_question_id"]:
            tasks_by_question[t["so_question_id"]].append(t["task_id"])

    rows = []
    urls = [meta["url"] for meta in kept["meta"]]
    roles = list(kept["so_role"])
    questions = (stackoverflow_question(text) for text in kept["text"])
    post_sigs = signatures(questions, config.OVERLAP_SHINGLE_WORDS, config.MINHASH_NUM_PERM,
                           config.NUM_PROC)
    for i, sig in enumerate(post_sigs):
        url = urls[i]
        qid = question_id(url)
        hits: dict[str, tuple[str, float | None]] = {
            task_id: ("odex_source_question", None) for task_id in tasks_by_question.get(qid, [])}
        if sig is not None:
            post_mh = to_minhash(sig)
            for task_id in lsh.query(post_mh):
                score = post_mh.jaccard(task_sigs[task_id])
                if score >= config.OVERLAP_JACCARD_THRESHOLD:
                    kind = hits.get(task_id, ("minhash", None))[0]
                    hits[task_id] = (kind, round(score, 3))
        for task_id, (kind, score) in hits.items():
            t = task_by_id[task_id]
            rows.append({"so_question_id": qid, "so_url": url, "so_role": roles[i],
                         "task_id": task_id, "dataset": t["dataset"], "split": split[task_id],
                         "match_type": kind, "jaccard_estimate": score})
    return rows


def write_csv(path, rows: list[dict], fields: list[str]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def main() -> None:
    targets = read_json(require(config.TARGETS_FILE, DAY3))["targets"]
    tasks = list(read_jsonl(require(config.TASKS_FILE, DAY3)))
    split = read_json(require(config.SPLITS_FILE, DAY3))["split"]
    matcher = LibraryMatcher(targets, config.SO_NAME_MATCH_EXCLUDE)

    so = load_raw(config.STACKOVERFLOW_KEY)
    print(f"Stack Overflow posts: {len(so):,}")
    if config.SO_FILTERED_DIR.exists():
        # Resume: the filter already ran; reuse it (delete the folder to rebuild).
        from datasets import load_from_disk
        print(f"{config.SO_FILTERED_DIR} exists; reusing it")
        kept = load_from_disk(str(config.SO_FILTERED_DIR))
        pool_size = (load_stats("day4_filter_stackoverflow") or {}).get("unmatched_python_pool")
    else:
        kept, pool_size = filter_posts(so, matcher)
        local = config.LOCAL_TMP_DIR / "stackoverflow_filtered"
        kept.save_to_disk(str(local))
        copy_dir_atomic(local, config.SO_FILTERED_DIR)
        print(f"saved {config.SO_FILTERED_DIR}")
    per_library = Counter(lib for libs in kept["libraries"] for lib in libs)
    roles = Counter(kept["so_role"])
    print(f"kept {len(kept):,} posts: {dict(roles)}; unmatched Python pool {pool_size}")
    print("posts per library:", dict(per_library.most_common()))

    rows = find_overlap(kept, tasks, split)
    write_csv(config.SO_OVERLAP_FILE, rows, ["so_question_id", "so_url", "so_role", "task_id",
                                             "dataset", "split", "match_type", "jaccard_estimate"])
    flagged_posts = len({r["so_url"] for r in rows})
    by_group = Counter(f"{r['dataset']}/{r['split']}/{r['match_type']}" for r in rows)
    print(f"overlap: {len(rows):,} (post, task) pairs, {flagged_posts:,} distinct posts")
    print("by dataset/split/type:", dict(sorted(by_group.items())))

    save_stats("day4_filter_stackoverflow", {
        "so_total_posts": len(so), "kept_posts": len(kept), "kept_by_role": dict(roles),
        "unmatched_python_pool": pool_size, "posts_per_library": dict(per_library.most_common()),
        "name_match_excluded": sorted(config.SO_NAME_MATCH_EXCLUDE),
        "overlap_threshold": config.OVERLAP_JACCARD_THRESHOLD,
        "overlap_pairs": len(rows), "overlap_distinct_posts": flagged_posts,
        "overlap_by_dataset_split_type": dict(sorted(by_group.items())),
        "overlap_test_split_pairs": sum(r["split"] == "test" for r in rows),
    })


if __name__ == "__main__":
    main()
