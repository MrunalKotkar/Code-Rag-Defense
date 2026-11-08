"""Write notes/week1_numbers.md from the stats files the week-1 steps saved.

Every number comes from stats/*.json, so nothing is copied by hand. Steps that
have not run yet show as "not run yet". Day 7 calls this automatically; it can
also be run on its own at any time.

Runs on: anywhere with Drive mounted (Colab), or locally with CODING_TRACK_DRIVE set.

Usage:  python -m coding_track.steps.week1_notes
"""

from __future__ import annotations

from coding_track import config
from coding_track.common.io_utils import atomic_write_text, load_stats, today_iso

NOT_RUN = "_not run yet_"


def _section(title: str, stats: dict | None, body) -> list[str]:
    lines = [f"## {title}", ""]
    lines += body(stats) if stats else [NOT_RUN]
    return lines + [""]


def _downloads(s: dict) -> list[str]:
    rows = ["| dataset | Hub repo @ revision | rows | columns | license | downloaded |",
            "|---|---|---:|---|---|---|"]
    for key, d in s["datasets"].items():
        rows.append(f"| {key} | {d['repo_id']} @ {d['revision'][:8]} | {d['num_rows']:,} | "
                    f"{', '.join(d['columns'])} | {d['license']} | {d['downloaded_on']} |")
    return rows


def _tasks(s: dict) -> list[str]:
    lines = [f"- Tasks per dataset: {s['tasks_per_dataset']}",
             f"- Split (proxy/test) per dataset: {s['split_counts']}",
             f"- Rare libraries (tasks): {s['rare_libraries']}",
             f"- BigCodeBench tasks where ast imports differ from its `libs` field: "
             f"{s['bigcodebench_libs_field_disagreements']}", "",
             "| library | role | fake name | PyPI (checked) | tasks | proxy | test |",
             "|---|---|---|---|---:|---:|---:|"]
    for t in s["targets"]:
        lines.append(f"| {t['package']} | {t['role']} | `{t['fake_name']}` | {t['pypi_status']} "
                     f"({t['pypi_checked_on']}) | {t['tasks_total']} | {t['n_proxy']} | {t['n_test']} |")
    for lib, g in s["groups"].items():
        if g["role"] == "control":
            lines.append(f"| {lib} | control | - | - | {g['proxy'] + g['test']} | {g['proxy']} | {g['test']} |")
    return lines


def _fresh(s: dict) -> list[str]:
    lines = [f"- DevDocs `{slug}` ({d['library']}, release {d['release']}): {d['pages']:,} pages, "
             f"fetched {d['fetched']}" for slug, d in s["devdocs"].items()]
    lines += [f"- PyPI README `{p}` {d['version']}, fetched {d['fetched']}" for p, d in s["pypi"].items()]
    return lines + [f"- Not on DevDocs: {s['not_on_devdocs']}"]


def _stackoverflow(s: dict) -> list[str]:
    return [f"- Posts in CodeRAG-Bench SO set: {s['so_total_posts']:,}",
            f"- Kept: {s['kept_posts']:,} {s['kept_by_role']}",
            f"- Posts per library: {s['posts_per_library']}",
            f"- Overlap flags (MinHash >= {s['overlap_threshold']} or ODEX source question): "
            f"{s['overlap_pairs']:,} pairs, {s['overlap_distinct_posts']:,} posts, "
            f"{s['overlap_test_split_pairs']:,} pairs with a test task",
            f"- By dataset/split/type: {s['overlap_by_dataset_split_type']}",
            f"- List for Prof. Stamp: `{config.SO_OVERLAP_FILE}`"]


def _clean(s: dict) -> list[str]:
    lines = [f"- Documents before cleaning: {s['docs_before']:,}; after: {s['docs_after']:,}",
             f"- Dropped: {s['dropped_by_reason']}", "", "| source | counts |", "|---|---|"]
    lines += [f"| {src} | {c} |" for src, c in s["per_source"].items()]
    return lines


def _chunks(s: dict) -> list[str]:
    return [f"- **Total chunks: {s['chunks']:,}** from {s['documents']:,} documents",
            f"- Per source: {s['chunks_per_source']}",
            f"- Tokens per chunk: max {s['max_tokens']}, mean {s['mean_tokens']} "
            f"(limit {s['chunk_tokens']}, over limit: {s['chunks_over_budget']})",
            f"- Chunks from code blocks too long to keep whole: {s['chunks_from_split_code_blocks']:,}"]


def _embed(s: dict) -> list[str]:
    return [f"- GPU: {s.get('gpu', '?')}; {s['n_chunks']:,} chunks in {s['n_shards']} shards",
            f"- First shard: {s.get('first_shard_seconds')} s -> estimated total "
            f"**{s.get('estimated_total_hours')} h**",
            f"- Measured total: {s.get('measured_total_hours', '?')} h "
            f"({s.get('shards_done', '?')}/{s['n_shards']} shards done)"]


def _chroma(s: dict) -> list[str]:
    return [f"- ChromaDB {s['chromadb_version']}, collection `{s['collection']}`, space {s['space']}",
            f"- Count {s['count']:,} (chunks {s['n_chunks']:,}); build {s['build_minutes']} min",
            f"- Zip {s['zip_bytes'] / 1e9:.2f} GB, sha256 {s['zip_sha256'][:16]}..."]


def _check(s: dict) -> list[str]:
    lines = [f"- Count check: {s['count_check']}"]
    if "eyeball" in s:
        lines.append(f"- Eyeball (20 test questions): {s['eyeball']}")
    if "ndcg" in s:
        lines += [f"- NDCG@10 {name}: {v}" for name, v in s["ndcg"].items()]
    return lines


def build_notes() -> str:
    lines = [f"# Week 1 numbers (generated {today_iso()})", "",
             "All values below are read from stats/*.json written by the pipeline.", ""]
    lines += _section("Day 1 - Ollama context check", load_stats("day1_ollama_check"),
                      lambda s: [f"- {s['model']} on Ollama {s['ollama_version']}: "
                                 f"prompt_eval_count {s['prompt_eval_count']:,} with num_ctx "
                                 f"{s['num_ctx']}; recalled first line: {s['secret_recalled']}; "
                                 f"passed: {s['passed']}"])
    lines += _section("Day 2 - Downloads (real row counts)", load_stats("day2_download"), _downloads)
    lines += _section("Day 3 - Tasks, split, targets", load_stats("day3_tasks_splits_targets"), _tasks)
    lines += _section("Day 4 - Fresh docs", load_stats("day4_fresh_docs"), _fresh)
    lines += _section("Day 4 - Stack Overflow filter and overlap",
                      load_stats("day4_filter_stackoverflow"), _stackoverflow)
    lines += _section("Day 5 - Cleaning and manifest", load_stats("day5_clean_and_manifest"), _clean)
    lines += _section("Day 5 - Chunks", load_stats("day5_chunk"), _chunks)
    lines += _section("Day 6 - Embedding time", load_stats("day6_embed"), _embed)
    lines += _section("Day 6 - ChromaDB", load_stats("day6_build_chroma"), _chroma)
    lines += _section("Day 7 - Checks", load_stats("day7_check_index"), _check)
    return "\n".join(lines)


def write_notes() -> None:
    text = build_notes()
    atomic_write_text(config.NOTES_FILE, text)
    print(text)
    print(f"\n[notes] wrote {config.NOTES_FILE}")


if __name__ == "__main__":
    write_notes()
