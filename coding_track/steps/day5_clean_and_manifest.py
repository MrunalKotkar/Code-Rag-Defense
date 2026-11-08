"""Day 5 (part 1): gather every document, clean it, remove duplicates, write the manifest.

Runs on: Colab High-RAM CPU runtime, Drive mounted (needs Days 2-4).

Sources and IDs (original IDs are kept in `orig_id`; `doc_id` = "<prefix>:<orig_id>"):
  libdoc    library-documentation   orig_id = its doc_id (these are what the
                                    DS-1000/ODEX labels point to)
  devdocs   DevDocs pages           orig_id = "<slug>/<page path>"
  pypi      PyPI READMEs            orig_id = package name
  solution  programming-solutions   orig_id = "<task_name>/<task_id>"
  tutorial  online-tutorials        orig_id = row number (the dataset has no ID column)
  so        filtered Stack Overflow orig_id = question id from the post URL

Cleaning:
  * DevDocs HTML -> text, <pre> blocks kept as ``` fenced code;
    programming solutions (pure Python) are wrapped as one fenced block;
    tutorial pages get their title as the first line;
  * drop pages with fewer than MIN_DOC_WORDS words;
  * drop exact duplicates (SHA-256 of whitespace-collapsed text);
  * drop near-duplicates (MinHash, estimated Jaccard >= NEAR_DUP_THRESHOLD).
Documents are processed in priority order and the first copy wins. Library
docs that DS-1000/ODEX label as correct are "protected": never dropped, so
NDCG@10 can be measured against them.

Outputs (Drive, derived/): clean_docs.parquet, manifest.jsonl, dedup_dropped.jsonl

Usage:  python -m coding_track.steps.day5_clean_and_manifest
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np

from coding_track import config
from coding_track.common.hf_data import load_raw
from coding_track.common.imports import import_to_package
from coding_track.common.io_utils import (
    atomic_copy, atomic_write_jsonl, load_stats, read_json, require, save_stats)
from coding_track.common.minhash_utils import new_lsh, signatures, to_minhash
from coding_track.common.so_filter import question_id
from coding_track.common.text_clean import (
    collapse_whitespace, fence, html_to_text, normalize_text, word_count)

FRESH_INDEX = config.FRESH_DIR / "fresh_index.json"
UNKNOWN = "unknown"
# library-documentation doc_id prefixes that are not import names.
LIBDOC_PREFIX_TO_LIBRARY = {
    "python": "python-stdlib",
    "matplotlib_configuration_api": "matplotlib",
    "django_rest_framework": "djangorestframework",
}


@dataclass
class Doc:
    doc_id: str
    orig_id: str
    source: str
    url: str | None
    library: str
    lib_version: str
    fetched: str
    license: str
    text: str
    protected: bool = False


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def labelled_doc_ids() -> set[str]:
    ids = set()
    for key in ("ds1000_coderag", "odex_coderag"):
        for docs in load_raw(key)["docs"]:
            ids.update(d["title"] for d in docs or [])
    return ids


def library_docs(meta: dict, labelled: set[str]) -> Iterator[Doc]:
    for row in load_raw("library-documentation"):
        prefix = row["doc_id"].split("#")[0].split(".")[0]
        library = LIBDOC_PREFIX_TO_LIBRARY.get(prefix) or import_to_package(prefix)
        yield Doc(f"libdoc:{row['doc_id']}", row["doc_id"], "library-documentation", None,
                  library, UNKNOWN, meta["downloaded_on"], meta["license"],
                  row["doc_content"], protected=row["doc_id"] in labelled)


def devdocs_pages(index: dict) -> Iterator[Doc]:
    for slug, entry in sorted(index.get("devdocs", {}).items()):
        attribution = html_to_text(entry["attribution"]) if entry["attribution"] else "see source"
        pages = read_json(config.FRESH_DIR / "devdocs" / entry["file"])
        for path, html in pages.items():
            yield Doc(f"devdocs:{slug}/{path}", f"{slug}/{path}", "devdocs",
                      config.DEVDOCS_PAGE_URL.format(slug=slug, path=path), entry["library"],
                      entry["release"], entry["fetched"], attribution, html_to_text(html))


def pypi_readmes(index: dict) -> Iterator[Doc]:
    for package, entry in sorted(index.get("pypi", {}).items()):
        info = read_json(config.FRESH_DIR / "pypi" / entry["file"])["info"]
        yield Doc(f"pypi:{package}", package, "pypi-readme", entry["url"], package,
                  entry["version"], entry["fetched"], entry["license"], info.get("description") or "")


def programming_solutions(meta: dict) -> Iterator[Doc]:
    for row in load_raw("programming-solutions"):
        orig_id = f"{row['meta']['task_name']}/{row['meta']['task_id']}"
        yield Doc(f"solution:{orig_id}", orig_id, "programming-solutions", None, UNKNOWN, UNKNOWN,
                  meta["downloaded_on"], meta["license"], fence(row["text"], "python"))


def online_tutorials(meta: dict) -> Iterator[Doc]:
    for i, row in enumerate(load_raw("online-tutorials")):
        text = f"{row['title']}\n\n{row['text']}" if row["title"] else row["text"]
        yield Doc(f"tutorial:{i}", str(i), "online-tutorials", None, UNKNOWN, UNKNOWN,
                  meta["downloaded_on"], meta["license"], text)


def stackoverflow_posts(meta: dict) -> Iterator[Doc]:
    from datasets import load_from_disk

    kept = load_from_disk(str(require(config.SO_FILTERED_DIR,
                                      "coding_track.steps.day4_filter_stackoverflow")))
    for i, row in enumerate(kept):
        url = row["meta"]["url"]
        orig_id = question_id(url) or f"row{i}"
        yield Doc(f"so:{orig_id}", orig_id, "stackoverflow-posts", url,
                  ",".join(row["libraries"]) or UNKNOWN, UNKNOWN, meta["downloaded_on"],
                  meta["license"], row["text"])


def all_docs() -> list[Doc]:
    """Every document, in priority order (first copy wins in de-duplication)."""
    day2 = load_stats("day2_download")
    if day2 is None:
        raise FileNotFoundError("stats/day2_download.json missing; run day2_download first.")
    meta = day2["datasets"]
    index = read_json(require(FRESH_INDEX, "coding_track.steps.day4_fresh_docs"))
    labelled = labelled_doc_ids()

    docs: list[Doc] = []
    for name, source in [
        ("library-documentation", library_docs(meta["library-documentation"], labelled)),
        ("devdocs", devdocs_pages(index)),
        ("pypi", pypi_readmes(index)),
        ("programming-solutions", programming_solutions(meta["programming-solutions"])),
        ("online-tutorials", online_tutorials(meta["online-tutorials"])),
        ("stackoverflow-posts", stackoverflow_posts(meta[config.STACKOVERFLOW_KEY])),
    ]:
        before = len(docs)
        docs.extend(source)
        print(f"loaded {len(docs) - before:>9,} docs from {name}")

    found = sum(d.protected for d in docs)
    print(f"labelled library docs found: {found:,} of {len(labelled):,}")
    if found != len(labelled):
        raise SystemExit("Some labelled doc ids are missing from library-documentation.")
    docs.sort(key=lambda d: not d.protected)          # stable: protected first
    return docs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def make_ids_unique(docs: list[Doc]) -> int:
    seen: Counter = Counter()
    renamed = 0
    for d in docs:
        seen[d.doc_id] += 1
        if seen[d.doc_id] > 1:
            d.doc_id = f"{d.doc_id}#{seen[d.doc_id]}"
            renamed += 1
    return renamed


def deduplicate(docs: list[Doc]) -> tuple[list[Doc], list[dict]]:
    kept: list[Doc] = []
    dropped: list[dict] = []

    # 1. too short, 2. exact duplicates
    first_by_hash: dict[str, str] = {}
    candidates: list[Doc] = []
    for d in docs:
        if not d.protected and word_count(d.text) < config.MIN_DOC_WORDS:
            dropped.append({"doc_id": d.doc_id, "source": d.source, "reason": "too_short"})
            continue
        digest = hashlib.sha256(collapse_whitespace(d.text).encode()).hexdigest()
        if digest in first_by_hash and not d.protected:
            dropped.append({"doc_id": d.doc_id, "source": d.source, "reason": "exact_duplicate",
                            "duplicate_of": first_by_hash[digest]})
            continue
        first_by_hash.setdefault(digest, d.doc_id)
        candidates.append(d)

    # 3. near duplicates
    lsh = new_lsh(config.NEAR_DUP_THRESHOLD, config.MINHASH_NUM_PERM)
    stored: dict[str, np.ndarray] = {}
    sigs = signatures((d.text for d in candidates), config.DEDUP_SHINGLE_WORDS,
                      config.MINHASH_NUM_PERM, config.NUM_PROC)
    for n, (d, sig) in enumerate(zip(candidates, sigs, strict=True), 1):
        if sig is None:
            dropped.append({"doc_id": d.doc_id, "source": d.source, "reason": "no_words"})
            continue
        mh = to_minhash(sig)
        match = next((key for key in lsh.query(mh)
                      if np.mean(stored[key] == sig) >= config.NEAR_DUP_THRESHOLD), None)
        if match is not None and not d.protected:
            dropped.append({"doc_id": d.doc_id, "source": d.source, "reason": "near_duplicate",
                            "duplicate_of": match})
            continue
        lsh.insert(d.doc_id, mh)
        stored[d.doc_id] = sig
        kept.append(d)
        if n % 100_000 == 0:
            print(f"  near-dup check: {n:,}/{len(candidates):,}")
    return kept, dropped


def write_outputs(kept: list[Doc]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["doc_id", "orig_id", "source", "url", "library", "lib_version", "fetched",
               "license", "text"]
    table = pa.table({c: [getattr(d, c) for d in kept] for c in columns}
                     | {"is_poison": [False] * len(kept)})
    local = config.LOCAL_TMP_DIR / config.CLEAN_DOCS_FILE.name
    local.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, local)
    atomic_copy(local, config.CLEAN_DOCS_FILE)

    def manifest_line(d: Doc) -> dict:
        rec = {k: v for k, v in asdict(d).items() if k not in ("text", "protected")}
        return {**rec, "is_poison": False, "n_words": word_count(d.text),
                "labelled_for_ndcg": d.protected,
                "text_sha256": hashlib.sha256(d.text.encode()).hexdigest()}

    atomic_write_jsonl(config.MANIFEST_FILE, (manifest_line(d) for d in kept))


def main() -> None:
    docs = all_docs()
    for d in docs:
        d.text = normalize_text(d.text)
    renamed = make_ids_unique(docs)
    print(f"total docs before cleaning: {len(docs):,} (duplicate ids renamed: {renamed})")

    kept, dropped = deduplicate(docs)
    write_outputs(kept)
    atomic_write_jsonl(config.DEDUP_LOG_FILE, dropped)

    per_source: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        per_source[d.source]["input"] += 1
    for d in kept:
        per_source[d.source]["kept"] += 1
    for r in dropped:
        per_source[r["source"]][r["reason"]] += 1
    print(f"\n{'source':24s} counts")
    for source, c in per_source.items():
        print(f"{source:24s} {dict(c)}")
    print(f"\nmanifest: {len(kept):,} documents -> {config.MANIFEST_FILE}")
    save_stats("day5_clean_and_manifest", {
        "docs_before": len(docs), "docs_after": len(kept), "duplicate_ids_renamed": renamed,
        "dropped_by_reason": dict(Counter(r["reason"] for r in dropped)),
        "per_source": {s: dict(c) for s, c in per_source.items()},
        "min_doc_words": config.MIN_DOC_WORDS, "near_dup_threshold": config.NEAR_DUP_THRESHOLD,
    })


if __name__ == "__main__":
    main()
