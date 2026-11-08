"""Day 6 (part 2): load the embedding shards into ChromaDB on local disk, then zip it to Drive.

Runs on: Colab (L4 session from part 1 is fine; no GPU needed), Drive mounted.

* The database is built in LOCAL_CHROMA_DIR (/content), never on Drive: live
  writes to Drive silently lost most of the NQ index.
* The collection is created with configuration={"hnsw": {"space": "cosine"}}.
* Chunks are added in batches of CHROMA_ADD_BATCH (capped by Chroma's own
  maximum), with their text and manifest metadata. ChromaDB rejects None
  metadata values, so a missing URL is stored as "".
* Re-running skips shards already present (first and last id found) and uses
  upsert, so a half-loaded shard is completed without duplicates.
* At the end the folder is zipped locally and copied to Drive (temp + rename).

Usage:  python -m coding_track.steps.day6_build_chroma
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np

from coding_track import config
from coding_track.common.io_utils import atomic_copy, require, save_stats, sha256_file
from coding_track.common.retrieval import chroma_client, create_or_open_collection

METADATA_COLUMNS = ["doc_id", "orig_id", "source", "url", "library", "lib_version", "fetched",
                    "license", "is_poison", "chunk_index", "n_tokens"]


def to_metadata(row: dict) -> dict:
    return {c: ("" if row[c] is None else row[c]) for c in METADATA_COLUMNS}


def load_shard(i: int) -> tuple[list[str], np.ndarray]:
    path = config.EMBED_DIR / f"shard_{i:05d}.npz"
    with np.load(path, allow_pickle=False) as data:
        return data["ids"].tolist(), data["embeddings"]


def shard_present(collection, ids: list[str]) -> bool:
    return len(collection.get(ids=[ids[0], ids[-1]], include=[])["ids"]) == 2


def zip_to_drive() -> None:
    base = config.LOCAL_TMP_DIR / config.CHROMA_ZIP_FILE.stem
    archive = shutil.make_archive(str(base), "zip", root_dir=config.LOCAL_CHROMA_DIR)
    atomic_copy(Path(archive), config.CHROMA_ZIP_FILE)
    print(f"zipped index -> {config.CHROMA_ZIP_FILE}")


def main() -> None:
    import chromadb
    import pyarrow.parquet as pq

    src = require(config.CHUNKS_FILE, "coding_track.steps.day5_chunk")
    local = config.LOCAL_TMP_DIR / src.name
    if not local.exists() or local.stat().st_size != src.stat().st_size:
        atomic_copy(src, local)
    table = pq.read_table(local, columns=["chunk_id", "text", *METADATA_COLUMNS], memory_map=True)
    n_chunks = table.num_rows
    n_shards = math.ceil(n_chunks / config.EMBED_SHARD_SIZE)
    missing = [i for i in range(n_shards) if not (config.EMBED_DIR / f"shard_{i:05d}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} embedding shards missing (e.g. {missing[:5]}). "
                         "Finish day6_embed first.")

    client = chroma_client()
    collection = create_or_open_collection(client)
    batch = min(config.CHROMA_ADD_BATCH, client.get_max_batch_size())
    print(f"ChromaDB {chromadb.__version__} at {config.LOCAL_CHROMA_DIR}; add batch {batch}")

    start = time.time()
    for i in range(n_shards):
        ids, vectors = load_shard(i)
        rows = table.slice(i * config.EMBED_SHARD_SIZE, config.EMBED_SHARD_SIZE).to_pylist()
        if [r["chunk_id"] for r in rows] != ids:
            raise SystemExit(f"Shard {i} ids do not match chunks.parquet; re-run day6_embed.")
        if shard_present(collection, ids):
            continue
        for s in range(0, len(ids), batch):
            part = rows[s:s + batch]
            collection.upsert(ids=ids[s:s + batch], embeddings=vectors[s:s + batch],
                              documents=[r["text"] for r in part],
                              metadatas=[to_metadata(r) for r in part])
        print(f"shard {i + 1}/{n_shards} loaded; collection count {collection.count():,}")

    count = collection.count()
    print(f"\nChromaDB count {count:,} vs chunks {n_chunks:,}")
    if count != n_chunks:
        raise SystemExit("Count mismatch: run day7_check_index to find the missing shards.")
    del collection, client                     # release file handles before zipping
    zip_to_drive()
    save_stats("day6_build_chroma", {
        "chromadb_version": chromadb.__version__, "collection": config.CHROMA_COLLECTION,
        "space": config.CHROMA_SPACE, "count": count, "n_chunks": n_chunks,
        "build_minutes": round((time.time() - start) / 60, 1),
        "zip_bytes": config.CHROMA_ZIP_FILE.stat().st_size,
        "zip_sha256": sha256_file(config.CHROMA_ZIP_FILE),
    })


if __name__ == "__main__":
    main()
