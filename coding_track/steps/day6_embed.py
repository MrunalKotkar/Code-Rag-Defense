"""Day 6 (part 1): embed every chunk with mGTE, in resumable shards of 10,000 chunks.

Runs on: Colab L4 GPU runtime, Drive mounted (needs day5_chunk). Works on CPU
too, just far slower.

Each shard is one .npz file (chunk ids + float32 vectors) written to local
disk, copied to Drive under a temp name, then renamed: a disconnect can never
leave half a shard. On restart, shards already on Drive are skipped.
embeddings/spec.json pins which chunks file the shards belong to; if
chunks.parquet changes, the script stops instead of mixing old and new shards.

The first shard embedded is timed and multiplied by the number of shards to
estimate the total; both the estimate and the real total go into the stats.

Usage:  python -m coding_track.steps.day6_embed
"""

from __future__ import annotations

import math
import time

import numpy as np

from coding_track import config
from coding_track.common.io_utils import (
    atomic_copy, atomic_write_json, load_stats, read_json, require, save_stats, sha256_file)
from coding_track.common.retrieval import embed, load_retriever


def shard_path(i: int):
    return config.EMBED_DIR / f"shard_{i:05d}.npz"


def check_spec(chunks_sha: str, n_chunks: int) -> None:
    spec = {"chunks_sha256": chunks_sha, "n_chunks": n_chunks, "shard_size": config.EMBED_SHARD_SIZE,
            "model": config.RETRIEVER_MODEL, "revision": config.RETRIEVER_REVISION,
            "max_seq_len": config.EMBED_MAX_SEQ_LEN}
    if config.EMBED_SPEC_FILE.exists():
        old = read_json(config.EMBED_SPEC_FILE)
        if old != spec:
            raise SystemExit(f"Existing shards were made for a different setup:\n{old}\nvs now\n{spec}\n"
                             f"Delete {config.EMBED_DIR} to start over.")
    else:
        atomic_write_json(config.EMBED_SPEC_FILE, spec)


def main() -> None:
    import pyarrow.parquet as pq
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no GPU found. Day 6 is meant for an L4 runtime; this will be slow.")
    src = require(config.CHUNKS_FILE, "coding_track.steps.day5_chunk")
    local = config.LOCAL_TMP_DIR / src.name
    if not local.exists() or local.stat().st_size != src.stat().st_size:
        atomic_copy(src, local)
    table = pq.read_table(local, columns=["chunk_id", "text"], memory_map=True)
    n_chunks = table.num_rows
    n_shards = math.ceil(n_chunks / config.EMBED_SHARD_SIZE)
    check_spec(sha256_file(local), n_chunks)
    todo = [i for i in range(n_shards) if not shard_path(i).exists()]
    print(f"{n_chunks:,} chunks -> {n_shards} shards; {n_shards - len(todo)} done, {len(todo)} to go")

    stats = load_stats("day6_embed") or {}
    shard_seconds: dict[str, float] = stats.get("shard_seconds", {})
    model = load_retriever() if todo else None
    for i in todo:
        rows = table.slice(i * config.EMBED_SHARD_SIZE, config.EMBED_SHARD_SIZE)
        ids = rows.column("chunk_id").to_pylist()
        start = time.time()
        vectors = embed(model, rows.column("text").to_pylist(), show_progress=True)
        seconds = time.time() - start
        assert vectors.shape == (len(ids), config.EMBED_DIM), vectors.shape

        tmp = config.LOCAL_TMP_DIR / f"shard_{i:05d}.npz"
        np.savez(tmp, ids=np.array(ids), embeddings=vectors)
        atomic_copy(tmp, shard_path(i))
        tmp.unlink()
        shard_seconds[str(i)] = round(seconds, 1)
        print(f"shard {i + 1}/{n_shards}: {len(ids):,} chunks in {seconds:.0f} s")
        if "first_shard_seconds" not in stats:
            stats["first_shard_seconds"] = round(seconds, 1)
            stats["estimated_total_hours"] = round(seconds * n_shards / 3600, 2)
            stats["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            print(f"ESTIMATE: {seconds:.0f} s x {n_shards} shards = "
                  f"{stats['estimated_total_hours']} h of embedding")
        save_stats("day6_embed", {**stats, "n_chunks": n_chunks, "n_shards": n_shards,
                                  "shard_seconds": shard_seconds})

    done = sum(shard_path(i).exists() for i in range(n_shards))
    total_hours = round(sum(shard_seconds.values()) / 3600, 2)
    print(f"shards on Drive: {done}/{n_shards}; measured embedding time {total_hours} h")
    save_stats("day6_embed", {**stats, "n_chunks": n_chunks, "n_shards": n_shards,
                              "shards_done": done, "measured_total_hours": total_hours,
                              "shard_seconds": shard_seconds})


if __name__ == "__main__":
    main()
