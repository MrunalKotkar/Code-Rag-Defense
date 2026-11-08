"""Day 5 (part 2): cut the clean documents into chunks of at most 1024 mGTE tokens.

Runs on: Colab High-RAM CPU runtime, Drive mounted (needs day5_clean_and_manifest).

Tokens are counted with mGTE's own tokenizer (no special tokens). Chunks break
only at paragraph or code-block edges and never inside a ``` code block
(rules and the one exception are in common/chunking.py). Each chunk carries
its document's manifest fields, ready to become ChromaDB metadata.

Output (Drive, derived/): chunks.parquet. Prints the total number of chunks.

Usage:  python -m coding_track.steps.day5_chunk
"""

from __future__ import annotations

import os
from collections import Counter

from coding_track import config
from coding_track.common.chunking import Chunker, HFTokenCounter
from coding_track.common.io_utils import atomic_copy, require, save_stats

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # we parallelize with processes

DOC_COLUMNS = ["doc_id", "orig_id", "source", "url", "library", "lib_version", "fetched",
               "license", "is_poison"]
CHUNK_COLUMNS = [*DOC_COLUMNS, "chunk_id", "chunk_index", "n_tokens", "from_split_code", "text"]

_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """One tokenizer per worker process, loaded on first use."""
    global _chunker
    if _chunker is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.RETRIEVER_MODEL,
                                                  revision=config.RETRIEVER_REVISION)
        tokenizer.model_max_length = 10**9    # we only count tokens; silence length warnings
        _chunker = Chunker(HFTokenCounter(tokenizer), config.CHUNK_TOKENS)
    return _chunker


def chunk_batch(batch: dict) -> dict:
    chunker = get_chunker()
    out: dict[str, list] = {c: [] for c in CHUNK_COLUMNS}
    for i, text in enumerate(batch["text"]):
        for j, chunk in enumerate(chunker.chunk(text)):
            for c in DOC_COLUMNS:
                out[c].append(batch[c][i])
            out["chunk_id"].append(f"{batch['doc_id'][i]}_c{j}")
            out["chunk_index"].append(j)
            out["n_tokens"].append(chunk.n_tokens)
            out["from_split_code"].append(chunk.from_split_code)
            out["text"].append(chunk.text)
    return out


def main() -> None:
    from datasets import Dataset

    src = require(config.CLEAN_DOCS_FILE, "coding_track.steps.day5_clean_and_manifest")
    docs = Dataset.from_parquet(str(src))
    print(f"documents: {len(docs):,}")
    chunks = docs.map(chunk_batch, batched=True, batch_size=256, num_proc=config.NUM_PROC,
                      remove_columns=docs.column_names, desc="chunking")

    local = config.LOCAL_TMP_DIR / config.CHUNKS_FILE.name
    local.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(str(local))
    atomic_copy(local, config.CHUNKS_FILE)

    n_tokens = chunks["n_tokens"]
    over = sum(n > config.CHUNK_TOKENS for n in n_tokens)
    docs_without_chunks = len(docs) - len(set(chunks["doc_id"]))
    per_source = dict(Counter(chunks["source"]))
    print(f"\nTOTAL CHUNKS: {len(chunks):,}")
    print(f"per source: {per_source}")
    print(f"max tokens {max(n_tokens)}, mean {sum(n_tokens) / len(n_tokens):.0f}; "
          f"chunks over {config.CHUNK_TOKENS}: {over}; docs with no chunk: {docs_without_chunks}")
    save_stats("day5_chunk", {
        "documents": len(docs), "chunks": len(chunks), "chunks_per_source": per_source,
        "chunk_tokens": config.CHUNK_TOKENS, "max_tokens": max(n_tokens),
        "mean_tokens": round(sum(n_tokens) / len(n_tokens), 1), "chunks_over_budget": over,
        "chunks_from_split_code_blocks": sum(chunks["from_split_code"]),
        "docs_without_chunks": docs_without_chunks,
    })
    if over or docs_without_chunks:
        raise SystemExit("Chunk check failed (see numbers above).")


if __name__ == "__main__":
    main()
