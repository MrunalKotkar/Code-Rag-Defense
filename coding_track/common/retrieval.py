"""The victim retriever (mGTE) and the ChromaDB collection, used by Days 6 and 7."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from coding_track import config


def load_retriever(device: str | None = None):
    """mGTE via sentence-transformers. fp16 on GPU, fp32 on CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(config.RETRIEVER_MODEL, revision=config.RETRIEVER_REVISION,
                                trust_remote_code=True, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = config.EMBED_MAX_SEQ_LEN
    print(f"[retriever] {config.RETRIEVER_MODEL} on {device}, max_seq_length={model.max_seq_length}")
    return model


def embed(model, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
    """L2-normalized float32 embeddings, shape (len(texts), EMBED_DIM)."""
    vectors = model.encode(list(texts), batch_size=config.EMBED_BATCH_SIZE,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=show_progress)
    return vectors.astype(np.float32, copy=False)


def chroma_client(path: Path = config.LOCAL_CHROMA_DIR):
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def create_or_open_collection(client):
    """The clean-corpus collection, with cosine distance fixed at creation."""
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, embedding_function=None,
        configuration={"hnsw": {"space": config.CHROMA_SPACE}})
    space = collection.configuration["hnsw"]["space"]
    if space != config.CHROMA_SPACE:
        raise RuntimeError(f"Collection uses '{space}' distance, expected '{config.CHROMA_SPACE}'. "
                           f"Delete {config.LOCAL_CHROMA_DIR} and rebuild.")
    return collection


def open_collection(path: Path = config.LOCAL_CHROMA_DIR):
    return chroma_client(path).get_collection(config.CHROMA_COLLECTION, embedding_function=None)


def search(collection, query_vectors: np.ndarray, k: int = config.TOP_K) -> list[list[dict]]:
    """Top-k hits per query: {'chunk_id', 'score' (cosine similarity), 'metadata', 'text'}."""
    res = collection.query(query_embeddings=query_vectors, n_results=k,
                           include=["metadatas", "distances", "documents"])
    return [[{"chunk_id": cid, "score": 1.0 - dist, "metadata": md, "text": doc}
             for cid, dist, md, doc in zip(ids, dists, mds, docs, strict=True)]
            for ids, dists, mds, docs in zip(res["ids"], res["distances"],
                                             res["metadatas"], res["documents"], strict=True)]
