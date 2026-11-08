"""Retrieval metrics."""

from __future__ import annotations

import math
from typing import Sequence


def ndcg_at_k(relevance: Sequence[int], n_relevant: int, k: int) -> float:
    """Binary NDCG@k.

    relevance:  1/0 for each retrieved item, best-ranked first.
    n_relevant: how many relevant items exist in the whole index. Here a chunk
                is relevant when its parent document is labelled, so this is the
                number of chunks belonging to labelled documents. Using chunks
                (not documents) for the ideal ranking keeps the score <= 1 when
                several chunks of one labelled document are retrieved.
    """
    if n_relevant <= 0:
        raise ValueError("ndcg_at_k needs at least one relevant item")
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance[:k]))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, n_relevant)))
    return dcg / idcg
