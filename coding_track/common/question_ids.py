"""Stable task ids for the question sets.

ODEX's task_id is the Stack Overflow question id, and one question can have
several English intents (439 rows, 333 distinct ids; some intents are word-for-
word identical). Each row gets a unique id "<question id>_<k>", where k counts
earlier rows with the same question id in dataset order. CodeRAG-Bench's ODEX
copy has the same row order, so the same function labels both.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def odex_task_ids(question_ids: Iterable) -> list[str]:
    seen: Counter = Counter()
    ids = []
    for qid in question_ids:
        ids.append(f"{qid}_{seen[qid]}")
        seen[qid] += 1
    return ids
