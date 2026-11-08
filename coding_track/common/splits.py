"""The 80/20 proxy/test split, made once per task (never per library).

A task is split once: a BigCodeBench task can use several libraries, and if it
were proxy for pandas but test for numpy the attacker would have seen a test
question. For the same reason near-copies of one question (a "split group",
see config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER) are split as one unit.

The split is stratified so that every target library gets about 20% test
tasks (a plain random split could leave a 5-task rare library with none).
`assign_splits` works on unit ids, which are task ids or split-group ids.
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping

PROXY, TEST = "proxy", "test"


def assign_splits(strata: Mapping[str, Iterable[str]], test_fraction: float,
                  seed: int) -> dict[str, str]:
    """Return {task_id: 'proxy' | 'test'}.

    Within each stratum, task ids are sorted, shuffled with one seeded RNG
    (strata visited in sorted order), and the first round-half-up(n * fraction)
    become test. Same input and seed always give the same split.
    """
    rng = random.Random(seed)
    split: dict[str, str] = {}
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        rng.shuffle(ids)
        n_test = int(len(ids) * test_fraction + 0.5)
        for i, task_id in enumerate(ids):
            if task_id in split:
                raise ValueError(f"Task {task_id} appears in more than one stratum")
            split[task_id] = TEST if i < n_test else PROXY
    return split


def rarest_library(packages: Iterable[str], library_counts: Mapping[str, int]) -> str | None:
    """The least-used of `packages` (ties broken by name), or None if empty."""
    candidates = sorted(packages, key=lambda p: (library_counts[p], p))
    return candidates[0] if candidates else None


def group_by_library(task_packages: Mapping[str, Iterable[str]], split: Mapping[str, str],
                     libraries: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    """{library: {'proxy': [...], 'test': [...]}} for the given libraries."""
    wanted = set(libraries)
    groups: dict[str, dict[str, list[str]]] = {lib: {PROXY: [], TEST: []} for lib in sorted(wanted)}
    for task_id, packages in task_packages.items():
        for lib in set(packages) & wanted:
            groups[lib][split[task_id]].append(task_id)
    for g in groups.values():
        g[PROXY].sort()
        g[TEST].sort()
    return groups
