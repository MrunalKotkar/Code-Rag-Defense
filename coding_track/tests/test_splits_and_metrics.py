import math

import pytest

from coding_track.common.metrics import ndcg_at_k
from coding_track.common.splits import assign_splits, group_by_library, rarest_library


def test_split_is_deterministic_and_stratified():
    strata = {"a": [f"a{i}" for i in range(10)], "b": [f"b{i}" for i in range(5)]}
    s1 = assign_splits(strata, 0.2, seed=42)
    s2 = assign_splits(strata, 0.2, seed=42)
    assert s1 == s2
    assert sum(s1[t] == "test" for t in strata["a"]) == 2
    assert sum(s1[t] == "test" for t in strata["b"]) == 1     # 5 * 0.2 = 1
    assert assign_splits(strata, 0.2, seed=7) != s1


def test_split_rounds_half_up():
    s = assign_splits({"x": [f"t{i}" for i in range(106)]}, 0.2, seed=42)   # 21.2 -> 21
    assert sum(v == "test" for v in s.values()) == 21


def test_task_in_two_strata_is_rejected():
    with pytest.raises(ValueError):
        assign_splits({"a": ["t1"], "b": ["t1"]}, 0.2, seed=42)


def test_grouping_uses_the_task_level_split():
    split = {"t1": "test", "t2": "proxy"}
    groups = group_by_library({"t1": ["pandas", "numpy"], "t2": ["numpy"]}, split, ["pandas", "numpy"])
    assert groups["pandas"] == {"proxy": [], "test": ["t1"]}
    assert groups["numpy"] == {"proxy": ["t2"], "test": ["t1"]}


def test_rarest_library():
    assert rarest_library({"pandas", "rsa"}, {"pandas": 700, "rsa": 6}) == "rsa"
    assert rarest_library(set(), {}) is None


def test_ndcg():
    assert ndcg_at_k([1, 0, 0], n_relevant=1, k=10) == 1.0
    assert ndcg_at_k([0, 1], n_relevant=1, k=10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([0] * 10, n_relevant=3, k=10) == 0.0
    # two chunks of the same labelled doc at ranks 1-2 never exceed 1
    assert ndcg_at_k([1, 1], n_relevant=2, k=10) == 1.0
