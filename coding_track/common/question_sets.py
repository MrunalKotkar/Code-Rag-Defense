"""Which questions Week 2 uses, read from Week 1's tasks, split and targets (never hard-coded).

  scored  test-split tasks that use an attacked library (6 popular + rare) or a
          control library (PyTorch/TensorFlow). These are the questions the
          clean runs and later the attack runs are scored on.
  anchor  every DS-1000 and ODEX task. Only for one Llama 3.1 8B no-retrieval
          run that is compared with CodeRAG-Bench's published no-retrieval
          numbers (the scored set has too few ODEX tasks for that comparison).
"""

from __future__ import annotations

import random

from coding_track import config
from coding_track.common.io_utils import read_json, read_jsonl, require

DAY3 = "coding_track.steps.day3_tasks_splits_targets"


def load_tasks() -> dict[str, dict]:
    return {t["task_id"]: t for t in read_jsonl(require(config.TASKS_FILE, DAY3))}


def load_split() -> dict[str, str]:
    return read_json(require(config.SPLITS_FILE, DAY3))["split"]


def load_targets() -> dict:
    return read_json(require(config.TARGETS_FILE, DAY3))


def fake_names(targets: dict) -> list[str]:
    return [t["fake_name"] for t in targets["targets"]]


def question_ids(name: str, tasks: dict[str, dict] | None = None) -> list[str]:
    tasks = tasks or load_tasks()
    if name == "scored":
        split, targets = load_split(), load_targets()
        libraries = {t["package"] for t in targets["targets"]} | set(targets["control"])
        return sorted(tid for tid, t in tasks.items()
                      if split[tid] == "test" and set(t["packages"]) & libraries)
    if name == "anchor":
        return sorted(tid for tid, t in tasks.items() if t["dataset"] in ("ds1000", "odex"))
    raise ValueError(f"unknown question set {name!r}; choose from {config.QUESTION_SETS}")


def pilot_ids(ids: list[str]) -> list[str]:
    """PILOT_N questions drawn once with the project seed."""
    return sorted(random.Random(config.SEED).sample(ids, min(config.PILOT_N, len(ids))))
