"""Week 2: re-read every saved answer with the current answer reader.

Runs on: main Colab notebook, Drive mounted, CPU is enough. Never while a
generation or GPT collect cell is running (they write the same files).

The raw answers are not touched. Only the fields read from them (code, imports,
pip installs, fake names, no-code and unfenced flags) are recomputed with
common/answers.py, so a fix to the reader reaches answers that were already
generated, without running any model again. Running it twice changes nothing
the second time.

The generation summaries in stats/ keep the counts from generation time; this
step's stats file has the counts after re-reading.

Output: week2/generations/*.jsonl rewritten (local copy + Drive),
        stats/w2_reread_answers.json

Usage:  python -m coding_track.steps.w2_reread_answers
"""

from __future__ import annotations

from pathlib import Path

from coding_track import config
from coding_track.common.generation import answer_fields
from coding_track.common.io_utils import (
    atomic_copy,
    atomic_write_jsonl,
    now_iso,
    read_jsonl,
    save_stats,
)
from coding_track.common.question_sets import fake_names, load_targets

COUNTS = {
    "code_found": lambda r: not r["abstained"],
    "code_found_unfenced": lambda r: bool(r.get("code_unfenced")),
    "fake_name_answers": lambda r: bool(r["fake_import"] or r["fake_install"]),
}
SHOW_CHANGED = 50          # task ids printed per file


def counts(records: list[dict]) -> dict[str, int]:
    return {name: sum(test(r) for r in records) for name, test in COUNTS.items()}


def newest_copy(drive: Path) -> Path:
    """Same rule as generation.load_done: the larger file holds more finished answers."""
    local = config.LOCAL_GENERATIONS_DIR / drive.name
    if local.exists() and local.stat().st_size > drive.stat().st_size:
        return local
    return drive


def main() -> None:
    files = sorted(config.GENERATIONS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No answers found in {config.GENERATIONS_DIR}")
    fakes = fake_names(load_targets())

    stats: dict = {"reread_at": now_iso(), "files": {}}
    for drive in files:
        old = list(read_jsonl(newest_copy(drive)))
        new, changed = [], []
        for record in old:
            fields = answer_fields(record["raw_answer"], fakes)
            if any(record.get(k) != v for k, v in fields.items()):
                changed.append(record["task_id"])
            new.append({**record, **fields})

        local = config.LOCAL_GENERATIONS_DIR / drive.name
        local.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(local, new)
        atomic_copy(local, drive)

        before, after = counts(old), counts(new)
        print(f"{drive.stem}: {len(new)} answers, {len(changed)} changed")
        for name in COUNTS:
            print(f"  {name}: {before[name]} -> {after[name]}")
        if changed:
            more = f" (+{len(changed) - SHOW_CHANGED} more)" if len(changed) > SHOW_CHANGED else ""
            print(f"  changed: {', '.join(changed[:SHOW_CHANGED])}{more}")
        stats["files"][drive.stem] = {"answers": len(new), "changed": changed,
                                      "before": before, "after": after}
    save_stats("w2_reread_answers", stats)


if __name__ == "__main__":
    main()
