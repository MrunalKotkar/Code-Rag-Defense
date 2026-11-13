"""One saved line per (model, setting, question), the same for Ollama and OpenAI runs.

Results are appended to a local JSONL file, copied to Drive every
COPY_TO_DRIVE_EVERY lines (temp file + rename), and finished task ids are
skipped on restart, so a dropped Colab session loses at most one copy interval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from coding_track import config
from coding_track.common.answers import fake_names_found, read_answer
from coding_track.common.io_utils import atomic_copy, now_iso, read_jsonl


def run_name(model: str, setting: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}__{setting}"


def generation_paths(model: str, setting: str) -> tuple[Path, Path]:
    name = run_name(model, setting) + ".jsonl"
    return config.LOCAL_GENERATIONS_DIR / name, config.GENERATIONS_DIR / name


def load_done(model: str, setting: str) -> tuple[Path, Path, set[str]]:
    """Local path, Drive path and finished task ids (Drive copy restores a new session)."""
    local, drive = generation_paths(model, setting)
    local.parent.mkdir(parents=True, exist_ok=True)
    if drive.exists() and (not local.exists() or drive.stat().st_size > local.stat().st_size):
        atomic_copy(drive, local)
    done = {r["task_id"] for r in read_jsonl(local)} if local.exists() else set()
    return local, drive, done


def append(local: Path, record: dict) -> None:
    with open(local, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(prompt_row: dict, *, model: str, model_id: str, setting: str, raw_answer: str,
                fakes: list[str], **measurements) -> dict:
    reading = read_answer(raw_answer)
    found = fake_names_found(reading, fakes)
    return {
        "task_id": prompt_row["task_id"], "dataset": prompt_row["dataset"],
        "native_id": prompt_row["native_id"], "packages": prompt_row["packages"],
        "question_sets": prompt_row["question_sets"],
        "model": model, "model_id": model_id, "setting": setting,
        "retrieved_ids": prompt_row["retrieved_ids"], "prompt_chars": len(prompt_row["prompt"]),
        **measurements,
        "raw_answer": raw_answer, "code": reading.code, "n_blocks": len(reading.blocks),
        "all_code": "\n\n".join(reading.blocks), "imports": reading.imports,
        "pip_installs": reading.pip_installs, "unparsable_blocks": reading.unparsable_blocks,
        "abstained": reading.abstained, "fake_import": found["import"], "fake_install": found["install"],
        "created_at": now_iso(),
    }
