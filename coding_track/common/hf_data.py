"""Download Hugging Face datasets once, keep them on Drive, read them from local disk.

Drive is slow for random reads, so `load_raw` first copies a saved dataset to
Colab's local disk (LOCAL_RAW_CACHE) and memory-maps it from there.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from coding_track import config
from coding_track.config import HFDataset

ALL_DATASETS: dict[str, HFDataset] = {
    d.key: d for d in (*config.CORPUS_DATASETS, *config.QUESTION_DATASETS, *config.LABEL_DATASETS)
}


def drive_path(ds: HFDataset) -> Path:
    parent = config.RAW_CORPUS_DIR if ds in config.CORPUS_DATASETS else config.RAW_QUESTIONS_DIR
    return parent / ds.key


def local_path(ds: HFDataset) -> Path:
    return config.LOCAL_RAW_CACHE / ds.key


def copy_dir_atomic(src: Path, dst: Path) -> None:
    """Copy a folder to dst via a temp folder + rename, so dst is never half-copied."""
    if dst.exists():
        return
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)


def download_from_hub(ds: HFDataset):
    """Load one dataset from the Hub at its pinned revision."""
    from datasets import load_dataset

    if ds.data_files:
        # Repos with a loading script are refused by datasets>=4 even when
        # data_files is given, so read the file with the generic JSON builder.
        url = f"hf://datasets/{ds.repo_id}@{ds.revision}/{ds.data_files}"
        return load_dataset("json", data_files=url, split=ds.split)
    return load_dataset(ds.repo_id, ds.config, revision=ds.revision, split=ds.split)


def load_raw(key: str):
    """Return a saved dataset by key, copying it from Drive to local disk first if needed."""
    from datasets import load_from_disk

    ds = ALL_DATASETS[key]
    local, remote = local_path(ds), drive_path(ds)
    if not local.exists():
        if not remote.exists():
            raise FileNotFoundError(
                f"{remote} not found. Run `python -m coding_track.steps.day2_download` first.")
        print(f"[load_raw] copying {remote} -> {local}")
        copy_dir_atomic(remote, local)
    return load_from_disk(str(local))
