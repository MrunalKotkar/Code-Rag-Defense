"""Day 2: download the 4 CodeRAG-Bench document sets and the question sets, once.

Runs on: Colab High-RAM CPU runtime, Drive mounted. (Stack Overflow is ~2M posts.)

Downloads (each at the commit pinned in config.py):
  corpus     library-documentation, online-tutorials, programming-solutions,
             stackoverflow-posts
  questions  DS-1000, ODEX (English), BigCodeBench
  labels     CodeRAG-Bench's copies of DS-1000 and ODEX (their labelled correct docs)

Each dataset is saved with `save_to_disk` on local disk, then copied to Drive
through a temp folder + rename. A dataset already on Drive is never downloaded
again. For every dataset the script prints the real column names and row count
and records them (plus license and file checksums) in stats/day2_download.json.

Usage:  python -m coding_track.steps.day2_download            # all datasets
        python -m coding_track.steps.day2_download --only odex # just some
"""

from __future__ import annotations

import argparse

from coding_track.common.hf_data import (
    ALL_DATASETS, copy_dir_atomic, download_from_hub, drive_path, load_raw, local_path)
from coding_track.common.io_utils import load_stats, save_stats, sha256_file, today_iso


def hub_license(repo_id: str, revision: str) -> str:
    """License from the dataset card metadata, or 'see source' if the card has none."""
    from huggingface_hub import HfApi

    card = HfApi().dataset_info(repo_id, revision=revision).card_data
    license_ = card.get("license") if card is not None else None
    return license_ if isinstance(license_, str) and license_ else "see source"


def preview(row: dict, width: int = 120) -> dict:
    return {k: (repr(v)[:width] + ("..." if len(repr(v)) > width else "")) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets once and save them to Drive.")
    parser.add_argument("--only", nargs="+", choices=sorted(ALL_DATASETS), metavar="KEY",
                        help=f"subset of: {', '.join(ALL_DATASETS)}")
    args = parser.parse_args()

    previous = (load_stats("day2_download") or {}).get("datasets", {})
    results = dict(previous)           # keep earlier entries when running a subset
    for key, ds in ALL_DATASETS.items():
        if args.only and key not in args.only:
            continue
        print(f"\n=== {key}  ({ds.repo_id} @ {ds.revision[:8]}, split={ds.split})")
        remote = drive_path(ds)
        if remote.exists():
            print(f"already on Drive: {remote}")
            data = load_raw(key)
            downloaded_on = previous.get(key, {}).get("downloaded_on", "unknown")
        else:
            data = download_from_hub(ds)
            data.save_to_disk(str(local_path(ds)))
            copy_dir_atomic(local_path(ds), remote)
            downloaded_on = today_iso()
            print(f"saved to {remote}")

        print(f"rows    : {data.num_rows:,}")
        print(f"columns : {data.column_names}")
        print(f"features: {data.features}")
        print(f"row 0   : {preview(data[0])}")

        files = sorted(p for p in local_path(ds).rglob("*") if p.is_file())
        results[key] = {
            "repo_id": ds.repo_id, "revision": ds.revision, "split": ds.split,
            "data_files": ds.data_files, "num_rows": data.num_rows,
            "columns": data.column_names, "features": str(data.features),
            "license": hub_license(ds.repo_id, ds.revision),
            "downloaded_on": downloaded_on,
            "files_sha256": {str(p.relative_to(local_path(ds))): sha256_file(p) for p in files},
        }

    save_stats("day2_download", {"datasets": results})
    print("\nRow counts:")
    for key, r in results.items():
        print(f"  {key:24s} {r['num_rows']:>10,}")
    missing = [k for k in ALL_DATASETS if k not in results]
    if missing:
        print(f"Not downloaded yet: {missing}")


if __name__ == "__main__":
    main()
