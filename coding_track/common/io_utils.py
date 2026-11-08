"""Small file and network helpers shared by every step.

All writes that matter go through `atomic_*`: write a temporary file next to the
target, then rename it. A disconnect can never leave a half-written file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator

from coding_track import config


def today_iso() -> str:
    return dt.date.today().isoformat()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------
def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Stream records to a temp file, then rename. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy src to dst through a temp name in dst's folder (safe on Drive)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Size mismatch while copying {src} -> {dst}")
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, produced_by: str) -> Path:
    """Fail early with a clear message when an input from an earlier step is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `{produced_by}` first.")
    return path


# ---------------------------------------------------------------------------
# Stats: one JSON per step with the real numbers (feeds the week-1 notes)
# ---------------------------------------------------------------------------
def save_stats(step: str, stats: dict) -> Path:
    path = config.STATS_DIR / f"{step}.json"
    atomic_write_json(path, {**stats, "step": step, "written_at": now_iso()})
    print(f"[stats] wrote {path}")
    return path


def load_stats(step: str) -> dict | None:
    path = config.STATS_DIR / f"{step}.json"
    return read_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
# HTTP (standard library only)
# ---------------------------------------------------------------------------
def http_get(url: str) -> tuple[int, bytes]:
    """GET with retries. Returns (status, body). A 404 is returned, not raised."""
    last_err: Exception | None = None
    for attempt in range(1, config.HTTP_RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": config.HTTP_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=config.HTTP_TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            last_err = e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(2 * attempt)
    raise RuntimeError(f"GET {url} failed after {config.HTTP_RETRIES} tries: {last_err}")


def http_get_json(url: str) -> Any:
    status, body = http_get(url)
    if status != 200:
        raise RuntimeError(f"GET {url} returned HTTP {status}")
    return json.loads(body)
