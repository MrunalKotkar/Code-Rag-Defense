"""Day 4 (part 1): collect fresh docs for the 6 popular targets and the rare libraries.

Runs on: Colab High-RAM CPU runtime, Drive mounted (needs Day 3's targets.json).

Sources:
  DevDocs  https://devdocs.io/docs.json lists every doc set. For each library we
           take the entry whose name matches the package or import name
           (case-insensitive) with the newest `mtime`, then download its db.json
           ({page path: page HTML}). Not every library is on DevDocs (scipy,
           for one, is not); those get only their PyPI README.
  PyPI     https://pypi.org/pypi/<package>/json; the README is info.description.

The raw responses are saved unchanged under fresh/, with URL, version and
fetch date recorded in fresh/fresh_index.json. HTML -> text happens on Day 5.
Already-downloaded files are skipped, so the script can be re-run safely.

Usage:  python -m coding_track.steps.day4_fresh_docs
"""

from __future__ import annotations

import json

from coding_track import config
from coding_track.common.io_utils import (
    atomic_write_bytes, atomic_write_json, http_get, http_get_json, read_json, require,
    save_stats, today_iso)

DEVDOCS_DIR = config.FRESH_DIR / "devdocs"
PYPI_DIR = config.FRESH_DIR / "pypi"
FRESH_INDEX = config.FRESH_DIR / "fresh_index.json"


def pick_devdocs_entry(entries: list[dict], names: set[str]) -> dict | None:
    """Newest DevDocs doc set whose display name matches one of `names`."""
    wanted = {n.lower() for n in names}
    matches = [e for e in entries if e["name"].lower() in wanted]
    return max(matches, key=lambda e: e["mtime"]) if matches else None


def fetch_devdocs(target: dict, entries: list[dict], index: dict) -> None:
    entry = pick_devdocs_entry(entries, {target["package"], target["import_name"]})
    if entry is None:
        print(f"  DevDocs: no doc set for {target['package']}")
        return
    slug = entry["slug"]
    path = DEVDOCS_DIR / f"{slug}.db.json"
    url = config.DEVDOCS_DB_URL.format(slug=slug, mtime=entry["mtime"])
    if path.exists():
        print(f"  DevDocs: {slug} already downloaded")
    else:
        status, body = http_get(url)
        if status != 200:
            raise RuntimeError(f"{url} returned HTTP {status}")
        json.loads(body)                      # fail now if the body is not JSON
        atomic_write_bytes(path, body)
        index["devdocs"][slug] = {"fetched": today_iso()}
    pages = len(read_json(path))
    print(f"  DevDocs: {slug} (release {entry['release']}) -> {pages} pages")
    index["devdocs"][slug] = {
        **index["devdocs"].get(slug, {"fetched": "unknown"}),
        "library": target["package"], "slug": slug, "name": entry["name"],
        "release": entry["release"], "mtime": entry["mtime"], "db_url": url,
        "attribution": entry.get("attribution", ""), "file": path.name, "pages": pages,
    }


def fetch_pypi(target: dict, index: dict) -> None:
    package = target["package"]
    path = PYPI_DIR / f"{package}.json"
    if path.exists():
        print(f"  PyPI   : {package} already downloaded")
        data = read_json(path)
    else:
        data = http_get_json(config.PYPI_JSON_URL.format(name=package))
        atomic_write_json(path, data)
        index["pypi"][package] = {"fetched": today_iso()}
    info = data["info"]
    print(f"  PyPI   : {package} {info['version']} README {len(info.get('description') or ''):,} chars")
    index["pypi"][package] = {
        **index["pypi"].get(package, {"fetched": "unknown"}),
        "library": package, "version": info["version"],
        "url": config.PYPI_PROJECT_URL.format(name=package, version=info["version"]),
        "license": info.get("license_expression") or info.get("license") or "see source",
        "content_type": info.get("description_content_type"), "file": path.name,
    }


def main() -> None:
    targets = read_json(require(config.TARGETS_FILE, "coding_track.steps.day3_tasks_splits_targets"))
    index = read_json(FRESH_INDEX) if FRESH_INDEX.exists() else {"devdocs": {}, "pypi": {}}

    entries = http_get_json(config.DEVDOCS_LIST_URL)
    atomic_write_json(DEVDOCS_DIR / "docs.json", entries)
    index["devdocs_list_fetched"] = today_iso()

    for target in targets["targets"]:
        print(f"\n{target['package']} ({target['role']})")
        fetch_devdocs(target, entries, index)
        fetch_pypi(target, index)
        atomic_write_json(FRESH_INDEX, index)     # save progress after every library

    covered = {v["library"] for v in index["devdocs"].values()}
    missing = [t["package"] for t in targets["targets"] if t["package"] not in covered]
    print(f"\nNot on DevDocs (PyPI README only): {missing}")
    save_stats("day4_fresh_docs", {
        "devdocs": {slug: {"library": v["library"], "release": v["release"], "pages": v["pages"],
                           "fetched": v["fetched"]} for slug, v in index["devdocs"].items()},
        "pypi": {p: {"version": v["version"], "fetched": v["fetched"]} for p, v in index["pypi"].items()},
        "not_on_devdocs": missing,
    })


if __name__ == "__main__":
    main()
