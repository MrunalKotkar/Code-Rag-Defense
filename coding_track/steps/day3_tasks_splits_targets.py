"""Day 3: tag every task with its libraries, pick the rare libraries, split 80/20, check fake names.

Runs on: Colab High-RAM CPU runtime, Drive mounted (needs Day 2's downloads).
Also runs on a laptop: it only reads the three small question sets.

Where each task's libraries come from (reference solution imports, read with ast):
  DS-1000       imports in the prompt's starter code + reference_code, plus the
                task's own metadata["library"] label (every task has one).
  ODEX          the dataset's `library` field. ODEX solutions are bare expressions
                (e.g. `subprocess.check_output(...)`); their imports only exist in
                the test harness, so the labelled field is the reliable source.
                Any imports in prompt/solution are added too.
  BigCodeBench  imports in code_prompt + canonical_solution.
Standard-library modules are dropped; import names are mapped to PyPI package names.

Task ids: "ds1000:<problem_id>", "odex:<SO question id>_<k>" (ODEX has several
intents per question, see common/question_ids.py), "bigcodebench:BigCodeBench/<n>".

The split is made once per task and never per library. Near-copies are one
split unit so a test question's twin can never be in the attacker's proxy set:
ODEX rows of the same SO question always; DS-1000 variants of one origin problem
when config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER is True.

Outputs (Drive, derived/):
  tasks.jsonl                one line per task: id, dataset, imports, packages, texts
  task_library_counts.json   tasks per library (per dataset and total)
  splits.json                the one 80/20 split (seed 42) + tasks grouped by library
  targets.json               6 popular + rare targets, fake names, PyPI 404 check date

Usage:  python -m coding_track.steps.day3_tasks_splits_targets
"""

from __future__ import annotations

from collections import Counter, defaultdict

from coding_track import config
from coding_track.common.hf_data import load_raw
from coding_track.common.imports import import_to_package, third_party_packages, top_level_imports
from coding_track.common.io_utils import (
    atomic_write_json, atomic_write_jsonl, http_get, save_stats, today_iso)
from coding_track.common.question_ids import odex_task_ids
from coding_track.common.splits import PROXY, TEST, assign_splits, group_by_library, rarest_library


# ---------------------------------------------------------------------------
# Building task records
# ---------------------------------------------------------------------------
def _task(dataset: str, native_id, imports: set[str], packages: set[str], primary: str | None,
          question_text: str, query_text: str, split_group: str | None = None,
          so_question_id: str | None = None) -> dict:
    task_id = f"{dataset}:{native_id}"
    return {
        "task_id": task_id, "dataset": dataset, "native_id": str(native_id),
        "split_group": split_group or task_id, "so_question_id": so_question_id,
        "imports": sorted(imports), "packages": sorted(packages), "primary_library": primary,
        "question_text": question_text, "query_text": query_text,
    }


def ds1000_tasks() -> list[dict]:
    tasks = []
    for row in load_raw("ds1000"):
        meta = row["metadata"]
        imports = top_level_imports(row["prompt"] + "\n" + row["reference_code"])
        primary = config.DS1000_LIBRARY_TO_PACKAGE[meta["library"]]
        # Variants of one origin problem share (library, perturbation_origin_id).
        group = (f"ds1000:{meta['library']}:{meta['perturbation_origin_id']}"
                 if config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER else None)
        # Problem statement = the prompt before its "A:" code section (if any).
        statement = row["prompt"].split("\nA:\n", 1)[0]
        tasks.append(_task("ds1000", meta["problem_id"], imports,
                           third_party_packages(imports) | {primary}, primary,
                           question_text=statement, query_text=row["prompt"], split_group=group))
    return tasks


def odex_tasks() -> list[dict]:
    rows = load_raw("odex")
    tasks = []
    for native_id, row in zip(odex_task_ids(rows["task_id"]), rows, strict=True):
        code = row["prompt"] + row["canonical_solution"] + row["suffix"]
        imports = set(row["library"]) | top_level_imports(code)
        qid = str(row["task_id"])
        tasks.append(_task("odex", native_id, imports, third_party_packages(imports), None,
                           question_text=row["intent"], query_text=row["intent"],
                           split_group=f"odex:{qid}", so_question_id=qid))
    return tasks


def bigcodebench_tasks() -> tuple[list[dict], int]:
    """Returns the tasks and how many disagree with the dataset's own `libs` field."""
    import ast

    tasks, disagreements = [], 0
    for row in load_raw("bigcodebench"):
        imports = top_level_imports(row["code_prompt"] + row["canonical_solution"])
        disagreements += imports != set(ast.literal_eval(row["libs"]))
        tasks.append(_task("bigcodebench", row["task_id"], imports, third_party_packages(imports),
                           None, question_text=row["instruct_prompt"],
                           query_text=row["instruct_prompt"]))
    return tasks, disagreements


# ---------------------------------------------------------------------------
# Counting and choosing
# ---------------------------------------------------------------------------
def library_counts(tasks: list[dict]) -> dict[str, dict]:
    """{package: {'total', 'by_dataset', 'import_names'}} sorted by total, descending."""
    total: Counter = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    import_names: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        for pkg in t["packages"]:
            total[pkg] += 1
            by_dataset[pkg][t["dataset"]] += 1
        for name in t["imports"]:
            import_names[import_to_package(name)][name] += 1
    return {pkg: {"total": n, "by_dataset": dict(by_dataset[pkg]),
                  "import_names": dict(import_names[pkg].most_common())}
            for pkg, n in total.most_common()}


def choose_rare(counts: dict[str, dict]) -> list[str]:
    """The N least-used libraries with at least RARE_MIN_TASKS tasks (ties by name)."""
    excluded = {t.package for t in config.POPULAR_TARGETS} | set(config.CONTROL_PACKAGES)
    candidates = [(c["total"], pkg) for pkg, c in counts.items()
                  if pkg not in excluded and c["total"] >= config.RARE_MIN_TASKS]
    return [pkg for _, pkg in sorted(candidates)[:config.N_RARE_LIBRARIES]]


def make_strata(tasks: list[dict], attack_libs: set[str], counts: dict[str, dict]) -> dict[str, list[str]]:
    """Split groups per stratum.

    DS-1000: its own library label. ODEX/BigCodeBench: the rarest target library
    used by any task in the group (or "-" when none).
    """
    totals = {pkg: c["total"] for pkg, c in counts.items()}
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        groups[t["split_group"]].append(t)
    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        dataset = members[0]["dataset"]
        if dataset == "ds1000":
            key = members[0]["primary_library"]
        else:
            used = set().union(*(m["packages"] for m in members)) & attack_libs
            key = rarest_library(used, totals) or "-"
        strata[f"{dataset}|{key}"].append(group_id)
    return strata


def check_fake_names_absent(fake_names: list[str]) -> dict[str, int]:
    """Every fake name must return 404 on PyPI. Stops the script otherwise."""
    statuses = {}
    for name in fake_names:
        status, _ = http_get(config.PYPI_JSON_URL.format(name=name))
        statuses[name] = status
        print(f"  PyPI {name:24s} -> HTTP {status}")
    taken = [n for n, s in statuses.items() if s != 404]
    if taken:
        raise SystemExit(f"These fake names are NOT free on PyPI (need 404): {taken}. "
                         "Pick other names in config.py.")
    return statuses


# ---------------------------------------------------------------------------
def main() -> None:
    bcb, bcb_disagreements = bigcodebench_tasks()
    tasks = ds1000_tasks() + odex_tasks() + bcb
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    by_dataset = Counter(t["dataset"] for t in tasks)
    print("Tasks per dataset:", dict(by_dataset))
    print(f"BigCodeBench tasks whose ast imports differ from its `libs` field: {bcb_disagreements}")

    counts = library_counts(tasks)
    print("\nTasks per third-party library (top 40):")
    for pkg, c in list(counts.items())[:40]:
        print(f"  {pkg:22s} {c['total']:5d}  {c['by_dataset']}")

    rare = choose_rare(counts)
    print(f"\nRare libraries (least-used with >= {config.RARE_MIN_TASKS} tasks): "
          f"{[(p, counts[p]['total']) for p in rare]}")

    popular = {t.package: t for t in config.POPULAR_TARGETS}
    attack_libs = set(popular) | set(rare)
    group_split = assign_splits(make_strata(tasks, attack_libs, counts),
                                config.TEST_FRACTION, config.SEED)
    split = {t["task_id"]: group_split[t["split_group"]] for t in tasks}
    n_groups = len(group_split)
    print(f"\nSplit units: {n_groups:,} groups for {len(tasks):,} tasks "
          f"(DS-1000 variants kept together: {config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER})")
    task_packages = {t["task_id"]: t["packages"] for t in tasks}
    groups = group_by_library(task_packages, split, attack_libs | set(config.CONTROL_PACKAGES))

    # Targets: popular ones are fixed in config; rare fake name = most common import name + suffix.
    targets = [{"role": "popular", "import_name": t.import_name, "package": t.package,
                "fake_name": t.fake_name} for t in config.POPULAR_TARGETS]
    for pkg in rare:
        import_name = next(iter(counts[pkg]["import_names"]), pkg)
        targets.append({"role": "rare", "import_name": import_name, "package": pkg,
                        "fake_name": f"{import_name}{config.RARE_FAKE_SUFFIX}"})
    print("\nChecking that every fake name is not on PyPI:")
    statuses = check_fake_names_absent([t["fake_name"] for t in targets])
    checked_on = today_iso()
    for t in targets:
        g = groups[t["package"]]
        t.update({"pypi_status": statuses[t["fake_name"]], "pypi_checked_on": checked_on,
                  "tasks_total": counts.get(t["package"], {}).get("total", 0),
                  "tasks_by_dataset": counts.get(t["package"], {}).get("by_dataset", {}),
                  "n_proxy": len(g[PROXY]), "n_test": len(g[TEST])})

    roles = {t["package"]: t["role"] for t in targets} | {p: "control" for p in config.CONTROL_PACKAGES}
    split_counts = {ds: dict(Counter(split[t["task_id"]] for t in tasks if t["dataset"] == ds))
                    for ds in by_dataset}

    atomic_write_jsonl(config.TASKS_FILE, tasks)
    atomic_write_json(config.TASK_LIBRARY_COUNTS_FILE, counts)
    atomic_write_json(config.SPLITS_FILE, {
        "seed": config.SEED, "test_fraction": config.TEST_FRACTION, "created_on": checked_on,
        "method": "one split per task (never per library); near-copies split as one group "
                  "(ODEX rows of one SO question; DS-1000 variants of one origin problem: "
                  f"{config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER}); stratified by dataset "
                  "and library (DS-1000: its library label; ODEX/BigCodeBench: rarest target "
                  "library used)",
        "n_split_groups": n_groups,
        "counts": split_counts,
        "split": dict(sorted(split.items())),
        "groups": {lib: {"role": roles[lib], **g} for lib, g in groups.items()},
    })
    atomic_write_json(config.TARGETS_FILE, {"pypi_checked_on": checked_on, "targets": targets,
                                            "control": list(config.CONTROL_PACKAGES)})

    print("\nProxy/test per target and control library:")
    for lib, g in groups.items():
        print(f"  {lib:16s} {roles[lib]:8s} proxy={len(g[PROXY]):4d} test={len(g[TEST]):4d}")
    print("Split per dataset:", split_counts)

    save_stats("day3_tasks_splits_targets", {
        "tasks_per_dataset": dict(by_dataset), "bigcodebench_libs_field_disagreements": bcb_disagreements,
        "rare_libraries": {p: counts[p]["total"] for p in rare},
        "split_counts": split_counts, "n_split_groups": n_groups,
        "ds1000_variants_kept_together": config.SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER,
        "groups": {lib: {"role": roles[lib], "proxy": len(g[PROXY]), "test": len(g[TEST])}
                   for lib, g in groups.items()},
        "targets": targets,
    })


if __name__ == "__main__":
    main()
