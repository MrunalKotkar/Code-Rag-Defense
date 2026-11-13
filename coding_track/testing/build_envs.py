"""Build the three test environments (DS-1000, BigCodeBench, ODEX) on the test runtime.

Runs on: the separate Colab CPU test notebook. No Drive, no keys.

Each environment is a Python 3.10 venv made by uv (`uv venv --seed`, so it has
its own pip). Packages are installed with that environment's pip, following
the pins in env_spec.py. Colab's own Python is newer than 3.10, so tests must
always use these environments' Python, never `python`.
After building, `pip freeze` of each environment is saved to
/content/test_outputs/env_<name>_freeze.txt so the exact versions are on record.

Usage:  python -m coding_track.testing.build_envs [ds1000 bcb odex]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from coding_track.testing import env_spec


def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_venv(name: str, python_version: str) -> list[str]:
    target = env_spec.ENVS_DIR / name
    if not env_spec.env_python(name).exists():
        sh([sys.executable, "-m", "uv", "venv", "--seed", "--python", python_version, str(target)])
    pip = [str(env_spec.env_python(name)), "-m", "pip", "install", "--quiet"]
    sh(pip + [env_spec.SETUPTOOLS_PIN])
    return pip


def freeze(name: str) -> None:
    env_spec.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([str(env_spec.env_python(name)), "-m", "pip", "freeze"],
                         check=True, capture_output=True, text=True).stdout
    (env_spec.OUTPUT_DIR / f"env_{name}_freeze.txt").write_text(out)
    print(f"[{name}] {len(out.splitlines())} packages recorded")


def build_ds1000() -> None:
    pip = make_venv("ds1000", env_spec.DS1000_PYTHON)
    torch_req, torch_index = env_spec.DS1000_TORCH
    sh(pip + ["--index-url", torch_index, torch_req])
    sh(pip + list(env_spec.DS1000_PACKAGES))
    freeze("ds1000")


def build_bcb() -> None:
    sh(["apt-get", "-qq", "install", "-y", *env_spec.BCB_APT_PACKAGES])
    pip = make_venv("bcb", env_spec.BCB_PYTHON)
    for step in env_spec.BCB_STEPS:
        sh(pip + step)
    freeze("bcb")


def build_odex() -> None:
    pip = make_venv("odex", env_spec.ODEX_PYTHON)
    sh(pip + list(env_spec.ODEX_PACKAGES))
    freeze("odex")


BUILDERS = {"ds1000": build_ds1000, "bcb": build_bcb, "odex": build_odex}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the test environments.")
    parser.add_argument("envs", nargs="*", choices=sorted(BUILDERS), default=list(BUILDERS))
    args = parser.parse_args()
    sh([sys.executable, "-m", "pip", "install", "--quiet", f"uv=={env_spec.UV_VERSION}",
        f"bandit=={env_spec.BANDIT_VERSION}"])
    for name in args.envs:
        print(f"\n=== building {name}")
        BUILDERS[name]()


if __name__ == "__main__":
    main()
