"""Pinned test environments and data sources for the test runtime (checked 2026-11-13).

Runs on: the separate Colab CPU "test" notebook only. No Drive, no API keys.
Everything here comes from the benchmarks' own repositories; nothing is guessed.
"""

from __future__ import annotations

import os
from pathlib import Path

# Overridable for local testing; on Colab the defaults are used.
ENVS_DIR = Path(os.environ.get("CODING_TRACK_ENVS", "/content/envs"))
OUTPUT_DIR = Path(os.environ.get("CODING_TRACK_TEST_OUTPUT", "/content/test_outputs"))
UV_VERSION = "0.13.6"
BANDIT_VERSION = "1.9.4"
# setuptools 82.0.0 removed pkg_resources ("environments relying on pkg_resources
# ... should depend on older versions of setuptools", setuptools NEWS). All three
# benchmark environments predate it, and ODEX's obspy imports pkg_resources.
SETUPTOOLS_PIN = "setuptools<82"

# ---------------------------------------------------------------------------
# DS-1000: github.com/xlang-ai/DS-1000 @ b39aab7 (environment.yml)
# conda's "yaml" entry is the LibYAML C library, not a Python package, so it
# has no pip equivalent. pytorch::cpuonly + pytorch==2.2.0 -> torch 2.2.0 from
# PyTorch's CPU wheel index.
# ---------------------------------------------------------------------------
DS1000_REPO_COMMIT = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"
DS1000_PYTHON = "3.10"
DS1000_TORCH = ("torch==2.2.0", "https://download.pytorch.org/whl/cpu")
DS1000_PACKAGES = (
    "gensim==4.3.2", "matplotlib==3.8.4", "numpy==1.26.4", "pandas==1.5.3",
    "seaborn==0.13.2", "scikit-learn==1.4.0", "scipy==1.12.0", "statsmodels==0.14.1",
    "xgboost==2.0.3", "tensorflow-cpu==2.16.1",
)
DS1000_TIMEOUT_S = 120          # test_ds1000.py uses timeout=120
DS1000_HF = ("xlangai/DS-1000", "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44", "test.jsonl")
DS1000_CODEX_ANSWERS_URL = (
    f"https://raw.githubusercontent.com/xlang-ai/DS-1000/{DS1000_REPO_COMMIT}"
    "/data/codex002-answers.jsonl")
# Expected Codex-002 scores, DS-1000 README "Expected output".
DS1000_CODEX_EXPECTED = {
    "Matplotlib": 0.548, "Numpy": 0.432, "Pandas": 0.265, "Pytorch": 0.397,
    "Scipy": 0.349, "Sklearn": 0.435, "Tensorflow": 0.378, "overall": 0.388,
}

# ---------------------------------------------------------------------------
# BigCodeBench: github.com/bigcode-project/bigcodebench, Docker/Evaluate.Dockerfile
# (python:3.10-slim). Install order copied from that Dockerfile: numpy/pyarrow,
# the package without dependencies, its runtime dependencies, then the pinned
# evaluation requirements with -I, then datasets==2.17.0. The package is the
# PyPI release 0.2.5 (its evaluator scores dataset split v0.1.4, our pinned
# version); e2b is capped as in the repo's current setup.cfg.
# ---------------------------------------------------------------------------
BCB_PYTHON = "3.10"
BCB_STEPS = (
    ["numpy==1.24.3", "pyarrow==14.0.1"],
    ["--no-deps", "bigcodebench==0.2.5"],
    ["appdirs", "fire", "multipledispatch", "pqdm", "tempdir", "termcolor", "tqdm",
     "tree_sitter", "tree-sitter-python", "wget", "transformers", "datasets", "gradio-client",
     "numpy", "rich", "accelerate", "anthropic", "google-genai", "mistralai<1.0.0",
     "openai", "e2b<=1.11.1"],
    ["-I", "-r", "https://raw.githubusercontent.com/bigcode-project/bigcodebench/"
     "36a905d2fd46761f3269b2d48c51b8a572971f73/Requirements/requirements-eval.txt"],
    ["datasets==2.17.0"],
)
BCB_APT_PACKAGES = ("g++", "python3-tk", "zip", "unzip", "procps", "r-base", "libgdal-dev",
                    "libfreetype6-dev", "libpng-dev", "pkg-config", "python3-dev")
BCB_SPLIT = "instruct"
BCB_SUBSET = "full"

# ---------------------------------------------------------------------------
# ODEX: the third-party modules its prompts, reference solutions and tests
# import (read from the data: pandas, numpy, matplotlib, requests, bs4, flask,
# django, scipy, mock, multidict, regex, sqlalchemy, obspy, pytz).
# ODEX's library/requirements.txt pins only sqlalchemy==1.3.5. Libraries that
# DS-1000 also uses get DS-1000's pins; the rest are unpinned and recorded by
# `pip freeze` when the environment is built.
# ---------------------------------------------------------------------------
ODEX_PYTHON = "3.10"
ODEX_PACKAGES = (
    "numpy==1.26.4", "pandas==1.5.3", "matplotlib==3.8.4", "scipy==1.12.0",
    "sqlalchemy==1.3.5",
    "requests", "beautifulsoup4", "flask", "django", "mock", "multidict", "regex",
    "obspy", "pytz",
    "html5lib",   # parser named by string in one test: BeautifulSoup(..., 'html5lib')
)
ODEX_TIMEOUT_S = 60
ODEX_HF = ("neulab/odex", "c3741a66c486b1a23beefdf6c75b06dba288d4f9", "en_test.jsonl")


def env_python(name: str) -> Path:
    if os.name == "nt":                       # local Windows testing only
        return ENVS_DIR / name / "Scripts" / "python.exe"
    return ENVS_DIR / name / "bin" / "python"


def hf_file_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
