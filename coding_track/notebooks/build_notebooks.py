"""Generate the Colab notebooks in this folder (run on the laptop after editing).

    python coding_track/notebooks/build_notebooks.py

Notebooks:
  week1_colab.ipynb        main notebook, Week 1 (data and index)
  week2_main_colab.ipynb   main notebook, Week 2 (retrieval, prompts, generation, scoring)
  week2_tests_colab.ipynb  separate TEST notebook: CPU runtime, no Drive, no API keys
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/MrunalKotkar/Code-Rag-Defense"
HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def write(name: str, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


def main_setup(need_ollama: bool, models: list[str]) -> str:
    return f'''
# ===== SETUP: run first in every session =====
REPO_URL = "{REPO_URL}"
REPO_DIR = "/content/repo"
NEED_OLLAMA = {need_ollama}          # True when this session runs local models
OLLAMA_MODELS = {models!r}

import os, subprocess, time
from google.colab import drive

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if "<" in REPO_URL:
    raise SystemExit("Set REPO_URL (first line of this cell) to your GitHub repo's HTTPS URL, then re-run.")

drive.mount("/content/drive")
if os.path.isdir(REPO_DIR):
    sh(f"git -C {{REPO_DIR}} pull --ff-only")
else:
    sh(f"git clone {{REPO_URL}} {{REPO_DIR}}")
os.chdir(REPO_DIR)
sh("pip install -q -r coding_track/requirements-colab.txt")

if NEED_OLLAMA:
    sh("apt-get -qq install -y pciutils zstd > /dev/null")   # pciutils helps Ollama find a GPU
    sh("curl -fsSL https://ollama.com/install.sh | sh")
    env = dict(os.environ, OLLAMA_CONTEXT_LENGTH="16384")
    subprocess.Popen(["ollama", "serve"], env=env,
                     stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(5)
    for m in OLLAMA_MODELS:
        sh(f"ollama pull {{m}}")
    sh("ollama list")
print("setup done")
'''


def week1() -> list[dict]:
    return [
        md("""
# Coding-docs RAG poisoning - Week 1 (data + index)

Run the **setup cell first in every new session**, then only the cell for today.
Every step is a script in `coding_track/steps/`; this notebook just calls them.
All outputs go to `MyDrive/coding_track/` (see `coding_track/README.md`).

| Day | Runtime to pick (Runtime > Change runtime type) | Cells |
|---|---|---|
| 1 | High-RAM CPU, setup with `NEED_OLLAMA = True` | Day 1 |
| 2-5 | **High-RAM CPU** | Day 2 ... Day 5 |
| 6 | **L4 GPU** | Day 6 (and Day 7 in the same session if time allows) |
| 7 | L4 (same session as Day 6) or High-RAM CPU | Day 7 |

Steps are safe to re-run: finished downloads, shards and files are detected and skipped.
"""),
        code(main_setup(False, ["llama3.2:3b"])),
        md("## Day 1 (Mon) - Ollama context check  \nNeeds the setup cell with `NEED_OLLAMA = True`. "
           "Expect `PASS` and `prompt_eval_count` around 9,500."),
        code("!python -m coding_track.steps.day1_ollama_check"),
        md("## Day 2 (Tue) - Download everything (High-RAM CPU)  \n"
           "Prints each dataset's real columns and row count. Stack Overflow is ~2M posts."),
        code("!python -m coding_track.steps.day2_download"),
        md("## Day 3 (Wed) - Tag tasks, pick rare libraries, split, check fake names on PyPI"),
        code("!python -m coding_track.steps.day3_tasks_splits_targets"),
        md("## Day 4 (Thu) - Fresh docs, then the Stack Overflow filter and overlap list"),
        code("!python -m coding_track.steps.day4_fresh_docs"),
        code("!python -m coding_track.steps.day4_filter_stackoverflow"),
        md("## Day 5 (Fri) - Clean, de-duplicate, manifest, then chunk"),
        code("!python -m coding_track.steps.day5_clean_and_manifest"),
        code("!python -m coding_track.steps.day5_chunk"),
        md("## Day 6 (Sat) - L4 GPU: embed in shards, then build ChromaDB on /content and zip it to Drive  \n"
           "If the session drops, run the setup cell and this cell again: finished shards are skipped."),
        code("!nvidia-smi -L\n!python -m coding_track.steps.day6_embed"),
        code("!python -m coding_track.steps.day6_build_chroma"),
        md("## Day 7 (Sun) - Count check, 20 eyeballed questions, NDCG@10, notes file"),
        code("!python -m coding_track.steps.day7_check_index"),
        md("Day 7 extra: where the top 10 come from, and NDCG@10 over library docs only."),
        code("!python -m coding_track.steps.day7_retrieval_diagnostics"),
        md("## Any time - regenerate the notes file from the saved stats"),
        code("!python -m coding_track.steps.week1_notes"),
    ]


def week2_main() -> list[dict]:
    models = ["llama3.1:8b", "qwen2.5-coder:7b", "llama3.2:3b"]
    gen = "!python -m coding_track.steps.w2_generate_ollama"
    return [
        md("""
# Coding-docs RAG poisoning - Week 2 MAIN notebook (clean baseline)

This is the **main** notebook (Drive, GPU, API key). Unit tests run in the separate
**test** notebook (`week2_tests_colab.ipynb`: CPU, no Drive, no keys).

| Step | Runtime | Cells |
|---|---|---|
| Day 4 | **L4 GPU** | setup, freeze retrieval, build prompts, pilot |
| Days 5-6 | **L4 GPU** | full runs, one model at a time; GPT-5.4 mini batch (any runtime) |
| Day 6 | any | export answers -> test notebook |
| Day 7 | any | score, hand-check sample |

Run the setup cell first in every session. Every run resumes where it stopped.
"""),
        code(main_setup(True, models)),
        md("## Day 4 - Retrieve once and freeze it (top 40 per scored question)  \n"
           "Unzips the Week 1 index from Drive the first time (takes a while)."),
        code("!python -m coding_track.steps.w2_freeze_retrieval"),
        md("## Day 3/4 - Build every prompt once (ImportSnare A.4 template)"),
        code("!python -m coding_track.steps.w2_build_prompts"),
        md("## Day 4 - Pilot: 20 questions, Llama 3.1 8B, clean RAG  \n"
           "Check: max prompt_eval_count well under 16384, code found, the time estimate. "
           "Then export the pilot answers and test them in the test notebook."),
        code(f"{gen} --model llama3.1:8b --setting clean_rag --pilot"),
        code("!python -m coding_track.steps.w2_export_for_tests --only llama3.1-8b__clean_rag "
             "--out pilot_answers.jsonl\n"
             "from google.colab import files\n"
             "files.download('/content/drive/MyDrive/coding_track/week2/pilot_answers.jsonl')"),
        md("## Days 5-6 - Full clean runs, one model at a time (order: Llama 3.1 8B, Qwen2.5-Coder 7B, "
           "Llama 3.2 3B)  \nIf the session drops: run setup, then the same cell; finished questions are skipped."),
        code(f"{gen} --model llama3.1:8b --setting no_retrieval\n{gen} --model llama3.1:8b --setting clean_rag"),
        md("Optional anchor run (compare with CodeRAG-Bench's published no-retrieval numbers on the full "
           "DS-1000 and ODEX): "),
        code(f"{gen} --model llama3.1:8b --setting no_retrieval --question-set anchor"),
        code(f"{gen} --model qwen2.5-coder:7b --setting no_retrieval\n"
             f"{gen} --model qwen2.5-coder:7b --setting clean_rag"),
        code(f"{gen} --model llama3.2:3b --setting no_retrieval\n{gen} --model llama3.2:3b --setting clean_rag"),
        md("## GPT-5.4 mini through the Batch API  \nAdd your key in Colab Secrets (key icon) as "
           "`OPENAI_API_KEY` first. Probe once, then submit both settings; collect when completed."),
        code("import os\nfrom google.colab import userdata\n"
             "os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')\n"
             "!python -m coding_track.steps.w2_openai_batch probe"),
        code("!python -m coding_track.steps.w2_openai_batch submit --setting no_retrieval\n"
             "!python -m coding_track.steps.w2_openai_batch submit --setting clean_rag"),
        code("!python -m coding_track.steps.w2_openai_batch status --setting no_retrieval\n"
             "!python -m coding_track.steps.w2_openai_batch status --setting clean_rag"),
        code("!python -m coding_track.steps.w2_openai_batch collect --setting no_retrieval\n"
             "!python -m coding_track.steps.w2_openai_batch collect --setting clean_rag"),
        md("## Day 6 - Export all answers for the test notebook"),
        code("!python -m coding_track.steps.w2_export_for_tests\n"
             "from google.colab import files\n"
             "files.download('/content/drive/MyDrive/coding_track/week2/answers_for_tests.jsonl')"),
        md("## Day 7 - After the test notebook: upload its newest `test_outputs.zip` "
           "(it is unzipped into Drive `coding_track/week2/test_results/`), then score"),
        code("import os, shutil\nfrom google.colab import files\n"
             "dst = '/content/drive/MyDrive/coding_track/week2/test_results'\nos.makedirs(dst, exist_ok=True)\n"
             "for name in files.upload():\n"
             "    if name.endswith('.zip'):\n"
             "        shutil.unpack_archive(name, dst, 'zip')\n"
             "        os.remove(name)\n"
             "    else:\n"
             "        shutil.move(name, os.path.join(dst, name))\n"
             "print(sorted(os.listdir(dst)))"),
        code("!python -m coding_track.steps.w2_score"),
        code("!python -m coding_track.steps.w2_handcheck"),
    ]


def week2_tests() -> list[dict]:
    run = "!python -m coding_track.testing.run_tests"
    return [
        md("""
# Coding-docs RAG poisoning - Week 2 TEST notebook

**Runtime: CPU (High-RAM if you have it). Do NOT mount Drive. Do NOT add API keys.**
Network stays on (some BigCodeBench tests use it). This machine only runs code and
gives back pass/fail.

Order: setup -> build environments -> (new session only: restore) -> DS-1000 Codex check ->
reference solutions, one dataset per cell -> upload answers -> score them.

Everything in this session (environments and `/content/test_outputs`) is lost when the
session ends. So every check cell ends by downloading `test_outputs.zip` with **all**
results so far. Keep the newest zip. In a new session: setup, build environments, then
the restore cell with that zip, and continue where you stopped.
"""),
        code(f'''
REPO_URL = "{REPO_URL}"
import os, shutil, subprocess
from google.colab import files
if not os.path.isdir("/content/repo"):
    subprocess.run(["git", "clone", REPO_URL, "/content/repo"], check=True)
else:
    subprocess.run(["git", "-C", "/content/repo", "pull", "--ff-only"], check=True)
os.chdir("/content/repo")
OUTPUTS = "/content/test_outputs"

def save_outputs():
    """Download everything in /content/test_outputs as one zip (all results so far)."""
    if not os.path.isdir(OUTPUTS) or not os.listdir(OUTPUTS):
        print("nothing to save yet")
        return
    shutil.make_archive("/content/test_outputs", "zip", OUTPUTS)
    print("saving:", sorted(os.listdir(OUTPUTS)))
    files.download("/content/test_outputs.zip")

print("ready")
'''),
        md("## Day 1-2 - Build the three environments (Python 3.10 each; takes a while)"),
        code("!python -m coding_track.testing.build_envs ds1000 bcb odex"),
        md("## New session only - restore the newest `test_outputs.zip` you downloaded\n"
           "Skip this in the first session."),
        code("uploaded = list(files.upload())\n"
             "shutil.unpack_archive(uploaded[0], OUTPUTS, 'zip')\n"
             "print(sorted(os.listdir(OUTPUTS)))"),
        md("## Day 1 - DS-1000 check: reproduce Codex-002 (expected overall 0.388)"),
        code(f"{run} codex\nsave_outputs()"),
        md("## Day 2 - Official reference solutions, one dataset per cell\n"
           "Each cell adds its dataset to `reference_results.jsonl` (the others are kept) "
           "and downloads the zip."),
        code(f"{run} reference --datasets ds1000\nsave_outputs()"),
        code(f"{run} reference --datasets odex\nsave_outputs()"),
        code(f"{run} reference --datasets bigcodebench\nsave_outputs()"),
        md("## Pilot / full scoring - upload the answers file exported by the main notebook"),
        code("uploaded = list(files.upload())\nprint(uploaded)"),
        code(f"{run} answers --input {{uploaded[0]}}\nsave_outputs()"),
    ]


if __name__ == "__main__":
    write("week1_colab.ipynb", week1())
    write("week2_main_colab.ipynb", week2_main())
    write("week2_tests_colab.ipynb", week2_tests())
