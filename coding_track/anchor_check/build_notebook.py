"""Generate anchor_check_colab.ipynb (run on the laptop after editing).

    python coding_track/anchor_check/build_notebook.py

One notebook for the whole anchor check: generation (L4 GPU + Ollama), then the DS-1000
tests, then saving. It does not touch the Week 1 / Week 2 notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/MrunalKotkar/Code-Rag-Defense"
HERE = Path(__file__).parent
MODEL = "llama3.1:8b"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


SETUP = f'''
# ===== STEP 1 - SETUP (run first in every session) =====
REPO_URL = "{REPO_URL}"
REPO_DIR = "/content/repo"
MODEL = "{MODEL}"

import os, subprocess, time
from google.colab import drive

def sh(cmd, tries=1):
    """Run a shell command, print the end of its output; retry network steps."""
    print("$", cmd, flush=True)
    for attempt in range(1, tries + 1):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
        lines = (proc.stdout + proc.stderr).replace("\\r", "\\n").splitlines()
        tail = [l for l in lines if l.strip()][-15:]
        if tail:
            print("\\n".join(tail))
        if proc.returncode == 0:
            return
        print(f"!! exit status {{proc.returncode}} (try {{attempt}} of {{tries}})", flush=True)
        if attempt < tries:
            time.sleep(15)
    raise RuntimeError(f"command failed: {{cmd}}")

drive.mount("/content/drive")
if os.path.isdir(REPO_DIR):
    sh(f"git -C {{REPO_DIR}} pull --ff-only")
else:
    sh(f"git clone {{REPO_URL}} {{REPO_DIR}}")
os.chdir(REPO_DIR)
sh("pip install -q ollama==0.6.3")      # same client version as Week 2

sh("apt-get -qq install -y pciutils zstd > /dev/null")     # pciutils helps Ollama find the GPU
sh("curl -fsSL https://ollama.com/install.sh | sh", tries=3)
env = dict(os.environ, OLLAMA_CONTEXT_LENGTH="16384")     # same server setting as Week 2
subprocess.Popen(["ollama", "serve"], env=env,
                 stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
sh(f"ollama pull {{MODEL}}", tries=3)
sh("ollama list")
sh("nvidia-smi -L")
print("setup done")
'''

SAVE_AND_UNMOUNT = '''
import shutil
os.makedirs("/content/drive/MyDrive/coding_track/anchor_check", exist_ok=True)
for name in os.listdir("/content/coding_track_local/anchor_check"):
    shutil.copy(f"/content/coding_track_local/anchor_check/{name}",
                f"/content/drive/MyDrive/coding_track/anchor_check/{name}")
print("saved to Drive:", sorted(os.listdir("/content/drive/MyDrive/coding_track/anchor_check")))
drive.flush_and_unmount()
print("Drive unmounted. It stays unmounted while the model's code runs.")
'''

SAVE_RESULTS = '''
import shutil
drive.mount("/content/drive")
dst = "/content/drive/MyDrive/coding_track/anchor_check"
os.makedirs(dst, exist_ok=True)
for name in os.listdir("/content/coding_track_local/anchor_check"):
    shutil.copy(f"/content/coding_track_local/anchor_check/{name}", f"{dst}/{name}")
print("saved to Drive:", sorted(os.listdir(dst)))

# A zip as well, in case the Drive copy fails.
from google.colab import files
shutil.make_archive("/content/anchor_check_results", "zip", "/content/coding_track_local/anchor_check")
files.download("/content/anchor_check_results.zip")
'''

CELLS = [
    md(f"""
# Anchor check - DS-1000 with CodeRAG-Bench's own prompt (one notebook)

**Why.** Week 2's anchor (Llama 3.1 8B, no retrieval, ImportSnare chat prompt) scored DS-1000
**15.7%** against CodeRAG-Bench Table 5's **28.9%**. CodeRAG-Bench does not use a chat prompt for
DS-1000: it sends DS-1000's own prompt text unchanged and lets the model continue it (stops at
`</code>`). This notebook repeats the anchor that way, with the same model (`{MODEL}`), temperature 0,
seed 100, and the Week 2 test harness unchanged. It does not read or write any Week 1 / Week 2 file.
Outputs go to `MyDrive/coding_track/anchor_check/`.

**Runtime: L4 GPU** (Runtime > Change runtime type). Total time about 1 to 1.5 hours.

| Step | What | Time |
|---|---|---|
| 1 | Setup (mount Drive, repo, Ollama, model) | ~5-10 min |
| 2 | Trial: 20 problems, look at the answers | ~2 min |
| 3 | Full generation: 1000 problems | ~35-60 min |
| 4 | Save to Drive, then **unmount Drive** | seconds |
| 5 | Build the DS-1000 test environment | ~10 min |
| 6 | Run the 1000 answers, print pass@1 | ~10-30 min |
| 7 | Mount Drive again, save results (also downloads a zip) | seconds |

If the session drops: run step 1 and the cell you were on. Generation resumes; finished problems are skipped.
Steps 5-6 run code written by a model, so Drive is **unmounted** for them and no API keys are in this notebook.
"""),
    code(SETUP),
    md("## STEP 2 - Trial on 20 problems  \nCheck the printed first answer: the raw text should be code only (no \"Sure, here is...\"), "
       "and \"empty code\" should be near 0."),
    code("!python -m coding_track.anchor_check.anchor generate --model {MODEL} --limit 20".replace("{MODEL}", MODEL)),
    md("## STEP 3 - Full generation (all 1000 DS-1000 problems)  \nResumable. Copies to Drive every 50 answers."),
    code("!python -m coding_track.anchor_check.anchor generate --model {MODEL}".replace("{MODEL}", MODEL)),
    md("## STEP 4 - Save to Drive, then unmount Drive  \nDo not skip the unmount: the next steps execute model-written code."),
    code(SAVE_AND_UNMOUNT),
    md("## STEP 5 - Build the DS-1000 test environment (Python 3.10, DS-1000's pins)  \n"
       "Same builder as Week 2, only the `ds1000` environment."),
    code("!python -m coding_track.testing.build_envs ds1000"),
    md("## STEP 6 - Run the answers and print pass@1  \nPrints the result next to 28.9% (published) and 15.7% (Week 2, chat prompt)."),
    code(f"!python -m coding_track.anchor_check.anchor score --model {MODEL}"),
    md("## STEP 7 - Mount Drive again and save the results  \nA Google permission popup appears again."),
    code(SAVE_RESULTS),
]


def main() -> None:
    nb = {"cells": CELLS, "metadata": {"colab": {"provenance": [], "gpuType": "L4"},
                                       "accelerator": "GPU",
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = HERE / "anchor_check_colab.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
