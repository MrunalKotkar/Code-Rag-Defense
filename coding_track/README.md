# coding_track

Code for the coding-docs RAG poisoning track (plan: `Coding-Docs RAG Poisoning
Full Requirements and Setup.pdf`, Nov 7, 2026). Week 1 builds the clean search
index and the question splits.

## Where things run

| Where | What |
|---|---|
| **Laptop** | Write code, run the unit tests (`pytest`), push to GitHub. No data lives here. |
| **Colab, High-RAM CPU** | Days 1-5: Ollama check, downloads, tagging/splits, fresh docs, Stack Overflow filter, cleaning, chunking. |
| **Colab, L4 GPU** | Day 6: embedding with mGTE, building ChromaDB. Day 7 can run in the same session. |
| **Google Drive** `MyDrive/coding_track/` | Every output (raw downloads, derived files, shards, index zip, stats, notes). |
| **Colab local disk** `/content/coding_track_local/` | Fast scratch: local copies of datasets, the ChromaDB being built. Lost when the session ends. |

Open `notebooks/week1_colab.ipynb` in Colab, run the setup cell, then the cell for the day.

## Layout

```
coding_track/
  config.py                  every fixed setting: paths, seed, chunk size, model, targets, thresholds
  requirements-colab.txt     pinned packages for Colab (see the comment about transformers 4.x)
  notebooks/week1_colab.ipynb  setup cell + one cell per day
  common/                    reusable helpers (no step logic)
    io_utils.py              atomic writes, JSON/JSONL, stats files, HTTP
    hf_data.py               download once / save to Drive / load via local copy
    imports.py               read imports with ast, stdlib check, import -> PyPI name
    question_ids.py          unique ODEX row ids
    splits.py                the stratified 80/20 split and grouping by library
    so_filter.py             which SO posts mention a library / are Python
    minhash_utils.py         MinHash signatures and LSH (datasketch 2.x)
    text_clean.py            HTML -> text with fenced code, whitespace tidying
    chunking.py              1024-token chunks that never cut a code block
    retrieval.py             mGTE loading/embedding, ChromaDB helpers
    metrics.py               NDCG@k
  steps/                     one script per step: python -m coding_track.steps.<name>
    day1_ollama_check.py
    day2_download.py
    day3_tasks_splits_targets.py
    day4_fresh_docs.py
    day4_filter_stackoverflow.py
    day5_clean_and_manifest.py
    day5_chunk.py
    day6_embed.py
    day6_build_chroma.py
    day7_check_index.py
    week1_notes.py           builds notes/week1_numbers.md from the stats files
  tests/                     unit tests for common/ (run on the laptop)
  docs/                      drafts (e.g. the Monday update)
```

## Week 1 outputs on Drive (`MyDrive/coding_track/`)

| Path | Made by | What |
|---|---|---|
| `raw/corpus/*`, `raw/questions/*` | day2 | the 4 corpora, 3 question sets, 2 label sets (`save_to_disk`) |
| `derived/tasks.jsonl` | day3 | one line per task: id, dataset, imports, packages, texts |
| `derived/task_library_counts.json` | day3 | tasks per library |
| `derived/splits.json` | day3 | the 80/20 split + tasks grouped by library |
| `derived/targets.json` | day3 | targets, fake names, PyPI 404 check and date |
| `fresh/` | day4 | raw DevDocs `db.json` and PyPI JSON + `fresh_index.json` (URL, version, date) |
| `derived/stackoverflow_filtered/` | day4 | kept SO posts with `libraries`, `so_role` |
| `derived/stackoverflow_overlap.csv` | day4 | posts that nearly copy a task (for Prof. Stamp) |
| `derived/clean_docs.parquet` | day5 | cleaned documents with text |
| `derived/manifest.jsonl` | day5 | one line per document (id, source, url, library, version, date, license, is_poison) |
| `derived/dedup_dropped.jsonl` | day5 | every dropped document and why |
| `derived/chunks.parquet` | day5 | chunks + metadata |
| `embeddings/shard_*.npz`, `spec.json` | day6 | embedding shards |
| `index/chroma_clean_index.zip` | day6 | the ChromaDB index |
| `derived/eyeball_top10.txt` | day7 | 20 test questions with their top 10 |
| `stats/*.json` | every step | the real numbers |
| `notes/week1_numbers.md` | day7 / week1_notes | the notes file with all real numbers |

## Week 2: clean baseline

Two Colab notebooks, on purpose:

| Notebook | Runtime | Has | Runs |
|---|---|---|---|
| `notebooks/week2_main_colab.ipynb` | L4 GPU (CPU for scoring) | Drive, OpenAI key | retrieval freeze, prompts, Ollama + GPT runs, export, scoring |
| `notebooks/week2_tests_colab.ipynb` | CPU | **no Drive, no keys** (network on) | the three test environments, DS-1000 Codex check, reference solutions, answer tests, Bandit |

Files moved between them by download/upload: `answers_for_tests.jsonl` (main -> test)
and `test_outputs.zip` (test -> main, unzipped into `week2/test_results/`).
The test notebook has no Drive, so every check cell downloads `test_outputs.zip` with all
results so far; after a disconnect, rebuild the environments and restore the newest zip.
`run_tests reference` runs one dataset per cell and saves after each one.

New code:
```
common/prompt.py          ImportSnare A.4 prompt (verbatim) + "Return only code, in one python block."
common/answers.py         code-block extraction, ast import reader (regex fallback), pip-install finder
common/question_sets.py   scored set (test split, attacked/control library) and anchor set
common/generation.py      one saved line per answer; resume + copy to Drive every 50
steps/w2_freeze_retrieval.py   top-40 chunks per scored question, frozen to a file
steps/w2_build_prompts.py      every prompt built once (no_retrieval, clean_rag)
steps/w2_generate_ollama.py    one local model at a time (temp 0, seed 100, num_ctx 16384)
steps/w2_openai_batch.py       GPT-5.4 mini: probe, submit, status, collect (Batch API)
steps/w2_export_for_tests.py   answers -> test notebook
steps/w2_score.py              baseline table + the three pre-Week-3 checks
steps/w2_handcheck.py          50 random answers for a manual check
testing/env_spec.py       pinned environments, each pin with its source
testing/build_envs.py     uv venvs (Python 3.10) + pip installs, pip freeze saved
testing/exec_program.py   runs one test program in a fresh process
testing/harness.py        DS-1000 / ODEX programs, BigCodeBench official evaluator
testing/run_tests.py      modes: codex, reference, answers
```

Week 2 outputs on Drive (`MyDrive/coding_track/week2/`): `retrieval_top40.jsonl`,
`retrieval_chunks.jsonl`, `prompts/`, `generations/`, `openai_batch/`,
`answers_for_tests.jsonl`, `test_results/`, `baseline_table.md`, `handcheck_50.md`.

Week 2 decisions (and why):
* **Prompt:** ImportSnare Appendix A.4, word for word (the paper's "don´t" typesetting artifact
  written as "don't"), plus the one line "Return only code, in one python block.".
  The same template is used with an empty context for no retrieval.
* **BigCodeBench Instruct split.** ImportSnare does not say which split it used.
* **ODEX prompt** follows ODEX's own nl2code format (header, intent docstring, body prefix).
  ODEX scoring runs **all** test cases (ODEX's script samples one); a full-function answer
  is used as is, otherwise it is treated as a completion (prompt + answer + suffix).
* **DS-1000** scoring copies DS-1000's test_ds1000.py program and its 120 s time limit.
  **BigCodeBench** uses its official evaluator (`bigcodebench.evaluate`, local, calibrated).
* **Scored set:** test-split questions using an attacked or control library (364:
  DS-1000 199, BigCodeBench 148, ODEX 17). The optional **anchor** run (Llama 3.1 8B,
  no retrieval, all DS-1000 + ODEX) is only for comparing with CodeRAG-Bench Table 5
  (Llama3-8B w/o: DS-1000 28.9, ODEX 37.4).
* **GPT-5.4 mini parameters are probed, not assumed**: temperature 0, seed 100 and
  reasoning effort "none" are sent; anything the API rejects is dropped and recorded.
* **setuptools<82** in every test environment (82.0.0 removed pkg_resources; the
  benchmark environments predate it; ODEX's obspy needs it).

## Testing on the laptop

```
python -m venv .venv
.venv/Scripts/python -m pip install -r coding_track/requirements-colab.txt torch pytest
.venv/Scripts/python -m pytest coding_track/tests -q
```

Any step can run locally against scratch folders by setting
`CODING_TRACK_DRIVE` and `CODING_TRACK_LOCAL`.

## Decisions made while building (and why)

* **Pinned transformers 4.57.6 / sentence-transformers 5.7.0.** mGTE's custom
  model code breaks under transformers 5.x (garbage position ids). With these
  pins it reproduces its model card's similarity scores exactly.
* **ODEX is loaded from its `en_test.jsonl` file**, because the repo's loading
  script is refused by `datasets` >= 4.
* **ODEX row ids are `<SO question id>_<k>`.** ODEX has 439 rows but 333 distinct
  task ids (several intents per Stack Overflow question).
* **Near-copies share one side of the split.** ODEX rows of one SO question are
  always split together; DS-1000 variants of one origin problem (1,000 tasks
  from 453 problems) are split together while
  `SPLIT_KEEP_DS1000_PERTURBATIONS_TOGETHER = True`. Set it to False for a plain
  per-task split.
* **Split is stratified** (DS-1000 by its library label; ODEX/BigCodeBench by the
  rarest target library used) so every target, including 6-task rare ones, gets
  test tasks.
* **Where task libraries come from:** ast imports of the reference program;
  DS-1000 also adds its own library label; ODEX uses its `library` field
  (its solutions are bare expressions; imports exist only in the test code).
* **Stack Overflow "Python post" and "mentions" are text heuristics** (the SO
  set has no tags): see `common/so_filter.py`. `requests`, `regex`, `rsa` count
  only via import / `pip install` lines.
* **Labelled library docs are protected** from the short-page and duplicate
  filters, so NDCG@10 can be measured against them.
* **Short-page floor is 5 words**; 20 words dropped ~30% of real API entries.
* **ChromaDB rejects `None` metadata**, so a missing URL is stored as `""`.
