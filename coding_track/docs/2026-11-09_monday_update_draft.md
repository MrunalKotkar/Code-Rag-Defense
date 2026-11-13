Subject: Weekly update (Nov 9) - coding-docs track, Week 1 (data and index)

Dear Prof. Stamp,

Here is my update on the coding-docs track. Week 1 builds the data and the
search index. Most of it is done.

## 1. Status

- Done: setup, downloads, question tagging, 80/20 split, target list, fresh
  docs, Stack Overflow filter, cleaning, chunking, embedding.
- Left this week: load the index into ChromaDB, then check it (count check,
  20 sample questions, retrieval quality NDCG@10).
- Everything runs on Colab. Data and results are on Google Drive. Code is on GitHub.

## 2. The dataset in short

The benchmark has four parts:

- Documents: what the coding assistant searches.
- Questions: coding tasks with unit tests.
- Targets: the libraries we attack, each with a fake package name.
- Split: 80% of questions the attacker may see, 20% kept hidden for scoring.

### 2a. Documents (the search corpus)

From CodeRAG-Bench, plus a fresh slice I collected:

- Library documentation: 34,003 official API pages (pandas, numpy, matplotlib,
  sklearn, Python standard library, and others).
- Online tutorials: 76,643 how-to articles.
- Programming solutions: 1,128 short Python programs (HumanEval and MBPP).
- Stack Overflow: 23,539,018 posts. The plan's 1.97M figure was wrong; this
  matches the paper's 23.5M.
  - I kept 548,001 posts that mention a target library.
  - I added 548,001 random other Python posts as background.
- Fresh slice: 9,931 current DevDocs pages and 11 PyPI READMEs.
  - Scipy, pillow, regex and rsa are not on DevDocs. They have only their README.

After cleaning:

- Duplicates and near-empty pages removed: 1,207,822 documents remain.
- Cut into 1,812,241 chunks of at most 1,024 tokens. No code block is cut in half.
- About 88% of the chunks are Stack Overflow posts.
- A manifest lists every document: source, URL, library, version, date,
  license, and is_poison (false for all).

### 2b. Questions

Every question has a reference solution and unit tests.

- DS-1000: 1,000 data-science questions from Stack Overflow, 7 libraries.
- ODEX: 439 short Python tasks, also from Stack Overflow.
- BigCodeBench: 1,140 longer tasks using many libraries. ImportSnare took its
  Python questions from this set.
- Total: 2,579 questions.
- I tagged each question with the libraries its solution imports.

### 2c. Targets and fake names

- Popular targets (fake name):
  - pandas (pandas_v2)
  - numpy (numpy_secure)
  - matplotlib (matplotlib_safe)
  - scikit-learn (sklearn_robust)
  - scipy (full_scipy)
  - requests (requests_secure)
- Rare targets: pillow, regex, rsa (6 tasks each), flask and statsmodels
  (8 each). Fake name = name + "_secure".
- Control libraries, never attacked: PyTorch and TensorFlow.
- I checked all 11 fake names on PyPI. None of them exists. I will not
  register any of them.

### 2d. Split

- Proxy (attacker may see): 2,066 questions.
- Test (hidden, used for scoring): 513 questions.
- Each question is split once, never per library.
- Test questions per target:
  - pandas 207, numpy 219, matplotlib 100
  - sklearn 57, scipy 41, requests 9
  - each rare library 1 or 2

## 3. Real numbers that replace estimates in my plan

- Ollama context check: passed. It read the full 9,538-token prompt with
  num_ctx 16384.
- Stack Overflow set: 23.5M posts, not 1.97M.
- Corpus: 1,207,822 documents and 1,812,241 chunks.
- Embedding time: 4.56 hours on an L4, about 7.8 Colab units. The plan
  guessed 3 to 8 hours.

## 4. Questions for you

1. Adding BigCodeBench.
   I added BigCodeBench (1,140 questions) next to DS-1000 and ODEX.
   ImportSnare used it, so our comparison is closer to the paper.
   Is that okay?

2. Stack Overflow posts that copy a test question.
   DS-1000 and ODEX were built from Stack Overflow, so some original posts
   are also in our search corpus. For those questions the model can just
   copy the answer.
   - 107 such matches involve test questions (65 DS-1000, 42 ODEX).
   - I can remove these posts without redoing the embedding.
   Should I remove them, or keep them because a real corpus would have them?

3. Budget.
   Paid models and extra Colab units should cost about $40 to $85.
   Nothing has been spent on paid models yet.
   Is this amount okay, and is there funding?

4. Keeping reworded questions together in the split.
   Many questions are rewordings of each other (DS-1000 has 453 originals
   among its 1,000). If an original is in the test set and its rewording is
   in the attacker's set, the attacker has in effect seen the test question.
   So I kept each original and its rewordings on the same side.
   The test set is still about 20%, but not exactly the counts in my plan.
   Is this okay? Switching to a plain per-question split is easy.

## 5. Next steps

- This week: build the ChromaDB index, check it, and record NDCG@10 on DS-1000
  and ODEX.
- Week 2: build the three test environments, check that the reference
  solutions pass, and run the clean baselines (no retrieval, and clean
  retrieval) for the local models and GPT-5.4 mini.

Best regards,
Mrunal
