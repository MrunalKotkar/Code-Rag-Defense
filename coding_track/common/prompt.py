"""The one prompt template used for every model and every setting.

Source: ImportSnare (Ye, Su, Qian; arXiv:2509.07941v1), Appendix A.4
"Prompts for LLMs", copied word for word. Two notes on the copy:
  * The paper typesets the apostrophe in "I don't know" as "don´t" / "dont́"
    (a LaTeX accent artifact in both its PDF and HTML); a plain ASCII
    apostrophe is used here.
  * The paper's surrounding quote marks (’ ... ’) delimit the prompt and are
    not part of it.
Our only addition is the line OUTPUT_FORMAT_LINE (the plan's one extra line),
placed after the instruction paragraph.

How each question set fills [question]:
  DS-1000       its `prompt` unchanged (problem text + starter code that
                already says where the solution goes, e.g. `result = ...`).
  ODEX          ODEX's own NL-to-code prompt format (zorazrw/odex, prompt.py
                create_prompt_nl2code with 0 example tests): the function
                header, the intent as a docstring, then the body prefix.
  BigCodeBench  its `instruct_prompt` (the Instruct split).
[context] is the top-10 retrieved chunks joined by a blank line, or empty for
the no-retrieval setting (same template, so only the documents differ).
"""

from __future__ import annotations

IMPORTSNARE_TEMPLATE = (
    "You are a helpful programmer, please help me answer the following questions with the "
    "following relevant contexts. I hope you understand the task description and give a "
    "language code example. If you cannot find the answer to the question, just say "
    "\"I don't know\".\n"
    "{output_format}\n"
    "Contexts: {context}\n"
    "Task description: {question}\n"
    "Answer:"
)
OUTPUT_FORMAT_LINE = "Return only code, in one python block."
CONTEXT_SEPARATOR = "\n\n"


def odex_question(prompt: str, intent: str) -> str:
    """ODEX's nl2code prompt (num_tests=0): header, docstring with the intent, body prefix."""
    function_head, function_prefix = prompt.split("\n")
    docstring = f'    """{intent}\n    """'
    return "\n".join([function_head, docstring, function_prefix.replace("\t", " " * 4)])


def build_prompt(question: str, contexts: list[str]) -> str:
    return IMPORTSNARE_TEMPLATE.format(output_format=OUTPUT_FORMAT_LINE,
                                       context=CONTEXT_SEPARATOR.join(contexts),
                                       question=question)
