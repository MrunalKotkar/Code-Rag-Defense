"""Text cleaning: HTML to text (code blocks kept as ``` fences) and whitespace tidying.

Only real HTML sources (DevDocs pages) go through `html_to_text`. The Hugging
Face corpora are already plain text; running an HTML parser on them would eat
literal tags that appear in docs about HTML or templates.
"""

from __future__ import annotations

import re

_BLOCK_TAGS = (
    "p", "div", "section", "article", "header", "footer", "table", "tr", "ul", "ol", "dl",
    "li", "dt", "dd", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "br",
)
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PLACEHOLDER = "\x00CODEBLOCK{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00CODEBLOCK(\d+)\x00")


def fence(code: str, language: str = "") -> str:
    return f"```{language}\n{code.strip(chr(10))}\n```"


def html_to_text(html: str) -> str:
    """Convert an HTML page to text. Each <pre> becomes a fenced code block."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    code_blocks: list[str] = []
    for pre in soup.find_all("pre"):
        language = (pre.get("data-language") or "").strip()
        code_blocks.append(fence(pre.get_text(), language))
        pre.replace_with(f"\n\n{_PLACEHOLDER.format(len(code_blocks) - 1)}\n\n")

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n")
        if tag.name not in ("br", "hr"):
            tag.insert_before("\n")

    prose = soup.get_text()
    lines = [_SPACES.sub(" ", line).strip() for line in prose.split("\n")]
    text = _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
    return _PLACEHOLDER_RE.sub(lambda m: code_blocks[int(m.group(1))], text)


def normalize_text(text: str) -> str:
    """Tidy plain text without touching indentation (code must survive)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def collapse_whitespace(text: str) -> str:
    """Single-spaced version used for exact-duplicate hashing."""
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(text.split())


def stackoverflow_question(post: str) -> str:
    """The question part of a CodeRAG-Bench SO post ('Q: ...\\n\\nA: ...')."""
    question = re.split(r"\n\s*\nA:", post, maxsplit=1)[0]
    return question.removeprefix("Q:").strip()
