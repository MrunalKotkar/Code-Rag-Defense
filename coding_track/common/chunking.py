"""Cut documents into chunks of at most CHUNK_TOKENS tokens.

Rules (from the plan):
  * count tokens with the retriever's own tokenizer (mGTE);
  * split only at paragraph or code-block edges;
  * never cut through a fenced ``` code block.

The one unavoidable exception: a single code block longer than the budget on
its own. It is split at line edges and each piece is re-fenced, so it stays
marked as code. Such pieces are flagged (`from_split_code`) and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

_FENCE_RE = re.compile(r"^```[^\n]*\n.*?^```[ \t]*$", re.M | re.S)
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")
BLOCK_JOINER = "\n\n"
LINE_JOINER = "\n"


class TokenCounter(Protocol):
    def count(self, texts: Sequence[str]) -> list[int]: ...
    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]: ...


class HFTokenCounter:
    """TokenCounter backed by a Hugging Face fast tokenizer (no special tokens)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        offsets = self.tokenizer(text, add_special_tokens=False,
                                 return_offsets_mapping=True)["offset_mapping"]
        starts = [offsets[i][0] for i in range(0, len(offsets), max_tokens)]
        bounds = starts[1:] + [len(text)]
        return [text[s:e] for s, e in zip(starts, bounds, strict=True) if text[s:e].strip()]


@dataclass
class Block:
    text: str
    is_code: bool
    from_split_code: bool = False


@dataclass
class Chunk:
    text: str
    n_tokens: int
    from_split_code: bool


def split_blocks(text: str) -> list[Block]:
    """Fenced code blocks stay whole; the prose between them splits at blank lines."""
    blocks: list[Block] = []
    pos = 0
    for match in _FENCE_RE.finditer(text):
        blocks.extend(_paragraphs(text[pos:match.start()]))
        blocks.append(Block(match.group(0), is_code=True))
        pos = match.end()
    blocks.extend(_paragraphs(text[pos:]))
    return blocks


def _paragraphs(text: str) -> list[Block]:
    return [Block(p.strip("\n"), is_code=False)
            for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _greedy_groups(lengths: Sequence[int], joiner_cost: int, budget: int) -> list[list[int]]:
    """Group consecutive items so each group's estimated size stays within budget."""
    groups: list[list[int]] = []
    current: list[int] = []
    size = 0
    for i, n in enumerate(lengths):
        extra = n + (joiner_cost if current else 0)
        if current and size + extra > budget:
            groups.append(current)
            current, size = [], 0
            extra = n
        current.append(i)
        size += extra
    if current:
        groups.append(current)
    return groups


class Chunker:
    def __init__(self, counter: TokenCounter, max_tokens: int):
        self.counter = counter
        self.max_tokens = max_tokens
        # Cost of the separators, measured once with the real tokenizer.
        self.block_joiner_cost = max(1, self._joiner_cost(BLOCK_JOINER))
        self.line_joiner_cost = max(1, self._joiner_cost(LINE_JOINER))

    def _joiner_cost(self, joiner: str) -> int:
        a, b, ab = self.counter.count(["x", "y", f"x{joiner}y"])
        return ab - a - b

    # -- public -------------------------------------------------------------
    def chunk(self, text: str) -> list[Chunk]:
        blocks = split_blocks(text)
        if not blocks:
            return []
        fitted: list[Block] = []
        for block, n in zip(blocks, self.counter.count([b.text for b in blocks]), strict=True):
            fitted.extend([block] if n <= self.max_tokens else self._split_oversized(block))
        lengths = self.counter.count([b.text for b in fitted])
        groups = _greedy_groups(lengths, self.block_joiner_cost, self.max_tokens)
        return self._finalize([[fitted[i] for i in g] for g in groups])

    # -- helpers ------------------------------------------------------------
    def _finalize(self, groups: list[list[Block]]) -> list[Chunk]:
        """Measure each chunk exactly; split any group that still ends up too long."""
        texts = [BLOCK_JOINER.join(b.text for b in g) for g in groups]
        chunks: list[Chunk] = []
        for group, text, n in zip(groups, texts, self.counter.count(texts), strict=True):
            if n <= self.max_tokens or len(group) == 1:
                chunks.append(Chunk(text, n, any(b.from_split_code for b in group)))
            else:
                mid = len(group) // 2
                chunks.extend(self._finalize([group[:mid], group[mid:]]))
        return chunks

    def _split_oversized(self, block: Block) -> list[Block]:
        """Split one too-long block at line edges (code pieces are re-fenced)."""
        lines = block.text.split("\n")
        opening, closing = "", ""
        if block.is_code:
            opening, closing, lines = lines[0], lines[-1], lines[1:-1]
        wrap_cost = self.counter.count([f"{opening}\n\n{closing}"])[0] if block.is_code else 0
        budget = self.max_tokens - wrap_cost - 2 * self.line_joiner_cost

        units: list[str] = []
        for line, n in zip(lines, self.counter.count(lines), strict=True):
            units.extend([line] if n <= budget else self._split_hard(line, budget))
        groups = _greedy_groups(self.counter.count(units), self.line_joiner_cost, budget)

        pieces = []
        for g in groups:
            body = LINE_JOINER.join(units[i] for i in g)
            text = f"{opening}\n{body}\n{closing}" if block.is_code else body
            if text.strip():
                pieces.append(Block(text, block.is_code, from_split_code=block.is_code))
        return pieces

    def _split_hard(self, text: str, budget: int) -> list[str]:
        """Last resort for a single line longer than the budget: cut at token edges."""
        size = budget
        while size > 0:
            pieces = self.counter.split_by_tokens(text, size)
            if all(n <= budget for n in self.counter.count(pieces)):
                return pieces
            size -= max(1, size // 10)
        raise ValueError("Could not split text into pieces within the token budget")
