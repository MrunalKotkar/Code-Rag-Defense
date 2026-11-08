from coding_track.common.chunking import Chunker, split_blocks


class WordCounter:
    """Test tokenizer: one token per whitespace-separated word."""

    def count(self, texts):
        return [len(t.split()) for t in texts]

    def split_by_tokens(self, text, max_tokens):
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_split_blocks_keeps_code_fences_whole():
    text = "intro para\n\n```python\nx = 1\n\ny = 2\n```\n\nouter para"
    blocks = split_blocks(text)
    assert [b.is_code for b in blocks] == [False, True, False]
    assert blocks[1].text == "```python\nx = 1\n\ny = 2\n```"   # blank line inside code kept


def test_chunks_respect_budget_and_paragraph_edges():
    text = "\n\n".join(words(30, f"p{i}_") for i in range(5))
    chunks = Chunker(WordCounter(), max_tokens=70).chunk(text)
    assert all(c.n_tokens <= 70 for c in chunks)
    # paragraphs are never cut: each chunk is made of whole paragraphs
    for c in chunks:
        assert all(len(p.split()) == 30 for p in c.text.split("\n\n"))
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_code_block_is_not_cut_when_it_fits():
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(20)) + "\n```"
    text = words(50) + "\n\n" + code
    chunks = Chunker(WordCounter(), max_tokens=80).chunk(text)
    assert any(c.text == code for c in chunks)
    assert not any(c.from_split_code for c in chunks)


def test_oversized_code_block_is_split_and_refenced():
    code = "```python\n" + "\n".join(f"v{i} = {i}" for i in range(100)) + "\n```"
    chunks = Chunker(WordCounter(), max_tokens=60).chunk(code)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("```python\n") and c.text.endswith("\n```")
        assert c.from_split_code and c.n_tokens <= 60


def test_single_huge_line_is_cut_at_token_edges():
    chunks = Chunker(WordCounter(), max_tokens=50).chunk(words(400))
    assert all(c.n_tokens <= 50 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == words(400).split()
