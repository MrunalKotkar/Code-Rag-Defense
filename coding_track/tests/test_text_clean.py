from coding_track.common.text_clean import html_to_text, normalize_text, stackoverflow_question


def test_html_pre_becomes_fenced_code_and_keeps_indentation():
    html = ('<h1>Merge</h1> <p>Use <code>merge</code> to join.</p>'
            '<pre data-language="python">def f():\n    return pd.merge(a, b)</pre><p>Done.</p>')
    text = html_to_text(html)
    assert "```python\ndef f():\n    return pd.merge(a, b)\n```" in text
    assert "Use merge to join." in text          # inline code stays inline
    assert text.startswith("Merge")


def test_normalize_keeps_indentation_and_trims_blank_runs():
    assert normalize_text("a\r\n\n\n\n    b  \n") == "a\n\n    b"


def test_stackoverflow_question_part():
    post = "Q: How do I sort?\nI tried x.\n\nA: Use sorted().\n\nA: Or list.sort()."
    assert stackoverflow_question(post) == "How do I sort?\nI tried x."
