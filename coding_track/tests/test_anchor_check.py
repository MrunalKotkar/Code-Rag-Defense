from coding_track.anchor_check.anchor import extract_fenced, postprocess


def test_plain_completion_is_kept_as_is():
    assert postprocess("result = df.iloc[List]\n") == "result = df.iloc[List]"


def test_stops_at_end_of_code_tag():
    assert postprocess("result = 1\n</code>\nEND SOLUTION\nmore") == "result = 1"


def test_stops_at_solution_end_comment():
    assert postprocess("plt.plot(x)\n# SOLUTION END\nplt.show()") == "plt.plot(x)"


def test_text_after_the_start_markers_is_kept():
    assert postprocess("BEGIN SOLUTION\n<code>\nresult = 2\n</code>") == "result = 2"
    assert postprocess("# SOLUTION START\nplt.grid()\n# SOLUTION END") == "plt.grid()"
    assert postprocess("Here you go:\n<code>\nx = 3\n</code>") == "x = 3"


def test_python_fence_is_unwrapped():
    assert postprocess("```python\nresult = 4\n```\nDone") == "result = 4"


def test_bare_fence_keeps_the_first_line_intact():
    # CodeRAG-Bench's own helper would drop the first characters here; ours does not.
    assert postprocess("```\nresult = 5\n```") == "result = 5"


def test_unclosed_fence_reads_to_the_end():
    assert extract_fenced("```python\nx = 1\ny = 2", "```python") == "x = 1\ny = 2"
