import numpy as np

from coding_track.common.minhash_utils import new_lsh, signature, to_minhash
from coding_track.common.so_filter import LibraryMatcher, question_id

TARGETS = [{"import_name": "pandas", "package": "pandas"},
           {"import_name": "sklearn", "package": "scikit-learn"},
           {"import_name": "requests", "package": "requests"}]
MATCHER = LibraryMatcher(TARGETS, frozenset({"requests"}))


def test_import_lines_count():
    assert MATCHER.libraries("Q: hi\nimport pandas as pd\ndf = pd.DataFrame()") == ["pandas"]
    assert MATCHER.libraries("Q: x\nfrom sklearn.svm import SVC") == ["scikit-learn"]


def test_bare_name_needs_a_python_post():
    assert MATCHER.libraries("Q: Is scikit-learn good with Python?") == ["scikit-learn"]
    assert MATCHER.libraries("Q: pandas are cute animals in China") == []


def test_ordinary_word_names_need_import_or_pip():
    assert MATCHER.libraries("Q: Python server handles many requests per second") == []
    assert MATCHER.libraries("Q: in python I ran pip install requests") == ["requests"]
    assert MATCHER.libraries("Q: x\nimport requests") == ["requests"]


def test_python_detection():
    assert LibraryMatcher.is_python("from os import path")
    assert not LibraryMatcher.is_python("import java.util.List;\npublic class A {}")


def test_question_id():
    assert question_id("https://stackoverflow.com/questions/3283984") == "3283984"
    assert question_id(None) is None


def test_minhash_finds_near_copy_and_ignores_unrelated():
    base = "how do I merge two pandas dataframes on a shared key column and keep all rows"
    lsh = new_lsh(0.5, 128)
    lsh.insert("task", to_minhash(signature(base, 3, 128)))
    near = to_minhash(signature(base + " please", 3, 128))
    far = to_minhash(signature("compile a rust crate for an arm microcontroller target", 3, 128))
    assert lsh.query(near) == ["task"]
    assert lsh.query(far) == []
    assert signature("", 3, 128) is None
    assert isinstance(signature(base, 3, 128), np.ndarray)
