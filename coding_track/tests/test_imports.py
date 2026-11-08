from coding_track.common.imports import (
    import_to_package, is_stdlib, third_party_packages, top_level_imports)


def test_reads_import_and_from_lines():
    code = "import numpy as np\nfrom matplotlib import pyplot as plt\nimport os.path\nx = 1\n"
    assert top_level_imports(code) == {"numpy", "matplotlib", "os"}


def test_skips_relative_imports():
    assert top_level_imports("from . import utils\nfrom .x import y\n") == set()


def test_falls_back_to_line_parsing_for_prose():
    # A DS-1000-style prompt: prose (not valid Python) around a code section.
    prompt = ("Problem:\nI have a DataFrame and want to sort it.\n"
              "from the docs I could not find it.\nA:\n<code>\nimport pandas as pd\n"
              "from scipy import sparse\n</code>\nresult = ... # put solution here\n")
    assert top_level_imports(prompt) == {"pandas", "scipy"}


def test_stdlib_includes_removed_modules():
    assert is_stdlib("os") and is_stdlib("re")
    assert is_stdlib("cgi") and is_stdlib("imp")       # removed from newer Pythons
    assert not is_stdlib("pandas")


def test_package_mapping():
    assert import_to_package("sklearn") == "scikit-learn"
    assert import_to_package("PIL") == "pillow"
    assert import_to_package("mpl_toolkits") == "matplotlib"
    assert import_to_package("flask_mail") == "flask-mail"      # PEP 503 normalization
    assert third_party_packages({"os", "sklearn", "numpy"}) == {"scikit-learn", "numpy"}
