"""Read which libraries a piece of Python code imports.

`top_level_imports` uses Python's `ast`. If the whole text does not parse (for
example a DS-1000 prompt mixes prose and code), it falls back to parsing each
line that starts with `import` or `from` on its own; prose lines that merely
start with "from" fail to parse and are ignored.
"""

from __future__ import annotations

import ast
import re
import sys
import warnings

from coding_track import config

_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s+\w")


def _imports_in_tree(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])   # relative imports are skipped
    return names


def _parse(code: str) -> ast.AST:
    # Dataset code often has escapes like "\s" in plain strings; ast.parse warns
    # about them (SyntaxWarning). Irrelevant for reading imports, so silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def top_level_imports(code: str) -> set[str]:
    """Top-level module names imported by `code` (e.g. 'matplotlib' for matplotlib.pyplot)."""
    try:
        return _imports_in_tree(_parse(code))
    except SyntaxError:
        pass
    names: set[str] = set()
    for line in code.splitlines():
        if _IMPORT_LINE.match(line):
            try:
                names |= _imports_in_tree(_parse(line.strip()))
            except SyntaxError:
                continue
    return names


def is_stdlib(module: str) -> bool:
    return module in sys.stdlib_module_names or module in config.REMOVED_STDLIB_MODULES


def normalize_package(name: str) -> str:
    """PEP 503 normalization: lower case, runs of -_. become a single '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def import_to_package(module: str) -> str:
    return normalize_package(config.IMPORT_TO_PACKAGE.get(module, module))


def third_party_packages(modules: set[str]) -> set[str]:
    """Map import names to PyPI package names, dropping standard-library modules."""
    return {import_to_package(m) for m in modules if not is_stdlib(m)}
