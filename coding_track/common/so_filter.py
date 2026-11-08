"""Decide which Stack Overflow posts mention a target/rare library, and which are Python.

CodeRAG-Bench's SO posts are plain text ("Q: ...\\n\\nA: ...") with no tags and no
code fences, and meta.language is the natural language ("en"), not the
programming language. So both decisions are text heuristics:

  mentions library L   an import line (`import L`, `from L ...`), a
                       `pip install <package>` line, or - in a Python post -
                       the bare name (`pandas`, `scikit-learn`). Names listed in
                       config.SO_NAME_MATCH_EXCLUDE are ordinary words
                       ("requests") and only count through import/pip lines.
  is a Python post     the word "python", or a Python-shaped import line
                       (`from x import y`, or `import x [as y]` alone on a line;
                       Java's `import a.b.C;` does not match).
"""

from __future__ import annotations

import re

_PY_WORD = re.compile(r"\bpython\b", re.I)
_PY_IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+[\w.]+[ \t]+import[ \t]|import[ \t]+[\w.]+(?:[ \t]+as[ \t]+\w+)?[ \t]*$)",
    re.M)
_SO_URL = re.compile(r"stackoverflow\.com/questions/(\d+)")


def _alternation(names) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


class LibraryMatcher:
    def __init__(self, targets: list[dict], name_match_exclude: frozenset[str]):
        """targets: dicts with 'import_name' and 'package' (from targets.json)."""
        self._by_import: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        for t in targets:
            self._by_import[t["import_name"]] = t["package"]
            for name in {t["import_name"].lower(), t["package"].lower()}:
                if name not in name_match_exclude:
                    self._by_name[name] = t["package"]
        self._pip_names = {**{k.lower(): v for k, v in self._by_import.items()},
                           **{t["package"].lower(): t["package"] for t in targets}}

        self._import_re = re.compile(
            rf"^[ \t]*(?:import|from)[ \t]+({_alternation(self._by_import)})\b", re.M)
        self._pip_re = re.compile(
            rf"\bpip3?[ \t]+install[ \t]+(?:-\S+[ \t]+)*({_alternation(self._pip_names)})(?![\w\-])", re.I)
        self._name_re = (re.compile(rf"(?<![\w.\-])({_alternation(self._by_name)})(?![\w\-])", re.I)
                         if self._by_name else None)

    @staticmethod
    def is_python(text: str) -> bool:
        return bool(_PY_WORD.search(text) or _PY_IMPORT.search(text))

    def libraries(self, text: str) -> list[str]:
        """Sorted packages that `text` mentions."""
        found = {self._by_import[m] for m in self._import_re.findall(text)}
        found |= {self._pip_names[m.lower()] for m in self._pip_re.findall(text)}
        if self._name_re is not None and self.is_python(text):
            found |= {self._by_name[m.lower()] for m in self._name_re.findall(text)}
        return sorted(found)


def question_id(url: str | None) -> str | None:
    """'https://stackoverflow.com/questions/4249160' -> '4249160'."""
    match = _SO_URL.search(url or "")
    return match.group(1) if match else None
