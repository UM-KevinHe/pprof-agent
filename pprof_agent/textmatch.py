"""Did the analyst actually write this column name? One predicate (chapter II `textmatch.py`; ties between
names of equal length broken deterministically, 2026-10-28).

Exactly three spellings of a column name count as writing it: the name itself, underscores as spaces,
underscores as hyphens. No stemming, no edit distance, no synonyms. A name the message does not contain
is never `quoted`, whatever the model says.
"""
from __future__ import annotations

import re
from typing import Iterable, List


def variants(name: str) -> List[str]:
    v = {name, name.replace("_", " "), name.replace("_", "-")}
    return sorted(v, key=len, reverse=True)


def _pattern(form: str) -> str:
    return rf"(?<![a-z0-9_]){re.escape(form)}(?![a-z0-9_])"


def names_mentioned(text: str, columns: Iterable[str]) -> List[str]:
    """The names the text writes. Longer names claim their words first; between names of equal length (`pid`
    and the column PID) the one written in the text's own case comes first, then the alphabet, so the answer
    never depends on the order of a set."""
    low = text.lower()
    found = []
    for col in sorted(set(columns), key=lambda c: (-len(c), c not in text, c)):
        hit = False
        for form in variants(col.lower()):
            pat = _pattern(form)
            if re.search(pat, low):
                hit = True
                low = re.sub(pat, " \x00 ", low)
        if hit:
            found.append(col)
    return found


def mentions(text: str, name: str, columns: Iterable[str]) -> bool:
    if not name or not str(name).strip():
        return False
    cands = set(columns) | {str(name)}
    return str(name) in names_mentioned(text, cands)


def phrase_in(text: str, phrase: str) -> bool:
    norm = lambda x: " ".join(str(x).lower().split())  # noqa: E731
    ph = norm(phrase)
    return bool(ph) and ph in norm(text)
