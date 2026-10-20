"""Knowledge the harness shows on request, retrieved without a model (chapter II component 8).

Two hand-written files: `knowledge/display_cards.md` (the figure catalogue the model chooses from) and
`knowledge/profiling_playbook.md` (situation notes for the planner, digit-free). Retrieval is by the
situation the data facts trigger; the text shown is the text in the file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def display_catalogue() -> str:
    return (KNOWLEDGE_DIR / "display_cards.md").read_text(encoding="utf-8")


def _notes(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    out: Dict[str, str] = {}
    key = None
    for line in text.splitlines():
        m = re.match(r"^## (\S+)", line)
        if m:
            key = m.group(1)
            out[key] = ""
        elif key:
            out[key] += line + "\n"
    return {k: v.strip() for k, v in out.items()}


def situations(session: Any) -> List[str]:
    card = session.card or {}
    adm = session.admissible
    out = ["always"]
    if (card.get("invariant_providers") or {}).get("total"):
        out.append("invariant_outcomes")
    if card.get("repeats"):
        out.append("repeated_observations")
    if adm is not None and adm.variance_default == "stabilized":
        out.append("few_patients")
    if (card.get("small_providers") or {}).get("le_3"):
        out.append("small_providers")
    if adm is not None and "empirical" in adm.references_computed:
        out.append("empirical_null")
    if session.spec is not None and session.spec.standardization in ("direct", "both"):
        out.append("direct_standardization")
    if session.spec is not None and session.spec.select_from:
        out.append("selection")
    en = session.get(f"en:{adm.default_test}") if adm is not None else None
    if en is not None:
        out.append("after_empirical_null")
    return out


def playbook(session: Any) -> str:
    notes = _notes(KNOWLEDGE_DIR / "profiling_playbook.md")
    keys = situations(session)
    return "\n".join(f"- {notes[k]}" for k in keys if k in notes)


def method_answer(question: str) -> str:
    """A method question answered from the playbook by keyword overlap; no model."""
    notes = _notes(KNOWLEDGE_DIR / "profiling_playbook.md")
    q = set(re.findall(r"[a-z]+", question.lower()))
    best, score = None, 0
    for k, v in notes.items():
        s = len(q & set(re.findall(r"[a-z]+", (k + " " + v).lower())))
        if s > score:
            best, score = k, s
    return notes.get(best, "") if best else ""
