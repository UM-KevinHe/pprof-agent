"""The instructions each model act runs under, as files (chapter II component 2, carried over).

One policy file per act in `policies/`, sent verbatim as the system prompt of that one constrained
call, five fixed headings, hashed into every run's provenance. A JSON Schema `description` never
reaches the model under constrained decoding, so the purpose of every field is stated in the policy.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict

POLICY_DIR = Path(__file__).resolve().parent / "policies"
NAMES = ("intent", "spec_extraction", "next_step", "figure_selection", "report_prose", "ask", "refusal", "explain",
         "choose_procedure")
HEADINGS = ("## What you are doing", "## What the system is", "## Before you decide", "## The fields",
            "## What you must not do")


def path(name: str) -> Path:
    if name not in NAMES:
        raise KeyError(f"no policy named {name!r}; known: {NAMES}")
    return POLICY_DIR / f"{name}.md"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    text = path(name).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"policy {name} is empty")
    return text


def sha256(name: str) -> str:
    return "sha256:" + hashlib.sha256(load(name).encode("utf-8")).hexdigest()[:16]


def describe() -> Dict[str, Dict[str, object]]:
    return {n: {"sha256": sha256(n), "chars": len(load(n))} for n in NAMES}


def sections(name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    current = None
    for line in load(name).split("\n"):
        if line.startswith("## "):
            current = line.strip()
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return out
