"""The state of one analysis, typed, in one place.

Everything the harness decides and everything the model is shown derives from `Session`. The phase is
derived from what the session holds, never stored. Results live in a ledger keyed by a result id
(`fit`, `sm`, `test:robust_score`, `en:robust_score`, ...); the model is shown summaries and references,
never a table.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .specification import MeasureSpecification, Verification


@dataclass
class Result:
    id: str
    action: str
    args: Dict[str, Any]
    summary: Dict[str, Any]
    tables: Dict[str, str]
    step: int
    source: str                      # planner | harness | core | harness_default
    seconds: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    data_path: str
    run_dir: str
    profile: Optional[Dict[str, Any]] = None
    spec: Optional[MeasureSpecification] = None
    open_items: List[Any] = field(default_factory=list)
    verification: Optional[Verification] = None
    admissible: Optional[Any] = None
    results: List[Result] = field(default_factory=list)
    steps: int = 0
    budget: int = 30
    cost_used: int = 0
    plan_log: List[Dict[str, Any]] = field(default_factory=list)
    model_calls: List[Dict[str, Any]] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)
    recommended: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    turns: List[str] = field(default_factory=list)
    config_hash: Optional[str] = None
    question: Optional[str] = None
    failed: List[str] = field(default_factory=list)   # result ids whose engine step failed
    prose: Dict[str, str] = field(default_factory=dict)  # the model's report paragraphs, references unresolved
    transcript: List[Dict[str, Any]] = field(default_factory=list)  # the conversation as shown: role, text, figures

    # ------------------------------------------------------------------ ledger
    def ids(self) -> List[str]:
        return [r.id for r in self.results]

    def has(self, rid: str) -> bool:
        if rid == "sm":
            return any(r.id == "sm" for r in self.results)
        return any(r.id == rid for r in self.results)

    def get(self, rid: str) -> Optional[Result]:
        for r in reversed(self.results):
            if r.id == rid:
                return r
        return None

    def add(self, result: Result) -> None:
        self.results = [r for r in self.results if r.id != result.id] + [result]

    def results_of(self, action: str) -> List[Result]:
        return [r for r in self.results if r.action == action]

    @property
    def card(self) -> Dict[str, Any]:
        return self.verification.card if self.verification else {}

    # ------------------------------------------------------------------ phase and state line
    @property
    def phase(self) -> str:
        if self.profile is None:
            return "EMPTY"
        if self.spec is None or self.open_items:
            return "DECLARING"
        if self.verification is None:
            return "DECLARED"
        if not self.verification.ok:
            return "REFUSED"
        if self.finished:
            return "DONE"
        return "RUNNING" if self.results else "VERIFIED"

    def state_line(self) -> str:
        parts = [f"phase: {self.phase}"]
        if self.spec is not None:
            parts.append(f"outcome: {self.spec.outcome or 'open'}; unit: {self.spec.unit or 'open'}; "
                         f"adjusters: {len(self.spec.adjusters or [])}")
        if self.verification is not None:
            c = self.verification.card
            parts.append(f"eligible providers: {c.get('eligible_providers')}; patients: {c.get('patients')}; "
                         f"event rate: {100 * (c.get('event_rate') or 0):.1f}%")
        parts.append(f"results: {len(self.results)}; budget left: {self.budget - self.cost_used}; steps: {self.steps}")
        return " | ".join(parts)

    # ------------------------------------------------------------------ provenance
    def to_provenance(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path, "run_dir": self.run_dir, "phase": self.phase,
            "specification": self.spec.to_dict() if self.spec else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "admissible": self.admissible.to_dict() if self.admissible else None,
            "results": [r.to_dict() for r in self.results],
            "steps": self.steps, "budget": self.budget, "cost_used": self.cost_used,
            "plan_log": self.plan_log, "model_calls": self.model_calls, "refusals": self.refusals,
            "recommended": self.recommended, "finished": self.finished, "config_hash": self.config_hash,
            "profile": self.profile, "prose": self.prose, "transcript": self.transcript, "turns": self.turns,
            "failed": self.failed, "question": self.question,
        }

    @classmethod
    def load(cls, run_dir: str, name: str = "trace.json") -> "Session":
        """A session restored from its trace: the ledger, the specification, the verification, the admissible
        set and the recommendation, enough to export, to break down a provider or to answer a question
        without the model. The run directory must still hold the tables and the fitted objects."""
        from . import rules
        t = json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))
        s = cls(data_path=t.get("data_path", ""), run_dir=str(run_dir), budget=int(t.get("budget") or 30))
        s.profile = t.get("profile")
        if s.profile is None:                       # traces written before the profile was saved: the engine output is kept
            p = Path(run_dir) / "outputs" / "00_profile.json"
            if p.exists():
                out = json.loads(p.read_text(encoding="utf-8"))
                s.profile = out.get("result", out) if isinstance(out, dict) else None
        if t.get("specification"):
            s.spec = MeasureSpecification.from_dict(t["specification"])
        if t.get("verification"):
            v = t["verification"]
            s.verification = Verification(bool(v.get("ok")), list(v.get("refusals") or []), list(v.get("warnings") or []), dict(v.get("card") or {}))
        if t.get("admissible"):
            a = dict(t["admissible"])
            s.admissible = rules.Admissible(**{k: a[k] for k in a if k in rules.Admissible.__dataclass_fields__})
        s.results = [Result(**{k: r[k] for k in r if k in Result.__dataclass_fields__}) for r in t.get("results") or []]
        s.steps, s.cost_used = int(t.get("steps") or 0), int(t.get("cost_used") or 0)
        s.plan_log, s.model_calls, s.refusals = list(t.get("plan_log") or []), list(t.get("model_calls") or []), list(t.get("refusals") or [])
        s.recommended, s.finished, s.config_hash = dict(t.get("recommended") or {}), bool(t.get("finished")), t.get("config_hash")
        s.prose, s.transcript, s.turns = dict(t.get("prose") or {}), list(t.get("transcript") or []), list(t.get("turns") or [])
        s.failed, s.question = list(t.get("failed") or []), t.get("question")
        return s

    def save(self, name: str = "trace.json") -> Path:
        p = Path(self.run_dir) / name
        p.write_text(json.dumps(self.to_provenance(), indent=1, default=str), encoding="utf-8")
        return p
