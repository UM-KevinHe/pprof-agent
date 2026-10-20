"""The measure specification: the typed state the planner reads and the harness verifies against.

It fixes what the analysis is ABOUT (the policy choices of a measure, DESIGN.md §3), never what runs
next. Every field carries a source: `quoted` (the analyst wrote it), `published` (a specification
file), `inferred` (a deterministic rule where the data leave one possibility), `default` (a value
disclosed in the report), `asked` (the analyst answered the numbered question), `remembered`.
The model never fills a field: `complete()` is rules, `parse_reply()` is a parser, `verify()` is a
check against the consequence card computed by R.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

ENUMS: Dict[str, Tuple[str, ...]] = {
    "outcome_type": ("binary", "binomial"),
    "event_direction": ("higher_is_worse", "higher_is_better"),
    "unit": ("patient_month", "index_discharge", "patient_record", "patient_year", "other"),
    "standardization": ("indirect", "direct", "both"),
    "norm": ("median", "size_weighted_mean"),
    "reference": ("theoretical", "empirical", "both"),
    "consequential_tail": ("worse", "better", "both"),
}
FIELD_NUMBERS: Dict[str, int] = {
    "outcome": 1, "event_direction": 2, "unit": 3, "provider": 4, "patient": 5, "adjusters": 6,
    "eligibility": 7, "standardization": 8, "norm": 9, "reference": 10, "consequential_tail": 11, "trials": 12,
}
NUMBER_FIELDS = {v: k for k, v in FIELD_NUMBERS.items()}
DEFAULT_ELIGIBILITY = {"patient_month": 2, "index_discharge": 3, "patient_record": 2, "patient_year": 2, "other": 10}
SOURCES = ("quoted", "published", "inferred", "default", "asked", "remembered")
# the fields that change what is fitted; everything else is provenance
CONFIG_FIELDS = ("outcome", "outcome_type", "trials", "event_direction", "unit", "provider", "patient",
                 "eligibility", "adjusters", "select_from", "groups", "standardization", "norm", "reference",
                 "consequential_tail", "level", "crossed_with")
REQUIRED = ("outcome", "event_direction", "unit", "provider", "patient")


def _aslist(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


@dataclass(frozen=True)
class MeasureSpecification:
    outcome: Optional[str] = None
    outcome_type: Optional[str] = None
    trials: Optional[str] = None
    event_direction: Optional[str] = None
    unit: Optional[str] = None
    provider: Optional[str] = None
    patient: Optional[str] = None            # a column, or "none"
    eligibility: Optional[int] = None
    adjusters: Optional[List[str]] = None
    select_from: Optional[List[str]] = None
    groups: Optional[Dict[str, int]] = None
    standardization: Optional[str] = None
    norm: Optional[str] = None
    reference: Optional[str] = None
    consequential_tail: Optional[str] = None
    level: Optional[float] = None
    crossed_with: str = "none"
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ updates
    def set(self, name: str, value: Any, source: str) -> "MeasureSpecification":
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if name in ENUMS and value is not None and value not in ENUMS[name]:
            raise ValueError(f"{name} must be one of {ENUMS[name]}, not {value!r}")
        src = dict(self.sources)
        src[name] = source
        return replace(self, **{name: value, "sources": src})

    def note(self, text: str) -> "MeasureSpecification":
        return replace(self, notes=list(self.notes) + [text])

    # ------------------------------------------------------------------ queries
    def settled(self, name: str) -> bool:
        v = getattr(self, name)
        if name == "adjusters":
            return bool(v) or bool(self.select_from)
        return v is not None

    def missing(self) -> List[str]:
        out = [f for f in REQUIRED if not self.settled(f)]
        if not self.settled("adjusters"):
            out.append("adjusters")
        return out

    def config(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in CONFIG_FIELDS}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], source: str = "quoted") -> "MeasureSpecification":
        known = {k: v for k, v in d.items() if k in CONFIG_FIELDS}
        src = dict(d.get("sources") or {})
        for k in known:
            src.setdefault(k, source)
        if "adjusters" in known and known["adjusters"] is not None:
            known["adjusters"] = [str(a) for a in _aslist(known["adjusters"])]
        if "select_from" in known and known["select_from"] is not None:
            known["select_from"] = [str(a) for a in _aslist(known["select_from"])]
        return cls(**known, sources=src, notes=list(d.get("notes") or []))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)


# ---------------------------------------------------------------------------------------- profile access
def profile_columns(profile: Dict[str, Any]) -> List[str]:
    return [c["name"] for c in profile.get("columns", [])]


def profile_column(profile: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for c in profile.get("columns", []):
        if c["name"] == name:
            return c
    return None


def candidates(profile: Dict[str, Any], role: str) -> List[str]:
    return [str(x) for x in _aslist((profile.get("candidates") or {}).get(role))]


def resolve_column(profile: Dict[str, Any], text: str) -> Optional[str]:
    """A column named by the analyst, matched exactly or ignoring case and underscores."""
    cols = profile_columns(profile)
    t = text.strip().strip("`'\"")
    if t in cols:
        return t
    key = re.sub(r"[\s_\-]", "", t.lower())
    for c in cols:
        if re.sub(r"[\s_\-]", "", c.lower()) == key:
            return c
    return None


# ---------------------------------------------------------------------------------------- completion
@dataclass
class OpenItem:
    field: str
    number: int
    question: str
    options: List[str]
    evidence: str = ""


def complete(spec: MeasureSpecification, profile: Dict[str, Any]) -> Tuple[MeasureSpecification, List[OpenItem]]:
    """Fill what a deterministic rule can settle from the profile; list what must be asked."""
    sp = spec
    items: List[OpenItem] = []
    cols = profile_columns(profile)

    if not sp.outcome:
        oc = candidates(profile, "outcome")
        if len(oc) == 1:
            sp = sp.set("outcome", oc[0], "inferred").note(f"outcome: the only 0/1 column is {oc[0]}")
        else:
            items.append(OpenItem("outcome", 1, "Which column is the outcome (1 = the event occurred)?",
                                  oc or [c for c in cols if (profile_column(profile, c) or {}).get("binary")],
                                  f"{len(oc)} columns hold only 0 and 1"))
    if sp.outcome and not sp.trials and not (profile_column(profile, sp.outcome) or {}).get("binary", True):
        # a count outcome needs its trials (the denominator): settled when only one column can be it, asked otherwise
        tc = [c for c in candidates(profile, "trials") if c != sp.outcome]
        if len(tc) == 1:
            sp = sp.set("trials", tc[0], "inferred").note(f"trials: the outcome {sp.outcome} is a count, and the only trials-like column is {tc[0]}")
        else:
            items.append(OpenItem("trials", 12, f"The outcome {sp.outcome} is a count, not 0/1: which column holds its trials (the denominator)?",
                                  tc or [c for c in cols if c != sp.outcome], "integer columns at least one and at least the outcome"))
    if sp.outcome and not sp.outcome_type:
        sp = sp.set("outcome_type", "binomial" if sp.trials else "binary", "inferred")

    if not sp.provider:
        pc = candidates(profile, "provider")
        if len(pc) == 1:
            sp = sp.set("provider", pc[0], "inferred").note(f"provider: the only grouping column is {pc[0]}")
        else:
            items.append(OpenItem("provider", 4, "Which column identifies the provider?", pc,
                                  "columns with many levels and several rows per level"))

    if not sp.patient:
        kc = [c for c in candidates(profile, "patient") if c != sp.provider]
        if len(kc) == 1:
            sp = sp.set("patient", kc[0], "inferred").note(f"patient: the only id column with more levels than the provider column is {kc[0]}")
        elif len(kc) == 0 and sp.provider:
            sp = sp.set("patient", "none", "inferred").note(
                "patient: no column with more levels than the provider column looks like a patient id; rows are taken as independent")
        elif sp.provider:
            items.append(OpenItem("patient", 5, "Which column identifies the patient (`none` if every row is a different patient)?",
                                  kc + ["none"], "columns with more levels than the provider column"))

    if not sp.event_direction:
        items.append(OpenItem("event_direction", 2, "Is a higher event rate worse (readmission, death) or better (fistula use)?",
                              ["higher_is_worse", "higher_is_better"], "the data cannot tell"))

    if not sp.unit:
        tc = candidates(profile, "time")
        ev = f"time-like columns: {', '.join(tc)}" if tc else "no time-like column"
        items.append(OpenItem("unit", 3, "What is one row: a patient-month, an index discharge, a patient-record, a patient-year, or other?",
                              list(ENUMS["unit"]), ev))

    if sp.unit and sp.eligibility is None:
        sp = sp.set("eligibility", DEFAULT_ELIGIBILITY.get(sp.unit, 10), "default")

    if not sp.settled("adjusters"):
        cov = [c for c in candidates(profile, "covariates") if c not in (sp.outcome, sp.provider, sp.patient, sp.trials)]
        published = [str(x) for x in _aslist(profile.get("published_adjusters"))]
        options = []
        if published:
            options.append(f"A = the published adjustment set ({len(published)} columns)")
        options.append(f"B = every eligible numeric column ({len(cov)}): {', '.join(cov)}")
        options.append("or list the columns; `B minus X, Y` and `A plus X` are accepted")
        items.append(OpenItem("adjusters", 6, "Which columns are the risk adjusters?", options,
                              "non-constant numeric columns that are not an id, the outcome or a time column"))

    for name, default in (("standardization", "both"), ("norm", "median"), ("reference", "both"),
                          ("consequential_tail", "worse")):
        if getattr(sp, name) is None:
            sp = sp.set(name, default, "default")
    if sp.level is None:
        sp = sp.set("level", 0.95, "default")
    items.sort(key=lambda it: it.number)
    return sp, items


def render_question(items: List[OpenItem]) -> str:
    if not items:
        return ""
    lines = ["Please answer by number, one item per line (for example `2) higher_is_better`):"]
    for it in items:
        opts = "; ".join(it.options) if it.options else ""
        lines.append(f"{it.number}) {it.question}" + (f"  [{opts}]" if opts else "") + (f"  ({it.evidence})" if it.evidence else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- the numbered reply
_ITEM = re.compile(r"(?:^|\s)(\d{1,2})\)\s*(.*?)(?=(?:\s\d{1,2}\)\s)|$)", re.S)

_SYNONYMS = {
    "event_direction": {"worse": "higher_is_worse", "bad": "higher_is_worse", "higher_is_worse": "higher_is_worse",
                        "lower_is_better": "higher_is_worse", "better": "higher_is_better", "good": "higher_is_better",
                        "higher_is_better": "higher_is_better", "lower_is_worse": "higher_is_better"},
    "unit": {"month": "patient_month", "patient_month": "patient_month", "patient-month": "patient_month",
             "discharge": "index_discharge", "index_discharge": "index_discharge", "record": "patient_record",
             "patient_record": "patient_record", "year": "patient_year", "patient_year": "patient_year", "other": "other"},
    "standardization": {"indirect": "indirect", "direct": "direct", "both": "both"},
    "norm": {"median": "median", "mean": "size_weighted_mean", "weighted": "size_weighted_mean",
             "size_weighted_mean": "size_weighted_mean"},
    "reference": {"theoretical": "theoretical", "empirical": "empirical", "both": "both"},
    "consequential_tail": {"worse": "worse", "better": "better", "both": "both"},
}


def _enum_value(name: str, text: str) -> Optional[str]:
    t = text.strip().lower().replace("-", "_").replace(" ", "_")
    table = _SYNONYMS[name]
    if t in table:
        return table[t]
    for k, v in table.items():
        if k in t:
            return v
    return None


def _parse_adjusters(text: str, profile: Dict[str, Any], spec: MeasureSpecification) -> Tuple[Optional[List[str]], List[str]]:
    problems: List[str] = []
    t = text.strip()
    cov = [c for c in candidates(profile, "covariates") if c not in (spec.outcome, spec.provider, spec.patient, spec.trials)]
    published = [str(x) for x in _aslist(profile.get("published_adjusters"))]
    m = re.match(r"^\s*([AB])\b(.*)$", t, re.I)
    if m:
        base = published if m.group(1).upper() == "A" else cov
        if m.group(1).upper() == "A" and not published:
            return None, ["there is no published adjustment set to take as A"]
        rest = m.group(2)
        chosen = list(base)
        for op, names in re.findall(r"(minus|plus|without|with|\-|\+)\s+([A-Za-z0-9_,\s]+?)(?=\s+(?:minus|plus|without|with|\-|\+)\s|$)", rest, re.I):
            cols = [resolve_column(profile, n) for n in re.split(r"[,\s]+", names.strip()) if n]
            bad = [n for n, c in zip(re.split(r"[,\s]+", names.strip()), cols) if n and c is None]
            problems += [f"{b} is not a column of the file" for b in bad]
            cols = [c for c in cols if c]
            if op.lower() in ("minus", "without", "-"):
                chosen = [c for c in chosen if c not in cols]
            else:
                chosen += [c for c in cols if c not in chosen]
        return chosen, problems
    names = [n for n in re.split(r"[,;\s]+", t) if n]
    out: List[str] = []
    for n in names:
        c = resolve_column(profile, n)
        if c is None:
            problems.append(f"{n} is not a column of the file")
        elif c not in out:
            out.append(c)
    return (out or None), problems


def parse_reply(text: str, spec: MeasureSpecification, profile: Dict[str, Any]) -> Tuple[MeasureSpecification, List[str]]:
    """Read a numbered reply deterministically; no model. Returns the updated specification and problems."""
    sp = spec
    problems: List[str] = []
    pairs = _ITEM.findall(" " + text.strip())
    if not pairs:
        return sp, ["no numbered item found (write `1) column`, `2) higher_is_better`, ...)"]
    for num, val in pairs:
        n = int(num)
        name = NUMBER_FIELDS.get(n)
        val = val.strip().rstrip(".")
        if name is None:
            problems.append(f"there is no item {n}")
            continue
        if name in ("outcome", "provider", "trials"):
            c = resolve_column(profile, val)
            if c is None:
                problems.append(f"{n}) {val}: not a column of the file")
            else:
                sp = sp.set(name, c, "asked")
        elif name == "patient":
            if val.lower() in ("none", "no", "-"):
                sp = sp.set("patient", "none", "asked")
            else:
                c = resolve_column(profile, val)
                if c is None:
                    problems.append(f"{n}) {val}: not a column of the file")
                else:
                    sp = sp.set("patient", c, "asked")
        elif name == "adjusters":
            cols, probs = _parse_adjusters(val, profile, sp)
            problems += [f"{n}) {p}" for p in probs]
            if cols:
                sp = sp.set("adjusters", cols, "asked")
        elif name == "eligibility":
            m = re.search(r"\d+", val)
            if m:
                sp = sp.set("eligibility", int(m.group()), "asked")
            else:
                problems.append(f"{n}) {val}: not a number")
        else:
            v = _enum_value(name, val)
            if v is None:
                problems.append(f"{n}) {val}: not one of {ENUMS[name]}")
            else:
                sp = sp.set(name, v, "asked")
    return sp, problems


# ---------------------------------------------------------------------------------------- verification
@dataclass
class Verification:
    ok: bool
    refusals: List[str]
    warnings: List[str]
    card: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "refusals": self.refusals, "warnings": self.warnings, "card": self.card}


def verify(spec: MeasureSpecification, card: Dict[str, Any]) -> Verification:
    """Deterministic rules on the consequence card: refuse what V1 cannot represent; warn otherwise."""
    refusals: List[str] = []
    warnings: List[str] = []
    missing = spec.missing()
    if missing:
        refusals.append("not settled: " + ", ".join(missing))
    if spec.crossed_with and spec.crossed_with != "none":
        refusals.append(f"the analysis declares a second provider level ({spec.crossed_with}); V1 fits one level only "
                        "(the crossed facility-hospital model is not available yet)")
    for p in _aslist(card.get("problems")):
        refusals.append(f"{p.get('field')}: {p.get('problem')}")
    for p in _aslist(card.get("adjuster_problems")):
        prob = str(p.get("problem", ""))
        if "not numeric" in prob or "constant" in prob:
            refusals.append(f"adjuster {p.get('name')}: {prob} (the agent does not recode data; declare another column)")
        else:
            warnings.append(f"adjuster {p.get('name')}: {prob}")
    if spec.unit in ("patient_month", "patient_year") and spec.patient == "none":
        refusals.append(f"a {spec.unit} file needs a patient column for the within-patient correlation; none was declared")
    if card.get("eligible_providers", 0) < 2:
        refusals.append(f"only {card.get('eligible_providers', 0)} providers meet the eligibility threshold of {card.get('eligibility')}")
    rate = card.get("event_rate")
    if rate is not None and (rate <= 0 or rate >= 1):
        refusals.append(f"the event rate among eligible providers is {rate}; nothing can be estimated")
    if card.get("has_patient") and card.get("repeats") and spec.unit in ("index_discharge", "other", "patient_record"):
        warnings.append(f"patients contribute up to {card.get('max_rows_per_patient')} rows; the patient-clustered variances apply")
    inv = (card.get("invariant_providers") or {}).get("total", 0)
    if inv:
        warnings.append(f"{inv} eligible providers have invariant outcomes (all events or none): Wald-type statistics are not reliable there")
    small = (card.get("small_providers") or {}).get("le_3", 0)
    if small:
        warnings.append(f"{small} eligible providers have three patients or fewer")
    dropped = card.get("rows_dropped_incomplete", 0)
    if dropped:
        warnings.append(f"{dropped} rows with a missing outcome, provider or adjuster are left out (the agent does not impute)")
    return Verification(ok=not refusals, refusals=refusals, warnings=warnings, card=card)


def consequence_lines(spec: MeasureSpecification, card: Dict[str, Any]) -> List[str]:
    """The consequence card as lines the analyst can falsify at a glance."""
    rpp = card.get("rows_per_patient") or {}
    ppp = card.get("patients_per_provider") or {}
    inv = card.get("invariant_providers") or {}
    return [
        f"rows {card.get('rows')} (complete {card.get('rows_complete')}), patients {card.get('patients')}, providers {card.get('providers')}",
        f"eligible providers (>= {card.get('eligibility')} patients): {card.get('eligible_providers')}; excluded {card.get('excluded_providers')}",
        f"events {card.get('events')} of {card.get('trials_total')} ({100 * (card.get('event_rate') or 0):.1f}%)",
        f"rows per patient: median {rpp.get('median')}, max {rpp.get('max')}" if card.get("has_patient") else "no patient column: rows taken as independent",
        f"patients per provider: min {ppp.get('min')}, median {ppp.get('median')}, max {ppp.get('max')}",
        f"providers with invariant outcomes: {inv.get('total', 0)} ({inv.get('none', 0)} none, {inv.get('all', 0)} all)",
    ]
