"""Choosing the reported procedure as an act of the model (planned; its training data are built first).

The act reads the declared question and the harness's diagnostics and answers one statistic, one variance
treatment and one reference distribution. This module holds what the act sees (`observation`), what it may
answer (`allowed`, the hard constraints), the answer's schema and how an answer becomes a test. The training
data (train/choice_tasks.py) are built from these functions, so the prompt that is learned is the prompt
that will run.

Only constraints that are not a judgement are hard here: a Wald-type statistic cannot be reported when some
provider's outcomes are all events or none (its estimate and standard error diverge); the exact test is not
computed above the trials limit; the stabilized variance is not computed for patients with too many rows;
the empirical null needs enough providers to fill its size groups. Whether the naive variance is too small,
whether the robust sandwich is too unstable, whether the theoretical null over-flags: those are the act's
judgement, scored in training by simulated type I error and power (summarize_oc.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import rules

CHOICE_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["reasoning", "statistic", "variance", "reference", "reason"],
    "properties": {
        "reasoning": {"type": "string"},
        "statistic": {"type": "string", "enum": ["score", "wald", "exact"]},
        "variance": {"type": "string", "enum": ["naive", "robust", "stabilized"]},
        "reference": {"type": "string", "enum": ["theoretical", "empirical"]},
        "reason": {"type": "string", "maxLength": 300},
    },
}
_TEST = {("exact", "naive"): "exact", ("score", "naive"): "score", ("wald", "naive"): "wald",
         ("score", "robust"): "robust_score", ("wald", "robust"): "robust_wald",
         ("score", "stabilized"): "stabilized_score", ("wald", "stabilized"): "stabilized_wald"}


def test_of(statistic: str, variance: str) -> Optional[str]:
    return _TEST.get((statistic, variance))


def parts_of(test: str) -> Dict[str, str]:
    return {"statistic": rules.STATISTIC_OF[test], "variance": rules.VARIANCE_OF[test]}


def allowed(session: Any) -> List[str]:
    """The procedures the act may answer, as 'test|reference': computed tests (the trials and rows limits
    already applied by rules.admissible), less the Wald-type ones when invariant providers exist; the
    empirical null where it was fitted for that test."""
    adm = session.admissible
    inv = int((session.card.get("invariant_providers") or {}).get("total", 0) or 0)
    out = []
    for t in adm.tests_computed:
        if inv > 0 and t in rules.WALD_TYPE:
            continue
        if session.get(f"test:{t}") is None:
            continue
        out.append(f"{t}|theoretical")
        if session.get(f"en:{t}") is not None:
            out.append(f"{t}|empirical")
    return out


def _q(d: Optional[Dict[str, Any]], keys=("min", "q1", "median", "q3", "max"), fmt="{:.0f}") -> str:
    if not d:
        return "n/a"
    return " / ".join(fmt.format(d[k]) if d.get(k) is not None else "-" for k in keys)


def _by_size(d: Optional[Dict[str, Any]], fmt="{:.2f}") -> str:
    if not d:
        return "n/a"
    return " ".join(fmt.format(d[q]) if d.get(q) is not None else "-" for q in ("q1", "q2", "q3", "q4"))


def observation(session: Any) -> str:
    """The user turn of the act: the question, the data, the fit, and per computed test how its statistics
    read against a standard normal and what the empirical null fitted; then the allowed choices."""
    spec, card = session.spec, session.card
    fit = session.get("fit").summary
    worse = "higher" if spec.event_direction == "higher_is_worse" else "lower"
    tail = spec.consequential_tail or "worse"
    inv = card.get("invariant_providers") or {}
    lines = ["The question (declared by the analyst):",
             f"  outcome {spec.outcome}{' out of ' + spec.trials if spec.trials else ''}; a {worse} rate is worse; "
             f"the consequential tail: {tail}; {spec.standardization or 'indirect'} standardization; "
             f"providers with fewer than {spec.eligibility} patients are not profiled",
             "", "The data:",
             f"  providers profiled {card.get('eligible_providers')} (excluded below eligibility {card.get('excluded_providers')})",
             f"  patients per provider (min / q1 / median / q3 / max): {_q(card.get('patients_per_provider'))}",
             f"  rows per patient (min / q1 / median / q3 / max): {_q(card.get('rows_per_patient'))}",
             f"  event rate {card.get('event_rate', 0):.3f}; trials per provider, largest {_q(card.get('trials_per_provider'), keys=('max',))}",
             f"  providers whose outcomes are all events or none: {inv.get('total', 0)}",
             "", "The fit:",
             f"  events per adjuster {fit.get('events_per_adjuster', 0):.0f}; providers at the estimation bound "
             f"{fit.get('at_bound_providers', 0)}; within-patient correlation of residuals "
             + (f"{fit['within_patient_corr']:.3f}" if fit.get("within_patient_corr") is not None else "n/a (no repeats)"),
             "", "Each computed test, its statistics read against a standard normal (size quartiles q1 smallest .. q4 largest):",
             "  test              beyond 1.96 %  by size q1..q4          scale (MAD) by size q1..q4    empirical null: scale by size group, location max"]
    for t in session.admissible.tests_computed:
        r = session.get(f"test:{t}")
        if r is None:
            continue
        s = r.summary
        en = session.get(f"en:{t}")
        en_txt = "not fitted"
        if en is not None:
            g = en.summary.get("groups") or []
            scales = [x.get("scale") for x in (g if isinstance(g, list) else [])]
            en_txt = (" ".join(f"{v:.2f}" for v in scales if v is not None) or "-") + f", {en.summary.get('location_max_abs', 0):.2f}"
        lines.append(f"  {t:<17} {s.get('z_beyond_pct', 0):>6.1f}         {_by_size(s.get('z_beyond_pct_by_size'), '{:.1f}'):<22}  "
                     f"{_by_size(s.get('z_scale_by_size')):<28}  {en_txt}")
    allow = allowed(session)
    lines += ["", "Allowed choices (test | reference): " + ", ".join(allow)]
    blocked = []
    if int(inv.get("total", 0) or 0) > 0:
        blocked.append("Wald-type statistics (providers with all events or none)")
    for t in rules.TESTS:
        if t not in session.admissible.tests_computed:
            blocked.append(f"{t} (not computed for these data)")
    if blocked:
        lines.append("Not allowed: " + "; ".join(blocked))
    return "\n".join(lines)
