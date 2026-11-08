"""The closed action set: the operations the planner may call, each with a card.

A card is what the planner reads (purpose, arguments and their allowed values, cost, what it needs);
`validate()` is what the harness checks before anything runs. Arguments are enumerations or column
lists, never free text that reaches the engine. A planner that proposes an action outside this set,
or an argument outside its enumeration, is refused with the reason by name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import rules

DISPLAYS = ("funnel", "caterpillar", "observed_expected", "coefficient_forest", "null_calibration",
            "flag_stability", "data_quality", "calibration_plot")


@dataclass(frozen=True)
class Card:
    name: str
    purpose: str
    args: Dict[str, Dict[str, Any]]
    cost: int
    needs: Tuple[str, ...]
    produces: str
    available: bool = True
    note: str = ""


CARDS: Dict[str, Card] = {
    "fit": Card(
        "fit", "fit the fixed provider effects logistic model on the declared adjusters (SerBIN)",
        {"adjusters": {"type": "columns", "doc": "the adjustment set; null for the declared set"}},
        cost=2, needs=(), produces="fit"),
    "select_adjusters": Card(
        "select_adjusters", "select the adjustment set by penalized regression with the provider effects kept unpenalized (cross-validated lasso or group lasso)",
        {"candidates": {"type": "columns", "doc": "candidate columns; null for the declared candidate set"},
         "method": {"enum": ["lasso", "group"]}, "nfolds": {"type": "int", "min": 3, "max": 20}},
        cost=5, needs=(), produces="selection"),
    "standardize": Card(
        "standardize", "indirect and direct standardized ratios and rates against the norm",
        {"stdz": {"enum": ["indirect", "direct", "both"]}, "norm": {"enum": ["median", "size_weighted_mean"]}},
        cost=1, needs=("fit",), produces="sm"),
    "test_providers": Card(
        "test_providers", "test every provider's effect against the norm with one statistic and variance treatment",
        {"test": {"enum": list(rules.TESTS)}},
        cost=1, needs=("fit",), produces="test"),
    "empirical_null": Card(
        "empirical_null", "fit the size-stratified empirical null to one test's statistics and re-flag under it",
        {"test": {"enum": list(rules.TESTS)}, "n_groups": {"type": "int", "min": 2, "max": 10},
         "size_by": {"enum": ["patients", "rows"]}},
        cost=1, needs=("test:{test}",), produces="en"),
    "test_measure": Card(
        "test_measure", "test the directly standardized rate on its own scale (the transported test)",
        {"se_source": {"enum": ["model", "robust_wald", "stabilized_wald"]},
         "reference": {"enum": ["theoretical", "empirical"]}},
        cost=1, needs=("sm",), produces="measure"),
    "evaluate_model": Card(
        "evaluate_model", "discrimination (C statistic overall and within provider) and calibration by decile of the risk-adjustment model",
        {}, cost=2, needs=("fit",), produces="evaluation"),
    "compare_axes": Card(
        "compare_axes", "flags under the other admissible choice on one axis, with the providers that change status",
        {"axis": {"enum": ["statistic", "variance", "reference", "standardization"]}},
        cost=0, needs=("fit",), produces="comparison"),
    "figure": Card(
        "figure", "render one display from the catalogue for one result",
        {"display": {"enum": list(DISPLAYS)}, "of": {"type": "result_id"}},
        cost=0, needs=(), produces="figure"),
    "calibrate_tests": Card(
        "calibrate_tests", "simulate this design (outcomes drawn from the fitted risk model at the norm, with the within-patient correlation matched), with departures of a given odds ratio planted in a small share of the providers, and measure each admissible test's type I error per tail on the providers at the norm and its power on the planted ones, under both reference distributions; the reported test is then the one with the most power whose type I error stays near the nominal level",
        {"tests": {"type": "tests", "doc": "tests to calibrate beyond the reportable ones (always included); null for all computed"},
         "n_sim": {"type": "int", "min": 20, "max": 500}, "max_providers": {"type": "int", "min": 50, "max": 5000},
         "effect_or": {"type": "float", "min": 1.1, "max": 4.0, "doc": "the planted departure as an odds ratio (default 1.5)"},
         "planted_frac": {"type": "float", "min": 0.02, "max": 0.2, "doc": "the share of providers planted (default 0.05)"}},
        cost=10, needs=("fit",), produces="calibration"),
    "subset": Card(
        "subset", "one provider's observed and expected events by patient group, and its case mix against the population (why is this provider flagged?)",
        {"provider": {"type": "provider_id"}, "by": {"type": "columns", "doc": "adjusters to split by; null for all"}},
        cost=2, needs=("fit",), produces="subset"),
    "finish": Card("finish", "end the analysis and write the report", {}, cost=0, needs=(), produces="finish"),
}


def result_id(name: str, args: Dict[str, Any]) -> str:
    a = args or {}
    if name == "test_providers":
        return f"test:{a.get('test')}"
    if name == "empirical_null":
        return f"en:{a.get('test')}"
    if name == "test_measure":
        return f"measure:direct_rate:{a.get('se_source', 'model')}:{a.get('reference', 'theoretical')}"
    if name == "compare_axes":
        return f"compare:{a.get('axis')}"
    if name == "subset":
        return f"subset:{a.get('provider')}"
    if name == "figure":
        return f"figure:{a.get('display')}:{a.get('of')}"
    card = CARDS.get(name)
    return card.produces if card is not None else name


def _needs(card: Card, args: Dict[str, Any]) -> List[str]:
    return [n.format(**{k: (args or {}).get(k) for k in ("test",)}) for n in card.needs]


def how_to_get(need: str) -> str:
    """The action that produces a result id, so a refusal names the step that would lift it."""
    if need == "fit":
        return "run fit"
    if need == "sm":
        return "run standardize (it produces sm)"
    if need.startswith("test:"):
        return f"run test_providers with test {need[5:]} (it produces {need})"
    if need.startswith("en:"):
        return f"run empirical_null with test {need[3:]} (it produces {need})"
    if need == "selection":
        return "run select_adjusters"
    return f"compute {need}"


def validate(name: str, args: Optional[Dict[str, Any]], session: Any, budget: bool = True) -> List[str]:
    """Problems with a proposed action, by name; empty when the harness accepts it. `budget=False` for an
    action the harness runs on the analyst's behalf after the loop (a breakdown for a question)."""
    problems: List[str] = []
    args = dict(args or {})
    card = CARDS.get(name)
    if card is None:
        return [f"{name} is not an action of this system (choose from {', '.join(CARDS)})"]
    if not card.available:
        return [f"{name}: {card.note}"]
    adm = getattr(session, "admissible", None)
    spec = getattr(session, "spec", None)
    for k, v in args.items():
        if k not in card.args and k not in ("reason",):
            problems.append(f"{name} takes no argument {k}")
    for k, rule in card.args.items():
        v = args.get(k)
        if v is None:
            continue
        if "enum" in rule and v not in rule["enum"]:
            problems.append(f"{name}.{k} must be one of {rule['enum']}, not {v!r}")
        if rule.get("type") == "int":
            try:
                iv = int(v)
            except (TypeError, ValueError):
                problems.append(f"{name}.{k} must be an integer")
                continue
            if iv < rule.get("min", -10**9) or iv > rule.get("max", 10**9):
                problems.append(f"{name}.{k} must be between {rule.get('min')} and {rule.get('max')}")
        if rule.get("type") == "float":
            try:
                fv = float(v)
            except (TypeError, ValueError):
                problems.append(f"{name}.{k} must be a number")
                continue
            if not (rule.get("min", -1e18) <= fv <= rule.get("max", 1e18)):
                problems.append(f"{name}.{k} must be between {rule.get('min')} and {rule.get('max')}")
        if rule.get("type") == "columns":
            cols = [c["name"] for c in (getattr(session, "profile", None) or {}).get("columns", [])]
            bad = [c for c in (v if isinstance(v, list) else [v]) if cols and c not in cols]
            if bad:
                problems.append(f"{name}.{k}: not columns of the file: {', '.join(map(str, bad))}")
        if rule.get("type") == "tests" and adm is not None:
            bad = [t for t in (v if isinstance(v, list) else [v]) if t not in adm.tests_computed]
            if bad:
                problems.append(f"{name}.{k}: not admissible here: {', '.join(map(str, bad))}")
        if rule.get("type") == "provider_id":
            fit = session.get("fit") if hasattr(session, "get") else None
            if fit is not None and fit.tables.get("gamma"):
                import csv
                with open(fit.tables["gamma"], encoding="utf-8") as fh:
                    ids = {row["provider"] for row in csv.DictReader(fh)}
                if str(v) not in ids:
                    problems.append(f"{name}.{k}: {v} is not a fitted provider")
    if name == "test_providers" and adm is not None:
        t = args.get("test")
        if t and t not in adm.tests_computed:
            problems.append(f"test {t} is not admissible here: {adm.reasons.get('stabilized') or adm.reasons.get('naive')}")
    if name == "empirical_null" and adm is not None and "empirical" not in adm.references_computed:
        problems.append(f"no empirical null here: {adm.reasons.get('empirical_null')}")
    if name == "test_measure" and adm is not None and not adm.direct_transport:
        problems.append("the specification does not include direct standardization")
    if name == "select_adjusters" and spec is not None and not (args.get("candidates") or spec.select_from):
        problems.append("no candidate set to select from: the specification declares a fixed adjustment set")
    if result_id(name, args) in getattr(session, "failed", ()):
        problems.append(f"{name} with these arguments failed in the engine earlier in this run; it is not retried")
    if name != "finish" and session.has(result_id(name, args)):
        problems.append(f"{name} with these arguments has already run ({result_id(name, args)} is in the ledger); choose another step or finish")
    if name == "test_measure" and args.get("se_source") in ("robust_wald", "stabilized_wald") \
            and not session.has(f"test:{args['se_source']}"):
        problems.append(f"test_measure with se_source {args['se_source']} needs test:{args['se_source']} first: "
                        + how_to_get(f"test:{args['se_source']}"))
    if name == "test_measure" and args.get("reference") == "empirical" and adm is not None \
            and not session.has(f"en:{adm.default_test}"):
        problems.append(f"test_measure under the empirical null needs en:{adm.default_test} first: " + how_to_get(f"en:{adm.default_test}"))
    for need in _needs(card, args):
        if not session.has(need):
            problems.append(f"{name} needs {need} first: " + how_to_get(need))
    if name == "figure":
        of = args.get("of")
        if of and not session.has(of):
            problems.append(f"figure: no result {of}")
    budget_left = getattr(session, "budget", 0) - getattr(session, "cost_used", 0)
    if budget and card.cost > 0 and card.cost > budget_left:        # a free action (a figure, a comparison) is never refused for budget
        problems.append(f"{name} costs {card.cost} units; {budget_left} left")
    if name == "finish":
        pending = [n for n, _ in protected_core(session) if not session.has(result_id(n, _))]
        if pending:
            problems.append("the protected core is not complete; the harness will run it before finishing: " + ", ".join(pending))
    return problems


_NEEDS = re.compile(r"needs (\S+) first")


def prerequisites_for(problems: List[str], session: Any) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """When every problem of a proposal is a missing prerequisite the harness can supply (a fit, the standardization,
    a test, an empirical null), the actions that supply them, in order; None when any problem is of another kind."""
    spec, adm = getattr(session, "spec", None), getattr(session, "admissible", None)
    if spec is None or adm is None or not problems:
        return None
    out: List[Tuple[str, Dict[str, Any]]] = []

    def supply(need: str) -> bool:
        """Add the action that produces `need`, after whatever it needs itself (fit before a test, the test before its null)."""
        if need == "fit":
            out.append(("fit", {"adjusters": None}))
        elif need == "sm":
            if not session.has("fit"):
                supply("fit")
            out.append(("standardize", {"stdz": spec.standardization or "both", "norm": spec.norm or "median"}))
        elif need.startswith("test:") and need[5:] in adm.tests_computed:
            if not session.has("fit"):
                supply("fit")
            out.append(("test_providers", {"test": need[5:]}))
        elif need.startswith("en:") and need[3:] in adm.tests_computed and "empirical" in adm.references_computed:
            if not session.has(f"test:{need[3:]}"):
                supply(f"test:{need[3:]}")
            out.append(("empirical_null", {"test": need[3:], "n_groups": 4, "size_by": "patients"}))
        else:
            return False
        return True

    for p in problems:
        m = _NEEDS.search(p)
        if not m or not supply(m.group(1).rstrip(":;,")):
            return None
    seen = set()
    uniq = []
    for n, a in out:
        rid = result_id(n, a)
        if rid not in seen:
            seen.add(rid)
            uniq.append((n, a))
    return uniq


def protected_core(session: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """What always runs whatever the planner says, so no admissible alternative is ever suppressed."""
    spec = session.spec
    adm = session.admissible
    if spec is None or adm is None:
        return []
    core: List[Tuple[str, Dict[str, Any]]] = [("fit", {"adjusters": None})]
    stdz = spec.standardization or "both"
    core.append(("standardize", {"stdz": stdz, "norm": spec.norm or "median"}))
    for t in adm.tests_computed:
        core.append(("test_providers", {"test": t}))
    if "empirical" in adm.references_computed:
        for t in adm.tests_reported:
            core.append(("empirical_null", {"test": t, "n_groups": 4, "size_by": "patients"}))
    if adm.direct_transport:
        se_source = "robust_wald" if "robust_wald" in adm.tests_computed else "model"
        core.append(("test_measure", {"se_source": se_source, "reference": "theoretical"}))
    core.append(("evaluate_model", {}))
    return core


def card_lines(session: Any) -> List[str]:
    """The cards as the planner sees them, with the enumerations narrowed to what is admissible here."""
    adm = session.admissible
    lines = []
    for c in CARDS.values():
        if not c.available:
            continue
        args = []
        for k, rule in c.args.items():
            if "enum" in rule:
                vals = rule["enum"]
                if c.name in ("test_providers", "empirical_null") and k == "test" and adm is not None:
                    vals = adm.tests_computed
                args.append(f"{k} in {{{', '.join(map(str, vals))}}}")
            elif rule.get("type") == "int":
                args.append(f"{k}: integer between {rule.get('min')} and {rule.get('max')}")
            elif rule.get("type") == "float":
                args.append(f"{k}: number between {rule.get('min')} and {rule.get('max')}")
            elif rule.get("type") == "result_id":
                args.append(f"{k}: the id of a result in the ledger")
            elif rule.get("type") == "provider_id":
                args.append(f"{k}: the id of a fitted provider")
            else:
                args.append(f"{k}: {rule.get('type')}")
        lines.append(f"- {c.name} (cost {c.cost}): {c.purpose}" + (f"; args: {'; '.join(args)}" if args else "")
                     + (f"; needs {', '.join(c.needs)}" if c.needs else ""))
    return lines


def calibration_seconds(session: Any, n_sim: int, max_providers: int = 500) -> Optional[float]:
    """A rough cost of calibrate_tests: every simulated data set is refitted and every calibrated test re-run, on
    at most `max_providers` providers. From the measured seconds of the fit and of the tests run so far (a test not
    yet run is taken at half a fit); None before the fit."""
    fit = session.get("fit") if hasattr(session, "get") else None
    if fit is None:
        return None
    fs = fit.summary or {}
    fit_s = float(fs.get("fit_seconds") or fs.get("seconds") or fit.seconds or 0)
    adm = getattr(session, "admissible", None)
    tests = list(adm.tests_computed) if adm else []
    per_test = []
    for t in tests:
        r = session.get(f"test:{t}")
        per_test.append(float((r.summary or {}).get("seconds") or r.seconds or 0) if r is not None else fit_s / 2)
    n_prov = int(fs.get("n_providers_fitted") or 0)
    share = min(1.0, max_providers / n_prov) if n_prov else 1.0
    return n_sim * share * (fit_s + sum(per_test))


def available_lines(session: Any) -> List[str]:
    """The concrete steps the harness would accept now (not yet run, prerequisites met, affordable), each with
    its arguments: what the planner may still propose. Figures and breakdowns are described, not enumerated."""
    adm = session.admissible
    spec = session.spec
    if adm is None or spec is None:
        return []
    cands: List[Tuple[str, Dict[str, Any]]] = [("fit", {"adjusters": None})]
    if spec.select_from:
        cands.append(("select_adjusters", {"method": "lasso"}))
    cands.append(("standardize", {"stdz": spec.standardization or "both", "norm": spec.norm or "median"}))
    cands += [("test_providers", {"test": t}) for t in adm.tests_computed]
    if "empirical" in adm.references_computed:
        cands += [("empirical_null", {"test": t, "n_groups": 4, "size_by": "patients"}) for t in adm.tests_computed]
    if adm.direct_transport:
        cands += [("test_measure", {"se_source": se, "reference": ref}) for se in ("model", "robust_wald", "stabilized_wald")
                  for ref in ("theoretical", "empirical")]
    cands.append(("evaluate_model", {}))
    cands += [("compare_axes", {"axis": a}) for a in ("statistic", "variance", "reference", "standardization")]
    cands.append(("calibrate_tests", {"n_sim": 30}))
    out = []
    for name, args in cands:
        if not validate(name, args, session):
            line = name + "".join(f" {k}={v}" for k, v in args.items() if v is not None)
            if name == "calibrate_tests":
                est = calibration_seconds(session, int(args.get("n_sim") or 30))
                if est is not None:
                    line += f" (about {max(1, round(est / 60))} min)" if est >= 60 else " (under a minute)"
            out.append(line)
    if session.has("fit"):
        out.append("subset provider=<a fitted provider id>")
    if session.results:
        out.append("figure display=<from the catalogue> of=<a result id in the ledger>")
    out.append("finish (the harness then completes the protected core and writes the report)")
    return out
