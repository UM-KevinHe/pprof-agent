"""The loop: plan -> verify -> run -> read -> re-plan, until finish.

The planner is a callable `planner(observation) -> proposal`, where a proposal is
    {"reading": str, "action": str, "args": dict, "reason": str}
Two planners exist: `default_planner`, which walks the protected core and finishes (no model; what the
tests and the no-model deployment use), and the model planner in `boundary.next_step` (later). The
harness validates every proposal (actions.validate); a refused proposal goes back with the problems
by name, MAX_TRIES times, after which the harness takes its own default next action and records the
fallback. The protected core always runs before `finish` is accepted, whoever planned.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from . import actions, rules
from .engine import executor
from .engine.rbridge import RError

MAX_STEPS = int(os.environ.get("PPROF_AGENT_MAX_STEPS", "24") or 24)   # planner steps; the core is completed by the harness after
MAX_TRIES = 3
Planner = Callable[[Dict[str, Any]], Dict[str, Any]]


def missing_core(session: Any) -> List[Any]:
    failed = set(getattr(session, "failed", []))
    return [(n, a) for n, a in actions.protected_core(session)
            if not session.has(actions.result_id(n, a)) and actions.result_id(n, a) not in failed]


def observe(session: Any, problems: Optional[List[str]] = None, last: Optional[Any] = None) -> Dict[str, Any]:
    """What the planner sees: the state line, the specification, the card, the admissible set with the
    cards, the results so far (summaries only, no table), the budget, and the last refusal's problems."""
    spec = session.spec
    adm = session.admissible
    return {
        "state_line": session.state_line(),
        "specification": spec.config() if spec else None,
        "sources": dict(spec.sources) if spec else {},
        "card": session.card,
        "admissible": adm.to_dict() if adm else None,
        "actions": actions.card_lines(session),
        "available": actions.available_lines(session),
        "core_pending": [actions.result_id(n, a) for n, a in missing_core(session)],
        "results": [{"id": r.id, "action": r.action, "args": r.args, "source": r.source,
                     "summary": _compact(r.summary)} for r in session.results],
        "last_result": None if last is None else {"id": last.id, "summary": _compact(last.summary)},
        "budget_left": session.budget - session.cost_used,
        "steps_left": MAX_STEPS - session.steps,
        "problems": list(problems or []),
    }


def _compact(summary: Dict[str, Any], limit: int = 40) -> Dict[str, Any]:
    """A summary small enough to show: scalars and short lists; tables and long lists are dropped."""
    out: Dict[str, Any] = {}
    for k, v in (summary or {}).items():
        if k in ("tables", "_stderr", "groups", "calibration"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and len(v) <= 12:
            out[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool)) or vv is None}
        elif isinstance(v, list) and len(v) <= limit and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = v
    return out


def default_planner(session: Any) -> Planner:
    """The harness's own plan: the protected core in order, then finish."""
    def plan(obs: Dict[str, Any]) -> Dict[str, Any]:
        pending = missing_core(session)
        if pending:
            name, args = pending[0]
            return {"reading": "", "action": name, "args": args, "reason": "protected core"}
        return {"reading": "", "action": "finish", "args": {}, "reason": "the protected core is complete"}
    return plan


def recommend(session: Any) -> Dict[str, Any]:
    """The reported configuration: the rule default test; the reference decided by the fitted empirical
    null of that test (rules.reference_rule); the evidence recorded. A calibration simulation, when one
    has run, overrides the test choice (not built yet)."""
    adm = session.admissible
    test = adm.default_test
    reasons = {"test": adm.reasons.get("variance") or adm.reasons.get("statistic") or "rule default",
               "naive": adm.reasons.get("naive"), "wald": adm.reasons.get("wald")}
    ref, why = "theoretical", adm.reasons.get("empirical_null", "")
    en = session.get(f"en:{test}")
    if en is not None:
        ref, why = rules.reference_rule(en.summary)
    cal = session.get("calibration")
    if cal is not None and isinstance(cal.summary.get("rates"), list):
        pick = calibrated_test(session, cal.summary["rates"], ref)
        if pick is not None:
            test, why_t = pick
            reasons["test"] = why_t
    rec = {"test": test, "reference": ref, "reasons": {**reasons, "reference": why},
           "statistic": rules.STATISTIC_OF[test], "variance": rules.VARIANCE_OF[test],
           "evidence": "calibration simulation" if cal is not None else "rules"}
    session.recommended = rec
    return rec


def calibrated_test(session: Any, rates: List[Dict[str, Any]], reference: str) -> Optional[Any]:
    """Among the admissible-as-reported tests, the one the calibration evidence picks under the reported reference.
    With simulated power (planted departures), the Neyman-Pearson score of rules.np_score over both tails; without,
    the test whose simulated type I error in the consequential tail is nearest the nominal one-sided level."""
    adm = session.admissible
    spec = session.spec
    tail_raw = ("higher" if spec.event_direction == "higher_is_worse" else "lower") if spec.consequential_tail != "better" \
        else ("lower" if spec.event_direction == "higher_is_worse" else "higher")
    if any(r.get("power_pct") is not None and r.get("power_pct") == r.get("power_pct") for r in rates):
        return _np_test(session, rates, reference, tail_raw)
    rows = [r for r in rates if r.get("reference") == reference and r.get("tail") == tail_raw
            and r.get("test") in adm.tests_reported]
    if not rows:
        return None
    nominal = float(rows[0]["nominal_pct"])
    def gap(r):
        return abs(float(r["rate_pct"]) - nominal)
    def calibrated(r):
        tol = max(2 * float(r.get("se_pct") or 0), 0.5)
        return gap(r) <= tol
    cal = [r for r in rows if calibrated(r)]
    default = next((r for r in rows if r["test"] == adm.default_test), None)
    if default is None:
        return None               # the rule default was not simulated: no evidence against it, it stands
    n_tests = len({r["test"] for r in rates})
    if default is not None and calibrated(default):
        return default["test"], (f"the calibration simulation ({n_tests} tests) put the rule default {default['test']}'s type I error "
                                 f"in the {spec.consequential_tail} tail at {float(default['rate_pct']):.1f}% against a nominal "
                                 f"{nominal:.1f}% under the {reference} null, within simulation error; it stands")
    pool = cal or rows
    best = min(pool, key=gap)
    why = (f"the calibration simulation ({n_tests} tests) put {best['test']}'s type I error in the {spec.consequential_tail} tail at "
           f"{float(best['rate_pct']):.1f}% against a nominal {nominal:.1f}% under the {reference} null")
    if default is not None:
        why += f"; the rule default {default['test']} was at {float(default['rate_pct']):.1f}%"
    why += ", so the calibrated test is reported" if cal else ", the nearest of the admissible tests (none within simulation error)"
    return best["test"], why


def np_scores(session: Any, rates: List[Dict[str, Any]], reference: str, tail_raw: str) -> Dict[str, float]:
    """Each reportable test's Neyman-Pearson score under the reference, weighting the consequential tail."""
    adm = session.admissible
    other = "lower" if tail_raw == "higher" else "higher"
    by = {(r.get("test"), r.get("tail")): r for r in rates if r.get("reference") == reference}
    w = rules.CALIBRATION["w_consequential"]
    def tail_score(r):
        return rules.np_score(float(r["rate_pct"]), r.get("power_pct"), float(r["nominal_pct"]))
    return {t: w * tail_score(by[(t, tail_raw)]) + (1 - w) * tail_score(by[(t, other)])
            for t in adm.tests_reported if (t, tail_raw) in by and (t, other) in by}


def _np_test(session: Any, rates: List[Dict[str, Any]], reference: str, tail_raw: str) -> Optional[Any]:
    """The reportable test with the best Neyman-Pearson score (ties keep the rule default), with the evidence in words."""
    adm, spec = session.admissible, session.spec
    scores = np_scores(session, rates, reference, tail_raw)
    default = adm.default_test
    if default not in scores:
        return None               # the rule default was not simulated: no evidence against it, it stands
    best = max(scores, key=lambda t: (scores[t], t == default))
    row = {r.get("test"): r for r in rates if r.get("reference") == reference and r.get("tail") == tail_raw}
    def pct(v):
        return 0.0 if v is None or v != v else float(v)
    b, d = row[best], row[default]
    n_tests = len({r["test"] for r in rates})
    why = (f"the calibration simulation ({n_tests} tests, departures of odds ratio {float(b.get('effect_or') or 0):.2f} planted in some "
           f"providers) scored each reportable test by its power, less a penalty for type I error beyond the nominal level by more "
           f"than a tenth: {best} had type I error {pct(b['rate_pct']):.1f}% and power {pct(b.get('power_pct')):.1f}% in the "
           f"{spec.consequential_tail} tail against a nominal {float(b['nominal_pct']):.1f}% under the {reference} null")
    if best == default:
        why += ", and it is the rule default; it stands"
    else:
        why += (f"; the rule default {default} had type I error {pct(d['rate_pct']):.1f}% and power {pct(d.get('power_pct')):.1f}%, "
                "so the calibrated test is reported")
    return best, why


def run_loop(session: Any, planner: Optional[Planner] = None, max_steps: int = MAX_STEPS,
             max_tries: int = MAX_TRIES, log: Callable[[str], None] = print) -> Any:
    if session.verification is None or not session.verification.ok:
        raise RuntimeError("the specification is not verified; nothing runs")
    planner_source = "planner" if planner is not None else "harness"
    planner = planner or default_planner(session)
    last = None
    problems: List[str] = []
    while session.steps < max_steps and not session.finished:
        obs = observe(session, problems, last)
        proposal: Optional[Dict[str, Any]] = None
        source = planner_source
        for attempt in range(max_tries):
            cand = planner(obs) or {}
            name, args = cand.get("action"), dict(cand.get("args") or {})
            probs = actions.validate(name, args, session)
            if name == "finish" and probs and all("protected core" in p for p in probs):
                probs = []            # the harness runs the core itself before finishing
            supplied: List[str] = []
            pre = actions.prerequisites_for(probs, session) if probs else None
            if pre:                   # an admissible step whose prerequisites the harness can supply: supply them, then accept
                for pn, pa in pre:
                    if session.has(actions.result_id(pn, pa)) or session.steps >= max_steps or actions.validate(pn, pa, session):
                        continue
                    try:
                        res = executor.execute(session, pn, pa, source="harness", reason=f"prerequisite of the planner's {name}")
                        supplied.append(res.id)
                        last = res
                        log(f"step {res.step}: {res.id} ({res.seconds}s) prerequisite of the planner's {name}")
                    except RError as exc:
                        session.refusals.append(f"{pn} failed in the engine: {exc.info.get('message', '')}")
                        session.failed.append(actions.result_id(pn, pa))
                        session.steps += 1
                if supplied:
                    probs = actions.validate(name, args, session)
            session.plan_log.append({"step": session.steps + 1, "attempt": attempt + 1, "proposal": cand,
                                     "problems": probs, **({"prerequisites_run": supplied} if supplied else {})})
            if not probs:
                proposal = cand
                break
            obs = observe(session, probs, last)
            problems = probs
        if proposal is None:
            pending = missing_core(session)
            if pending:
                name, args = pending[0]
                proposal = {"reading": "", "action": name, "args": args, "reason": "harness default after refusals"}
            else:
                proposal = {"reading": "", "action": "finish", "args": {}, "reason": "harness default after refusals"}
            source = "harness_default"
            session.plan_log.append({"step": session.steps + 1, "fallback": proposal})
        name, args = proposal["action"], dict(proposal.get("args") or {})
        if name == "finish":
            _run_core(session, log)
            session.finished = True
            log(f"finish: {proposal.get('reason', '')}")
            break
        try:
            last = executor.execute(session, name, args, source=source, reason=proposal.get("reason", ""))
            problems = []
            log(f"step {last.step}: {last.id} ({last.seconds}s) {proposal.get('reason', '')}")
        except RError as exc:
            problems = [f"{name} failed in the engine: {exc.info.get('message', '')}"]
            session.refusals.append(problems[0])
            session.failed.append(actions.result_id(name, args))
            session.plan_log.append({"step": session.steps + 1, "engine_error": exc.info})
            log(f"engine error in {name}: {exc.info.get('message', '')}")
            session.steps += 1
    if not session.finished:
        _run_core(session, log)
        session.finished = True
        log("max steps reached: the protected core was completed and the analysis finished")
    recommend(session)
    for axis in ("statistic", "variance", "reference", "standardization"):
        try:
            executor.execute(session, "compare_axes", {"axis": axis}, source="harness", reason="flags under alternatives")
        except Exception as exc:  # noqa: BLE001
            session.refusals.append(f"compare_axes {axis}: {exc}")
    return session


def _run_core(session: Any, log: Callable[[str], None]) -> None:
    for name, args in missing_core(session):
        try:
            res = executor.execute(session, name, args, source="core", reason="protected core")
            log(f"core {res.step}: {res.id} ({res.seconds}s)")
        except RError as exc:
            session.refusals.append(f"{name} failed in the engine: {exc.info.get('message', '')}")
            session.failed.append(actions.result_id(name, args))
            log(f"core engine error in {name}: {exc.info.get('message', '')}")
