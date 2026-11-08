"""Execute one action of the closed set: build its payload, run the engine, record the result.

Every payload and every output is kept under the run directory (`payloads/`, `outputs/`), numbered by
step, which is what `repro.R` replays without the model. The harness steps `profile` and `verify` go
through the same bridge but are not planner actions.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from .. import actions, rules
from ..state import Result, Session
from . import rbridge

SCRIPTS = {
    "profile": "profile_data.R",
    "verify": "verify_spec.R",
    "fit": "fit_fe.R",
    "standardize": "standardize.R",
    "test_providers": "test_providers.R",
    "empirical_null": "empirical_null.R",
    "test_measure": "test_measure.R",
    "select_adjusters": "select_adjusters.R",
    "evaluate_model": "evaluate_model.R",
    "calibrate_tests": "calibrate_tests.R",
    "subset": "subset.R",
}
DEFAULT_TOL = 1e-6
DEFAULT_BOUND = 10.0
DEFAULT_SEED = 20261018


def _spec_payload(session: Session, adjusters: Optional[list] = None) -> Dict[str, Any]:
    s = session.spec
    return {"outcome": s.outcome, "trials": s.trials, "provider": s.provider,
            "patient": s.patient, "adjusters": list(adjusters or s.adjusters or []),
            "eligibility": s.eligibility, "crossed_with": s.crossed_with}


def _paths(session: Session, label: str):
    run = Path(session.run_dir)
    return run / "payloads" / f"{label}.json", run / "outputs" / f"{label}.json"


def _call(session: Session, script_key: str, payload: Dict[str, Any], label: str, timeout_s: int = 3600) -> Dict[str, Any]:
    pin, pout = _paths(session, label)
    return rbridge.run_r(SCRIPTS[script_key], payload, pin, pout, timeout_s=timeout_s)


# ------------------------------------------------------------------------------------- harness steps
def profile(session: Session) -> Dict[str, Any]:
    out = _call(session, "profile", {"data_path": session.data_path}, "00_profile")
    session.profile = out
    return out


def verify_card(session: Session) -> Dict[str, Any]:
    payload = {"data_path": session.data_path, "spec": _spec_payload(session)}
    return _call(session, "verify", payload, "00_verify")


# ------------------------------------------------------------------------------------- planner actions
def build_payload(session: Session, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    s = session.spec
    run_dir = session.run_dir
    a = dict(args or {})
    if name == "fit":
        return {"data_path": session.data_path, "run_dir": run_dir,
                "spec": _spec_payload(session, a.get("adjusters")), "tol": DEFAULT_TOL, "bound": DEFAULT_BOUND,
                "algorithm": "SerBIN"}
    if name == "standardize":
        stdz = a.get("stdz") or s.standardization or "both"
        return {"run_dir": run_dir, "stdz": ["indirect", "direct"] if stdz == "both" else [stdz],
                "norm": a.get("norm") or s.norm or "median"}
    if name == "test_providers":
        return {"run_dir": run_dir, "test": a["test"], "null": "median",
                "alternative": a.get("alternative") or "two.sided", "level": s.level or 0.95, "key": a["test"]}
    if name == "empirical_null":
        return {"run_dir": run_dir, "key": a["test"], "n_groups": int(a.get("n_groups") or 4),
                "size_by": a.get("size_by") or "patients", "level": s.level or 0.95}
    if name == "test_measure":
        p = {"run_dir": run_dir, "measure": "direct_rate", "se_source": a.get("se_source") or "model",
             "level": s.level or 0.95, "alternative": "two.sided",
             "key": f"direct_rate_{a.get('se_source') or 'model'}_{a.get('reference') or 'theoretical'}"}
        if a.get("reference") == "empirical":
            t = session.admissible.default_test if session.admissible else None
            if t and session.has(f"en:{t}"):
                p["en_key"] = t
        return p
    if name == "select_adjusters":
        cands = a.get("candidates") or s.select_from or s.adjusters
        p = {"data_path": session.data_path, "run_dir": run_dir, "spec": _spec_payload(session),
             "candidates": list(cands), "nfolds": int(a.get("nfolds") or 10), "seed": DEFAULT_SEED,
             "method": a.get("method") or ("group" if s.groups else "lasso")}
        if p["method"] == "group" and s.groups:
            p["groups"] = [int(s.groups.get(c, 1)) for c in cands]
        return p
    if name == "evaluate_model":
        return {"run_dir": run_dir, "n_groups": 10}
    if name == "calibrate_tests":
        adm = session.admissible
        tests = list(a.get("tests") or (adm.tests_computed if adm else []))
        if adm:                       # the reportable tests are always calibrated; the planner may add alternatives
            tests = [t for t in adm.tests_computed if t in set(tests) | set(adm.tests_reported)]
        # power is measured by default (planted departures); an explicit null effect_or asks for the size-only simulation
        effect_or = a["effect_or"] if "effect_or" in a else rules.CALIBRATION["effect_or"]
        return {"run_dir": run_dir, "tests": list(tests), "n_sim": int(a.get("n_sim") or 100),
                "max_providers": int(a.get("max_providers") or 500), "seed": DEFAULT_SEED, "level": s.level or 0.95,
                "n_groups": 4, "effect_or": None if effect_or is None else float(effect_or),
                "planted_frac": float(a.get("planted_frac") or rules.CALIBRATION["planted_frac"])}
    if name == "subset":
        return {"run_dir": run_dir, "provider": str(a.get("provider")), "by": list(a.get("by") or s.adjusters or []), "max_by": 8}
    raise ValueError(f"no payload for action {name}")


def execute(session: Session, name: str, args: Optional[Dict[str, Any]], source: str = "planner",
            reason: str = "") -> Result:
    """Run one action; the harness has already validated it."""
    args = dict(args or {})
    rid = actions.result_id(name, args)
    step = session.steps + 1
    label = f"{step:02d}_{rid.replace(':', '_')}"
    t0 = time.time()
    if name in SCRIPTS:
        out = _call(session, name, build_payload(session, name, args), label,
                    timeout_s=7200 if name in ("fit", "select_adjusters", "calibrate_tests") else 3600)
        summary = {k: v for k, v in out.items() if k not in ("tables", "_stderr", "status")}
        tables = dict(out.get("tables") or {})
    elif name == "compare_axes":
        from .. import compare
        summary, tables = compare.compare_axis(session, args.get("axis"))
    elif name == "figure":
        from . import pycore
        summary, tables = pycore.render_figure(session, args.get("display"), args.get("of"))
    else:
        raise ValueError(f"action {name} cannot be executed")
    res = Result(id=rid, action=name, args=args, summary=summary, tables=tables, step=step, source=source,
                 seconds=round(time.time() - t0, 2), reason=reason)
    session.add(res)
    session.steps = step
    session.cost_used += actions.CARDS[name].cost
    return res
