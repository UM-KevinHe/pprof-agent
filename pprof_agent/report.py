"""The report: harness-rendered tables and a closed set of named references.

Every number the analyst sees is read from a table the engine wrote or from a result summary; the
model (when one is configured) writes prose with `[reference]` names, which `resolve()` fills, and
`check_no_digits()` refuses prose that carries a digit. This module renders Markdown; the HTML
report with figures is assembled by `engine.pycore` from the same pieces.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import compare, rules
from .specification import consequence_lines

_REF = re.compile(r"\[([a-z_0-9:]+)\]")
_DIGIT = re.compile(r"\d")


def _fmt(x: Any) -> str:
    if x is None:
        return "not available"
    if isinstance(x, float):
        if x != x:
            return "not available"
        return f"{x:.4g}" if abs(x) < 1e-3 or abs(x) >= 1e5 else f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def build_references(session: Any) -> Dict[str, Any]:
    """The typed set of quantities prose may name."""
    spec, card, adm = session.spec, session.card, session.admissible
    rec = session.recommended or {}
    refs: Dict[str, Any] = {
        "n_rows": card.get("rows_complete"), "n_patients": card.get("patients"), "n_providers": card.get("providers"),
        "n_eligible": card.get("eligible_providers"), "eligibility": card.get("eligibility"),
        "event_rate_pct": round(100 * (card.get("event_rate") or 0), 1), "n_invariant": (card.get("invariant_providers") or {}).get("total"),
        "rows_per_patient_max": card.get("max_rows_per_patient"),
        "patients_per_provider_median": (card.get("patients_per_provider") or {}).get("median"),
        "outcome": spec.outcome, "unit": spec.unit, "event_direction": spec.event_direction,
        "provider_column": spec.provider, "patient_column": spec.patient, "trials_column": spec.trials,
        "n_adjusters": len(spec.adjusters or []), "standardization": spec.standardization, "norm": spec.norm,
        "consequential_tail": spec.consequential_tail, "level": spec.level,
        "recommended_test": rec.get("test"), "recommended_reference": rec.get("reference"),
        "recommended_statistic": rec.get("statistic"), "recommended_variance": rec.get("variance"),
    }
    refs.update({"few_patients_threshold": int(rules.THRESHOLDS["few_patients_median"]), "en_min_providers": int(rules.THRESHOLDS["en_min_providers"]),
                 "stabilized_max_rows": int(rules.THRESHOLDS["stabilized_max_rows"]), "exact_max_trials": int(rules.THRESHOLDS["exact_max_trials"]),
                 "en_scale_threshold": float(rules.THRESHOLDS["en_scale_max"]), "en_location_threshold": float(rules.THRESHOLDS["en_location_max"])})
    fit = session.get("fit")
    if fit:
        refs.update({"auc": fit.summary.get("auc"), "iterations": fit.summary.get("iterations"),
                     "n_providers_fitted": fit.summary.get("n_providers_fitted"),
                     "n_providers_excluded": fit.summary.get("n_providers_excluded")})
    ev = session.get("evaluation")
    if ev:
        refs.update({"c_overall": ev.summary.get("c_overall"), "c_within": ev.summary.get("c_within_provider")})
    sm = session.get("sm")
    if sm:
        refs.update({"sm_flips": (sm.summary.get("flips") or {}).get("n"),
                     "population_rate_pct": round(100 * (sm.summary.get("population_rate") or 0), 1)})
    t = rec.get("test")
    if t:
        df = compare.table_of(session, t, rec.get("reference", "theoretical"))
        c = compare.counts(df)
        refs.update({"flagged_worse": c["worse"], "flagged_better": c["better"], "flagged_as_expected": c["as_expected"],
                     "flagged_worse_pct": round(100 * c["worse"] / max(len(df), 1), 1) if df is not None else None,
                     "flagged_better_pct": round(100 * c["better"] / max(len(df), 1), 1) if df is not None else None})
        en = session.get(f"en:{t}")
        if en:
            refs.update({"en_scale_max": en.summary.get("scale_max"), "en_scale_ratio": en.summary.get("scale_ratio"),
                         "en_location_max": en.summary.get("location_max_abs")})
    for axis in ("statistic", "variance", "reference", "standardization"):
        r = session.get(f"compare:{axis}")
        if r:
            refs[f"changed_max_{axis}"] = r.summary.get("max_changed")
            for c in r.summary.get("comparisons") or []:
                tag = compare.alternative_tag(axis, c["alternative"])
                refs[f"worse_under_{tag}"] = c["worse_alt"]
                refs[f"better_under_{tag}"] = c["better_alt"]
                refs[f"changed_under_{tag}"] = c["changed"]
    sel = session.get("selection")
    if sel:
        refs.update({"n_selected": sel.summary.get("n_selected"), "n_candidates": sel.summary.get("n_candidates")})
    stab = compare.stability(session)
    if stab:
        refs.update({"worse_under_every_alternative": len(stab["worse"]["stable"]),
                     "better_under_every_alternative": len(stab["better"]["stable"]), "n_alternatives": stab["n_alternatives"]})
    cal = session.get("calibration")
    if cal and t and isinstance(cal.summary.get("rates"), list):
        tail = consequential_tail_raw(spec)
        row = next((r for r in cal.summary["rates"] if r.get("test") == t and r.get("reference") == rec.get("reference", "theoretical")
                    and r.get("tail") == tail), None)
        if row is not None:
            refs.update({"calibration_rate_pct": round(float(row["rate_pct"]), 1), "calibration_nominal_pct": float(row["nominal_pct"]),
                         "calibration_n_sim": int(row.get("n_sim") or cal.summary.get("n_sim") or 0)})
            if _num(row.get("power_pct")) is not None:
                refs.update({"calibration_power_pct": round(float(row["power_pct"]), 1),
                             "calibration_effect_or": float(cal.summary.get("effect_or") or row.get("effect_or"))})
        if adm is not None and adm.default_test != t:
            drow = next((r for r in cal.summary["rates"] if r.get("test") == adm.default_test and r.get("reference") == rec.get("reference", "theoretical")
                         and r.get("tail") == tail), None)
            if drow is not None:
                refs["calibration_default_rate_pct"] = round(float(drow["rate_pct"]), 1)
                if _num(drow.get("power_pct")) is not None:
                    refs["calibration_default_power_pct"] = round(float(drow["power_pct"]), 1)
    return refs


def _num(v: Any) -> Optional[float]:
    """A finite number, or None (a missing value arrives as None or NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def consequential_tail_raw(spec: Any) -> str:
    """The raw direction (higher / lower than the norm) of the consequential tail."""
    worse_is_higher = spec.event_direction == "higher_is_worse"
    if spec.consequential_tail == "better":
        return "lower" if worse_is_higher else "higher"
    return "higher" if worse_is_higher else "lower"


def resolve(text: str, refs: Dict[str, Any]) -> str:
    """Fill the references; a percent-valued reference carries its sign so it reads as one."""
    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key not in refs:
            raise KeyError(f"unknown reference [{key}]")
        v = refs[key]
        s = _fmt(v)
        if key.endswith("_pct") and isinstance(v, (int, float)) and not isinstance(v, bool) and s != "not available":
            s += "%"
        return s
    text = _REF.sub(sub, text)
    return text.replace("%%", "%")


def check_no_digits(text: str) -> Optional[str]:
    stripped = _REF.sub("", text)
    if _DIGIT.search(stripped):
        return "the prose contains a digit; numbers enter only as [references]"
    return None


# ----------------------------------------------------------------------------------------- tables
def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_none_"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in the table file_")
    return "\n".join(lines)


def _read(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not Path(path).exists():
        return None
    return pd.read_csv(path, dtype={"provider": str, "term": str})


def flagged_table(session: Any, test: str, reference: str, tail: str, max_rows: int = 30) -> Optional[pd.DataFrame]:
    df = compare.table_of(session, test, reference)
    if df is None:
        return None
    d = df[df["status"] == tail].copy()
    zcol = "z_en" if reference == "empirical" and "z_en" in d.columns else "stat"
    d["abs_z"] = d[zcol].abs()
    d = d.sort_values("abs_z", ascending=False)
    cols = [c for c in ("provider", "observed", "expected", "estimate", zcol, "p", "n_patients", "n_rows") if c in d.columns]
    return d[cols].head(max_rows)


def render_markdown(session: Any, prose: Optional[Dict[str, str]] = None, figures: Optional[List[Dict[str, Any]]] = None) -> str:
    spec, card, adm = session.spec, session.card, session.admissible
    rec = session.recommended or {}
    refs = build_references(session)
    prose = prose or {}
    L: List[str] = []
    L.append(f"# Provider profiling report: {spec.outcome} per {spec.unit}\n")
    L.append("## 1. The measure as declared\n")
    src = spec.sources
    rows = [{"field": f, "value": _fmt(getattr(spec, f)) if f != "adjusters" else f"{len(spec.adjusters or [])} columns",
             "source": src.get(f, "")} for f in ("outcome", "event_direction", "unit", "provider", "patient", "eligibility",
                                                   "adjusters", "standardization", "norm", "reference", "consequential_tail", "level")]
    L.append(_md_table(pd.DataFrame(rows)))
    L.append("\n**Consequences on this file**\n")
    L += [f"- {line}" for line in consequence_lines(spec, card)]
    if session.verification and session.verification.warnings:
        L.append("\n**Warnings**\n")
        L += [f"- {w}" for w in session.verification.warnings]
    if prose.get("declared"):
        L.append("\n" + resolve(prose["declared"], refs))

    L.append("\n## 2. What the data admit\n")
    L.append(f"- statistics computed: {', '.join(adm.tests_computed)}")
    L.append(f"- recommended as the reported test: **{rec.get('test')}** ({rules.STATISTIC_OF.get(rec.get('test', ''), '')} statistic, "
             f"{rules.VARIANCE_OF.get(rec.get('test', ''), '')} variance). {rec.get('reasons', {}).get('test', '')}")
    for k in ("naive", "wald"):
        if adm.reasons.get(k):
            L.append(f"- {adm.reasons[k]}")
    L.append(f"- reference distribution: **{rec.get('reference')}**. {rec.get('reasons', {}).get('reference', '')}")
    for c in adm.caveats:
        L.append(f"- caveat: {c}")

    L.append("\n## 3. The risk-adjustment model\n")
    fit = session.get("fit")
    if fit:
        s = fit.summary
        L.append(f"- fixed provider effects logistic model ({s.get('algorithm')}), {_fmt(s.get('n_rows_fitted'))} rows, "
                 f"{_fmt(s.get('n_providers_fitted'))} providers fitted ({_fmt(s.get('n_providers_excluded'))} below the eligibility threshold), "
                 f"{_fmt(s.get('p'))} adjusters, {_fmt(s.get('iterations'))} iterations; AUC {_fmt(s.get('auc'))}")
        beta = _read((fit.tables or {}).get("beta"))
        if beta is not None:
            L.append("\n" + _md_table(beta, max_rows=60))
    sel = session.get("selection")
    if sel:
        L.append(f"\n- adjusters selected by cross-validated {sel.summary.get('method')} with the provider effects unpenalized: "
                 f"{_fmt(sel.summary.get('n_selected'))} of {_fmt(sel.summary.get('n_candidates'))} candidates kept; dropped: "
                 f"{', '.join(sel.summary.get('dropped') or []) or 'none'}")
    ev = session.get("evaluation")
    if ev:
        L.append(f"\n- discrimination: C statistic overall {_fmt(ev.summary.get('c_overall'))}, within provider (the risk adjustment "
                 f"alone) {_fmt(ev.summary.get('c_within_provider'))}")
        cal = _read((ev.tables or {}).get("calibration"))
        if cal is not None:
            L.append("- calibration by decile of the risk score (observed vs expected events):\n")
            L.append(_md_table(cal[cal["scale"] == "risk_score"][["decile", "n", "observed", "expected", "oe", "observed_rate", "mean_predicted"]]))
    if prose.get("model"):
        L.append("\n" + resolve(prose["model"], refs))

    L.append("\n## 4. Standardized measures\n")
    sm = session.get("sm")
    if sm:
        s = sm.summary
        L.append(f"- norm: {s.get('norm')} of the provider effects ({_fmt(s.get('norm_value'))}); population rate {_fmt(100 * (s.get('population_rate') or 0))}%")
        if s.get("flips"):
            f = s["flips"]
            L.append(f"- providers whose verdict flips between indirect and direct standardization (above the norm under one, below under the other): "
                     f"**{f.get('n')}**" + (f" ({', '.join(f.get('providers') or [])})" if f.get("n") else ""))
        tab = _read((sm.tables or {}).get("sm"))
        if tab is not None:
            L.append("\n" + _md_table(tab.sort_values(tab.columns[3] if len(tab.columns) > 3 else tab.columns[0], ascending=False), max_rows=20))
    if prose.get("measures"):
        L.append("\n" + resolve(prose["measures"], refs))

    L.append("\n## 5. Flags\n")
    t, ref = rec.get("test"), rec.get("reference", "theoretical")
    if t:
        df = compare.table_of(session, t, ref)
        c = compare.counts(df)
        n = len(df) if df is not None else 0
        tails = ["worse", "better"] if spec.consequential_tail != "better" else ["better", "worse"]
        L.append(f"Under the reported configuration ({t}, {ref} null, level {spec.level}): "
                 f"**{c['worse']} providers worse than expected ({100 * c['worse'] / max(n, 1):.1f}%)**, "
                 f"{c['better']} better ({100 * c['better'] / max(n, 1):.1f}%), {c['as_expected']} as expected. "
                 f"The {spec.consequential_tail} tail carries the consequences.")
        for tail in tails:
            ft = flagged_table(session, t, ref, tail)
            L.append(f"\n**{tail.capitalize()} than expected**\n")
            L.append(_md_table(ft))
    if prose.get("flags"):
        L.append("\n" + resolve(prose["flags"], refs))

    cal = session.get("calibration")
    if cal:
        s = cal.summary
        planted = (f" except {_fmt(s.get('n_planted_per_tail'))} providers per tail moved off it by an odds ratio of {_fmt(s.get('effect_or'))}"
                   if s.get("power") else "")
        L.append(f"\n**Calibration by simulation**: {_fmt(s.get('n_sim'))} data sets drawn from the fitted risk model with every provider at "
                 f"the norm{planted} ({_fmt(s.get('n_providers_used'))} providers, patient random-effect scale {_fmt(s.get('sigma_patient'))} matching a "
                 f"within-patient residual correlation of {_fmt(s.get('rho_observed'))}); type I error per tail (%, nominal {_fmt(s.get('nominal_pct'))}"
                 + (", counted on the providers at the norm" if s.get("power") else "") + "):\n")
        ct = _read((cal.tables or {}).get("calibration_tests"))
        if ct is not None:
            piv = ct.pivot_table(index=["test", "reference"], columns="tail", values="rate_pct").reset_index()
            L.append(_md_table(piv))
            if s.get("power") and "power_pct" in ct.columns and ct["power_pct"].notna().any():
                L.append("\nPower per tail (%, the planted providers of that tail found):\n")
                L.append(_md_table(ct.pivot_table(index=["test", "reference"], columns="tail", values="power_pct").reset_index()))
    subs = session.results_of("subset")
    if subs:
        L.append("\n**Provider breakdowns**\n")
        for r in subs:
            L.append(f"- {r.args.get('provider')}: observed {_fmt(r.summary.get('observed'))}, expected {_fmt(r.summary.get('expected'))}, "
                     f"ratio {_fmt(r.summary.get('ratio'))}")
            bt = _read(list((r.tables or {}).values())[0]) if r.tables else None
            if bt is not None:
                L.append("\n" + _md_table(bt, max_rows=40))
    L.append("\n## 6. Flags under the other admissible choices\n")
    for axis in ("statistic", "variance", "reference", "standardization"):
        r = session.get(f"compare:{axis}")
        if not r or not r.summary.get("comparisons"):
            L.append(f"- {axis}: no alternative here")
            continue
        for cmp_ in r.summary["comparisons"]:
            L.append(f"- **{axis}**: {cmp_['base']} vs {cmp_['alternative']}: worse {cmp_['worse_base']} vs {cmp_['worse_alt']} "
                     f"(only under the first {cmp_['worse_only_base']}, only under the second {cmp_['worse_only_alt']}); "
                     f"better {cmp_['better_base']} vs {cmp_['better_alt']}; **{cmp_['changed']} providers change status**"
                     + (f": {', '.join(cmp_['changed_providers'])}" + (" ..." if cmp_['changed'] > len(cmp_['changed_providers']) else "") if cmp_['changed'] else ""))
    if prose.get("alternatives"):
        L.append("\n" + resolve(prose["alternatives"], refs))

    L.append("\n## 7. Figures\n")
    figs = figures if figures is not None else [r.summary for r in session.results_of("figure")]
    if not figs:
        L.append("_no figure rendered_")
    for f in figs:
        if f.get("error"):
            L.append(f"- {f.get('display')} of {f.get('of')}: not rendered ({f['error']})")
        else:
            L.append(f"- **{f.get('display')}** of {f.get('of')}: `{f.get('path')}`" + (f"\n  {f.get('caption')}" if f.get("caption") else ""))
            if f.get("path"):
                L.append(f"\n![{f.get('display')}](figures/{Path(f['path']).name})\n")

    L.append("\n## 8. How the analysis was planned, and how the language model was used\n")
    for r in session.results:
        reason = r.reason if r.reason and not _DIGIT.search(r.reason) else ("(reason kept in the trace: it carried a number)" if r.reason else "")
        L.append(f"- step {r.step}: {r.id} [{r.source}] {reason}".rstrip())
    fallbacks = [p for p in session.plan_log if p.get("fallback")]
    if fallbacks:
        L.append(f"- the harness fell back to its own default action {len(fallbacks)} time(s) after refused proposals")
    if session.model_calls:
        L.append(f"- language model calls: {len(session.model_calls)}")
        for mc in session.model_calls:
            L.append(f"  - {mc.get('name')}: {mc.get('prompt_tokens')} prompt tokens, {mc.get('completion_tokens')} completion tokens")
    else:
        L.append("- no language model was used: the protected core ran under the harness's own plan")

    L.append("\n## 9. What this analysis does not establish\n")
    L += ["- No causal claim: a flag is a departure from the norm after adjustment for the declared risk factors, not a judgement of care.",
          "- No ranking: providers are not ordered against one another; indirectly standardized measures are not comparable between providers.",
          "- Nothing beyond this file: the model and the norm describe these providers and this period.",
          ("- The reported test was checked by simulation under the fitted risk model (section 5)." if session.has("calibration")
           else "- No calibration check of the tests by simulation was run; the recommendation rests on the rules of section 2.")]
    L.append("\n## Artifacts\n")
    run = Path(session.run_dir)
    L += [f"- specification: `{run / 'spec.json'}`", f"- trace: `{run / 'trace.json'}`", f"- tables: `{run / 'tables'}`",
          f"- replay without the model: `{run / 'repro.R'}`", f"- configuration hash: `{session.config_hash}`"]
    return "\n".join(L) + "\n"
