"""The presentation layer: pprof_py displays built from the engine's tables, plus harness-drawn figures.

The R tables carry what a `ProviderProfile` needs (estimate, flag, se, interval, O, E, the funnel
coordinates whose limits agree with the flags, degeneracy), so the displays of pprof_py draw from the
same numbers the report quotes. Displays pprof_py cannot draw from tables (coefficient forest without
a model, null calibration, the calibration-by-decile plot, flag stability across our own axes) are
drawn here with matplotlib in the same theme. Every renderer returns a summary and never raises: a
figure that cannot be drawn is recorded as not rendered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import compare, rules

ROLES = {
    "provider_id": "provider", "estimate": "estimate", "se": "se", "ci_lower": "ci_lower", "ci_upper": "ci_upper",
    "null_value": "null_value", "p_value": "p", "z_raw": "stat", "observed": "observed", "expected": "expected",
    "denominator": "trials", "funnel_estimate": "funnel_estimate", "funnel_precision": "funnel_precision",
    "funnel_lower": "funnel_lower", "funnel_upper": "funnel_upper",
    "zero_events": "zero_events", "all_events": "all_events", "finite_estimate": "finite_estimate",
}


def _theme():
    try:
        from pprof_py.presentation import Theme
        return Theme.report()
    except Exception:  # noqa: BLE001
        return None


def _figures_dir(session: Any) -> Path:
    d = Path(session.run_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profile_from_result(session: Any, rid: str):
    """A pprof_py ProviderProfile from a `test:<t>` or `en:<t>` result table."""
    from pprof_py.presentation import ProviderProfile
    kind, test = rid.split(":", 1)
    reference = "empirical" if kind == "en" else "theoretical"
    df = compare.table_of(session, test, reference)
    if df is None:
        raise ValueError(f"no table for {rid}")
    res = session.get(rid)
    tsum = (session.get(f"test:{test}") or res).summary
    roles = dict(ROLES)
    roles["flag"] = "flag_en_raw" if reference == "empirical" else "flag_raw"
    if reference == "empirical":
        roles.update({"z_adjusted": "z_en", "null_mean": "null_mean", "null_sd": "null_sd", "null_group": "group"})
    else:
        roles["z_adjusted"] = "stat"
    d = df.copy()
    d[roles["flag"]] = d[roles["flag"]].astype("float")
    wald = test in rules.WALD_TYPE
    prov = {
        "model": "LogisticFixedEffectModel", "estimator": "fixed effect (unshrunken)",
        "measure": "gamma" if wald else "indirect_ratio", "scale": "effect" if wald else "ratio",
        "test_method": test, "reference": "median", "reference_value": tsum.get("null_value"),
        "null_model": ("empirical null, size-stratified (location and scale per group)" if reference == "empirical"
                       else "theoretical N(0, 1)"),
        "alternative": "two_sided", "level": tsum.get("level", 0.95), "critical": tsum.get("crit"),
        "covariates": tuple(session.spec.adjusters or []),
        "funnel": {"estimate_kind": tsum.get("estimate_kind"), "precision_kind": tsum.get("precision_kind"),
                   "limit_rule": tsum.get("limit_rule"), "curve_kind": None, "levels": (tsum.get("level", 0.95),)},
    }
    return ProviderProfile.from_frame(d, roles=roles, provenance=prov)


def _save(fig, path: Path) -> Dict[str, Any]:
    fig.save(str(path))
    try:
        fig.save(str(path.with_suffix(".png")), dpi=200)
    except Exception:  # noqa: BLE001
        pass
    return {"path": str(path), "caption": str(getattr(fig, "caption", "") or ""),
            "alt_text": str(getattr(fig, "alt_text", "") or "")}


def render_figure(session: Any, display: str, of: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    out: Dict[str, Any] = {"display": display, "of": of}
    try:
        path = _figures_dir(session) / f"{display}_{(of or 'run').replace(':', '_')}.svg"
        if display in ("funnel", "caterpillar", "observed_expected", "data_quality"):
            import pprof_py.presentation as P
            prof = profile_from_result(session, of)
            fn = getattr(P, display)
            fig = fn(prof, theme="report") if display != "data_quality" else fn(prof, theme="report")
            out.update(_save(fig, path))
            out["status_counts"] = {k: v for k, v in prof.status_counts().items()}
        elif display == "coefficient_forest":
            out.update(_forest(session, path))
        elif display == "null_calibration":
            out.update(_null_calibration(session, of, path))
        elif display == "calibration_plot":
            out.update(_calibration(session, path))
        elif display == "flag_stability":
            out.update(_flag_stability(session, path))
        else:
            out["error"] = f"unknown display {display}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    tables = {}
    if out.get("path"):
        tables[f"figure_{display}"] = out["path"]
    return out, tables


# ------------------------------------------------------------------------------- harness-drawn displays
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _forest(session: Any, path: Path) -> Dict[str, Any]:
    plt = _mpl()
    fit = session.get("fit")
    beta = pd.read_csv(fit.tables["beta"])
    beta = beta.sort_values("beta")
    sel = session.get("selection")
    kept = None
    if sel is not None:
        s = pd.read_csv(sel.tables["selection"])
        kept = set(s.loc[s["selected"], "term"].astype(str))
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(beta) + 1.2))
    y = np.arange(len(beta))
    ax.errorbar(beta["beta"], y, xerr=1.96 * beta["se_model"], fmt="o", color="#1E2B38", ecolor="#6B7684", capsize=2, ms=3)
    ax.axvline(0, color="#1E2B38", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}{'' if kept is None or t in kept else ' (dropped)'}" for t in beta["term"]], fontsize=8)
    ax.set_xlabel("coefficient (log-odds) with 95% model-based interval")
    ax.set_title("Risk-adjustment coefficients", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path), "caption": "Coefficients of the declared adjusters with model-based 95% intervals."}


def _null_calibration(session: Any, of: Optional[str], path: Path) -> Dict[str, Any]:
    plt = _mpl()
    from scipy.stats import norm
    if not of or ":" not in of:
        if not session.recommended:
            from ..loop import recommend
            recommend(session)
        of = f"en:{session.recommended.get('test')}"
    test = of.split(":", 1)[1]
    en = session.get(f"en:{test}")
    if en is None:
        raise ValueError(f"no empirical null for {test}")
    tab = pd.read_csv(en.tables[f"en_{test}"])
    gi = pd.read_csv(en.tables[f"en_groups_{test}"])
    z = tab["stat"].to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    r = max(4.0, float(np.quantile(np.abs(z), 0.995)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.hist(np.clip(z, -r, r), bins=60, density=True, color="#9ECAE1", edgecolor="white")
    xs = np.linspace(-r, r, 400)
    ax.plot(xs, norm.pdf(xs), color="#1E2B38", lw=1.4, label="theoretical null")
    w = gi["n"] / gi["n"].sum()
    mix = sum(wi * norm.pdf(xs, loc, sc) for wi, loc, sc in zip(w, gi["location"], gi["scale"]))
    ax.plot(xs, mix, color="#C0392B", lw=1.4, ls=":", label="empirical null (size-stratified)")
    crit = float(en.summary.get("crit") or 1.96)
    for c in (-crit, crit):
        ax.axvline(c, color="black", lw=0.7, ls="--")
    ax.set_xlabel(f"{test} statistic")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(a) statistics and the two nulls", fontsize=10)
    ax = axes[1]
    x = np.arange(len(gi))
    ax.bar(x - 0.2, gi["scale"], 0.4, color="#3182BD", label="scale")
    ax.bar(x + 0.2, gi["location"], 0.4, color="#9ECAE1", label="location")
    ax.axhline(1, color="black", lw=0.7, ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{int(g)}\n{int(m)} pts" for g, m in zip(gi["group"], gi["median_size"])], fontsize=8)
    ax.set_xlabel("provider size group (median patients)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) the empirical null by size", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path),
            "caption": f"Is the theoretical null credible? The {test} statistics against N(0,1) and the size-stratified empirical null."}


def _calibration(session: Any, path: Path) -> Dict[str, Any]:
    plt = _mpl()
    ev = session.get("evaluation")
    if ev is None:
        raise ValueError("no evaluation result")
    cal = pd.read_csv(ev.tables["calibration"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, scale, title in zip(axes, ("risk_score", "prediction"), ("(a) by decile of the risk score", "(b) by decile of the full prediction")):
        d = cal[cal["scale"] == scale]
        ax.plot(d["mean_predicted"], d["observed_rate"], "o-", color="#1E2B38")
        lim = [0, max(float(d["mean_predicted"].max()), float(d["observed_rate"].max())) * 1.1]
        ax.plot(lim, lim, color="#6B7684", lw=0.8, ls="--")
        ax.set_xlabel("mean predicted rate")
        ax.set_ylabel("observed rate")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path), "caption": "Calibration of the risk-adjustment model: observed against mean predicted rate by decile."}


def _flag_stability(session: Any, path: Path) -> Dict[str, Any]:
    plt = _mpl()
    if not session.recommended:                 # asked for mid-loop: decide the reported configuration now
        from ..loop import recommend
        recommend(session)
    rows = []
    for axis in ("statistic", "variance", "reference", "standardization"):
        r = session.get(f"compare:{axis}")
        if not r:
            continue
        for c in r.summary.get("comparisons", []):
            rows.append((axis, c["alternative"], c["worse_alt"], c["better_alt"], c["changed"]))
    rec = session.recommended
    base = compare.table_of(session, rec["test"], rec["reference"])
    cb = compare.counts(base)
    rows.insert(0, ("reported", f"{rec['test']}/{rec['reference']}", cb["worse"], cb["better"], 0))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[2] for r in rows], 0.4, color="#C0392B", label="worse")
    ax.barh(y - 0.2, [r[3] for r in rows], 0.4, color="#3182BD", label="better")
    for yi, r in zip(y, rows):
        if r[4]:
            ax.text(max(r[2], r[3]) + 0.5, yi, f"{r[4]} change status", va="center", fontsize=8, color="#556372")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}: {r[1]}" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("providers flagged")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Flags under the reported configuration and each alternative", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path), "caption": "How many providers each admissible alternative flags, and how many change status."}


def default_figures(session: Any, log=print) -> List[Dict[str, Any]]:
    """The harness's own choice when no model chooses: the funnel and null calibration of the reported
    test, the coefficient forest, the calibration plot and the flag stability."""
    from . import executor
    rec = session.recommended or {}
    test, ref = rec.get("test"), rec.get("reference", "theoretical")
    rid = f"en:{test}" if ref == "empirical" and session.has(f"en:{test}") else f"test:{test}"
    wanted = [("funnel", rid), ("observed_expected", rid), ("data_quality", rid), ("coefficient_forest", "fit"),
              ("calibration_plot", "evaluation"), ("flag_stability", "fit")]
    if session.has(f"en:{test}"):
        wanted.append(("null_calibration", f"en:{test}"))
    if test in rules.WALD_TYPE:
        wanted.append(("caterpillar", rid))
    done = []
    for display, of in wanted:
        try:
            res = executor.execute(session, "figure", {"display": display, "of": of}, source="harness",
                                   reason="default display of the reported configuration")
            done.append(res.summary)
            log(f"figure {display} of {of}: {'ok' if res.summary.get('path') else res.summary.get('error')}")
        except Exception as exc:  # noqa: BLE001
            log(f"figure {display} failed: {exc}")
    return done


def render_html(session: Any, markdown_text: str) -> Optional[Path]:
    """One self-contained HTML report through pprof_py's Report, with the figures embedded."""
    try:
        from pprof_py.presentation import Report
    except Exception:  # noqa: BLE001
        return None
    spec = session.spec
    rep = Report(f"Provider profiling: {spec.outcome} per {spec.unit}", subtitle=f"run {Path(session.run_dir).name}", theme="report")
    section = None
    buf: List[str] = []

    def flush():
        if buf:
            rep.text("\n".join(buf))
            buf.clear()

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            rep.section(section)
        elif line.startswith("# "):
            continue
        elif line.startswith("!["):
            continue
        else:
            buf.append(line)
    flush()
    p = Path(session.run_dir) / "report.html"
    rep.save(str(p))
    return p
