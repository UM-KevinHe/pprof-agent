"""Flags under the other admissible choices, one axis at a time -- computed by the harness.

A flag is `worse` or `better` only after the raw direction (higher / lower than the norm) is read with
the declared event direction. The comparison names, for each alternative on the axis, how many
providers are flagged in each tail under each choice and which providers change status.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import rules

TAILS = ("worse", "better", "as_expected")


def tail_of(raw: Any, event_direction: str) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    r = int(raw)
    if r == 0:
        return "as_expected"
    higher = r == 1
    if event_direction == "higher_is_worse":
        return "worse" if higher else "better"
    return "better" if higher else "worse"


def table_of(session: Any, test: str, reference: str) -> Optional[pd.DataFrame]:
    """The flag table of a test under a reference, with a `status` column in the measure's tails."""
    run = Path(session.run_dir) / "tables"
    if reference == "empirical":
        p = run / f"en_{test}.csv"
        col = "flag_en_raw"
    else:
        p = run / f"test_{test}.csv"
        col = "flag_raw"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"provider": str})
    ed = session.spec.event_direction
    df["status"] = [tail_of(v, ed) for v in df[col]]
    return df


def measure_table_of(session: Any, key: str) -> Optional[pd.DataFrame]:
    p = Path(session.run_dir) / "tables" / f"measure_{key}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"provider": str})
    ed = session.spec.event_direction
    df["status"] = [tail_of(v, ed) for v in df["flag_raw"]]
    return df


def counts(df: Optional[pd.DataFrame]) -> Dict[str, int]:
    if df is None:
        return {t: 0 for t in TAILS}
    return {t: int((df["status"] == t).sum()) for t in TAILS}


def _ids(m: pd.DataFrame, mask: pd.Series, limit: int = 10) -> List[str]:
    return m.loc[mask, "provider"].astype(str).head(limit).tolist()


def alternative_tag(axis: str, alt_name: str) -> str:
    """A reference-name tag for one alternative: `theoretical_null`, `robust_score`, `direct`, `direct_empirical`."""
    test, _, ref = alt_name.partition("/")
    if axis == "reference":
        return f"{ref}_null"
    if axis == "standardization":
        se = test.partition(":")[2]
        return "direct" + ("" if se in ("", "model") else f"_{se}") + ("" if ref in ("", "theoretical") else f"_{ref}")
    return test


def alternative_words(tag: str) -> str:
    """The tag in words, for the facts and the reference meanings."""
    if tag.endswith("_null"):
        return f"the {tag[:-5]} null"
    if tag.startswith("direct"):
        parts = [p for p in tag[len("direct"):].split("_") if p]
        words = "the directly standardized rate tested on its own scale"
        if parts and parts[-1] in ("empirical", "theoretical"):
            words += f" under the {parts[-1]} null"
            parts = parts[:-1]
        if parts:
            words += f" with the {' '.join(parts)} standard error"
        return words
    return f"the {tag.replace('_', ' ')} test"


def compare_two(base: pd.DataFrame, alt: pd.DataFrame, base_name: str, alt_name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    m = base[["provider", "status"]].merge(alt[["provider", "status"]], on="provider", how="outer",
                                           suffixes=("_base", "_alt"))
    changed = m[(m["status_base"] != m["status_alt"]) & m["status_base"].notna() & m["status_alt"].notna()]
    cb, ca = counts(base), counts(alt)
    out = {
        "base": base_name, "alternative": alt_name,
        "worse_base": cb["worse"], "worse_alt": ca["worse"], "better_base": cb["better"], "better_alt": ca["better"],
        "worse_only_base": int(((m["status_base"] == "worse") & (m["status_alt"] != "worse")).sum()),
        "worse_only_alt": int(((m["status_alt"] == "worse") & (m["status_base"] != "worse")).sum()),
        "better_only_base": int(((m["status_base"] == "better") & (m["status_alt"] != "better")).sum()),
        "better_only_alt": int(((m["status_alt"] == "better") & (m["status_base"] != "better")).sum()),
        "changed": int(len(changed)),
        "changed_providers": changed["provider"].astype(str).head(20).tolist(),
        "worse_only_base_providers": _ids(m, (m["status_base"] == "worse") & (m["status_alt"] != "worse")),
        "worse_only_alt_providers": _ids(m, (m["status_alt"] == "worse") & (m["status_base"] != "worse")),
        "better_only_base_providers": _ids(m, (m["status_base"] == "better") & (m["status_alt"] != "better")),
        "better_only_alt_providers": _ids(m, (m["status_alt"] == "better") & (m["status_base"] != "better")),
    }
    flips = changed.rename(columns={"status_base": base_name, "status_alt": alt_name})
    flips.insert(0, "comparison", f"{base_name} vs {alt_name}")
    return out, flips


def alternative_table(session: Any, axis: str, alt_name: str) -> Optional[pd.DataFrame]:
    """The flag table of one alternative named in a comparison (`test/ref` or `direct_rate:se/ref`)."""
    test, _, ref = alt_name.partition("/")
    ref = ref or "theoretical"
    if axis == "standardization":
        se = test.partition(":")[2] or "model"
        return measure_table_of(session, f"direct_rate_{se}_{ref}")
    return table_of(session, test, ref)


def stability(session: Any) -> Dict[str, Any]:
    """Which providers flagged under the reported configuration keep that status under every computed
    alternative (the deck's 'flagged under k of n choices'), per tail."""
    rec = session.recommended or {}
    if not rec.get("test"):
        return {}
    base = table_of(session, rec["test"], rec.get("reference", "theoretical"))
    if base is None:
        return {}
    alts: List[pd.DataFrame] = []
    for axis in ("statistic", "variance", "reference", "standardization"):
        r = session.get(f"compare:{axis}")
        for c in (r.summary.get("comparisons") if r is not None else None) or []:
            t = alternative_table(session, axis, c["alternative"])
            if t is not None:
                alts.append(t)
    out: Dict[str, Any] = {"n_alternatives": len(alts)}
    for tail in ("worse", "better"):
        ids = base.loc[base["status"] == tail, "provider"].astype(str).tolist()
        keep = set(ids)
        for t in alts:
            keep &= set(t.loc[t["status"] == tail, "provider"].astype(str))
        out[tail] = {"flagged": ids, "stable": [i for i in ids if i in keep], "fragile": [i for i in ids if i not in keep]}
    return out


def compare_axis(session: Any, axis: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    adm = session.admissible
    rec = session.recommended or {}
    test = rec.get("test") or adm.default_test
    ref = rec.get("reference") or "theoretical"
    base = table_of(session, test, ref)
    if base is None:
        return {"axis": axis, "comparisons": [], "note": f"no table for {test} under the {ref} null"}, {}
    comps: List[Dict[str, Any]] = []
    flips: List[pd.DataFrame] = []
    base_name = f"{test}/{ref}"
    if axis in ("statistic", "variance"):
        for t in rules.alternatives(adm, test)[axis]:
            alt = table_of(session, t, ref)
            if alt is None:
                continue
            c, f = compare_two(base, alt, base_name, f"{t}/{ref}")
            comps.append(c)
            flips.append(f)
    elif axis == "reference":
        other = "empirical" if ref == "theoretical" else "theoretical"
        alt = table_of(session, test, other)
        if alt is not None:
            c, f = compare_two(base, alt, base_name, f"{test}/{other}")
            comps.append(c)
            flips.append(f)
    elif axis == "standardization":
        for r in session.results_of("test_measure"):
            key = r.summary.get("key")
            alt = measure_table_of(session, key)
            if alt is None:
                continue
            c, f = compare_two(base, alt, base_name, f"direct_rate:{r.args.get('se_source', 'model')}/{r.args.get('reference', 'theoretical')}")
            comps.append(c)
            flips.append(f)
    tables: Dict[str, str] = {}
    if flips:
        allf = pd.concat(flips, ignore_index=True)
        p = Path(session.run_dir) / "tables" / f"compare_{axis}.csv"
        allf.to_csv(p, index=False)
        tables[f"compare_{axis}"] = str(p)
    summary = {"axis": axis, "base": base_name, "comparisons": comps,
               "max_changed": max([c["changed"] for c in comps], default=0)}
    return summary, tables
