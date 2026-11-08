"""Score every procedure per design from its operating characteristics (oc.R) and report how the best
procedure varies across designs: the pilot's question, since training data in which one procedure always
wins would teach the model a constant.

A procedure is a provider test (statistic x variance) under a reference distribution (theoretical or the
size-stratified empirical null). Its reward on a design is the harness's Neyman-Pearson score
(rules.np_score: power, less a penalty for type I error beyond the tolerance) per tail, the worse tail
weighted by rules.CALIBRATION["w_consequential"]. The type I error enters as an upper confidence bound
(estimate + Z_UCB standard errors), so Monte Carlo error cannot make an oversized test look valid.
Inadmissible procedures are left out: the Wald-type tests when the design has all-zero providers, and any
test or reference the simulation could not compute (the exact test above the trials limit, the empirical
null with too few providers).

    python summarize_oc.py <dir with designs.jsonl and oc/>      -> <dir>/scores.json and a printed report
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from pprof_agent import rules

Z_UCB = 0.84            # one-sided 80% upper bound, the convention of the group's agent notes
TIE = 0.02              # procedures within this of the best count as tied
FIXED = ("stabilized_score|empirical", "stabilized_score|theoretical", "robust_score|theoretical",
         "robust_score|empirical", "robust_wald|empirical", "score|theoretical")


def tails(design: Dict[str, Any]) -> Tuple[str, str]:
    worse = "higher" if design.get("event_direction", "higher_is_worse") == "higher_is_worse" else "lower"
    return worse, ("lower" if worse == "higher" else "higher")


def scores(oc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    design = oc["design"]
    nominal = float(oc.get("nominal_pct", 2.5))
    w = rules.CALIBRATION["w_consequential"]
    worse, other = tails(design)
    by: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for r in oc["rates"]:
        r = {k: (None if v == "NA" else v) for k, v in r.items()}     # older oc files wrote R's NA as a string
        by[f"{r['test']}|{r['reference']}"][r["tail"]] = r
    out = {}
    for key, t in by.items():
        test = key.split("|")[0]
        if design.get("n_invariant", 0) > 0 and test in rules.WALD_TYPE:
            continue
        if worse not in t or other not in t or t[worse]["size_pct"] is None:
            continue
        def tail_score(r):
            size = float(r["size_pct"]) + Z_UCB * float(r["size_se_pct"] or 0)
            return rules.np_score(size, r["power_pct"], nominal)
        out[key] = {"reward": w * tail_score(t[worse]) + (1 - w) * tail_score(t[other]),
                    "size_worse": t[worse]["size_pct"], "power_worse": t[worse]["power_pct"],
                    "size_other": t[other]["size_pct"], "power_other": t[other]["power_pct"]}
    return out


def main(argv=None) -> int:
    root = (argv or sys.argv[1:])[0]
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, "oc", "*.json"))):
        with open(path) as fh:
            oc = json.load(fh)
        sc = scores(oc)
        if not sc:
            continue
        best_r = max(v["reward"] for v in sc.values())
        ranked = sorted(sc, key=lambda k: -sc[k]["reward"])
        tied = [k for k in ranked if sc[k]["reward"] >= best_r - TIE]
        rows.append({"id": oc["id"], "design": oc["design"], "n_rep": oc["n_rep"], "seconds": oc["seconds"],
                     "n_providers": oc["n_providers_median"], "scores": sc, "best": ranked[0], "tied": tied,
                     "gap_to_second": best_r - sc[ranked[1]]["reward"] if len(ranked) > 1 else None,
                     "regret": {f: (sc[f]["reward"] - best_r) if f in sc else None for f in FIXED}})
    with open(os.path.join(root, "scores.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print(f"{len(rows)} designs scored\n")
    print(f"{'id':<11} {'kind':<13} {'prov':>4} {'med':>5} {'sig':>4} {'tau':>4} {'plant':>5} {'cmix':>4} {'inv':>3} "
          f"{'best':<30} {'reward':>6} {'gap':>5} {'n_tied':>6} {'secs':>5}")
    for r in rows:
        d = r["design"]
        print(f"{r['id']:<11} {d['kind']:<13} {r['n_providers']:>4.0f} {d['median_patients']:>5.1f} {d['sigma_patient']:>4.1f} "
              f"{d['tau']:>4.2f} {d['planted_frac']:>5.2f} {d['casemix']:>4.1f} {d['n_invariant']:>3d} {r['best']:<30} "
              f"{r['scores'][r['best']]['reward']:>6.3f} {r['gap_to_second'] or 0:>5.3f} {len(r['tied']):>6d} {r['seconds']:>5.0f}")
    print("\nbest procedure (counting each design once; ties shared):")
    share: Dict[str, float] = collections.defaultdict(float)
    for r in rows:
        for k in r["tied"]:
            share[k] += 1 / len(r["tied"])
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v:5.1f}")
    print("\nmean regret of always using one procedure (0 = always best; designs where it is inadmissible left out):")
    for f in FIXED:
        vals = [r["regret"][f] for r in rows if r["regret"][f] is not None]
        if vals:
            print(f"  {f:<32} {sum(vals) / len(vals):7.3f}  (best in {sum(1 for r in rows if f in r['tied'])}/{len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
