"""How much do the diagnostics the model sees say about which procedure to report? A statistical "teacher".

For every choice task the model is shown a data profile (counts), fit summaries and, for each computed test, summaries
of its z statistics (choice_tasks.py). Here an ordinary regression model per procedure (gradient boosting, and ridge as
a check) predicts the procedure's reward from those same numbers; the teacher reports the allowed procedure with the
highest predicted reward. Its regret against the simulation-best procedure is what the shown information can support.

Feature sets, cumulative, to see where the information is:
  kind      the unit of observation only (the per-kind lookup table)
  profile   + counts from the data, no model needed (sizes, rows per patient, event rate, trials, invariant providers)
  fit       + the fit's summaries (within-patient residual correlation, events per adjuster, providers at the bound)
  tests     + the z summaries of every computed test (share beyond 1.96, MAD scale by size quartile, empirical null)
Evaluation: leave-one-family-out over the training families (the honest estimate on many families) and the held-out
families of the SFT/DPO evaluation (fit on all training families), next to the lookup table, always
robust_score|empirical and the trained models' greedy answers. A depth-3 tree fitted to the teacher's choices shows
what its rule looks like. The teacher's choices (out of fold for training tasks) are written for later use as labels.

    python teacher.py --tasks choice_tasks_v2.jsonl --scores scores.json --heldout build_report.json \
                      --arms sft=<greedy.jsonl> dpo=<greedy.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, export_text

TESTS = ["exact", "score", "wald", "robust_wald", "robust_score", "stabilized_wald", "stabilized_score"]
KINDS = ["discharge", "patient_month", "binomial_days"]


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def _five(text: str, label: str) -> List[float]:
    m = re.search(label + r" \(min / q1 / median / q3 / max\): ([^\n]+)", text)
    return [_num(x.strip()) for x in m.group(1).split("/")] if m else [math.nan] * 5


def features(task: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """The numbers the model is shown, by feature set."""
    text, d = task["messages"][1]["content"], task["diag"]
    kind = task["family"].split("|")[0]
    g = lambda pat, i=1: _num(m.group(i)) if (m := re.search(pat, text)) else math.nan
    pp, rp = _five(text, "patients per provider"), _five(text, "rows per patient")
    out: Dict[str, Dict[str, float]] = {"kind": {f"kind_{k}": float(kind == k) for k in KINDS}}
    out["profile"] = {
        "providers": g(r"providers profiled (\d+)"), "excluded": g(r"excluded below eligibility (\d+)"),
        "eligibility": g(r"fewer than (\d+) patients"), "binomial": float(" out of " in text.split("\n")[1]),
        "higher_worse": float("higher rate is worse" in text), "tail_worse": float("consequential tail: worse" in text),
        "direct": float("direct standardization" in text and "indirect" not in text),
        **{f"pp_{q}": v for q, v in zip(("min", "q1", "med", "q3", "max"), pp)},
        **{f"rows_{q}": v for q, v in zip(("min", "q1", "med", "q3", "max"), rp)},
        "event_rate": g(r"event rate ([\d.]+)"), "log_trials_max": math.log10(max(g(r"largest (\d+)"), 1)),
        "invariant": float(d.get("invariant") or 0)}
    out["fit"] = {"log_events_per_adjuster": math.log10(max(g(r"events per adjuster ([\d.]+)"), 0.1)),
                  "at_bound": g(r"estimation bound (\d+)"), "rho": d.get("within_patient_corr", math.nan)}
    t: Dict[str, float] = {}
    for name in TESTS:
        s = (d.get("tests") or {}).get(name) or {}
        sc = [(s.get("z_scale_by_size") or {}).get(q) for q in ("q1", "q2", "q3", "q4")]
        sc = [math.nan if v is None else v for v in sc]
        en = [v for v in (s.get("en_scales") or []) if v is not None]
        t.update({f"{name}_beyond": s.get("z_beyond_pct", math.nan) if s.get("z_beyond_pct") is not None else math.nan,
                  **{f"{name}_scale_q{i + 1}": v for i, v in enumerate(sc)},
                  f"{name}_scale_mean": float(np.nanmean(sc)) if any(not math.isnan(v) for v in sc) else math.nan,
                  f"{name}_scale_q4_over_q1": sc[3] / sc[0] if sc[0] and not math.isnan(sc[0]) and not math.isnan(sc[3]) else math.nan,
                  f"{name}_en_fitted": float(bool(en)),
                  f"{name}_en_max": max(en) if en else math.nan, f"{name}_en_min": min(en) if en else math.nan,
                  f"{name}_en_loc": s.get("en_location_max") if s.get("en_location_max") is not None else math.nan})
    out["tests"] = t
    return out


SETS = {"kind": ["kind"], "profile": ["kind", "profile"], "fit": ["kind", "profile", "fit"],
        "tests": ["kind", "profile", "fit", "tests"]}


def matrix(feats: List[Dict[str, Dict[str, float]]], sets: List[str]):
    names = [n for s in sets for n in feats[0][s]]
    return np.array([[f[s][n] for s in sets for n in f[s]] for f in feats], float), names


def fit_predict(kind_model: str, X: np.ndarray, Y: Dict[str, np.ndarray], tr: np.ndarray, te: np.ndarray,
                allowed: List[List[str]]) -> Dict[str, np.ndarray]:
    """Predicted reward of every procedure for the test rows, each procedure's model fitted on the training rows where
    it was allowed."""
    pred = {}
    for k, y in Y.items():
        rows = np.array([i for i in tr if k in allowed[i] and not math.isnan(y[i])])
        if len(rows) < 30:
            continue
        if kind_model == "gbm":
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=0)
            m.fit(X[rows], y[rows])
            pred[k] = m.predict(X[te])
        else:
            mu = np.nanmedian(X[rows], axis=0)
            mu = np.where(np.isnan(mu), 0.0, mu)
            fill = lambda A: np.where(np.isnan(A), mu, A)
            A, B = fill(X[rows]), fill(X[te])
            sd = A.std(axis=0) + 1e-9
            m = Ridge(alpha=10.0).fit((A - A.mean(0)) / sd, y[rows])
            pred[k] = m.predict((B - A.mean(0)) / sd)
    return pred


def choose(pred: Dict[str, np.ndarray], te: np.ndarray, allowed: List[List[str]]) -> List[str]:
    out = []
    for j, i in enumerate(te):
        cand = {k: v[j] for k, v in pred.items() if k in allowed[i]}
        out.append(max(cand, key=cand.get) if cand else allowed[i][0])
    return out


def score(picks: List[str], idx: np.ndarray, R: List[Dict[str, float]]) -> Dict[str, float]:
    reg = [(R[i][k] if k in R[i] else min(R[i].values())) - max(R[i].values()) for k, i in zip(picks, idx)]
    return {"regret": float(np.mean(reg)), "within": float(np.mean([r >= -0.02 for r in reg])), "n": len(reg)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--arms", nargs="*", default=[], help="name=greedy answers (eval_report format)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    S = {r["id"]: r for r in json.load(open(a.scores))}
    held = set(json.load(open(a.heldout))["heldout_families"])
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    tasks = [t for t in tasks if t["act"] == "choose_procedure" and t["design_id"] in S]
    R = [{k: S[t["design_id"]]["scores"][k]["reward"] for k in t["allowed"] if k in S[t["design_id"]]["scores"]} for t in tasks]
    allowed = [list(r) for r in R]
    keys = sorted({k for r in R for k in r})
    Y = {k: np.array([r.get(k, math.nan) for r in R]) for k in keys}
    fam = np.array([t["family"] for t in tasks])
    kind = np.array([f.split("|")[0] for f in fam])
    tr = np.array([i for i in range(len(tasks)) if fam[i] not in held])
    ev = np.array([i for i in range(len(tasks)) if fam[i] in held])
    feats = [features(t) for t in tasks]
    print(f"{len(tasks)} choice tasks: {len(tr)} in {len(set(fam[tr]))} training families, {len(ev)} in "
          f"{len(set(fam[ev]))} held-out families; {len(keys)} procedures", flush=True)

    rows: List[Any] = []
    labels: Dict[str, Any] = {}
    # reference policies
    def per_kind_table(idx_fit):
        return {kd: max(keys, key=lambda k: np.mean([(R[i][k] if k in R[i] else min(R[i].values())) - max(R[i].values())
                                                     for i in idx_fit if kind[i] == kd]))
                for kd in KINDS}
    table = per_kind_table(tr)
    cv_table = []
    for f in sorted(set(fam[tr])):
        rest = np.array([i for i in tr if fam[i] != f])
        tb = per_kind_table(rest)
        cv_table += [(i, tb[kind[i]]) for i in tr if fam[i] == f]
    cv_table.sort()
    rows.append(("lookup table by kind", score([k for _, k in cv_table], np.array([i for i, _ in cv_table]), R),
                 score([table[kind[i]] for i in ev], ev, R)))
    rows.append(("always robust_score|empirical", score(["robust_score|empirical"] * len(tr), tr, R),
                 score(["robust_score|empirical"] * len(ev), ev, R)))
    for name, path in (x.split("=", 1) for x in a.arms):
        got = {r["id"]: r["samples"][0].get("key") or "invalid" for r in map(json.loads, open(path))}
        picks = [got.get(tasks[i]["id"], "invalid") for i in ev]
        rows.append((f"{name} (LLM, greedy)", None, score(picks, ev, R)))

    teacher_choice: Dict[str, Dict[int, str]] = {}
    for model in ("gbm", "ridge"):
        for set_name, sets in SETS.items():
            if model == "ridge" and set_name != "tests":
                continue
            X, names = matrix(feats, sets)
            oof: Dict[int, str] = {}
            for f in sorted(set(fam[tr])):
                te = np.array([i for i in tr if fam[i] == f])
                rest = np.array([i for i in tr if fam[i] != f])
                for i, k in zip(te, choose(fit_predict(model, X, Y, rest, te, allowed), te, allowed)):
                    oof[int(i)] = k
            pred_ev = fit_predict(model, X, Y, tr, ev, allowed)
            ev_pick = choose(pred_ev, ev, allowed)
            tag = f"teacher {model} [{set_name}]"
            rows.append((tag, score([oof[int(i)] for i in tr], tr, R), score(ev_pick, ev, R)))
            teacher_choice[tag] = {**oof, **{int(i): k for i, k in zip(ev, ev_pick)}}
            print(f"  done {tag} ({len(names)} features)", flush=True)

    print("\n=== regret (0 = the best allowed procedure); share within 0.02 of the best")
    print(f"{'policy':40s} {'train families, leave-one-out':>32s} {'held-out families':>24s}")
    for name, cv, ho in rows:
        c = f"{cv['regret']:.3f} / {cv['within']:.0%}" if cv else "-"
        print(f"{name:40s} {c:>32s} {ho['regret']:.3f} / {ho['within']:.0%}".rjust(0))

    best_tag = max((r for r in rows if r[0].startswith("teacher") and r[1]), key=lambda r: r[1]["regret"])[0]
    ch = teacher_choice[best_tag]
    print(f"\n=== the best teacher by leave-family-out regret: {best_tag}")
    for kd in KINDS:
        for part, idx in (("train (out of fold)", tr), ("held-out", ev)):
            sub = [i for i in idx if kind[i] == kd]
            if not sub:
                continue
            best = collections.Counter(max(R[i], key=R[i].get) for i in sub)
            picks = collections.Counter(ch[int(i)] for i in sub)
            s_t = score([ch[int(i)] for i in sub], np.array(sub), R)
            s_l = score([table[kd]] * len(sub), np.array(sub), R)
            print(f"  {kd:13s} {part:20s} n {len(sub):4d}  teacher {s_t['regret']:.3f}  table {s_l['regret']:.3f}  "
                  f"picks {picks.most_common(4)}  best {best.most_common(4)}")

    # what the teacher's rule looks like: a depth-3 tree on its out-of-fold choices
    X, names = matrix(feats, SETS["tests"])
    Xf = np.where(np.isnan(X), -9.0, X)
    yt = [ch[int(i)] for i in tr]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=0).fit(Xf[tr], yt)
    print("\n=== a depth-3 tree imitating the teacher (missing = -9)\n" + export_text(tree, feature_names=names, max_depth=3))
    tree_ev = list(tree.predict(Xf[ev]))
    tree_ev = [k if k in allowed[i] else table[kind[i]] for k, i in zip(tree_ev, ev)]
    print(f"tree as a policy on the held-out families: {score(tree_ev, ev, R)}")

    with open(os.path.join(a.out, "teacher_labels.jsonl"), "w") as fh:
        for i, t in enumerate(tasks):
            fh.write(json.dumps({"id": t["id"], "part": "eval" if fam[i] in held else "train", "teacher": best_tag,
                                 "key": ch[int(i)], "best": max(R[i], key=R[i].get),
                                 "regret": (R[i].get(ch[int(i)], min(R[i].values())) - max(R[i].values()))}) + "\n")
    json.dump({"rows": [(n, c, h) for n, c, h in rows], "best_teacher": best_tag, "table": table},
              open(os.path.join(a.out, "teacher_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
