"""Build the SFT and DPO data sets from the tasks, their scored samples and (for the procedure choice) the
designs' rewards; split them by design family so that whole families are kept for evaluation.

SFT (one chat per task: the deployed system and user turns, the answer to learn):
  choose_procedure   the best allowed procedure with reasoning a hosted model wrote for it and the checks passed
                     (rationalize_api.py; source "rationale"); else the best-scored sampled answer when it is
                     within TIE of the best allowed procedure (source "sampled"); else the best allowed
                     procedure with reasoning written from the task's diagnostics by template (source "teacher")
  spec_extraction    the gold extraction (source "gold"); a sampled answer only replaces it if it scores 1
DPO (prompt, chosen, rejected): the chosen answer is the SFT answer (the best available); the rejected ones
are sampled answers scoring at least MARGIN below it, one per distinct answer (a distinct procedure for the
choice, a distinct set of extracted fields for the question), so one task shows several different mistakes
rather than one mistake several times; at most PAIRS per task, the mistakes the model made most often first.

Output, in the chat formats TRL reads:
  sft_train.jsonl, sft_eval.jsonl        {"messages": [...], "meta": {...}}
  dpo_train.jsonl, dpo_eval.jsonl        {"prompt": [system, user], "chosen": [assistant], "rejected": [assistant], "meta": {...}}
  heldout_families.json, build_report.json

    python build_datasets.py --choice-tasks ... --spec-tasks ... --samples ... --scores scores.json --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from typing import Any, Dict, List, Optional

from pprof_agent import choose

TIE = 0.02
MARGIN = 0.10
PAIRS = 3
HELDOUT_SHARE = 0.15


def load(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths or []:
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def _fmt(v: Optional[float], f: str = "{:.2f}") -> str:
    return "n/a" if v is None else f.format(v)


RHO_SMALL = 0.05            # a within-patient residual correlation at or below this reads as close to independent
SCALE_NEAR_ONE = (0.85, 1.15)
FOR_DESIGNS_LIKE_THIS = "holds its type I error with the most power for designs like this"


def teacher_answer(task: Dict[str, Any], key: str) -> str:
    """The best allowed procedure as an answer, its reasoning read off the task's diagnostics. Every clause is
    chosen by what the diagnostics show, so no clause contradicts them (an independent-rows argument is never
    made when the residuals are correlated, a wide spread never claimed when the scales are near one); where no
    argument from the diagnostics supports the choice, the clause says only that it is the choice that holds its
    type I error with the most power for designs like this, which is what the reward measures."""
    test, ref = key.split("|")
    parts = choose.parts_of(test)
    d = task.get("diag") or {}
    td = (d.get("tests") or {}).get(test, {})
    rho, med = d.get("within_patient_corr"), d.get("median_patients")
    scales = td.get("z_scale_by_size") or {}
    reading = [f"Within-patient correlation of residuals {_fmt(rho, '{:.3f}')}; median provider {_fmt(med, '{:.0f}')} patients; "
               f"providers with all events or none {d.get('invariant', 0)}.",
               f"For {test}, the statistics' scale by size quartile is " + " ".join(_fmt(scales.get(q)) for q in ("q1", "q2", "q3", "q4"))
               + (f", and the empirical null fits scales {' '.join(_fmt(s) for s in td['en_scales'] if s is not None)}." if td.get("en_scales") else ".")]
    small_rho = rho is not None and rho <= RHO_SMALL
    sc = [s for s in (scales.get(q) for q in ("q1", "q2", "q3", "q4")) if s is not None]
    near_one = bool(sc) and all(SCALE_NEAR_ONE[0] <= s <= SCALE_NEAR_ONE[1] for s in sc)
    span = f"{min(sc):.2f} to {max(sc):.2f}" if sc else "n/a"
    rho_s, med_s = _fmt(rho, "{:.3f}"), _fmt(med, "{:.0f}")
    if parts["variance"] == "naive":
        if small_rho:
            why = f"the rows of a patient are close to independent here (correlation {rho_s}), so the working-independence variance is not too small and keeps its power"
        elif ref == "empirical":
            why = (f"the rows of a patient are correlated ({rho_s}), but the size-stratified empirical null rescales the statistics and takes up "
                   "the extra spread, so the working-independence variance keeps its power")
        elif near_one:
            why = f"the rows of a patient are correlated ({rho_s}), yet the statistics' scale stays near one in every size quartile, so the working-independence variance is not too small here"
        else:
            why = f"the working-independence variance {FOR_DESIGNS_LIKE_THIS}"
    elif parts["variance"] == "robust":
        why = f"rows repeat within patients and providers have enough patients (median {med_s}) for a provider-specific sandwich, which stays valid whatever the correlation"
    else:
        why = f"providers have few patients (median {med_s}), so the within-patient correlation is pooled over all patients rather than estimated provider by provider"
    if parts["statistic"] == "score":
        stat = "the score statistic is evaluated at the null, so it stays usable for small providers and providers with extreme outcomes"
    elif parts["statistic"] == "wald":
        stat = "no provider has all events or none, so the Wald statistic is defined for every provider"
    elif small_rho:
        stat = "the exact test is exact for independent rows at these counts"
    else:
        stat = "the exact test keeps its power at small counts" + (", and the empirical null corrects the spread the correlation adds" if ref == "empirical" else "")
    if ref == "empirical":
        refw = (f"the statistics' scale departs from one across sizes ({span}), so the size-stratified empirical null is the honest reference" if not near_one
                else f"the statistics' scale is near one ({span}), and the size-stratified empirical null keeps the flags calibrated within each size group")
    else:
        refw = (f"the statistics' scale stays near one in every size quartile ({span}), so the theoretical null holds and keeps its power" if near_one
                else f"the theoretical null {FOR_DESIGNS_LIKE_THIS}")
    reasoning = " ".join(reading) + f" So: {why}; {stat}; {refw}."
    reason = f"{why[0].upper() + why[1:]}; {refw}."
    if len(reason) > choose.CHOICE_SCHEMA["properties"]["reason"]["maxLength"]:
        reason = f"{why[0].upper() + why[1:]}."
    return json.dumps({"reasoning": reasoning, "statistic": parts["statistic"], "variance": parts["variance"],
                       "reference": ref, "reason": reason[:choose.CHOICE_SCHEMA["properties"]["reason"]["maxLength"]]})


def answer_key(task: Dict[str, Any], sample: Dict[str, Any]) -> str:
    """What an answer decided: the procedure for the choice, the extracted fields for the question."""
    if task["act"] == "choose_procedure":
        return sample.get("key") or "invalid"
    try:
        return json.dumps(sorted((f["field"], str(f.get("value"))) for f in json.loads(sample["content"])["fields"]))
    except Exception:  # noqa: BLE001
        return "invalid"


def rejected_answers(task: Dict[str, Any], samples: List[Dict[str, Any]], chosen_score: float) -> List[Dict[str, Any]]:
    """One sampled answer per distinct decision scoring at least MARGIN below the chosen answer, the decisions the
    model made most often first (then the lower score), at most PAIRS."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        if chosen_score - s["reward"] >= MARGIN:
            groups.setdefault(answer_key(task, s), []).append(s)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]["reward"], kv[0]))
    return [{**g[0], "decision": k, "times": len(g)} for k, g in ranked[:PAIRS]]


def split_families(tasks: List[Dict[str, Any]], seed: int) -> set:
    fams = sorted({t.get("family") or t.get("meta", {}).get("family") for t in tasks})
    rng = random.Random(seed)
    k = max(1, round(HELDOUT_SHARE * len(fams)))
    return set(rng.sample(fams, k))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--choice-tasks", nargs="*", default=[])
    ap.add_argument("--spec-tasks", nargs="*", default=[])
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--rationales", nargs="*", default=[], help="rationalize_api.py output")
    ap.add_argument("--scores", help="scores.json of the designs")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rewards = {row["id"]: {k: v["reward"] for k, v in row["scores"].items()} for row in json.load(open(a.scores))} if a.scores else {}
    tasks = {t["id"]: t for t in load(a.choice_tasks) + load(a.spec_tasks) if "error" not in t}
    samples = {s["id"]: s.get("samples") or [] for s in load(a.samples) if "error" not in s}
    rationales = {x["id"]: x for x in load(a.rationales) if x.get("ok")}
    held =split_families(list(tasks.values()), a.seed)
    files = {k: open(os.path.join(a.out, k + ".jsonl"), "w") for k in ("sft_train", "sft_eval", "dpo_train", "dpo_eval")}
    stats: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for tid, task in tasks.items():
        act = task["act"]
        fam = task.get("family") or task.get("meta", {}).get("family")
        part = "eval" if fam in held else "train"
        prompt = task["messages"][:2]
        sm = sorted((s for s in samples.get(tid, []) if s.get("content")), key=lambda s: -s["reward"])
        # the SFT answer and its score
        if act == "choose_procedure":
            r = {k: v for k, v in rewards.get(task["design_id"], {}).items() if k in task["allowed"]}
            if not r:
                stats[act]["no_reward"] += 1
                continue
            best_key = max(r, key=r.get)
            if tid in rationales and rationales[tid]["key"] == best_key:
                answer, source, ans_score = rationales[tid]["content"], "rationale", r[best_key]
            elif sm and sm[0]["valid"] and sm[0]["reward"] >= r[best_key] - TIE:
                answer, source, ans_score = sm[0]["content"], "sampled", sm[0]["reward"]
            else:
                answer, source, ans_score = teacher_answer(task, best_key), "teacher", r[best_key]
            meta = {"act": act, "task": tid, "design": task["design_id"], "family": fam, "source": source,
                    "best": best_key, "rule_choice": task.get("rule_choice"),
                    "rule_regret": (r.get(task.get("rule_choice"), min(r.values())) - r[best_key])}
            stats[act]["rule_is_best"] += int(task.get("rule_choice") in r and r[task["rule_choice"]] >= r[best_key] - TIE)
        else:
            if sm and sm[0]["reward"] >= 1.0:
                answer, source = sm[0]["content"], "sampled"
            else:
                answer, source = task["messages"][2]["content"], "gold"
            ans_score = 1.0
            meta = {"act": act, "task": tid, "family": fam, "source": source}
        files[f"sft_{part}"].write(json.dumps({"messages": prompt + [{"role": "assistant", "content": answer}], "meta": meta}) + "\n")
        stats[act][f"sft_{part}"] += 1
        stats[act][f"source_{source}"] += 1
        # DPO: the SFT answer against the model's distinct mistakes on this task
        for lo in rejected_answers(task, sm, ans_score):
            files[f"dpo_{part}"].write(json.dumps({"prompt": prompt, "chosen": [{"role": "assistant", "content": answer}],
                                                   "rejected": [{"role": "assistant", "content": lo["content"]}],
                                                   "meta": {**meta, "chosen_score": ans_score, "rejected_score": lo["reward"],
                                                            "rejected_decision": lo["decision"], "rejected_times": lo["times"]}}) + "\n")
            stats[act][f"dpo_{part}"] += 1
            stats[act]["dpo_chosen_teacher" if source == "teacher" else f"dpo_chosen_{source}"] += 1

    for f in files.values():
        f.close()
    report = {"heldout_families": sorted(held), "counts": {k: dict(v) for k, v in stats.items()},
              "tie": TIE, "margin": MARGIN, "pairs_per_task": PAIRS}
    json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(json.dumps(report["counts"], indent=1))
    print(f"held-out families ({len(held)}): {sorted(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
