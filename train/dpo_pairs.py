"""On-policy DPO pairs from the SFT model's own samples (eval.sbatch STEPS=sample: 8 answers per task at T 0.9).

Both sides come from the model being trained wherever possible, so a pair contrasts decisions and not writing styles:
  chosen    the model's own best-scored valid sample when it is within TIE of the best allowed procedure (choice) or
            scores 1 (question); else the SFT answer of the task (the hosted model's rationale for the best procedure,
            else the template teacher; the gold extraction), which the SFT model was trained to write
  rejected  the model's own samples scoring at least MARGIN below the chosen one, one per distinct decision, the
            decisions it made most often first, at most PAIRS (build_datasets.rejected_answers; an invalid procedure
            pair such as exact with robust variance counts at the floor reward)
The held-out families are those of the SFT build (its build_report.json), so the DPO evaluation stays on families
neither stage has seen. Output in the same chat format as build_datasets.py: dpo_train.jsonl, dpo_eval.jsonl,
build_report.json.

    python dpo_pairs.py --choice-tasks ... --spec-tasks ... --samples samples_sft_v2.jsonl --rationales ...
                        --scores scores.json --heldout datasets_v2/build_report.json --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Dict

from build_datasets import MARGIN, PAIRS, TIE, load, rejected_answers, teacher_answer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--choice-tasks", nargs="*", default=[])
    ap.add_argument("--spec-tasks", nargs="*", default=[])
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--rationales", nargs="*", default=[])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--heldout", required=True, help="build_report.json of the SFT data set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rewards = {row["id"]: {k: v["reward"] for k, v in row["scores"].items()} for row in json.load(open(a.scores))}
    tasks = {t["id"]: t for t in load(a.choice_tasks) + load(a.spec_tasks) if "error" not in t}
    samples = {s["id"]: s.get("samples") or [] for s in load(a.samples) if "error" not in s}
    rationales = {x["id"]: x for x in load(a.rationales) if x.get("ok")}
    held = set(json.load(open(a.heldout))["heldout_families"])
    files = {k: open(os.path.join(a.out, k + ".jsonl"), "w") for k in ("dpo_train", "dpo_eval")}
    stats: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for tid, task in tasks.items():
        act = task["act"]
        fam = task.get("family") or task.get("meta", {}).get("family")
        part = "eval" if fam in held else "train"
        sm = sorted((s for s in samples.get(tid, []) if s.get("content")), key=lambda s: -s["reward"])
        if not sm:
            stats[act]["no_samples"] += 1
            continue
        meta: Dict[str, Any] = {"act": act, "task": tid, "family": fam}
        if act == "choose_procedure":
            r = {k: v for k, v in rewards.get(task["design_id"], {}).items() if k in task["allowed"]}
            if not r:
                stats[act]["no_reward"] += 1
                continue
            best_key = max(r, key=r.get)
            if sm[0]["valid"] and sm[0]["reward"] >= r[best_key] - TIE:
                chosen, source, score = sm[0]["content"], "own", sm[0]["reward"]
            elif tid in rationales and rationales[tid]["key"] == best_key:
                chosen, source, score = rationales[tid]["content"], "rationale", r[best_key]
            else:
                chosen, source, score = teacher_answer(task, best_key), "teacher", r[best_key]
            meta.update(design=task["design_id"], best=best_key, chosen_key=sm[0]["key"] if source == "own" else best_key)
        else:
            if sm[0]["reward"] >= 1.0:
                chosen, source = sm[0]["content"], "own"
            else:
                chosen, source = task["messages"][2]["content"], "gold"
            score = 1.0
        stats[act][f"chosen_{source}_{part}"] += 1
        lows = rejected_answers(task, sm, score)
        if not lows:
            stats[act][f"no_pair_{part}"] += 1
        for lo in lows:
            files[f"dpo_{part}"].write(json.dumps({
                "prompt": task["messages"][:2], "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": lo["content"]}],
                "meta": {**meta, "source": source, "chosen_score": score, "rejected_score": lo["reward"],
                         "rejected_decision": lo["decision"], "rejected_times": lo["times"]}}) + "\n")
            stats[act][f"dpo_{part}"] += 1
            stats[act][f"rejected_invalid_{part}"] += int(lo["decision"] in ("invalid",) or not lo.get("valid", True))

    for f in files.values():
        f.close()
    report = {"heldout_families": sorted(held), "counts": {k: dict(sorted(v.items())) for k, v in stats.items()},
              "tie": TIE, "margin": MARGIN, "pairs_per_task": PAIRS, "samples": a.samples}
    json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(json.dumps(report["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
