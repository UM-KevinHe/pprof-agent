"""Score arms of sampled answers (sample.py output) on the post-training tasks, held-out and training families apart.

Choice tasks: regret = reward of the answered procedure minus the best allowed one (0 = best), for the first answer,
the mean over a task's answers and the best of them, against the rule default and always robust_score|empirical
on the same tasks; the procedures chosen; answers whose reasoning quotes a number the system and user turns do not
contain. Question tasks: F1, reward (F1 minus the charge for invented items), answers with an invented item.

    python eval_report.py --choice-tasks choice_tasks_v2.jsonl --spec-tasks spec_tasks.jsonl --scores scores.json
                          --heldout build_report.json --arm untrained=samples_choice_v2.jsonl --arm sft=samples_sft.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from typing import Any, Dict, List

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
NEAR = 0.02


def jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def line(name: str, v: List[float]) -> str:
    return f"  {name:30s} {st.mean(v):7.3f}   within {NEAR} of best {sum(x >= -NEAR for x in v) / len(v):.2f}   n={len(v)}"


def unlisted_numbers(task: Dict[str, Any], content: str) -> List[str]:
    try:
        reasoning = json.loads(content).get("reasoning", "")
    except ValueError:
        return []
    allowed = set(NUMBER.findall(task["messages"][0]["content"] + "\n" + task["messages"][1]["content"]))
    return sorted({x for x in NUMBER.findall(reasoning) if x not in allowed})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--choice-tasks", required=True)
    ap.add_argument("--spec-tasks", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--heldout", required=True, help="build_report.json (its heldout_families)")
    ap.add_argument("--arm", action="append", required=True,
                    help="name=samples.jsonl[=act] (repeatable; the optional act keeps only that act's tasks)")
    ap.add_argument("--parts", default="eval,train")
    a = ap.parse_args(argv)
    held = set(json.load(open(a.heldout))["heldout_families"])
    rewards = {r["id"]: {k: v["reward"] for k, v in r["scores"].items()} for r in json.load(open(a.scores))}
    tasks = {t["id"]: t for t in jsonl(a.choice_tasks) + jsonl(a.spec_tasks) if "error" not in t}

    def part_of(t: Dict[str, Any]) -> str:
        return "eval" if (t.get("family") or t.get("meta", {}).get("family")) in held else "train"

    for arm in a.arm:
        name, path, *act = arm.split("=")
        samples = {s["id"]: s["samples"] for s in jsonl(path) if s.get("samples") and (not act or s.get("act") == act[0])}
        errors = sum(1 for s in jsonl(path) if "error" in s)
        for part in a.parts.split(","):
            ch = [(tasks[i], sm) for i, sm in samples.items() if i in tasks and tasks[i]["act"] == "choose_procedure"
                  and part_of(tasks[i]) == part]
            sp = [(tasks[i], sm) for i, sm in samples.items() if i in tasks and tasks[i]["act"] == "spec_extraction"
                  and part_of(tasks[i]) == part]
            if not ch and not sp:
                continue
            print(f"\n===== {name} | {part} families | {len(ch)} choice + {len(sp)} question tasks "
                  f"({max((len(sm) for _, sm in ch + sp), default=0)} answers per task; {errors} request errors in the file)")
            if ch:
                rows: Dict[str, List[float]] = collections.defaultdict(list)
                by_kind: Dict[str, List[float]] = collections.defaultdict(list)
                picks: collections.Counter = collections.Counter()
                valid = total = unlisted = 0
                examples: List[str] = []
                for t, sm in ch:
                    r = {k: v for k, v in rewards[t["design_id"]].items() if k in t["allowed"]}
                    best = max(r.values())
                    rows["model: first answer"].append(sm[0]["reward"] - best)
                    rows["model: mean over answers"].append(st.mean(s["reward"] for s in sm) - best)
                    if len(sm) > 1:
                        rows[f"model: best of {len(sm)}"].append(max(s["reward"] for s in sm) - best)
                    rows["rule default"].append(r.get(t["rule_choice"], min(r.values())) - best)
                    rows["always robust_score|empirical"].append(r.get("robust_score|empirical", min(r.values())) - best)
                    by_kind[t["family"].split("|")[0]].append(sm[0]["reward"] - best)
                    picks[sm[0].get("key")] += 1
                    for s in sm:
                        total += 1
                        valid += bool(s.get("valid"))
                        bad = unlisted_numbers(t, s["content"])
                        unlisted += bool(bad)
                        if bad and len(examples) < 3:
                            examples.append(f"{t['id']}: {', '.join(bad[:6])}")
                print("choice: mean regret (0 = the best allowed procedure)")
                for k in ("model: first answer", "model: mean over answers") + tuple(k for k in rows if k.startswith("model: best")) \
                        + ("rule default", "always robust_score|empirical"):
                    print(line(k, rows[k]))
                print("  first answer by kind:", ", ".join(f"{k} {st.mean(v):.3f} (n={len(v)})" for k, v in sorted(by_kind.items())))
                print(f"  valid answers {valid}/{total}; answers quoting a number not in the prompt {unlisted}/{total}"
                      + (f" (e.g. {'; '.join(examples)})" if examples else ""))
                print("  first answers:", ", ".join(f"{k} {c}" for k, c in picks.most_common(7)))
                bests = collections.Counter()
                for t, _ in ch:
                    r = {k: v for k, v in rewards[t["design_id"]].items() if k in t["allowed"]}
                    bests[max(r, key=r.get)] += 1
                print("  best procedures:", ", ".join(f"{k} {c}" for k, c in bests.most_common(7)))
            if sp:
                first = [sm[0] for _, sm in sp]
                every = [s for _, sm in sp for s in sm]
                print("question: first answer F1 {:.3f}, reward {:.3f}, any invented item {:.0%}, reward 1 {:.0%}; "
                      "over all answers F1 {:.3f}, reward {:.3f}, any invented {:.0%}".format(
                          st.mean(s.get("f1", 0) for s in first), st.mean(s["reward"] for s in first),
                          st.mean(s.get("invented", 0) > 0 for s in first), st.mean(s["reward"] >= 1 for s in first),
                          st.mean(s.get("f1", 0) for s in every), st.mean(s["reward"] for s in every),
                          st.mean(s.get("invented", 0) > 0 for s in every)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
