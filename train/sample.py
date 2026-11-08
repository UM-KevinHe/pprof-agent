"""Sample the local model's answers on the training tasks and score every answer: the material SFT and DPO
are built from (build_datasets.py).

The calls are the deployed ones: the task's own system and user turns, constrained decoding with the act's
schema, thinking off. Several answers per task come from one request (`n`) at a sampling temperature, so
the answers differ and some are better than others. Scores:

  choose_procedure   the reward of the answered procedure on the task's design (summarize_oc.py: the
                     Neyman-Pearson score per tail, the worse tail weighted); an answer outside the allowed
                     set scores the floor
  spec_extraction    F1 of the extraction against the gold fields, after the harness's verifier
                     (spec_tasks.score), so only what the harness would accept counts, minus INVENTED for every
                     item the verifier dropped that the message does not state (a column, a value or evidence
                     the analyst never wrote); a column the message does write but the file lacks is not
                     charged, since the harness's note on it is what tells the analyst

    python sample.py --tasks choice_tasks.jsonl --scores scores.json --endpoint http://localhost:PORT/v1 --n 8 --out samples.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pprof_agent import boundary, choose, rules  # noqa: E402
from pprof_agent.specification import MeasureSpecification  # noqa: E402
import spec_tasks  # noqa: E402

SCHEMAS = {"choose_procedure": (choose.CHOICE_SCHEMA, 700), "spec_extraction": (boundary.SPEC_SCHEMA, 900)}
INVENTED = 0.05


def rewards_by_design(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    return {row["id"]: {k: v["reward"] for k, v in row["scores"].items()} for row in json.load(open(path))}


def score_choice(task: Dict[str, Any], content: str, rewards: Dict[str, float]) -> Dict[str, Any]:
    try:
        a = json.loads(content)
        test = choose.test_of(a["statistic"], a["variance"])
        key = f"{test}|{a['reference']}"
    except Exception:  # noqa: BLE001
        return {"key": None, "valid": False, "reward": rules.CALIBRATION["floor"]}
    valid = test is not None and key in task["allowed"] and key in rewards
    return {"key": key, "valid": valid, "reward": rewards[key] if valid else rules.CALIBRATION["floor"]}


def score_spec(task: Dict[str, Any], content: str) -> Dict[str, Any]:
    user = task["messages"][1]["content"]
    cols = user.split("\n", 1)[0].replace("Columns of the file: ", "").split(", ")
    message = user.split("The analyst wrote:\n", 1)[1]
    try:
        spec, notes = boundary.verify_extraction(json.loads(content), message, {"columns": [{"name": c} for c in cols]},
                                                 MeasureSpecification())
    except Exception:  # noqa: BLE001
        return {"valid": False, "reward": 0.0}
    s = spec_tasks.score(task["gold"], spec)
    k = invented(notes, message)
    return {"valid": True, "reward": max(0.0, s["f1"] - INVENTED * k), "f1": s["f1"], "precision": s["precision"],
            "recall": s["recall"], "notes": len(notes), "invented": k}


def _written(name: str, message: str) -> bool:
    """Whether the message writes this name, ignoring case and separators (`pid` for PID, `DUAL ELIGIBLE`)."""
    norm = lambda x: " ".join(re.sub(r"[_\W]+", " ", str(x).lower()).split())  # noqa: E731
    return bool(norm(name)) and f" {norm(name)} " in f" {norm(message)} "


def invented(notes: List[str], message: str) -> int:
    """How many of the verifier's drops are things the message does not state (boundary.verify_extraction's
    notes): a dropped column or adjuster the message does not write in any case or spacing, a number or
    evidence not in the message, a value outside the allowed set. A name the message does write (in another
    spelling, or absent from the file) is not counted: the first already costs F1, the second's note is what
    tells the analyst."""
    k = 0
    for note in notes:
        m = re.match(r"adjusters: not written in the message or not in the file: (.*)$", note)
        if m:
            k += sum(1 for x in m.group(1).split(",") if x.strip() and not _written(x, message))
            continue
        m = re.match(r"\w+: (.*) is not written in the message; dropped$", note)
        if m:
            k += 0 if _written(m.group(1), message) else 1
        elif "is named in the message but is not a column" not in note:
            k += 1
    return k


def run_task(client: Any, model: str, task: Dict[str, Any], n: int, temperature: float,
             rewards: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    schema, max_tokens = SCHEMAS[task["act"]]
    try:
        resp = client.chat.completions.create(
            model=model, messages=task["messages"][:2], n=n, temperature=temperature, top_p=0.95, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": task["act"], "schema": schema, "strict": True}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as exc:  # noqa: BLE001
        return {"id": task["id"], "error": str(exc)[:300]}
    out = []
    for c in resp.choices:
        content = c.message.content or ""
        if task["act"] == "choose_procedure":
            sc = score_choice(task, content, rewards.get(task["design_id"], {}))
        else:
            sc = score_spec(task, content)
        out.append({"content": content, "finish": c.finish_reason, **sc})
    return {"id": task["id"], "act": task["act"], "samples": out}


def main(argv=None) -> int:
    from openai import OpenAI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, nargs="+")
    ap.add_argument("--scores", help="scores.json of the designs (choice tasks)")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="qwen3-8b-awq")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rewards = rewards_by_design(a.scores)
    tasks: List[Dict[str, Any]] = []
    for p in a.tasks:
        tasks += [t for t in (json.loads(l) for l in open(p) if l.strip()) if "error" not in t]
    tasks = [t for t in tasks if t["act"] != "choose_procedure" or t["design_id"] in rewards]
    if a.limit:
        tasks = tasks[:a.limit]
    client = OpenAI(base_url=a.endpoint, api_key="none", timeout=600)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex, open(a.out, "w") as fh:
        for r in ex.map(lambda t: run_task(client, a.model, t, a.n, a.temperature, rewards), tasks):
            fh.write(json.dumps(r) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(tasks)} tasks sampled", flush=True)
    print(f"{done} tasks sampled -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
