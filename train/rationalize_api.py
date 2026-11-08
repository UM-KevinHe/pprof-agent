"""A hosted model writes the reasoning for the procedure that simulation found best, for the SFT answers of the
procedure-choice tasks. The choice is never the hosted model's: the best allowed procedure on the task's design
(the reward table) is given to it, the answer schema fixes the three choice fields to that procedure, and the model
writes only `reasoning` and `reason` from the diagnostics, as the statistician who reaches that choice.

Every answer is checked by code and, on a problem, sent back with the problem named (at most TRIES calls):
  - the three choice fields are the given procedure;
  - every number in `reasoning` is written in the diagnostics or the policy (no rounded or invented values);
  - `reason` has no digit and at most MAX_REASON characters (the deployed schema and policy);
  - no claim that rows are close to independent, or that the correlation is near zero, when the within-patient
    correlation of residuals is above RHO_SMALL;
  - no mention of simulation, known truth, or of the choice having been given.
An answer that never passes is recorded with its problems; build_datasets.py then falls back to its template.

    python rationalize_api.py --tasks choice_tasks_v2.jsonl --scores scores.json --model gpt-6.1-sol --limit 20 --out r.jsonl
The key is read from OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pprof_agent import choose  # noqa: E402
import build_datasets  # noqa: E402
import sample  # noqa: E402

TRIES = 3
MAX_REASON = choose.CHOICE_SCHEMA["properties"]["reason"]["maxLength"]
RHO_SMALL = build_datasets.RHO_SMALL
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
INDEPENDENT = re.compile(r"(close to|nearly|approximately|essentially|effectively|roughly)\s+independent|"
                         r"correlation\s+(is\s+)?(near|close to|about)\s+zero|negligible\s+(within-patient\s+)?correlation", re.I)
LEAK = re.compile(r"simulat|known truth|was (given|provided|told)|told to|the given (procedure|choice)|reward", re.I)

INSTRUCTION = """

## The procedure for these data
For this task the procedure to report has already been chosen: {choice}. Write `reasoning` and `reason` as the statistician who reaches this choice by reading the diagnostics above: say what in the diagnostics supports each part of it (statistic, variance, reference). Use only what the diagnostics show, and quote any number exactly as it is written there. If the diagnostics do not single out a part of the choice, say which features are consistent with it and do not invent a cause. Never write that the rows of a patient are close to independent when the within-patient correlation of residuals is clearly above zero. Do not mention that the choice was given, and do not mention simulation."""


def schema_for(parts: Dict[str, str]) -> Dict[str, Any]:
    s = copy.deepcopy(choose.CHOICE_SCHEMA)
    s["properties"]["reason"].pop("maxLength", None)
    for f in ("statistic", "variance", "reference"):
        s["properties"][f] = {"type": "string", "enum": [parts[f]]}
    return s


def problems_of(content: str, parts: Dict[str, str], allowed_text: str, rho: Any) -> List[str]:
    try:
        a = json.loads(content)
    except Exception:  # noqa: BLE001
        return ["the answer is not valid JSON"]
    out = []
    for f in ("statistic", "variance", "reference"):
        if a.get(f) != parts[f]:
            out.append(f"{f} must be {parts[f]}")
    allowed = set(NUMBER.findall(allowed_text))
    bad = sorted({x for x in NUMBER.findall(a.get("reasoning", "")) if x not in allowed})
    if bad:
        out.append("these numbers are not written in the diagnostics: " + ", ".join(bad[:8]) + " (quote numbers exactly as written)")
    r = a.get("reason", "")
    if re.search(r"\d", r):
        out.append("`reason` must contain no number")
    if len(r) > MAX_REASON:
        out.append(f"`reason` must be at most {MAX_REASON} characters (it has {len(r)})")
    if rho is not None and rho > RHO_SMALL and INDEPENDENT.search(a.get("reasoning", "") + " " + r):
        out.append("the within-patient correlation is clearly above zero here, so do not say the rows are close to independent "
                   "or the correlation is near zero")
    if LEAK.search(a.get("reasoning", "") + " " + r):
        out.append("do not mention simulation, known truth, rewards, or that the choice was given")
    return out


def call(client: Any, kw: Dict[str, Any], tries: int = 5) -> Any:
    for i in range(tries):
        try:
            return client.chat.completions.create(**kw)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "insufficient_quota" in msg or "credit_balance" in msg:
                raise SystemExit("the API account has no credits left: " + msg[:300])
            if "reasoning_effort" in msg and "reasoning_effort" in kw:
                kw.pop("reasoning_effort")
                continue
            if i == tries - 1 or not any(s in msg for s in ("429", "Rate limit", "timeout", "Timeout", "502", "503", "Connection")):
                raise
            time.sleep(2 ** i + random.random())


def run_task(client: Any, a: argparse.Namespace, task: Dict[str, Any], rewards: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    r = {k: v for k, v in rewards.get(task["design_id"], {}).items() if k in task["allowed"]}
    if not r:
        return {"id": task["id"], "ok": False, "problems": ["no reward for this design"]}
    key = max(r, key=r.get)
    test, ref = key.split("|")
    parts = {**choose.parts_of(test), "reference": ref}
    choice = f"statistic {parts['statistic']}, variance {parts['variance']}, reference {ref} (the test {test} read against the {ref} null)"
    system = task["messages"][0]["content"] + INSTRUCTION.format(choice=choice)
    user = task["messages"][1]["content"]
    rho = (task.get("diag") or {}).get("within_patient_corr")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw: Dict[str, Any] = {"model": a.model, "max_completion_tokens": a.max_tokens,
                          "response_format": {"type": "json_schema", "json_schema": {"name": "choose_procedure", "schema": schema_for(parts), "strict": True}}}
    if a.effort:
        kw["reasoning_effort"] = a.effort
    tok = {"prompt": 0, "completion": 0}
    content, probs = "", ["no answer"]
    for attempt in range(1, TRIES + 1):
        try:
            resp = call(client, {**kw, "messages": msgs})
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"id": task["id"], "key": key, "ok": False, "problems": ["api error: " + str(exc)[:200]], "tokens": tok}
        tok["prompt"] += resp.usage.prompt_tokens
        tok["completion"] += resp.usage.completion_tokens
        content = resp.choices[0].message.content or ""
        probs = problems_of(content, parts, user + "\n" + task["messages"][0]["content"], rho)
        if not probs:
            return {"id": task["id"], "key": key, "ok": True, "attempts": attempt, "content": content, "tokens": tok}
        msgs = msgs + [{"role": "assistant", "content": content},
                       {"role": "user", "content": "Write the answer again. Problems with the last one: " + "; ".join(probs)}]
    return {"id": task["id"], "key": key, "ok": False, "attempts": TRIES, "content": content, "problems": probs, "tokens": tok}


def main(argv=None) -> int:
    from openai import OpenAI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rewards = sample.rewards_by_design(a.scores)
    tasks = [t for t in (json.loads(l) for l in open(a.tasks) if l.strip()) if "error" not in t and t.get("act") == "choose_procedure"]
    if a.limit:
        tasks = tasks[:a.limit]
    client = OpenAI(timeout=600)
    tok = {"prompt": 0, "completion": 0}
    ok = done = 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex, open(a.out, "w") as fh:
        for r in ex.map(lambda t: run_task(client, a, t, rewards), tasks):
            fh.write(json.dumps(r) + "\n")
            done += 1
            ok += r["ok"]
            for k in tok:
                tok[k] += (r.get("tokens") or {}).get(k, 0)
            if done % 100 == 0:
                print(f"{done}/{len(tasks)} tasks, {ok} passed, tokens {tok}", flush=True)
    print(f"{done} tasks, {ok} passed the checks -> {a.out}; tokens {tok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
