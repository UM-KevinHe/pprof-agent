"""A hosted model answers the procedure-choice tasks the local model is trained on: the benchmark arm "a general
purpose model, no training". The calls carry exactly what the local model sees (the task's system policy and
user observation, the act's schema as structured output); each answer is scored by the same table
(sample.score_choice: the reward of the chosen procedure on the task's design), and the token use of every call
is recorded so the cost is known.

Differences from the local calls, forced by the hosted API: the schema's `maxLength` is dropped (strict structured
output does not take it; an over-long `reason` is cut afterwards and counted), reasoning models take no temperature
(their default sampling is used), and one answer is drawn per request.

    python benchmark_api.py --tasks choice_tasks.jsonl --scores scores.json --families heldout.json \
        --model gpt-5 --n 1 --limit 20 --out bench.jsonl
The key is read from OPENAI_API_KEY, never from an argument.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pprof_agent import choose  # noqa: E402
import sample  # noqa: E402

MAX_REASON = choose.CHOICE_SCHEMA["properties"]["reason"]["maxLength"]


def api_schema() -> Dict[str, Any]:
    s = copy.deepcopy(choose.CHOICE_SCHEMA)
    s["properties"]["reason"].pop("maxLength", None)
    return s


def ask(client: Any, model: str, task: Dict[str, Any], effort: str, max_tokens: int, tries: int = 5) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"model": model, "messages": task["messages"][:2], "max_completion_tokens": max_tokens,
                          "response_format": {"type": "json_schema",
                                              "json_schema": {"name": "choose_procedure", "schema": api_schema(), "strict": True}}}
    if effort:
        kw["reasoning_effort"] = effort
    for i in range(tries):
        try:
            r = client.chat.completions.create(**kw)
            u = r.usage
            det = getattr(u, "completion_tokens_details", None)
            return {"content": r.choices[0].message.content or "", "finish": r.choices[0].finish_reason,
                    "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None}
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "insufficient_quota" in msg or "credit_balance" in msg:
                raise SystemExit("the API account has no credits left: " + msg[:300])
            if "reasoning_effort" in msg and "reasoning_effort" in kw:   # a model that takes no effort setting
                kw.pop("reasoning_effort")
                continue
            if i == tries - 1 or not any(s in msg for s in ("429", "Rate limit", "timeout", "Timeout", "502", "503", "Connection")):
                return {"content": "", "finish": "error", "error": msg[:300]}
            time.sleep(2 ** i + random.random())
    return {"content": "", "finish": "error"}


def run_task(client: Any, a: argparse.Namespace, task: Dict[str, Any], rewards: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out = []
    for _ in range(a.n):
        r = ask(client, a.model, task, a.effort, a.max_tokens)
        content, cut = r["content"], False
        try:
            obj = json.loads(content)
            if len(obj.get("reason", "")) > MAX_REASON:
                obj["reason"], cut = obj["reason"][:MAX_REASON], True
                content = json.dumps(obj)
        except Exception:  # noqa: BLE001
            pass
        out.append({**r, "content": content, "reason_cut": cut,
                    **sample.score_choice(task, content, rewards.get(task["design_id"], {}))})
    return {"id": task["id"], "act": task["act"], "model": a.model, "effort": a.effort, "samples": out}


def main(argv=None) -> int:
    from openai import OpenAI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", required=True, nargs="+")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--families", help="build_report.json: only tasks of its held-out families")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--effort", default="low", help="reasoning effort ('' to leave the model's default)")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rewards = sample.rewards_by_design(a.scores)
    tasks: List[Dict[str, Any]] = []
    for p in a.tasks:
        tasks += [t for t in (json.loads(l) for l in open(p) if l.strip())
                  if "error" not in t and t.get("act") == "choose_procedure" and t.get("design_id") in rewards]
    if a.families:
        held = set(json.load(open(a.families))["heldout_families"])
        tasks = [t for t in tasks if t["family"] in held]
    if a.limit:
        tasks = tasks[:a.limit]
    client = OpenAI(timeout=600)
    tok = {"prompt": 0, "completion": 0, "reasoning": 0}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex, open(a.out, "w") as fh:
        for r in ex.map(lambda t: run_task(client, a, t, rewards), tasks):
            fh.write(json.dumps(r) + "\n")
            for s in r["samples"]:
                tok["prompt"] += s.get("prompt_tokens") or 0
                tok["completion"] += s.get("completion_tokens") or 0
                tok["reasoning"] += s.get("reasoning_tokens") or 0
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(tasks)} tasks, tokens so far {tok}", flush=True)
    print(f"{done} tasks x {a.n} -> {a.out}; tokens {tok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
