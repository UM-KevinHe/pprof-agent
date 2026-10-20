"""A fake OpenAI-compatible client for the tests: answers every act with canned JSON that obeys its
schema, so the boundaries, their verifiers and the model planner path run without a model.

Each act's answer is a function of the user text (so a test can plant a ghost column, a wrong
evidence phrase or a digit and see the harness refuse it). `FakeClient(answers=...)` overrides acts.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional


class _Usage:
    prompt_tokens = 100
    completion_tokens = 50


class _Message:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = None
        self.model_extra = {}


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Completion:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def _default_intent(user: str) -> Dict[str, Any]:
    msg = user.split("The analyst wrote:\n", 1)[-1]
    low = msg.lower()
    if "hospital" in low:
        return {"reasoning": "out", "intents": [{"kind": "out_of_scope", "evidence": msg[:40], "out_of_scope_reason": "crossed_providers"}]}
    if any(w in low for w in ("outcome is", "adjust for", "identifies", "one row per", "higher rate")):
        return {"reasoning": "declares", "intents": [{"kind": "declare_specification", "evidence": msg[:40], "out_of_scope_reason": None}]}
    if "why" in low or "what changes" in low or "what would change" in low or ("which" in low and "flag" in low):
        return {"reasoning": "asks", "intents": [{"kind": "ask_about_result", "evidence": msg[:40], "out_of_scope_reason": None}]}
    if "funnel" in low or "plot" in low:
        return {"reasoning": "figure", "intents": [{"kind": "request_figure", "evidence": msg[:40], "out_of_scope_reason": None}]}
    if "how does" in low or "what is" in low:
        return {"reasoning": "method", "intents": [{"kind": "ask_about_method", "evidence": msg[:40], "out_of_scope_reason": None}]}
    return {"reasoning": "other", "intents": [{"kind": "other", "evidence": "", "out_of_scope_reason": None}]}


def _default_spec(user: str) -> Dict[str, Any]:
    """Reads a few fixed phrasings: 'the outcome is X', 'X identifies the facility/patient', 'adjust for A, B and C',
    'one row per patient-month', 'a higher rate is better/worse'."""
    msg = user.split("The analyst wrote:\n", 1)[-1]
    fields = []
    m = re.search(r"outcome is (\w+)", msg, re.I)
    if m:
        fields.append({"field": "outcome", "value": m.group(1), "evidence": m.group(0)})
    m = re.search(r"out of (\w+)", msg, re.I)
    if m:
        fields.append({"field": "trials", "value": m.group(1), "evidence": m.group(0)})
    m = re.search(r"(\w+) identifies the (facility|provider)", msg, re.I)
    if m:
        fields.append({"field": "provider", "value": m.group(1), "evidence": m.group(0)})
    m = re.search(r"(\w+) (?:identifies )?the patient", msg, re.I)
    if m:
        fields.append({"field": "patient", "value": m.group(1), "evidence": m.group(0)})
    m = re.search(r"adjust for ([\w, ]+?)(?:\.|$)", msg, re.I)
    if m:
        names = [n.strip() for n in re.split(r",|\band\b", m.group(1)) if n.strip()]
        fields.append({"field": "adjusters", "value": ", ".join(names), "evidence": m.group(0)})
    if re.search(r"patient-month|patient month", msg, re.I):
        fields.append({"field": "unit", "value": "patient_month", "evidence": re.search(r"patient[- ]month", msg, re.I).group(0)})
    if re.search(r"index discharge|per discharge", msg, re.I):
        fields.append({"field": "unit", "value": "index_discharge", "evidence": re.search(r"index discharge|per discharge", msg, re.I).group(0)})
    m = re.search(r"higher rate is (better|worse)", msg, re.I)
    if m:
        fields.append({"field": "event_direction", "value": f"higher_is_{m.group(1).lower()}", "evidence": m.group(0)})
    return {"reasoning": "quoted", "fields": fields}


def _default_next_step(user: str) -> Dict[str, Any]:
    """Walks the protected core as the harness lists it, then finishes; mirrors the default planner but
    through the schema, with a wordy reading."""
    m = re.search(r"Protected core still to run.*?: (.+)", user)
    pending = [] if not m or m.group(1).strip() == "complete" else [x.strip() for x in m.group(1).split(",")]
    args = {k: None for k in ("test", "stdz", "norm", "n_groups", "size_by", "se_source", "reference",
                              "method", "nfolds", "axis", "display", "of")}
    if pending:
        rid = pending[0]
        if rid == "fit":
            action = "fit"
        elif rid == "sm":
            action = "standardize"
            args.update({"stdz": "both", "norm": "median"})
        elif rid.startswith("test:"):
            action = "test_providers"
            args.update({"test": rid.split(":")[1]})
        elif rid.startswith("en:"):
            action = "empirical_null"
            args.update({"test": rid.split(":")[1], "n_groups": 4, "size_by": "patients"})
        elif rid.startswith("measure:"):
            action = "test_measure"
            args.update({"se_source": "robust_wald", "reference": "theoretical"})
        elif rid == "evaluation":
            action = "evaluate_model"
        else:
            action = "finish"
        return {"reasoning": "the core first", "reading": "The last result is as the rules expected.", "action": action,
                "args": args, "reason": "the protected core is not complete"}
    return {"reasoning": "done", "reading": "Nothing affordable would change the flags.", "action": "finish", "args": args,
            "reason": "the core is complete and the flags are stable"}


def _default_figures(user: str) -> Dict[str, Any]:
    m = re.search(r"Results available \(use their ids in `of`\): (.+)", user)
    ids = [x.strip() for x in (m.group(1).split(",") if m else [])]
    figs = []
    en = [i for i in ids if i.startswith("en:")]
    tests = [i for i in ids if i.startswith("test:")]
    if en:
        figs.append({"display": "funnel", "of": en[0], "reason": "the reported test under its reference"})
        figs.append({"display": "null_calibration", "of": en[0], "reason": "whether the theoretical null is credible"})
    elif tests:
        figs.append({"display": "funnel", "of": tests[0], "reason": "the reported test"})
    if "fit" in ids:
        figs.append({"display": "coefficient_forest", "of": "fit", "reason": "the adjusters' effects"})
        figs.append({"display": "flag_stability", "of": "fit", "reason": "how fragile the flags are"})
    figs.append({"display": "caterpillar", "of": "test:score", "reason": "intervals"})     # refused: the score test has none
    return {"reasoning": "chosen", "figures": figs}


def _default_prose(user: str) -> Dict[str, Any]:
    return {"reasoning": "refs", "declared": "The measure counts [outcome] per [unit]; [n_eligible] providers meet the threshold.",
            "model": "The model adjusts for [n_adjusters] risk factors with a within-provider concordance of [c_within].",
            "measures": "Both conventions were computed and [sm_flips] providers flip between them.",
            "flags": "Under the [recommended_test] with the [recommended_reference] null, [flagged_worse] providers are worse than expected.",
            "alternatives": "The reference axis changes the most providers [changed_max_reference]; the other axes change 3 at most."}


def _default_word(user: str) -> Dict[str, Any]:
    return {"reasoning": "words", "text": "A few fields are still open because the data cannot tell them apart."}


DEFAULTS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "intent": _default_intent, "spec_extraction": _default_spec, "next_step": _default_next_step,
    "figure_selection": _default_figures, "report_prose": _default_prose,
    "ask": _default_word, "refusal": _default_word, "explain": _default_word,
}


class FakeClient:
    def __init__(self, answers: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None):
        self.answers = dict(DEFAULTS)
        self.answers.update(answers or {})
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model: str, messages, max_tokens: int = 0, response_format=None, **kw):
        name = (response_format or {}).get("json_schema", {}).get("name")
        user = messages[-1]["content"]
        self.calls.append({"name": name, "user": user, "system": messages[0]["content"]})
        fn = self.answers.get(name)
        if fn is None:
            raise ValueError(f"fake client has no answer for act {name}")
        return _Completion(json.dumps(fn(user)))
