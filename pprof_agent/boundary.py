"""Where the language model acts, and how: one constrained generation per act, no tools, no transcript.

Every act sends `[system = the policy file, user = this turn's material]`, constrains the answer to
the act's JSON schema (`response_format: json_schema`, strict; vLLM's xgrammar admits only tokens the
grammar allows), and parses the result. `reasoning` is the first field of every schema (reason, then
commit) and is recorded, never acted on. Quoting acts (intent, spec_extraction, figure_selection) run
at temperature 0; writing acts (report_prose, ask, refusal, explain) at 0.7; the planner (next_step)
thinks (Qwen3's switch) with its own token allowance. Every call is logged with tokens, time, finish
reason and the raw text.

Each act has a verifier here: what the model returned is checked against the message, the file, the
closed sets and the session before it takes effect. A failed check drops the item or refuses the
proposal by name; nothing is repaired by asking the model again with its own output.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import actions, policy, textmatch
from .specification import ENUMS, MeasureSpecification, profile_columns

QUOTING_ACTS = ("intent", "spec_extraction", "figure_selection")
WRITING_ACTS = ("report_prose", "ask", "refusal", "explain")
THINKING_ACTS = ("next_step",)
THINKING_ALLOWANCE = int(os.environ.get("PPROF_AGENT_THINKING_ALLOWANCE", "6000") or 0)
CALLS: List[Dict[str, Any]] = []
_DIGIT = re.compile(r"\d")


def thinks(name: str) -> bool:
    env = os.environ.get("PPROF_AGENT_THINKING", "").strip().lower()
    if env in ("on", "1", "true"):
        return True
    if env in ("off", "0", "false"):
        return False
    return name in THINKING_ACTS


def decoding_for(name: str) -> Dict[str, Any]:
    if thinks(name):
        d: Dict[str, Any] = {"temperature": 0.6, "top_p": 0.95}
    elif name in WRITING_ACTS:
        d = {"temperature": 0.7, "top_p": 0.8}
    else:
        d = {"temperature": 0.0, "top_p": 1.0}
    if os.environ.get("PPROF_AGENT_SEED", "").strip():
        d["seed"] = int(os.environ["PPROF_AGENT_SEED"])
    d["thinking"] = thinks(name)
    return d


def _chat(client: Any, model: str, system: str, user: str, schema: Dict[str, Any], name: str,
          max_tokens: int = 1200, record_to: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    t0 = time.time()
    dec = decoding_for(name)
    kw: Dict[str, Any] = {"temperature": dec["temperature"], "top_p": dec["top_p"]}
    if "seed" in dec:
        kw["seed"] = dec["seed"]
    allowance = 0
    kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(dec["thinking"])}}
    if dec["thinking"]:
        allowance = THINKING_ALLOWANCE
        max_tokens += allowance
    fmt = {"response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, **fmt, **kw)
    choice = completion.choices[0]
    text = choice.message.content or ""
    usage = getattr(completion, "usage", None)
    rec: Dict[str, Any] = {"name": name, "model": model, "temperature": dec["temperature"], "thinking": dec["thinking"],
                           "max_tokens": max_tokens, "thinking_allowance": allowance,
                           "prompt_tokens": getattr(usage, "prompt_tokens", None),
                           "completion_tokens": getattr(usage, "completion_tokens", None),
                           "finish_reason": getattr(choice, "finish_reason", None),
                           "seconds": round(time.time() - t0, 2), "policy_sha256": policy.sha256(name)}
    think = getattr(choice.message, "reasoning_content", None)
    if not think:
        extra = getattr(choice.message, "model_extra", None) or {}
        think = extra.get("reasoning") or extra.get("reasoning_content")
    if think:
        rec["thinking_chars"] = len(str(think))
        rec["thinking_text"] = str(think)[:4000]
    if not text.strip():
        rec["error"] = f"no answer (finish_reason={rec['finish_reason']})"
        rec["raw_text"] = ""
        (record_to if record_to is not None else CALLS).append(rec)
        raise ValueError(f"{name}: the model wrote no answer ({rec['error']})")
    try:
        out = json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = re.sub(r"\\'", "'", text)
        try:
            out = json.loads(repaired)
            rec["repaired"] = "escaped single quote"
        except json.JSONDecodeError:
            rec["error"] = f"invalid JSON: {exc}"
            rec["raw_text"] = text[:4000]
            (record_to if record_to is not None else CALLS).append(rec)
            raise ValueError(f"{name}: the model's output was not valid JSON: {exc}") from exc
    rec["raw_text"] = text[:4000]
    rec["first_key"] = next(iter(out), None) if isinstance(out, dict) else None
    (record_to if record_to is not None else CALLS).append(rec)
    return out


def _nullable(sub: Dict[str, Any]) -> Dict[str, Any]:
    return {"anyOf": [sub, {"type": "null"}]}


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


# ===================================================================================== spec extraction
SPEC_FIELDS = ("outcome", "event_direction", "unit", "provider", "patient", "trials", "adjusters", "eligibility",
               "standardization", "norm", "reference", "consequential_tail")
COLUMN_FIELDS = ("outcome", "provider", "patient", "trials")

SPEC_SCHEMA = _obj({
    "reasoning": {"type": "string"},
    "fields": {"type": "array", "maxItems": 12,
               "items": _obj({"field": {"type": "string", "enum": list(SPEC_FIELDS)},
                              "value": {"type": "string", "maxLength": 400},
                              "evidence": {"type": "string", "maxLength": 300}})},
})


def spec_extraction_user(message: str, profile: Dict[str, Any]) -> str:
    """The user turn of the extraction call (also what the training data for this act is built on)."""
    cols = profile_columns(profile)
    return ("Columns of the file: " + ", ".join(cols) + "\n\n"
            + "Fields and their allowed values: " + "; ".join(
                f"{f} in {{{', '.join(ENUMS[f])}}}" if f in ENUMS else f"{f}: a column name" if f in COLUMN_FIELDS
                else f"{f}: {'column names, comma separated' if f == 'adjusters' else 'an integer'}" for f in SPEC_FIELDS)
            + "\n\nThe analyst wrote:\n" + message)


def extract_specification(client: Any, model: str, message: str, profile: Dict[str, Any],
                          spec: MeasureSpecification, record_to: Optional[list] = None) -> Tuple[MeasureSpecification, List[str]]:
    """Boundary 1: read the specification out of prose; the harness checks every field."""
    raw = _chat(client, model, policy.load("spec_extraction"), spec_extraction_user(message, profile), SPEC_SCHEMA,
                "spec_extraction", max_tokens=900, record_to=record_to)
    return verify_extraction(raw, message, profile, spec)


def verify_extraction(raw: Dict[str, Any], message: str, profile: Dict[str, Any],
                      spec: MeasureSpecification) -> Tuple[MeasureSpecification, List[str]]:
    cols = profile_columns(profile)
    notes: List[str] = []
    sp = spec
    for item in raw.get("fields") or []:
        f, v, ev = item.get("field"), str(item.get("value") or "").strip(), str(item.get("evidence") or "")
        if f not in SPEC_FIELDS or not v:
            continue
        if not textmatch.phrase_in(message, ev):
            notes.append(f"{f}: the quoted evidence is not in the message; dropped")
            continue
        if f in COLUMN_FIELDS:
            if f == "patient" and v.lower() == "none":
                sp = sp.set("patient", "none", "quoted")
                continue
            if not textmatch.mentions(message, v, cols):
                notes.append(f"{f}: {v} is not written in the message; dropped")
                continue
            if v not in cols:
                notes.append(f"{f}: {v} is named in the message but is not a column of the file")
                continue
            sp = sp.set(f, v, "quoted")
        elif f == "adjusters":
            names = [n.strip() for n in re.split(r"[,;]+", v) if n.strip()]
            kept, bad = [], []
            for n in names:
                if n in cols and textmatch.mentions(message, n, cols):
                    kept.append(n)
                else:
                    bad.append(n)
            if bad:
                notes.append("adjusters: not written in the message or not in the file: " + ", ".join(bad))
            if kept:
                sp = sp.set("adjusters", kept, "quoted")
        elif f == "eligibility":
            m = re.search(r"\d+", v)
            if m and textmatch.phrase_in(message, m.group()):
                sp = sp.set("eligibility", int(m.group()), "quoted")
            else:
                notes.append("eligibility: the number is not in the message; dropped")
        else:
            vv = normalize_enum(f, v)
            if vv in ENUMS[f]:
                sp = sp.set(f, vv, "quoted")
            else:
                notes.append(f"{f}: {v} is not one of {ENUMS[f]}; dropped")
    return sp, notes


_ENUM_KEYWORDS = {
    "unit": (("discharge", "index_discharge"), ("month", "patient_month"), ("year", "patient_year"), ("record", "patient_record"),
             ("other", "other")),
    "event_direction": (("worse", "higher_is_worse"), ("better", "higher_is_better")),
    "standardization": (("both", "both"), ("indirect", "indirect"), ("direct", "direct")),
    "norm": (("weight", "size_weighted_mean"), ("mean", "size_weighted_mean"), ("median", "median")),
    "reference": (("both", "both"), ("empirical", "empirical"), ("theoretical", "theoretical")),
    "consequential_tail": (("both", "both"), ("worse", "worse"), ("better", "better")),
}


def normalize_enum(field: str, value: str) -> str:
    """A choice field's value as the model wrote it, mapped onto the enumeration: exact after case and
    separators, else by the first keyword that identifies one value (`patient_discharge` -> index_discharge,
    `patient month` -> patient_month). A value no keyword identifies is returned as written and then dropped."""
    vv = value.lower().strip().replace(" ", "_").replace("-", "_")
    if vv in ENUMS.get(field, ()):
        return vv
    for word, target in _ENUM_KEYWORDS.get(field, ()):
        if word in vv:
            return target
    return vv


# ===================================================================================== the planner
ARG_SCHEMA = _obj({
    "test": _nullable({"type": "string", "enum": list(actions.rules.TESTS)}),
    "stdz": _nullable({"type": "string", "enum": ["indirect", "direct", "both"]}),
    "norm": _nullable({"type": "string", "enum": ["median", "size_weighted_mean"]}),
    "n_groups": _nullable({"type": "integer"}),
    "size_by": _nullable({"type": "string", "enum": ["patients", "rows"]}),
    "se_source": _nullable({"type": "string", "enum": ["model", "robust_wald", "stabilized_wald"]}),
    "reference": _nullable({"type": "string", "enum": ["theoretical", "empirical"]}),
    "method": _nullable({"type": "string", "enum": ["lasso", "group"]}),
    "nfolds": _nullable({"type": "integer"}),
    "axis": _nullable({"type": "string", "enum": ["statistic", "variance", "reference", "standardization"]}),
    "display": _nullable({"type": "string", "enum": list(actions.DISPLAYS)}),
    "of": _nullable({"type": "string", "maxLength": 60}),
    "tests": _nullable({"type": "array", "items": {"type": "string", "enum": list(actions.rules.TESTS)}, "maxItems": 7}),
    "n_sim": _nullable({"type": "integer"}),
    "max_providers": _nullable({"type": "integer"}),
    "provider": _nullable({"type": "string", "maxLength": 40}),
    "by": _nullable({"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": 8}),
})
PLANNER_ACTIONS = [n for n, c in actions.CARDS.items() if c.available]
NEXT_STEP_SCHEMA = _obj({
    "reasoning": {"type": "string"},
    "reading": {"type": "string", "maxLength": 600},
    "action": {"type": "string", "enum": PLANNER_ACTIONS},
    "args": ARG_SCHEMA,
    "reason": {"type": "string", "maxLength": 300},
})


def render_observation(obs: Dict[str, Any], playbook: str = "") -> str:
    spec = obs.get("specification") or {}
    card = obs.get("card") or {}
    adm = obs.get("admissible") or {}
    lines = [f"State: {obs.get('state_line')}", "",
             "The measure as declared (field: value [source]):"]
    for k, v in spec.items():
        if v is not None and k not in ("select_from", "groups"):
            vv = f"{len(v)} columns" if k == "adjusters" else v
            lines.append(f"  {k}: {vv} [{(obs.get('sources') or {}).get(k, '')}]")
    lines += ["", "The data (consequence card):",
              f"  rows {card.get('rows_complete')}, patients {card.get('patients')}, eligible providers {card.get('eligible_providers')} of {card.get('providers')}, "
              f"event rate {100 * (card.get('event_rate') or 0):.1f}%, rows per patient up to {card.get('max_rows_per_patient')}, "
              f"median patients per provider {(card.get('patients_per_provider') or {}).get('median')}, "
              f"providers with invariant outcomes {(card.get('invariant_providers') or {}).get('total')}, "
              f"providers with three patients or fewer {(card.get('small_providers') or {}).get('le_3')}",
              "", "What the data admit (the harness's rules; you choose within this):",
              f"  statistics computed in the protected core: {', '.join(adm.get('tests_computed') or [])}",
              f"  admissible as the reported test: {', '.join(adm.get('tests_reported') or [])}; rule default {adm.get('default_test')}",
              f"  reference distributions computed: {', '.join(adm.get('references_computed') or [])}",
              f"  direct standardization tested on its own scale: {adm.get('direct_transport')}"]
    for k, v in (adm.get("reasons") or {}).items():
        lines.append(f"  - {k}: {v}")
    lines += ["", "Actions (the closed set):"] + ["  " + a for a in obs.get("actions") or []]
    pend = obs.get("core_pending") or []
    lines += ["", f"Protected core still to run (the harness runs it before finishing whatever you decide): {', '.join(pend) if pend else 'complete'}",
              f"Budget left: {obs.get('budget_left')} units; steps left: {obs.get('steps_left')}", ""]
    if obs.get("results"):
        lines.append("Already run, never to be proposed again: " + ", ".join(r["id"] for r in obs["results"]))
    if obs.get("available"):
        lines.append("Still open to you now, with arguments the harness accepts (anything else, and any repeat, is refused): "
                     + "; ".join(obs["available"]))
        lines.append("Results so far (summaries; numbers here are for your reading, never for the report):")
        for r in obs["results"]:
            lines.append(f"  {r['id']} [{r['source']}]: " + json.dumps(r.get("summary") or {}, default=str)[:700])
    if obs.get("last_result"):
        lines.append("The last result: " + json.dumps(obs["last_result"], default=str)[:1500])
    if obs.get("problems"):
        lines += ["", "Your previous proposal was refused:"] + ["  - " + p for p in obs["problems"]]
    if playbook:
        lines += ["", "Notes from the profiling playbook that apply here:", playbook]
    return "\n".join(lines)


def next_step(client: Any, model: str, obs: Dict[str, Any], playbook: str = "",
              record_to: Optional[list] = None) -> Dict[str, Any]:
    """The planner act: one constrained call returning a proposal the loop validates."""
    raw = _chat(client, model, policy.load("next_step"), render_observation(obs, playbook), NEXT_STEP_SCHEMA,
                "next_step", max_tokens=900, record_to=record_to)
    args = {k: v for k, v in (raw.get("args") or {}).items() if v is not None}
    name = raw.get("action")
    # keep only the arguments the card takes; the harness validates the rest
    card = actions.CARDS.get(name)
    if card is not None:
        args = {k: v for k, v in args.items() if k in card.args}
    reading = str(raw.get("reading") or "")
    return {"reading": reading, "action": name, "args": args, "reason": str(raw.get("reason") or ""),
            "reading_has_digit": bool(_DIGIT.search(reading))}


def model_planner(client: Any, model: str, session: Any, playbook_fn: Optional[Callable[[Any], str]] = None) -> Callable:
    def plan(obs: Dict[str, Any]) -> Dict[str, Any]:
        pb = playbook_fn(session) if playbook_fn else ""
        try:
            prop = next_step(client, model, obs, pb, record_to=session.model_calls)
        except ValueError as exc:
            session.refusals.append(f"planner call failed: {exc}")
            return {"reading": "", "action": "finish", "args": {}, "reason": "planner call failed; harness default"}
        return prop
    return plan


# ===================================================================================== figures
FIGURE_SCHEMA = _obj({
    "reasoning": {"type": "string"},
    "figures": {"type": "array", "maxItems": 6,
                "items": _obj({"display": {"type": "string", "enum": list(actions.DISPLAYS)},
                               "of": {"type": "string", "maxLength": 60},
                               "reason": {"type": "string", "maxLength": 200}})},
})


def select_figures(client: Any, model: str, session: Any, catalogue: str, record_to: Optional[list] = None) -> List[Dict[str, Any]]:
    rec = session.recommended or {}
    ids = [r.id for r in session.results if r.action in ("test_providers", "empirical_null", "fit", "evaluate_model", "sm")]
    user = ("The reported configuration: " + json.dumps({k: rec.get(k) for k in ("test", "reference", "statistic", "variance")})
            + "\nResults available (use their ids in `of`): " + ", ".join(ids)
            + "\nThe comparisons across alternatives: " + json.dumps({a: (session.get(f"compare:{a}") or type("R", (), {"summary": {}})()).summary.get("max_changed")
                                                                     for a in ("statistic", "variance", "reference", "standardization")})
            + "\n\nThe display catalogue:\n" + catalogue)
    raw = _chat(client, model, policy.load("figure_selection"), user, FIGURE_SCHEMA, "figure_selection",
                max_tokens=700, record_to=record_to)
    chosen = []
    for f in raw.get("figures") or []:
        probs = [p for p in actions.validate("figure", {"display": f.get("display"), "of": f.get("of")}, session)
                 if "already run" not in p]
        if probs:
            session.refusals.append(f"figure {f.get('display')} of {f.get('of')} not applicable: {'; '.join(probs)}")
            continue
        chosen.append({"display": f["display"], "of": f["of"], "reason": str(f.get("reason") or "")})
    return chosen


# ===================================================================================== prose
PROSE_SECTIONS = ("declared", "model", "measures", "flags", "alternatives")
PROSE_SCHEMA = _obj({"reasoning": {"type": "string"},
                     **{s: {"type": "string", "maxLength": 900} for s in PROSE_SECTIONS}})
_REF = re.compile(r"\[([a-z_0-9:]+)\]")


def write_prose(client: Any, model: str, refs: Dict[str, Any], facts: List[str], record_to: Optional[list] = None,
                columns: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[str]]:
    user = ("The references you may name (each with what it is):\n" + "\n".join(f"  [{k}]: {_meaning(k)}" for k in refs)
            + "\n\nThe facts of this analysis, in words:\n" + "\n".join("  - " + f for f in facts))
    raw = _chat(client, model, policy.load("report_prose"), user, PROSE_SCHEMA, "report_prose", max_tokens=1400,
                record_to=record_to)
    out: Dict[str, str] = {}
    notes: List[str] = []
    for s in PROSE_SECTIONS:
        text = str(raw.get(s) or "").strip()
        if not text:
            continue
        bad = [m for m in _REF.findall(text) if m not in refs]
        if bad:
            notes.append(f"{s}: unknown references dropped: {', '.join(bad)}")
            text = _REF.sub(lambda m: m.group(0) if m.group(1) in refs else "", text)
        masked = _REF.sub("", text)
        for tok in sorted(set(columns or []), key=len, reverse=True):
            if tok:
                masked = masked.replace(tok, " ")
        if _DIGIT.search(masked):
            notes.append(f"{s}: the paragraph wrote a digit; dropped")
            continue
        out[s] = text
    return out, notes


_MEANINGS = {
    "n_rows": "rows analysed", "n_patients": "patients", "n_providers": "providers in the file", "n_eligible": "providers meeting the eligibility threshold",
    "eligibility": "the eligibility threshold (patients)", "event_rate_pct": "the event rate in percent", "n_invariant": "providers whose outcomes are all events or none",
    "rows_per_patient_max": "the most rows one patient contributes", "patients_per_provider_median": "the median provider's number of patients",
    "outcome": "the outcome column", "unit": "what one row is", "event_direction": "whether a higher rate is worse or better",
    "provider_column": "the provider column", "patient_column": "the patient column", "trials_column": "the trials column (binomial outcomes)",
    "n_adjusters": "the number of risk adjusters", "standardization": "the standardization convention", "norm": "the norm",
    "consequential_tail": "the tail that carries consequences", "level": "the confidence level", "recommended_test": "the reported test",
    "recommended_reference": "the reported reference distribution", "recommended_statistic": "the statistic of the reported test",
    "recommended_variance": "the variance treatment of the reported test", "auc": "the C statistic of the fitted model including provider effects",
    "iterations": "Newton iterations to convergence", "n_providers_fitted": "providers fitted", "n_providers_excluded": "providers below the threshold",
    "c_overall": "the C statistic overall", "c_within": "the C statistic within provider (the risk adjustment alone)",
    "sm_flips": "providers whose verdict flips between indirect and direct standardization", "population_rate_pct": "the population rate in percent",
    "flagged_worse": "providers flagged worse than expected", "flagged_better": "providers flagged better than expected",
    "flagged_as_expected": "providers as expected", "flagged_worse_pct": "percent flagged worse", "flagged_better_pct": "percent flagged better",
    "en_scale_max": "the widest size group's scale under the empirical null", "en_scale_ratio": "the ratio of the widest to the narrowest group scale",
    "en_location_max": "the largest absolute location of a size group under the empirical null",
    "en_scale_threshold": "the group scale above which the empirical null is reported", "en_location_threshold": "the group location beyond which the empirical null is reported",
    "changed_max_statistic": "the most providers that change status under another statistic", "changed_max_variance": "... under another variance treatment",
    "changed_max_reference": "... under the other reference distribution", "changed_max_standardization": "... under the other standardization",
    "n_selected": "adjusters kept by the selection", "n_candidates": "candidate adjusters",
    "few_patients_threshold": "the median patients per provider below which the stabilized variance replaces the robust one",
    "en_min_providers": "the fewest eligible providers for a size-stratified empirical null",
    "stabilized_max_rows": "the most rows per patient for which the stabilized variance is computed",
    "exact_max_trials": "the largest provider size for which the exact test is enumerated",
    "worse_under_every_alternative": "providers flagged worse under the reported configuration and under every computed alternative",
    "better_under_every_alternative": "providers flagged better under the reported configuration and under every computed alternative",
    "n_alternatives": "the number of computed alternative configurations",
    "calibration_rate_pct": "the reported test's simulated type I error in the consequential tail, in percent",
    "calibration_nominal_pct": "the nominal one-sided level in percent",
    "calibration_n_sim": "the number of simulated data sets in the calibration",
    "calibration_default_rate_pct": "the rule default test's simulated type I error in the consequential tail, in percent",
    "calibration_power_pct": "the reported test's simulated power in the consequential tail (share of planted providers found), in percent",
    "calibration_default_power_pct": "the rule default test's simulated power in the consequential tail, in percent",
    "calibration_effect_or": "the odds ratio of the departures planted in the calibration simulation",
}


def _meaning(key: str) -> str:
    if key in _MEANINGS:
        return _MEANINGS[key]
    from . import compare
    for prefix, what in (("worse_under_", "providers flagged worse under "), ("better_under_", "providers flagged better under "),
                         ("changed_under_", "providers whose status changes under ")):
        if key.startswith(prefix):
            return what + compare.alternative_words(key[len(prefix):]) + " instead of the reported configuration"
    return key.replace("_", " ")


# ===================================================================================== intent, ask, refusal, explain
INTENT_KINDS = ("declare_specification", "edit_specification", "request_analysis", "ask_about_result",
                "ask_about_method", "request_figure", "out_of_scope", "other")
OUT_OF_SCOPE_REASONS = ("crossed_providers", "time_to_event_outcome", "count_outcome", "mixed_effects", "reliability",
                        "individual_prediction", "causal_claim", "data_processing", "other")
INTENT_SCHEMA = _obj({
    "reasoning": {"type": "string"},
    "intents": {"type": "array", "minItems": 1, "maxItems": 4,
                "items": _obj({"kind": {"type": "string", "enum": list(INTENT_KINDS)},
                               "evidence": {"type": "string", "maxLength": 300},
                               "out_of_scope_reason": _nullable({"type": "string", "enum": list(OUT_OF_SCOPE_REASONS)})})},
})


def classify(client: Any, model: str, message: str, state_line: str, record_to: Optional[list] = None) -> List[Dict[str, Any]]:
    user = f"State: {state_line}\n\nThe analyst wrote:\n{message}"
    raw = _chat(client, model, policy.load("intent"), user, INTENT_SCHEMA, "intent", max_tokens=500, record_to=record_to)
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in raw.get("intents") or []:
        kind = it.get("kind")
        if kind not in INTENT_KINDS or kind in seen:
            continue
        ev = str(it.get("evidence") or "")
        verified = textmatch.phrase_in(message, ev)
        if kind == "out_of_scope" and not verified:
            kind = "other"
        if kind == "ask_about_result" and "result: present" not in state_line and "DONE" not in state_line:
            kind = "other"
        seen.add(kind)
        out.append({"kind": kind, "evidence": ev, "evidence_verified": verified,
                    "out_of_scope_reason": it.get("out_of_scope_reason")})
    return out or [{"kind": "other", "evidence": "", "evidence_verified": False, "out_of_scope_reason": None}]


WORDING_SCHEMA = _obj({"reasoning": {"type": "string"}, "text": {"type": "string", "maxLength": 1200}})


def word(client: Any, model: str, act: str, material: str, columns: List[str], record_to: Optional[list] = None,
         allow: Optional[List[str]] = None) -> Optional[str]:
    """A writing act (ask / refusal / explain): the words only; refused when they carry a digit (other than
    inside a column name, a reference or an allowed token such as a provider id) or advice."""
    raw = _chat(client, model, policy.load(act), material, WORDING_SCHEMA, act, max_tokens=900 if act == "explain" else 600,
                record_to=record_to)
    text = str(raw.get("text") or "").strip()
    masked = _REF.sub("", text)
    for tok in sorted(set(list(columns) + list(allow or [])), key=len, reverse=True):
        if tok:
            masked = masked.replace(tok, " ")
    if not text or _DIGIT.search(masked):
        return None
    if act == "refusal" and re.search(r"\b(you should|you could|you can|you may|you might|try |consider )", text.lower()):
        return None
    return text
