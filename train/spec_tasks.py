"""Training tasks for reading the measure out of the analyst's words (the extraction act, boundary 1).

Each task is a file's column list and a message an analyst might write; its answer is the gold extraction in
the act's own schema (SPEC_SCHEMA): the fields the message states, each with the column or value and the
verbatim phrase that states it. The prompt is the deployed one (policy `spec_extraction` +
`boundary.spec_extraction_user`), so what is learned is what runs. Every gold answer is passed through the
harness's verifier (`boundary.verify_extraction`) and kept only if all its fields are accepted unchanged.

Situations covered, each a reason the act fails in practice:
  styles        prose, bullets, terse key-value, a forwarded note with filler sentences
  columns       several naming conventions; names written in another case or with spaces for underscores
  binomial      "X out of Y" names the trials column
  left open     fields the message never states (they must be left out, not guessed)
  descriptions  "adjust for the usual demographics", "by facility": a description is not a column
  ghosts        a named adjuster that is not in the file (dropped from the gold)
  hedges        "maybe direct, not sure yet": not a statement, left out
  preferences   "adjust for all of these", "keep the model small": recorded in meta for the act that uses them

    python spec_tasks.py --n 2000 --seed 11 --out spec_tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from pprof_agent import boundary, policy
from pprof_agent.specification import MeasureSpecification

PROVIDER = ("FACILITY", "PROVFS", "CCN", "facility_id", "PROVIDER", "DIALYSIS_FACILITY")
PATIENT = ("PATIENT", "KECC_ID", "patient_id", "BENE_ID", "PID")
OUTCOMES = {
    "patient_month": (("HOSPITALIZED", "higher_is_worse", "a hospital admission that month"),
                      ("AVF", "higher_is_better", "an arteriovenous fistula in use"),
                      ("TRANSFUSED", "higher_is_worse", "a transfusion that month"),
                      ("CATHETER_90", "higher_is_worse", "a catheter in use for 90 days or more")),
    "index_discharge": (("READMIT30", "higher_is_worse", "an unplanned readmission within 30 days"),
                        ("ED30", "higher_is_worse", "an emergency visit within 30 days"),
                        ("UNPLANNED_READMIT", "higher_is_worse", "an unplanned readmission")),
    "binomial": (("HOSP_DAYS", "higher_is_worse", "days spent in hospital"),
                 ("DAYS_OUT", "higher_is_better", "days alive and out of hospital")),
}
TRIALS = ("DAYS_AT_RISK", "ELIGIBLE_DAYS", "N_DAYS")
COVARIATES = ("AGE", "FEMALE", "DIABETES", "CHF", "COPD", "BMI", "YEARS_ON_DIALYSIS", "NURSING_HOME", "ALBUMIN",
              "HEMOGLOBIN", "RACE_BLACK", "DUAL_ELIGIBLE", "CANCER", "STROKE", "PVD", "DEMENTIA")
GHOSTS = ("ALBUMIN_LAB", "KTV", "FRAILTY", "INCOME", "HISPANIC")
EXTRA = {"patient_month": ("MONTH", "YEAR"), "index_discharge": ("DISCHARGE_ID", "ADMIT_MONTH", "HOSPITAL"),
         "binomial": ("MONTH", "YEAR")}
FILLER = ("This is for the 2025 annual report.", "Thanks in advance.", "The file came from the KECC extract.",
          "Let me know if anything is unclear.", "We ran a similar analysis last year.", "Data are on the shared drive.")


def written(col: str, rng: random.Random) -> str:
    """A column as an analyst types it: usually as is, sometimes lower case, sometimes with spaces."""
    r = rng.random()
    if r < 0.15:
        return col.lower()
    if r < 0.25 and "_" in col:
        return col.replace("_", " ")
    return col


def join(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def phrases(field: str, value: Any, ctx: Dict[str, Any], rng: random.Random) -> str:
    """One phrase that states the field; it is also the evidence, so it is copied into the message verbatim."""
    w = lambda c: written(c, rng)  # noqa: E731
    if field == "outcome":
        o = w(value)
        if ctx.get("trials_stated"):
            return f"{o} out of {w(ctx['trials'])}"
        return rng.choice([f"the outcome is {o}", f"{o} is the event", f"outcome column {o}",
                           f"we model {o} (1 means {ctx['event_desc']})"])
    if field == "event_direction":
        if value == "higher_is_worse":
            return rng.choice(["a higher rate is worse", "more events mean worse care", "higher is worse",
                               "the event is a bad outcome"])
        return rng.choice(["a higher rate is better", "higher is better", "more of it means better care"])
    if field == "unit":
        return {"patient_month": rng.choice(["each row is a patient-month", "one row per patient per month",
                                             "rows are patient months"]),
                "index_discharge": rng.choice(["each row is an index discharge", "one row per index discharge"]),
                "patient_year": "each row is a patient-year"}[value]
    if field == "provider":
        p = w(value)
        return rng.choice([f"providers are identified by {p}", f"profile each {p}", f"the facility column is {p}",
                           f"by {p}"])
    if field == "patient":
        if value == "none":
            return rng.choice(["there is no patient identifier", "no patient ID in this file"])
        p = w(value)
        return rng.choice([f"patients are identified by {p}", f"{p} is the patient ID", f"patient column {p}"])
    if field == "trials":
        return f"{w(ctx['outcome'])} out of {w(value)}"
    if field == "adjusters":
        names = [w(a) for a in ctx["adjusters_written"]]
        return rng.choice([f"adjust for {join(names)}", f"risk-adjust on {join(names)}",
                           f"covariates: {', '.join(names)}"])
    if field == "eligibility":
        return rng.choice([f"facilities with fewer than {value} patients are excluded", f"at least {value} patients",
                           f"a minimum of {value} patients per facility"])
    if field == "standardization":
        return {"indirect": rng.choice(["an indirectly standardized ratio", "observed over expected",
                                        "indirect standardization"]),
                "direct": rng.choice(["a directly standardized rate", "direct standardization"]),
                "both": "report both indirect and direct standardization"}[value]
    if field == "norm":
        return {"median": rng.choice(["the median facility as the norm", "compared with the national median"]),
                "size_weighted_mean": rng.choice(["the size-weighted national mean as the norm",
                                                  "compared with the national average weighted by size"])}[value]
    if field == "reference":
        return {"empirical": rng.choice(["use an empirical null", "calibrate the z-scores with an empirical null"]),
                "theoretical": rng.choice(["the theoretical null", "a standard normal reference"]),
                "both": "show both the theoretical and the empirical null"}[value]
    if field == "consequential_tail":
        return {"worse": rng.choice(["only worse-than-expected flags have consequences",
                                     "the worse tail is what matters"]),
                "better": "the better tail is what matters",
                "both": rng.choice(["both directions matter", "both tails have consequences"])}[value]
    raise KeyError(field)


# descriptions and hedges: sentences that look like declarations but state no field, so nothing is extracted
NON_STATEMENTS = {
    "adjusters": ("adjust for the usual demographics", "use the standard risk factors"),
    "provider": ("one profile per dialysis unit",),
    "standardization": ("maybe the direct rate, I am not sure yet",),
    "reference": ("we have not decided on the null yet",),
    "eligibility": ("small facilities may need a threshold, to be decided",),
}
PREFERENCES = {"all": "please keep all of these in the model even if some look unimportant",
               "small": "we would rather keep the model small"}


def make_task(i: int, rng: random.Random) -> Optional[Dict[str, Any]]:
    unit_kind = rng.choice(("patient_month", "index_discharge", "binomial"))
    outcome, direction, event_desc = rng.choice(OUTCOMES[unit_kind])
    provider, patient = rng.choice(PROVIDER), rng.choice(PATIENT)
    trials = rng.choice(TRIALS) if unit_kind == "binomial" else None
    covs = rng.sample(COVARIATES, rng.randint(5, 14))
    columns = [provider, patient, outcome] + ([trials] if trials else []) + covs + list(EXTRA[unit_kind])
    rng.shuffle(columns)
    profile = {"columns": [{"name": c} for c in columns]}

    gold: Dict[str, Any] = {}
    stated = lambda p: rng.random() < p  # noqa: E731
    if stated(0.95):
        gold["outcome"] = outcome
    if stated(0.85):
        gold["provider"] = provider
    if stated(0.7):
        gold["patient"] = patient if stated(0.92) else "none"
    if stated(0.7):
        gold["event_direction"] = direction
    if stated(0.6):
        gold["unit"] = "patient_month" if unit_kind == "binomial" else unit_kind
    ctx: Dict[str, Any] = {"outcome": outcome, "trials": trials, "event_desc": event_desc}
    if trials and "outcome" in gold and stated(0.8):
        gold["trials"] = trials
        ctx["trials_stated"] = True
    ghost = None
    if stated(0.7):
        adj = rng.sample(covs, rng.randint(1, min(8, len(covs))))
        written_adj = list(adj)
        if stated(0.2):
            ghost = rng.choice(GHOSTS)
            written_adj.insert(rng.randint(0, len(written_adj)), ghost)
        gold["adjusters"] = adj
        ctx["adjusters_written"] = written_adj
    if stated(0.5):
        gold["eligibility"] = rng.choice((3, 5, 10, 11, 20, 26))
    for f, vals, p in (("standardization", ("indirect", "direct", "both"), 0.45), ("norm", ("median", "size_weighted_mean"), 0.3),
                       ("reference", ("empirical", "theoretical", "both"), 0.35), ("consequential_tail", ("worse", "both", "better"), 0.35)):
        if stated(p):
            gold[f] = rng.choice(vals)
    # a first message is often short ("profile READMIT30 by FACILITY"), and sometimes states nothing at all
    r = rng.random()
    if r < 0.03:
        gold = {}
    elif r < 0.25:
        keep = set(rng.sample(sorted(gold), min(len(gold), rng.randint(1, 3))))
        if "trials" in keep and "outcome" not in keep:
            keep.add("outcome")
        if "outcome" in keep and "trials" in gold and ctx.get("trials_stated"):
            keep.add("trials")                # "X out of Y" states both or neither
        gold = {f: v for f, v in gold.items() if f in keep}
        ctx["trials_stated"] = "trials" in gold

    pieces: List[Tuple[Optional[str], str]] = []
    for f, v in gold.items():
        if f == "trials":
            continue                      # stated by the outcome's "out of" phrase
        pieces.append((f, phrases(f, v, ctx, rng)))
    open_fields = [f for f in NON_STATEMENTS if f not in gold]
    for f in rng.sample(open_fields, min(len(open_fields), rng.choice((0, 0, 1, 2)))):
        pieces.append((None, rng.choice(NON_STATEMENTS[f])))
    preference = None
    if "adjusters" in gold and stated(0.25):
        preference = rng.choice(tuple(PREFERENCES))
        pieces.append((None, PREFERENCES[preference]))
    if not pieces:
        pieces.append((None, rng.choice(("please run the usual profiling on this file",
                                         "could you profile the facilities in the attached data"))))
    rng.shuffle(pieces)
    style = rng.choice(("prose", "bullets", "terse", "note"))
    texts = [t for _, t in pieces]
    if style == "prose":
        message = " ".join(t[0].upper() + t[1:] + "." for t in texts)
    elif style == "bullets":
        message = "\n".join("- " + t for t in texts)
    elif style == "terse":
        message = "; ".join(texts)
    else:
        fill = rng.sample(FILLER, 2)
        message = fill[0] + " " + " ".join(t[0].upper() + t[1:] + "." for t in texts) + " " + fill[1]

    evidence = {f: t for f, t in pieces if f}
    if "trials" in gold:
        evidence["trials"] = evidence["outcome"]
    order = sorted(gold, key=lambda f: message.lower().find(evidence[f].lower()))
    fields = [{"field": f, "value": ", ".join(gold[f]) if f == "adjusters" else str(gold[f]), "evidence": evidence[f]}
              for f in order]
    reasoning = "; ".join(f'"{evidence[f]}" names {f}' for f in order) if order else "The message names no field."
    answer = {"reasoning": reasoning, "fields": fields}

    # the gold answer must pass the harness's verifier unchanged (a ghost adjuster is in the message, never in
    # the answer, so it raises no note either)
    spec, notes = boundary.verify_extraction(answer, message, profile, MeasureSpecification())
    if notes:
        return None
    for f, v in gold.items():
        if getattr(spec, f) != v:
            return None

    return {"id": f"spec_{i:05d}", "act": "spec_extraction",
            "messages": [{"role": "system", "content": policy.load("spec_extraction")},
                         {"role": "user", "content": boundary.spec_extraction_user(message, profile)},
                         {"role": "assistant", "content": json.dumps(answer)}],
            "gold": {k: v for k, v in gold.items()},
            "meta": {"unit_kind": unit_kind, "style": style, "n_stated": len(gold), "ghost": ghost,
                     "preference": preference, "open_statements": [t for f, t in pieces if f is None and t not in PREFERENCES.values()],
                     "family": f"{unit_kind}|{style}"}}


def score(gold: Dict[str, Any], spec: MeasureSpecification) -> Dict[str, float]:
    """The reward of an extraction after the harness's verifier: F1 over fields, where a field counts as right
    only with the gold value, and a field set that the message does not state counts against precision."""
    got = {f: getattr(spec, f) for f in boundary.SPEC_FIELDS if getattr(spec, f) is not None}
    right = sum(1 for f, v in got.items() if gold.get(f) == v)
    prec = right / len(got) if got else (1.0 if not gold else 0.0)
    rec = right / len(gold) if gold else 1.0
    return {"precision": prec, "recall": rec, "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    tasks, tries = [], 0
    while len(tasks) < a.n and tries < 5 * a.n:
        tries += 1
        t = make_task(len(tasks) + 1, rng)
        if t is not None:
            tasks.append(t)
    with open(a.out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    styles: Dict[str, int] = {}
    for t in tasks:
        styles[t["meta"]["family"]] = styles.get(t["meta"]["family"], 0) + 1
    print(f"{len(tasks)} tasks ({tries - len(tasks)} drafts rejected by the verifier) -> {a.out}")
    for k, v in sorted(styles.items()):
        print(f"  {k:<28} {v}")
    print(f"  with a ghost adjuster: {sum(1 for t in tasks if t['meta']['ghost'])}; "
          f"with a preference: {sum(1 for t in tasks if t['meta']['preference'])}; "
          f"stating nothing beyond one field: {sum(1 for t in tasks if t['meta']['n_stated'] <= 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
