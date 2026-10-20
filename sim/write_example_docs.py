"""Write the documents that accompany each example data set: the data dictionary (in the register of a
measure's technical specification), the published measure specification the agent can pre-fill from, the
scripted scenarios of the demo, and a README. Reads only truth.json; never the data.

    python sim/write_example_docs.py <examples dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = {
    "PROVFS": "Facility identifier (synthetic; the six-character CMS certification number in production).",
    "KECC_ID": "Patient identifier (synthetic). A patient contributes one row per unit of observation.",
    "AVF": "1 if the patient dialyzed through an arteriovenous fistula in the month, 0 otherwise. The outcome; a higher rate is better.",
    "READMIT30": "1 if an unplanned readmission followed the index discharge within 30 days, 0 otherwise. The outcome; a higher rate is worse.",
    "HOSP_DAYS": "Days spent in hospital during the month. The outcome count; a higher rate is worse.",
    "DAYS_AT_RISK": "Days of the month the patient was at risk (alive, on dialysis, not transplanted). The trials of the binomial outcome.",
    "AGE": "Age in years at the start of the month (or at the index admission).",
    "FEMALE": "1 if female.",
    "DIABETES": "1 if diabetes is recorded as the cause of ESRD or a comorbidity.",
    "CHF": "1 if congestive heart failure is recorded.",
    "COPD": "1 if chronic obstructive pulmonary disease is recorded.",
    "BMI": "Body mass index at the start of dialysis.",
    "YRS_ESRD": "Years since the start of ESRD (dialysis vintage).",
    "NH": "1 if the patient resides in a nursing home.",
    "ALB": "Serum albumin (g/dL), the most recent value in the month.",
    "HGB": "Hemoglobin (g/dL), the most recent value in the month.",
    "LOS": "Length of the index hospital stay in days.",
    "PRIOR_ADMITS": "Hospital admissions in the year before the index admission.",
    "DX_CARDIAC": "1 if the principal diagnosis of the index stay was cardiovascular.",
    "DX_INFECTION": "1 if the principal diagnosis of the index stay was an infection.",
    "DX_ACCESS": "1 if the principal diagnosis of the index stay concerned the vascular access.",
    "HOSPITAL": "Discharging hospital (synthetic identifier). The measure is attributed to the dialysis facility; the hospital is a second, crossed provider that the current implementation does not adjust for.",
    "DISCHARGE_ID": "Identifier of the index discharge.",
    "ADMIT_MONTH": "Calendar month of the index admission.",
    "MONTH": "Calendar month of the patient-month within the year.",
    "YEAR": "Reporting year (constant within a file).",
    "FACILITY": "Dialysis facility identifier (synthetic).",
    "PATIENT": "Patient identifier (synthetic). A patient contributes one row per month on dialysis.",
    "HOSPITALIZED": "1 if the patient was admitted to hospital during the month, 0 otherwise. The outcome; a higher rate is worse.",
    "HEART_FAILURE": "1 if congestive heart failure is recorded.",
    "YEARS_ON_DIALYSIS": "Years since the start of dialysis.",
    "NURSING_HOME": "1 if the patient resides in a nursing home.",
    "ALBUMIN": "Serum albumin (g/dL), the most recent value in the month.",
    "HEMOGLOBIN": "Hemoglobin (g/dL), the most recent value in the month.",
}

MEASURES = {
    "sfr": dict(title="Standardized Fistula Rate (SFR), synthetic", outcome="AVF", unit="patient_month", direction="higher_is_better",
                standardization="direct", eligibility=11, noun="patient-month", higher="better", figure="fistula use"),
    "srr": dict(title="Standardized Readmission Ratio (SRR), synthetic, one-stage", outcome="READMIT30", unit="index_discharge",
                direction="higher_is_worse", standardization="indirect", eligibility=11, noun="index discharge", higher="worse", figure="readmission"),
    "aoh": dict(title="Standardized Hospital Days (AOH-like), synthetic", outcome="HOSP_DAYS", unit="patient_month", direction="higher_is_worse",
                standardization="direct", eligibility=11, noun="patient-month", higher="worse", figure="hospital days"),
    "demo": dict(title="Monthly hospitalization among dialysis patients", outcome="HOSPITALIZED", unit="patient_month", direction="higher_is_worse",
                 standardization="indirect", eligibility=11, noun="patient-month", higher="worse", figure="hospitalization"),
}


def kind_of(name: str) -> str:
    return name.split("_")[0]


def dictionary(name: str, t: dict) -> str:
    m = MEASURES[kind_of(name)]
    cols = [t["provider"], t["patient"], t["outcome"]] + ([t["trials"]] if t.get("trials") else []) + list(t["covariates"])
    cols += ["HOSPITAL", "DISCHARGE_ID", "ADMIT_MONTH"] if t["kind"] == "discharge" else ["MONTH"]
    cols += ["YEAR"]
    lines = [f"# {m['title']} — data dictionary ({name})", "",
             f"One row is one {m['noun']}. Rows are attributed to the dialysis facility (`{t['provider']}`) where the patient was treated. "
             f"The outcome is `{t['outcome']}`" + (f" out of `{t['trials']}` trials" if t.get("trials") else "") +
             f"; a higher rate is {m['higher']}. The risk adjusters are the committee set listed in `specification.json`.", "",
             "**This file is synthetic.** It was generated by `Agent/sim/make_examples.R` from a model with known facility effects, "
             "patient-level dependence and planted departures; `truth.json` records them. No real patient or facility data entered it.", "",
             "| column | meaning |", "|---|---|"]
    for c in cols:
        lines.append(f"| `{c}` | {COLUMNS.get(c, '')} |")
    lines += ["", "## What is planted (from `truth.json`)", "",
              f"- Facilities truly worse than the norm: {', '.join(t['worse'])} (two per size tertile; effects {t['shift_by_size_class']} on the logit scale, larger where the facility is smaller).",
              f"- Facilities truly better: {', '.join(t['better'])}.",
              f"- Facilities whose outcomes are all zero (invariant): {', '.join(t['invariant'])}.",
              f"- Facilities with three patients or fewer: {len(t['tiny'])}.",
              f"- A decoy with an extreme case mix and a true effect of zero: {t['decoy']}.",
              f"- A patient random effect of scale {t['sigma_patient']} (within-patient correlation); ordinary facilities have a small spread of effects (sd {t['sd_gamma_ordinary']})."]
    if t.get("year2_changes"):
        y = t["year2_changes"]
        lines.append(f"- Changes from the previous year: {y['truly_worse']} became truly worse; {y['casemix_worse']}'s case mix deteriorated ({y['casemix_change']}) with its effect unchanged.")
    if t.get("hospitals"):
        h = t["hospitals"]
        lines.append(f"- Hospital effects in the generating model: {h['n']} hospitals, two of them ({', '.join(h['bad'])}) with a large excess; facilities whose home hospital is one of them: {', '.join(h['facilities_with_a_bad_home_hospital'])}.")
    return "\n".join(lines) + "\n"


def specification(name: str, t: dict) -> dict:
    m = MEASURES[kind_of(name)]
    spec = {"outcome": t["outcome"], "event_direction": m["direction"], "unit": m["unit"], "provider": t["provider"], "patient": t["patient"],
            "eligibility": m["eligibility"], "adjusters": list(t["covariates"]), "standardization": m["standardization"], "norm": "median",
            "consequential_tail": "worse"}
    if t.get("trials"):
        spec["trials"] = t["trials"]
    return spec


def scenarios(name: str, t: dict) -> dict:
    m = MEASURES[kind_of(name)]
    adj = list(t["covariates"])
    adj_words = ", ".join(adj[:-1]) + " and " + adj[-1]
    noun = m["noun"]
    trials = f" out of {t['trials']}" if t.get("trials") else ""
    declare = (f"The outcome is {t['outcome']}{trials}, one row per {noun}, and a higher rate is {m['higher']}. "
               f"{t['provider']} identifies the facility and {t['patient']} the patient. Facilities with fewer than {m['eligibility']} patients are excluded. "
               f"Adjust for {adj_words}.")
    small_worse = [p for p, c in t["worse_size_class"].items() if c == "large"][:1] or t["worse"][:1]   # the facility the demo asks about: a large one, reliably flagged
    large_worse = [p for p, c in t["worse_size_class"].items() if c == "large"][:1] or t["worse"][-1:]
    s = {}
    s["demo"] = [f"# The full conversation: declare the measure in a sentence, ask about the flags, a facility, an alternative, the test; draw figures", declare,
                 "Which facilities are flagged worse than expected, and how sure can I be about them?",
                 f"Why is facility {small_worse[0]} flagged worse?",
                 "What would change if we used the theoretical null instead?",
                 "Which test is reported, and why not the robust score test?",
                 "Draw the funnel plot and the observed-against-expected plot."]
    if t["kind"] == "discharge":
        s["demo"].insert(3, "Can you adjust for the discharging hospital as well?")
    s["ghost_column"] = [f"# A declaration that names a column the file does not have", declare.replace(f" and {adj[-1]}.", f", {adj[-1]} and HEMOGLOBIN_LAB.")]
    s["edit_after"] = ["# Changing the eligibility threshold after the analysis: the agent re-runs it", declare,
                       "Use an eligibility threshold of 20 patients instead."]
    s["explain_large"] = ["# A large flagged facility, and the direct standardization", declare,
                          f"Why is facility {large_worse[0]} flagged worse?", "What would change under direct standardization?"]
    s["decoy"] = ["# A decoy: much sicker patients, but no true departure", declare,
                  f"Is facility {t['decoy']} flagged? Its patients look much sicker than elsewhere."]
    if t.get("year2_changes"):
        y = t["year2_changes"]
        s["year_over_year"] = ["# (needs the two-year operation, not built yet) why is this facility flagged this year?", declare,
                               f"Why is facility {y['casemix_worse']} flagged this year when it was not last year?",
                               f"And facility {y['truly_worse']}?"]
    return s


def readme(names: list) -> str:
    if names == ["demo"]:
        return "\n".join([
            "# The demo data set", "",
            "`demo/` is one synthetic measure file shaped like a CMS dialysis quality measure: one row per patient-month, the outcome "
            "`HOSPITALIZED`, a facility and a patient identifier, ten risk adjusters. It was generated by `sim/make_examples.R` from a "
            "model with known facility effects (`truth.json`: six facilities truly worse, four truly better, two with no events, a decoy "
            "with a much sicker case mix and no true departure). **Nothing here derives from real data.**", "",
            "`dictionary.md` describes the columns, `specification.json` is a published specification the agent can pre-fill from, "
            "`scenarios/*.txt` are the scripted conversations (one message per line), and `runs/<scenario>/` the recorded runs with the "
            "local model that the web app replays.", ""])
    return "\n".join([
        "# Example data sets", "",
        "Synthetic measure files shaped like the CMS dialysis measures, with planted truth, for the end-to-end tests, the evaluation "
        "and the public demo. Generated by `sim/make_examples.R`; documents by `sim/write_example_docs.py`. **Nothing here derives from real data.**", "",
        "| folder | shape | outcome | use |", "|---|---|---|---|",
        "| `sfr_2024` | patient-month | AVF, higher is better | the previous year of `sfr_2025` (the year-over-year scenario) |",
        "| `sfr_2025` | patient-month | AVF, higher is better | the main demo: full run, both standardizations, empirical null, calibration |",
        "| `srr_2025` | index discharge | READMIT30, higher is worse | the worse tail; the hospital column the gate refuses |",
        "| `aoh_2025` | patient-month, binomial | HOSP_DAYS of DAYS_AT_RISK, higher is worse | the trials-column path |", "",
        "Each folder holds `data.csv`, `dictionary.md`, `specification.json` (the published specification the agent pre-fills from, "
        "`python -m pprof_agent.cli run --spec`), `truth.json`, `scenarios/*.txt` (one message per line for `python -m pprof_agent.chat --script`), "
        "and, once run, `runs/<scenario>/` with the tables, figures, report and trace of a run with the local model.", ""])


def main(root: str) -> None:
    rootp = Path(root)
    names = []
    for d in sorted(rootp.iterdir()):
        if not (d / "truth.json").exists():
            continue
        t = json.load(open(d / "truth.json", encoding="utf-8"))
        names.append(d.name)
        (d / "dictionary.md").write_text(dictionary(d.name, t), encoding="utf-8")
        (d / "specification.json").write_text(json.dumps(specification(d.name, t), indent=1), encoding="utf-8")
        sc = d / "scenarios"
        sc.mkdir(exist_ok=True)
        for k, lines in scenarios(d.name, t).items():
            (sc / f"{k}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(d.name, "documents written:", ", ".join(scenarios(d.name, t).keys()))
    (rootp / "README.md").write_text(readme(names), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
