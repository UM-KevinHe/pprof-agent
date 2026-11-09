from pprof_agent import specification as S


def col(name, type_="numeric", binary=False, nd=50, id_like=False):
    return {"name": name, "type": type_, "binary": binary, "n_distinct": nd, "constant": False, "id_like": id_like,
            "n_missing": 0, "rows_per_level": 1000 / max(nd, 1)}


PROFILE = {
    "rows": 1000,
    "columns": [col("PROVFS", "character", nd=40, id_like=True), col("KECC_ID", "character", nd=300, id_like=True),
                col("AVF", "integer", binary=True, nd=2), col("MONTH", "integer", nd=12), col("AGE", "integer", nd=60),
                col("FEMALE", "integer", binary=True, nd=2), col("HGB", "numeric", nd=200)],
    "candidates": {"outcome": ["AVF", "FEMALE"], "provider": ["PROVFS"], "patient": ["KECC_ID"], "time": ["MONTH"],
                   "covariates": ["AGE", "FEMALE", "HGB"], "trials": [], "hospital": [], "categorical": []},
}


def test_complete_infers_only_single_candidates():
    spec, items = S.complete(S.MeasureSpecification(), PROFILE)
    assert spec.provider == "PROVFS" and spec.sources["provider"] == "inferred"
    assert spec.patient == "KECC_ID" and spec.sources["patient"] == "inferred"
    assert spec.outcome is None                      # two binary columns: asked, never guessed
    assert {i.field for i in items} == {"outcome", "event_direction", "unit", "adjusters"}
    assert spec.standardization == "both" and spec.sources["standardization"] == "default"
    assert [i.number for i in items] == sorted(i.number for i in items)


def test_parse_reply_settles_and_completes():
    spec, items = S.complete(S.MeasureSpecification(), PROFILE)
    spec2, problems = S.parse_reply("1) avf 2) higher is better 3) month 6) B minus HGB", spec, PROFILE)
    assert problems == []
    assert spec2.outcome == "AVF" and spec2.sources["outcome"] == "asked"
    assert spec2.event_direction == "higher_is_better"
    assert spec2.unit == "patient_month"
    assert spec2.adjusters == ["AGE", "FEMALE"]
    spec3, items3 = S.complete(spec2, PROFILE)
    assert items3 == []
    assert spec3.eligibility == 2 and spec3.sources["eligibility"] == "default"
    assert spec3.outcome_type == "binary"
    assert spec3.missing() == []


def test_parse_reply_reports_problems():
    spec, _ = S.complete(S.MeasureSpecification(), PROFILE)
    _, problems = S.parse_reply("6) AGE, NOPE 9) nonsense 13) x", spec, PROFILE)
    assert any("NOPE" in p for p in problems)
    assert any("9)" in p for p in problems)
    assert any("no item 13" in p for p in problems)
    _, problems = S.parse_reply("just prose", spec, PROFILE)
    assert problems and "numbered" in problems[0]


def test_parse_reply_explicit_adjusters_and_none_patient():
    spec, _ = S.complete(S.MeasureSpecification(), PROFILE)
    spec2, problems = S.parse_reply("5) none 6) AGE HGB", spec, PROFILE)
    assert problems == []
    assert spec2.patient == "none" and spec2.adjusters == ["AGE", "HGB"]


def test_render_question_lists_open_items():
    _, items = S.complete(S.MeasureSpecification(), PROFILE)
    text = S.render_question(items)
    for n in ("1)", "2)", "3)", "6)"):
        assert n in text
    assert "higher_is_worse" in text


def card(**kw):
    base = {"rows": 1000, "rows_complete": 1000, "patients": 300, "providers": 40, "eligible_providers": 38,
            "excluded_providers": 2, "eligibility": 2, "events": 300, "trials_total": 1000, "event_rate": 0.3,
            "rows_per_patient": {"median": 3, "max": 12}, "max_rows_per_patient": 12, "repeats": True,
            "has_patient": True, "patients_per_provider": {"min": 2, "median": 8, "max": 30},
            "rows_per_provider": {"max": 200}, "invariant_providers": {"none": 1, "all": 0, "total": 1},
            "small_providers": {"le_3": 3, "le_10": 10}, "problems": [], "adjuster_problems": []}
    base.update(kw)
    return base


def full_spec(**kw):
    d = dict(outcome="AVF", outcome_type="binary", event_direction="higher_is_better", unit="patient_month",
             provider="PROVFS", patient="KECC_ID", eligibility=2, adjusters=["AGE", "HGB"], standardization="both",
             norm="median", reference="both", consequential_tail="worse", level=0.95)
    d.update(kw)
    return S.MeasureSpecification.from_dict(d)


def test_verify_accepts_and_warns():
    v = S.verify(full_spec(), card())
    assert v.ok and v.refusals == []
    assert any("invariant" in w for w in v.warnings)


def test_verify_refuses():
    assert not S.verify(full_spec(crossed_with="HOSP"), card()).ok
    assert not S.verify(full_spec(patient="none"), card(has_patient=False, repeats=False)).ok
    v = S.verify(full_spec(), card(problems=[{"field": "outcome", "problem": "constant"}]))
    assert not v.ok and "constant" in v.refusals[0]
    v = S.verify(full_spec(), card(adjuster_problems=[{"name": "HGB", "problem": "not numeric"}]))
    assert not v.ok and "HGB" in v.refusals[0]
    assert not S.verify(full_spec(), card(eligible_providers=1)).ok
    assert not S.verify(full_spec(adjusters=None), card()).ok


def test_config_excludes_provenance():
    sp = full_spec().note("a note")
    assert "sources" not in sp.config() and "notes" not in sp.config()
    assert sp.config()["adjusters"] == ["AGE", "HGB"]


def test_complete_settles_trials_for_a_count_outcome():
    prof = {"columns": [{"name": "PROVFS", "binary": False}, {"name": "KECC_ID", "binary": False}, {"name": "HOSP_DAYS", "binary": False},
                        {"name": "DAYS_AT_RISK", "binary": False}, {"name": "AGE", "binary": False}, {"name": "FEMALE", "binary": True}],
            "candidates": {"outcome": ["FEMALE"], "trials": ["DAYS_AT_RISK"], "provider": ["PROVFS"], "patient": ["KECC_ID", "PROVFS"],
                           "time": [], "covariates": ["AGE", "FEMALE"]}}
    sp, items = S.complete(S.MeasureSpecification.from_dict({"outcome": "HOSP_DAYS", "unit": "patient_month", "event_direction": "higher_is_worse",
                                                             "adjusters": ["AGE"]}), prof)
    assert sp.trials == "DAYS_AT_RISK" and sp.sources["trials"] == "inferred" and sp.outcome_type == "binomial"
    assert not any(it.field == "trials" for it in items)
    prof["candidates"]["trials"] = ["DAYS_AT_RISK", "AGE"]
    sp, items = S.complete(S.MeasureSpecification.from_dict({"outcome": "HOSP_DAYS", "unit": "patient_month", "event_direction": "higher_is_worse",
                                                             "adjusters": ["AGE"]}), prof)
    assert sp.trials is None and any(it.field == "trials" and it.number == 12 for it in items)
    sp, items = S.complete(S.MeasureSpecification.from_dict({"outcome": "FEMALE", "unit": "patient_month", "event_direction": "higher_is_worse",
                                                             "adjusters": ["AGE"]}), prof)
    assert sp.trials is None and sp.outcome_type == "binary"


def test_reply_sets_trials_by_number():
    prof = {"columns": [{"name": "HOSP_DAYS", "binary": False}, {"name": "DAYS_AT_RISK", "binary": False}], "candidates": {}}
    sp, probs = S.parse_reply("12) DAYS_AT_RISK", S.MeasureSpecification(outcome="HOSP_DAYS"), prof)
    assert not probs and sp.trials == "DAYS_AT_RISK" and sp.sources["trials"] == "asked"


def test_phrases_fill_unit_direction_and_eligibility_the_model_left_open():
    msg = ("This file is the AOH measure. The outcome is AOH_DAYS out of DAYS_AT_RISK, one row per patient-month, "
           "and a higher rate is better. FACILITY identifies the facility. Leave out facilities with fewer than 11 patients.")
    found = dict((f, (v, e)) for f, v, e in S.read_phrases(msg))
    assert found["unit"][0] == "patient_month" and "row per patient-month" in found["unit"][1]
    assert found["event_direction"][0] == "higher_is_better"
    assert found["eligibility"][0] == 11
    sp, notes = S.fill_from_phrases(S.MeasureSpecification(outcome="AOH_DAYS", event_direction="higher_is_worse"), msg)
    assert sp.unit == "patient_month" and sp.sources["unit"] == "quoted"
    assert sp.event_direction == "higher_is_worse"            # a field the model settled is never overwritten
    assert sp.eligibility == 11 and len(notes) == 2
    assert S.read_phrases("Which facilities are flagged worse than expected?") == []
    assert dict((f, v) for f, v, _ in S.read_phrases("each row is an index discharge; a lower rate is better"))["unit"] == "index_discharge"
    assert dict((f, v) for f, v, _ in S.read_phrases("a lower rate is better"))["event_direction"] == "higher_is_worse"
