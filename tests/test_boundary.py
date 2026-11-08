"""The model boundaries with the fake client: typing, extraction and its checks, the planner's proposal
shape, figure selection against the session, prose checks, wording checks."""
import json

from pprof_agent import actions, boundary, rules, policy
from pprof_agent.fake import FakeClient
from pprof_agent.specification import MeasureSpecification, Verification
from pprof_agent.state import Result, Session

COLS = ["PROVFS", "KECC_ID", "AVF", "AGE", "FEMALE", "DIABETES", "HGB", "MONTH"]
PROFILE = {"rows": 100, "columns": [{"name": c, "type": "integer", "binary": c in ("AVF", "FEMALE", "DIABETES"),
                                      "n_distinct": 5, "constant": False, "id_like": c.endswith("ID") or c == "PROVFS"} for c in COLS],
           "candidates": {"outcome": ["AVF", "FEMALE", "DIABETES"], "provider": ["PROVFS"], "patient": ["KECC_ID"],
                          "time": ["MONTH"], "covariates": ["AGE", "FEMALE", "DIABETES", "HGB"]}}
CARD = {"invariant_providers": {"total": 0}, "repeats": True, "has_patient": True, "patients_per_provider": {"median": 40, "max": 300},
        "rows_per_provider": {"max": 3000}, "eligible_providers": 200, "max_rows_per_patient": 12, "small_providers": {"le_3": 2},
        "rows_complete": 1000, "patients": 300, "providers": 200, "event_rate": 0.3, "eligibility": 2}


def test_mentions_does_not_depend_on_set_order():
    from pprof_agent import textmatch
    for cols in (["PID", "AGE"], ["AGE", "PID"]):
        assert textmatch.names_mentioned("Patient column pid", cols + ["pid"]) == ["pid"]
        assert textmatch.mentions("Patient column pid", "pid", cols)
        assert textmatch.mentions("Patient column pid", "PID", cols)


def test_policies_have_five_headings():
    for n in policy.NAMES:
        secs = policy.sections(n)
        for h in policy.HEADINGS:
            assert h in secs, (n, h)
    assert policy.describe()["next_step"]["sha256"].startswith("sha256:")


def test_classify_verifies_evidence_and_phase():
    c = FakeClient()
    out = boundary.classify(c, "fake", "The outcome is AVF and PROVFS identifies the facility.", "phase: DECLARING | result: none")
    assert out[0]["kind"] == "declare_specification" and out[0]["evidence_verified"]
    out = boundary.classify(c, "fake", "Why is facility A flagged?", "phase: DECLARING | result: none")
    assert out[0]["kind"] == "other"              # a result question before a result is demoted
    out = boundary.classify(c, "fake", "Why is facility A flagged?", "phase: DONE | result: present")
    assert out[0]["kind"] == "ask_about_result"
    c2 = FakeClient({"intent": lambda u: {"reasoning": "", "intents": [{"kind": "out_of_scope", "evidence": "not in the message", "out_of_scope_reason": "other"}]}})
    out = boundary.classify(c2, "fake", "hello", "phase: DECLARING")
    assert out[0]["kind"] == "other"              # out_of_scope without evidence in the message is demoted


def test_extraction_is_checked_against_message_and_file():
    c = FakeClient()
    msg = "The outcome is AVF, one row per patient-month, and a higher rate is better. PROVFS identifies the facility and KECC_ID the patient. Adjust for AGE, FEMALE and HGB."
    spec, notes = boundary.extract_specification(c, "fake", msg, PROFILE, MeasureSpecification())
    assert spec.outcome == "AVF" and spec.sources["outcome"] == "quoted"
    assert spec.provider == "PROVFS" and spec.patient == "KECC_ID"
    assert spec.unit == "patient_month" and spec.event_direction == "higher_is_better"
    assert spec.adjusters == ["AGE", "FEMALE", "HGB"]
    assert notes == []
    # a ghost column named by the model but not in the file; a column not written in the message; bad evidence
    c2 = FakeClient({"spec_extraction": lambda u: {"reasoning": "", "fields": [
        {"field": "outcome", "value": "AVF2", "evidence": "The outcome is AVF2"},
        {"field": "provider", "value": "PROVFS", "evidence": "PROVFS identifies the facility"},
        {"field": "patient", "value": "KECC_ID", "evidence": "this phrase is not in the message"},
        {"field": "event_direction", "value": "sideways", "evidence": "The outcome"}]}})
    spec, notes = boundary.extract_specification(c2, "fake", "The outcome is AVF2 and PROVFS identifies the facility", PROFILE, MeasureSpecification())
    assert spec.outcome is None and any("AVF2" in n and "not a column" in n for n in notes)
    assert spec.provider == "PROVFS"
    assert spec.patient is None and any("patient" in n for n in notes)
    assert spec.event_direction is None


def make_session():
    s = Session(data_path="x.csv", run_dir="/tmp/x")
    s.profile = PROFILE
    s.spec = MeasureSpecification.from_dict(dict(outcome="AVF", provider="PROVFS", patient="KECC_ID", adjusters=["AGE", "HGB"],
                                                 event_direction="higher_is_better", unit="patient_month", eligibility=2,
                                                 standardization="both", norm="median", reference="both", consequential_tail="worse", level=0.95))
    s.verification = Verification(True, [], [], CARD)
    s.admissible = rules.admissible(CARD, s.spec)
    return s


def test_next_step_proposal_shape_and_filtering():
    from pprof_agent import loop
    s = make_session()
    obs = loop.observe(s)
    text = boundary.render_observation(obs, "note")
    assert "Protected core still to run" in text and "fit" in text
    prop = boundary.next_step(FakeClient(), "fake", obs)
    assert prop["action"] == "fit" and prop["args"] == {}       # null args dropped, only the card's args kept
    assert actions.validate(prop["action"], prop["args"], s) == []
    s.add(Result(id="fit", action="fit", args={}, summary={}, tables={}, step=1, source="core"))
    prop = boundary.next_step(FakeClient(), "fake", loop.observe(s))
    assert prop["action"] == "standardize" and prop["args"] == {"stdz": "both", "norm": "median"}
    assert "alternative" not in boundary.ARG_SCHEMA["properties"]
    planner = boundary.model_planner(FakeClient(), "fake", s)
    assert planner(loop.observe(s))["action"] == "standardize"


def test_figure_selection_is_validated():
    s = make_session()
    for rid, act in (("fit", "fit"), ("test:robust_score", "test_providers"), ("en:robust_score", "empirical_null"), ("test:score", "test_providers")):
        s.add(Result(id=rid, action=act, args={}, summary={}, tables={}, step=len(s.results) + 1, source="core"))
    chosen = boundary.select_figures(FakeClient(), "fake", s, "catalogue")
    names = [(c["display"], c["of"]) for c in chosen]
    assert ("funnel", "en:robust_score") in names and ("coefficient_forest", "fit") in names
    assert ("caterpillar", "test:score") in names         # applicability of the display itself is decided at render time
    c2 = FakeClient({"figure_selection": lambda u: {"reasoning": "", "figures": [{"display": "funnel", "of": "test:wald", "reason": "x"}]}})
    assert boundary.select_figures(c2, "fake", s, "catalogue") == []    # no such result
    assert any("not applicable" in r for r in s.refusals)


def test_prose_checks():
    refs = {"outcome": "AVF", "unit": "patient_month", "n_eligible": 200, "n_adjusters": 10, "c_within": 0.61, "sm_flips": 3,
            "recommended_test": "robust_score", "recommended_reference": "empirical", "flagged_worse": 6, "changed_max_reference": 53}
    prose, notes = boundary.write_prose(FakeClient(), "fake", refs, ["fact one"])
    assert "declared" in prose and "flags" in prose
    assert "alternatives" not in prose and any("digit" in n for n in notes)   # the fake wrote "3 at most"
    c2 = FakeClient({"report_prose": lambda u: {"reasoning": "", "declared": "see [nope] and [n_eligible]", "model": "", "measures": "", "flags": "", "alternatives": ""}})
    prose, notes = boundary.write_prose(c2, "fake", refs, [])
    assert prose["declared"] == "see  and [n_eligible]" and any("unknown" in n for n in notes)


def test_wording_checks():
    assert boundary.word(FakeClient(), "fake", "ask", "m", COLS)
    c2 = FakeClient({"refusal": lambda u: {"reasoning": "", "text": "You should drop the hospital column."}})
    assert boundary.word(c2, "fake", "refusal", "m", COLS) is None
    c3 = FakeClient({"explain": lambda u: {"reasoning": "", "text": "There were 12 flagged."}})
    assert boundary.word(c3, "fake", "explain", "m", COLS) is None


def test_decoding_settings():
    assert boundary.decoding_for("intent")["temperature"] == 0.0
    assert boundary.decoding_for("report_prose")["temperature"] == 0.7
    assert boundary.decoding_for("next_step")["thinking"] is True
    assert json.dumps(boundary.NEXT_STEP_SCHEMA)


def test_wording_allows_provider_ids_and_columns():
    c = FakeClient({"explain": lambda u: {"reasoning": "", "text": "F00078 is flagged because its ratio is low among HGB patients."}})
    assert boundary.word(c, "fake", "explain", "m", COLS + ["HGB"], allow=["F00078"])
    assert boundary.word(c, "fake", "explain", "m", COLS) is None


def test_prose_allows_column_names_with_digits():
    refs = {"outcome": "READMIT30", "n_eligible": 200}
    c = FakeClient({"report_prose": lambda u: {"reasoning": "", "declared": "The outcome READMIT30 is counted for [n_eligible] providers.",
                                                 "model": "", "measures": "", "flags": "", "alternatives": ""}})
    prose, notes = boundary.write_prose(c, "fake", refs, [], columns=["READMIT30", "PROVFS"])
    assert "declared" in prose and notes == []
    prose, notes = boundary.write_prose(c, "fake", refs, [])
    assert "declared" not in prose
