"""End to end on synthetic data: needs R, pprofR, grplasso and PPROF_AGENT_DATA=<dir with data.csv, truth.json>."""
import json
import os
import re
from pathlib import Path

import pytest

DATA = os.environ.get("PPROF_AGENT_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="PPROF_AGENT_DATA not set")

from pprof_agent import compare, pipeline, report  # noqa: E402


def truth():
    return json.load(open(Path(DATA) / "truth.json"))


def spec_of(t):
    return {"outcome": t["outcome"], "provider": t["provider"], "patient": t["patient"], "adjusters": t["covariates"],
            "event_direction": t["event_direction"], "unit": "patient_month" if t["kind"] == "patient_month" else "index_discharge"}


def test_full_run(tmp_path):
    t = truth()
    s = pipeline.run_analysis(str(Path(DATA) / "data.csv"), str(tmp_path), spec_dict=spec_of(t), name="full", log=lambda m: None)
    assert s.phase == "DONE", s.refusals
    assert s.has("fit") and s.has("sm") and s.has("evaluation")
    for test in s.admissible.tests_computed:
        assert s.has(f"test:{test}")
    rec = s.recommended
    assert rec["test"] in s.admissible.tests_reported
    df = compare.table_of(s, rec["test"], rec["reference"])
    counts = compare.counts(df)
    refs = report.build_references(s)
    assert refs["flagged_worse"] == counts["worse"] and refs["flagged_better"] == counts["better"]
    worse = set(t["worse"])
    flagged = set(df.loc[df["status"] == "worse", "provider"].astype(str))
    assert len(worse & flagged) >= len(worse) // 2, (worse, flagged)
    run = Path(s.run_dir)
    for f in ("report.md", "repro.R", "trace.json", "spec.json", "tables/gamma.csv", "tables/sm.csv"):
        assert (run / f).exists(), f
    assert s.has("compare:reference") and s.has("compare:variance")
    figs = [r for r in s.results if r.action == "figure"]
    assert figs, "no figure attempted"
    assert any(r.summary.get("path") for r in figs), [r.summary.get("error") for r in figs]
    assert report.check_no_digits("the rate was [event_rate_pct] percent") is None
    assert report.check_no_digits("the rate was 12 percent") is not None
    assert report.resolve("[n_eligible] providers", refs).startswith(str(refs["n_eligible"]))


def test_loop_supplies_prerequisites_of_a_proposal(tmp_path):
    """A planner that asks for the empirical null before its test: the harness fits and tests first, then accepts."""
    from pprof_agent import loop
    from pprof_agent.state import Session
    from pprof_agent.engine import executor
    from pprof_agent import specification as S, rules
    t = truth()
    run = pipeline.new_run_dir(str(tmp_path), "prereq")
    s = Session(data_path=str(Path(DATA) / "data.csv"), run_dir=str(run), budget=30)
    executor.profile(s)
    spec, items = S.complete(S.MeasureSpecification.from_dict(spec_of(t), source="quoted"), s.profile)
    s.spec, s.open_items = spec, items
    assert not items
    card = executor.verify_card(s)
    s.verification = S.verify(spec, card)
    s.admissible = rules.admissible(card, spec)
    default = s.admissible.default_test
    calls = []
    def planner(obs):
        calls.append(obs["steps_left"])
        if len(calls) == 1:
            return {"reading": "", "action": "empirical_null", "args": {"test": default, "n_groups": 4, "size_by": "patients"}, "reason": "the null first"}
        return {"reading": "", "action": "finish", "args": {}, "reason": "done"}
    loop.run_loop(s, planner=planner, log=lambda m: None)
    assert s.has("fit") and s.has(f"test:{default}") and s.has(f"en:{default}")
    first = next(e for e in s.plan_log if e.get("proposal", {}).get("action") == "empirical_null")
    assert first["problems"] == [] and set(first["prerequisites_run"]) == {"fit", f"test:{default}"}, first
    assert [r.source for r in s.results if r.id in ("fit", f"test:{default}")] == ["harness", "harness"]
    assert s.get(f"en:{default}").source == "planner"


def test_open_items_ask(tmp_path):
    t = truth()
    s = pipeline.run_analysis(str(Path(DATA) / "data.csv"), str(tmp_path), spec_dict={"adjusters": t["covariates"]},
                              name="ask", log=lambda m: None, figures=False)
    assert s.phase == "DECLARING"
    assert "2)" in s.question and "3)" in s.question


def test_reply_then_run(tmp_path):
    t = truth()
    unit = "month" if t["kind"] == "patient_month" else "discharge"
    direction = "better" if t["event_direction"] == "higher_is_better" else "worse"
    reply = f"1) {t['outcome']} 2) {direction} 3) {unit} 4) {t['provider']} 5) {t['patient']} 6) B"
    s = pipeline.run_analysis(str(Path(DATA) / "data.csv"), str(tmp_path), spec_dict={}, reply=reply,
                              name="reply", log=lambda m: None, figures=False)
    assert s.phase == "DONE", (s.question, s.refusals, s.verification and s.verification.refusals)
    assert s.spec.sources["outcome"] == "asked" and s.spec.sources["eligibility"] == "default"
    assert set(s.spec.adjusters) >= set(t["covariates"])


def test_refusal_without_patient(tmp_path):
    t = truth()
    sp = spec_of(t)
    sp["patient"] = "none"
    sp["unit"] = "patient_month"
    s = pipeline.run_analysis(str(Path(DATA) / "data.csv"), str(tmp_path), spec_dict=sp, name="refuse", log=lambda m: None, figures=False)
    assert s.phase == "REFUSED"
    assert any("patient column" in r for r in s.verification.refusals)


def test_chat_with_fake_model(tmp_path):
    """The whole conversation path with the fake client: prose declaration -> extraction -> run with the
    model planner -> figures chosen by the model -> prose -> a result question -> a refusal."""
    from pprof_agent.chat import Agent
    from pprof_agent.fake import FakeClient
    t = truth()
    unit = "patient-month" if t["kind"] == "patient_month" else "index discharge"
    direction = "better" if t["event_direction"] == "higher_is_better" else "worse"
    agent = Agent(str(Path(DATA) / "data.csv"), str(tmp_path), client=FakeClient(), model="fake", name="chat", log=lambda m: None)
    opening = agent.start()
    assert "rows" in opening
    msg = (f"The outcome is {t['outcome']}, one row per {unit}, and a higher rate is {direction}. "
           f"{t['provider']} identifies the facility and {t['patient']} the patient. Adjust for " + ", ".join(t["covariates"][:-1]) + " and " + t["covariates"][-1] + ".")
    reply = agent.submit(msg)
    s = agent.session
    assert s.phase == "DONE", (reply, s.refusals)
    assert s.spec.sources["outcome"] == "quoted" and s.spec.sources["adjusters"] == "quoted"
    assert "Done." in reply and "Report:" in reply
    assert any(r.source == "planner" for r in s.results if r.action in ("fit", "standardize"))
    figs = [r for r in s.results if r.action == "figure"]
    assert any(r.summary.get("path") for r in figs)
    assert any(r.summary.get("error") for r in figs if r.args.get("display") == "caterpillar")   # the score test has no interval
    names = {mc["name"] for mc in s.model_calls}
    assert {"intent", "spec_extraction", "next_step", "figure_selection", "report_prose"} <= names
    md = (Path(s.run_dir) / "report.md").read_text()
    assert "providers meet the threshold" in md            # the fake's prose, with references resolved
    assert "[n_eligible]" not in md
    out = agent.submit("Why is the empirical null the reported reference?")
    assert out and "could not" not in out
    refs = report.build_references(s)
    med = refs["patients_per_provider_median"]
    worded = agent._words(f"the median provider has {med} patients (below {refs['few_patients_threshold']}); 1 of 2 groups; scale 9.87 against 1")
    assert "[patients_per_provider_median]" in worded and "[few_patients_threshold]" in worded, worded
    assert "one of two groups" in worded and "scale some against one" in worded, worded
    assert not re.search(r"\d", re.sub(r"\[[a-z_]+\]", "", worded)), worded
    assert any(f.startswith("why not robust_score") for f in agent._facts()), agent._facts()
    if refs.get("en_location_max") is not None:
        worded = agent._words(f"the size groups have scale at most {refs['en_scale_max']:.2f} and location within {refs['en_location_max']:.2f} of zero")
        assert "[en_scale_max]" in worded and "[en_location_max]" in worded and "some" not in worded, worded
    facts, ids = agent._axis_facts("What would change if we used the theoretical null instead?")
    other = "theoretical" if s.recommended["reference"] == "empirical" else "empirical"
    assert any(f"[worse_under_{other}_null]" in f for f in facts), facts
    refs = report.build_references(s)
    assert f"worse_under_{other}_null" in refs and f"changed_under_{other}_null" in refs
    assert all(i in agent._providers_named(" ".join(ids)) for i in ids)       # the ids are fitted providers
    assert not agent._axis_facts("Why is the model fitted with provider effects?")[0]
    same = agent._axis_facts(f"What would change if we used the {s.recommended['reference']} null instead?")[0]
    assert same and same[0].startswith("the question names the") and "already the reported configuration" in same[0], same
    assert "already the reported" not in " ".join(agent._axis_facts(f"What would change under the {other} null?")[0])
    assert agent._levels_named("why not the robust score test") == {"variance": "robust", "statistic": "score"}
    assert agent._levels_named("under indirect standardization") == {"standardization": "indirect"}
    ff, fids = agent._flag_facts("Which facilities are flagged worse, and how sure can I be about them?")
    assert ff and ff[0].startswith("flagged worse under the reported configuration ([flagged_worse] providers): "), ff
    assert any("[worse_under_every_alternative]" in f and "[n_alternatives]" in f for f in ff), ff
    assert not any("flagged better under" in f for f in ff)        # the question named one tail
    assert any("calibration" in f for f in ff)
    assert fids and all(i in agent._providers_named(" ".join(fids)) for i in fids)
    stab = compare.stability(s)
    assert stab["n_alternatives"] >= 1 and set(stab["worse"]["stable"]) <= set(stab["worse"]["flagged"])
    refs = report.build_references(s)
    assert refs["worse_under_every_alternative"] == len(stab["worse"]["stable"])
    assert report.resolve("[flagged_worse_pct] worse", refs).endswith("% worse")
    assert agent._flag_facts("Why is the model fitted with provider effects?") == ([], [])
    out = agent.submit("Which facilities are flagged worse than expected, and how sure can I be about them?")
    assert out and "could not" not in out
    assert "Flagged worse under the reported configuration (" in out and stab["worse"]["flagged"][0] in out, out
    assert agent._flag_lists("Why is the model fitted with provider effects?") == ""
    sf = [f for f in agent._axis_facts("What would change under direct standardization?")[0] if "instead of the reported" in f]
    assert len(sf) == 1 and "direct" in sf[0] and "model" not in sf[0].split("instead")[0].replace("model standard error", ""), sf
    out = agent.submit("What would change if we used the theoretical null instead?")
    assert out and "could not" not in out
    assert agent._tests_named("Which test is reported, and why not the robust score test?") == ["robust_score"]
    assert agent._tests_named("why not the score test or the exact one") == ["exact", "score"]
    pid = t["worse"][0]
    out = agent.submit(f"Why is facility {pid} flagged?")
    assert out and "could not" not in out
    assert s.has(f"subset:{pid}")
    bf = agent._breakdown_facts(pid)
    assert bf and not re.search(r"\d", re.sub(r"\[[a-z_]+\]|F\d+", "", " ".join(bf))), bf
    assert bf[0].startswith(f"{pid} is flagged worse") or bf[0].startswith(f"{pid} is not flagged"), bf[0]
    assert "as_expected" not in " ".join(bf)
    assert agent._level_words("COPD", 1) == "with COPD" and agent._level_words("AGE", "low") == "with AGE in its low third"
    out = agent.submit("Can you adjust for the discharging hospital too?")
    assert "not" in out.lower()
    # the transcript, the session loader and the bundle for the web app
    s.save("trace.json")
    assert s.transcript and s.transcript[0]["role"] == "agent" and s.transcript[1]["text"].startswith("The outcome is")
    assert all(set(m) == {"role", "text", "figures"} for m in s.transcript)
    from pprof_agent.state import Session
    s2 = Session.load(s.run_dir)
    assert s2.phase == "DONE" and s2.has("fit") and s2.recommended == s.recommended and s2.spec.config() == s.spec.config()
    assert [r.id for r in s2.results] == [r.id for r in s.results] and len(s2.transcript) == len(s.transcript)
    from pprof_agent import export
    b = export.bundle(s.run_dir)
    assert b["tables"]["gamma"] and b["figures"] and b["references"]["flagged_worse"] == report.build_references(s)["flagged_worse"]
    assert b["stability"]["worse"]["flagged"] and b["transcript"] and b["recommended"]["test"] == s.recommended["test"]
    assert all(isinstance(v, (int, float, str, type(None), bool)) for row in b["tables"]["gamma"] for v in row.values())
    p = export.write_bundle(s.run_dir, str(tmp_path / "bundle.json"))
    assert p.exists() and json.loads(p.read_text())["run"] == Path(s.run_dir).name
    n = export.subset_flagged(s2)
    assert n >= 0 and all(s2.has(f"subset:{pid}") for pid in b["stability"]["worse"]["flagged"])
