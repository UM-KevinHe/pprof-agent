from pprof_agent import actions, rules
from pprof_agent.specification import MeasureSpecification, Verification
from pprof_agent.state import Result, Session

CARD = {"invariant_providers": {"total": 3}, "repeats": True, "has_patient": True,
        "patients_per_provider": {"median": 40, "max": 300}, "rows_per_provider": {"max": 3000},
        "eligible_providers": 200, "max_rows_per_patient": 12, "small_providers": {"le_3": 2}}


def session(**kw):
    s = Session(data_path="x.csv", run_dir="/tmp/x", budget=kw.get("budget", 20))
    s.profile = {"columns": [{"name": n} for n in ("AVF", "PROVFS", "KECC_ID", "AGE", "HGB")], "candidates": {}}
    s.spec = MeasureSpecification.from_dict(dict(outcome="AVF", provider="PROVFS", patient="KECC_ID", adjusters=["AGE", "HGB"],
                                                 event_direction="higher_is_better", unit="patient_month", eligibility=2,
                                                 standardization="both", norm="median", reference="both",
                                                 consequential_tail="worse", level=0.95))
    s.verification = Verification(True, [], [], CARD)
    s.admissible = rules.admissible(CARD, s.spec)
    return s


def done(s, name, args):
    s.add(Result(id=actions.result_id(name, args), action=name, args=args, summary={}, tables={}, step=len(s.results) + 1, source="core"))


def test_unknown_and_unavailable_actions_are_refused():
    s = session()
    assert actions.validate("run_sql", {}, s)
    probs = actions.validate("calibrate_tests", {"tests": ["score"]}, s)
    assert len(probs) == 1 and probs[0].startswith("calibrate_tests needs fit first: run fit")


def test_enums_and_needs():
    s = session()
    assert actions.validate("test_providers", {"test": "robust_score"}, s)[0].startswith("test_providers needs fit first")
    done(s, "fit", {"adjusters": None})
    assert actions.validate("test_providers", {"test": "robust_score"}, s) == []
    assert actions.validate("test_providers", {"test": "bayes"}, s)
    probs = actions.validate("empirical_null", {"test": "robust_score"}, s)
    assert len(probs) == 1 and "run test_providers with test robust_score" in probs[0], probs
    assert actions.validate("fit", {"adjusters": ["AGE", "NOPE"]}, s)
    assert actions.validate("standardize", {"stdz": "both", "norm": "median", "extra": 1}, s)


def test_budget_and_core():
    s = session(budget=1)
    assert any("costs" in p for p in actions.validate("fit", {}, s))
    s.cost_used = 5                                        # the protected core may overrun the budget; free actions still run
    done(s, "fit", {"adjusters": None})
    assert actions.validate("compare_axes", {"axis": "reference"}, s) == []
    assert any("costs" in p for p in actions.validate("evaluate_model", {}, s))
    s = session()
    core = actions.protected_core(s)
    names = [n for n, _ in core]
    assert names[0] == "fit" and names[1] == "standardize" and names[-1] == "evaluate_model"
    assert names.count("test_providers") == len(s.admissible.tests_computed)
    assert names.count("empirical_null") == len(s.admissible.tests_reported)
    assert "test_measure" in names
    probs = actions.validate("finish", {}, s)
    assert probs and "protected core" in probs[0]


def test_result_ids():
    assert actions.result_id("test_providers", {"test": "score"}) == "test:score"
    assert actions.result_id("empirical_null", {"test": "score"}) == "en:score"
    assert actions.result_id("compare_axes", {"axis": "reference"}) == "compare:reference"
    assert actions.result_id("fit", {}) == "fit"


def test_card_lines_narrow_tests_to_admissible():
    s = session()
    lines = "\n".join(actions.card_lines(s))
    assert "stabilized_score" in lines and "calibrate_tests" in lines


def test_calibrate_and_subset_cards():
    s = session()
    done(s, "fit", {"adjusters": None})
    assert actions.validate("calibrate_tests", {"tests": ["robust_score", "score"], "n_sim": 50}, s) == []
    assert any("not admissible" in p for p in actions.validate("calibrate_tests", {"tests": ["bayes"], "n_sim": 50}, s))
    assert any("between" in p for p in actions.validate("calibrate_tests", {"n_sim": 5}, s))
    assert actions.result_id("subset", {"provider": "F00001"}) == "subset:F00001"
    assert actions.validate("subset", {"provider": "F00001", "by": ["AGE"]}, s) == []   # no gamma table in a fake session: ids unchecked
    assert "calibrate_tests" in "\n".join(actions.card_lines(s))


def test_no_repeats_and_measure_prerequisites():
    s = session()
    done(s, "fit", {"adjusters": None})
    done(s, "compare_axes", {"axis": "variance"})
    assert any("already run" in p for p in actions.validate("compare_axes", {"axis": "variance"}, s))
    assert actions.validate("compare_axes", {"axis": "reference"}, s) == []
    done(s, "standardize", {"stdz": "both", "norm": "median"})
    assert any("needs test:robust_wald" in p and "run test_providers with test robust_wald" in p
               for p in actions.validate("test_measure", {"se_source": "robust_wald"}, s))
    assert actions.validate("test_measure", {"se_source": "model", "reference": "theoretical"}, s) == []
    assert any("needs en:" in p for p in actions.validate("test_measure", {"se_source": "model", "reference": "empirical"}, s))


def test_calibration_payload_keeps_the_reportable_tests():
    from pprof_agent.engine import executor
    s = session()
    done(s, "fit", {"adjusters": None})
    p = executor.build_payload(s, "calibrate_tests", {"tests": ["score"], "n_sim": 30})
    assert set(s.admissible.tests_reported) <= set(p["tests"]) and "score" in p["tests"]
    p = executor.build_payload(s, "calibrate_tests", {"n_sim": 30})
    assert set(p["tests"]) == set(s.admissible.tests_computed)


def test_calibrated_test_needs_the_default_simulated():
    from pprof_agent import loop
    s = session()
    other = next(t for t in s.admissible.tests_reported if t != s.admissible.default_test)      # a reportable alternative
    rows = [{"test": other, "reference": "theoretical", "tail": "lower", "rate_pct": 2.6, "se_pct": 0.3, "nominal_pct": 2.5}]
    assert loop.calibrated_test(s, rows, "theoretical") is None
    rows.append({"test": s.admissible.default_test, "reference": "theoretical", "tail": "lower", "rate_pct": 1.0, "se_pct": 0.3, "nominal_pct": 2.5})
    pick = loop.calibrated_test(s, rows, "theoretical")
    assert pick is not None and pick[0] == other, pick


def test_available_lines_list_only_what_the_harness_accepts():
    s = session()
    lines = actions.available_lines(s)
    assert lines[0] == "fit" and len(lines) == 2 and lines[1].startswith("finish")   # nothing else has its prerequisite
    done(s, "fit", {"adjusters": None})
    lines = actions.available_lines(s)
    assert "fit" not in lines and "standardize stdz=both norm=median" in lines
    assert "test_providers test=robust_score" in lines and "evaluate_model" in lines
    assert "calibrate_tests n_sim=30" in lines and any(l.startswith("subset provider=") for l in lines)
    assert not any(l.startswith("test_measure") for l in lines)     # needs sm
    assert not any(l.startswith("empirical_null") for l in lines)   # needs its test
    done(s, "test_providers", {"test": "robust_score"})
    lines = actions.available_lines(s)
    assert "test_providers test=robust_score" not in lines and "empirical_null test=robust_score n_groups=4 size_by=patients" in lines
    assert "integer between 20 and 500" in "\n".join(actions.card_lines(s))


def test_prerequisites_for_supplies_what_the_harness_can():
    s = session()
    pre = actions.prerequisites_for(actions.validate("empirical_null", {"test": "robust_score"}, s), s)
    assert pre == [("fit", {"adjusters": None}), ("test_providers", {"test": "robust_score"})]
    done(s, "fit", {"adjusters": None})
    pre = actions.prerequisites_for(actions.validate("test_measure", {"se_source": "robust_wald", "reference": "empirical"}, s), s)
    names = [n for n, _ in pre]
    assert names[0] == "test_providers" and {"standardize", "empirical_null"} <= set(names), pre
    assert names.index("test_providers") < names.index("empirical_null")         # the test before its null
    assert actions.prerequisites_for(["calibrate_tests.n_sim must be between 20 and 500"], s) is None
    assert actions.prerequisites_for(actions.validate("empirical_null", {"test": "bayes"}, s), s) is None


def test_validate_without_the_planner_budget():
    s = session(budget=1)
    done(s, "fit", {"adjusters": None})
    s.cost_used = 5
    assert any("costs" in p for p in actions.validate("subset", {"provider": "F00001"}, s))
    assert actions.validate("subset", {"provider": "F00001"}, s, budget=False) == []


def test_words_with_refs_percent_and_collisions():
    from pprof_agent.chat import words_with_refs
    from pprof_agent.chat import Agent
    refs = {"eligibility": 3, "calibration_rate_pct": 3.0, "calibration_nominal_pct": 2.5, "calibration_default_rate_pct": 1.6,
            "patients_per_provider_median": 23.5, "few_patients_threshold": 30}
    out = words_with_refs("type I error at 3.0% against a nominal 2.5%; the default was at 1.6%; threshold 3 patients; median 23.5 (below 30); 2 tails; 7 tests; 9.87 ratio",
                          refs, Agent._NUMBER_REFS)
    assert "[calibration_rate_pct] against a nominal [calibration_nominal_pct]" in out, out
    assert "default was at [calibration_default_rate_pct]" in out and "threshold [eligibility] patients" in out, out
    assert "median [patients_per_provider_median] (below [few_patients_threshold])" in out and "two tails; seven tests; some ratio" in out, out
    assert "%" not in out


def test_resolve_percent_references_carry_their_sign():
    from pprof_agent import report
    refs = {"flagged_worse_pct": 5.7, "flagged_worse": 17, "calibration_nominal_pct": 2.5}
    assert report.resolve("[flagged_worse_pct] of providers ([flagged_worse])", refs) == "5.7% of providers (17)"
    assert report.resolve("nominal [calibration_nominal_pct]%", refs) == "nominal 2.5%"
