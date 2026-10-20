from pprof_agent import rules
from pprof_agent.specification import MeasureSpecification


def card(**kw):
    base = {"invariant_providers": {"total": 0}, "repeats": True, "has_patient": True,
            "patients_per_provider": {"median": 40, "max": 300}, "rows_per_provider": {"max": 3000},
            "eligible_providers": 200, "max_rows_per_patient": 12, "small_providers": {"le_3": 2}}
    base.update(kw)
    return base


SPEC = MeasureSpecification(standardization="both")


def test_repeats_give_robust_defaults():
    adm = rules.admissible(card(), SPEC)
    assert set(adm.tests_computed) == set(rules.TESTS)
    assert "score" not in adm.tests_reported and "exact" not in adm.tests_reported
    assert adm.default_test == "robust_score" and adm.variance_default == "robust"
    assert adm.references_computed == ["theoretical", "empirical"]
    assert adm.direct_transport


def test_few_patients_give_stabilized():
    adm = rules.admissible(card(patients_per_provider={"median": 10, "max": 50}), SPEC)
    assert adm.default_test == "stabilized_score" and adm.variance_default == "stabilized"


def test_long_patients_drop_stabilized():
    adm = rules.admissible(card(max_rows_per_patient=9000, patients_per_provider={"median": 10, "max": 50}), SPEC)
    assert not any(t in adm.tests_computed for t in rules.STABILIZED)
    assert adm.default_test == "robust_score"


def test_invariant_drops_wald_from_reported():
    adm = rules.admissible(card(invariant_providers={"total": 5}), SPEC)
    assert not any(t in adm.tests_reported for t in rules.WALD_TYPE)
    assert "robust_wald" in adm.tests_computed
    assert any("invariant" in c for c in adm.caveats)


def test_no_patient_gives_naive_exact():
    adm = rules.admissible(card(has_patient=False, repeats=False), SPEC)
    assert adm.tests_computed == ["exact", "score", "wald"]
    assert adm.default_test == "exact"
    adm = rules.admissible(card(has_patient=False, repeats=False, rows_per_provider={"max": 50000}), SPEC)
    assert adm.default_test == "score"


def test_few_providers_no_empirical_null():
    adm = rules.admissible(card(eligible_providers=30), SPEC)
    assert adm.references_computed == ["theoretical"]


def test_reference_rule():
    assert rules.reference_rule({"scale_max": 2.5, "location_max_abs": 0.1, "scale_ratio": 1.4})[0] == "empirical"
    assert rules.reference_rule({"scale_max": 1.05, "location_max_abs": 0.9, "scale_ratio": 1.1})[0] == "empirical"
    assert rules.reference_rule({"scale_max": 1.05, "location_max_abs": 0.1, "scale_ratio": 1.1})[0] == "theoretical"


def test_why_not():
    adm = rules.admissible(card(patients_per_provider={"median": 10, "max": 50}), SPEC)
    assert adm.default_test == "stabilized_score"
    assert "median provider" in rules.why_not(adm, "robust_score") and "10" in rules.why_not(adm, "robust_score")
    assert "independent" in rules.why_not(adm, "score")
    assert "score statistic" in rules.why_not(adm, "stabilized_wald")
    assert rules.why_not(adm, "stabilized_score") == "it is the rule default"
    adm = rules.admissible(card(invariant_providers={"total": 5}), SPEC)
    assert "invariant" in rules.why_not(adm, "robust_wald")
    adm = rules.admissible(card(max_rows_per_patient=9000), SPEC)
    assert "not computed" in rules.why_not(adm, "stabilized_score")


def test_exact_dropped_for_large_trials():
    adm = rules.admissible(card(trials_per_provider={"median": 9000, "max": 64000}), SPEC)
    assert "exact" not in adm.tests_computed and "trials" in adm.reasons["exact"]
    assert "not computed" in rules.why_not(adm, "exact")
    adm = rules.admissible(card(repeats=False, has_patient=False, trials_per_provider={"median": 9000, "max": 64000}), SPEC)
    assert adm.tests_computed == ["score", "wald"] and adm.default_test == "score"


def test_alternatives_by_axis():
    adm = rules.admissible(card(), SPEC)
    alt = rules.alternatives(adm, "robust_score")
    assert alt["statistic"] == ["robust_wald"]
    assert set(alt["variance"]) == {"score", "stabilized_score"}
    assert alt["reference"] == ["theoretical", "empirical"]
