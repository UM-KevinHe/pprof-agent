"""The derivation rules: from the consequence card and the specification to the admissible set.

Deterministic, from the paper (`sec:inference`, `sec:agent`) and the proposal deck (slide 14):
  all-same outcomes       -> Wald-type statistics are not admitted as the reported test
  repeats per patient     -> the naive (working-independence) variances are not admitted
  few patients            -> the stabilized variance replaces the robust one as the default
  spread grows with size  -> the empirical null is the reported reference (decided after it is fitted)
  direct standardization  -> the test is transported to the standardized scale
The harness computes the admissible set and shows it to the planner; the planner chooses within it
and can never add to it. Everything admissible is computed whatever the planner does (the protected
core, loop.py), so an alternative can never be suppressed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

TESTS = ("exact", "score", "wald", "robust_wald", "robust_score", "stabilized_wald", "stabilized_score")
WALD_TYPE = ("wald", "robust_wald", "stabilized_wald")
SCORE_TYPE = ("score", "robust_score", "stabilized_score")
NAIVE = ("exact", "score", "wald")
ROBUST = ("robust_wald", "robust_score")
STABILIZED = ("stabilized_wald", "stabilized_score")
VARIANCE_OF = {"exact": "naive", "score": "naive", "wald": "naive", "robust_wald": "robust", "robust_score": "robust",
               "stabilized_wald": "stabilized", "stabilized_score": "stabilized"}
STATISTIC_OF = {"exact": "exact", "score": "score", "wald": "wald", "robust_wald": "wald", "robust_score": "score",
                "stabilized_wald": "wald", "stabilized_score": "score"}

THRESHOLDS: Dict[str, float] = {
    "few_patients_median": 30,       # median patients per provider below this: stabilized replaces robust
    "en_min_providers": 40,          # fewer eligible providers: no empirical null (4 groups of 10)
    "en_scale_max": 1.25,            # the empirical null is reported when a size group's scale exceeds this
    "en_location_max": 0.5,          # ... or a group's location moves this far from zero
    "stabilized_max_rows": 5000,     # pprofR pools a matrix over a patient's rows; larger patients refuse
    "exact_max_trials": 20000,       # the exact test enumerates trials; above this the score test is the default
}


@dataclass
class Admissible:
    tests_computed: List[str]
    tests_reported: List[str]
    default_test: str
    variance_default: str
    references_computed: List[str]
    default_reference: str
    direct_transport: bool
    caveats: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def admissible(card: Dict[str, Any], spec: Any, thresholds: Dict[str, float] = THRESHOLDS) -> Admissible:
    inv = int((card.get("invariant_providers") or {}).get("total", 0) or 0)
    repeats = bool(card.get("repeats"))
    has_patient = bool(card.get("has_patient"))
    med_pat = (card.get("patients_per_provider") or {}).get("median") or 0
    n_prov = int(card.get("eligible_providers") or 0)
    max_rows_pp = int(card.get("max_rows_per_patient") or 1)
    max_trials = (card.get("trials_per_provider") or card.get("rows_per_provider") or {}).get("max") or 0
    reasons: Dict[str, str] = {}
    caveats: List[str] = []
    exact_ok = max_trials <= thresholds["exact_max_trials"]

    if has_patient and repeats:
        computed = list(TESTS)
        if not exact_ok:
            computed = [t for t in computed if t != "exact"]
            reasons["exact"] = (f"the largest provider has {int(max_trials)} trials; the exact enumeration is not computed above "
                                f"{int(thresholds['exact_max_trials'])}")
        if max_rows_pp > thresholds["stabilized_max_rows"]:
            computed = [t for t in computed if t not in STABILIZED]
            reasons["stabilized"] = (f"a patient contributes up to {max_rows_pp} rows; the stabilized variance pools a matrix "
                                     f"over a patient's rows and is not computed beyond {int(thresholds['stabilized_max_rows'])}")
        reported = [t for t in computed if t not in NAIVE]
        reasons["naive"] = (f"patients contribute up to {max_rows_pp} rows; a variance that treats rows as independent is "
                            "too small, so the naive statistics are alternatives, not recommendations")
        few = med_pat < thresholds["few_patients_median"]
        if few and any(t in computed for t in STABILIZED):
            variance_default = "stabilized"
            reasons["variance"] = (f"the median provider has {med_pat} patients (below {int(thresholds['few_patients_median'])}); "
                                   "the provider-specific sandwich is unstable there, so the correlation pooled over all patients is used")
        else:
            variance_default = "robust"
            reasons["variance"] = (f"the median provider has {med_pat} patients; the patient-clustered sandwich variance is used")
    else:
        computed = ["exact", "score", "wald"] if exact_ok else ["score", "wald"]
        reported = list(computed)
        variance_default = "naive"
        if not exact_ok:
            reasons["exact"] = (f"the largest provider has {int(max_trials)} trials; the exact enumeration is not computed above "
                                f"{int(thresholds['exact_max_trials'])}")
        reasons["naive"] = ("each patient contributes one row" if has_patient else
                            "no patient column: rows are taken as independent (declare one if patients repeat)")
    if inv > 0:
        reported = [t for t in reported if t not in WALD_TYPE]
        reasons["wald"] = (f"{inv} providers have invariant outcomes (all events or none): their effect estimate and its "
                           "standard error diverge, so Wald-type statistics are alternatives, not recommendations")
        caveats.append("Wald-type statistics cannot flag the providers with invariant outcomes")
    if variance_default == "stabilized":
        default_test = "stabilized_score"
    elif variance_default == "robust":
        default_test = "robust_score"
    else:
        default_test = "exact" if exact_ok else "score"
        reasons["statistic"] = ("the exact test needs no large-sample approximation in the provider's size" if default_test == "exact"
                                else f"the largest provider has {int(max_trials)} trials; the exact enumeration is replaced by the score test")
    if default_test not in reported:
        default_test = next((t for t in reported if t in SCORE_TYPE), reported[0] if reported else computed[0])
    if n_prov >= thresholds["en_min_providers"]:
        references = ["theoretical", "empirical"]
        reasons["empirical_null"] = (f"{n_prov} eligible providers: the empirical null can be estimated in four size groups")
    else:
        references = ["theoretical"]
        reasons["empirical_null"] = (f"only {n_prov} eligible providers: too few to estimate a size-stratified empirical null")
    direct = getattr(spec, "standardization", "both") in ("direct", "both")
    if direct:
        reasons["transport"] = "a directly standardized rate is tested on its own scale, with the delta-method standard error"
    small = int((card.get("small_providers") or {}).get("le_3", 0) or 0)
    if small:
        caveats.append(f"{small} providers have three patients or fewer; no test is well calibrated there")
    return Admissible(tests_computed=computed, tests_reported=reported, default_test=default_test,
                      variance_default=variance_default, references_computed=references,
                      default_reference="theoretical", direct_transport=direct, caveats=caveats, reasons=reasons)


def reference_rule(en_summary: Dict[str, Any], thresholds: Dict[str, float] = THRESHOLDS) -> Tuple[str, str]:
    """Decide the reported reference distribution from the fitted empirical null of the default test."""
    scale_max = float(en_summary.get("scale_max") or 0)
    loc = float(en_summary.get("location_max_abs") or 0)
    ratio = float(en_summary.get("scale_ratio") or 1)
    if scale_max > thresholds["en_scale_max"]:
        return "empirical", (f"the statistics are overdispersed: the widest size group has scale {scale_max:.2f} against 1 "
                             f"under the theoretical null (ratio across groups {ratio:.2f})")
    if loc > thresholds["en_location_max"]:
        return "empirical", f"the statistics are off-centre: a size group has location {loc:.2f}"
    return "theoretical", (f"the size groups have scale at most {scale_max:.2f} and location within {loc:.2f} of zero; "
                           "the theoretical null describes them")


def why_not(adm: Admissible, test: str) -> str:
    """Why a test is not the rule default, from the reasons the rules recorded (the calibration evidence, when
    it overrides the default, is recorded by the recommendation itself)."""
    r = adm.reasons
    if test == adm.default_test:
        return "it is the rule default"
    if test not in adm.tests_computed:
        return (r.get("exact") if test == "exact" else r.get("stabilized")) or "it is not computed for this file"
    if test in NAIVE and adm.variance_default != "naive":
        return r.get("naive", "its variance treats rows as independent")
    if test in WALD_TYPE and test not in adm.tests_reported:
        return r.get("wald", "Wald-type statistics are not reported for this file")
    if VARIANCE_OF[test] != adm.variance_default and VARIANCE_OF[test] in ("robust", "stabilized"):
        return r.get("variance", "")
    if STATISTIC_OF[test] != STATISTIC_OF[adm.default_test]:
        if adm.variance_default == "naive":
            return r.get("statistic", "")
        return ("the score statistic is evaluated at the null and needs no estimate of the provider's own effect, so it is the "
                "default within a variance treatment; the Wald statistic is carried as an alternative")
    return "it is an admissible alternative with the same statistic and variance"


def alternatives(adm: Admissible, test: str) -> Dict[str, List[str]]:
    """The other admissible choices one axis at a time, from a reported test."""
    stat = STATISTIC_OF[test]
    var = VARIANCE_OF[test]
    out = {
        "statistic": [t for t in adm.tests_computed if VARIANCE_OF[t] == var and t != test],
        "variance": [t for t in adm.tests_computed if STATISTIC_OF[t] == stat and t != test],
        "reference": [r for r in adm.references_computed],
    }
    return out
