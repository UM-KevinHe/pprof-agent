# Profiling playbook: situation notes for the planner (no numbers)

## always
The protected core runs whatever you plan: the fit, both standardizations, every admissible test under both reference distributions, and the model evaluation. Plan only what adds to it, and finish when nothing affordable would change the flags or how they are read.

## invariant_outcomes
Some providers have all events or none. Their effect estimate diverges and so does its standard error, so a Wald-type statistic cannot flag them; score-type and exact tests use the observed count and remain finite. The rule keeps Wald-type statistics as alternatives, not as the reported test.

## repeated_observations
Patients contribute several rows, so the rows of one patient are correlated. A variance that treats them as independent is too small and flags too many providers. The patient-clustered sandwich variance is consistent whatever the correlation; it is the robust variance.

## few_patients
When the median provider has few patients, the provider-specific sandwich variance is badly variable and biased downward, so the robust test flags too many small providers. The stabilized variance pools one within-patient correlation over all patients and is far less variable, at the price of assuming the correlation is common across providers.

## small_providers
Providers with very few patients cannot be flagged reliably by any test: a departure there is within chance. The report shows them; a flag there deserves the caveat.

## empirical_null
The theoretical null presumes that only sampling variation remains after adjustment. Residual risk the adjusters do not capture stays in the provider effect, and because large providers are estimated precisely, the theoretical null flags them for departures of no consequence. The empirical null estimates the centre and spread of the statistics within size groups from their middle; if the spread grows with size, it is the reference the measure intends.

## after_empirical_null
Read the size groups: a scale near one in every group means the theoretical null describes the statistics; a scale that grows with size means overdispersion the empirical null absorbs. The reference rule decides from these; the comparison across the reference axis shows which providers change status.

## direct_standardization
Indirect standardization evaluates a provider on its own patients and is not comparable between providers; direct standardization evaluates every provider on the whole population and extrapolates. When a provider's performance varies with case mix, its verdict can flip between the two. The directly standardized rate is tested on its own scale with the delta-method standard error.

## selection
When a candidate set was declared, the penalized fit with the provider effects kept unpenalized selects the adjusters by cross-validation; the pooled model is never used, because it confounds the provider effect with every correlated covariate. Compare the flags under the selected and the declared sets if both were fitted.
