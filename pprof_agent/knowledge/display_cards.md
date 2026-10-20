# The display catalogue

Each card: the question the display answers; what it is drawn of; when it is informative; when it is refused.

## funnel
Question: which providers depart from the norm by more than the test's sampling variation, and how does that depend on precision? Drawn of: a test result (`test:<name>` under the theoretical null, `en:<name>` under the empirical null). Shows the observed-to-expected ratio (score-type tests) or the provider effect (Wald-type tests) against the test's precision, with the limits of the same test, so a provider lies outside its limits exactly when it is flagged. Informative for the reported test and reference. Refused when the result has no funnel coordinates.

## caterpillar
Question: how uncertain is each provider's estimate, and where does it sit against the norm? Drawn of: a Wald-type test result (`test:wald`, `test:robust_wald`, `test:stabilized_wald`), which carries an interval; score-type and exact tests carry none and are refused. Shows the interval of every provider ordered by estimate, with a volume panel.

## observed_expected
Question: are the extreme providers small or large? Drawn of: a score-type or exact test result. Observed against expected events on square-root axes with the line of equality; a provider far from the line with few expected events is a small provider.

## coefficient_forest
Question: how large are the risk adjusters' effects, and which were dropped by the selection? Drawn of: `fit`. Coefficients with model-based intervals; adjusters dropped by a selection are marked.

## null_calibration
Question: is the theoretical null credible, or do the statistics spread more with provider size? Drawn of: `en:<name>`. The statistics against the theoretical density and the size-stratified empirical null, and the location and scale of each size group. Informative whenever an empirical null was fitted; decisive when the reported reference is the empirical null.

## flag_stability
Question: how fragile are the flags? Drawn of: `fit` (it reads every comparison). Providers flagged in each tail under the reported configuration and under each admissible alternative, with the number that change status. Informative when any alternative changes a provider's status.

## data_quality
Question: who is excluded, untested or degenerate, and why? Drawn of: a test result. Providers below the eligibility threshold, providers whose outcomes are all events or none, providers whose statistic could not be computed.

## calibration_plot
Question: does the risk-adjustment model predict the right rate across the range of risk? Drawn of: `evaluation`. Observed against mean predicted rate by decile of the risk score and of the full prediction.
