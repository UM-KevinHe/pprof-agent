# test_providers.R -- one provider-effect test for every fitted provider (pprofR::test_fe).
# payload: {run_dir, test: exact|score|wald|robust_wald|robust_score|stabilized_wald|stabilized_score,
#           null: "median" | <number>, alternative: two.sided|greater|less, level (0.95), key}
# writes tables/test_<key>.csv; returns counts.
#
# flag_raw follows pprofR: 1 = significantly HIGHER than the null, -1 = lower, 0 = as expected. The
# harness maps higher/lower to worse/better with the declared event direction.
#
# The table also carries what a display needs (pprof_py's ProviderProfile roles): the observed and
# expected counts under the null, the variance the statistic was referred to, the estimate on the
# display scale (O/E for score-type tests, the effect gamma - gamma0 for Wald-type tests), an interval
# where the test has one (Wald-type), and the funnel coordinates whose limits agree with the flags:
#   score-type   z = (O - E) / sqrt(V)  ->  estimate O/E, precision E^2/V, limits 1 +/- crit sqrt(V)/E
#   Wald-type    z = (g - g0) / se      ->  estimate g - g0, precision 1/se^2, limits +/- crit se
# For the exact test the funnel uses the normal approximation with the null variance (approximate;
# the flags themselves are exact) and this is recorded in `limit_rule`.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprofR))
TEST_MAP <- c(exact = "exact.poisbinom", exact_bootstrap = "exact.bootstrap", score = "score", wald = "wald",
              robust_wald = "robust_wald", robust_score = "robust_score",
              stabilized_wald = "stabilized_wald", stabilized_score = "stabilized_score")
WALD_TYPE <- c("wald", "robust_wald", "stabilized_wald")
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir")
  test <- arg("test")
  if (!test %in% names(TEST_MAP)) stop("unknown test: ", test, call. = FALSE)
  null <- arg("null", "median")
  if (!identical(null, "median")) null <- as.numeric(null)
  alternative <- arg("alternative", "two.sided")
  level <- as.numeric(arg("level", 0.95))
  key <- arg("key", test)
  fit <- load_fit(run_dir)
  cl <- fit$char_list
  lev <- rownames(fit$gamma)
  gam <- as.numeric(fit$gamma[, 1])
  null_value <- if (identical(null, "median")) stats::median(gam) else null
  r <- test_fe(if (test == "wald") wald_fit(fit) else fit, test = TEST_MAP[[test]], null = null, alternative = alternative, level = level,
               threads = threads(), n = as.integer(arg("n_resample", 10000)))
  i <- match(lev, rownames(r))
  sizes <- provider_sizes(fit)
  d <- fit$data_include
  prov <- factor(pid(d[[cl$prov.char]]), levels = lev)
  N <- if (is.null(cl$N.char) || !cl$N.char %in% names(d)) rep(1, nrow(d)) else as.numeric(d[[cl$N.char]])
  Y <- as.numeric(d[[cl$Y.char]])
  p0 <- as.numeric(stats::plogis(null_value + as.numeric(fit$linear_pred)))
  O <- as.numeric(rowsum(Y, prov))
  E <- as.numeric(rowsum(N * p0, prov))
  V0 <- as.numeric(rowsum(N * p0 * (1 - p0), prov))
  Ntot <- as.numeric(rowsum(N, prov))
  stat <- as.numeric(r$stat[i])
  crit <- if (alternative == "two.sided") stats::qnorm(1 - (1 - level) / 2) else stats::qnorm(level)
  se <- if ("se.gamma" %in% names(r)) as.numeric(r$se.gamma[i]) else rep(NA_real_, length(lev))
  if (test %in% WALD_TYPE) {
    estimate <- gam - null_value
    variance <- se^2
    ci_lower <- gam - crit * se; ci_upper <- gam + crit * se
    funnel_estimate <- estimate; funnel_precision <- 1 / variance
    funnel_lower <- -crit * se; funnel_upper <- crit * se
    limit_rule <- "normal, model or sandwich variance"
    estimate_kind <- "effect"; precision_kind <- "inverse_variance"
  } else {
    U <- O - E
    variance <- if (test %in% c("exact", "exact_bootstrap")) V0 else ifelse(is.finite(stat) & stat != 0, (U / stat)^2, V0)
    estimate <- O / E
    ci_lower <- rep(NA_real_, length(lev)); ci_upper <- rep(NA_real_, length(lev))
    funnel_estimate <- estimate; funnel_precision <- E^2 / variance
    funnel_lower <- 1 - crit * sqrt(variance) / E; funnel_upper <- 1 + crit * sqrt(variance) / E
    limit_rule <- if (test %in% c("exact", "exact_bootstrap")) "normal approximation with the null variance (flags are exact)" else "normal, the test's own variance"
    estimate_kind <- "ratio"; precision_kind <- "inverse_null_variance"
  }
  tab <- data.frame(provider = lev, stat = stat, p = as.numeric(r$p[i]),
                    flag_raw = as.integer(as.character(r$flag[i])), se = se,
                    estimate = estimate, null_value = null_value, ci_lower = ci_lower, ci_upper = ci_upper,
                    observed = O, expected = E, variance = variance, trials = Ntot,
                    n_rows = sizes$n_rows, n_patients = sizes$n_patients,
                    zero_events = O == 0, all_events = O == Ntot, finite_estimate = !(O == 0 | O == Ntot),
                    funnel_estimate = funnel_estimate, funnel_precision = funnel_precision,
                    funnel_lower = funnel_lower, funnel_upper = funnel_upper, stringsAsFactors = FALSE)
  write_table(tab, run_dir, paste0("test_", key))
  ok <- !is.na(tab$flag_raw)
  # the statistics read against N(0, 1), overall and by provider size (rank quartiles of patients, q1 the
  # smallest): the share beyond the two-sided critical value (about 100 * (1 - level) when N(0, 1) describes
  # them) and a robust scale, MAD / 0.6745 (about 1); what the planner reads to judge the theoretical null
  z <- tab$stat; zf <- is.finite(z)
  crit2 <- stats::qnorm(1 - (1 - level) / 2)
  qs <- c("q1", "q2", "q3", "q4")
  quart <- if (sum(zf) >= 8) qs[cut(rank(tab$n_patients, ties.method = "first"), 4, labels = FALSE)] else rep(NA_character_, length(z))
  by_size <- function(f) stats::setNames(lapply(qs, function(q) {
    k <- zf & quart %in% q
    if (sum(k) >= 3) round(f(z[k]), 3) else NULL
  }), qs)
  beyond <- function(x) 100 * mean(abs(x) > crit2)
  list(test = test, key = key, null = null, null_value = null_value, alternative = alternative, level = level,
       crit = crit, n_tested = sum(ok), n_na = sum(!ok),
       flagged_higher = sum(tab$flag_raw == 1, na.rm = TRUE), flagged_lower = sum(tab$flag_raw == -1, na.rm = TRUE),
       pct_higher = 100 * mean(tab$flag_raw[ok] == 1), pct_lower = 100 * mean(tab$flag_raw[ok] == -1),
       stat = q5(tab$stat), estimate_kind = estimate_kind, precision_kind = precision_kind, limit_rule = limit_rule,
       z_beyond_pct = if (any(zf)) round(beyond(z[zf]), 3) else NULL,
       z_beyond_higher_pct = if (any(zf)) round(100 * mean(z[zf] > crit2), 3) else NULL,
       z_beyond_lower_pct = if (any(zf)) round(100 * mean(z[zf] < -crit2), 3) else NULL,
       z_beyond_pct_by_size = by_size(beyond),
       z_scale = if (sum(zf) >= 3) round(stats::mad(z[zf]), 3) else NULL, z_scale_by_size = by_size(stats::mad),
       seconds = seconds_since(t0),
       tables = stats::setNames(list(table_path(run_dir, paste0("test_", key))), paste0("test_", key)))
})
