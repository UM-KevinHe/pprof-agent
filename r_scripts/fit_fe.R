# fit_fe.R -- the fixed provider effects logistic model (pprof::fit_logistic_fe, SerBIN), binary or binomial.
# payload: {data_path, run_dir, spec: {outcome, trials, provider, patient, adjusters, eligibility},
#           tol (1e-6), bound (10), algorithm ("SerBIN")}
# writes run_dir/fit.rds, tables/gamma.csv, tables/beta.csv; returns the fit summary.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprof))
run_step({
  t0 <- Sys.time()
  spec <- PAYLOAD$spec
  run_dir <- arg("run_dir")
  outcome_col <- arg("outcome", from = spec); trials_col <- arg("trials", from = spec)
  provider_col <- arg("provider", from = spec); patient_col <- arg("patient", from = spec)
  if (identical(patient_col, "none")) patient_col <- NULL
  adjusters <- chr(arg("adjusters", from = spec))
  if (!length(adjusters)) stop("no adjusters declared", call. = FALSE)
  eligibility <- as.integer(arg("eligibility", 1, from = spec))
  tol <- as.numeric(arg("tol", 1e-6)); bound <- as.numeric(arg("bound", 10)); algorithm <- arg("algorithm", "SerBIN")
  need <- unique(c(outcome_col, trials_col, provider_col, patient_col, adjusters))
  df <- load_data(arg("data_path"), need)
  n0 <- nrow(df)
  df <- df[stats::complete.cases(df[, need, drop = FALSE]), , drop = FALSE]
  df[[provider_col]] <- pid(df[[provider_col]])
  if (!is.null(patient_col)) df[[patient_col]] <- pid(df[[patient_col]])
  for (a in adjusters) df[[a]] <- as.numeric(df[[a]])
  # the eligibility threshold counts patients when a patient column exists, rows otherwise
  pd <- pp_data(df, y = outcome_col, x = adjusters, provider = provider_col, patient = patient_col, n = trials_col,
                cutoff = eligibility, check = FALSE, message = FALSE)
  n_excluded <- sum(!pd$sizes$included)
  t1 <- Sys.time()
  fit <- fit_logistic_fe(pd, algorithm = algorithm, tol = tol, bound = bound, threads = threads(),
                         auc = is.null(trials_col), message = FALSE)
  fit_seconds <- seconds_since(t1)
  V <- vcov(fit)
  se_gamma <- sqrt(attr(V, "var_gamma"))
  lev <- fit_levels(fit)
  sizes <- provider_sizes(fit)
  inv <- as.logical(fit$sizes$no_events | fit$sizes$all_events)
  gam <- fit_gamma(fit)
  at_bound <- abs(gam - stats::median(gam)) >= bound - 1e-6
  Y <- fit_Y(fit); N <- fit_N(fit)
  prov <- factor(fit_prov(fit), levels = lev)
  obs <- as.numeric(rowsum(Y, prov))
  gamma_tab <- data.frame(provider = lev, gamma = gam, se_model = as.numeric(se_gamma[lev]),
                          observed = obs, n_rows = sizes$n_rows, n_patients = sizes$n_patients,
                          invariant = inv, at_bound = at_bound, stringsAsFactors = FALSE)
  write_table(gamma_tab, run_dir, "gamma")
  beta <- fit_beta(fit)
  beta_tab <- data.frame(term = names(beta), beta = as.numeric(beta), se_model = sqrt(diag(V)), stringsAsFactors = FALSE)
  write_table(beta_tab, run_dir, "beta")
  saveRDS(fit, fit_path(run_dir))
  # events per adjuster: the rarer of events and non-events over the number of adjusters (the usual EPV)
  n_events <- sum(Y)
  n_trials <- sum(N)
  # the within-patient correlation of standardized residuals: what a working-independence variance ignores
  rho <- NULL
  if (!is.null(patient_col)) {
    pr <- fit_pred(fit)
    e <- (Y - N * pr) / sqrt(pmax(N * pr * (1 - pr), 1e-12))
    pat <- factor(paste(fit_prov(fit), fit_pat(fit)))
    rho <- round(within_corr(e, pat), 4)
  }
  list(n_rows_read = n0, n_rows_fitted = nrow(fit$data), n_rows_dropped_incomplete = n0 - nrow(df),
       n_providers_fitted = length(lev), n_providers_excluded = n_excluded, n_patients = sum(sizes$n_patients),
       p = nrow(beta_tab), iterations = fit$control$iterations, algorithm = algorithm, tol = tol, bound = bound,
       event_rate = n_events / n_trials, events = n_events,
       events_per_adjuster = min(n_events, n_trials - n_events) / max(nrow(beta_tab), 1),
       auc = if (is.null(fit$auc) || is.na(fit$auc)) NULL else fit$auc,
       gamma_median = stats::median(gam), gamma_quantiles = q5(gam),
       invariant_providers = sum(inv), at_bound_providers = sum(at_bound), within_patient_corr = rho,
       fit_seconds = fit_seconds, seconds = seconds_since(t0),
       tables = list(gamma = table_path(run_dir, "gamma"), beta = table_path(run_dir, "beta")))
})
