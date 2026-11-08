# fit_fe.R -- the fixed provider effects logistic model (pprofR::logis_fe, SerBIN), binary or binomial.
# payload: {data_path, run_dir, spec: {outcome, trials, provider, patient, adjusters, eligibility},
#           tol (1e-6), bound (10), algorithm ("SerBIN")}
# writes run_dir/fit.rds, tables/gamma.csv, tables/beta.csv; returns the fit summary.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprofR))
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
  args_prep <- list(Y = df[, outcome_col, drop = FALSE], Z = df[, adjusters, drop = FALSE],
                    ProvID = df[, provider_col, drop = FALSE], cutoff = eligibility, check = FALSE, message = FALSE)
  if (!is.null(trials_col)) args_prep$N <- df[, trials_col, drop = FALSE]
  if (!is.null(patient_col)) args_prep$PatientID <- df[, patient_col, drop = FALSE]
  dp <- do.call(fe_data_prep, args_prep)
  n_excluded <- length(unique(dp$data[[provider_col]][dp$data$included == 0]))
  t1 <- Sys.time()
  fit <- logis_fe(dp, algorithm = algorithm, tol = tol, bound = bound, threads = threads(),
                  AUC = is.null(trials_col), message = FALSE)
  fit_seconds <- seconds_since(t1)
  V <- vcov(fit)
  se_gamma <- sqrt(attr(V, "var.gamma"))
  lev <- rownames(fit$gamma)
  sizes <- provider_sizes(fit)
  d <- fit$data_include
  inv <- tapply(d$no.events + d$all.events, factor(pid(d[[provider_col]]), levels = lev), max) > 0
  gam <- as.numeric(fit$gamma[, 1])
  at_bound <- abs(gam - stats::median(gam)) >= bound - 1e-6
  obs <- as.numeric(fit$df.prov$Obs_provider[match(lev, rownames(fit$df.prov))])
  gamma_tab <- data.frame(provider = lev, gamma = gam, se_model = as.numeric(se_gamma[lev]),
                          observed = obs, n_rows = sizes$n_rows, n_patients = sizes$n_patients,
                          invariant = as.logical(inv[lev]), at_bound = at_bound, stringsAsFactors = FALSE)
  write_table(gamma_tab, run_dir, "gamma")
  beta_tab <- data.frame(term = rownames(fit$beta), beta = as.numeric(fit$beta[, 1]), se_model = sqrt(diag(V)),
                         stringsAsFactors = FALSE)
  write_table(beta_tab, run_dir, "beta")
  saveRDS(fit, fit_path(run_dir))
  # events per adjuster: the rarer of events and non-events over the number of adjusters (the usual EPV)
  n_events <- sum(fit$obs)
  n_trials <- if (is.null(trials_col)) length(fit$obs) else sum(d[[fit$char_list$N.char]])
  # the within-patient correlation of standardized residuals: what a working-independence variance ignores
  rho <- NULL
  if (!is.null(patient_col) && length(fit$pred) == nrow(d)) {
    Nr <- if (is.null(trials_col)) rep(1, nrow(d)) else as.numeric(d[[fit$char_list$N.char]])
    pr <- as.numeric(fit$pred)
    e <- (as.numeric(d[[fit$char_list$Y.char]]) - Nr * pr) / sqrt(pmax(Nr * pr * (1 - pr), 1e-12))
    pat <- factor(paste(pid(d[[provider_col]]), pid(d[[patient_col]])))
    rho <- round(within_corr(e, pat), 4)
  }
  list(n_rows_read = n0, n_rows_fitted = nrow(d), n_rows_dropped_incomplete = n0 - nrow(df),
       n_providers_fitted = length(lev), n_providers_excluded = n_excluded, n_patients = sum(sizes$n_patients),
       p = nrow(beta_tab), iterations = fit$control$iter, algorithm = algorithm, tol = tol, bound = bound,
       event_rate = n_events / n_trials, events = n_events,
       events_per_adjuster = min(n_events, n_trials - n_events) / max(nrow(beta_tab), 1),
       auc = if (is.null(fit$AUC) || is.na(fit$AUC)) NULL else fit$AUC,
       gamma_median = stats::median(gam), gamma_quantiles = q5(gam),
       invariant_providers = sum(inv), at_bound_providers = sum(at_bound), within_patient_corr = rho,
       fit_seconds = fit_seconds, seconds = seconds_since(t0),
       tables = list(gamma = table_path(run_dir, "gamma"), beta = table_path(run_dir, "beta")))
})
