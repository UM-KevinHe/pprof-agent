# verify_spec.R -- the consequence card: what a specification implies on this file, as numbers the
# analyst can falsify. payload: {data_path, spec: {outcome, trials, provider, patient, adjusters,
# eligibility, crossed_with}}. Does not fit anything.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
run_step({
  t0 <- Sys.time()
  spec <- PAYLOAD$spec
  outcome_col <- arg("outcome", from = spec); trials_col <- arg("trials", from = spec)
  provider_col <- arg("provider", from = spec); patient_col <- arg("patient", from = spec)
  if (identical(patient_col, "none")) patient_col <- NULL
  adjusters <- chr(arg("adjusters", from = spec))
  eligibility <- as.integer(arg("eligibility", 1, from = spec))
  crossed <- arg("crossed_with", "none", from = spec)
  need <- unique(c(outcome_col, trials_col, provider_col, patient_col, adjusters, if (!identical(crossed, "none")) crossed))
  df <- load_data(arg("data_path"))
  missing_cols <- setdiff(need, names(df))
  if (length(missing_cols)) stop("columns not in the file: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  dt <- as.data.table(df[, need, drop = FALSE])
  n <- nrow(dt)
  y <- dt[[outcome_col]]
  problems <- list()
  if (!is.numeric(y)) problems[[length(problems) + 1]] <- list(field = "outcome", problem = "not numeric")
  yvals <- sort(unique(y[!is.na(y)]))
  outcome_is_binary <- is.numeric(y) && all(yvals %in% c(0, 1))
  trials_ok <- TRUE
  if (!is.null(trials_col)) {
    tr <- dt[[trials_col]]
    trials_ok <- is.numeric(tr) && all(tr[!is.na(tr)] >= 1) && all(y[!is.na(y) & !is.na(tr)] <= tr[!is.na(y) & !is.na(tr)])
    if (!trials_ok) problems[[length(problems) + 1]] <- list(field = "trials", problem = "trials must be >= 1 and >= the outcome count")
  } else if (!outcome_is_binary) {
    problems[[length(problems) + 1]] <- list(field = "outcome", problem = paste0("not 0/1 (values seen: ", paste(head(yvals, 6), collapse = ", "), "); a count outcome needs its trials column declared (the denominator)"))
  }
  if (length(yvals) < 2) problems[[length(problems) + 1]] <- list(field = "outcome", problem = "constant")
  adj_problems <- list()
  for (a in adjusters) {
    x <- dt[[a]]
    if (!is.numeric(x) && !is.logical(x)) adj_problems[[length(adj_problems) + 1]] <- list(name = a, problem = "not numeric")
    else if (length(unique(x[!is.na(x)])) <= 1) adj_problems[[length(adj_problems) + 1]] <- list(name = a, problem = "constant")
    else if (mean(is.na(x)) > 0.5) adj_problems[[length(adj_problems) + 1]] <- list(name = a, problem = sprintf("%.0f%% missing", 100 * mean(is.na(x))))
  }
  keep <- stats::complete.cases(as.data.frame(dt)[, c(outcome_col, trials_col, provider_col, adjusters), drop = FALSE])
  dc <- dt[keep]
  n_rows_prov <- dc[, .N, by = c(provider_col)]
  if (!is.null(patient_col)) {
    rpp <- dc[, .N, by = c(provider_col, patient_col)]
    n_pat_prov <- rpp[, .N, by = c(provider_col)]
    rows_per_patient <- rpp$N
    n_patients <- nrow(unique(dc[, c(patient_col), with = FALSE]))
  } else {
    n_pat_prov <- n_rows_prov
    rows_per_patient <- rep(1L, nrow(dc))
    n_patients <- nrow(dc)
  }
  setnames(n_pat_prov, c("provider", "n_patients"))
  setnames(n_rows_prov, c("provider", "n_rows"))
  sizes <- merge(n_pat_prov, n_rows_prov, by = "provider")
  eligible <- sizes$n_patients >= eligibility
  elig_ids <- sizes$provider[eligible]
  de <- dc[pid(dc[[provider_col]]) %in% elig_ids]
  ev <- de[, .(events = sum(get(outcome_col)), ntrials = if (is.null(trials_col)) .N else sum(get(trials_col))), by = c(provider_col)]
  invariant_none <- sum(ev$events == 0); invariant_all <- sum(ev$events == ev$ntrials)
  res <- list(
    rows = n, rows_complete = nrow(dc), rows_dropped_incomplete = n - nrow(dc),
    patients = n_patients, providers = nrow(sizes), eligible_providers = sum(eligible),
    excluded_providers = sum(!eligible), eligibility = eligibility,
    events = sum(de[[outcome_col]]), trials_total = if (is.null(trials_col)) nrow(de) else sum(de[[trials_col]]),
    event_rate = if (is.null(trials_col)) mean(de[[outcome_col]]) else sum(de[[outcome_col]]) / sum(de[[trials_col]]),
    rows_per_patient = q5(rows_per_patient), max_rows_per_patient = max(rows_per_patient),
    repeats = any(rows_per_patient > 1), has_patient = !is.null(patient_col),
    patients_per_provider = q5(sizes$n_patients[eligible]), rows_per_provider = q5(sizes$n_rows[eligible]),
    trials_per_provider = q5(ev$ntrials),
    invariant_providers = list(none = invariant_none, all = invariant_all, total = invariant_none + invariant_all),
    small_providers = list(le_3 = sum(sizes$n_patients[eligible] <= 3), le_10 = sum(sizes$n_patients[eligible] <= 10)),
    outcome_values = I(as.character(head(yvals, 12))), outcome_is_binary = outcome_is_binary, trials_ok = trials_ok,
    problems = problems, adjuster_problems = adj_problems,
    hospitals = if (!identical(crossed, "none")) length(unique(dc[[crossed]])) else NULL,
    seconds = seconds_since(t0))
  res
})
