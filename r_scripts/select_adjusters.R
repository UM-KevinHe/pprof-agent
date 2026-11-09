# select_adjusters.R -- variable selection with the provider effects kept unpenalized
# (pprof::var_select, lasso or group lasso, and pprof::cv). payload: {data_path, run_dir, spec: {outcome,
# trials, provider, patient, eligibility}, candidates: [...], groups: [...] (optional, same length;
# 0 = unpenalized), nfolds (10), seed (20261018), method: lasso|group}
# writes tables/selection.csv, tables/cv_path.csv; returns the selected set.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprof))
run_step({
  t0 <- Sys.time()
  spec <- PAYLOAD$spec; run_dir <- arg("run_dir")
  outcome_col <- arg("outcome", from = spec); trials_col <- arg("trials", from = spec)
  provider_col <- arg("provider", from = spec); patient_col <- arg("patient", from = spec)
  if (identical(patient_col, "none")) patient_col <- NULL
  eligibility <- as.integer(arg("eligibility", 1, from = spec))
  candidates <- chr(arg("candidates")); groups <- arg("groups")
  method <- arg("method", if (is.null(groups)) "lasso" else "group")
  nfolds <- as.integer(arg("nfolds", 10)); seed <- as.integer(arg("seed", 20261018))
  if (length(candidates) < 2) stop("need at least two candidate adjusters", call. = FALSE)
  need <- unique(c(outcome_col, trials_col, provider_col, patient_col, candidates))
  df <- load_data(arg("data_path"), need)
  df <- df[stats::complete.cases(df[, need, drop = FALSE]), , drop = FALSE]
  df[[provider_col]] <- pid(df[[provider_col]])
  if (!is.null(patient_col)) df[[patient_col]] <- pid(df[[patient_col]])
  for (a in candidates) df[[a]] <- as.numeric(df[[a]])
  # eligibility: providers with fewer than `eligibility` patients (rows when no patient column) are left out
  pd <- pp_data(df, y = outcome_col, x = candidates, provider = provider_col, patient = patient_col, n = trials_col,
                cutoff = eligibility, check = FALSE, message = FALSE)
  if (identical(method, "group")) {
    g <- as.integer(unlist(groups)); if (length(g) != length(candidates)) stop("groups must match candidates", call. = FALSE)
    path <- var_select(pd, penalty = "group_lasso", group = g, threads = threads())
  } else {
    path <- var_select(pd, penalty = "lasso", threads = threads())
  }
  # folds are formed within each provider over its patients (rows without a patient column), so every
  # provider is in the training data of every fold; a provider unseen in a fold is predicted at the median effect
  cvo <- cv(path, n_folds = nfolds, seed = seed)
  b <- coef(path, which = cvo$index_min)$beta
  if (is.null(names(b))) names(b) <- candidates
  b <- b[candidates]
  sel <- data.frame(term = candidates, coef_at_lambda_min = as.numeric(b), selected = as.numeric(b) != 0,
                    group = if (identical(method, "group")) as.integer(unlist(groups)) else NA_integer_, stringsAsFactors = FALSE)
  write_table(sel, run_dir, "selection")
  cv_tab <- data.frame(lambda = cvo$lambda, cve = cvo$cve, cvse = cvo$cvse)
  write_table(cv_tab, run_dir, "cv_path")
  list(method = method, nfolds = nfolds, seed = seed, n_rows = sum(pd$data$included),
       n_providers = sum(pd$sizes$included), n_providers_excluded = sum(!pd$sizes$included),
       folds = "within each provider, patients dealt to the folds in turn",
       lambda_min = cvo$lambda_min, lambda_index = cvo$index_min, n_lambda = length(cvo$lambda),
       n_candidates = length(candidates), n_selected = sum(sel$selected),
       selected = I(sel$term[sel$selected]), dropped = I(sel$term[!sel$selected]),
       seconds = seconds_since(t0), tables = list(selection = table_path(run_dir, "selection"), cv_path = table_path(run_dir, "cv_path")))
})
