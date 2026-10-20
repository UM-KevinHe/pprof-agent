# select_adjusters.R -- variable selection with the provider effects kept unpenalized
# (grplasso::cv.pp.lasso / cv.grp.lasso). payload: {data_path, run_dir, spec: {outcome, provider,
# patient, eligibility}, candidates: [...], groups: [...] (optional, same length; 0 = unpenalized),
# nfolds (10), seed (20261018), method: lasso|group}
# writes tables/selection.csv, tables/cv_path.csv; returns the selected set.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(grplasso))
run_step({
  t0 <- Sys.time()
  spec <- PAYLOAD$spec; run_dir <- arg("run_dir")
  outcome_col <- arg("outcome", from = spec); provider_col <- arg("provider", from = spec); patient_col <- arg("patient", from = spec)
  if (identical(patient_col, "none")) patient_col <- NULL
  eligibility <- as.integer(arg("eligibility", 1, from = spec))
  candidates <- chr(arg("candidates")); groups <- arg("groups")
  method <- arg("method", if (is.null(groups)) "lasso" else "group")
  nfolds <- as.integer(arg("nfolds", 10)); seed <- as.integer(arg("seed", 20261018))
  if (length(candidates) < 2) stop("need at least two candidate adjusters", call. = FALSE)
  need <- unique(c(outcome_col, provider_col, patient_col, candidates))
  df <- load_data(arg("data_path"), need)
  df <- df[stats::complete.cases(df[, need, drop = FALSE]), , drop = FALSE]
  df[[provider_col]] <- pid(df[[provider_col]])
  for (a in candidates) df[[a]] <- as.numeric(df[[a]])
  # eligibility: providers with fewer than `eligibility` patients (rows when no patient column) are left out
  if (!is.null(patient_col)) {
    pp <- unique(df[, c(provider_col, patient_col)]); n_pat <- table(pp[[provider_col]])
  } else n_pat <- table(df[[provider_col]])
  keep_prov <- names(n_pat)[n_pat >= eligibility]
  df <- df[df[[provider_col]] %in% keep_prov, , drop = FALSE]
  df <- as.data.frame(df)
  # a provider whose rows all fall into one held-out fold is unseen in that fold's training data and makes
  # grplasso's prediction fail; so providers with a single row are left out of the selection, and the folds are
  # built per provider (rows ordered by outcome, assigned round-robin from a random start) so every provider with
  # two or more rows is in the training data of every fold
  n_rows_prov <- table(df[[provider_col]])
  df <- df[df[[provider_col]] %in% names(n_rows_prov)[n_rows_prov >= 2], , drop = FALSE]
  n_single <- sum(n_rows_prov < 2)
  set.seed(seed)
  fold <- integer(nrow(df))
  for (pv in split(seq_len(nrow(df)), df[[provider_col]])) {
    o <- pv[order(df[[outcome_col]][pv], stats::runif(length(pv)))]
    fold[o] <- ((seq_along(o) - 1L + sample.int(nfolds, 1L)) %% nfolds) + 1L
  }
  if (identical(method, "group")) {
    g <- as.integer(unlist(groups)); if (length(g) != length(candidates)) stop("groups must match candidates", call. = FALSE)
    cv <- cv.grp.lasso(df, outcome_col, candidates, provider_col, group = g, nfolds = nfolds, seed = seed, fold = fold)
  } else {
    cv <- cv.pp.lasso(df, outcome_col, candidates, provider_col, nfolds = nfolds, seed = seed, fold = fold)
  }
  b <- cv$fit$beta[, cv$min]
  if (is.null(names(b))) names(b) <- candidates
  b <- b[candidates]
  sel <- data.frame(term = candidates, coef_at_lambda_min = as.numeric(b), selected = as.numeric(b) != 0,
                    group = if (identical(method, "group")) as.integer(unlist(groups)) else NA_integer_, stringsAsFactors = FALSE)
  write_table(sel, run_dir, "selection")
  path <- data.frame(lambda = cv$lambda, cve = cv$cve, cvse = cv$cvse)
  write_table(path, run_dir, "cv_path")
  list(method = method, nfolds = nfolds, seed = seed, n_rows = nrow(df), n_providers = length(unique(df[[provider_col]])),
       n_single_row_providers_dropped = n_single, folds = "per provider, rows ordered by outcome, round-robin",
       lambda_min = cv$lambda.min, lambda_index = cv$min, n_lambda = length(cv$lambda),
       n_candidates = length(candidates), n_selected = sum(sel$selected),
       selected = I(sel$term[sel$selected]), dropped = I(sel$term[!sel$selected]),
       seconds = seconds_since(t0), tables = list(selection = table_path(run_dir, "selection"), cv_path = table_path(run_dir, "cv_path")))
})
