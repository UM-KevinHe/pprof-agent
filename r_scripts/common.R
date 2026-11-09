# common.R -- shared by every script in r_scripts/: the JSON handshake with the Python harness, data
# loading, thread pinning, and the run-directory conventions. Each script is called as
#     Rscript <script>.R <payload.json> <out.json>
# and must write exactly one JSON object to <out.json>: {"status": "ok", ...} or
# {"status": "error", "class": ..., "message": ...}. Nothing is printed for the harness to parse.
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

.args <- commandArgs(trailingOnly = TRUE)
if (length(.args) < 2) stop("usage: Rscript <script>.R <payload.json> <out.json>", call. = FALSE)
PAYLOAD <- jsonlite::fromJSON(.args[1], simplifyVector = TRUE)
OUT_PATH <- .args[2]

write_out <- function(x) {
  jsonlite::write_json(x, OUT_PATH, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
  invisible(x)
}

fail <- function(message, class = "error") {
  write_out(list(status = "error", class = class, message = paste(message, collapse = " ")))
  quit(save = "no", status = 1)
}

# Run the body of a script: errors become a JSON error object, a result becomes {"status": "ok", ...}.
run_step <- function(expr) {
  res <- tryCatch(expr, error = function(e) fail(conditionMessage(e), class(e)[1]))
  if (!is.list(res)) res <- list(value = res)
  if (is.null(res$status)) res$status <- "ok"
  write_out(res)
}

# a payload field, with a default when absent or null
arg <- function(name, default = NULL, from = PAYLOAD) {
  v <- from[[name]]
  if (is.null(v) || (length(v) == 1 && is.na(v) && !is.character(v))) default else v
}

# character vector from a JSON array that jsonlite may have simplified to a scalar or an empty list
chr <- function(x) if (is.null(x) || length(x) == 0) character(0) else as.character(unlist(x))

threads <- function() {
  t <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
  if (is.na(t) || t < 1) 1L else t
}

load_data <- function(path, columns = NULL) {
  columns <- if (is.null(columns)) NULL else unique(chr(columns))
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
    df <- as.data.frame(arrow::read_parquet(path, col_select = if (is.null(columns)) NULL else dplyr::all_of(columns)))
  } else {
    df <- data.table::fread(path, select = columns, na.strings = c("", "NA", "."), data.table = FALSE)
  }
  df
}

fit_path <- function(run_dir) file.path(run_dir, "fit.rds")
tables_dir <- function(run_dir) {
  d <- file.path(run_dir, "tables")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}
table_path <- function(run_dir, name) file.path(tables_dir(run_dir), paste0(name, ".csv"))
write_table <- function(df, run_dir, name) {
  p <- table_path(run_dir, name)
  data.table::fwrite(df, p)
  p
}
read_table <- function(run_dir, name) data.table::fread(table_path(run_dir, name), data.table = FALSE)

# five-number summary as a named list (JSON object)
q5 <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(list(min = NULL, q1 = NULL, median = NULL, q3 = NULL, max = NULL, n = 0L))
  q <- stats::quantile(x, c(0, .25, .5, .75, 1), names = FALSE)
  list(min = q[1], q1 = q[2], median = q[3], q3 = q[4], max = q[5], n = length(x))
}

# provider ids as character, always
pid <- function(x) as.character(x)

# the fitted model: a `logistic_fe` object of the pprof R package (2.0 or later), and accessors for
# what the scripts read from it
load_fit <- function(run_dir) {
  p <- fit_path(run_dir)
  if (!file.exists(p)) stop("no fitted model in the run directory (run `fit` first)", call. = FALSE)
  readRDS(p)
}
fit_levels <- function(fit) names(fit$coefficients$gamma)
fit_gamma <- function(fit) as.numeric(fit$coefficients$gamma)
fit_beta <- function(fit) fit$coefficients$beta
fit_Y <- function(fit) as.numeric(fit$data[[fit$vars$y]])
fit_N <- function(fit) if (is.null(fit$vars$n)) rep(1, nrow(fit$data)) else as.numeric(fit$data[[fit$vars$n]])
fit_prov <- function(fit) pid(fit$data[[fit$vars$provider]])
fit_pat <- function(fit) if (is.null(fit$vars$patient)) NULL else pid(fit$data[[fit$vars$patient]])
fit_Z <- function(fit) { z <- as.matrix(fit$data[, fit$vars$x, drop = FALSE]); storage.mode(z) <- "double"; z }
fit_pred <- function(fit) as.numeric(fit$fitted)
fit_lp <- function(fit) as.numeric(fit$linear_predictor)

# rows and patients per fitted provider, in the order of the provider effects
provider_sizes <- function(fit) {
  s <- fit$sizes
  n_pat <- if (all(is.na(s$patients))) s$rows else s$patients
  data.frame(provider = pid(s$provider), n_rows = as.integer(s$rows), n_patients = as.integer(n_pat),
             stringsAsFactors = FALSE)
}

# the agent's test names -> the arguments of pprof::test(); the naive Wald test uses pprof's case-mix
# variance, which does not depend on where the covariates are centred
TEST_ARGS <- list(
  exact = list(statistic = "exact", variance = "naive", exact_method = "poibin"),
  exact_bootstrap = list(statistic = "exact", variance = "naive", exact_method = "bootstrap"),
  score = list(statistic = "score", variance = "naive"),
  wald = list(statistic = "wald", variance = "naive"),
  robust_wald = list(statistic = "wald", variance = "robust"),
  robust_score = list(statistic = "score", variance = "robust"),
  stabilized_wald = list(statistic = "wald", variance = "stabilized"),
  stabilized_score = list(statistic = "score", variance = "stabilized"))

# one provider test; a data frame in the order of the provider effects: provider, stat, p, flag (1 above
# the null, -1 below, 0 as expected, NA not testable), se (Wald-type tests only)
run_test <- function(fit, test, null = "median", level = 0.95, alternative = "two.sided", n_resample = 10000) {
  a <- TEST_ARGS[[test]]
  if (is.null(a)) stop("unknown test: ", test, call. = FALSE)
  r <- do.call(pprof::test, c(list(fit, null = null, level = level, alternative = alternative,
                                   n_resample = as.integer(n_resample)), a))
  data.frame(provider = pid(r$provider), stat = as.numeric(r$statistic), p = as.numeric(r$p),
             flag = as.integer(r$flag), se = as.numeric(r$se), stringsAsFactors = FALSE)
}

# the size-stratified empirical null (pprof::null_model): providers ordered by size and cut into `ngroup`
# equal groups; within each, a Huber M-estimate of location and a MAD scale define the null, and the
# statistic is re-centred and re-scaled. Non-finite statistics are left out of the fit and get NA.
group_en <- function(z, size, ngroup = 4) {
  nm <- pprof::null_model("empirical", z = z, size = size, n_groups = ngroup, grouping = "cut",
                          estimator = "huber", fit_mask = is.finite(z))
  dg <- nm$diagnostics
  tab <- data.frame(group = dg$group, n = dg$n_fitted, min_size = dg$min_size, median_size = dg$median_size,
                    max_size = dg$max_size, location = dg$null_mean, scale = dg$null_sd, mean = dg$mean_z,
                    sd = dg$sd_z)
  list(z_en = as.numeric(pprof::adjust_z(z, nm)), group_idx = as.integer(nm$group), group_inf = tab, null = nm)
}

seconds_since <- function(t0) as.numeric(Sys.time() - t0, units = "secs")

# the within-patient correlation of standardized residuals: over patients with >= 2 rows, the mean pairwise
# product (the fit reports it; the calibration simulation matches its patient random effect to it)
within_corr <- function(e, pat) {
  s <- rowsum(e, pat); s2 <- rowsum(e^2, pat); n <- as.numeric(table(pat)[rownames(s)])
  k <- n >= 2
  if (!any(k)) return(0)
  pairs <- (s[k]^2 - s2[k]) / 2
  sum(pairs) / sum(n[k] * (n[k] - 1) / 2)
}
