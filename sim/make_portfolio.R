#!/usr/bin/env Rscript
# make_portfolio.R -- synthetic measure data shaped like the CMS dialysis measures, for development,
# the calibration simulation and evaluation. No real data is read anywhere in this file.
#
# kinds
#   patient_month  one row per patient-month (SFR / PPPW shape): binary outcome AVF, repeated rows per
#                  patient with a patient random effect, providers of very unequal size, MONTH column
#   discharge      one row per index discharge (SRR shape): outcome READMIT30, a patient contributes one
#                  or more discharges, a HOSP column so the crossed-design refusal can be tested
#
# Rscript make_portfolio.R kind=patient_month n_prov=300 mean_patients=40 months=12 seed=1 out=<dir>
# writes <dir>/data.csv and <dir>/truth.json (the generating values: gamma per provider, beta, the
# planted outliers, column roles). The function is also sourced by the agent's calibration simulation.

make_portfolio <- function(kind = "patient_month", n_prov = 300, mean_patients = 40, months = 12,
                           seed = 1, sigma_patient = 1.0, sigma_prov = 0.4, outlier_frac = 0.03,
                           outlier_shift = 1.0, base_rate = 0.3, beta = NULL) {
  set.seed(seed)
  kind <- match.arg(kind, c("patient_month", "discharge"))
  # provider sizes are heavy-tailed (log-normal), at least one patient
  n_pat <- pmax(1L, as.integer(round(exp(rnorm(n_prov, log(mean_patients), 0.8)))))
  prov_id <- sprintf("F%05d", seq_len(n_prov))
  gamma <- rnorm(n_prov, 0, sigma_prov)
  n_out <- max(1L, as.integer(round(outlier_frac * n_prov)))
  worse <- sample(seq_len(n_prov), n_out)
  better <- sample(setdiff(seq_len(n_prov), worse), n_out)
  # the planted outliers are shifted in the BAD direction of the measure: a higher rate is worse for
  # readmission (discharge kind) and better for fistula use (patient_month kind)
  bad_sign <- if (kind == "patient_month") -1 else 1
  gamma[worse] <- gamma[worse] + bad_sign * outlier_shift
  gamma[better] <- gamma[better] - bad_sign * outlier_shift
  gamma <- gamma - stats::median(gamma)
  total_pat <- sum(n_pat)
  pat_prov <- rep(seq_len(n_prov), n_pat)
  pat_id <- sprintf("K%07d", seq_len(total_pat))
  if (kind == "patient_month") {
    m <- pmin(as.integer(months), 1L + stats::rgeom(total_pat, prob = 1 / months))
  } else {
    m <- 1L + stats::rgeom(total_pat, prob = 0.6)
  }
  n <- sum(m)
  row_pat <- rep(seq_len(total_pat), m)
  row_prov <- pat_prov[row_pat]
  # patient-level covariates, constant within patient
  age <- round(stats::rnorm(total_pat, 62, 13))
  female <- stats::rbinom(total_pat, 1, 0.45)
  diabetes <- stats::rbinom(total_pat, 1, 0.45)
  chf <- stats::rbinom(total_pat, 1, 0.30)
  copd <- stats::rbinom(total_pat, 1, 0.15)
  bmi <- round(stats::rnorm(total_pat, 28, 6), 1)
  yrs <- round(stats::rexp(total_pat, 1 / 4), 2)
  nh <- stats::rbinom(total_pat, 1, 0.08)
  # row-level covariates
  alb <- round(stats::rnorm(n, 3.8, 0.5), 2)
  hgb <- round(stats::rnorm(n, 11, 1.2), 1)
  Z <- data.frame(AGE = age[row_pat], FEMALE = female[row_pat], DIABETES = diabetes[row_pat],
                  CHF = chf[row_pat], COPD = copd[row_pat], BMI = bmi[row_pat], YRS_ESRD = yrs[row_pat],
                  NH = nh[row_pat], ALB = alb, HGB = hgb)
  if (is.null(beta)) {
    beta <- c(AGE = 0.02, FEMALE = -0.2, DIABETES = 0.4, CHF = 0.5, COPD = 0, BMI = 0.01,
              YRS_ESRD = -0.08, NH = 0.6, ALB = 0, HGB = 0)
  }
  beta <- beta[names(Z)]
  Zc <- scale(as.matrix(Z), center = TRUE, scale = FALSE)
  b <- stats::rnorm(total_pat, 0, sigma_patient)
  eta <- stats::qlogis(base_rate) + gamma[row_prov] + as.numeric(Zc %*% beta) + b[row_pat]
  Y <- stats::rbinom(n, 1, stats::plogis(eta))
  outcome <- if (kind == "patient_month") "AVF" else "READMIT30"
  d <- data.frame(PROVFS = prov_id[row_prov], KECC_ID = pat_id[row_pat], stringsAsFactors = FALSE)
  d[[outcome]] <- Y
  d <- cbind(d, Z)
  if (kind == "patient_month") {
    d$MONTH <- unlist(lapply(m, seq_len))
  } else {
    d$HOSP <- sprintf("H%04d", sample(seq_len(max(2L, n_prov %/% 3)), n, replace = TRUE))
    d$DISCHARGE_ID <- seq_len(n)
  }
  truth <- list(kind = kind, seed = seed, n_prov = n_prov, n_patients = total_pat, n_rows = n,
                outcome = outcome, provider = "PROVFS", patient = "KECC_ID", covariates = names(Z),
                event_direction = if (kind == "patient_month") "higher_is_better" else "higher_is_worse",
                unit = kind, base_rate = base_rate, sigma_patient = sigma_patient, sigma_prov = sigma_prov,
                gamma = as.list(stats::setNames(gamma, prov_id)), beta = as.list(beta),
                worse = prov_id[worse], better = prov_id[better])
  list(data = d, truth = truth)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  for (a in args) { p <- strsplit(a, "=", fixed = TRUE)[[1]]; kv[[p[1]]] <- paste(p[-1], collapse = "=") }
  num <- function(k, d) if (is.null(kv[[k]])) d else as.numeric(kv[[k]])
  out <- if (is.null(kv$out)) stop("out=<dir> is required") else kv$out
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  res <- make_portfolio(kind = if (is.null(kv$kind)) "patient_month" else kv$kind,
                        n_prov = num("n_prov", 300), mean_patients = num("mean_patients", 40),
                        months = num("months", 12), seed = num("seed", 1),
                        sigma_patient = num("sigma_patient", 1.0), sigma_prov = num("sigma_prov", 0.4),
                        outlier_frac = num("outlier_frac", 0.03), outlier_shift = num("outlier_shift", 1.0),
                        base_rate = num("base_rate", 0.3))
  data.table::fwrite(res$data, file.path(out, "data.csv"))
  jsonlite::write_json(res$truth, file.path(out, "truth.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat(sprintf("%s: %d rows, %d patients, %d providers, event rate %.3f -> %s\n", res$truth$kind,
              res$truth$n_rows, res$truth$n_patients, res$truth$n_prov, mean(res$data[[res$truth$outcome]]), out))
}
