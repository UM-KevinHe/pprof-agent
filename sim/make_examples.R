#!/usr/bin/env Rscript
# make_examples.R -- the example portfolio: four synthetic measure files with planted truth, for the
# end-to-end tests, the evaluation and the public demo. No real data is read anywhere in this file.
#
#   sfr_2024, sfr_2025   patient-month rows, outcome AVF (1 = fistula in use; higher is better); the same
#                        facilities two years running, 70% of patients carried over; in 2025 one facility
#                        becomes truly worse and another's case mix deteriorates with its effect unchanged
#   srr_2025             index-discharge rows, outcome READMIT30 (higher is worse); a HOSPITAL column with
#                        hospital effects in the generating model (the crossed design V1 refuses)
#   aoh_2025             patient-month rows, binomial outcome HOSP_DAYS of DAYS_AT_RISK (higher is worse)
#
# Planted in every file (recorded in truth.json): six facilities truly worse (two per size tertile, the
# effect set outright, larger where the facility is smaller), four truly better, two facilities whose
# outcomes are all zero (invariant; not in the binomial file), fifteen facilities with three patients or
# fewer, one "decoy" with an extreme case mix and a true effect of exactly zero, and a patient random effect
# (within-patient correlation). Ordinary facilities have a small constant spread of effects: this is demo
# data for the agent, not a study of the reference distribution.
#
#   Rscript make_examples.R out=<dir> [which=all|sfr|srr|aoh] [seed=2026]

simulate_measure <- function(kind = c("patient_month", "discharge", "binomial_days", "demo"), n_prov = 300, seed = 2026,
                             target_rate = 0.63, sigma_patient = 1.0, months = 12, state = NULL,
                             year = 2025, year2_changes = FALSE, median_patients = 25) {
  kind <- match.arg(kind)
  set.seed(seed)
  bad_sign <- if (kind == "patient_month") -1 else 1   # the direction of a WORSE facility on the logit scale
  prov_id <- sprintf("F%05d", seq_len(n_prov))

  # ---- facilities: sizes, effects, planted structure (reused from `state` for a second year) ----
  if (is.null(state)) {
    n_pat <- pmin(300L, pmax(4L, as.integer(round(exp(rnorm(n_prov, log(median_patients), 0.9))))))
    tiny <- sort(sample(seq_len(n_prov), 15))
    n_pat[tiny] <- sample(1:3, length(tiny), replace = TRUE)
    # the invariant facilities are large enough to be tested (eligibility 11), so the Wald rule is exercised
    small_pool <- setdiff(which(n_pat >= 11 & n_pat <= 14), tiny)
    # no invariant facility for the binomial-days file: zero hospital days over thousands of patient-days is
    # not a plausible facility, and with invariant facilities only the score-type tests may be reported, which
    # are very conservative for binomial trials (type I error far below nominal in the calibration)
    invariant <- if (kind == "binomial_days") integer(0) else sort(sample(small_pool, 2))
    # planted facilities are drawn from those a published eligibility threshold (11 patients) would test
    ordinary <- setdiff(which(n_pat >= 12), c(tiny, invariant))
    # ordinary facilities: a small constant spread of effects
    sd_gamma <- rep(0.10, n_prov)
    gamma <- rnorm(n_prov, 0, sd_gamma)
    tert <- cut(rank(n_pat[ordinary], ties.method = "first"), 3, labels = c("small", "medium", "large"))
    pick <- function(lab, k) sample(ordinary[tert == lab], k)
    worse <- c(pick("large", 2), pick("medium", 2), pick("small", 2))
    rest <- setdiff(ordinary, worse)
    tert_rest <- tert[match(rest, ordinary)]
    better <- c(sample(rest[tert_rest == "large"], 1), sample(rest[tert_rest == "medium"], 2), sample(rest[tert_rest == "small"], 1))
    shift_of <- if (kind == "demo") c(large = 1.0, medium = 1.3, small = 1.7) else c(large = 0.8, medium = 1.0, small = 1.4)
    size_class <- as.character(tert[match(seq_len(n_prov), ordinary)])
    size_class[is.na(size_class)] <- ifelse(seq_len(n_prov) %in% tiny, "tiny", "below_threshold")
    # the planted effects are set outright (not added to a random draw), so each is a known departure
    gamma[worse] <- bad_sign * shift_of[size_class[worse]]
    gamma[better] <- -bad_sign * shift_of[size_class[better]]
    decoy <- sample(setdiff(rest[tert_rest == "medium"], better), 1)
    gamma[decoy] <- 0
    # facility-level case-mix offsets (what makes a pooled model and a fixed-effects model differ)
    nh_off <- rnorm(n_prov, 0, 0.5); alb_off <- rnorm(n_prov, 0, 0.15); age_off <- rnorm(n_prov, 0, 3)
    nh_off[decoy] <- 2.0; alb_off[decoy] <- -0.6
    hosp <- NULL
    if (kind == "discharge") {
      n_hosp <- max(2L, n_prov %/% 3)
      home <- sample(seq_len(n_hosp), n_prov, replace = TRUE)
      hosp_eff <- rnorm(n_hosp, 0, 0.15)
      bad_hosp <- sample(seq_len(n_hosp), 2)
      hosp_eff[bad_hosp] <- hosp_eff[bad_hosp] + 0.5
      hosp <- list(n_hosp = n_hosp, home = home, eff = hosp_eff, bad = bad_hosp)
    }
    state <- list(n_pat = n_pat, tiny = tiny, invariant = invariant, gamma = gamma, sd_gamma = sd_gamma,
                  worse = worse, better = better, decoy = decoy, size_class = size_class, shift_of = shift_of,
                  nh_off = nh_off, alb_off = alb_off, age_off = age_off, hosp = hosp, patients = NULL)
    changes <- list()
  } else {
    changes <- list()
  }
  n_pat <- state$n_pat; gamma <- state$gamma
  nh_off <- state$nh_off; alb_off <- state$alb_off; age_off <- state$age_off
  if (year2_changes) {
    # one ordinary medium facility becomes truly worse; another's case mix deteriorates, effect unchanged
    cand <- setdiff(which(state$size_class == "medium"), c(state$worse, state$better, state$decoy, state$tiny, state$invariant))
    truly_worse <- sample(cand, 1)
    casemix_worse <- sample(setdiff(cand, truly_worse), 1)
    gamma[truly_worse] <- bad_sign * 1.0
    nh_off[casemix_worse] <- nh_off[casemix_worse] + 2.0
    alb_off[casemix_worse] <- alb_off[casemix_worse] - 0.5
    changes <- list(truly_worse = prov_id[truly_worse], truly_worse_effect = bad_sign * 1.0,
                    casemix_worse = prov_id[casemix_worse], casemix_change = "NH more frequent, ALB lower")
  }
  gamma_med <- stats::median(gamma[-c(state$tiny)])

  # ---- patients: carried over from the previous year where a state holds them, new otherwise ----
  total_pat <- sum(n_pat)
  pat_prov <- rep(seq_len(n_prov), n_pat)
  new_patient <- function(idx) {
    k <- length(idx)
    data.frame(prov = pat_prov[idx],
               age = round(stats::rnorm(k, 63 + age_off[pat_prov[idx]], 13)),
               female = stats::rbinom(k, 1, 0.44), diabetes = stats::rbinom(k, 1, 0.58), chf = stats::rbinom(k, 1, 0.30),
               copd = stats::rbinom(k, 1, 0.14), bmi = round(pmax(15, stats::rnorm(k, 28, 6.5)), 1),
               yrs = round(stats::rexp(k, 1 / 4), 2),
               nh = stats::rbinom(k, 1, stats::plogis(stats::qlogis(0.06) + nh_off[pat_prov[idx]])),
               alb_mean = round(stats::rnorm(k, 3.8 + alb_off[pat_prov[idx]], 0.40), 2),
               hgb_mean = round(stats::rnorm(k, 11.0, 1.0), 1),
               b = stats::rnorm(k, 0, sigma_patient), stringsAsFactors = FALSE)
  }
  if (is.null(state$patients)) {
    P <- new_patient(seq_len(total_pat))
    P$id <- sprintf("K%07d", seq_len(total_pat))
    carried <- rep(FALSE, total_pat)
  } else {
    prev <- state$patients
    keep <- stats::rbinom(nrow(prev), 1, 0.7) == 1
    P_old <- prev[keep, ]
    P_old$age <- P_old$age + 1L
    P_old$yrs <- P_old$yrs + 1
    # the facility's patient count is kept: new patients fill the places of those who left
    need <- n_pat - as.integer(table(factor(P_old$prov, levels = seq_len(n_prov))))
    need <- pmax(need, 0L)
    idx_new <- rep(seq_len(n_prov), need)
    pat_prov <- idx_new
    P_new <- new_patient(seq_along(idx_new))
    P_new$id <- sprintf("K%07d", state$next_id - 1L + seq_len(nrow(P_new)))
    P <- rbind(P_old[, names(P_new)], P_new)
    carried <- c(rep(TRUE, nrow(P_old)), rep(FALSE, nrow(P_new)))
    o <- order(P$prov, P$id); P <- P[o, ]; carried <- carried[o]
    total_pat <- nrow(P)
  }
  rownames(P) <- NULL

  # ---- rows ----
  if (kind == "discharge") {
    m <- 1L + stats::rgeom(total_pat, prob = 0.55)
  } else {
    m <- pmin(as.integer(months), 1L + stats::rgeom(total_pat, prob = 1 / 8))   # attrition: mean about eight months
  }
  n <- sum(m)
  row_pat <- rep(seq_len(total_pat), m)
  row_prov <- P$prov[row_pat]
  Z <- data.frame(AGE = P$age[row_pat], FEMALE = P$female[row_pat], DIABETES = P$diabetes[row_pat], CHF = P$chf[row_pat],
                  COPD = P$copd[row_pat], BMI = P$bmi[row_pat], YRS_ESRD = P$yrs[row_pat], NH = P$nh[row_pat],
                  ALB = round(P$alb_mean[row_pat] + stats::rnorm(n, 0, 0.25), 2),
                  HGB = round(P$hgb_mean[row_pat] + stats::rnorm(n, 0, 0.7), 1))
  if (kind == "patient_month") {
    beta <- c(AGE = -0.015, FEMALE = -0.35, DIABETES = -0.25, CHF = -0.20, COPD = -0.10, BMI = -0.02,
              YRS_ESRD = 0.08, NH = -0.70, ALB = 0.40, HGB = 0.05)
  } else if (kind == "discharge") {
    Z$LOS <- 1L + stats::rpois(n, 4)
    Z$PRIOR_ADMITS <- stats::rpois(n, 0.8)
    dx <- sample(c("CARDIAC", "INFECTION", "ACCESS", "OTHER"), n, replace = TRUE, prob = c(0.25, 0.20, 0.20, 0.35))
    Z$DX_CARDIAC <- as.integer(dx == "CARDIAC"); Z$DX_INFECTION <- as.integer(dx == "INFECTION"); Z$DX_ACCESS <- as.integer(dx == "ACCESS")
    beta <- c(AGE = 0.010, FEMALE = 0.05, DIABETES = 0.20, CHF = 0.45, COPD = 0.30, BMI = -0.01, YRS_ESRD = 0.03,
              NH = 0.50, ALB = -0.50, HGB = -0.10, LOS = 0.04, PRIOR_ADMITS = 0.25, DX_CARDIAC = 0.30, DX_INFECTION = 0.40, DX_ACCESS = 0.10)
  } else {
    beta <- c(AGE = 0.012, FEMALE = 0.05, DIABETES = 0.20, CHF = 0.50, COPD = 0.30, BMI = -0.01, YRS_ESRD = 0.03,
              NH = 0.60, ALB = -0.55, HGB = -0.12)
  }
  beta <- beta[names(Z)]
  Zc <- scale(as.matrix(Z), center = TRUE, scale = FALSE)
  eta0 <- (gamma[row_prov] - gamma_med) + as.numeric(Zc %*% beta) + P$b[row_pat]
  hosp_col <- NULL
  if (kind == "discharge") {
    h <- state$hosp
    home <- h$home[row_prov]
    hosp_row <- ifelse(stats::runif(n) < 0.7, home, sample(seq_len(h$n_hosp), n, replace = TRUE))
    eta0 <- eta0 + h$eff[hosp_row]
    hosp_col <- sprintf("H%04d", hosp_row)
  }
  # the intercept that gives the target marginal rate
  alpha <- stats::uniroot(function(a) mean(stats::plogis(a + eta0)) - target_rate, c(-10, 10))$root
  p <- stats::plogis(alpha + eta0)
  if (kind == "binomial_days") {
    last <- !duplicated(row_pat, fromLast = TRUE)
    days <- sample(28:31, n, replace = TRUE)
    days[last & m[row_pat] < months] <- sample(5:27, sum(last & m[row_pat] < months), replace = TRUE)
    Y <- stats::rbinom(n, days, p)
  } else {
    days <- NULL
    Y <- stats::rbinom(n, 1, p)
  }
  if (length(state$invariant)) Y[row_prov %in% state$invariant] <- 0L

  outcome <- switch(kind, patient_month = "AVF", discharge = "READMIT30", binomial_days = "HOSP_DAYS", demo = "HOSPITALIZED")
  d <- data.frame(PROVFS = prov_id[row_prov], KECC_ID = P$id[row_pat], stringsAsFactors = FALSE)
  d[[outcome]] <- Y
  if (kind == "binomial_days") d$DAYS_AT_RISK <- days
  d <- cbind(d, Z)
  if (kind == "discharge") {
    d$HOSPITAL <- hosp_col
    d$DISCHARGE_ID <- sprintf("D%07d", seq_len(n))
    d$ADMIT_MONTH <- sample(1:12, n, replace = TRUE)
  } else {
    d$MONTH <- unlist(lapply(m, seq_len))
  }
  d$YEAR <- as.integer(year)

  tab <- function(idx) prov_id[idx]
  truth <- list(
    kind = kind, year = year, seed = seed, n_prov = n_prov, n_patients = total_pat, n_rows = n,
    outcome = outcome, trials = if (kind == "binomial_days") "DAYS_AT_RISK" else NULL,
    provider = "PROVFS", patient = "KECC_ID", covariates = names(Z),
    event_direction = if (kind == "patient_month") "higher_is_better" else "higher_is_worse",
    unit = if (kind == "discharge") "index_discharge" else "patient_month",
    target_rate = target_rate, event_rate = if (kind == "binomial_days") sum(Y) / sum(days) else mean(Y),
    sigma_patient = sigma_patient, intercept = alpha,
    gamma = as.list(stats::setNames(round(gamma - gamma_med, 4), prov_id)), beta = as.list(beta),
    sd_gamma_ordinary = state$sd_gamma[1],
    patients_per_facility = as.list(stats::setNames(as.integer(table(factor(P$prov, levels = seq_len(n_prov)))), prov_id)),
    worse = tab(state$worse), worse_size_class = as.list(stats::setNames(state$size_class[state$worse], tab(state$worse))),
    better = tab(state$better), better_size_class = as.list(stats::setNames(state$size_class[state$better], tab(state$better))),
    shift_by_size_class = as.list(state$shift_of), bad_direction_sign = bad_sign,
    invariant = tab(state$invariant), tiny = tab(state$tiny), decoy = tab(state$decoy),
    decoy_note = "an extreme case mix (NH frequent, ALB low) with a true effect of exactly zero: a pooled model flags it, a fixed-effects model does not",
    carried_over_patients = sum(carried), year2_changes = if (length(changes)) changes else NULL,
    hospitals = if (kind == "discharge") list(n = state$hosp$n_hosp, effect_sd = 0.15, bad = sprintf("H%04d", state$hosp$bad), bad_shift = 0.5,
                                              home_share = 0.7, facilities_with_a_bad_home_hospital = tab(which(state$hosp$home %in% state$hosp$bad))) else NULL
  )
  if (kind == "demo") {
    # the demo data set reads as a plain quality measure: monthly hospitalization, plain column names
    ren <- c(PROVFS = "FACILITY", KECC_ID = "PATIENT", CHF = "HEART_FAILURE", YRS_ESRD = "YEARS_ON_DIALYSIS", NH = "NURSING_HOME",
             ALB = "ALBUMIN", HGB = "HEMOGLOBIN")
    nm <- names(d); hit <- nm %in% names(ren); nm[hit] <- ren[nm[hit]]; names(d) <- nm
    truth$provider <- "FACILITY"; truth$patient <- "PATIENT"
    cv <- truth$covariates; hit <- cv %in% names(ren); cv[hit] <- ren[cv[hit]]; truth$covariates <- cv
    names(truth$beta) <- cv
    truth$decoy_note <- "an extreme case mix (nursing-home residents frequent, albumin low) with a true effect of exactly zero: a pooled model flags it, a fixed-effects model does not"
  }
  state$patients <- P
  state$next_id <- max(as.integer(sub("K", "", P$id))) + 1L
  list(data = d, truth = truth, state = state)
}

write_example <- function(res, out) {
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  data.table::fwrite(res$data, file.path(out, "data.csv"))
  jsonlite::write_json(res$truth, file.path(out, "truth.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")
  t <- res$truth
  cat(sprintf("%-10s %s: %6d rows %5d patients %3d facilities, event rate %.3f; worse %s; better %s\n",
              basename(out), t$kind, t$n_rows, t$n_patients, t$n_prov, t$event_rate,
              paste(t$worse, collapse = " "), paste(t$better, collapse = " ")))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  for (a in args) { p <- strsplit(a, "=", fixed = TRUE)[[1]]; kv[[p[1]]] <- paste(p[-1], collapse = "=") }
  out <- if (is.null(kv$out)) stop("out=<dir> is required") else kv$out
  which <- if (is.null(kv$which)) "all" else kv$which
  seed <- if (is.null(kv$seed)) 2026 else as.integer(kv$seed)
  if (which %in% c("all", "sfr")) {
    y1 <- simulate_measure("patient_month", n_prov = 300, seed = seed, target_rate = 0.63, year = 2024)
    write_example(y1, file.path(out, "sfr_2024"))
    y2 <- simulate_measure("patient_month", n_prov = 300, seed = seed + 1, target_rate = 0.63, state = y1$state, year = 2025, year2_changes = TRUE)
    write_example(y2, file.path(out, "sfr_2025"))
  }
  if (which %in% c("all", "srr")) write_example(simulate_measure("discharge", n_prov = 250, seed = seed + 2, target_rate = 0.26, year = 2025, median_patients = 40), file.path(out, "srr_2025"))
  if (which %in% c("all", "aoh")) write_example(simulate_measure("binomial_days", n_prov = 250, seed = seed + 3, target_rate = 0.04, year = 2025), file.path(out, "aoh_2025"))
  if (which == "demo") write_example(simulate_measure("demo", n_prov = 200, seed = seed + 4, target_rate = 0.15, year = 2025, median_patients = 25), file.path(out, "demo"))
}
