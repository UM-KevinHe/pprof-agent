# simulate.R -- one data set from a training design: the generator behind the agent's training tasks (SFT,
# DPO) and behind the operating characteristics that score them (oc.R). No real data is read anywhere.
#
# A design (designs.py) fixes the structure: the measure kind (patient-month rows, index discharges, or
# binomial days at risk), the number of providers and their sizes, the repeats per patient and the
# within-patient correlation (a patient random effect on the logit scale), the event rate, the ordinary
# provider-to-provider spread `tau`, the planted outliers, a case mix that differs by provider, providers
# whose outcomes are all zero, and the eligibility threshold the analysis applies.
#
# Outliers follow the project's definition (Yubo, 2026-10-28, "definition B"): a provider is unusual when it
# departs beyond the normal spread among providers. So the planted effects are set outright at +/- shift with
# shift = max(log(effect_or), 2.5 tau), beyond about 99% of the ordinary providers, and every other eligible
# provider is null, the ordinary spread included. The all-zero providers are neither: they exist to make
# the Wald-type tests inadmissible, as they do in real measures.

COVARIATES <- c("AGE", "FEMALE", "DIABETES", "CHF", "COPD", "BMI", "YRS", "NH", "ALB", "HGB")
BETA <- list(
  patient_month = c(AGE = -0.015, FEMALE = -0.35, DIABETES = -0.25, CHF = -0.20, COPD = -0.10, BMI = -0.02,
                    YRS = 0.08, NH = -0.70, ALB = 0.40, HGB = 0.05),
  discharge = c(AGE = 0.010, FEMALE = 0.05, DIABETES = 0.20, CHF = 0.45, COPD = 0.30, BMI = -0.01,
                YRS = 0.03, NH = 0.50, ALB = -0.50, HGB = -0.10),
  binomial_days = c(AGE = 0.012, FEMALE = 0.05, DIABETES = 0.20, CHF = 0.50, COPD = 0.30, BMI = -0.01,
                    YRS = 0.03, NH = 0.60, ALB = -0.55, HGB = -0.12))

simulate_design <- function(design, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion")   # explicit kind: called inside future workers
  kind <- design$kind
  n_prov <- as.integer(design$n_prov)
  elig <- as.integer(design$eligibility)
  ids <- sprintf("P%04d", seq_len(n_prov))

  # ---- providers: sizes, the ordinary spread, the planted outliers, the all-zero providers ----
  n_pat <- pmin(400L, pmax(1L, as.integer(round(exp(stats::rnorm(n_prov, log(design$median_patients), design$size_sdlog))))))
  eligible <- which(n_pat >= elig)
  invariant <- integer(0)
  if (design$n_invariant > 0 && kind != "binomial_days") {
    pool <- eligible[n_pat[eligible] <= elig + 5]       # small but tested, as in real measures
    if (length(pool) >= design$n_invariant) invariant <- sort(sample(pool, design$n_invariant))
  }
  gamma <- stats::rnorm(n_prov, 0, design$tau)
  shift <- max(log(design$effect_or), 2.5 * design$tau)
  cand <- setdiff(eligible, invariant)
  k <- if (design$planted_frac > 0) max(1L, as.integer(round(design$planted_frac * length(cand) / 2))) else 0L
  k <- min(k, length(cand) %/% 2L)
  pick <- if (k > 0) sample(cand, 2L * k) else integer(0)
  up <- sort(pick[seq_len(k)]); down <- sort(pick[k + seq_len(k)])
  gamma[up] <- shift; gamma[down] <- -shift

  # ---- patients: a case mix that differs by provider (scaled by `casemix`) and a patient random effect ----
  cm <- design$casemix
  off_age <- stats::rnorm(n_prov, 0, 4 * cm); off_chf <- stats::rnorm(n_prov, 0, 0.6 * cm)
  off_nh <- stats::rnorm(n_prov, 0, 0.8 * cm); off_alb <- stats::rnorm(n_prov, 0, 0.25 * cm)
  total <- sum(n_pat)
  pp <- rep(seq_len(n_prov), n_pat)
  P <- data.frame(age = stats::rnorm(total, 63 + off_age[pp], 13), female = stats::rbinom(total, 1, 0.44),
                  diabetes = stats::rbinom(total, 1, 0.58),
                  chf = stats::rbinom(total, 1, stats::plogis(stats::qlogis(0.30) + off_chf[pp])),
                  copd = stats::rbinom(total, 1, 0.14), bmi = pmax(15, stats::rnorm(total, 28, 6.5)),
                  yrs = stats::rexp(total, 1 / 4),
                  nh = stats::rbinom(total, 1, stats::plogis(stats::qlogis(0.06) + off_nh[pp])),
                  alb = stats::rnorm(total, 3.8 + off_alb[pp], 0.40), hgb = stats::rnorm(total, 11.0, 1.0),
                  b = stats::rnorm(total, 0, design$sigma_patient))

  # ---- rows: discharges per patient, or months with attrition (mean about eight) ----
  m <- if (kind == "discharge") 1L + stats::rgeom(total, prob = 0.55) else pmin(12L, 1L + stats::rgeom(total, prob = 1 / 8))
  n <- sum(m)
  rp <- rep(seq_len(total), m)
  rprov <- pp[rp]
  Z <- cbind(AGE = P$age[rp], FEMALE = P$female[rp], DIABETES = P$diabetes[rp], CHF = P$chf[rp], COPD = P$copd[rp],
             BMI = P$bmi[rp], YRS = P$yrs[rp], NH = P$nh[rp], ALB = P$alb[rp] + stats::rnorm(n, 0, 0.25),
             HGB = P$hgb[rp] + stats::rnorm(n, 0, 0.7))
  beta <- BETA[[kind]][COVARIATES]
  eta0 <- gamma[rprov] + as.numeric(scale(Z, center = TRUE, scale = FALSE) %*% beta) + P$b[rp]
  # the intercept that gives the target marginal rate
  alpha <- stats::uniroot(function(a) mean(stats::plogis(a + eta0)) - design$target_rate, c(-20, 20))$root
  p <- stats::plogis(alpha + eta0)
  if (kind == "binomial_days") {
    N <- sample(28:31, n, replace = TRUE)
    Y <- stats::rbinom(n, N, p)
  } else {
    N <- rep(1L, n)
    Y <- stats::rbinom(n, 1, p)
  }
  if (length(invariant)) Y[rprov %in% invariant] <- 0L

  d <- data.frame(PROV = ids[rprov], PAT = sprintf("K%06d", rp), Y = Y, N = N, round(Z, 3), stringsAsFactors = FALSE)
  list(data = d,
       truth = list(up = ids[up], down = ids[down], invariant = ids[invariant], shift = shift,
                    eligible = ids[eligible], gamma = stats::setNames(gamma, ids), intercept = alpha),
       n_patients = stats::setNames(n_pat, ids))
}
