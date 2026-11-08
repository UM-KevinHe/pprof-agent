# calibrate_tests.R -- the agent checks each admissible test by simulation (proposal deck, slide 14):
# copy the design (the fitted providers, their patients and rows, the covariates and the fitted risk
# model, the within-patient correlation), simulate outcomes with NO provider truly different (every
# provider at the norm), refit, run each test, and record the type I error in each tail against the
# nominal one-sided level. The empirical null is also applied to each simulated set of statistics.
#
# With `effect_or` (an odds ratio > 1) the simulation also measures power: in every simulated data set a
# random `planted_frac` of the providers is moved off the norm, half by +log(effect_or) on the logit scale
# (higher), half by -log(effect_or) (lower). The type I error is then counted on the providers left at the
# norm and the power per tail on the providers planted in that tail. Without it every provider is at the
# norm and only the type I error is measured (the size-only calibration).
#
# payload: {run_dir, tests: [...], n_sim (100), max_providers (500), seed (20261018), level (0.95),
#           n_groups (4), sigma_grid (optional), effect_or (null = size only), planted_frac (0.05)}
# writes tables/calibration_tests.csv; returns the rates and the design copied.
#
# The within-patient correlation is induced by a patient random effect on the logit scale; its scale is
# chosen on a grid so that the simulated within-patient correlation of standardized residuals matches
# the observed one (a transparent moment match, recorded in the output).
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "group_en.R"))
suppressPackageStartupMessages(library(pprofR))
TEST_MAP <- c(exact = "exact.poisbinom", score = "score", wald = "wald", robust_wald = "robust_wald",
              robust_score = "robust_score", stabilized_wald = "stabilized_wald", stabilized_score = "stabilized_score")

run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir")
  tests <- chr(arg("tests", names(TEST_MAP)))
  tests <- tests[tests %in% names(TEST_MAP)]
  if (!length(tests)) stop("no known test requested", call. = FALSE)
  n_sim <- as.integer(arg("n_sim", 100)); max_prov <- as.integer(arg("max_providers", 500))
  seed <- as.integer(arg("seed", 20261018)); level <- as.numeric(arg("level", 0.95))
  n_groups <- as.integer(arg("n_groups", 4))
  effect_or <- suppressWarnings(as.numeric(arg("effect_or", NA_real_)))
  power_on <- is.finite(effect_or) && effect_or > 1
  planted_frac <- as.numeric(arg("planted_frac", 0.05))
  fit <- load_fit(run_dir)
  cl <- fit$char_list
  d <- fit$data_include
  has_pat <- !is.null(cl$PatientID.char)
  clustered <- c("robust_wald", "robust_score", "stabilized_wald", "stabilized_score")
  if (!has_pat) tests <- setdiff(tests, clustered)
  set.seed(seed)
  lev <- rownames(fit$gamma)
  if (length(lev) > max_prov) {
    keep <- sort(sample(lev, max_prov))
    d <- d[pid(d[[cl$prov.char]]) %in% keep, , drop = FALSE]
  }
  d <- d[order(d[[cl$prov.char]], if (has_pat) d[[cl$PatientID.char]] else seq_len(nrow(d))), , drop = FALSE]
  Z <- as.matrix(d[, cl$Z.char, drop = FALSE])
  N <- if (is.null(cl$N.char) || !cl$N.char %in% names(d)) rep(1, nrow(d)) else as.numeric(d[[cl$N.char]])
  beta <- as.numeric(fit$beta[, 1])
  gamma0 <- stats::median(as.numeric(fit$gamma[, 1]))
  eta0 <- gamma0 + as.numeric(Z %*% beta)
  prov <- pid(d[[cl$prov.char]])
  pat <- if (has_pat) paste(prov, pid(d[[cl$PatientID.char]])) else seq_len(nrow(d))
  pat_f <- factor(pat, levels = unique(pat))
  n_pat <- nlevels(pat_f)
  # the observed within-patient correlation of standardized residuals, from the real fit
  pr <- as.numeric(fit$pred[match(rownames(d), rownames(fit$data_include))])
  if (anyNA(pr)) pr <- as.numeric(stats::plogis(rep(as.numeric(fit$gamma[match(prov, lev), 1]), 1) + as.numeric(Z %*% beta)))
  e_obs <- (as.numeric(d[[cl$Y.char]]) - N * pr) / sqrt(N * pr * (1 - pr))
  rho_obs <- if (has_pat) within_corr(e_obs, pat_f) else 0
  # the patient random effect scale: the grid value whose simulated correlation is nearest
  grid <- as.numeric(unlist(arg("sigma_grid", c(0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3))))
  # residuals are taken from the marginal probability (no random effect), as the observed residuals are
  # taken from the fitted model: the random effect is what makes a patient's rows correlated
  p_marg <- stats::plogis(eta0)
  rho_sim <- vapply(grid, function(s) {
    b <- stats::rnorm(n_pat, 0, s)[as.integer(pat_f)]
    y <- stats::rbinom(length(eta0), N, stats::plogis(eta0 + b))
    within_corr((y - N * p_marg) / sqrt(N * p_marg * (1 - p_marg)), pat_f)
  }, numeric(1))
  sigma <- if (has_pat) grid[which.min(abs(rho_sim - rho_obs))] else 0
  # the simulations
  # flags among the providers at the norm (type I error) and among the planted ones (power), per tail
  flags <- array(0L, dim = c(n_sim, length(tests), 2), dimnames = list(NULL, tests, c("higher", "lower")))
  flags_en <- flags; hits <- flags; hits_en <- flags
  prov_lev <- unique(prov)
  n_tested <- length(prov_lev)
  k_tail <- if (power_on) min(max(1L, as.integer(round(planted_frac * n_tested / 2))), n_tested %/% 2L) else 0L
  power_on <- power_on && k_tail >= 1L
  n_null <- n_tested - 2L * k_tail
  for (s in seq_len(n_sim)) {
    up <- down <- character(0)
    eta <- eta0
    if (power_on) {                    # drawn only when power is asked for, so the size-only stream is unchanged
      pick <- sample(prov_lev, 2L * k_tail)
      up <- pick[seq_len(k_tail)]; down <- pick[k_tail + seq_len(k_tail)]
      eta <- eta0 + log(effect_or) * ((prov %in% up) - (prov %in% down))
    }
    b <- stats::rnorm(n_pat, 0, sigma)[as.integer(pat_f)]
    y <- stats::rbinom(nrow(d), N, stats::plogis(eta + b))
    args_prep <- list(Y = data.frame(Y = y), Z = d[, cl$Z.char, drop = FALSE], ProvID = data.frame(P = prov),
                      cutoff = 1, check = FALSE, message = FALSE)
    if (!all(N == 1)) args_prep$N <- data.frame(N = N)
    if (has_pat) args_prep$PatientID <- data.frame(K = pat)
    dp <- do.call(fe_data_prep, args_prep)
    f <- logis_fe(dp, tol = 1e-6, threads = threads(), message = FALSE)
    sizes <- provider_sizes(f)
    for (t in tests) {
      r <- test_fe(if (t == "wald") wald_fit(f) else f, test = TEST_MAP[[t]], null = "median", level = level, threads = threads())
      fl <- as.integer(as.character(r$flag))
      at_norm <- !rownames(r) %in% c(up, down); is_up <- rownames(r) %in% up; is_down <- rownames(r) %in% down
      flags[s, t, "higher"] <- sum(fl[at_norm] == 1, na.rm = TRUE)
      flags[s, t, "lower"] <- sum(fl[at_norm] == -1, na.rm = TRUE)
      hits[s, t, "higher"] <- sum(fl[is_up] == 1, na.rm = TRUE)
      hits[s, t, "lower"] <- sum(fl[is_down] == -1, na.rm = TRUE)
      if (n_tested >= 3 * n_groups) {
        st <- as.numeric(r$stat)
        en <- tryCatch(Group_EN(st, sizes$n_patients[match(rownames(r), sizes$provider)], ngroup = n_groups), error = function(e) NULL)
        if (!is.null(en)) {
          crit <- stats::qnorm(1 - (1 - level) / 2)
          z <- en$z_score_GEN
          flags_en[s, t, "higher"] <- sum(z[at_norm] > crit, na.rm = TRUE)
          flags_en[s, t, "lower"] <- sum(z[at_norm] < -crit, na.rm = TRUE)
          hits_en[s, t, "higher"] <- sum(z[is_up] > crit, na.rm = TRUE)
          hits_en[s, t, "lower"] <- sum(z[is_down] < -crit, na.rm = TRUE)
        }
      }
    }
  }
  nominal <- 100 * (1 - level) / 2
  rows <- list()
  for (t in tests) for (ref in c("theoretical", "empirical")) for (tail in c("higher", "lower")) {
    v <- if (ref == "theoretical") flags[, t, tail] else flags_en[, t, tail]
    rate <- 100 * v / n_null
    h <- if (ref == "theoretical") hits[, t, tail] else hits_en[, t, tail]
    pw <- if (power_on) 100 * h / k_tail else rep(NA_real_, n_sim)
    rows[[length(rows) + 1]] <- data.frame(test = t, reference = ref, tail = tail, rate_pct = mean(rate),
                                           se_pct = stats::sd(rate) / sqrt(n_sim), nominal_pct = nominal, n_sim = n_sim,
                                           power_pct = mean(pw), power_se_pct = stats::sd(pw) / sqrt(n_sim),
                                           effect_or = if (power_on) effect_or else NA_real_, stringsAsFactors = FALSE)
  }
  tab <- do.call(rbind, rows)
  write_table(tab, run_dir, "calibration_tests")
  list(n_sim = n_sim, n_providers_used = n_tested, n_rows_used = nrow(d), n_patients_used = n_pat,
       subsampled = length(lev) > max_prov, sigma_patient = sigma, rho_observed = rho_obs,
       rho_simulated = rho_sim[which.min(abs(rho_sim - rho_obs))], sigma_grid = I(grid), rho_grid = I(rho_sim),
       level = level, nominal_pct = nominal, tests = I(tests),
       power = power_on, effect_or = if (power_on) effect_or else NULL, planted_frac = if (power_on) planted_frac else NULL,
       n_planted_per_tail = k_tail, n_providers_at_norm = n_null,
       rates = tab, seconds = seconds_since(t0),
       tables = list(calibration_tests = table_path(run_dir, "calibration_tests")))
})
