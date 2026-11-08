#!/usr/bin/env Rscript
# oc.R -- the operating characteristics of every provider test, for one training design. R data sets are
# simulated from the design (simulate.R); each is analysed the way the agent's engine analyses one (the
# eligibility threshold, the fixed-effects fit, every provider test, the size-stratified empirical null);
# the flags are counted against the truth per tail: the type I error among the null providers (the
# ordinary spread included, definition B) and the power among the providers planted in that tail.
# These scores are what judge a choice in the training data (summarize_oc.py turns them into the reward).
#
#   Rscript oc.R designs=<designs.jsonl> id=<design id> R=100 out=<dir>
# writes <out>/<id>.json. Repetitions run in parallel over SLURM_CPUS_PER_TASK workers, one thread each.
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
source(file.path(here, "simulate.R"))
source(file.path(here, "..", "r_scripts", "group_en.R"))
suppressPackageStartupMessages({ library(pprofR); library(future); library(future.apply); library(progressr) })

TEST_MAP <- c(exact = "exact.poisbinom", score = "score", wald = "wald", robust_wald = "robust_wald",
              robust_score = "robust_score", stabilized_wald = "stabilized_wald", stabilized_score = "stabilized_score")
# the Wald test on a copy of the fit with centred adjusters (r_scripts/common.R explains why)
wald_fit <- function(fit) {
  zc <- fit$char_list$Z.char
  f <- fit
  f$data_include[zc] <- lapply(f$data_include[zc], function(x) x - mean(x))
  f
}
EXACT_MAX_TRIALS <- 20000      # rules.THRESHOLDS["exact_max_trials"]
EN_MIN_PROVIDERS <- 40         # rules.THRESHOLDS["en_min_providers"]
LEVEL <- 0.95                  # two-sided, i.e. 2.5% in each tail

run_one_rep <- function(design, seed) {
  sim <- simulate_design(design, seed)
  d <- sim$data[sim$data$PROV %in% sim$truth$eligible, , drop = FALSE]
  args_prep <- list(Y = d["Y"], Z = d[, COVARIATES], ProvID = d["PROV"], PatientID = d["PAT"],
                    cutoff = 1, check = FALSE, message = FALSE)
  if (design$kind == "binomial_days") args_prep$N <- d["N"]
  dp <- do.call(fe_data_prep, args_prep)
  f <- logis_fe(dp, tol = 1e-6, threads = 1, message = FALSE)
  provs <- rownames(f$gamma)
  trials <- as.numeric(tapply(d$N, d$PROV, sum)[provs])
  events <- as.numeric(tapply(d$Y, d$PROV, sum)[provs])
  npat <- as.numeric(sim$n_patients[provs])
  tests <- names(TEST_MAP)
  if (max(trials) > EXACT_MAX_TRIALS) tests <- setdiff(tests, "exact")
  is_up <- provs %in% sim$truth$up; is_down <- provs %in% sim$truth$down
  is_null <- !(is_up | is_down | provs %in% sim$truth$invariant)
  crit <- stats::qnorm(1 - (1 - LEVEL) / 2)
  counts <- list()
  for (t in tests) {
    r <- tryCatch(test_fe(if (t == "wald") wald_fit(f) else f, test = TEST_MAP[[t]], null = "median", level = LEVEL, threads = 1),
                  error = function(e) NULL)
    if (is.null(r)) next
    i <- match(provs, rownames(r))
    fl <- as.integer(as.character(r$flag[i]))
    st <- as.numeric(r$stat[i])
    fen <- rep(NA_integer_, length(provs))
    if (length(provs) >= EN_MIN_PROVIDERS) {
      en <- tryCatch(Group_EN(st, npat, ngroup = 4), error = function(e) NULL)
      if (!is.null(en)) {
        z <- en$z_score_GEN
        fen <- ifelse(is.na(z), NA_integer_, ifelse(z > crit, 1L, ifelse(z < -crit, -1L, 0L)))
      }
    }
    for (ref in c("theoretical", "empirical")) {
      g <- if (ref == "theoretical") fl else fen
      if (all(is.na(g))) next
      counts[[paste(t, ref, sep = "|")]] <- c(
        fp_higher = sum(g[is_null] == 1, na.rm = TRUE), fp_lower = sum(g[is_null] == -1, na.rm = TRUE),
        tp_higher = sum(g[is_up] == 1, na.rm = TRUE), tp_lower = sum(g[is_down] == -1, na.rm = TRUE),
        n_null = sum(is_null), n_up = sum(is_up), n_down = sum(is_down), n_na = sum(is.na(g)))
    }
  }
  list(counts = counts, n_providers = length(provs), n_rows = nrow(d), max_trials = max(trials),
       invariant_observed = sum(events == 0 | events == trials))
}

# the pooled rate over repetitions and its Monte Carlo standard error (between-repetition spread)
pooled <- function(num, den) {
  ok <- den > 0
  if (!any(ok)) return(c(pct = NA_real_, se_pct = NA_real_))
  per <- 100 * num[ok] / den[ok]
  c(pct = 100 * sum(num[ok]) / sum(den[ok]), se_pct = if (sum(ok) > 1) stats::sd(per) / sqrt(sum(ok)) else NA_real_)
}

if (sys.nframe() == 0L) {
  kv <- list()
  for (a in commandArgs(trailingOnly = TRUE)) { p <- strsplit(a, "=", fixed = TRUE)[[1]]; kv[[p[1]]] <- paste(p[-1], collapse = "=") }
  lines <- readLines(kv$designs)
  designs <- lapply(lines[nzchar(lines)], jsonlite::fromJSON)
  design <- Filter(function(x) identical(x$id, kv$id), designs)[[1]]
  nrep <- as.integer(if (is.null(kv$R)) 100 else kv$R)
  out <- kv$out
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  t0 <- Sys.time()
  # the repetition seeds are fixed before the parallel call, so results do not depend on the workers
  SEEDS <- as.integer(design$seed) * 1000L + seq_len(nrep)
  parallel_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1"))
  plan(multisession, workers = parallel_cores)
  handlers(handler_txtprogressbar(file = stderr()))
  with_progress({
    prog <- progressor(steps = nrep)
    results_list <- future_lapply(
      X = seq_len(nrep),
      FUN = function(i) {
        prog()
        tryCatch(
          run_one_rep(design, SEEDS[i]),
          error = function(e) {
            message("Error in repetition ", i, ": ", conditionMessage(e))
            return(NULL)
          }
        )
      },
      future.seed = TRUE,
      future.scheduling = Inf,
      future.packages = c("pprofR", "MASS")
    )
  })
  plan(sequential)
  valid_results <- Filter(Negate(is.null), results_list)
  if (!length(valid_results)) stop("every repetition failed for design ", design$id)

  keys <- unique(unlist(lapply(valid_results, function(x) names(x$counts))))
  rates <- list()
  for (key in keys) {
    m <- do.call(rbind, lapply(valid_results, function(x) if (is.null(x$counts[[key]])) NULL else x$counts[[key]]))
    parts <- strsplit(key, "|", fixed = TRUE)[[1]]
    for (tail in c("higher", "lower")) {
      sz <- pooled(m[, paste0("fp_", tail)], m[, "n_null"])
      pw <- pooled(m[, paste0("tp_", tail)], m[, if (tail == "higher") "n_up" else "n_down"])
      rates[[length(rates) + 1]] <- list(test = parts[1], reference = parts[2], tail = tail,
                                         size_pct = unname(sz["pct"]), size_se_pct = unname(sz["se_pct"]),
                                         power_pct = unname(pw["pct"]), power_se_pct = unname(pw["se_pct"]),
                                         n_rep = nrow(m), n_na_mean = mean(m[, "n_na"]))
    }
  }
  summ <- function(f) stats::median(vapply(valid_results, f, numeric(1)))
  res <- list(id = design$id, design = design, n_rep = length(valid_results), n_failed = nrep - length(valid_results),
              nominal_pct = 100 * (1 - LEVEL) / 2,
              n_providers_median = summ(function(x) x$n_providers), n_rows_median = summ(function(x) x$n_rows),
              max_trials_median = summ(function(x) x$max_trials),
              invariant_observed_median = summ(function(x) x$invariant_observed),
              seconds = as.numeric(Sys.time() - t0, units = "secs"), rates = rates)
  jsonlite::write_json(res, file.path(out, paste0(design$id, ".json")), auto_unbox = TRUE, digits = NA, pretty = TRUE,
                       null = "null", na = "null")
  cat(sprintf("%s %-13s providers %4.0f rows %7.0f reps %d (failed %d) %.0fs\n", design$id, design$kind,
              res$n_providers_median, res$n_rows_median, res$n_rep, res$n_failed, res$seconds))
}
