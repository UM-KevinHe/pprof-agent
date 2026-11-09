# standardize.R -- indirect and direct standardized ratios and rates (pprof::standardize).
# payload: {run_dir, stdz: ["indirect","direct"], norm: "median" | "size_weighted_mean" | <number>, N_pop}
# writes tables/sm.csv; returns the norm used, population rate, and the verdict flips between the two.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprof))
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir")
  fit <- load_fit(run_dir)
  stdz <- chr(arg("stdz", c("indirect", "direct")))
  norm <- arg("norm", "median")
  gam <- fit_gamma(fit); lev <- fit_levels(fit)
  sizes <- provider_sizes(fit)
  norm_value <- if (identical(norm, "median")) stats::median(gam) else if (identical(norm, "size_weighted_mean")) {
    sum(sizes$n_rows * gam) / sum(sizes$n_rows)
  } else as.numeric(norm)
  N_pop <- arg("N_pop")
  pop_n <- if (is.null(N_pop)) sum(fit_N(fit)) else as.numeric(N_pop)
  chk <- function(m) { if (!identical(pid(m$provider), lev)) stop("provider order differs in standardize()", call. = FALSE); m }
  tab <- data.frame(provider = lev, stringsAsFactors = FALSE)
  if ("indirect" %in% stdz) {
    si <- chk(standardize(fit, type = "indirect", null = norm_value, n_pop = pop_n, threads = threads()))
    tab$observed <- as.numeric(si$observed)
    tab$expected_indirect <- as.numeric(si$expected)
    tab$indirect_ratio <- as.numeric(si$ratio)
    tab$indirect_rate <- as.numeric(si$rate)
  }
  if ("direct" %in% stdz) {
    sd_ <- chk(standardize(fit, type = "direct", n_pop = pop_n, threads = threads()))
    tab$direct_ratio <- as.numeric(sd_$ratio)
    tab$direct_rate <- as.numeric(sd_$rate)
    tab$direct_numer_se <- as.numeric(sd_$variance_expected)
    g <- read_table(run_dir, "gamma")
    tab$direct_rate_se_model <- tab$direct_numer_se / pop_n * g$se_model[match(lev, g$provider)]
  }
  write_table(tab, run_dir, "sm")
  flips <- NULL
  if (all(c("indirect", "direct") %in% stdz)) {
    s_i <- sign(tab$indirect_ratio - 1); s_d <- sign(tab$direct_ratio - 1)
    flip <- s_i != 0 & s_d != 0 & s_i != s_d
    flips <- list(n = sum(flip), providers = I(head(lev[flip], 50)),
                  correlation = if (length(lev) > 2) stats::cor(tab$indirect_ratio, tab$direct_ratio) else NULL)
  }
  list(norm = norm, norm_value = norm_value, N_pop = pop_n, population_rate = sum(fit_Y(fit)) / pop_n,
       indirect_ratio = if ("indirect" %in% stdz) q5(tab$indirect_ratio) else NULL,
       direct_ratio = if ("direct" %in% stdz) q5(tab$direct_ratio) else NULL,
       flips = flips, seconds = seconds_since(t0), tables = list(sm = table_path(run_dir, "sm")))
})
