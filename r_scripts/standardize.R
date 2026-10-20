# standardize.R -- indirect and direct standardized ratios and rates (pprofR::SR_output).
# payload: {run_dir, stdz: ["indirect","direct"], norm: "median" | "size_weighted_mean" | <number>, N_pop}
# writes tables/sm.csv; returns the norm used, population rate, and the verdict flips between the two.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprofR))
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir")
  fit <- load_fit(run_dir)
  stdz <- chr(arg("stdz", c("indirect", "direct")))
  norm <- arg("norm", "median")
  gam <- as.numeric(fit$gamma[, 1]); lev <- rownames(fit$gamma)
  sizes <- provider_sizes(fit)
  norm_value <- if (identical(norm, "median")) stats::median(gam) else if (identical(norm, "size_weighted_mean")) {
    sum(sizes$n_rows * gam) / sum(sizes$n_rows)
  } else as.numeric(norm)
  N_pop <- arg("N_pop")
  args_sm <- list(fit = fit, stdz = stdz, measure = c("rate", "ratio"), null = norm_value, threads = threads(), se = TRUE)
  if (!is.null(N_pop)) args_sm$N_pop <- as.numeric(N_pop)
  sm <- do.call(SR_output, args_sm)
  # SR_output returns one value per provider in the order of fit$gamma; check where names exist
  chk <- function(m) { rn <- rownames(m); if (!is.null(rn) && !identical(rn, lev)) stop("provider order differs in SR_output", call. = FALSE); as.numeric(m[, 1]) }
  tab <- data.frame(provider = lev, stringsAsFactors = FALSE)
  if ("indirect" %in% stdz) {
    tab$observed <- as.numeric(sm$OE$OE_indirect$Obs_provider)
    tab$expected_indirect <- as.numeric(sm$OE$OE_indirect$Exp.indirect_provider)
    tab$indirect_ratio <- chk(sm$indirect.ratio)
    tab$indirect_rate <- chk(sm$indirect.rate)
  }
  if ("direct" %in% stdz) {
    tab$direct_ratio <- chk(sm$direct.ratio)
    tab$direct_rate <- chk(sm$direct.rate)
    tab$direct_numer_se <- as.numeric(sm$numer_se[, 1])
    g <- read_table(run_dir, "gamma")
    tab$direct_rate_se_model <- tab$direct_numer_se / sm$N_pop * g$se_model[match(lev, g$provider)]
  }
  write_table(tab, run_dir, "sm")
  flips <- NULL
  if (all(c("indirect", "direct") %in% stdz)) {
    s_i <- sign(tab$indirect_ratio - 1); s_d <- sign(tab$direct_ratio - 1)
    flip <- s_i != 0 & s_d != 0 & s_i != s_d
    flips <- list(n = sum(flip), providers = I(head(lev[flip], 50)),
                  correlation = if (length(lev) > 2) stats::cor(tab$indirect_ratio, tab$direct_ratio) else NULL)
  }
  pop_n <- if (is.null(N_pop)) sum(if (is.null(fit$char_list$N.char)) rep(1, length(fit$obs)) else fit$data_include[[fit$char_list$N.char]]) else as.numeric(N_pop)
  list(norm = norm, norm_value = norm_value, N_pop = pop_n, population_rate = sum(fit$obs) / pop_n,
       indirect_ratio = if ("indirect" %in% stdz) q5(tab$indirect_ratio) else NULL,
       direct_ratio = if ("direct" %in% stdz) q5(tab$direct_ratio) else NULL,
       flips = flips, seconds = seconds_since(t0), tables = list(sm = table_path(run_dir, "sm")))
})
