# empirical_null.R -- the size-stratified empirical null (Group_EN) applied to one test's statistics.
# payload: {run_dir, key (the test table), n_groups (4), size_by: patients|rows, level (0.95), crit}
# writes tables/en_<key>.csv (the test table's display columns with z_en, group, flag_en_raw and the
# funnel limits re-drawn under the group's null) and tables/en_groups_<key>.csv.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "group_en.R"))
WALD_TYPE <- c("wald", "robust_wald", "stabilized_wald")
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir"); key <- arg("key"); n_groups <- as.integer(arg("n_groups", 4))
  size_by <- arg("size_by", "patients")
  level <- as.numeric(arg("level", 0.95))
  crit <- as.numeric(arg("crit", stats::qnorm(1 - (1 - level) / 2)))
  tab <- read_table(run_dir, paste0("test_", key))
  test <- sub("_.*$", "", key)   # the key is the test name unless the harness suffixes it
  wald_type <- any(startsWith(key, WALD_TYPE))
  size <- if (identical(size_by, "rows")) tab$n_rows else tab$n_patients
  en <- Group_EN(tab$stat, size, ngroup = n_groups)
  z <- en$z_score_GEN
  gi <- en$group_inf
  loc <- gi$location[en$group_idx]; sc <- gi$scale[en$group_idx]
  out <- tab
  out$z_en <- z
  out$group <- en$group_idx
  out$null_mean <- loc; out$null_sd <- sc
  out$flag_en_raw <- ifelse(is.na(z), NA_integer_, ifelse(z > crit, 1L, ifelse(z < -crit, -1L, 0L)))
  if (wald_type) {
    out$funnel_lower <- (loc - crit * sc) * tab$se
    out$funnel_upper <- (loc + crit * sc) * tab$se
    # the interval that inverts the empirical-null test: it excludes the null exactly when flagged
    g <- tab$estimate + tab$null_value
    out$ci_lower <- g - (loc + crit * sc) * tab$se
    out$ci_upper <- g - (loc - crit * sc) * tab$se
  } else {
    s <- sqrt(tab$variance) / tab$expected
    out$funnel_lower <- 1 + (loc - crit * sc) * s
    out$funnel_upper <- 1 + (loc + crit * sc) * s
  }
  write_table(out, run_dir, paste0("en_", key))
  write_table(gi, run_dir, paste0("en_groups_", key))
  ok <- is.finite(tab$stat)
  before <- ifelse(tab$stat > crit, 1L, ifelse(tab$stat < -crit, -1L, 0L))
  slope <- if (nrow(gi) >= 2) unname(stats::coef(stats::lm(gi$scale ~ log(gi$median_size)))[2]) else NA_real_
  list(key = key, n_groups = n_groups, size_by = size_by, level = level, crit = crit,
       groups = gi, scale_ratio = max(gi$scale) / min(gi$scale), scale_slope_log_size = slope,
       scale_max = max(gi$scale), location_max_abs = max(abs(gi$location)),
       flagged_theoretical = list(higher = sum(before[ok] == 1), lower = sum(before[ok] == -1), pct = 100 * mean(before[ok] != 0)),
       flagged_empirical = list(higher = sum(out$flag_en_raw[ok] == 1, na.rm = TRUE), lower = sum(out$flag_en_raw[ok] == -1, na.rm = TRUE),
                                pct = 100 * mean(out$flag_en_raw[ok] != 0, na.rm = TRUE)),
       changed = sum(before[ok] != out$flag_en_raw[ok], na.rm = TRUE),
       seconds = seconds_since(t0),
       tables = stats::setNames(list(table_path(run_dir, paste0("en_", key)), table_path(run_dir, paste0("en_groups_", key))),
                                c(paste0("en_", key), paste0("en_groups_", key))))
})
