# test_measure.R -- the measure-level test of a directly standardized rate (the transport of a
# provider test to the standardized scale; pprofR::test_SM). payload: {run_dir, measure: direct_rate,
# se_source: model | robust_wald | stabilized_wald (a test table with an se column), transform:
# identity|logit, location (0), scale (1) -- or en_key to take location/scale per provider from an
# empirical-null group table --, level, alternative, key}
# writes tables/measure_<key>.csv; returns counts.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(pprofR))
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir"); measure <- arg("measure", "direct_rate"); se_source <- arg("se_source", "model")
  transform <- arg("transform", "identity"); level <- as.numeric(arg("level", 0.95)); alternative <- arg("alternative", "two.sided")
  key <- arg("key", paste(measure, se_source, sep = "_"))
  sm <- read_table(run_dir, "sm")
  if (!measure %in% c("direct_rate")) stop("V1 tests the direct rate only (measure = direct_rate)", call. = FALSE)
  if (!"direct_rate" %in% names(sm)) stop("run standardize with direct first", call. = FALSE)
  fit <- load_fit(run_dir)
  N_pop <- arg("N_pop", sum(if (is.null(fit$char_list$N.char)) rep(1, length(fit$obs)) else fit$data_include[[fit$char_list$N.char]]))
  se_gamma <- if (identical(se_source, "model")) {
    g <- read_table(run_dir, "gamma"); g$se_model[match(sm$provider, g$provider)]
  } else {
    tt <- read_table(run_dir, paste0("test_", se_source))
    if (all(is.na(tt$se))) stop("test table ", se_source, " carries no standard error", call. = FALSE)
    tt$se[match(sm$provider, tt$provider)]
  }
  se <- sm$direct_numer_se / N_pop * se_gamma
  ref <- as.numeric(arg("ref", sum(fit$obs) / N_pop))
  location <- as.numeric(arg("location", 0)); scale <- as.numeric(arg("scale", 1))
  en_key <- arg("en_key")
  if (!is.null(en_key)) {
    en <- read_table(run_dir, paste0("en_", en_key)); gi <- read_table(run_dir, paste0("en_groups_", en_key))
    grp <- en$group[match(sm$provider, en$provider)]
    location <- gi$location[grp]; scale <- gi$scale[grp]
  }
  r <- test_SM(sm$direct_rate, se, ref = ref, transform = transform, location = location, scale = scale,
               level = level, alternative = alternative)
  tab <- data.frame(provider = sm$provider, measure = sm$direct_rate, se = se, ref = ref, z = r$z, p = r$p,
                    flag_raw = as.integer(r$flag), lower = r$lower, upper = r$upper, stringsAsFactors = FALSE)
  write_table(tab, run_dir, paste0("measure_", key))
  ok <- !is.na(tab$flag_raw)
  list(measure = measure, se_source = se_source, transform = transform, ref = ref, key = key,
       n_tested = sum(ok), flagged_higher = sum(tab$flag_raw == 1, na.rm = TRUE), flagged_lower = sum(tab$flag_raw == -1, na.rm = TRUE),
       pct_higher = 100 * mean(tab$flag_raw[ok] == 1), pct_lower = 100 * mean(tab$flag_raw[ok] == -1),
       seconds = seconds_since(t0), tables = stats::setNames(list(table_path(run_dir, paste0("measure_", key))), paste0("measure_", key)))
})
