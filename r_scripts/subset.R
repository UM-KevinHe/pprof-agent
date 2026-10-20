# subset.R -- one provider's observed and expected events by patient group ("why is facility A
# flagged?"): the provider's observed and expected (under the norm) overall and within the levels of
# each `by` column (a binary column by its two values; a continuous one by tertiles of the provider's
# own rows), and the provider's case mix against the population's.
# payload: {run_dir, provider, by: [...columns...] (default: every adjuster), max_by (8)}
# writes tables/subset_<provider>.csv; returns the breakdown.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir"); provider <- pid(arg("provider"))
  fit <- load_fit(run_dir)
  cl <- fit$char_list
  d <- fit$data_include
  lev <- rownames(fit$gamma)
  if (!provider %in% lev) stop("provider ", provider, " is not among the fitted providers", call. = FALSE)
  by <- chr(arg("by", cl$Z.char))
  by <- head(by[by %in% cl$Z.char], as.integer(arg("max_by", 8)))
  N <- if (is.null(cl$N.char) || !cl$N.char %in% names(d)) rep(1, nrow(d)) else as.numeric(d[[cl$N.char]])
  Y <- as.numeric(d[[cl$Y.char]])
  gamma0 <- stats::median(as.numeric(fit$gamma[, 1]))
  p0 <- as.numeric(stats::plogis(gamma0 + as.numeric(fit$linear_pred)))
  sel <- pid(d[[cl$prov.char]]) == provider
  rows <- list(data.frame(group = "all rows", level = "all", n = sum(sel), observed = sum(Y[sel]), expected = sum(N[sel] * p0[sel]),
                          stringsAsFactors = FALSE))
  casemix <- list()
  for (v in by) {
    x <- as.numeric(d[[v]])
    xs <- x[sel]
    casemix[[v]] <- list(provider_mean = mean(xs), population_mean = mean(x))
    if (length(unique(x[!is.na(x)])) <= 2) {
      g <- as.character(x)
    } else {
      q <- stats::quantile(xs, c(1 / 3, 2 / 3), na.rm = TRUE)
      g <- ifelse(x <= q[1], "low", ifelse(x <= q[2], "middle", "high"))
    }
    for (lv in unique(g[sel])) {
      k <- sel & g == lv
      rows[[length(rows) + 1]] <- data.frame(group = v, level = lv, n = sum(k), observed = sum(Y[k]), expected = sum(N[k] * p0[k]),
                                             stringsAsFactors = FALSE)
    }
  }
  tab <- do.call(rbind, rows)
  tab$ratio <- tab$observed / tab$expected
  write_table(tab, run_dir, paste0("subset_", provider))
  list(provider = provider, n_rows = sum(sel), observed = sum(Y[sel]), expected = sum(N[sel] * p0[sel]),
       ratio = sum(Y[sel]) / sum(N[sel] * p0[sel]), by = I(by), breakdown = tab, case_mix = casemix,
       seconds = seconds_since(t0),
       tables = stats::setNames(list(table_path(run_dir, paste0("subset_", provider))), paste0("subset_", provider)))
})
