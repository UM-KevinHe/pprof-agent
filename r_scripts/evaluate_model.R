# evaluate_model.R -- discrimination and calibration of the fitted risk-adjustment model.
# payload: {run_dir, n_groups (10)}. C statistics: overall (all comparable pairs, prediction with the
# provider effect) and WITHIN provider (pairs of patients of the same provider: the risk adjustment
# itself, the statistic the KECC measures report). Calibration: observed vs expected events by decile
# of the risk score Z*beta (the adjustment alone) and of the full prediction. Binary rows only for C.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
run_step({
  t0 <- Sys.time()
  run_dir <- arg("run_dir"); n_groups <- as.integer(arg("n_groups", 10))
  fit <- load_fit(run_dir)
  cl <- fit$char_list; d <- fit$data_include
  y <- as.numeric(d[[cl$Y.char]])
  N <- if (is.null(cl$N.char) || !cl$N.char %in% names(d)) rep(1, nrow(d)) else as.numeric(d[[cl$N.char]])
  pred <- as.numeric(fit$pred); risk <- as.numeric(fit$linear_pred)
  prov <- pid(d[[cl$prov.char]])
  binary <- all(N == 1)
  auc <- function(yy, pp) {
    n1 <- sum(yy == 1); n0 <- sum(yy == 0)
    if (n1 == 0 || n0 == 0) return(c(NA_real_, 0))
    r <- rank(pp)
    c((sum(r[yy == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0), n1 * n0)
  }
  c_overall <- c_within <- NA_real_; pairs_within <- 0
  if (binary) {
    c_overall <- auc(y, pred)[1]
    by_prov <- split(seq_along(y), prov)
    acc <- vapply(by_prov, function(ii) auc(y[ii], risk[ii]), numeric(2))
    w <- acc[2, ]; a <- acc[1, ]
    ok <- w > 0 & !is.na(a)
    c_within <- sum(a[ok] * w[ok]) / sum(w[ok]); pairs_within <- sum(w[ok])
  }
  calib <- function(score, label) {
    g <- cut(rank(score, ties.method = "first"), breaks = n_groups, labels = FALSE)
    data.frame(scale = label, decile = seq_len(n_groups),
               n = as.integer(table(factor(g, levels = seq_len(n_groups)))),
               observed = as.numeric(tapply(y, g, sum)), expected = as.numeric(tapply(N * pred, g, sum)),
               mean_predicted = as.numeric(tapply(pred, g, mean)),
               observed_rate = as.numeric(tapply(y, g, sum) / tapply(N, g, sum)), stringsAsFactors = FALSE)
  }
  tab <- rbind(calib(risk, "risk_score"), calib(pred, "prediction"))
  tab$oe <- tab$observed / tab$expected
  write_table(tab, run_dir, "calibration")
  rs <- tab[tab$scale == "risk_score", ]
  list(binary = binary, c_overall = c_overall, c_within_provider = c_within, pairs_within = pairs_within,
       n_rows = nrow(d), calibration = tab, oe_range_risk_score = c(min(rs$oe), max(rs$oe)),
       seconds = seconds_since(t0), tables = list(calibration = table_path(run_dir, "calibration")))
})
