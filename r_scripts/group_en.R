# group_en.R -- the size-stratified empirical null, ported unchanged from
# real_data/source_code/tests.R (Group_EN of the KECC Firth notebooks): providers ordered by size and
# cut into `ngroup` equal groups; within each, a Huber M-estimate of location (MASS::rlm) and a MAD
# scale define the null, and the statistic is re-centred and re-scaled. NA / infinite statistics are
# left out of the fit and get NA.
Group_EN <- function(z_score, size, ngroup = 4) {
  stopifnot(length(z_score) == length(size))
  ok <- is.finite(z_score)
  num <- length(z_score)
  ord <- order(size)
  group_idx <- integer(num)
  group_idx[ord] <- cut(seq_len(num), breaks = ngroup, labels = FALSE)
  tab <- data.frame(group = seq_len(ngroup), n = NA_integer_, min_size = NA_real_,
                    median_size = NA_real_, max_size = NA_real_,
                    location = NA_real_, scale = NA_real_, mean = NA_real_, sd = NA_real_)
  z_GEN <- rep(NA_real_, num)
  for (g in seq_len(ngroup)) {
    sel <- group_idx == g & ok
    zg <- z_score[sel]
    if (length(zg) < 3) stop("empirical null: fewer than three usable statistics in size group ", g, call. = FALSE)
    fit <- MASS::rlm(zg ~ 1, method = "M", psi = MASS::psi.huber, scale.est = "MAD")
    loc <- unname(stats::coef(fit)[1]); sc <- fit$s
    tab[g, c("n", "min_size", "median_size", "max_size")] <-
      c(sum(sel), min(size[sel]), stats::median(size[sel]), max(size[sel]))
    tab[g, c("location", "scale", "mean", "sd")] <- c(loc, sc, mean(zg), stats::sd(zg))
    z_GEN[sel] <- (zg - loc) / sc
  }
  list(z_score_GEN = z_GEN, group_idx = group_idx, group_inf = tab)
}
