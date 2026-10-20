# profile_data.R -- column facts of one data file, before any specification exists. Facts only:
# names, types, missingness, cardinality, ranges, and which columns are ELIGIBLE for which role.
# Never a row of data. payload: {data_path, max_levels (12)}
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
run_step({
  t0 <- Sys.time()
  path <- arg("data_path")
  max_levels <- arg("max_levels", 12)
  df <- load_data(path)
  n <- nrow(df)
  cols <- lapply(names(df), function(nm) {
    x <- df[[nm]]
    miss <- sum(is.na(x))
    xx <- x[!is.na(x)]
    type <- if (is.numeric(x)) (if (is.integer(x) || all(xx == floor(xx))) "integer" else "numeric")
            else if (is.logical(x)) "logical" else if (inherits(x, c("Date", "POSIXt", "IDate"))) "date" else "character"
    nd <- length(unique(xx))
    binary <- is.numeric(x) && nd == 2 && all(xx %in% c(0, 1))
    constant <- nd <= 1
    rows_per_level <- if (nd > 0) length(xx) / nd else NA_real_
    up <- toupper(nm)
    id_like <- (type %in% c("character", "integer")) && nd >= 10 && rows_per_level < 0.5 * length(xx) &&
      (grepl("ID|PROV|FAC|CCN|CENTER|CENTRE|HOSP|SITE|KECC|PATIENT|PT_|MRN|SUBJECT", up) || rows_per_level <= 50)
    out <- list(name = nm, type = type, n_missing = miss, n_distinct = nd, binary = binary, constant = constant,
                id_like = id_like, rows_per_level = rows_per_level)
    if (is.numeric(x) && nd > 0) {
      out$min <- min(xx); out$max <- max(xx); out$mean <- mean(xx)
    }
    if (nd <= max_levels && nd > 0) out$levels <- as.character(sort(unique(xx)))
    out
  })
  names(cols) <- names(df)
  up <- toupper(names(df))
  nd <- vapply(cols, function(c) c$n_distinct, numeric(1))
  rpl <- vapply(cols, function(c) ifelse(is.na(c$rows_per_level), 0, c$rows_per_level), numeric(1))
  typ <- vapply(cols, function(c) c$type, character(1))
  is_bin <- vapply(cols, function(c) isTRUE(c$binary), logical(1))
  # role eligibility, by facts
  outcome_cands <- names(df)[is_bin & !grepl("ID$|^PROV|FLAG_?EXCL", up)]
  trials_cands <- names(df)[typ == "integer" & !is_bin & vapply(cols, function(c) !is.null(c$min) && c$min >= 1 && c$max > 1, logical(1)) &
                              grepl("^N$|^N_|_N$|TRIAL|DAYS|DENOM|RISK|COUNT|EXPOS", up)]
  # providers: a character column, or an integer column whose name says so, with many levels and
  # several rows per level; patients: an id column with more levels than the provider candidate
  prov_hint <- grepl("PROV|FAC|CCN|CENTER|CENTRE|SITE|UNIT|HOSP", up)
  pat_hint <- grepl("PATIENT|KECC|PT_?ID|MRN|SUBJECT|PERSON|BENE", up)
  grp_cands <- names(df)[(typ == "character" | (typ == "integer" & (prov_hint | pat_hint))) & nd >= 10 & nd <= n / 2 & rpl >= 2]
  provider_cands <- names(df)[names(df) %in% grp_cands & (prov_hint | !pat_hint)]
  provider_cands <- provider_cands[order(!prov_hint[match(provider_cands, names(df))], nd[provider_cands])]
  min_prov_nd <- if (length(provider_cands)) min(nd[provider_cands]) else 0
  patient_cands <- names(df)[(names(df) %in% grp_cands | (typ %in% c("character", "integer") & nd >= 10 & nd >= n * 0.5)) &
                               nd > min_prov_nd & !names(df) %in% provider_cands[1]]
  patient_cands <- patient_cands[order(!pat_hint[match(patient_cands, names(df))], -nd[patient_cands])]
  time_cands <- names(df)[grepl("MONTH|QUARTER|QTR|PERIOD|DATE|YEAR|TIME|YRMO", up) & !is_bin]
  hospital_cands <- names(df)[grepl("HOSP", up) & names(df) %in% grp_cands]
  covariate_cands <- names(df)[typ %in% c("integer", "numeric", "logical") & !is_bin | is_bin]
  # binary columns stay candidates: one of them is the outcome, the others may be adjusters; the harness
  # removes the declared outcome from the options it offers
  covariate_cands <- setdiff(covariate_cands, c(provider_cands, patient_cands, time_cands, trials_cands, hospital_cands,
                                                names(df)[vapply(cols, function(c) isTRUE(c$constant) || isTRUE(c$id_like), logical(1))]))
  covariate_cands <- covariate_cands[!grepl("_ID$|^ID$", toupper(covariate_cands))]
  categorical <- names(df)[typ == "character" & nd <= max_levels & nd >= 2]
  list(data_path = path, rows = n, n_columns = ncol(df), columns = unname(cols),
       candidates = list(outcome = I(outcome_cands), trials = I(trials_cands), provider = I(provider_cands),
                         patient = I(patient_cands), time = I(time_cands), hospital = I(hospital_cands),
                         covariates = I(covariate_cands), categorical = I(categorical)),
       seconds = seconds_since(t0))
})
