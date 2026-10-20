# common.R -- shared by every script in r_scripts/: the JSON handshake with the Python harness, data
# loading, thread pinning, and the run-directory conventions. Each script is called as
#     Rscript <script>.R <payload.json> <out.json>
# and must write exactly one JSON object to <out.json>: {"status": "ok", ...} or
# {"status": "error", "class": ..., "message": ...}. Nothing is printed for the harness to parse.
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

.args <- commandArgs(trailingOnly = TRUE)
if (length(.args) < 2) stop("usage: Rscript <script>.R <payload.json> <out.json>", call. = FALSE)
PAYLOAD <- jsonlite::fromJSON(.args[1], simplifyVector = TRUE)
OUT_PATH <- .args[2]

write_out <- function(x) {
  jsonlite::write_json(x, OUT_PATH, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
  invisible(x)
}

fail <- function(message, class = "error") {
  write_out(list(status = "error", class = class, message = paste(message, collapse = " ")))
  quit(save = "no", status = 1)
}

# Run the body of a script: errors become a JSON error object, a result becomes {"status": "ok", ...}.
run_step <- function(expr) {
  res <- tryCatch(expr, error = function(e) fail(conditionMessage(e), class(e)[1]))
  if (!is.list(res)) res <- list(value = res)
  if (is.null(res$status)) res$status <- "ok"
  write_out(res)
}

# a payload field, with a default when absent or null
arg <- function(name, default = NULL, from = PAYLOAD) {
  v <- from[[name]]
  if (is.null(v) || (length(v) == 1 && is.na(v) && !is.character(v))) default else v
}

# character vector from a JSON array that jsonlite may have simplified to a scalar or an empty list
chr <- function(x) if (is.null(x) || length(x) == 0) character(0) else as.character(unlist(x))

threads <- function() {
  t <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
  if (is.na(t) || t < 1) 1L else t
}

load_data <- function(path, columns = NULL) {
  columns <- if (is.null(columns)) NULL else unique(chr(columns))
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
    df <- as.data.frame(arrow::read_parquet(path, col_select = if (is.null(columns)) NULL else dplyr::all_of(columns)))
  } else {
    df <- data.table::fread(path, select = columns, na.strings = c("", "NA", "."), data.table = FALSE)
  }
  df
}

fit_path <- function(run_dir) file.path(run_dir, "fit.rds")
tables_dir <- function(run_dir) {
  d <- file.path(run_dir, "tables")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}
table_path <- function(run_dir, name) file.path(tables_dir(run_dir), paste0(name, ".csv"))
write_table <- function(df, run_dir, name) {
  p <- table_path(run_dir, name)
  data.table::fwrite(df, p)
  p
}
read_table <- function(run_dir, name) data.table::fread(table_path(run_dir, name), data.table = FALSE)

# five-number summary as a named list (JSON object)
q5 <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(list(min = NULL, q1 = NULL, median = NULL, q3 = NULL, max = NULL, n = 0L))
  q <- stats::quantile(x, c(0, .25, .5, .75, 1), names = FALSE)
  list(min = q[1], q1 = q[2], median = q[3], q3 = q[4], max = q[5], n = length(x))
}

# provider ids as character, always
pid <- function(x) as.character(x)

# the fitted pprofR object and the names it was fitted under
load_fit <- function(run_dir) {
  p <- fit_path(run_dir)
  if (!file.exists(p)) stop("no fitted model in the run directory (run `fit` first)", call. = FALSE)
  readRDS(p)
}

# rows and patients per fitted provider, in the order of fit$gamma
provider_sizes <- function(fit) {
  cl <- fit$char_list
  d <- fit$data_include
  lev <- rownames(fit$gamma)
  prov <- factor(pid(d[[cl$prov.char]]), levels = lev)
  n_rows <- as.integer(table(prov))
  if (!is.null(cl$PatientID.char)) {
    pp <- unique(data.frame(p = pid(d[[cl$prov.char]]), k = pid(d[[cl$PatientID.char]]), stringsAsFactors = FALSE))
    n_pat <- as.integer(table(factor(pp$p, levels = lev)))
  } else {
    n_pat <- n_rows
  }
  data.frame(provider = lev, n_rows = n_rows, n_patients = n_pat, stringsAsFactors = FALSE)
}

seconds_since <- function(t0) as.numeric(Sys.time() - t0, units = "secs")
