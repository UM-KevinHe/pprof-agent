#!/usr/bin/env Rscript
# write_task_data.R -- one task's data set from a training design, written as the analyst's file would be
# (data.csv: PROV, PAT, Y, N for binomial days only, the covariates) with its truth beside it (truth.json).
#
#   Rscript write_task_data.R designs=<designs.jsonl> id=<design id> seed=<integer> out=<dir>
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
source(file.path(here, "simulate.R"))
kv <- list()
for (a in commandArgs(trailingOnly = TRUE)) { p <- strsplit(a, "=", fixed = TRUE)[[1]]; kv[[p[1]]] <- paste(p[-1], collapse = "=") }
lines <- readLines(kv$designs)
design <- Filter(function(x) identical(x$id, kv$id), lapply(lines[nzchar(lines)], jsonlite::fromJSON))[[1]]
sim <- simulate_design(design, as.integer(kv$seed))
d <- sim$data
if (design$kind != "binomial_days") d$N <- NULL
dir.create(kv$out, showWarnings = FALSE, recursive = TRUE)
data.table::fwrite(d, file.path(kv$out, "data.csv"))
t <- sim$truth
jsonlite::write_json(list(design_id = design$id, seed = as.integer(kv$seed), up = t$up, down = t$down, invariant = t$invariant,
                          shift = t$shift, n_eligible = length(t$eligible)),
                     file.path(kv$out, "truth.json"), auto_unbox = TRUE, digits = NA, null = "null", na = "null")
