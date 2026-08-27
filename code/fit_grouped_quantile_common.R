suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript fit_grouped_quantile_common.R <design_csv> <tau> <coef_csv> [group_col]")
}

design_csv <- args[[1]]
tau <- as.numeric(args[[2]])
coef_csv <- args[[3]]
group_col <- if (length(args) >= 4) args[[4]] else "group_id"

dt <- fread(design_csv)
if (!("y" %in% names(dt))) {
  stop("design_csv must contain a y column")
}
if (!(group_col %in% names(dt))) {
  stop(sprintf("design_csv must contain the group column '%s'", group_col))
}

predictor_cols <- setdiff(names(dt), c(group_col, "y"))
if (length(predictor_cols) == 0) {
  stop("No predictor columns found")
}

groups <- sort(unique(dt[[group_col]]))
out <- vector("list", length(groups))

for (idx in seq_along(groups)) {
  g <- groups[[idx]]
  local <- dt[dt[[group_col]] == g]
  x <- as.matrix(local[, ..predictor_cols])
  y <- local[["y"]]
  fit <- tryCatch(
    rq.fit(x, y, tau = tau, method = "fn"),
    error = function(e) rq.fit(x, y, tau = tau, method = "br")
  )
  out[[idx]] <- data.table(
    group_id = g,
    term = predictor_cols,
    estimate = as.numeric(fit$coefficients)
  )
}

fwrite(rbindlist(out), coef_csv)
