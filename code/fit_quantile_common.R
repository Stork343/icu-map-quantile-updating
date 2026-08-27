suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript fit_quantile_common.R <design_csv> <tau> <coef_csv>")
}

design_csv <- args[[1]]
tau <- as.numeric(args[[2]])
coef_csv <- args[[3]]

dt <- fread(design_csv)
if (!("y" %in% names(dt))) {
  stop("design_csv must contain a y column")
}
weight_col <- "fit_weight"
predictor_cols <- setdiff(
  names(dt),
  c("y", "stay_index", "stay_id", "time_hours", "index_flag", "late_flag", weight_col)
)
if (length(predictor_cols) == 0) {
  stop("No predictor columns found")
}

x <- as.matrix(dt[, ..predictor_cols])
y <- dt[["y"]]
weights <- if (weight_col %in% names(dt)) dt[[weight_col]] else rep(1.0, nrow(dt))
if (any(!is.finite(weights)) || any(weights <= 0)) {
  stop("fit_weight must be finite and strictly positive")
}
# Check loss is positively homogeneous: rho_tau(w * residual) = w * rho_tau(residual).
# Scaling y and every design row therefore gives the requested weighted objective.
fit <- rq.fit(x * weights, y * weights, tau = tau, method = "fn")
out <- data.table(term = predictor_cols, estimate = as.numeric(fit$coefficients))
fwrite(out, coef_csv)
