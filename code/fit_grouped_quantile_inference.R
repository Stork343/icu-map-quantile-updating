suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
})
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(3L, 4L))) {
  stop("Usage: Rscript fit_grouped_quantile_inference.R <design_csv> <tau> <coef_csv>")
}

dt <- fread(args[[1]])
tau <- as.numeric(args[[2]])
predictor_cols <- c("intercept", "age_z", "male", "urgent")
if (!all(c("group_id", "y", predictor_cols) %in% names(dt))) {
  stop("Training design lacks required columns")
}
groups <- sort(unique(dt$group_id))
out <- vector("list", length(groups))
for (idx in seq_along(groups)) {
  g <- groups[[idx]]
  local <- dt[group_id == g]
  x <- as.matrix(local[, ..predictor_cols])
  if (qr(x)$rank != ncol(x)) {
    stop(sprintf("Singular design in group %s", g))
  }
  fit <- rq.fit(x, local$y, tau = tau, method = "br")
  coefficients <- as.numeric(fit$coefficients)
  if (length(coefficients) != length(predictor_cols) || any(!is.finite(coefficients))) {
    stop(sprintf("Invalid coefficients for group %s", g))
  }
  residuals <- local$y - drop(x %*% coefficients)
  primal_loss <- sum(residuals * (tau - (residuals < 0)))
  dual_score <- fit$dual - (1 - tau)
  dual_loss <- sum(local$y * dual_score)
  dual_stationarity <- max(abs(drop(crossprod(x, dual_score))))
  objective_gap <- abs(primal_loss - dual_loss)
  relative_gap <- objective_gap / max(1, abs(primal_loss))
  if (any(!is.finite(dual_score)) || any(fit$dual < -1e-8) || any(fit$dual > 1 + 1e-8) ||
      dual_stationarity > 1e-6 || relative_gap > 1e-7) {
    stop(sprintf("Invalid primal/dual certificate for group %s: stationarity=%g gap=%g", g, dual_stationarity, relative_gap))
  }
  out[[idx]] <- data.table(group_id = g, term = predictor_cols, estimate = coefficients)
  out[[idx]][, `:=`(solver = "br", n_observations = nrow(x), primal_loss = primal_loss,
                   dual_loss = dual_loss, dual_stationarity = dual_stationarity,
                   relative_objective_gap = relative_gap)]
}
fwrite(rbindlist(out), args[[3]])
