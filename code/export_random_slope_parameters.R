suppressPackageStartupMessages({library(lqmm); library(data.table)})
export_slope_parameters <- function(fit, path) {
  if (!inherits(fit, "lqmm") || fit$opt$low_loop < 0 || fit$opt$upp_loop < 0) {
    stop("A converged lqmm model is required")
  }
  covariance <- VarCorr(fit)
  if (!identical(dim(covariance), c(2L, 2L))) stop("The saved model requires two random coefficients")
  output <- data.table(
  term = c("x_intercept", "age_z", "male", "emergency_or_urgent", "time_u",
    "variance_level", "covariance_level_slope", "variance_slope", "residual_scale",
    "residual_mean", "residual_variance", "log_likelihood", "lower_iterations",
    "upper_iterations", "nodes_per_dimension", "quadrature_nodes"),
  estimate = c(fit$theta_x, covariance[1, 1], covariance[1, 2], covariance[2, 2],
    fit$scale, meanAL(0, fit$scale, fit$tau), varAL(fit$scale, fit$tau), fit$logLik,
    fit$opt$low_loop, fit$opt$upp_loop, attr(fit$QUAD, "k"), nrow(fit$QUAD$nodes)))
  if (nrow(output) != 16L || any(!is.finite(output$estimate))) stop("Parameter export is invalid")
  fwrite(output, path)
  invisible(output)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2L) stop("Supply saved fitted model and parameter CSV")
  export_slope_parameters(readRDS(args[1]), args[2])
}

if (sys.nframe() == 0L) main()
