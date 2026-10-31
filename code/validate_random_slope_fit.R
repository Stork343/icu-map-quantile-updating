Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
suppressPackageStartupMessages({
  library(data.table)
  library(lqmm)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) stop("Supply design, tau, nodes, parameters, model and validation output")
design <- fread(args[[1]])
tau <- as.numeric(args[[2]])
nodes <- as.integer(args[[3]])
parameters <- fread(args[[4]])
fit <- readRDS(args[[5]])
required <- c("y", "stay_index", "age_z", "male", "emergency_or_urgent", "time_u")
if (!all(required %in% names(design))) stop("Current fitting design lacks required columns")
if (any(!is.finite(as.matrix(design[, ..required])))) stop("Current fitting design is nonfinite")
order_index <- order(factor(design$stay_index))
design <- design[order_index]
x <- cbind("(Intercept)" = 1, as.matrix(design[, .(age_z, male, emergency_or_urgent, time_u)]))
z <- cbind("(Intercept)" = 1, "time_u" = design$time_u)
current_group <- as.integer(factor(design$stay_index))
dimensions_match <- nrow(design) == fit$nobs && length(fit$y) == fit$nobs &&
  identical(dim(x), dim(fit$mmf)) && identical(dim(z), dim(fit$mmr)) &&
  length(unique(current_group)) == fit$ngroups && identical(colnames(x), colnames(fit$mmf))
if (!dimensions_match) stop("Saved LQMM dimensions differ from the current fitting design")
outcome_error <- max(abs(design$y - fit$y))
covariate_error <- max(abs(x - fit$mmf))
random_design_error <- max(abs(z - fit$mmr))
if (outcome_error > 1e-12 || covariate_error > 1e-12 || random_design_error > 1e-12 ||
    !identical(current_group, as.integer(factor(fit$group)))) {
  stop("Saved LQMM records, covariates or group membership differ from the current fitting design")
}
if (fit$tau != tau || nrow(fit$QUAD$nodes) != nodes^2 || attr(fit$QUAD, "k") != nodes ||
    fit$type != "normal" || fit$cov_name != "pdSymm" || fit$rule != 1 ||
    fit$opt$low_loop <= 0 || fit$opt$upp_loop <= 0 || any(fit$weights != 1)) {
  stop("Saved LQMM settings or optimizer state differ from the requested model")
}
covariance <- VarCorr(fit)
expected <- c(as.numeric(fit$theta_x), covariance[1, 1], covariance[1, 2], covariance[2, 2], fit$scale,
              meanAL(0, fit$scale, fit$tau), varAL(fit$scale, fit$tau), fit$logLik,
              fit$opt$low_loop, fit$opt$upp_loop, nodes, nrow(fit$QUAD$nodes))
expected_terms <- c("x_intercept", "age_z", "male", "emergency_or_urgent", "time_u",
                    "variance_level", "covariance_level_slope", "variance_slope",
                    "residual_scale", "residual_mean", "residual_variance", "log_likelihood",
                    "lower_iterations", "upper_iterations", "nodes_per_dimension", "quadrature_nodes")
if (!identical(parameters$term, expected_terms) || length(expected) != nrow(parameters) ||
    any(!is.finite(expected)) || fit$scale <= 0 ||
    min(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values) <= 0) stop("Invalid saved LQMM parameter structure")
parameter_error <- max(abs(parameters$estimate - expected) / pmax(1, abs(expected)))
if (parameter_error > 1e-10) stop("Saved LQMM parameter CSV differs from the fitted object")
fwrite(data.table(n_fitting_stays = fit$ngroups, n_observations = fit$nobs, tau = tau,
                  nodes_per_dimension = nodes, quadrature_nodes = nodes^2,
                  maximum_outcome_difference = outcome_error,
                  maximum_covariate_difference = covariate_error,
                  maximum_random_design_difference = random_design_error,
                  maximum_relative_parameter_difference = parameter_error,
                  group_membership_matches = TRUE, dimensions_match = TRUE,
                  lower_iterations = fit$opt$low_loop, upper_iterations = fit$opt$upp_loop,
                  log_likelihood = fit$logLik, validation_passed = TRUE), args[[6]])
