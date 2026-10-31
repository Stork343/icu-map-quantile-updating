args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("Supply code directory, completed model, initial model, expected nodes and output JSON")
source(file.path(args[1], "fit_random_slope_lqmm.R"))
fit <- readRDS(args[2])
initial <- readRDS(args[3])
nodes <- as.integer(args[4])
validate_slope_fit(fit)
fields <- c("y", "mmf", "mmr", "group", "weights", "control", "cov_name", "type", "rule", "dim_theta", "nobs", "ngroups")
unchanged <- vapply(fields, function(field) identical(fit[[field]], initial[[field]]), logical(1))
if (!all(unchanged)) stop("A completed random-slope model changed its fitting inputs or controls")
if (attr(fit$QUAD, "k") != nodes || nrow(fit$QUAD$nodes) != nodes^fit$dim_theta[2]) {
  stop("A completed random-slope model has an unexpected quadrature order")
}
report <- list(status = "verified", nodes_per_dimension = nodes,
                quadrature_nodes = nrow(fit$QUAD$nodes), fitting_stays = fit$ngroups,
                fitting_observations = fit$nobs, input_and_control_fields_identical = as.list(unchanged),
                finite_parameters = all(is.finite(fit$theta)), scale = fit$scale,
                covariance_minimum_eigenvalue = min(eigen(VarCorr(fit), symmetric = TRUE, only.values = TRUE)$values),
                optimization = fit$opt, stored_log_likelihood = fit$logLik)
jsonlite::write_json(report, args[5], auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(report)
