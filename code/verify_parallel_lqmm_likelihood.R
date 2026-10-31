Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
suppressPackageStartupMessages({library(lqmm); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(6L, 7L)) stop("Supply code directory, actual model, nodes, workers, subset stays, output JSON and optional objectives-only mode")
source(file.path(args[1], "fit_random_slope_lqmm.R"))
source(file.path(args[1], "parallel_lqmm_likelihood.R"))
fit <- readRDS(args[2])
nodes <- as.integer(args[3])
workers <- as.integer(args[4])
subset_stays <- as.integer(args[5])
validate_slope_fit(fit)
if (nodes != 61L || workers != 8L || subset_stays < workers) stop("The prespecified precision validation requires 61 nodes and eight stay blocks")
quadrature <- getFromNamespace("quad", "lqmm")(q = fit$dim_theta[2], k = nodes, type = fit$type, rule = fit$rule)
objective <- make_parallel_lqmm_objective(fit, quadrature, workers)
records <- list()
for (point in seq_len(5L)) {
  theta <- fit$theta
  sigma <- fit$scale * c(1, .99, 1.01, 1.001, .999)[point]
  if (point > 1L) {
    parameter <- if (point == 5L) fit$dim_theta[1] + 1L else point - 1L
    theta[parameter] <- theta[parameter] + .001 * (-1)^point
  }
  kind <- if (point %% 2L == 1L) "theta" else "sigma"
  official <- getFromNamespace(if (kind == "theta") "loglik.t" else "loglik.s", "lqmm")
  started <- proc.time()[[3]]
  serial <- do.call(official, c(list(theta = theta, sigma = sigma), objective$data$whole, objective$data$common))
  serial_seconds <- proc.time()[[3]] - started
  started <- proc.time()[[3]]
  parallel_value <- if (kind == "theta") objective$theta(theta, sigma) else objective$sigma(sigma, theta)
  parallel_seconds <- proc.time()[[3]] - started
  difference <- abs(serial - parallel_value)
  record <- list(point = point, objective_kind = kind, sigma = sigma, serial = serial,
                 parallel = parallel_value, absolute_difference = difference,
                 serial_seconds = serial_seconds, parallel_seconds = parallel_seconds)
  print(record)
  if (!is.finite(difference) || difference > 1e-7) stop("Independent-stay likelihood blocks differ from the official whole-data likelihood")
  records[[point]] <- record
}
parallel::stopCluster(objective$cluster)
if (length(args) == 7L) {
  if (args[7] != "objectives-only") stop("Unknown likelihood validation mode")
  write_json(list(lqmm_version = as.character(packageVersion("lqmm")),
                  full_data_stays = fit$ngroups, full_data_observations = fit$nobs,
                  nodes_per_dimension = nodes, quadrature_nodes = nrow(quadrature$nodes), workers = workers,
                  official_namespace_likelihood_checks = records, unchanged_control = fit$control),
             args[6], auto_unbox = TRUE, pretty = TRUE, digits = NA)
  quit(save = "no", status = 0L)
}
sizes <- as.integer(table(fit$group))
selected <- seq_len(sum(head(sizes, subset_stays)))
design <- data.table(y = fit$y[selected], stay_index = as.integer(fit$group[selected]),
                     age_z = fit$mmf[selected, "age_z"], male = fit$mmf[selected, "male"],
                     emergency_or_urgent = fit$mmf[selected, "emergency_or_urgent"],
                     time_u = fit$mmf[selected, "time_u"])
initial <- fit_slope_model(design, fit$tau, 7L, stages = 7L)
serial_fit <- refine_slope_fit(initial, nodes)
parallel_fit <- refine_slope_fit_parallel(initial, nodes, workers)
serial_blp <- as.matrix(ranef(serial_fit))
parallel_blp <- as.matrix(ranef(parallel_fit))
comparison <- list(stays = subset_stays, observations = nrow(design),
                    max_theta_difference = max(abs(serial_fit$theta - parallel_fit$theta)),
                    scale_difference = abs(serial_fit$scale - parallel_fit$scale),
                    log_likelihood_difference = abs(serial_fit$logLik - parallel_fit$logLik),
                    max_blp_difference = max(abs(serial_blp - parallel_blp)),
                    serial_convergence = serial_fit$opt, parallel_convergence = parallel_fit$opt)
print(comparison)
differences <- unlist(comparison[c("max_theta_difference", "scale_difference", "log_likelihood_difference", "max_blp_difference")])
if (any(!is.finite(differences)) || max(differences) > 1e-7) {
  stop("Serial and parallel subset parameters, likelihood or BLP exceed the equivalence tolerance")
}
report <- list(lqmm_version = as.character(packageVersion("lqmm")),
                full_data_stays = fit$ngroups, full_data_observations = fit$nobs,
                nodes_per_dimension = nodes, quadrature_nodes = nrow(quadrature$nodes), workers = workers,
                official_namespace_likelihood_checks = records, real_clinical_subset_comparison = comparison,
                equivalence_tolerance = 1e-7, equivalence_passed = TRUE, unchanged_control = fit$control)
write_json(report, args[6], auto_unbox = TRUE, pretty = TRUE, digits = NA)
