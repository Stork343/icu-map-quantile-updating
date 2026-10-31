args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) stop("Supply code directory, input model, nodes, workers, parameter CSV, output model and validation JSON")
source(file.path(args[1], "fit_random_slope_lqmm.R"))
source(file.path(args[1], "parallel_lqmm_likelihood.R"))
source(file.path(args[1], "export_random_slope_parameters.R"))
fit <- readRDS(args[2])
nodes <- as.integer(args[3])
workers <- as.integer(args[4])
validate_slope_fit(fit)
if (nodes <= attr(fit$QUAD, "k")) stop("Parallel refinement requires a higher quadrature order")
quadrature <- getFromNamespace("quad", "lqmm")(q = fit$dim_theta[2], k = nodes, type = fit$type, rule = fit$rule)
check <- make_parallel_lqmm_objective(fit, quadrature, workers)
official <- getFromNamespace("loglik.t", "lqmm")
serial_value <- do.call(official, c(list(theta = fit$theta, sigma = fit$scale), check$data$whole, check$data$common))
parallel_value <- check$theta(fit$theta, fit$scale)
parallel::stopCluster(check$cluster)
difference <- abs(serial_value - parallel_value)
if (!is.finite(difference) || difference > 1e-7) stop("Blocked likelihood does not match the official likelihood on this fitting dataset")
report <- list(lqmm_version = as.character(utils::packageVersion("lqmm")),
                fitting_stays = fit$ngroups, fitting_observations = fit$nobs,
                initial_nodes = attr(fit$QUAD, "k"), final_nodes = nodes,
                quadrature_nodes = nrow(quadrature$nodes), workers = workers,
                serial_negative_log_likelihood = serial_value,
                blocked_negative_log_likelihood = parallel_value,
                absolute_difference = difference, tolerance = 1e-7,
                unchanged_control = fit$control, status = "objective_verified")
dir.create(dirname(args[5]), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(report, args[7], auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(report)
started <- proc.time()
fit <- refine_slope_fit_parallel(fit, nodes, workers)
saveRDS(fit, args[6])
saveRDS(fit, paste0(args[6], ".stage_", nodes))
print(export_slope_parameters(fit, args[5]))
report$status <- "converged"
report$optimization <- fit$opt
report$elapsed <- unname((proc.time() - started)[[3]])
jsonlite::write_json(report, args[7], auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(proc.time() - started)
