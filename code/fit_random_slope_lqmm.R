Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
suppressPackageStartupMessages({
  library(data.table)
  library(lqmm)
})

validate_slope_fit <- function(fit) {
  if (fit$opt$low_loop < 0 || fit$opt$upp_loop < 0) stop("Random-slope LQMM did not converge")
  covariance <- VarCorr(fit)
  if (!identical(dim(covariance), c(2L, 2L)) ||
      any(!is.finite(fit$theta)) || !is.finite(fit$scale) || fit$scale <= 0 ||
      min(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    stop("Invalid random-slope LQMM parameters")
  }
  invisible(NULL)
}

refine_slope_fit <- function(fit, nodes) {
  quadrature <- getFromNamespace("quad", "lqmm")(
    q = fit$dim_theta[2], k = nodes, type = fit$type, rule = fit$rule)
  initial <- list(theta = fit$theta, sigma = fit$scale)
  refined <- lqmm.fit.df(
    theta_0 = initial$theta, x = fit$mmf, y = fit$y, z = fit$mmr,
    weights = fit$weights, cov_name = fit$cov_name,
    V = quadrature$nodes, W = quadrature$weights,
    sigma_0 = initial$sigma, tau = fit$tau, group = fit$group,
    control = fit$control)
  for (name in names(refined)) fit[[name]] <- refined[[name]]
  fit$theta_x <- fit$theta[seq_len(fit$dim_theta[1])]
  fit$theta_z <- fit$theta[-seq_len(fit$dim_theta[1])]
  fit$QUAD <- quadrature
  fit$InitialPar <- initial
  fit$call$nK <- nodes
  validate_slope_fit(fit)
  fit
}

fit_slope_model <- function(design, tau, nodes, checkpoint = NULL, stages = NULL) {
  required <- c("y", "stay_index", "age_z", "male", "emergency_or_urgent", "time_u")
  if (!all(required %in% names(design))) stop("Missing random-slope design columns")
  if (any(!is.finite(as.matrix(design[, ..required])))) stop("Nonfinite random-slope design")
  if (is.null(stages)) {
    stages <- unique(c(7L, 15L, nodes))
    stages <- stages[stages <= nodes]
  }
  if (length(stages) < 1L || any(!is.finite(stages)) || any(stages < 7L) ||
      any(stages != as.integer(stages)) || any(diff(stages) <= 0L) || tail(stages, 1L) != nodes) {
    stop("Quadrature stages must increase to the requested final order")
  }
  fit <- lqmm(
    fixed = y ~ age_z + male + emergency_or_urgent + time_u,
    random = ~ time_u, group = stay_index, covariance = "pdSymm",
    tau = tau, nK = stages[1], type = "normal", rule = 1, data = design,
    control = lqmmControl(method = "df", LP_tol_ll = 1e-5,
                         LP_max_iter = 5000, UP_tol = 1e-5, UP_max_iter = 60,
                         verbose = TRUE)
  )
  validate_slope_fit(fit)
  if (!is.null(checkpoint)) saveRDS(fit, paste0(checkpoint, ".stage_", stages[1]))
  if (length(stages) > 1L) {
    for (stage in stages[-1]) {
      cat(sprintf("Refining random-slope quadrature to %d nodes per dimension\n", stage))
      fit <- refine_slope_fit(fit, stage)
      if (!is.null(checkpoint)) saveRDS(fit, paste0(checkpoint, ".stage_", stage))
    }
  }
  fit
}

blp_coefficients <- function(y, x, z, beta, covariance, scale, tau) {
  psi <- varAL(scale, tau)
  residual <- y - drop(x %*% beta) - meanAL(0, scale, tau)
  solve(diag(psi, 2) + covariance %*% crossprod(z), covariance %*% crossprod(z, residual))
}

self_test <- function() {
  data("Orthodont", package = "nlme", envir = environment())
  design <- as.data.table(Orthodont)
  design[, `:=`(y = distance, stay_index = as.integer(Subject), age_z = age - 11,
                 male = as.integer(Sex == "Male"), emergency_or_urgent = 0,
                 time_u = (age - 11) / 4)]
  fit <- lqmm(y ~ male + time_u, random = ~ time_u, group = stay_index,
              covariance = "pdSymm", tau = 0.1, nK = 21,
              data = design, control = lqmmControl(method = "df", LP_max_iter = 5000,
                                                   UP_max_iter = 60, LP_tol_ll = 1e-5))
  if (fit$opt$low_loop < 0 || fit$opt$upp_loop < 0) stop("Official-data random-slope fit failed")
  reference <- as.matrix(ranef(fit))
  groups <- split(seq_along(fit$y), fit$group)
  manual <- t(vapply(groups, function(index) {
    drop(blp_coefficients(fit$y[index], fit$mmf[index, , drop = FALSE],
                          fit$mmr[index, , drop = FALSE], fit$theta_x,
                          VarCorr(fit), fit$scale, fit$tau))
  }, numeric(2)))
  error <- max(abs(manual - reference))
  if (!is.finite(error) || error > 1e-9) stop("Random-slope BLP differs from lqmm::ranef")
  fit$control <- lqmmControl(method = "df", LP_max_iter = 5000,
                             UP_max_iter = 60, LP_tol_ll = 1e-5)
  refined <- refine_slope_fit(fit, 31L)
  if (nrow(refined$QUAD$nodes) != 31L^2L ||
      !identical(refined$mmf, fit$mmf) || !identical(refined$group, fit$group)) {
    stop("Random-slope quadrature continuation changed the fitted design")
  }
  continued <- as.matrix(ranef(refined))
  independent <- t(vapply(groups, function(index) {
    drop(blp_coefficients(refined$y[index], refined$mmf[index, , drop = FALSE],
                          refined$mmr[index, , drop = FALSE], refined$theta_x,
                          VarCorr(refined), refined$scale, refined$tau))
  }, numeric(2)))
  if (max(abs(continued - independent)) > 1e-9) stop("Continued random-slope BLP check failed")
  cat(sprintf("Official Orthodont random-slope BLP maximum error %.12g\n", error))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 1 && args[[1]] == "--self-test") {
    self_test()
    return(invisible(NULL))
  }
  if (!length(args) %in% c(5L, 6L)) stop("Supply design CSV, tau, nodes, parameter CSV and model RDS")
  design <- fread(args[[1]])
  tau <- as.numeric(args[[2]])
  nodes <- as.integer(args[[3]])
  stages <- if (length(args) == 6L) scan(text = args[[6]], what = integer(), sep = ",", quiet = TRUE) else NULL
  fit <- fit_slope_model(design, tau, nodes, args[[5]], stages)
  covariance <- VarCorr(fit)
  output <- data.table(
    term = c("x_intercept", "age_z", "male", "emergency_or_urgent", "time_u",
             "variance_level", "covariance_level_slope", "variance_slope",
             "residual_scale", "residual_mean", "residual_variance", "log_likelihood",
             "lower_iterations", "upper_iterations", "nodes_per_dimension", "quadrature_nodes"),
    estimate = c(fit$theta_x, covariance[1, 1], covariance[1, 2], covariance[2, 2],
                 fit$scale, meanAL(0, fit$scale, fit$tau), varAL(fit$scale, fit$tau), fit$logLik,
                 fit$opt$low_loop, fit$opt$upp_loop, nodes, nrow(fit$QUAD$nodes))
  )
  if (nrow(output) != 16 || any(!is.finite(output$estimate))) stop("Invalid parameter export")
  fwrite(output, args[[4]])
  saveRDS(fit, args[[5]])
  print(output)
}

if (sys.nframe() == 0L) main()
