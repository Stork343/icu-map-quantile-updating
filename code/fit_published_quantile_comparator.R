Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
suppressPackageStartupMessages({
  library(data.table)
  library(lqmm)
  library(quantreg)
})

fit_calibration <- function(design, tau) {
  required <- c("y", "cal_intercept", "cal_predictor", "fit_weight")
  if (!all(required %in% names(design))) stop("Missing calibration columns")
  if (any(!is.finite(as.matrix(design[, ..required]))) || any(design$fit_weight <= 0)) stop("Invalid calibration input")
  x <- as.matrix(design[, .(cal_intercept, cal_predictor)]) * design$fit_weight
  y <- design$y * design$fit_weight
  if (qr(x)$rank != 2) stop("Singular affine calibration design")
  fit <- rq.fit(x, y, tau = tau, method = "br")
  residual <- y - drop(x %*% fit$coefficients)
  primal <- sum(residual * (tau - (residual < 0)))
  dual_score <- fit$dual - (1 - tau)
  dual <- sum(y * dual_score)
  stationarity <- max(abs(drop(crossprod(x, dual_score))))
  relative_gap <- abs(primal - dual) / max(1, abs(primal))
  if (any(!is.finite(fit$coefficients)) || any(!is.finite(dual_score)) ||
      any(fit$dual < -1e-8) || any(fit$dual > 1 + 1e-8) ||
      stationarity > 1e-6 || relative_gap > 1e-7) stop("Affine calibration primal/dual check failed")
  data.table(term = c("cal_intercept", "cal_predictor"), estimate = as.numeric(fit$coefficients),
             primal_loss = primal, dual_loss = dual, dual_stationarity = stationarity,
             relative_objective_gap = relative_gap, solver = "br")
}

fit_comparator <- function(design, tau, n_nodes) {
  required <- c("y", "stay_index", "age_z", "male", "emergency_or_urgent")
  if (!all(required %in% names(design))) stop("Missing LQMM design columns")
  if (any(!is.finite(as.matrix(design[, ..required])))) stop("Nonfinite LQMM input")
  fit <- lqmm(
    fixed = y ~ age_z + male + emergency_or_urgent,
    random = ~ 1, group = stay_index, covariance = "pdDiag",
    tau = tau, nK = n_nodes, type = "normal", rule = 1,
    data = design,
    control = lqmmControl(
      method = "df", LP_tol_ll = 1e-4, LP_max_iter = 3000,
      UP_tol = 1e-4, UP_max_iter = 40
    )
  )
  if (fit$opt$low_loop < 0 || fit$opt$upp_loop < 0) stop("LQMM optimization did not converge")
  if (any(!is.finite(fit$theta)) || !is.finite(fit$scale) || fit$scale <= 0) {
    stop("Invalid LQMM parameter estimates")
  }
  fit
}

validate_blp <- function() {
  data("Orthodont", package = "nlme", envir = environment())
  design <- as.data.table(Orthodont)
  design[, `:=`(
    y = distance, stay_index = as.integer(Subject), age_z = age - 11,
    male = as.integer(Sex == "Male"), emergency_or_urgent = as.integer(age >= 12)
  )]
  fit <- fit_comparator(design, 0.1, 11)
  variance <- as.numeric(VarCorr(fit))
  residual_mean <- as.numeric(meanAL(0, fit$scale, fit$tau))
  residual_variance <- as.numeric(varAL(fit$scale, fit$tau))
  fitted <- as.numeric(fit$mmf %*% fit$theta_x)
  reference <- ranef(fit)[, 1]
  manual <- tapply(fit$y - fitted - residual_mean, fit$group, function(value) {
    variance * sum(value) / (residual_variance + length(value) * variance)
  })
  difference <- max(abs(as.numeric(reference) - as.numeric(manual)))
  if (!is.finite(difference) || difference > 1e-10) stop("BLP differs from lqmm::ranef")
  calibration_design <- data.table(y = design$y, cal_intercept = 1,
                                   cal_predictor = design$age_z, fit_weight = 0.25)
  calibration <- fit_calibration(calibration_design, 0.1)
  if (calibration$relative_objective_gap[[1]] > 1e-7) stop("Calibration check failed")
  cat(sprintf("Official Orthodont data: BLP max absolute difference %.12g\n", difference))
  cat(sprintf("Official Orthodont data: affine primal/dual relative gap %.12g\n", calibration$relative_objective_gap[[1]]))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 1 && args[[1]] == "--self-test") {
    validate_blp()
    return(invisible(NULL))
  }
  if (length(args) == 4 && args[[1]] == "--calibrate") {
    calibration <- fit_calibration(fread(args[[2]]), as.numeric(args[[3]]))
    fwrite(calibration, args[[4]])
    return(invisible(NULL))
  }
  if (length(args) != 5) stop("Usage: Rscript fit_published_quantile_comparator.R design_csv tau n_nodes parameters_csv model_rds")
  if (args[[1]] == "--refit-from-model") {
    source_model <- readRDS(args[[2]])
    design <- data.table(
      y = source_model$y,
      stay_index = as.integer(source_model$group),
      age_z = source_model$mmf[, "age_z"],
      male = source_model$mmf[, "male"],
      emergency_or_urgent = source_model$mmf[, "emergency_or_urgent"]
    )
    fit <- fit_comparator(design, source_model$tau, as.integer(args[[3]]))
  } else {
    design <- fread(args[[1]])
    fit <- fit_comparator(design, as.numeric(args[[2]]), as.integer(args[[3]]))
  }
  parameters <- data.table(
    term = c("x_intercept", "age_z", "male", "emergency_or_urgent", "random_intercept_variance", "residual_scale", "residual_mean", "residual_variance", "log_likelihood", "lower_iterations", "upper_iterations", "quadrature_nodes"),
    estimate = c(as.numeric(fit$theta_x), as.numeric(VarCorr(fit)), fit$scale,
      meanAL(0, fit$scale, fit$tau), varAL(fit$scale, fit$tau), fit$logLik,
      fit$opt$low_loop, fit$opt$upp_loop, as.integer(args[[3]]))
  )
  parameters[term == "quadrature_nodes", estimate := as.integer(args[[3]])]
  if (nrow(parameters) != 12) stop("Unexpected LQMM parameter dimensions")
  fwrite(parameters, args[[4]])
  saveRDS(fit, args[[5]])
  print(parameters)
}

if (sys.nframe() == 0L) main()
