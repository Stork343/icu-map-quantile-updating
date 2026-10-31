initialize_lqmm_worker <- function(block, common) {
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
  loadNamespace("lqmm")
  assign(".lqmm_independent_block", block, envir = .GlobalEnv)
  assign(".lqmm_common_design", common, envir = .GlobalEnv)
  NULL
}


evaluate_lqmm_worker <- function(theta, sigma, kind) {
  block <- get(".lqmm_independent_block", envir = .GlobalEnv)
  common <- get(".lqmm_common_design", envir = .GlobalEnv)
  objective <- getFromNamespace(if (kind == "theta") "loglik.t" else "loglik.s", "lqmm")
  do.call(objective, c(list(theta = theta, sigma = sigma), block, common))
}


make_lqmm_likelihood_blocks <- function(fit, quadrature, workers = 8L) {
  if (!inherits(fit, "lqmm") || length(fit$tau) != 1L || workers < 1L) stop("Invalid lqmm likelihood input")
  group <- as.integer(fit$group)
  group_sizes <- as.integer(table(group))
  runs <- rle(group)
  if (!identical(group_sizes, as.integer(runs$lengths)) || anyDuplicated(runs$values) ||
      is.unsorted(group) || length(fit$weights) != length(group_sizes)) {
    stop("Likelihood blocks require the ordered, contiguous stay records of a fitted lqmm object")
  }
  groups <- length(group_sizes)
  if (workers > groups) stop("Worker count exceeds the number of independent stays")
  p <- ncol(fit$mmf)
  q <- ncol(fit$mmr)
  common <- list(Tq = getFromNamespace("Tfun", "lqmm")(n = q, type = fit$cov_name),
                 V = quadrature$nodes, W = quadrature$weights, tau = fit$tau,
                 p = p, q = q, m = getFromNamespace("theta.z.dim", "lqmm")(fit$cov_name, q),
                 Kq = nrow(quadrature$nodes))
  end_rows <- cumsum(group_sizes)
  start_rows <- c(1L, head(end_rows, -1L) + 1L)
  allocation <- cut(seq_len(groups), breaks = workers, labels = FALSE)
  blocks <- lapply(seq_len(workers), function(worker) {
    selected_groups <- which(allocation == worker)
    selected_rows <- seq.int(start_rows[min(selected_groups)], end_rows[max(selected_groups)])
    sizes <- group_sizes[selected_groups]
    list(x = fit$mmf[selected_rows, , drop = FALSE], y = fit$y[selected_rows],
         z = fit$mmr[selected_rows, , drop = FALSE], weights = fit$weights[selected_groups],
         M = length(sizes), N = length(selected_rows), minn = c(1L, head(cumsum(sizes), -1L) + 1L),
         maxn = cumsum(sizes))
  })
  whole <- list(x = fit$mmf, y = fit$y, z = fit$mmr, weights = fit$weights,
                M = groups, N = length(fit$y), minn = start_rows, maxn = end_rows)
  list(common = common, blocks = blocks, whole = whole)
}


make_parallel_lqmm_objective <- function(fit, quadrature, workers = 8L) {
  data <- make_lqmm_likelihood_blocks(fit, quadrature, workers)
  cluster <- parallel::makePSOCKcluster(workers, outfile = "")
  parallel::clusterMap(cluster, initialize_lqmm_worker, data$blocks, rep(list(data$common), workers), SIMPLIFY = FALSE)
  evaluate <- function(theta, sigma, kind) {
    values <- unlist(parallel::clusterCall(cluster, evaluate_lqmm_worker, theta, sigma, kind), use.names = FALSE)
    if (length(values) != workers || any(!is.finite(values))) stop("Nonfinite official likelihood block")
    sum(values)
  }
  list(cluster = cluster, data = data,
       theta = function(theta, sigma) evaluate(theta, sigma, "theta"),
       sigma = function(sigma, theta) evaluate(theta, sigma, "sigma"))
}


fit_lqmm_parallel_parameters <- function(fit, objective) {
  control <- fit$control
  if (control$method != "df" || control$UP_max_iter <= 0L) stop("Parallel continuation requires derivative-free lqmm controls")
  theta <- fit$theta
  scale <- fit$scale
  converged <- FALSE
  # 每个stay的C likelihood由官方namespace计算，参数与scale迭代沿用lqmm.fit.df的控制。
  for (upper in seq_len(control$UP_max_iter)) {
    if (control$verbose) cat(sprintf("Parallel upper iteration %d\n", upper))
    parameter_fit <- stats::optim(theta, objective$theta, sigma = scale, method = "Nelder-Mead",
                                  control = list(maxit = control$LP_max_iter, abstol = control$LP_tol_ll,
                                                 trace = control$verbose))
    next_theta <- parameter_fit$par
    scale_fit <- stats::optimize(objective$sigma, interval = c(.Machine$double.eps, 1000 * scale),
                                 theta = next_theta)
    next_scale <- scale_fit$minimum
    if (control$verbose) cat(sprintf("Parallel negative log likelihood %.12f, scale %.12f\n", parameter_fit$value, next_scale))
    if (abs(next_scale - scale) < control$UP_tol) {
      converged <- TRUE
      break
    }
    theta <- next_theta
    scale <- next_scale
  }
  lower <- if (parameter_fit$convergence == 1L) -1 else as.numeric(parameter_fit$counts[1])
  optimization <- list(low_loop = lower, upp_loop = if (converged) upper else -1)
  getFromNamespace("errorHandling", "lqmm")(optimization$low_loop, "low", control$LP_max_iter, control$LP_tol_ll, "lqmm")
  getFromNamespace("errorHandling", "lqmm")(optimization$upp_loop, "upp", control$UP_max_iter, control$UP_tol, "lqmm")
  list(theta = next_theta, scale = next_scale, logLik = -parameter_fit$value, opt = optimization)
}


refine_slope_fit_parallel <- function(fit, nodes, workers = 8L) {
  validate_slope_fit(fit)
  quadrature <- getFromNamespace("quad", "lqmm")(q = fit$dim_theta[2], k = nodes, type = fit$type, rule = fit$rule)
  initial <- list(theta = fit$theta, sigma = fit$scale)
  objective <- make_parallel_lqmm_objective(fit, quadrature, workers)
  on.exit(parallel::stopCluster(objective$cluster), add = TRUE)
  updated <- fit_lqmm_parallel_parameters(fit, objective)
  fit[names(updated)] <- updated
  fixed_dimension <- seq_len(fit$dim_theta[1])
  fit$theta_x <- fit$theta[fixed_dimension]
  fit$theta_z <- fit$theta[-fixed_dimension]
  fit$QUAD <- quadrature
  fit$InitialPar <- initial
  fit$call$nK <- nodes
  validate_slope_fit(fit)
  fit
}
