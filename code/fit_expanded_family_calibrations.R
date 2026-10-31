suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected input CSV, tau, and output CSV")
dt <- fread(args[[1]])
tau <- as.numeric(args[[2]])
if (!all(c("y", "weight") %in% names(dt))) stop("Missing outcome or weight")
if (any(!is.finite(dt$weight)) || any(dt$weight <= 0)) stop("Invalid weights")
predictors <- setdiff(names(dt), c("y", "weight"))
output <- vector("list", length(predictors))
for (i in seq_along(predictors)) {
  candidate <- predictors[[i]]
  x <- cbind(1, dt[[candidate]])
  fit <- rq.fit(x * dt$weight, dt$y * dt$weight, tau = tau, method = "fn")
  if (any(!is.finite(fit$coefficients))) stop("Nonfinite calibration")
  output[[i]] <- data.table(candidate = candidate,
                           intercept = as.numeric(fit$coefficients[[1]]),
                           slope = as.numeric(fit$coefficients[[2]]))
}
fwrite(rbindlist(output), args[[3]])
