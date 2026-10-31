library(data.table)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: verify_control_clinical.R <private_results> <public_results> <report_csv>")
}
private <- args[[1]]
public <- args[[2]]
temporal <- fread(file.path(public, "clinical_temporal_paired.csv"))
nested <- fread(file.path(public, "clinical_nested_pairs.csv"))
if (nrow(temporal) != 24L || any(temporal$comparison_family_size != 24L)) {
  stop("The temporal comparison family differs from the fixed design")
}
models <- unique(c(temporal$first, temporal$second))
critical <- qnorm(1 - 0.05 / (2 * 24))
report <- vector("list", 24L)
pvalues <- numeric(24L)
for (position in seq_len(nrow(temporal))) {
  row <- temporal[position]
  horizon <- row$horizon_hours
  columns <- c("global_stay_index", paste0("loss_", models))
  assessment <- fread(file.path(private, "temporal", paste0("assessment_h", horizon, ".csv")), select = columns)
  if (anyDuplicated(assessment$global_stay_index)) {
    stop("The temporal assessment contains repeated individuals")
  }
  difference <- assessment[[paste0("loss_", row$first)]] - assessment[[paste0("loss_", row$second)]]
  estimate <- mean(difference)
  standard_error <- sd(difference) / sqrt(length(difference))
  if (standard_error == 0 && any(difference != 0)) {
    stop("A nonzero clinical contrast has zero estimated variance")
  }
  pvalues[position] <- if (standard_error == 0) 1 else 2 * pnorm(-abs(estimate / standard_error))
  expected <- c(estimate, standard_error, estimate - 1.96 * standard_error,
                estimate + 1.96 * standard_error,
                estimate - critical * standard_error, estimate + critical * standard_error,
                pvalues[position], mean(difference < 0))
  actual <- unlist(row[, c("mean_difference", "se", "ci95_low", "ci95_high",
                          "simultaneous_ci95_low", "simultaneous_ci95_high", "raw_p_value",
                          "benefit_fraction"), with = FALSE], use.names = FALSE)
  error <- max(abs(expected - actual))
  if (!is.finite(error) || error > 2e-12 || length(difference) != row$n_stays) {
    stop("Independent R clinical inference differs from the reported result")
  }
  fold_values <- vector("list", 5L)
  ids <- integer()
  for (fold in seq_len(5L)) {
    name <- paste0("fold_", fold)
    frame <- fread(file.path(private, name, paste0("assessment_h", horizon, ".csv")), select = columns)
    if (any(frame$global_stay_index %in% ids)) {
      stop("Nested assessment folds overlap")
    }
    ids <- c(ids, frame$global_stay_index)
    values <- frame[[paste0("loss_", row$first)]] - frame[[paste0("loss_", row$second)]]
    stored <- nested[partition == name & horizon_hours == horizon & first == row$first & second == row$second]
    if (nrow(stored) != 1L || abs(mean(values) - stored$mean_difference) > 2e-12 || length(values) != stored$n_stays) {
      stop("A nested fold summary differs from the saved patient comparisons")
    }
    fold_values[[fold]] <- values
  }
  combined <- unlist(fold_values, use.names = FALSE)
  stored <- nested[partition == "nested_pooled" & horizon_hours == horizon & first == row$first & second == row$second]
  if (nrow(stored) != 1L || abs(mean(combined) - stored$mean_difference) > 2e-12 || length(combined) != stored$n_stays) {
    stop("The nested pooled summary differs from the individual comparisons")
  }
  report[[position]] <- data.frame(horizon_hours = horizon, first = row$first, second = row$second,
                                  temporal_stays = length(difference), nested_stays = length(combined),
                                  maximum_absolute_error = error)
}
if (max(abs(p.adjust(pvalues, method = "holm") - temporal$holm_p_value)) > 2e-12) {
  stop("The independently calculated Holm adjustment differs")
}
write.csv(do.call(rbind, report), args[[3]], row.names = FALSE)
cat("Independent R checks passed for all 24 temporal comparisons and 144 nested summaries\n")
