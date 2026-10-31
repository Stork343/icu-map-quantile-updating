args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: verify_control_summaries.R <simulation_directory> <report_csv>")
}
directory <- args[[1]]
paired <- read.csv(file.path(directory, "paired_replicates.csv"), check.names = FALSE)
methods <- read.csv(file.path(directory, "methods_replicates.csv"), check.names = FALSE)
summary <- read.csv(file.path(directory, "paired_summary.csv"), check.names = FALSE)
keys <- c("setting", "scope", "horizon_hours", "first", "second")
record_keys <- c(keys, "replicate")
if (nrow(paired) != 38400L || nrow(summary) != 192L || anyDuplicated(paired[record_keys])) {
  stop("The control comparison family is incomplete or duplicated")
}
groups <- split(seq_len(nrow(paired)), interaction(paired[keys], drop = TRUE))
if (length(groups) != 192L) {
  stop("The comparison family differs from the fixed design")
}
critical <- qt(1 - 0.05 / (2 * 192), df = 199)
report <- vector("list", length(groups))
position <- 0L
for (indices in groups) {
  local <- paired[indices, ]
  if (nrow(local) != 200L || !identical(sort(local$replicate), seq_len(200L))) {
    stop("A paired comparison does not retain all 200 independent datasets")
  }
  selected <- rep(TRUE, nrow(summary))
  for (name in keys) {
    selected <- selected & summary[[name]] == local[[name]][[1]]
  }
  observed <- summary[selected, ]
  if (nrow(observed) != 1L || any(!is.finite(local$mean_difference))) {
    stop("The paired comparison is not uniquely represented")
  }
  estimate <- mean(local$mean_difference)
  mcse <- sd(local$mean_difference) / sqrt(200)
  expected <- c(estimate, mcse, estimate - qt(0.975, 199) * mcse,
                estimate + qt(0.975, 199) * mcse,
                estimate - critical * mcse, estimate + critical * mcse,
                mean(local$mean_difference < 0),
                max((sum(local$mean_difference) - local$mean_difference) / 199))
  actual <- unlist(observed[c("mean_difference", "mcse", "pointwise_low", "pointwise_high",
                              "simultaneous_low", "simultaneous_high", "gain_frequency",
                              "leave_one_out_maximum_mean")], use.names = FALSE)
  error <- max(abs(expected - actual))
  if (!is.finite(error) || error > 2e-12) {
    stop("Independent R interval calculation differs from the exported summary")
  }
  position <- position + 1L
  report[[position]] <- data.frame(local[1, keys], independent_numeric_fields = 8L,
                                  maximum_absolute_difference = error)
}
join_keys <- c("setting", "scope", "horizon_hours", "replicate")
first <- methods[c(join_keys, "method", "mean_loss")]
second <- first
names(first)[names(first) == "method"] <- "first"
names(first)[names(first) == "mean_loss"] <- "first_loss"
names(second)[names(second) == "method"] <- "second"
names(second)[names(second) == "mean_loss"] <- "second_loss"
paired_methods <- merge(paired, first, by = c(join_keys, "first"), all.x = TRUE, sort = FALSE)
paired_methods <- merge(paired_methods, second, by = c(join_keys, "second"), all.x = TRUE, sort = FALSE)
if (nrow(paired_methods) != nrow(paired) ||
    any(!is.finite(paired_methods$first_loss)) || any(!is.finite(paired_methods$second_loss))) {
  stop("Method-level exports do not account for all paired records")
}
identity_error <- max(abs(paired_methods$first_loss - paired_methods$second_loss -
                           paired_methods$mean_difference))
if (identity_error > 2e-12) {
  stop("Paired contrasts differ from the two exported method means")
}
output <- do.call(rbind, report)
write.csv(output, args[[2]], row.names = FALSE)
cat("Independent R checks passed for 192 comparisons, 1536 summary values and 38400 paired identities\n")
cat("Maximum paired method identity error", format(identity_error, digits = 17), "\n")
cat("Bonferroni critical value", format(critical, digits = 17), "\n")
