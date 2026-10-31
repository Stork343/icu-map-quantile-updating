args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
data <- read.csv(args[1], stringsAsFactors = FALSE)
stopifnot(all(is.finite(data$mean_difference)), all(is.finite(data$se)), all(data$se >= 0))
data$raw_p_value <- ifelse(data$se > 0, 2 * pnorm(-abs(data$mean_difference / data$se)),
                         ifelse(data$mean_difference == 0, 1, 0))
data$holm_p_value <- NA_real_
data$family_size <- NA_integer_
for (indices in split(seq_len(nrow(data)), data$comparison_family)) {
    data$holm_p_value[indices] <- p.adjust(data$raw_p_value[indices], method = "holm")
    data$family_size[indices] <- length(indices)
}
data$holm_reject_005 <- data$holm_p_value < 0.05
write.csv(data, args[2], row.names = FALSE, na = "")
