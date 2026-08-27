# ADEMP v2 split-window simulation

## Monte Carlo design

- Independent unit: stay; independent Monte Carlo unit: complete simulated dataset.
- Replicates per scenario: 200.
- Target quantile: 0.1.
- Losses and calibration probabilities are first averaged within stay and then across stays.
- A discrete quantile obeys the probability-mass bracket P(Y < q) <= tau <= P(Y <= q); probability-mass bracket violation is the distance outside that bracket.
- Parenthesized values below are Monte Carlo standard errors across independent datasets.

## Performance summary

| Scenario | Method | Effective/attempted | Failure (%) | Loss (MCSE) | Regret (MCSE) | P(Y<q) | P(Y<=q) | Probability-mass bracket violation (MCSE) | Offset RMSE (MCSE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ideal: large, dense | oracle | 200/200 | 0.0 | 0.963 (0.002) | 0.000 (0.000) | 0.100 | 0.100 | 0.007 (0.000) | -- |
| Ideal: large, dense | population | 200/200 | 0.0 | 1.370 (0.004) | 0.407 (0.004) | 0.101 | 0.101 | 0.014 (0.001) | -- |
| Ideal: large, dense | raw_index_q10 | 200/200 | 0.0 | 1.085 (0.003) | 0.122 (0.002) | 0.127 | 0.127 | 0.027 (0.001) | -- |
| Ideal: large, dense | affine_calibrated_q10 | 200/200 | 0.0 | 1.061 (0.002) | 0.098 (0.002) | 0.102 | 0.102 | 0.011 (0.001) | -- |
| Ideal: large, dense | unpenalized_level | 200/200 | 0.0 | 1.085 (0.003) | 0.122 (0.002) | 0.127 | 0.127 | 0.027 (0.001) | 4.144 (0.024) |
| Ideal: large, dense | tuned_level | 200/200 | 0.0 | 1.064 (0.002) | 0.101 (0.002) | 0.114 | 0.114 | 0.016 (0.001) | 3.752 (0.023) |
| Ideal: large, dense | tuned_level_slope | 200/200 | 0.0 | 1.154 (0.003) | 0.191 (0.002) | 0.091 | 0.091 | 0.012 (0.001) | 3.864 (0.018) |
| Ideal: small, dense | oracle | 200/200 | 0.0 | 0.967 (0.003) | 0.000 (0.000) | 0.099 | 0.099 | 0.011 (0.001) | -- |
| Ideal: small, dense | population | 200/200 | 0.0 | 1.383 (0.008) | 0.416 (0.007) | 0.106 | 0.106 | 0.023 (0.001) | -- |
| Ideal: small, dense | raw_index_q10 | 200/200 | 0.0 | 1.088 (0.004) | 0.121 (0.003) | 0.126 | 0.126 | 0.027 (0.001) | -- |
| Ideal: small, dense | affine_calibrated_q10 | 200/200 | 0.0 | 1.069 (0.004) | 0.102 (0.002) | 0.100 | 0.100 | 0.018 (0.001) | -- |
| Ideal: small, dense | unpenalized_level | 200/200 | 0.0 | 1.088 (0.004) | 0.121 (0.003) | 0.126 | 0.126 | 0.027 (0.001) | 4.129 (0.038) |
| Ideal: small, dense | tuned_level | 200/200 | 0.0 | 1.073 (0.004) | 0.106 (0.003) | 0.112 | 0.112 | 0.020 (0.001) | 3.770 (0.038) |
| Ideal: small, dense | tuned_level_slope | 200/200 | 0.0 | 1.158 (0.004) | 0.191 (0.003) | 0.094 | 0.094 | 0.018 (0.001) | 3.848 (0.030) |
| Ideal: large, sparse | oracle | 200/200 | 0.0 | 0.968 (0.003) | 0.000 (0.000) | 0.101 | 0.101 | 0.010 (0.001) | -- |
| Ideal: large, sparse | population | 200/200 | 0.0 | 1.372 (0.005) | 0.404 (0.005) | 0.102 | 0.102 | 0.016 (0.001) | -- |
| Ideal: large, sparse | raw_index_q10 | 200/200 | 0.0 | 1.476 (0.007) | 0.508 (0.007) | 0.241 | 0.241 | 0.141 (0.002) | -- |
| Ideal: large, sparse | affine_calibrated_q10 | 200/200 | 0.0 | 1.146 (0.003) | 0.178 (0.003) | 0.101 | 0.101 | 0.016 (0.001) | -- |
| Ideal: large, sparse | unpenalized_level | 200/200 | 0.0 | 1.476 (0.007) | 0.508 (0.007) | 0.241 | 0.241 | 0.141 (0.002) | 6.654 (0.034) |
| Ideal: large, sparse | tuned_level | 200/200 | 0.0 | 1.204 (0.004) | 0.236 (0.003) | 0.125 | 0.125 | 0.029 (0.001) | 4.666 (0.022) |
| Ideal: large, sparse | tuned_level_slope | 200/200 | 0.0 | 1.260 (0.004) | 0.292 (0.004) | 0.096 | 0.096 | 0.014 (0.001) | 4.752 (0.022) |
| Serial dependence | oracle | 200/200 | 0.0 | 0.962 (0.002) | 0.000 (0.000) | 0.101 | 0.101 | 0.010 (0.001) | -- |
| Serial dependence | population | 200/200 | 0.0 | 1.370 (0.005) | 0.407 (0.005) | 0.101 | 0.101 | 0.015 (0.001) | -- |
| Serial dependence | raw_index_q10 | 200/200 | 0.0 | 1.200 (0.005) | 0.238 (0.004) | 0.167 | 0.167 | 0.067 (0.001) | -- |
| Serial dependence | affine_calibrated_q10 | 200/200 | 0.0 | 1.103 (0.003) | 0.141 (0.003) | 0.101 | 0.101 | 0.017 (0.001) | -- |
| Serial dependence | unpenalized_level | 200/200 | 0.0 | 1.200 (0.005) | 0.238 (0.004) | 0.167 | 0.167 | 0.067 (0.001) | 5.219 (0.027) |
| Serial dependence | tuned_level | 200/200 | 0.0 | 1.135 (0.004) | 0.173 (0.003) | 0.124 | 0.124 | 0.029 (0.002) | 4.356 (0.033) |
| Serial dependence | tuned_level_slope | 200/200 | 0.0 | 1.177 (0.003) | 0.214 (0.003) | 0.100 | 0.100 | 0.014 (0.001) | 4.217 (0.020) |
| Integer MAP rounding | oracle | 200/200 | 0.0 | 0.963 (0.002) | 0.000 (0.000) | 0.084 | 0.117 | 0.000 (0.000) | -- |
| Integer MAP rounding | population | 200/200 | 0.0 | 1.377 (0.005) | 0.414 (0.005) | 0.101 | 0.101 | 0.015 (0.001) | -- |
| Integer MAP rounding | raw_index_q10 | 200/200 | 0.0 | 1.133 (0.003) | 0.170 (0.003) | 0.123 | 0.158 | 0.023 (0.001) | -- |
| Integer MAP rounding | affine_calibrated_q10 | 200/200 | 0.0 | 1.089 (0.003) | 0.126 (0.002) | 0.099 | 0.102 | 0.012 (0.001) | -- |
| Integer MAP rounding | unpenalized_level | 200/200 | 0.0 | 1.133 (0.003) | 0.170 (0.003) | 0.123 | 0.158 | 0.023 (0.001) | 4.574 (0.026) |
| Integer MAP rounding | tuned_level | 200/200 | 0.0 | 1.105 (0.003) | 0.142 (0.002) | 0.108 | 0.135 | 0.012 (0.001) | 4.102 (0.025) |
| Integer MAP rounding | tuned_level_slope | 200/200 | 0.0 | 1.177 (0.003) | 0.215 (0.003) | 0.094 | 0.094 | 0.012 (0.001) | 4.062 (0.021) |
| Informative monitoring | oracle | 200/200 | 0.0 | 0.961 (0.003) | 0.000 (0.000) | 0.103 | 0.103 | 0.011 (0.001) | -- |
| Informative monitoring | population | 200/200 | 0.0 | 1.504 (0.005) | 0.543 (0.005) | 0.045 | 0.045 | 0.055 (0.001) | -- |
| Informative monitoring | raw_index_q10 | 200/200 | 0.0 | 1.324 (0.007) | 0.363 (0.006) | 0.202 | 0.202 | 0.102 (0.002) | -- |
| Informative monitoring | affine_calibrated_q10 | 200/200 | 0.0 | 1.113 (0.003) | 0.151 (0.003) | 0.099 | 0.099 | 0.018 (0.001) | -- |
| Informative monitoring | unpenalized_level | 200/200 | 0.0 | 1.324 (0.007) | 0.363 (0.006) | 0.202 | 0.202 | 0.102 (0.002) | 8.928 (0.044) |
| Informative monitoring | tuned_level | 200/200 | 0.0 | 1.182 (0.004) | 0.221 (0.004) | 0.091 | 0.091 | 0.018 (0.001) | 6.031 (0.050) |
| Informative monitoring | tuned_level_slope | 200/200 | 0.0 | 1.383 (0.004) | 0.421 (0.004) | 0.046 | 0.046 | 0.054 (0.001) | 5.036 (0.020) |
| Cluster size--level association | oracle | 200/200 | 0.0 | 0.959 (0.002) | 0.000 (0.000) | 0.099 | 0.099 | 0.008 (0.000) | -- |
| Cluster size--level association | population | 200/200 | 0.0 | 1.387 (0.003) | 0.428 (0.004) | 0.071 | 0.071 | 0.029 (0.001) | -- |
| Cluster size--level association | raw_index_q10 | 200/200 | 0.0 | 1.095 (0.003) | 0.136 (0.002) | 0.129 | 0.129 | 0.029 (0.001) | -- |
| Cluster size--level association | affine_calibrated_q10 | 200/200 | 0.0 | 1.058 (0.002) | 0.099 (0.001) | 0.100 | 0.100 | 0.013 (0.001) | -- |
| Cluster size--level association | unpenalized_level | 200/200 | 0.0 | 1.095 (0.003) | 0.136 (0.002) | 0.129 | 0.129 | 0.029 (0.001) | 5.369 (0.030) |
| Cluster size--level association | tuned_level | 200/200 | 0.0 | 1.084 (0.003) | 0.125 (0.002) | 0.108 | 0.108 | 0.014 (0.001) | 4.896 (0.035) |
| Cluster size--level association | tuned_level_slope | 200/200 | 0.0 | 1.211 (0.002) | 0.252 (0.003) | 0.068 | 0.068 | 0.032 (0.001) | 4.533 (0.021) |
| Misspecified common time | oracle | 200/200 | 0.0 | 0.967 (0.002) | 0.000 (0.000) | 0.101 | 0.101 | 0.008 (0.000) | -- |
| Misspecified common time | population | 200/200 | 0.0 | 1.437 (0.006) | 0.470 (0.005) | 0.146 | 0.146 | 0.046 (0.001) | -- |
| Misspecified common time | raw_index_q10 | 200/200 | 0.0 | 1.705 (0.007) | 0.738 (0.007) | 0.335 | 0.335 | 0.235 (0.001) | -- |
| Misspecified common time | affine_calibrated_q10 | 200/200 | 0.0 | 1.113 (0.003) | 0.146 (0.002) | 0.102 | 0.102 | 0.013 (0.001) | -- |
| Misspecified common time | unpenalized_level | 200/200 | 0.0 | 1.705 (0.007) | 0.738 (0.007) | 0.335 | 0.335 | 0.235 (0.001) | 6.453 (0.030) |
| Misspecified common time | tuned_level | 200/200 | 0.0 | 1.310 (0.004) | 0.344 (0.004) | 0.181 | 0.181 | 0.081 (0.002) | 4.701 (0.021) |
| Misspecified common time | tuned_level_slope | 200/200 | 0.0 | 1.304 (0.004) | 0.337 (0.004) | 0.174 | 0.174 | 0.074 (0.001) | 4.539 (0.020) |
| Persistent level plus shape | oracle | 200/200 | 0.0 | 0.961 (0.002) | 0.000 (0.000) | 0.101 | 0.101 | 0.007 (0.000) | -- |
| Persistent level plus shape | population | 200/200 | 0.0 | 1.220 (0.003) | 0.259 (0.003) | 0.103 | 0.103 | 0.011 (0.001) | -- |
| Persistent level plus shape | raw_index_q10 | 200/200 | 0.0 | 1.551 (0.007) | 0.590 (0.007) | 0.193 | 0.193 | 0.093 (0.001) | -- |
| Persistent level plus shape | affine_calibrated_q10 | 200/200 | 0.0 | 1.254 (0.004) | 0.293 (0.004) | 0.104 | 0.104 | 0.016 (0.001) | -- |
| Persistent level plus shape | unpenalized_level | 200/200 | 0.0 | 1.551 (0.007) | 0.590 (0.007) | 0.193 | 0.193 | 0.093 (0.001) | 6.194 (0.028) |
| Persistent level plus shape | tuned_level | 200/200 | 0.0 | 1.219 (0.003) | 0.258 (0.003) | 0.105 | 0.105 | 0.012 (0.001) | 3.988 (0.019) |
| Persistent level plus shape | tuned_level_slope | 200/200 | 0.0 | 1.211 (0.004) | 0.250 (0.003) | 0.111 | 0.111 | 0.015 (0.001) | 3.968 (0.019) |
| Treatment feedback | oracle | 200/200 | 0.0 | 0.960 (0.002) | 0.000 (0.000) | 0.099 | 0.099 | 0.008 (0.000) | -- |
| Treatment feedback | population | 200/200 | 0.0 | 1.272 (0.003) | 0.312 (0.003) | 0.081 | 0.081 | 0.021 (0.001) | -- |
| Treatment feedback | raw_index_q10 | 200/200 | 0.0 | 1.170 (0.003) | 0.211 (0.003) | 0.121 | 0.121 | 0.022 (0.001) | -- |
| Treatment feedback | affine_calibrated_q10 | 200/200 | 0.0 | 1.090 (0.003) | 0.131 (0.002) | 0.100 | 0.100 | 0.013 (0.001) | -- |
| Treatment feedback | unpenalized_level | 200/200 | 0.0 | 1.170 (0.003) | 0.211 (0.003) | 0.121 | 0.121 | 0.022 (0.001) | 4.202 (0.022) |
| Treatment feedback | tuned_level | 200/200 | 0.0 | 1.119 (0.003) | 0.159 (0.002) | 0.092 | 0.092 | 0.017 (0.001) | 3.402 (0.023) |
| Treatment feedback | tuned_level_slope | 200/200 | 0.0 | 1.145 (0.002) | 0.186 (0.002) | 0.076 | 0.076 | 0.024 (0.001) | 3.362 (0.017) |
| Transient non-persistence | oracle | 200/200 | 0.0 | 0.964 (0.002) | 0.000 (0.000) | 0.100 | 0.100 | 0.008 (0.000) | -- |
| Transient non-persistence | population | 200/200 | 0.0 | 1.004 (0.002) | 0.040 (0.001) | 0.058 | 0.058 | 0.042 (0.001) | -- |
| Transient non-persistence | raw_index_q10 | 200/200 | 0.0 | 1.791 (0.010) | 0.827 (0.010) | 0.222 | 0.222 | 0.122 (0.002) | -- |
| Transient non-persistence | affine_calibrated_q10 | 200/200 | 0.0 | 1.034 (0.002) | 0.070 (0.002) | 0.100 | 0.100 | 0.012 (0.001) | -- |
| Transient non-persistence | unpenalized_level | 200/200 | 0.0 | 1.791 (0.010) | 0.827 (0.010) | 0.222 | 0.222 | 0.122 (0.002) | 7.153 (0.033) |
| Transient non-persistence | tuned_level | 200/200 | 0.0 | 1.006 (0.002) | 0.042 (0.001) | 0.058 | 0.058 | 0.042 (0.001) | 0.202 (0.023) |
| Transient non-persistence | tuned_level_slope | 200/200 | 0.0 | 1.007 (0.002) | 0.043 (0.001) | 0.059 | 0.059 | 0.041 (0.001) | 0.294 (0.035) |
| Pure null with serial dependence | oracle | 200/200 | 0.0 | 0.960 (0.003) | 0.000 (0.000) | 0.100 | 0.100 | 0.010 (0.001) | -- |
| Pure null with serial dependence | population | 200/200 | 0.0 | 0.961 (0.003) | 0.001 (0.000) | 0.099 | 0.099 | 0.010 (0.001) | -- |
| Pure null with serial dependence | raw_index_q10 | 200/200 | 0.0 | 1.201 (0.005) | 0.241 (0.004) | 0.168 | 0.168 | 0.068 (0.001) | -- |
| Pure null with serial dependence | affine_calibrated_q10 | 200/200 | 0.0 | 1.019 (0.003) | 0.059 (0.001) | 0.101 | 0.101 | 0.015 (0.001) | -- |
| Pure null with serial dependence | unpenalized_level | 200/200 | 0.0 | 1.201 (0.005) | 0.241 (0.004) | 0.168 | 0.168 | 0.068 (0.001) | 3.783 (0.019) |
| Pure null with serial dependence | tuned_level | 200/200 | 0.0 | 0.962 (0.003) | 0.002 (0.000) | 0.100 | 0.100 | 0.011 (0.001) | 0.240 (0.018) |
| Pure null with serial dependence | tuned_level_slope | 200/200 | 0.0 | 0.962 (0.003) | 0.002 (0.000) | 0.101 | 0.101 | 0.011 (0.001) | 0.247 (0.018) |
| Weak persistent level | oracle | 200/200 | 0.0 | 0.965 (0.002) | 0.000 (0.000) | 0.100 | 0.100 | 0.007 (0.000) | -- |
| Weak persistent level | population | 200/200 | 0.0 | 1.064 (0.002) | 0.099 (0.001) | 0.101 | 0.101 | 0.010 (0.001) | -- |
| Weak persistent level | raw_index_q10 | 200/200 | 0.0 | 1.086 (0.003) | 0.121 (0.002) | 0.127 | 0.127 | 0.027 (0.001) | -- |
| Weak persistent level | affine_calibrated_q10 | 200/200 | 0.0 | 1.038 (0.002) | 0.072 (0.001) | 0.101 | 0.101 | 0.011 (0.001) | -- |
| Weak persistent level | unpenalized_level | 200/200 | 0.0 | 1.086 (0.003) | 0.121 (0.002) | 0.127 | 0.127 | 0.027 (0.001) | 2.952 (0.015) |
| Weak persistent level | tuned_level | 200/200 | 0.0 | 1.025 (0.002) | 0.059 (0.001) | 0.104 | 0.104 | 0.010 (0.001) | 1.989 (0.011) |
| Weak persistent level | tuned_level_slope | 200/200 | 0.0 | 1.032 (0.002) | 0.067 (0.001) | 0.102 | 0.102 | 0.010 (0.000) | 2.058 (0.010) |
| Heavy-tailed t3 residual | oracle | 200/200 | 0.0 | 1.353 (0.005) | 0.000 (0.000) | 0.099 | 0.099 | 0.006 (0.000) | -- |
| Heavy-tailed t3 residual | population | 200/200 | 0.0 | 1.670 (0.006) | 0.316 (0.004) | 0.101 | 0.101 | 0.011 (0.001) | -- |
| Heavy-tailed t3 residual | raw_index_q10 | 200/200 | 0.0 | 1.557 (0.006) | 0.204 (0.004) | 0.126 | 0.126 | 0.026 (0.001) | -- |
| Heavy-tailed t3 residual | affine_calibrated_q10 | 200/200 | 0.0 | 1.502 (0.006) | 0.149 (0.003) | 0.101 | 0.101 | 0.012 (0.001) | -- |
| Heavy-tailed t3 residual | unpenalized_level | 200/200 | 0.0 | 1.557 (0.006) | 0.204 (0.004) | 0.126 | 0.126 | 0.026 (0.001) | 6.493 (0.156) |
| Heavy-tailed t3 residual | tuned_level | 200/200 | 0.0 | 1.482 (0.006) | 0.129 (0.002) | 0.104 | 0.104 | 0.013 (0.001) | 4.311 (0.029) |
| Heavy-tailed t3 residual | tuned_level_slope | 200/200 | 0.0 | 1.522 (0.005) | 0.169 (0.002) | 0.093 | 0.093 | 0.011 (0.001) | 4.242 (0.018) |

## Data-generating-mechanism checks

| Scenario | Mean index n | Mean later n | Count--level r | Observed error lag-1 r | Treated fraction |
|---|---:|---:|---:|---:|---:|
| Ideal: large, dense | 12.00 | 12.00 | 0.004 | 0.014 | 0.000 |
| Ideal: small, dense | 11.99 | 11.99 | -0.007 | 0.013 | 0.000 |
| Ideal: large, sparse | 3.50 | 6.00 | -0.001 | 0.015 | 0.000 |
| Serial dependence | 10.02 | 10.01 | -0.002 | 0.502 | 0.000 |
| Integer MAP rounding | 9.99 | 9.99 | 0.003 | 0.221 | 0.000 |
| Informative monitoring | 5.83 | 5.84 | -0.720 | 0.312 | 0.000 |
| Cluster size--level association | 11.50 | 11.51 | -0.933 | 0.014 | 0.000 |
| Misspecified common time | 10.00 | 10.00 | -0.001 | 0.219 | 0.000 |
| Persistent level plus shape | 12.00 | 11.99 | 0.003 | 0.249 | 0.000 |
| Treatment feedback | 10.00 | 10.01 | -0.002 | 0.256 | 0.270 |
| Transient non-persistence | 10.01 | 9.99 | nan | 0.255 | 0.000 |
| Pure null with serial dependence | 9.99 | 10.00 | nan | 0.503 | 0.000 |
| Weak persistent level | 12.00 | 12.00 | -0.005 | 0.014 | 0.000 |
| Heavy-tailed t3 residual | 12.00 | 12.00 | 0.004 | 0.008 | 0.000 |

## Penalty selection

| Scenario | Method | Penalty | Selected n/N (%) |
|---|---|---:|---:|
| Ideal: large, dense | tuned_level | 0 | 11/200 (5.5) |
| Ideal: large, dense | tuned_level | 0.03 | 175/200 (87.5) |
| Ideal: large, dense | tuned_level | 0.1 | 14/200 (7.0) |
| Ideal: large, dense | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Ideal: small, dense | tuned_level | 0 | 36/200 (18.0) |
| Ideal: small, dense | tuned_level | 0.03 | 129/200 (64.5) |
| Ideal: small, dense | tuned_level | 0.1 | 35/200 (17.5) |
| Ideal: small, dense | tuned_level_slope | 0.1 | 199/200 (99.5) |
| Ideal: small, dense | tuned_level_slope | 0.3 | 1/200 (0.5) |
| Ideal: large, sparse | tuned_level | 0.03 | 153/200 (76.5) |
| Ideal: large, sparse | tuned_level | 0.1 | 47/200 (23.5) |
| Ideal: large, sparse | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Serial dependence | tuned_level | 0 | 3/200 (1.5) |
| Serial dependence | tuned_level | 0.03 | 90/200 (45.0) |
| Serial dependence | tuned_level | 0.1 | 106/200 (53.0) |
| Serial dependence | tuned_level | 0.3 | 1/200 (0.5) |
| Serial dependence | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Integer MAP rounding | tuned_level | 0 | 14/200 (7.0) |
| Integer MAP rounding | tuned_level | 0.03 | 150/200 (75.0) |
| Integer MAP rounding | tuned_level | 0.1 | 36/200 (18.0) |
| Integer MAP rounding | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Informative monitoring | tuned_level | 0 | 6/200 (3.0) |
| Informative monitoring | tuned_level | 0.03 | 194/200 (97.0) |
| Informative monitoring | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Cluster size--level association | tuned_level | 0 | 53/200 (26.5) |
| Cluster size--level association | tuned_level | 0.03 | 147/200 (73.5) |
| Cluster size--level association | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Misspecified common time | tuned_level | 0.1 | 117/200 (58.5) |
| Misspecified common time | tuned_level | 0.3 | 83/200 (41.5) |
| Misspecified common time | tuned_level_slope | 0.1 | 185/200 (92.5) |
| Misspecified common time | tuned_level_slope | 0.3 | 15/200 (7.5) |
| Persistent level plus shape | tuned_level | 0.1 | 1/200 (0.5) |
| Persistent level plus shape | tuned_level | 0.3 | 24/200 (12.0) |
| Persistent level plus shape | tuned_level | 1 | 87/200 (43.5) |
| Persistent level plus shape | tuned_level | 3 | 39/200 (19.5) |
| Persistent level plus shape | tuned_level | 10 | 49/200 (24.5) |
| Persistent level plus shape | tuned_level_slope | 0.1 | 43/200 (21.5) |
| Persistent level plus shape | tuned_level_slope | 0.3 | 78/200 (39.0) |
| Persistent level plus shape | tuned_level_slope | 1 | 51/200 (25.5) |
| Persistent level plus shape | tuned_level_slope | 3 | 13/200 (6.5) |
| Persistent level plus shape | tuned_level_slope | 10 | 15/200 (7.5) |
| Treatment feedback | tuned_level | 0.03 | 101/200 (50.5) |
| Treatment feedback | tuned_level | 0.1 | 97/200 (48.5) |
| Treatment feedback | tuned_level | 0.3 | 2/200 (1.0) |
| Treatment feedback | tuned_level_slope | 0.1 | 200/200 (100.0) |
| Transient non-persistence | tuned_level | 0.3 | 5/200 (2.5) |
| Transient non-persistence | tuned_level | 1 | 6/200 (3.0) |
| Transient non-persistence | tuned_level | 3 | 13/200 (6.5) |
| Transient non-persistence | tuned_level | 10 | 176/200 (88.0) |
| Transient non-persistence | tuned_level_slope | 0.1 | 1/200 (0.5) |
| Transient non-persistence | tuned_level_slope | 0.3 | 23/200 (11.5) |
| Transient non-persistence | tuned_level_slope | 1 | 8/200 (4.0) |
| Transient non-persistence | tuned_level_slope | 3 | 10/200 (5.0) |
| Transient non-persistence | tuned_level_slope | 10 | 158/200 (79.0) |
| Pure null with serial dependence | tuned_level | 0.3 | 5/200 (2.5) |
| Pure null with serial dependence | tuned_level | 1 | 53/200 (26.5) |
| Pure null with serial dependence | tuned_level | 3 | 39/200 (19.5) |
| Pure null with serial dependence | tuned_level | 10 | 103/200 (51.5) |
| Pure null with serial dependence | tuned_level_slope | 0.3 | 15/200 (7.5) |
| Pure null with serial dependence | tuned_level_slope | 1 | 69/200 (34.5) |
| Pure null with serial dependence | tuned_level_slope | 3 | 28/200 (14.0) |
| Pure null with serial dependence | tuned_level_slope | 10 | 88/200 (44.0) |
| Weak persistent level | tuned_level | 0.03 | 2/200 (1.0) |
| Weak persistent level | tuned_level | 0.1 | 62/200 (31.0) |
| Weak persistent level | tuned_level | 0.3 | 135/200 (67.5) |
| Weak persistent level | tuned_level | 1 | 1/200 (0.5) |
| Weak persistent level | tuned_level_slope | 0.1 | 77/200 (38.5) |
| Weak persistent level | tuned_level_slope | 0.3 | 122/200 (61.0) |
| Weak persistent level | tuned_level_slope | 1 | 1/200 (0.5) |
| Heavy-tailed t3 residual | tuned_level | 0 | 1/200 (0.5) |
| Heavy-tailed t3 residual | tuned_level | 0.03 | 112/200 (56.0) |
| Heavy-tailed t3 residual | tuned_level | 0.1 | 87/200 (43.5) |
| Heavy-tailed t3 residual | tuned_level_slope | 0.1 | 200/200 (100.0) |

## Precision boundary

Monte Carlo confidence intervals quantify simulation error only. They do not validate the data-generating mechanisms, and no fixed replicate count is treated as automatically sufficient; the reported MCSE should be compared with the smallest method difference the manuscript intends to interpret.
