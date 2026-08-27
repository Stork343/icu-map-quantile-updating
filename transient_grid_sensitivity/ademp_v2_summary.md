# ADEMP v2 split-window simulation

## Monte Carlo design

- Independent unit: stay; independent Monte Carlo unit: complete simulated dataset.
- Replicates per scenario: 200.
- Target quantile: 0.1.
- Losses and calibration probabilities are first averaged within stay and then across stays.
- A discrete quantile obeys the identifying probability-mass bracket when P(Y < q) <= tau <= P(Y <= q); probability-mass bracket violation is the distance outside that bracket.
- Parenthesized values below are Monte Carlo standard errors across independent datasets.

## Performance summary

| Scenario | Method | Effective/attempted | Failure (%) | Loss (MCSE) | Regret (MCSE) | P(Y<q) | P(Y<=q) | Probability-mass bracket violation (MCSE) | Offset RMSE (MCSE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transient non-persistence | oracle | 200/200 | 0.0 | 0.964 (0.002) | 0.000 (0.000) | 0.100 | 0.100 | 0.008 (0.000) | -- |
| Transient non-persistence | population | 200/200 | 0.0 | 1.004 (0.002) | 0.040 (0.001) | 0.058 | 0.058 | 0.042 (0.001) | -- |
| Transient non-persistence | raw_index_q10 | 200/200 | 0.0 | 1.791 (0.010) | 0.827 (0.010) | 0.222 | 0.222 | 0.122 (0.002) | -- |
| Transient non-persistence | affine_calibrated_q10 | 200/200 | 0.0 | 1.034 (0.002) | 0.070 (0.002) | 0.100 | 0.100 | 0.012 (0.001) | -- |
| Transient non-persistence | unpenalized_level | 200/200 | 0.0 | 1.791 (0.010) | 0.827 (0.010) | 0.222 | 0.222 | 0.122 (0.002) | 7.153 (0.033) |
| Transient non-persistence | tuned_level | 200/200 | 0.0 | 1.005 (0.002) | 0.041 (0.001) | 0.058 | 0.058 | 0.042 (0.001) | 0.111 (0.025) |
| Transient non-persistence | tuned_level_slope | 200/200 | 0.0 | 1.006 (0.002) | 0.042 (0.001) | 0.059 | 0.059 | 0.041 (0.001) | 0.232 (0.037) |

## Data-generating-mechanism checks

| Scenario | Mean index n | Mean later n | Count--level r | Observed error lag-1 r | Treated fraction |
|---|---:|---:|---:|---:|---:|
| Transient non-persistence | 10.01 | 9.99 | nan | 0.255 | 0.000 |

## Penalty selection

| Scenario | Method | Penalty | Selected n/N (%) |
|---|---|---:|---:|
| Transient non-persistence | tuned_level | 0.3 | 5/200 (2.5) |
| Transient non-persistence | tuned_level | 1 | 6/200 (3.0) |
| Transient non-persistence | tuned_level | 3 | 13/200 (6.5) |
| Transient non-persistence | tuned_level | 10 | 10/200 (5.0) |
| Transient non-persistence | tuned_level | 30 | 4/200 (2.0) |
| Transient non-persistence | tuned_level | 100 | 2/200 (1.0) |
| Transient non-persistence | tuned_level | 300 | 160/200 (80.0) |
| Transient non-persistence | tuned_level_slope | 0.1 | 1/200 (0.5) |
| Transient non-persistence | tuned_level_slope | 0.3 | 23/200 (11.5) |
| Transient non-persistence | tuned_level_slope | 1 | 8/200 (4.0) |
| Transient non-persistence | tuned_level_slope | 3 | 9/200 (4.5) |
| Transient non-persistence | tuned_level_slope | 10 | 9/200 (4.5) |
| Transient non-persistence | tuned_level_slope | 30 | 1/200 (0.5) |
| Transient non-persistence | tuned_level_slope | 100 | 1/200 (0.5) |
| Transient non-persistence | tuned_level_slope | 300 | 148/200 (74.0) |

## Precision boundary

Monte Carlo confidence intervals quantify simulation error only. They do not validate the data-generating mechanisms, and no fixed replicate count is treated as automatically sufficient; the reported MCSE should be compared with the smallest method difference the manuscript intends to interpret.
