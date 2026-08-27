# ADEMP v2: manuscript-ready key conclusions

Negative paired loss differences favor the first named method. Monte Carlo intervals quantify simulation error only.

## Directly supported summary

- The tuned level update had lower loss than the population-only rule with a Monte Carlo 95% interval below zero in 0 scenarios: none.
- It had higher loss than the population-only rule with a Monte Carlo 95% interval above zero in 1 scenario: Transient non-persistence.
- The tuned level update had lower loss than the tuning-calibrated affine q10 comparator in 1 scenario: Transient non-persistence.
- The affine q10 comparator had lower loss than the tuned level update in 0 scenarios: none.

## Scenario-level paired estimates

| Scenario | Contrast (A - B) | Difference | MCSE | Monte Carlo 95% interval | Effective replicates |
|---|---|---:|---:|---:|---:|
| Transient non-persistence | tuned_level - population | 0.0012 | 0.0003 | [0.0006, 0.0018] | 200 |
| Transient non-persistence | affine_calibrated_q10 - population | 0.0306 | 0.0018 | [0.0270, 0.0342] | 200 |
| Transient non-persistence | tuned_level - affine_calibrated_q10 | -0.0294 | 0.0019 | [-0.0331, -0.0257] | 200 |
| Transient non-persistence | tuned_level_slope - tuned_level | 0.0013 | 0.0005 | [0.0002, 0.0023] | 200 |
| Transient non-persistence | raw_index_q10 - affine_calibrated_q10 | 0.7567 | 0.0100 | [0.7371, 0.7764] | 200 |

## Interpretation boundary

These experiments support statements about predictive check loss under the declared data-generating mechanisms. They do not establish clinical utility, external transportability, or universal superiority over other longitudinal quantile methods.
