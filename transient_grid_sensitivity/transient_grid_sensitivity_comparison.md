# Post hoc transient-penalty-grid sensitivity

This analysis re-used the same 200 transient non-persistence datasets, split assignments, and common-component fits as the primary ADEMP-v2 simulation. Population-only losses were identical replicate by replicate. The only intended change was expansion of the level-penalty grid from `{0, 0.03, 0.10, 0.30, 1, 3, 10}` to `{0, 0.03, 0.10, 0.30, 1, 3, 10, 30, 100, 300}`. It is a post hoc tuning-boundary sensitivity and does not replace the prespecified primary grid.

| Estimand | Mean paired loss difference | MCSE | Monte Carlo 95% interval | Effective/attempted | Failure rate |
|---|---:|---:|---:|---:|---:|
| Primary-grid tuned level minus population | +0.002032 | 0.000298 | [0.001449, 0.002615] | 200/200 | 0% |
| Extended-grid tuned level minus population | +0.001152 | 0.000306 | [0.000552, 0.001751] | 200/200 | 0% |
| Extended-grid minus primary-grid paired difference | -0.000881 | 0.000068 | [-0.001014, -0.000748] | 200/200 | 0% |

The extended grid reduced, but did not eliminate, the small null-setting loss penalty. The selected level penalty remained at the new upper boundary (`lambda = 300`) in 160/200 replicates (80%); the remaining selections were 0.3 (2.5%), 1 (3%), 3 (6.5%), 10 (5%), 30 (2%), and 100 (1%). Thus the primary transient result is partly grid-boundary sensitive, while the continued upper-bound selection indicates that a formally included no-update candidate (or a still wider grid) would be needed to claim exact null safety. This sensitivity should be reported as a limitation rather than used to replace the primary-grid result.
