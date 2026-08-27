# ADEMP v2: manuscript-ready key conclusions

Negative paired loss differences favor the first named method. Monte Carlo intervals quantify simulation error only.

## Directly supported summary

- The tuned level update had lower loss than the population-only rule with a Monte Carlo 95% interval below zero in 11 scenarios: Ideal: large, dense, Ideal: small, dense, Ideal: large, sparse, Serial dependence, Integer MAP rounding, Informative monitoring, Cluster size--level association, Misspecified common time, Treatment feedback, Weak persistent level, Heavy-tailed t3 residual.
- It had higher loss than the population-only rule with a Monte Carlo 95% interval above zero in 2 scenarios: Transient non-persistence, Pure null with serial dependence.
- The tuned level update had lower loss than the tuning-calibrated affine q10 comparator in 5 scenarios: Persistent level plus shape, Transient non-persistence, Pure null with serial dependence, Weak persistent level, Heavy-tailed t3 residual.
- The affine q10 comparator had lower loss than the tuned level update in 9 scenarios: Ideal: large, dense, Ideal: small, dense, Ideal: large, sparse, Serial dependence, Integer MAP rounding, Informative monitoring, Cluster size--level association, Misspecified common time, Treatment feedback.

## Scenario-level paired estimates

| Scenario | Contrast (A - B) | Difference | MCSE | Monte Carlo 95% interval | Effective replicates |
|---|---|---:|---:|---:|---:|
| Ideal: large, dense | tuned_level - population | -0.3060 | 0.0045 | [-0.3148, -0.2971] | 200 |
| Ideal: large, dense | affine_calibrated_q10 - population | -0.3094 | 0.0044 | [-0.3181, -0.3007] | 200 |
| Ideal: large, dense | tuned_level - affine_calibrated_q10 | 0.0034 | 0.0011 | [0.0013, 0.0056] | 200 |
| Ideal: large, dense | tuned_level_slope - tuned_level | 0.0896 | 0.0023 | [0.0852, 0.0940] | 200 |
| Ideal: large, dense | raw_index_q10 - affine_calibrated_q10 | 0.0242 | 0.0011 | [0.0221, 0.0264] | 200 |
| Ideal: small, dense | tuned_level - population | -0.3095 | 0.0075 | [-0.3242, -0.2948] | 200 |
| Ideal: small, dense | affine_calibrated_q10 - population | -0.3138 | 0.0073 | [-0.3280, -0.2995] | 200 |
| Ideal: small, dense | tuned_level - affine_calibrated_q10 | 0.0043 | 0.0020 | [0.0005, 0.0081] | 200 |
| Ideal: small, dense | tuned_level_slope - tuned_level | 0.0846 | 0.0038 | [0.0772, 0.0920] | 200 |
| Ideal: small, dense | raw_index_q10 - affine_calibrated_q10 | 0.0194 | 0.0019 | [0.0157, 0.0232] | 200 |
| Ideal: large, sparse | tuned_level - population | -0.1676 | 0.0045 | [-0.1764, -0.1588] | 200 |
| Ideal: large, sparse | affine_calibrated_q10 - population | -0.2261 | 0.0045 | [-0.2350, -0.2173] | 200 |
| Ideal: large, sparse | tuned_level - affine_calibrated_q10 | 0.0585 | 0.0026 | [0.0534, 0.0636] | 200 |
| Ideal: large, sparse | tuned_level_slope - tuned_level | 0.0554 | 0.0030 | [0.0494, 0.0614] | 200 |
| Ideal: large, sparse | raw_index_q10 - affine_calibrated_q10 | 0.3305 | 0.0062 | [0.3183, 0.3427] | 200 |
| Serial dependence | tuned_level - population | -0.2345 | 0.0046 | [-0.2436, -0.2255] | 200 |
| Serial dependence | affine_calibrated_q10 - population | -0.2667 | 0.0045 | [-0.2755, -0.2580] | 200 |
| Serial dependence | tuned_level - affine_calibrated_q10 | 0.0322 | 0.0022 | [0.0279, 0.0364] | 200 |
| Serial dependence | tuned_level_slope - tuned_level | 0.0415 | 0.0026 | [0.0364, 0.0466] | 200 |
| Serial dependence | raw_index_q10 - affine_calibrated_q10 | 0.0975 | 0.0035 | [0.0907, 0.1043] | 200 |
| Integer MAP rounding | tuned_level - population | -0.2718 | 0.0050 | [-0.2815, -0.2620] | 200 |
| Integer MAP rounding | affine_calibrated_q10 - population | -0.2876 | 0.0047 | [-0.2968, -0.2784] | 200 |
| Integer MAP rounding | tuned_level - affine_calibrated_q10 | 0.0159 | 0.0017 | [0.0126, 0.0191] | 200 |
| Integer MAP rounding | tuned_level_slope - tuned_level | 0.0725 | 0.0030 | [0.0665, 0.0784] | 200 |
| Integer MAP rounding | raw_index_q10 - affine_calibrated_q10 | 0.0436 | 0.0018 | [0.0401, 0.0471] | 200 |
| Informative monitoring | tuned_level - population | -0.3217 | 0.0042 | [-0.3300, -0.3134] | 200 |
| Informative monitoring | affine_calibrated_q10 - population | -0.3916 | 0.0051 | [-0.4016, -0.3815] | 200 |
| Informative monitoring | tuned_level - affine_calibrated_q10 | 0.0699 | 0.0033 | [0.0634, 0.0765] | 200 |
| Informative monitoring | tuned_level_slope - tuned_level | 0.2001 | 0.0034 | [0.1935, 0.2067] | 200 |
| Informative monitoring | raw_index_q10 - affine_calibrated_q10 | 0.2118 | 0.0053 | [0.2014, 0.2222] | 200 |
| Cluster size--level association | tuned_level - population | -0.3028 | 0.0038 | [-0.3103, -0.2954] | 200 |
| Cluster size--level association | affine_calibrated_q10 - population | -0.3292 | 0.0036 | [-0.3363, -0.3221] | 200 |
| Cluster size--level association | tuned_level - affine_calibrated_q10 | 0.0264 | 0.0015 | [0.0234, 0.0294] | 200 |
| Cluster size--level association | tuned_level_slope - tuned_level | 0.1269 | 0.0027 | [0.1216, 0.1323] | 200 |
| Cluster size--level association | raw_index_q10 - affine_calibrated_q10 | 0.0377 | 0.0017 | [0.0343, 0.0410] | 200 |
| Misspecified common time | tuned_level - population | -0.1265 | 0.0038 | [-0.1340, -0.1190] | 200 |
| Misspecified common time | affine_calibrated_q10 - population | -0.3240 | 0.0054 | [-0.3345, -0.3135] | 200 |
| Misspecified common time | tuned_level - affine_calibrated_q10 | 0.1976 | 0.0037 | [0.1903, 0.2048] | 200 |
| Misspecified common time | tuned_level_slope - tuned_level | -0.0066 | 0.0019 | [-0.0104, -0.0028] | 200 |
| Misspecified common time | raw_index_q10 - affine_calibrated_q10 | 0.5923 | 0.0065 | [0.5795, 0.6050] | 200 |
| Persistent level plus shape | tuned_level - population | -0.0017 | 0.0009 | [-0.0035, 0.0001] | 200 |
| Persistent level plus shape | affine_calibrated_q10 - population | 0.0336 | 0.0022 | [0.0293, 0.0379] | 200 |
| Persistent level plus shape | tuned_level - affine_calibrated_q10 | -0.0353 | 0.0022 | [-0.0395, -0.0311] | 200 |
| Persistent level plus shape | tuned_level_slope - tuned_level | -0.0076 | 0.0011 | [-0.0097, -0.0055] | 200 |
| Persistent level plus shape | raw_index_q10 - affine_calibrated_q10 | 0.2969 | 0.0060 | [0.2852, 0.3087] | 200 |
| Treatment feedback | tuned_level - population | -0.1530 | 0.0031 | [-0.1592, -0.1469] | 200 |
| Treatment feedback | affine_calibrated_q10 - population | -0.1817 | 0.0029 | [-0.1875, -0.1760] | 200 |
| Treatment feedback | tuned_level - affine_calibrated_q10 | 0.0287 | 0.0016 | [0.0256, 0.0317] | 200 |
| Treatment feedback | tuned_level_slope - tuned_level | 0.0263 | 0.0019 | [0.0226, 0.0301] | 200 |
| Treatment feedback | raw_index_q10 - affine_calibrated_q10 | 0.0801 | 0.0022 | [0.0757, 0.0845] | 200 |
| Transient non-persistence | tuned_level - population | 0.0020 | 0.0003 | [0.0014, 0.0026] | 200 |
| Transient non-persistence | affine_calibrated_q10 - population | 0.0306 | 0.0018 | [0.0270, 0.0342] | 200 |
| Transient non-persistence | tuned_level - affine_calibrated_q10 | -0.0285 | 0.0019 | [-0.0322, -0.0249] | 200 |
| Transient non-persistence | tuned_level_slope - tuned_level | 0.0010 | 0.0005 | [-0.0000, 0.0020] | 200 |
| Transient non-persistence | raw_index_q10 - affine_calibrated_q10 | 0.7567 | 0.0100 | [0.7371, 0.7764] | 200 |
| Pure null with serial dependence | tuned_level - population | 0.0009 | 0.0004 | [0.0002, 0.0017] | 200 |
| Pure null with serial dependence | affine_calibrated_q10 - population | 0.0573 | 0.0015 | [0.0543, 0.0603] | 200 |
| Pure null with serial dependence | tuned_level - affine_calibrated_q10 | -0.0564 | 0.0015 | [-0.0593, -0.0535] | 200 |
| Pure null with serial dependence | tuned_level_slope - tuned_level | -0.0002 | 0.0003 | [-0.0007, 0.0003] | 200 |
| Pure null with serial dependence | raw_index_q10 - affine_calibrated_q10 | 0.1828 | 0.0038 | [0.1753, 0.1903] | 200 |
| Weak persistent level | tuned_level - population | -0.0394 | 0.0012 | [-0.0418, -0.0371] | 200 |
| Weak persistent level | affine_calibrated_q10 - population | -0.0263 | 0.0015 | [-0.0292, -0.0233] | 200 |
| Weak persistent level | tuned_level - affine_calibrated_q10 | -0.0131 | 0.0010 | [-0.0150, -0.0112] | 200 |
| Weak persistent level | tuned_level_slope - tuned_level | 0.0077 | 0.0007 | [0.0064, 0.0090] | 200 |
| Weak persistent level | raw_index_q10 - affine_calibrated_q10 | 0.0485 | 0.0014 | [0.0456, 0.0513] | 200 |
| Heavy-tailed t3 residual | tuned_level - population | -0.1875 | 0.0035 | [-0.1943, -0.1806] | 200 |
| Heavy-tailed t3 residual | affine_calibrated_q10 - population | -0.1676 | 0.0040 | [-0.1753, -0.1598] | 200 |
| Heavy-tailed t3 residual | tuned_level - affine_calibrated_q10 | -0.0199 | 0.0025 | [-0.0249, -0.0150] | 200 |
| Heavy-tailed t3 residual | tuned_level_slope - tuned_level | 0.0397 | 0.0017 | [0.0362, 0.0431] | 200 |
| Heavy-tailed t3 residual | raw_index_q10 - affine_calibrated_q10 | 0.0549 | 0.0022 | [0.0506, 0.0592] | 200 |

## Interpretation boundary

These experiments support statements about predictive check loss under the declared data-generating mechanisms. They do not establish clinical utility, external transportability, or universal superiority over other longitudinal quantile methods.
