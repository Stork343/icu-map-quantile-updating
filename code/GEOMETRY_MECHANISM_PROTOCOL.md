# Geometry mechanism evaluation

## Scientific question

The evaluation asks how the interval over which early measurements are available changes the relative predictive risk of history and horizon regularization. The prediction landmark remains 12 hours. The early sampling intervals have widths 4, 8, and 12 hours; the forecast windows have lengths 4, 8, and 12 hours. All combinations are reported in each of the four existing mixed population settings.

This protocol specifies an additional analysis of the completed simulation study. The overall study results were already available when the analysis was designed. Coverage specific risk estimates have not been used to choose the protocol. No additional data generating settings, penalty values, or prediction rules are introduced.

## Prediction rules

Full geometry selection chooses among all 42 structured base profiles and their globally fitted affine calibrated forms. Horizon geometry selection restricts this menu to gamma equal to zero. Diagonal profile selection chooses among the 48 positive diagonal level and slope matrices and their affine calibrated forms. The latter is distinct from the original complete family, which also contains scalar profiles and empirical summaries.

Each replicate and forecast horizon uses the same population coefficients and complete tuning losses from the completed study. Selection takes place across all 120 tuning stays. The chosen rule and affine coefficients are frozen before evaluating coverage groups. Tuning is not repeated within a coverage group. Tie handling follows the original candidate order. The three contrasts are full geometry minus diagonal, horizon geometry minus diagonal, and full geometry minus horizon geometry. Negative differences mean smaller later check loss for the first rule.

A direct only analysis selects among the 42, 6, and 48 base rules, respectively. It examines regularization without selecting an affine calibration. Joint selection evaluates the complete prediction procedure; the direct comparison distinguishes the role of the penalty from changes in the selected calibration. Both analyses are reported under the same coverage groups and forecasting targets.

## Records and estimation

All four settings and all 200 completed replicates per setting enter the analysis. The original dataset seed regenerates the original 600 stay dataset and its fitting, tuning, and assessment allocation. The retained population coefficients are reused. The saved independent reference seed regenerates the original 2,000 stay reference population. For each frozen rule, losses average the later observations within each stay. Paired differences are averaged within each assigned early sampling interval, and these group means are then averaged equally across the 200 independent replicates. The reference population provides the conditional fitted rule risk comparison. The original 120 assessment stays provide a companion comparison and an independent replay check. Every group must contain stays in every replicate and evaluation population. A missing group or inconsistent replay terminates the calculation.

Assigned coverage denotes the width of the sampling interval ending at the landmark. Actual first to last observation span and index count are also reported. A four hour sampling interval can yield an actual span below four hours because times are sampled from its grid.

The saved full geometry, horizon geometry, and original family stay losses are matched to the regenerated assessment stay indices. Replayed reference summaries for the full geometry and horizon geometry selections must match the retained original study summaries. A newly scored diagonal rule is checked against the saved original family results whenever the two selections coincide. All numerical profile solutions retain the established primal and dual certificate thresholds.

## Uncertainty and reporting

The independent statistical unit is the simulation replicate. Monte Carlo standard errors are the standard deviations of 200 replicate paired means divided by the square root of 200. Pointwise 95 percent intervals use a Student t quantile with 199 degrees of freedom. The complete family contains 432 contrasts, formed by four settings, three early intervals, three forecast windows, three rule contrasts, two selection modes, and two evaluation populations. Bonferroni simultaneous 95 percent intervals use the corresponding Student t quantile for this family. The complete numerical table reports both interval types.

The figure displays both geometry procedures relative to diagonal profile selection for every setting and forecast window. Zero reference lines and the same vertical scale support comparison. Interpretation distinguishes a local risk difference from the overall mixed population risk and identifies whether a favorable direction is supported by the simultaneous interval. Selection frequencies describe how often the full geometry rule uses positive gamma. All planned conditions and contrasts remain in the outputs.
