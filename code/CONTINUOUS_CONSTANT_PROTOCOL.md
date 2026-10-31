# Continuous constant component comparison

The comparison separates continuous penalty optimization from dependence on the early record. All choices below are fixed before assessment of the added rules.

## Fitted rules

The continuous constant model uses the existing two component penalty functions with all six history coefficients fixed at zero. Its six horizon intercepts retain the original bounds of minus twelve to twelve. The penalty ranges remain 0.03 to 10 for level and 0.03 to 100 for slope. The original exact 48 matrix grid branch retains the endpoints of those ranges.

The zero history model uses the original L-BFGS-B options, a maximum of 100 iterations per start and the same two deterministic starts. Population fits, index records, tuning records and grouping of horizons within stay are retained. Individual profiles must pass the existing optimality checks. Optimizer stopping conditions and the numbers of objective and profile evaluations are recorded.

Grouped threefold validation fits the continuous constant model inside each original inner training fold and assesses it on that fold's held out tuning stays. These validation scores select between the continuous constant model and the exact grid branch, with the grid branch retained in numerical ties. A final continuous constant fit uses all tuning stays and the retained branch decision. The history dependent procedure and its original validation choices remain frozen, permitting a direct comparison of the two fitted procedures.

Both the continuous constant model and its validation selected version are assessed. Each receives a separate horizon specific affine calibration estimated on tuning records by the original stay weighted objective. Direct forecasts and affine calibrated forecasts remain distinct comparisons.

## Simulation comparisons

Reuse all four original settings and all 200 independent datasets per setting. Each dataset retains its original fitting and tuning allocation, 120 assessment stays and 2,000 independent reference stays. All three forecast horizons are evaluated. No replicate is removed on the basis of its result.

The eight paired contrasts are history continuous direct minus continuous constant direct; history selected direct minus selected constant direct; history continuous affine minus continuous constant affine; history selected affine minus selected constant affine; continuous constant direct minus grid constant direct; selected constant direct minus grid constant direct; continuous constant affine minus grid constant affine; and selected constant affine minus grid constant affine.

The new family contains 192 setting, horizon, population and contrast combinations. Simultaneous two sided 95 percent intervals use paired replicate means, their Monte Carlo standard errors and the Bonferroni adjusted Student t critical value with 199 degrees of freedom. Pointwise intervals and replicate gain frequencies are also supplied. Original comparison families retain their original results and multiplicity definitions.

## Clinical comparisons

Use the five original nested partitions and the original temporal partition, retaining the population coefficients and fitting only age standardization for each partition. Fit each added rule using the original tuning stays and grouped validation seed. The assessment observations and eligibility at four, eight and twelve hours remain unchanged.

Nested results report pooled out of fold losses, the five fold means and omission diagnostics. They do not use a pooled independent sample interval as inference for the retrained procedure. Temporal comparisons condition on the frozen rules and use paired stay differences. The eight temporal contrasts at three horizons form a family of 24 comparisons, with simultaneous normal intervals and Holm adjusted two sided tests. Calibration brackets, fitting choices and numerical diagnostics are retained.

## Reproduction

The output records original seeds, source fitted rule identifiers, validation assignments, penalty bounds and solver configuration. Original history dependent predictions are replayed against their saved results before the new comparisons are interpreted. Public clinical exports contain aggregate quantities; individual clinical records remain in credentialed local storage.
