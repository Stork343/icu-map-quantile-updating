# Learning separate penalties from early records

## Research question

The study asks whether a patient's early record can determine different strengths of regularization for level and slope, using later quantile prediction risk to learn both strengths. The two positive penalties remain diagonal. Each is a continuous function of early record features and the forecast window.

This protocol follows the completed geometry comparison. Its overall and coverage specific results were available when this method was designed. The learning rule, parameter domain, complexity choices, and evaluation populations are fixed before evaluating the new rules on assessment outcomes.

## Features and penalty functions

Let u=(t-12)/12 and let m be the early observation count. Three bounded history features are clip(log(m/12)/log(4),-1,1), clip(12 var(u)-1,-1,1), and the clipped difference between the recent and earlier rank half residual q10 values divided by 10 mmHg. All features use measurements available at the 12 hour landmark. These deterministic transformations use no later outcomes.

The remaining three features are piecewise linear hat functions of the forecast horizon over the knots 4, 8, and 12 hours. They sum to one and equal the corresponding coordinate vector at each knot. They define a continuous interpolation in the requested horizon. The resulting feature vector has six entries.

Separate parameter vectors specify the level and slope penalties. Their log values interpolate smoothly between the log lower and upper bounds through the logistic function of the feature linear predictor. The level penalty lies between 0.03 and 10, and the slope penalty between 0.03 and 100. Horizon intercept parameters lie in [-12,12], and history coefficients in [-4,4]. The profile itself minimizes ordinary index check loss plus the resulting diagonal quadratic penalty. Forecasts remain unrestricted.

Exact constant diagonal profiles are retained as a separate candidate branch, including the endpoints of the original 48 matrix grid. At each horizon this branch selects among all 48 direct predictions. This branch also provides the initial penalties for fitting the continuous functions. Optimization uses deterministic starting points and records the returned objective and numerical diagnostics. A numerical stopping criterion is not interpreted as a certificate of a global outer optimum.

## Learning and complexity selection

Each fitting partition learns one pair of functions jointly across all three horizons. The empirical prediction loss first averages later observations within each stay, then gives equal weight to eligible stays at each horizon, and finally averages the three horizon losses equally. Eligibility remains horizon specific. All occurrences of the same stay share an inner fold.

Threefold cross validation within the original tuning population selects the history coefficient penalty from 0, 0.001, 0.01, and 0.1. The training objective adds this coefficient times the sum of squares of the six history coefficients. Training of each inner fold uses its own records and its own constant grid initializations. Held out tuning records choose the complexity. The exact constant branch is included as a fifth cross validation candidate. The selected continuous complexity is refitted on the complete tuning population; the selected procedure can retain the constant branch. Both the best continuous candidate and the cross validation selected procedure are evaluated.

Direct prediction is the initial learning target. For each horizon, an ordinary unrestricted affine map is subsequently fitted on the full tuning population for both new procedures. Both direct and calibrated predictions are reported. Assessment outcomes choose neither function complexity nor calibration form.

## Evaluation

The mixed population study retains all four previously specified settings and uses 200 new independent replicates per setting under master seed 20261022. Each replicate contains 600 newly generated stays allocated to 360 fitting, 120 tuning, and 120 assessment stays, together with a newly generated independent 2,000 stay reference population. Population quantile coefficients are fitted on the new fitting records. Comparison methods include constant diagonal selection and the feature selector, together with the original complete family and calibrated q10. Every comparison method is fitted and tuned on the same new records. Paired loss differences use identical stay records. Monte Carlo uncertainty treats the replicate as the independent unit.

The clinical study retains all five nested assessment folds and the established temporal partition. Each partition fits its maps jointly across the 4, 8, and 12 hour forecast windows. Fitting coefficients and age transformations remain those of the corresponding original fitting population. Internal complexity selection uses tuning stays. Later assessment losses give each eligible ICU stay equal total weight. Public clinical files contain aggregate quantities and learned function parameters. Individual clinical records remain in the credentialed local directory.

## Numerical and scientific checks

Every profile must pass the existing primal and dual objective, projected optimality, and coefficient stationarity checks. A compiled batch solver is compared with the existing certified Clarabel implementation on actual numerical quantile objectives. Hypergradients are compared with finite differences away from active set transitions. Replay checks establish agreement with the original constant diagonal scores when the same rules are used.

The analysis reports the empirical risk of the returned fitted rules. The theoretical analysis distinguishes uniform comparisons valid for any returned parameter from oracle results involving an explicit optimization error. The benefit of individual learning is evaluated against its estimation and approximation costs. All four simulation settings and all clinical horizons remain in the reported outputs.
