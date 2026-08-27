# ADEMP-v2 manuscript integration brief

This file is a ready-to-integrate technical brief for the revised Simulation Study. The primary results are the prespecified 14-scenario, original-penalty-grid results. The transient extended-grid analysis is post hoc and is reported separately at the end.

## Methods paragraph

We evaluated the split-window update in an ADEMP simulation with 14 data-generating mechanisms and 200 independent Monte Carlo replicates per mechanism (2,800 complete simulated datasets). For stay (i) at time (t_{ij}), the marginal 0.10-quantile model was generated from

\[
Y_{ij}=x_i^\top\beta+g(t_{ij})+b_i+s_i u_{ij}+\delta_i I(t_{ij}\le 12)+A_i(t_{ij})+\sigma_i\varepsilon_{ij},
\qquad u_{ij}=(t_{ij}-12)/12,
\]

where the marginal 0.10 quantile of \(\varepsilon_{ij}\) was zero. The common component was (x_i^\top\beta), with a nonlinear omitted (g(t)) in the common-time-misspecification scenario; (b_i) was a persistent stay level, (s_i) a stay-specific time shape, (delta_i) an index-only shift, and (A_i(t)) a treatment-feedback effect. Gaussian serial errors were generated on a latent 15-minute grid through a Gaussian-copula AR(1) process; the heavy-tail scenario transformed that copula to a Student-(t_3) marginal and centered it at the exact (t_3) 0.10 quantile. Training, tuning, and assessment stays comprised 60%, 20%, and 20% of each dataset.

The scenarios were grouped as follows:

1. **Sample-size and observation-density controls:** ideal large/dense (600 stays; 8--16 index and later observations), ideal small/dense (240 stays), and ideal large/sparse (600 stays; 2--5 index and 4--8 later observations).
2. **Dependence and observation-process stressors:** serial dependence (15-minute AR coefficient 0.82), informative monitoring (baseline observation probability 0.12; logit effects (-0.60b_i/5.5+0.85\operatorname{clip}[(70-Y_{i,\mathrm{prev}})/8])), cluster-size--level association (count probability \(\operatorname{logit}^{-1}[-1.05b_i/5.5]\)), and integer MAP rounding.
3. **Trajectory misspecification and feedback:** omitted nonlinear common time (amplitude 5 mmHg), persistent level plus random shape ((\operatorname{SD}(b_i)=3), (\operatorname{SD}(s_i)=5)), treatment feedback (early q10 below 68 mmHg triggered an 8-mmHg effect decaying over 7 h), and transient non-persistence ((\operatorname{SD}(\delta_i)=6), no persistent level).
4. **Signal and distribution controls:** pure null with serial dependence (no level, shape, or index shift), weak persistent level ((\operatorname{SD}(b_i)=2.5)), and heavy-tailed (t_3) residuals (persistent-level SD 5.5; residual scale multiplier 0.85).

Seven prespecified prediction rules were assessed: the structural marginal oracle, population-only quantile regression, raw index-window empirical q10 carry-forward, a tuning-fitted affine q10 comparator, the unpenalized scalar level update, the tuning-selected penalized level update, and a tuning-selected penalized level-plus-slope update. The affine comparator was fitted by stay-weighted quantile loss using a HiGHS linear program. Scalar levels used the exact convex profiled solution; level-plus-slope updates used a Fenchel-dual box optimization with a primal--dual certificate.

Performance measures were mean stay-level later check loss, regret relative to the structural marginal oracle, paired loss differences, discrete-quantile calibration (P(Y<q)), (P(Y\le q)), and

\[
\max\{P(Y<q)-0.10,\;0.10-P(Y\le q),\;0\},
\]

plus offset bias, RMSE and correlation, penalty-selection frequencies, and realized DGM diagnostics. Every outcome was first averaged within stay. Monte Carlo standard errors were computed across complete independent datasets; plotted intervals are Monte Carlo mean \(\pm1.96\) MCSE.

## Results paragraph

All 14 scenarios contributed 200/200 effective replicates for every method (19,600 method evaluations; failure rate 0%). The tuned level update had lower assessment loss than population-only prediction in 11 scenarios, including the ideal large/dense setting (paired difference (-0.3060\), MCSE 0.0045), informative monitoring ((-0.3217\), 0.0042), cluster-size informativeness ((-0.3028\), 0.0038), omitted common time ((-0.1265\), 0.0038), weak persistent levels ((-0.0394\), 0.0012), and heavy-tailed (t_3) errors ((-0.1875\), 0.0035). The interval included zero for persistent level plus shape ((-0.0017\), 0.0009), while the tuned update had a small positive loss difference under transient non-persistence ((+0.0020\), 0.0003) and the serial null ((+0.0009\), 0.0004).

The strong affine q10 comparator materially bounded the method claim. It had lower loss than the tuned level update in nine scenarios, including sparse observation ((\Delta_{\mathrm{level-cal}}=+0.0585\), MCSE 0.0026), informative monitoring ((+0.0699\), 0.0033), and common-time misspecification ((+0.1976\), 0.0037). Conversely, the tuned level update had lower loss in five scenarios: level plus shape ((-0.0353\), 0.0022), transient non-persistence ((-0.0285\), 0.0019), the serial null ((-0.0564\), 0.0015), weak persistent levels ((-0.0131\), 0.0010), and heavy-tailed errors ((-0.0199\), 0.0025). Thus the simulations support persistent lower-tail updating relative to population-only prediction, but not universal superiority over a tuning-calibrated empirical q10 rule.

The level-plus-slope extension improved on the tuned level update only when a richer time component was useful: omitted common time (paired difference (-0.0066\), MCSE 0.0019) and persistent level plus shape ((-0.0076\), 0.0011). It increased loss in the ideal, sparse, serial, informative-monitoring, cluster-size, weak-signal, heavy-tail, and treatment-feedback scenarios. This pattern supports retaining the scalar update as the primary parsimonious rule and treating richer update structures as tuning-selected extensions rather than defaults.

## Realized-mechanism and numerical QA

- The informative-monitoring scenario realized mean index/later counts of 5.83/5.84, a count--latent-level correlation of (-0.720), and an observed-error lag-1 correlation of 0.312.
- The cluster-size scenario realized a count--latent-level correlation of (-0.933).
- The serial-dependence and serial-null scenarios realized observed-error lag-1 correlations of 0.502 and 0.503, respectively.
- The treatment-feedback scenario treated 27.0% of stays.
- The large/sparse scenario realized 3.50 index and 6.00 later observations per stay on average; dense scenarios realized approximately 12/12.
- In the integer-rounding scenario, the structural oracle had (P(Y<q)=0.0843), (P(Y\le q)=0.1171), and mean probability-mass bracket violation 0.000292, demonstrating why strict-below error alone is invalid with ties.
- All 2,800 affine calibration linear programs succeeded; the largest equality-constraint residual was (1.14\times10^{-13}). All selected level-plus-slope assessment fits passed their numerical certificate, and every scenario-level mean solver-convergence fraction was 1.00.

## Post hoc transient-grid boundary sensitivity

In the primary transient scenario, 176/200 replicates (88%) selected the prespecified upper level penalty (\lambda=10). Re-running the identical 200 datasets with the grid extended through 30, 100 and 300 reduced the tuned-level-minus-population difference from (+0.002032) (MCSE 0.000298; Monte Carlo 95% interval 0.001449 to 0.002615) to (+0.001152) (0.000306; 0.000552 to 0.001751). The paired change was (-0.000881) (0.000068; (-0.001014) to (-0.000748)), but 160/200 replicates (80%) selected the new upper boundary (\lambda=300). The small null-setting penalty is therefore partly grid-boundary sensitive but was not eliminated. This analysis is post hoc, does not replace the primary grid, and indicates that exact null safety would require an explicit no-update candidate or an effectively infinite penalty.

## Artifact map

- Main results and MCSE: `ademp_v2_summary.csv`
- Complete method-level replicates: `ademp_v2_method_replicates.csv`
- Paired contrasts: `ademp_v2_paired_loss_comparisons.csv`
- Effective-replicate/failure audit: `ademp_v2_failure_summary.csv`
- DGM diagnostics: `ademp_v2_scenario_diagnostics.csv`
- Penalty selection: `ademp_v2_penalty_summary.csv`
- Submission table/figure: `ademp_v2_summary.tex`, `ademp_v2_summary_figure.pdf`, `ademp_v2_summary_figure.png`
- Post hoc sensitivity: `../simulation_transient_grid_sensitivity_v2/transient_grid_sensitivity_comparison.md`
