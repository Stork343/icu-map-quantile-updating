# Continuous constant penalty comparisons

This study separates continuous optimization of horizon specific penalties from modulation by early history. The continuous constant fit fixes the six history coefficients at zero and optimizes the six horizon intercepts within the same bounds as the history dependent model. Grouped validation also retains the original 48 matrix grid branch. Population fits, observation records and original history dependent predictions remain fixed. The complete design is in `code/CONTINUOUS_CONSTANT_PROTOCOL.md` in the accompanying reproducibility directory.

## Simulation outputs

The `simulation` directory contains results for all 800 repeated datasets under the four original settings. Its 192 paired summaries cover eight contrasts, three forecast horizons and both assessment and reference populations. Each summary retains all 200 independent repeated datasets for its setting. `paired_summary.csv` supplies pointwise and simultaneous intervals, while `optimizer_summary.csv` records the numerical stopping conditions.

The validation selected constant procedure reduces reference loss relative to the original grid in the nine informative setting and horizon combinations. The selected history procedure has no simultaneous reference interval below the selected continuous constant procedure in these simulations. All settings, calibrated comparisons and original prediction replay checks are supplied.

## Clinical outputs

The `clinical` directory supplies aggregate results for the five nested partitions and the temporal partition. History dependent direct loss is lower than continuously tuned constant loss at every nested horizon and in all fifteen fold and horizon combinations. Temporal differences vary with the forecast window. The direct difference is positive at four hours and negative at eight and twelve hours, with simultaneous intervals excluding zero for all three.

Nested fold means and omissions describe saved out of fold forecasts. Temporal intervals condition on frozen rules. The 24 temporal contrasts use Bonferroni simultaneous normal intervals and separately calculated Holm adjusted tests. The two procedures can give different significance decisions. The clinical directory contains no individual clinical records.

## Reproduction commands

Run these commands from the extracted reproducibility directory. The existing `component_learning/simulation` directory supplies the original fitted objects and seed records.

```sh
python code/run_continuous_constant_simulation.py --source component_learning/simulation --output local_runs/continuous_constant --workers 6
python code/export_continuous_constant_simulation.py --source local_runs/continuous_constant --output local_runs/continuous_constant_public
Rscript code/verify_continuous_constant_summaries.R analysis_continuous_constant/simulation simulation_verification.csv
python code/format_continuous_constant_tables.py --source analysis_continuous_constant --output local_runs/continuous_constant_tables
```

Reconstructing the clinical comparison requires credentialed MIMIC inputs and the original private sample configuration. Its runner accepts explicit paths to the original public `clinical_learning_fits.json` and `clinical_affine_coefficients.csv` exports. `clinical/README.md` documents these inputs and the verification commands. Independent R summary and interval checks are included alongside the respective public aggregates.

## Directional theorem verification

`nonsmooth_gain_verification.json` and `nonsmooth_profile_direction_checks.csv` report theorem verification on synthetic mathematical examples. These calculations evaluate the directional profile and risk formulas in the supplementary material, separately from the 800 replicate simulation study. They do not contribute observations, estimates or comparison intervals to that study.

The verification uses the production `solve_spd_profile` solver. Eight constructed index records cover residual ties with interior and boundary subgradients, a zero critical cone and a one dimensional critical cone. Another 64 records have nonzero residuals. The calculations compare the directional quadratic program with one sided profile differences in 144 matrix directions and check 1,152 objective expansions. Fifteen later residual examples test the check loss derivative, including positive and negative changes at a tie.

Twelve synthetic stays then evaluate the centered history path over 4, 8 and 12 hour horizons using `make_profile_batch` and `penalty_values`. Their index counts range from 4 to 16, and all three history features vary. Two future outcome constructions include residual ties. One has a positive directional risk derivative; the other has derivative -0.516232, and its risk satisfies the proposed local descent inequality at all six checked positive step sizes. Across all examples, 1,368 production profile solves pass their numerical certificates. Finite numerical checks accompany the mathematical proof.

```sh
python code/verify_nonsmooth_gain.py --output-dir analysis_continuous_constant
```
