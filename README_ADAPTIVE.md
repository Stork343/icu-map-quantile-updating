# History and horizon adaptive quantile profiling

This extension evaluates a structured individual penalty at the 12 hour landmark for subsequent windows of 4, 8, and 12 hours. Its 42 base profiles and ordinary affine calibrated forms join the original 116 prediction rules. The mixed population simulation evaluates all 200 rules at each requested horizon. A feature selector provides a comparison using the same early record features and the original rule family.

The numerical environment uses Python 3.12 with the versions in `requirements.txt`, R 4.3, `quantreg` 6.1, `lqmm` 1.5.8, and `data.table` 1.17.8. Positive definite profile objectives use Clarabel followed by active set polishing and independent primal and dual checks. Clinical coefficients inherited from the original diagonal penalty family retain their OSQP certificates.

The R package `jsonlite` writes numerical validation summaries. Install the required R packages with `install.packages(c("data.table", "quantreg", "lqmm", "jsonlite"))`.

## Mixed population simulation

Run the full study from the extracted reproducibility directory:

```bash
python code/run_adaptive_mixed_simulation.py \
  --output local_runs/adaptive/simulation \
  --replicates 200 \
  --n-stays 600 \
  --reference-stays 2000 \
  --workers 4 \
  --seed 20261021
python code/verify_adaptive_mixed_simulation.py local_runs/adaptive/simulation
```

The four settings combine persistent level, transient recovery, and persistent shape within a single population. Each replicate fits the common component on 360 stays, tunes on 120, and assesses 120. A separate reference population contains 2,000 stays. The information limit setting makes the complete index law independent of latent type. The serial integer setting adds AR(1) dependence and integer rounding. Generated outcomes lie in the retained clinical MAP range; forecasts use their ordinary values and ordinary check loss.

Use `--resume` with the same output directory and scientific settings to continue an interrupted calculation. Worker count changes execution scheduling. Completed replicates are verified before reuse. Aggregate files report both the mean replicate calibration violation and the violation of the mean probability mass bracket. Paired differences include Monte Carlo standard errors. Reference risks evaluate the selected fitted rules.

## Clinical calculations

The existing README gives cache construction and fitting, tuning, and assessment commands for credentialed MIMIC-IV 3.1 records. The extended family calculation uses those same partitions and population fits. After running `run_expanded_family_validation.py`, supply the resulting cache and coefficients through explicit paths:

```bash
python code/run_adaptive_clinical.py \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --source-cache-dir local_runs/mimic_cache \
  --inherited-family-dir local_runs/complete_family \
  --nested-population-dir local_runs/validation \
  --temporal-population-file local_runs/temporal/temporal_validation_results.json \
  --patients-file /path/to/mimiciv/3.1/hosp/patients.csv.gz \
  --output-dir local_runs/adaptive/clinical \
  --workers 4
python code/run_random_slope_comparator.py \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --patients-path /path/to/mimiciv/3.1/hosp/patients.csv.gz \
  --output-dir local_runs/adaptive/random_slope
python code/summarize_adaptive_clinical.py \
  --clinical-dir local_runs/adaptive/clinical \
  --random-slope-dir local_runs/adaptive/random_slope \
  --output-dir local_runs/adaptive/clinical_aggregate
```

Random slope LQMM fitting uses the baseline covariates and normalized time as fixed effects, with a correlated Gaussian random intercept and slope. Product Gauss Hermite quadrature continues from 7 to 15 and then 31 nodes per dimension. Both optimization levels must converge. New stay effects use index measurements and the implemented best linear predictor with asymmetric Laplace moments. Affine calibration is fitted separately for each horizon.

The temporal 61 node sensitivity continues the saved 31 node fit with unchanged controls. Run this command after the 31 node comparison:

```bash
Rscript code/refine_random_slope_parallel_checked.R \
  code \
  local_runs/adaptive/random_slope/temporal/model.rds \
  61 8 \
  local_runs/adaptive/random_slope_61/temporal/parameters.csv \
  local_runs/adaptive/random_slope_61/temporal/model.rds \
  local_runs/adaptive/random_slope_61/temporal/parallel_validation.json
python code/run_random_slope_comparator.py \
  --scope temporal --nodes 61 --quadrature-stages 7 15 31 61 \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --patients-path /path/to/mimiciv/3.1/hosp/patients.csv.gz \
  --output-dir local_runs/adaptive/random_slope_61
```

The refinement checks the full fitting likelihood against the installed `lqmm` likelihood before using additive independent stay blocks. The scoring program checks the saved model's outcomes, covariates, random design, and group membership against the current fitting data before evaluating it. The 31 and 61 node results are reported separately.

Clinical public outputs contain aggregate model scores and fitted coefficients. Individual records, model objects containing records, and row level predictions remain in the credentialed research directory. Horizon specific eligibility requires a later observation in the requested window. The nested cohorts therefore contain 59,969, 60,241, and 60,310 stays for 4, 8, and 12 hours, respectively.

## Numerical verification

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -p 'test_adaptive*.py'
PYTHONPATH=code python -m unittest code/tests/test_random_slope_comparator.py
Rscript code/fit_random_slope_lqmm.R --self-test
```

The profile tests exercise primal and dual equality on actual quantile objectives, including near-bound dual variables. The random slope test compares the matrix calculation with `lqmm::ranef` on the package's Orthodont data. Simulation verification reconstructs aggregate estimates from completed replicate files and checks candidate membership, frozen rules, loss metrics, and solver certificates.

The quadrature comparison uses tuning records and separately fitted tuning affine maps:

```bash
python code/check_random_slope_quadrature.py \
  --model-dir local_runs/adaptive/random_slope \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --partition temporal \
  --nodes 15 31 \
  --output-dir local_runs/adaptive/quadrature_precision
```

For the 31 to 61 node comparison, set `--nodes 31 61` and add `--second-model-dir local_runs/adaptive/random_slope_61`. The actual fitting data comparison of the official and parallel likelihood, including a 64 stay serial and parallel fit, can be rerun with:

```bash
Rscript code/verify_parallel_lqmm_likelihood.R \
  code local_runs/adaptive/random_slope/temporal/model.rds \
  61 8 64 local_runs/adaptive/parallel_likelihood_validation.json
```

## Sampling coverage and forecast horizon

The follow-up mechanism analysis reuses every completed mixed population replicate and its frozen fitting and tuning records. Full geometry, horizon geometry with gamma equal to zero, and diagonal level with slope profiling are compared in every early sampling interval and forecast window. Joint selection considers direct and affine calibrated rules. A direct only comparison examines the penalty without choosing a calibration. Selection always uses all 120 tuning stays. The original assessment population and the independently generated 2,000 stay reference population are evaluated after the rules have been frozen.

The protocol in `code/GEOMETRY_MECHANISM_PROTOCOL.md` identifies this as an additional analysis designed after the overall simulation results were available and before coverage specific estimates were examined. Assigned coverage is the interval from which early times were sampled. Actual first to last observation spans are reported separately.

From the extracted package root, run:

```bash
python code/run_geometry_mechanism_validation.py \
  --source adaptive_study/simulation \
  --output local_runs/geometry_mechanism \
  --workers 12
python code/verify_geometry_mechanism.py \
  local_runs/geometry_mechanism \
  --source adaptive_study/simulation \
  --workers 4
python code/format_geometry_mechanism_tables.py \
  --source local_runs/geometry_mechanism/paired_summary.csv \
  --output-dir local_runs/geometry_tables
python code/plot_geometry_mechanism.py \
  --input local_runs/geometry_mechanism/paired_summary.csv \
  --selection-mode joint --scope reference \
  --output local_runs/geometry_figures/geometry_mechanism.pdf
python code/plot_geometry_mechanism.py \
  --input local_runs/geometry_mechanism/paired_summary.csv \
  --selection-mode direct --scope reference \
  --output local_runs/geometry_figures/geometry_mechanism_direct_reference.pdf
```

The supplied `geometry_mechanism/` directory contains all replicate stay metrics and aggregate results. It can be independently checked with the verification command above after replacing `local_runs/geometry_mechanism` with `geometry_mechanism`. The independent unit for Monte Carlo uncertainty is the replicate. Pointwise intervals use Student t quantiles; simultaneous intervals cover all 432 planned contrasts using Bonferroni adjustment. Both interval types are retained in the full numerical tables, and the figures display pointwise intervals.

The plotting commands require Times New Roman to be installed. The exported vector PDFs embed the figure fonts. The code checks font size, text bounds and the numeric values passed to every plotted point and interval.

## Learned component penalties

The continuous component extension learns separate level and slope penalties jointly across the three forecast horizons. It uses threefold validation grouped by ICU stay, with two deterministic optimization starts and a 100 iteration limit. `README_COMPONENT_LEARNING.md` provides complete package relative commands for fresh simulation, independent replay, clinical reproduction, tests, and table generation.

The new simulation uses seed `20261022`, with 200 fresh replicates in each of the four settings. Its supplied results are in `component_learning/simulation/`; independently verified aggregate clinical results are in `component_learning/clinical/`. Public clinical files contain aggregate scores and fitted parameters, while individual clinical calculations use credentialed local records. The component solver requires `numba` 0.61.2 and `llvmlite` 0.44.0 from `requirements.txt`.
