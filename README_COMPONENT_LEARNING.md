# Learning separate level and slope penalties

This study learns two continuous penalty functions from early record features and the requested 4, 8, or 12 hour forecast window. Each fitting partition shares one pair of functions across the three windows. The protocol is supplied in `code/COMPONENT_LEARNING_PROTOCOL.md`.

Commands below run from the extracted `reproducibility_code/` directory. The component programs have development defaults referring to local research directories. Supply every input and output path shown in these commands when using the extracted package.

## Environment and tests

Install the versions in `requirements.txt` using Python 3.12. The component solver uses `numba` 0.61.2 and `llvmlite` 0.44.0. Clinical affine calibration additionally uses the installed R packages described in `README.md`.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=code python -m unittest discover -s code/tests -p 'test_component*.py' -v
```

The tests compare actual quadratic quantile profiles with Clarabel, check profile derivatives against finite differences, and exercise grouped validation on generated quantile records. They also verify the ordinary retained outcome scoring convention and horizon specific weighting.

## Supplied results

`component_learning/simulation/` contains 800 newly generated replicates under seed `20261022`, with 200 replicates for each of the four settings. Every replicate fits a population model on 360 stays, learns prediction rules on 120 tuning stays, and scores 120 assessment stays plus an independent 2,000 stay reference population. All comparison methods are fitted and tuned on these same generated records.

The directory supplies replicate metrics and frozen parameters, method summaries, 192 paired contrast summaries, and selection frequencies. Pointwise and simultaneous intervals use the independent replicate as the Monte Carlo unit. The simultaneous family contains all 192 planned contrasts. Both continuous and validation selected procedures are retained in direct and postcalibrated forms.

`component_learning/clinical/` contains aggregate nested and temporal losses, probability mass bracket violations, learned parameters, affine coefficients, validation scores, and numerical checks. Its twelve files follow `clinical_public_schema.json`. The five nested folds and temporal partition give 18 horizon comparisons. The nested assessment counts are 59,969, 60,241, and 60,310 at 4, 8, and 12 hours; the corresponding temporal counts are 21,843, 21,976, and 22,005.

The clinical table retains seven prediction rules in each assessment panel and the learned minus matched constant differences for both direct prediction and affine calibration. Those temporal intervals are pointwise paired normal 95 percent intervals, conditional on the frozen fitted rules. `clinical_temporal_paired.csv` retains all 120 temporal comparisons.

Individual clinical features and predictions remain in the credentialed local calculation directory. Public clinical exports contain aggregate quantities and fitted function parameters. Synthetic per stay simulation metrics are supplied with the simulation results.

## Learning and scoring

Threefold validation uses the original tuning population. All horizon records from the same stay share one fold. Training loss gives equal weight to the three horizons and equal weight to eligible stays within each horizon. The validation candidates use history coefficient penalties `0`, `0.001`, `0.01`, and `0.1`, together with the exact constant diagonal branch.

Every inner training fold chooses its own 48 matrix constant baseline separately at each horizon. Continuous optimization uses two deterministic starts and at most 100 L-BFGS-B iterations. After validation, the chosen complexity is refitted on the full tuning population. The best continuous rule and the selected procedure are both stored. Ordinary affine maps are fitted afterward on all tuning stays separately at each horizon.

Published check loss and strict or weak tail probabilities compare the raw retained outcome with the actual returned forecast. The learning objective keeps its defined residual arithmetic. Saved profile diagnostics include objective gaps, projected optimality checks, and stationarity checks; fitted maps also record the outer optimizer stopping status and gradient norms.

## Replay the supplied simulation

The independent verifier reconstructs tuning selections and the inner validation predictions from saved parameters, then rebuilds the assessment and reference scores and all paired summaries.

```bash
NUMBA_NUM_THREADS=1 python code/verify_component_learning_simulation.py \
  component_learning/simulation --workers 4
python code/verify_component_selection_exports.py component_learning/simulation
```

The completed report checks 800 replicates and 192 interval summaries. It records replay of 3,168,000 assessment scores across stays, horizons, and all eleven rules, together with the separate reference risk replay and every baseline tuning choice. Frozen parameters are retained during this verification. `selection_export_verification.json` separately checks all 800 exported branch decisions and the 31 selection frequency records.

## Generate the fresh simulation

The simulation uses the fixed master seed `20261022` and the complete frozen protocol. Run the full calculation with:

```bash
NUMBA_NUM_THREADS=1 python code/run_component_learning_simulation.py \
  --output local_runs/component_learning/simulation \
  --replicates 200 --workers 4
NUMBA_NUM_THREADS=1 python code/verify_component_learning_simulation.py \
  local_runs/component_learning/simulation --workers 4
```

For an executable eight replicate calculation with the same sample sizes and method settings, use `--replicates 2` and a separate output directory. Completed calculation directories can be reused by the runner after its configuration and frozen result checks. The full manuscript table formatter requires 200 replicates in each setting.

## Reproduce the clinical calculations

Build the credentialed MIMIC-IV cache and the complete family calculations using `README.md`. Generate the history and horizon allocations using the clinical commands in `README_ADAPTIVE.md`. The component calculation reuses those population coefficients, age transformations, sample assignments, and original tuning profiles.

```bash
NUMBA_NUM_THREADS=2 python code/run_component_clinical.py \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --source-clinical local_runs/adaptive/clinical \
  --inherited-family local_runs/complete_family \
  --source-cache local_runs/mimic_cache \
  --patients-file /path/to/mimiciv/3.1/hosp/patients.csv.gz \
  --output local_runs/component_learning/clinical \
  --public-output local_runs/component_learning/public/clinical \
  --scope all --max-iter 100 --starts 2
NUMBA_NUM_THREADS=2 python code/verify_component_clinical.py \
  --dataset-cache local_runs/complete_family/local_dataset.pkl \
  --source local_runs/adaptive/clinical \
  --private local_runs/component_learning/clinical \
  --public local_runs/component_learning/public/clinical
```

The clinical seed is `20261022`. The independent replay checks the six fitting partitions and all 18 horizon comparisons. It reconstructs the original assessment membership and scores, 252 model summary rows, and all 120 temporal paired intervals. Across the partitions it checks 78 inner validation or full tuning fits and 246,344 assessment horizon records. Repeated numerical profile evaluations are computation counts, and the eligible ICU stay remains the unit for clinical loss aggregation.

## Regenerate numerical tables

The following commands read the supplied public results and write new CSV files under `local_runs/`. The manuscript and bilingual reader use the same CSV values and shared LaTeX templates in the accompanying source packages.

```bash
python code/format_component_simulation_tables.py \
  --source component_learning/simulation \
  --output-dir local_runs/component_tables \
  --manifest local_runs/component_tables/simulation_validation.json
python code/format_component_clinical_table.py \
  --source component_learning/clinical \
  --output local_runs/component_tables/component_clinical_comparison.csv \
  --manifest local_runs/component_tables/clinical_validation.json
```

The simulation formatter checks the entire setting and contrast grid and reconstructs both interval types. The clinical formatter checks equal assessment counts across compared rules, the bracket violation definition, and the matched paired interval calculations before writing the table.
