# ICU MAP penalized quantile regression

Reproducibility materials for the manuscript **Penalized Quantile Regression for Lower Tail Persistence in ICU Mean Arterial Pressure Trajectories**.

Repository: <https://github.com/Stork343/icu-map-quantile-updating>

Submission release: `v1.2.1`

## What is included

- `code/`: cohort construction, primary analysis, nested cross fitting, calibration, sensitivity analyses, figure generation, simulation, and regression tests.
- `simulation_ademp_v2/`: complete synthetic results for the 14 mechanism ADEMP simulation with 200 replicates per mechanism.
- `transient_grid_sensitivity/`: synthetic post hoc penalty grid sensitivity for the transient mechanism.
- `empirical_aggregate/`: aggregate CSV, JSON, TeX, and figure artifacts used in the MIMIC-IV application.
- `validation_aggregate/`: aggregate nested cross fitting, fixed observation opportunity, comparator, and calibration outputs.
- `source_sensitivity/`: aggregate invasive-only and noninvasive-only results.
- `supplement_aggregate/`: source tables, source CSV files, and figures for the expanded mathematical, simulation, and MIMIC-IV supplement.

Figure 1 is supplied as an editable draw.io file, a standalone SVG, a vector PDF, and a 300 dpi PNG. Its Times New Roman typography matches the manuscript body. The training, tuning, and held-out assessment lanes use solid connectors for records or predictions and dashed connectors for fitted quantities frozen before assessment.

## Data boundary

This repository contains no row-level MIMIC-IV data, patient identifiers, stay identifiers, row-level feature files, or row-level out-of-fold predictions. Reproducing the empirical analysis from source requires credentialed local access to [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) and acceptance of the PhysioNet data use agreement. All simulation data and replicate outputs in this repository are synthetic.

## Software environment

The archived analysis used Python 3.12.1 and R 4.3.2. Install the Python dependencies with

```bash
python -m pip install -r requirements.txt
```

Install the two R dependencies with

```r
install.packages(c("data.table", "quantreg"))
```

The archived versions were `data.table` 1.17.8 and `quantreg` 6.1. A conda specification is also provided in `environment.yml`.

## Verification tests

From the repository root, run

```bash
python -m unittest discover -s code/tests -v
```

The release was checked with 38 passing tests. They cover empirical quantile conventions, scalar and vector profiled updates, stable tie handling, data split isolation, fixed observation opportunity, probability mass calibration, informative monitoring, and deidentified export.

## Regenerate expanded supplement artifacts

The expanded supplemental tables and figures are generated from public aggregate files:

```bash
python code/generate_expanded_supplement_tables.py
python code/plot_expanded_supplement_figures.py
```

These commands write LaTeX tables, aggregate source CSV files, vector PDFs, and PNG previews to `supplement_aggregate/`. They do not require row-level MIMIC-IV data.

## Reproduce a synthetic smoke run

The following command runs one small replicate of the primary simulation without MIMIC-IV data:

```bash
python code/run_split_window_simulation.py \
  --design ademp-v2 \
  --n-rep 1 \
  --n-stays 60 \
  --scenarios ideal_large_dense \
  --out-dir local_runs/smoke/output \
  --work-dir local_runs/smoke/work \
  --progress-every 1
```

The complete archived design used seed `20260529`, all 14 mechanisms, and 200 replicates per mechanism:

```bash
python code/run_split_window_simulation.py \
  --design ademp-v2 \
  --n-rep 200 \
  --seed 20260529 \
  --out-dir local_runs/ademp_v2/output \
  --work-dir local_runs/ademp_v2/work
```

The simulation calls `Rscript code/fit_grouped_quantile_common.R` for common quantile fits.

## Reproduce the empirical analysis

Assume `/path/to/mimiciv/3.1` contains the credentialed MIMIC-IV 3.1 files. Build the first-day MAP cache with

```bash
python code/build_revised_mimic_map_cache.py \
  --data-root /path/to/mimiciv/3.1 \
  --cache-dir local_runs/mimic_cache \
  --metadata-output local_runs/mimic_cache/metadata.json \
  --map-source combined
```

Run the primary fit, tune, and assessment analysis with

```bash
python code/run_split_window_mixed_effects_analysis.py \
  --obs-cache local_runs/mimic_cache/mimic_map_observations.parquet \
  --stays-cache local_runs/mimic_cache/mimic_map_stays.parquet \
  --data-root /path/to/mimiciv/3.1 \
  --artifact-dir local_runs/primary/artifacts \
  --work-dir local_runs/primary/work \
  --output local_runs/primary/results.json \
  --clinical-output local_runs/primary/clinical_results.json \
  --seed 20260522
```

Run nested fivefold assessment, calibrated scalar comparators, fixed observation opportunity analyses, and tie aware calibration with

```bash
python code/run_submission_validation_extensions.py \
  --obs-cache local_runs/mimic_cache/mimic_map_observations.parquet \
  --stays-cache local_runs/mimic_cache/mimic_map_stays.parquet \
  --results-json local_runs/primary/results.json \
  --output-dir local_runs/validation \
  --work-dir local_runs/validation/work \
  --seed 20260522 \
  --outer-folds 5
```

The empirical analysis is computationally intensive. The aggregate outputs needed to inspect the reported manuscript results are already included in this repository.

## Reported uncertainty

Empirical loss intervals are descriptive intervals based on stay-level standard errors conditional on the realized fitted or fold-trained rules unless a table states otherwise. Monte Carlo intervals quantify simulation error across independent replicates.

## Citation

Please use the metadata in `CITATION.cff` and cite the accompanying manuscript when it becomes available.
