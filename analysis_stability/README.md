# Analysis stability and interval accuracy

The learning, clinical and simulation directories contain the aggregate results supporting the stability section of the Supplementary Material. The original study estimates remain in their original result directories. These files report replicate influence, optimization budget sensitivity, saved clinical fold comparisons, complete temporal comparison families and a fixed rule interval coverage experiment.

The clinical directory contains aggregate quantities. Reconstructing those quantities from individual MIMIC-IV records requires credentialed local inputs, supplied as command-line arguments to `code/audit_saved_clinical.py`. Its companion `code/adjust_clinical_families.R` uses base R Holm adjustment. No individual clinical records are included here.

The learning directory includes a README with commands to reconstruct all 192 learning and 432 geometry stability summaries and the fixed 211 record optimization budget check. Fifteen controls additionally repeat complete grouped cross validation at the two expanded budgets. Budget sensitivity reuses the existing simulation datasets.

The additional coverage study retains 600 fitted rule sets across three settings. It replays the original 120 assessment stays and adds independent assessment cohorts of 600 and 1,800 stays. Each rule set has a separately generated reference population of 100,000 stays. All 81 coverage cells, 16,200 intervals and reference block summaries are supplied. From the extracted reproducibility directory, run:

```bash
python code/run_frozen_risk_coverage.py \
  --inference-dir risk_inference \
  --out-dir local_runs/frozen_coverage \
  --workers 6
python code/format_analysis_stability_tables.py \
  --source analysis_stability \
  --output local_runs/stability_tables
```

The scripts use the package's declared Python and R environment. Seed 20261025 and the full fixed design are recorded in `simulation/coverage_extension/design.json`. Completion records validate the configuration when resuming. Additional intervals are evaluated against an estimated reference risk; the reference Monte Carlo precision is reported separately.
