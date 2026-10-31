# Saved clinical result audit

The program audits the saved clinical results for the 60,310 eligible first adult ICU stays in this article. It checks the complete fitting, tuning and assessment allocations, reconstructs training-sample age standardization, and reproduces all 252 model-summary cells from retained stay-level scores. The output contains aggregate statistics.

Python requires NumPy and pandas, using the article's recorded environment. The clinical cache contains an object defined in the article's `split_window_data.py`, so supply the article's code directory through `--project-code`. R is used for its base `p.adjust(method="holm")` function. All intermediate R files are created within the output directory.

Credentialed analysis inputs are supplied through command-line parameters. They comprise the saved clinical cache, component-learning assessment scores, adaptive clinical partition configurations, random-slope assessment scores at 31 and 61 quadrature nodes, and the previously generated aggregate comparison tables. These input records remain in their governed analysis environment.

```bash
python audit_saved_clinical.py \
  --component-clinical-root PATH_TO_COMPONENT_CLINICAL \
  --adaptive-clinical-root PATH_TO_ADAPTIVE_CLINICAL \
  --cached-dataset PATH_TO_CLINICAL_CACHE \
  --component-public-root PATH_TO_COMPONENT_AGGREGATES \
  --adaptive-public-root PATH_TO_ADAPTIVE_AGGREGATES \
  --original-complete-paired PATH_TO_COMPLETE_FAMILY_PAIRED_CSV \
  --slope31-root PATH_TO_31_NODE_RESULTS \
  --slope61-root PATH_TO_61_NODE_RESULTS \
  --project-code PATH_TO_ARTICLE_CODE \
  --output-root PATH_TO_AGGREGATE_OUTPUT
```

The component root contains `fold_1` through `fold_5` and `temporal`, each with `assessment_h4.csv`, `assessment_h8.csv` and `assessment_h12.csv`. The adaptive root contains `horizon_4`, `horizon_8` and `horizon_12`; nested partition configurations reside in `nested/fold_1` through `nested/fold_5`, and chronological configurations in `temporal`. Each random-slope root contains `temporal/assessment_horizon_12.csv`.

The five additional audit contrasts retain learned direct versus matched constant direct profiling, learned affine versus matched constant affine profiling, each learned form versus calibrated q10, and learned affine profiling versus the finite feature selector. Every outer fold and forecast horizon is retained. The out-of-fold and fold-omission summaries describe the saved fitted forecasts. Within-fold intervals and chronological intervals concern paired risks of frozen fitted rules.

Three-horizon chronological intervals use two-sided Bonferroni normal limits within each stated contrast. Holm adjustment retains each complete comparison family separately: 42 original complete-family comparisons, 255 adaptive horizon comparisons, 120 component-learning comparisons, six quadrature-sensitivity comparisons and four additional count interactions. The original pointwise intervals remain available alongside the adjusted P values.

Count groups use the median early observation count among all 60,310 eligible stays, which equals 13. The threshold is calculated from timestamps by the 12 hour landmark. The count interactions are additional descriptive analyses. Influence diagnostics include all single-stay omissions, predefined demographic groups, count groups, early-q10 quintiles and both favorable and unfavorable outcome-ranked deletions. Outcome-ranked deletion changes the population composition and is reported as a diagnostic.

`clinical_public_manifest.json` records the aggregate schemas and verifies that identifier columns and personal filesystem paths are absent from the output.
