# Component learning stability checks

The files summarize the four component learning settings and the coverage conditional geometry analysis. Each setting contains 200 independent simulation datasets. Learning comparisons use the simulation seed 20261022; geometry comparisons use the retained study under seed 20261021. The replicate is the uncertainty unit.

`learning_stability.csv` retains all 192 planned learning comparisons, and `geometry_stability.csv` retains all 432 geometry comparisons. Their original means and simultaneous intervals are reproduced before the additional diagnostics are calculated. Gain proportions count strictly negative replicate differences, with exact pointwise 95 percent binomial intervals. Leave one out means and the means obtained after removing the ten smallest or ten largest replicate differences quantify influence. The seed block files report the first and second groups of 100 replicates and four consecutive groups of 50; these are partitions of the completed simulation experiment.

All nine informative setting and horizon comparisons of direct selected learning with constant diagonal profiling have negative reference means after every single replicate deletion and after removing the ten most favorable replicates. Their 18 half block means and 36 quarter block means are negative. Strict replicate gain proportions range from 0.64 to 0.73. Assessment mean differences are negative in the same nine comparisons, with eight simultaneous intervals below zero.

## Optimization budgets

The fixed job list contains all 203 replicates whose returned full tuning continuous fit reached the original 100 iteration budget. It also contains the first successful full tuning fit in each setting and selected branch, adding eight records. Every one of these 211 records is fitted with budgets of 300 and 500 iterations, using the original complexity coefficient, the same two deterministic initializations and the original fitting and tuning partitions. Affine calibrations and assessment and reference predictions are recalculated for the returned fits.

Complete threefold complexity selection is additionally repeated for 15 controls, chosen as the first replicate in each observed combination of setting, original full fit termination status and selected branch. The control identities are fixed in `optimization_budget_jobs.csv` by `redo_cv`. Both budgets retain the original complexity and branch in all 15 controls. At 500 iterations all 211 full tuning fits return successful termination. The 100 iteration results remain the original experiment; these calculations assess their sensitivity to the optimization budget.

`budget_paired_summary.csv` reconstructs each of the 192 planned comparisons at each budget by replacing the 211 affected fits and retaining the remaining 589 original fits. It reports intervals adjusted for 192 comparisons within a budget and for 384 comparisons across both budgets. All 18 direct reference comparisons in informative settings retain intervals below zero after the latter adjustment. The largest absolute change in the selected direct reference mean risk is 0.0000442705 mmHg. The per record objective changes, full CV decisions and mean risk changes are supplied in separate CSV files.

## Recalculation

From the extracted replication directory, run the supplied code with its frozen input files.

```bash
python code/audit_component_stability.py \
  --learning-source component_learning/simulation \
  --geometry-source geometry_mechanism \
  --output local_runs/stability
python code/prepare_component_budget_plan.py \
  --source component_learning/simulation \
  --output local_runs/budget_plan
python code/run_component_budget_sensitivity.py \
  --source component_learning/simulation \
  --jobs local_runs/budget_plan/optimization_budget_jobs.csv \
  --output local_runs/component_budget \
  --workers 4
python code/summarize_component_budget_sensitivity.py \
  --source component_learning/simulation \
  --results local_runs/component_budget \
  --jobs local_runs/budget_plan/optimization_budget_jobs.csv \
  --output local_runs/budget_summary
```

The runner checks the source design and fixed job identities and resumes completed records after checking their original complexity, selected branch and objective. New outputs contain replicate means and fitted function parameters. The supplied public summaries contain no individual stay scores.
