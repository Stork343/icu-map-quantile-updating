# Frozen rule interval coverage

Place `run_frozen_risk_coverage.py` in `code/` beside the supplied risk inference and simulation modules. The input directory contains the original `risk_inference/replicates/` records with frozen coefficients, tuning choices and seeds.

```bash
OPENBLAS_NUM_THREADS=1 python code/run_frozen_risk_coverage.py \
  --inference-dir risk_inference \
  --out-dir local_runs/interval_coverage \
  --scenarios common_time_misspecified,ideal_large_dense,cluster_size_informative \
  --n-rep 200 \
  --assessment-sizes 120,600,1800 \
  --reference-n 100000 \
  --reference-block-size 10000 \
  --bootstrap-replicates 2000 \
  --seed 20261025 \
  --workers 6
```

The first assessment size replays the original saved assessment population. The new cohort has the largest requested size, and its prefixes supply the smaller new assessments. Population coefficients, the scalar penalty and affine calibration coefficients are read from each saved rule set.

The program stores per-replicate assessment loss arrays, interval endpoints, reference-block sums and squared sums, and completion metadata. Its final outputs include all interval records, coverage summaries with exact binomial intervals, and the completion manifest. `--summarize-only` reconstructs these summaries from completed records. `--benchmark` evaluates the first complete rule set in a separately specified output directory.

Studentized intervals require positive finite bootstrap standard errors. A draw that fails this condition terminates its replicate. The completed study contains all 600 planned rule sets.
