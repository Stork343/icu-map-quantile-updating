# ICU MAP quantile profiling

Reproducibility materials for **Quantile Profiling for Lower Tail Persistence in ICU Mean Arterial Pressure** by Jian Hou and Maozai Tian. Version 1.1.0 accompanies the manuscript prepared on 31 October 2026.

The method profiles individual lower tail level and slope around a population component. Two positive penalty functions use the early record and requested horizon. Later check loss, with equal weight per ICU stay, trains the functions. Matched continuously tuned constants distinguish history modulation from the resolution of penalty search.

## Results and inputs

`component_learning/` supplies the 800 independent component learning simulation replicates and aggregate nested and temporal MIMIC-IV results. `analysis_continuous_constant/` supplies the matched constant controls on all 800 replicates and six clinical fitting partitions. Its simulation family has 192 paired contrasts; its temporal clinical family has 24. All original history dependent fits and selections remain fixed in this comparison. The protocol and independent R checks are included.

Continuously tuned constants attained lower mean reference risk than history dependent candidates in the mixed population simulations. Nested clinical direct losses favored history modulation in all 15 fold and horizon comparisons after matching continuous tuning. Temporal direct comparisons favored history modulation at eight and twelve hours and shared continuous penalties at four hours. Direct and affine calibrated results are supplied separately.

`adaptive_study/` retains the finite family mixed population study. `geometry_mechanism/` evaluates how early observation coverage relates to forecast geometry. `analysis_stability/` contains replicate deletion, optimization budget and grouped clinical comparisons, with the frozen rule assessment size experiment. `risk_inference/` contains independent assessment and cross fitting outputs, including the retained fitted inputs for 600 frozen rule replicates. Their reconstruction and replay checks are documented in that directory.

The original 14 mechanism prediction study is in `simulation_ademp_v2/`. The `strengthening_aggregate/` directory supplies complete prediction family and published LQMM comparisons. Source and monitoring sensitivities are in `source_sensitivity/` and `validation_aggregate/`. Numerical files used by the supplementary tables are in `supplement_aggregate/`.

## Software

Install the pinned Python packages from `requirements.txt`. The analysis uses R with `data.table`, `quantreg`, `lqmm` and `jsonlite`. The recorded study environments are in `environments/`. R scripts run through `Rscript`.

```bash
python -m pip install -r requirements.txt
```

```r
install.packages(c("data.table", "quantreg", "lqmm", "jsonlite"))
```

Run commands from the extracted package root. Keep generated runs in the ignored `local_runs/` directory.

## Verification

```bash
python -m unittest discover -s code/tests -v
python code/verify_public_package.py
```

The integration command regenerates seven supplementary tables and two diagnostic figures from aggregate inputs. It checks that the tables reproduce exactly, input files remain unchanged, and editable sources for the two introductory figures are present.

The continuous constant comparison has separate arithmetic checks in Python and R. The public `analysis_continuous_constant/validation.json` records the completed checks of 800 simulation datasets and six clinical partitions. Patient level clinical scores used for the independent verification require credentialed local access.

## Component learning and continuous constant control

`README_COMPONENT_LEARNING.md` gives the original simulation and clinical commands. `README_CONTINUOUS_CONSTANT.md` gives the matched constant commands and file descriptions. The exact study definitions are in:

- `code/COMPONENT_LEARNING_PROTOCOL.md`
- `code/CONTINUOUS_CONSTANT_PROTOCOL.md`
- `code/GEOMETRY_MECHANISM_PROTOCOL.md`

The constant control fixes six history coefficients at zero while retaining the six horizon intercepts. It uses the same penalty ranges, grouped validation assignments, initializations and optimizer budget. Each record still supplies its own individual quantile profile. The control selects between continuous constants and the original 48 matrix branch without changing the original history dependent procedure.

Simulation reproduction uses `code/run_continuous_constant_simulation.py`; aggregate export uses `code/export_continuous_constant_simulation.py`. `code/verify_continuous_constant_summaries.R` independently checks paired numerical summaries. Clinical reproduction uses `code/run_continuous_constant_clinical.py` with the original fitted coefficient and affine coefficient exports. `code/verify_continuous_constant_clinical.py` checks the paired clinical outputs; `code/verify_continuous_constant_clinical_intervals.R` independently calculates the temporal intervals and nested fold summaries.

## Original prediction studies

For the 14 mechanism prediction study, run:

```bash
python code/run_split_window_simulation.py \
  --design ademp-v2 --n-rep 200 --seed 20260529 \
  --out-dir local_runs/ademp_v2/output \
  --work-dir local_runs/ademp_v2/work
```

The finite family history and horizon study is documented in `README_ADAPTIVE.md`. Risk inference and paired bootstrap use `code/run_risk_inference_simulation.py` and `code/run_paired_bootstrap_validation.py`, respectively. Their run manifests retain the seeds and assessment sample sizes used for the supplied results.

## Clinical reproduction

MIMIC-IV 3.1 source data require PhysioNet credentialing and the data use agreement. This archive supplies aggregate clinical quantities. Individual clinical records and patient identifiers remain in the credentialed environment.

Create the first day MAP cache using the admission age proxy:

```bash
python code/build_revised_mimic_map_cache.py \
  --data-root /path/to/mimiciv/3.1 \
  --cache-dir local_runs/mimic_cache \
  --metadata-output local_runs/mimic_cache/mimic_map_full_cache_metadata.json \
  --map-source combined --age-definition icu_proxy
```

`code/run_split_window_mixed_effects_analysis.py` fits the population component, selects the scalar penalty and assesses prediction. `code/run_submission_validation_extensions.py` supplies nested fivefold assessment and fixed observation opportunity analyses. `code/run_temporal_validation.py` assigns stays by aligned complete containment in the 2008 to 2013 fitting period, 2014 to 2016 tuning period and 2017 to 2022 assessment period.

The common clinical configuration is `config/analysis_config.json`. Every clock landmark requires at least eight total, four index and one later observation. Horizon specific evaluation retains the original stay partitions and tests later record availability separately for each horizon.

`code/run_expanded_family_validation.py` compares the complete prediction family. `code/run_published_quantile_comparator.py` fits Gaussian random intercept LQMM prediction. Random slope comparisons and their quadrature sensitivities are documented in the adaptive study materials. `code/summarize_strengthening_comparisons.py` combines only matching assessment records and fitting definitions.

## Figures

Figure 1 has an editable TikZ source at `code/figure_specs/introduction_overview.tex`. Figure 2 is generated by `code/plot_method_illustration.py`. Their numerical source records and fitted coordinates are in `empirical_aggregate/introduction_figure_source/` and `empirical_aggregate/method_figure_source/`.

Both figures include vector PDF, SVG and PNG exports. Figure 2 shares `code/manuscript_plot_style.py` with the later clinical figures. `code/figure_specs/README.md` supplies the reproduction commands and numerical layout checks.

## Uncertainty

Simulation intervals describe Monte Carlo uncertainty across independent datasets. Paired temporal clinical intervals condition on fitted and tuned rules. Nested clinical outputs report pooled losses, fold comparisons and fold omission summaries. The continuous constant comparison retains its separate 192 contrast simulation family and 24 contrast temporal family.

## Citation

`CITATION.cff` records the software version and authors. Cite the accompanying manuscript together with this archive.
