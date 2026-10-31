# Completeness corrections: 2026-09-16

The public package now regenerates the supplied aggregate tables and figures. See `revision_20260916.json` for current validation and file hashes. Earlier completion statements below refer to prior deliveries. Author confirmation of declarations and independent acceptance of the mathematical arguments remain pending.

# Luna revision baseline

## Gate 2 corrections (supersedes earlier completion claims)

The author identified incomplete condition linkage, stale result prose, clipped Figure 3 values, and mixed temporal denominators in the previous delivery. Earlier VERIFIED statements did not establish these checks and are superseded for the affected items by this correction record.

Actual changes: Assumption 4 now requires Assumptions 2 and 3 in each outer fold, disjoint inner samples of proportional size, and the numerical approximate minimization requirements of Theorems 1 and 2. The cross fitting proof explicitly invokes these conditions. All other proof arguments are retained.

The S10 narrative now uses the frozen, unrounded temporal strata CSV: differences 0.035867872364125704, -0.023562672758329958, and -0.48617228404125373. Their weighted mean equals the aggregate calibrated-q10 minus profile loss to numerical precision. S4.3 uses losses 2.2256, 1.9575, and 1.9065; mortality profile AUC is consistently 0.565 (source value 0.5650326399067727).

Existing synthetic replicate losses give 12 negative mechanism means and 11 paired Monte Carlo intervals entirely below zero. The abstract and simulation prose now distinguish these counts. Source and calculations: `gate1_approved_run/manifests/simulation_count_audit.csv`.

The metadata-only cross-tabulation `gate1_approved_run/manifests/temporal_status_by_12h.csv` gives, among 60,310 eligible stays, 53,472 assigned, 5,681 boundary uncertain, and 1,157 out of range; missing/invalid count is zero. Among 2,647 ineligible cache stays the corresponding counts are 2,329, 248, and 70. Both denominators reconcile exactly. Main and supplement now report the eligible denominator.

Figure 3 was redrawn from all ten frozen aggregate rows, retaining lambda 0.03 and sample counts. The 1 h and 2 h reductions are -3.126784% and -1.186447%. The left axis now includes both negative points and zero; the right count axis remains independent. The plotting code regression check verifies all ten points lie inside the limits and a zero reference line exists. No model fitting, parameter selection, or simulation rerun was performed for these corrections.

Checks run: unrounded stratum/overall identity, simulation replicate mean/MCSE recount, temporal eligibility cross-tabulation, Figure 3 axis regression and rendered visual check, condition traceability audit, and compilation of both documents. Code patch: `a760375`. Independent author acceptance of the corrected theory and delivery remains pending.

Date: 2026-09-07

The execution baseline is the current working tree, with `v1.0.0` used as the repository reference named in the workplan. The reproducibility repository is `recovery_20260822/github_release/icu-map-quantile-updating`; the baseline reference was `1701ce147b7565f78343470835dc9f83eb72dd2b`. The approved rerun code is now at local `HEAD` `9ae92ca`; `v1.0.0` remains unchanged and no remote push has been made.

The formal manuscript source is `manuscript/manuscript.tex`. The complete supplementary source is `manuscript/supplementary_material.tex`, with the proof and formulation inputs in `manuscript/proofs.tex` and `manuscript/supplement_statistical_formulation.tex`. The local MIMIC-IV source and cache paths exist under `data/mimic-iv-3.1/`, `code/mimic_full_cache/`, and `recovery_20260822/cache/`; no row-level clinical data were loaded into model context during this baseline check.

The required baseline command was run from the repository:

```text
python -m unittest discover -s code/tests -v
```

Result: 41 tests passed.

## Work-package status

| Work package | Status | Baseline evidence |
|---|---|---|
| A. Admission age and cache dependencies | VERIFIED | `split_window_data.py` retains `anchor_year`, the high-age code, and the approved ICU proxy; schema 4 records the choice and rejects old stay caches without provenance. The approved run uses `icu_proxy`. |
| B. Temporal admission-year alignment | VERIFIED | `run_temporal_validation.py` provides `aligned_full_containment`, explicit boundary and range states, and an aggregate audit payload. The approved run uses the complete-containment 2008 to 2022 phase design. |
| C. Fixed-clock landmark versus count split | VERIFIED | `clock_split_indices` and `count_split_indices` are separate. Fixed-clock callers no longer fall back to half-record splits; failed qualifications return empty windows and clinical callers drop those stays. The approved 6, 12, and 18 hour queues and horizon-specific Table 7 results are recorded in the gate 1 run. Simulation callers explicitly allow their declared sparse counts. |
| D. Theorem 3 conditions and proof | VERIFIED | The complete supplementary source is available. The current four-condition main theorem, cross-fit condition chain, vector profile continuity proof, reference audit, and independent 19-page/23-page compilation were checked without unresolved theorem or cross-reference findings. |
| E. Dependent re-runs and numeric propagation | VERIFIED | Gate 1 passed on 2026-09-07. Schema 4 caches, primary, horizon-specific landmark, aligned temporal, nested validation, source, weighting, calibration, and structural branches completed in one isolated run. Aggregate before/after comparison: `recovery_20260822/gate1_approved_run/manifests/results_before_after.csv`. |
| F. Text, figures, and submission package | VERIFIED | Locked results propagated to the formal main/supplement sources, tables, figures, PDFs, source archive, and aggregate-only code Data File. Final author/package acceptance remains at gate 2. |

The gate 1 design decisions are closed and recorded below; no further decision is requested for those three definitions.

## Current handoff

Actual code changes are in the reproducibility repository and mirrored into the top-level `code/` source: age provenance and cache invalidation in `split_window_data.py` and `build_revised_mimic_map_cache.py`; aligned temporal assignment in `run_temporal_validation.py`; and strict landmark routing in `split_window_analysis_core.py` and `split_window_clinical_core.py`. The synthetic regression file is `code/tests/test_luna_revision_contracts.py`.

The repository test suite now passes 56 tests; the top-level mirrored suite passes 53 tests. The source-only checks use no MIMIC records. The local code changes are committed through `9ae92ca`; they have not been pushed and `v1.0.0` has not been moved. Before gate 1 no full empirical branch had run, and no abstract or result text had been changed for a new analysis. The gate 1 configuration is recorded in `recovery_20260822/gate1_approved_run/analysis_config.json`.

## Human gate 1: passed 2026-09-07

The author approved the following definitions before the affected MIMIC-IV branch was run.

1. Age definition: use `icu_proxy = anchor_age + year(intime) - anchor_year` at ICU admission. It controls adult qualification, the age covariate, age groups, and training-only standardization. The cache retains `anchor_age`, `anchor_year`, `age_at_icu_proxy`, `age_year_delta`, and `high_age_coded`; a coded value of 91 is not treated as a recovered true age and no high-age model is added.
2. Temporal assignment: use `aligned_full_containment` with fitting 2008--2013, tuning 2014--2016, and assessment 2017--2022. A shifted anchor-year range must be wholly contained in one phase. Boundary-uncertain, out-of-range, or missing cases leave only the strict temporal analysis; they remain in random and nested queues.
3. Landmark qualification: use strict clock splits with `t <= h` for index and `t > h` for later, including observations exactly at h in index. At 6, 12, and 18 hours require at least 8 total observations, 4 index observations, and 1 later observation. Each horizon has its own eligible queue from the same upstream base queue; existing assignments are preserved and newly eligible stays are assigned once by the fixed outcome-independent rule. Table 7 refits and retunes per horizon; Figure 3 retains its audited descriptive specification and freezes the primary 12-hour penalty. Count-based splitting remains an explicit separate mode and is never a fallback.

No performance value, loss, calibration result, sample count, or old manuscript wording was used to select these definitions. The complete run configuration is in `recovery_20260822/gate1_approved_run/analysis_config.json`.

## Gate 1 execution and result lock

The approved run configuration and command/commit/input/output record are `recovery_20260822/gate1_approved_run/analysis_config.json` and `recovery_20260822/gate1_approved_run/manifests/run_manifest.json`. The local reproducibility repository is at commit `9ae92ca`; `v1.0.0` remains unchanged and no remote push was made.

The rebuilt combined schema 4 cache contains 62,957 stays and 1,583,414 MAP observations. Its observation digest is identical to the retained schema 3 combined cache. The ICU proxy is available for every retained stay, changes the numerical age for 16,981 stays, and leaves the adult first-stay set unchanged. The final strict-clock queues contain 58,995, 60,310, and 55,230 stays at 6, 12, and 18 hours, respectively; no observation-count fallback is used.

The main fixed split refit 60,310 12-hour stays (36,186/12,062/12,062) and selected $\lambda_b=0.03$. Historical assessment losses were 2.2256 for the population component, 1.9575 for the profiled offset, and 1.8878 for calibrated q10. Nested fivefold losses were 2.2222, 1.9531, and 1.8896 on 60,310 stays.

The aligned temporal run used 20,855 fitting, 10,612 tuning, and 22,005 assessment stays after the 12-hour landmark rule. It selected $\lambda_b=0.10$ and calibrated q10. Assessment losses were 2.3220, 2.0484, and 1.8904 for the population, profiled, and calibrated q10 rules. There were 5,929 boundary-uncertain and 1,227 out-of-range stays in the schema 4 cache; they remained in the random and nested queues.

Table 7 was refit separately at each approved horizon. The profiled reductions were 7.8% (6 h; 11,778 assessment stays), 12.0% (12 h; 12,062), and 15.4% (18 h; 11,083). The descriptive Figure 3 sequence and its frozen primary penalty were regenerated separately. Invasive and noninvasive schema 4 source runs, stay-weighted fitting, random-effect structure, fixed-opportunity calibration, and clinical association artifacts were also regenerated.

The authoritative numeric comparison is the 47-row aggregate CSV above; result paths and keys are recorded for every row. No full simulation was rerun because the shared strict-clock audit showed that simulation callers explicitly retain their declared sparse counts and all 56 contract tests passed.

Formal source updates are in `manuscript/manuscript.tex`, `manuscript/supplement_empirical.tex`, and `manuscript/supplement_reproducibility.tex`, with corresponding generated tables under `manuscript/tables/` and figures under `manuscript/figures/`. Main and supplementary PDFs compile successfully to 20 and 24 pages, respectively; rendered pages were visually inspected. `submission/submission_source.zip` and `submission/reproducibility_code.zip` were rebuilt, with the latter checked to contain no stay-level prediction, identifier, or raw-cache files. No unresolved citation or cross-reference warning remains in the final logs (class and layout overfull warnings are pre-existing template behavior).

Final package hashes are recorded in `recovery_20260822/gate1_approved_run/manifests/run_manifest.json`: main PDF `ca6d5155dae52804084ce11a3ad8030f6b96dd9f3ecbe9183fa1b96d8db52a59`, supplementary PDF `c5418199e268c0f3cccee21d86419e351a5038407dafb74f0ea9c92369a630c5`, source archive `4a8e516b98594b7468f258aa6f3bffb48cc09fe33e8e3054c310097d84b8a78b`, and aggregate code archive `c0728ec4701088070fb78db4e55b1b33e885777d7b480a1a8692c05149888792`.

## Outstanding gate 2 items

The current work is not an author or journal acceptance. Gate 2 still requires independent author/theory acceptance of the preserved condition chain and complete proofs, confirmation of the final funding/ORCID/AI-use metadata, and an explicit decision on any remote GitHub push or new release. Restricted records and row-level derived data remain local; only approved aggregate artifacts are in the handoff.

## D condition map

| Result | Conditions used | Proof step checked |
|---|---|---|
| Common component consistency | A1, A2 | Cluster-level law of large numbers, compact-net bound, and separated argmin argument. Rounded outcomes require no density expansion. |
| Tuning risk consistency | A1, A3 | Plug-in convergence from fitting records, uniform tuning law over finite candidates and compact calibration spaces, followed by the oracle inequality. |
| Evaluation risk and paired comparison | A1, A4 | Conditional Lyapunov or degenerate-direction Chebyshev argument and covariance consistency for independent evaluation; cross-fit decomposition with fold-exclusion, unique fitting and tuning limits, profile continuity, and envelope control. |

The complete supplementary proof source was available at baseline. The theorem and reference audits reported no unresolved traceability findings. No new D theorem, bootstrap claim, or high-dimensional result was added.
