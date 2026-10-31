from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import split_window_data as data_utils
from expanded_family_core import (
    LEVEL_GRID, MODEL_ORDER, PROFILE_KKT_TOLERANCE, SCALAR_GRID, SLOPE_GRID, aggregate_metrics, assess_rules,
    build_layout, build_profile_grid, fit_calibration_batch, selected_rules,
)
from expanded_profile_qp import QP_SETTINGS
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import balanced_fold_ids, global_index_lookup, inner_fit_tune_indices, write_json
from run_temporal_validation import paired_bootstrap_interval, paired_interval, temporal_indices
from split_window_analysis_core import design_frame
from split_window_data import _safe_read_frame, build_dataset_from_cache


def verify_population_source(source: dict[str, object], fit: np.ndarray, tune: np.ndarray, assess: np.ndarray,
                             age_info: dict[str, float], temporal: bool, tau: float, index_hours: float) -> np.ndarray:
    metadata = source["design"] if temporal else source
    count_names = ("fit_stays", "tuning_stays", "assessment_stays") if temporal else (
        "inner_fit_stays", "inner_tuning_stays", "outer_assessment_stays")
    for name, indices in zip(count_names, (fit, tune, assess)):
        if int(metadata[name]) != indices.size:
            raise RuntimeError(f"Cached population fit has different {name}")
    for key, value in age_info.items():
        if not np.isclose(float(metadata["age_standardization"][key]), float(value), rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Cached population fit has different {key}")
    if temporal:
        if metadata["assignment_rule"] != "aligned_full_containment":
            raise RuntimeError("Temporal population fit uses a different assignment rule")
        if float(metadata["tau"]) != tau or float(metadata["index_hours"]) != index_hours:
            raise RuntimeError("Temporal population fit uses a different target or landmark")
    beta = np.asarray(source["population_coefficients"], dtype=float)
    if beta.shape != (4,) or not np.all(np.isfinite(beta)):
        raise RuntimeError("Invalid cached population coefficients")
    return beta


def evaluate_partition(dataset: dict[str, object], stays: pd.DataFrame, fit: np.ndarray, tune: np.ndarray,
                       assess: np.ndarray, source_path: Path, work_dir: Path, args: argparse.Namespace,
                       temporal: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if len(np.unique(np.concatenate([fit, tune, assess]))) != fit.size + tune.size + assess.size:
        raise RuntimeError("Fitting, tuning, and assessment partitions overlap")
    standardized, age_info = apply_training_age_standardization(dataset, stays, fit)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not temporal:
        expected_fold = int(work_dir.name.removeprefix("fold_"))
        if int(source["outer_fold"]) != expected_fold:
            raise RuntimeError("Cached population source belongs to a different fold")
        coefficient_source = json.loads((source_path.parent / "population" / "population_coefficients.json").read_text(encoding="utf-8"))
        if coefficient_source["predictors"] != ["x_intercept", "age_z", "male", "emergency_or_urgent"] or float(coefficient_source["tau"]) != args.tau:
            raise RuntimeError("Cached population fit uses different covariates or target")
        if coefficient_source["coefficients"] != source["population_coefficients"]:
            raise RuntimeError("Cached population coefficient records disagree")
    beta = verify_population_source(source, fit, tune, assess, age_info, temporal, args.tau, args.index_hours)
    prior_metadata_path = work_dir / "partition_results.json"
    if args.resume and prior_metadata_path.exists():
        prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
        if prior_metadata["population_coefficients"] != beta.tolist() or prior_metadata["age_standardization"] != age_info:
            raise RuntimeError("Cached family candidates use a different population fit or age standardization")
        for name, indices in (("fit_stays", fit), ("tuning_stays", tune), ("assessment_stays", assess)):
            if int(prior_metadata[name]) != indices.size:
                raise RuntimeError(f"Cached family candidates use different {name}")
    tune_design, _ = design_frame(data_utils.subset_cluster_data(standardized, tune))
    assess_design, _ = design_frame(data_utils.subset_cluster_data(standardized, assess))
    lookup = global_index_lookup(dataset)
    tuning_layout = build_layout(tune_design, beta, lookup, args.index_hours)
    assessment_layout = build_layout(assess_design, beta, lookup, args.index_hours)
    if not np.array_equal(np.sort(tuning_layout.global_indices), tune) or not np.array_equal(np.sort(assessment_layout.global_indices), assess):
        raise RuntimeError("Layout indices disagree with the fixed partitions")
    started = time.perf_counter()
    print(f"Computing {work_dir.name}: fit={fit.size}, tuning={tune.size}, assessment={assess.size}", flush=True)
    grid_path = work_dir / "tuning_grid.json"
    calibrations_path = work_dir / "calibration" / "calibrations.json"
    if args.resume and grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        if any(row["basis"] == "level_slope" and (row.get("solver_method") != "osqp_fenchel_dual_qp_polished" or row.get("solver_settings") != QP_SETTINGS) for row in grid):
            raise RuntimeError("Cached profile grid uses a different solver")
    else:
        grid = build_profile_grid(tuning_layout, args.tau, args.index_hours, work_dir / "tuning_profiles", args.workers)
        write_json(grid_path, grid)
    if args.resume and calibrations_path.exists():
        calibrations = json.loads(calibrations_path.read_text(encoding="utf-8"))
    else:
        calibrations = fit_calibration_batch(tuning_layout, work_dir / "tuning_profiles", work_dir / "calibration",
                                             args.tau, Path(__file__).with_name("fit_expanded_family_calibrations.R"))
    rules = selected_rules(grid, calibrations, tuning_layout, args.tau)
    records, solver_diagnostics = assess_rules(assessment_layout, rules, calibrations, args.tau, args.index_hours)
    for row in grid:
        key = str(row["key"])
        row["affine_tuning_loss"] = calibrations[key]["tuning_loss"]
        row["affine_intercept"] = calibrations[key]["intercept"]
        row["affine_slope"] = calibrations[key]["slope"]
    pd.DataFrame(grid).to_csv(work_dir / "tuning_grid.csv", index=False)
    metadata = {"fit_stays": fit.size, "tuning_stays": tune.size, "assessment_stays": assess.size,
                "age_standardization": age_info, "population_coefficients": beta.tolist(),
                "population_fit_source": str(source_path.relative_to(args.temporal_results_file.parent if temporal else args.nested_results_dir)), "selected_rules": rules,
                "calibrations": calibrations, "assessment_solver_diagnostics": solver_diagnostics,
                "runtime_seconds": time.perf_counter() - started}
    write_json(work_dir / "partition_results.json", metadata)
    records.to_csv(work_dir / "assessment_metrics.csv", index=False)
    print(f"Completed {work_dir.name}: selected={rules['selected_family']['selected_model']} seconds={metadata['runtime_seconds']:.1f}", flush=True)
    return records, metadata


def temporal_comparisons(records: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for reference in ("population", "scalar_profile", "calibrated_q10"):
        for model in MODEL_ORDER:
            if model == reference:
                continue
            candidate = records[f"loss_{model}"].to_numpy(dtype=float)
            control = records[f"loss_{reference}"].to_numpy(dtype=float)
            normal = paired_interval(candidate, control)
            bootstrap = paired_bootstrap_interval(candidate, control, args.bootstrap_replicates, args.bootstrap_seed)
            rows.append({"candidate": model, "reference": reference, **normal,
                         "bootstrap_ci95_low": bootstrap["ci95_low"], "bootstrap_ci95_high": bootstrap["ci95_high"],
                         "relative_reduction_percent": 100.0 * (1.0 - float(np.mean(candidate)) / float(np.mean(control)))})
    return pd.DataFrame(rows)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    root = repo.parent
    recovery = repo / "recovery_20260822" / "gate1_approved_run"
    parser = argparse.ArgumentParser(description="Evaluate the complete quantile profiling candidate family")
    parser.add_argument("--obs-cache", type=Path, default=recovery / "cache" / "mimic_map_observations.parquet")
    parser.add_argument("--stays-cache", type=Path, default=recovery / "cache" / "mimic_map_stays.parquet")
    parser.add_argument("--patients-file", type=Path, default=repo / "data" / "mimic-iv-3.1" / "hosp" / "patients.csv.gz")
    parser.add_argument("--nested-results-dir", type=Path, default=recovery / "validation_extensions")
    parser.add_argument("--temporal-results-file", type=Path, default=recovery / "temporal" / "temporal_validation_results.json")
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work" / "strengthening" / "expanded_family")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--index-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scope", choices=("all", "nested", "temporal"), default="all")
    args = parser.parse_args()
    if args.tau != 0.1 or args.index_hours != 12.0 or args.seed != 20260522:
        raise ValueError("This analysis requires the fixed target, landmark, and split seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"tau": args.tau, "index_hours": args.index_hours, "seed": args.seed, "outer_folds": 5,
              "age_definition": "icu_proxy", "cache_schema_version": 4, "qualification": [8, 4, 1],
              "scalar_penalty_grid": SCALAR_GRID, "level_penalty_grid": LEVEL_GRID, "slope_penalty_grid": SLOPE_GRID,
              "matrix_grid": "all diagonal matrices from the Cartesian product of positive level and slope grids",
              "affine_calibration": "a+c*q(t); equal total later-observation weight per tuning stay",
              "selection": "minimum equal-stay tuning later check loss; parameters and candidate frozen before assessment",
              "equivalent_candidates": {"scalar_lambda_zero": "raw_q10", "calibrated_scalar_lambda_zero": "calibrated_q10"},
              "family_tie_order": MODEL_ORDER[:-1], "nested_inference": "pooled risk estimate and fold-specific risk; no pooled fixed-rule confidence intervals",
              "temporal_assignment": "aligned_full_containment", "temporal_periods": ["2008 - 2013", "2014 - 2016", "2017 - 2022"],
              "public_output_policy": "aggregate results only; local assessment metrics stay in ignored .review-work"}
    config_path = args.output_dir / "analysis_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
        raise RuntimeError("Existing analysis configuration differs from the fixed configuration")
    write_json(config_path, config)
    write_json(args.output_dir / "solver_precision.json", {
        "solver": "OSQP Fenchel dual convex quadratic program with polishing and primal-dual certificates",
        "primal_dual_gap_absolute_tolerance": 1e-7,
        "primal_dual_gap_relative_tolerance": 1e-8,
        "projected_kkt_absolute_tolerance": PROFILE_KKT_TOLERANCE,
        "projected_kkt_definition": "infinity norm of a-clip(a+r-Xb,tau-1,tau)",
        "coefficient_stationarity_absolute_tolerance": 1e-10,
        "osqp_settings": QP_SETTINGS,
    })
    cache_metadata = json.loads((args.obs_cache.parent / "mimic_map_full_cache_metadata.json").read_text(encoding="utf-8"))
    if int(cache_metadata["cache_schema_version"]) != 4 or cache_metadata["age_definition"] != "icu_proxy":
        raise RuntimeError("The analysis requires schema 4 and ICU proxy age")
    legacy = json.loads((args.nested_results_dir / "submission_validation_extensions_results.json").read_text(encoding="utf-8"))
    legacy_settings = legacy["settings"]
    for name, expected in (("seed", args.seed), ("age_definition", "icu_proxy"), ("tau", args.tau),
                           ("index_hours", args.index_hours), ("outer_folds", 5)):
        if legacy_settings[name] != expected:
            raise RuntimeError(f"Cached nested population analysis has a different {name}")
    dataset_cache = args.output_dir / "local_dataset.pkl"
    cache_signature = {str(path.resolve()): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                       for path in (args.obs_cache, args.stays_cache)}
    if dataset_cache.exists():
        with dataset_cache.open("rb") as handle:
            cached = pickle.load(handle)
        if cached["source_signature"] != cache_signature:
            raise RuntimeError("Analytic source files changed after the local dataset was constructed")
        dataset, data_summary, stays = cached["dataset"], cached["data_summary"], cached["stays"]
    else:
        obs, stays = _safe_read_frame(args.obs_cache), _safe_read_frame(args.stays_cache)
        dataset, data_summary, _ = build_dataset_from_cache(obs, stays, 0, args.seed, 24.0)
        with dataset_cache.open("wb") as handle:
            pickle.dump({"source_signature": cache_signature, "dataset": dataset, "data_summary": data_summary, "stays": stays}, handle)
    if int(data_summary["fit_stays"]) != 60310:
        raise RuntimeError("The analytic cache does not reproduce the fixed 60310-stay cohort")
    output = {"config": config, "data_summary": data_summary}
    aggregate_dir = args.output_dir / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    if args.scope in ("all", "nested"):
        folds = balanced_fold_ids(60310, 5, args.seed)
        frames, metadata, fold_summary = [], [], []
        for fold in range(5):
            fit, tune = inner_fit_tune_indices(np.flatnonzero(folds != fold), fold, args.seed)
            assess = np.flatnonzero(folds == fold)
            fold_dir = args.output_dir / "nested" / f"fold_{fold + 1}"
            records, info = evaluate_partition(dataset, stays, fit, tune, assess,
                args.nested_results_dir / "work" / "nested_crossfit" / f"fold_{fold + 1}" / "fold_metadata.json", fold_dir, args)
            records["outer_fold"] = fold + 1
            frames.append(records)
            metadata.append(info)
            fold_summary.append(aggregate_metrics(records, f"nested_fold_{fold + 1}", args.tau))
        combined = pd.concat(frames, ignore_index=True).sort_values("global_stay_index")
        if combined.shape[0] != 60310 or combined["global_stay_index"].nunique() != 60310:
            raise RuntimeError("Every analytic stay must receive exactly one assessment prediction")
        combined.to_csv(args.output_dir / "nested_outer_assessment_metrics.csv", index=False)
        pd.concat([aggregate_metrics(combined, "nested_pooled", args.tau), *fold_summary]).to_csv(aggregate_dir / "nested_model_summary.csv", index=False)
        output["nested_partitions"] = metadata
    if args.scope in ("all", "temporal"):
        patients = pd.read_csv(args.patients_file, compression="infer", usecols=["subject_id", "anchor_year", "anchor_year_group"])
        fit, tune, assess, _ = temporal_indices(dataset, stays, patients, ("2008 - 2013",), ("2014 - 2016",), ("2017 - 2022",), "aligned_full_containment")
        records, info = evaluate_partition(dataset, stays, fit, tune, assess,
            args.temporal_results_file, args.output_dir / "temporal", args, temporal=True)
        records.to_csv(args.output_dir / "temporal_assessment_metrics.csv", index=False)
        aggregate_metrics(records, "temporal_2017_2022", args.tau).to_csv(aggregate_dir / "temporal_model_summary.csv", index=False)
        temporal_comparisons(records, args).to_csv(aggregate_dir / "temporal_paired_comparisons.csv", index=False)
        output["temporal_partition"] = info
    write_json(aggregate_dir / f"expanded_family_results_{args.scope}.json", output)


if __name__ == "__main__":
    main()
