from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import split_window_data as data_utils
from generate_split_window_hard_result_tables import X_PREDICTORS, contiguous_stay_slices, strict_split_indices
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import (
    balanced_fold_ids,
    calibration_long_frame,
    global_index_lookup,
    inner_fit_tune_indices,
    write_json,
)
from run_temporal_validation import temporal_indices
from split_window_analysis_core import check_loss, design_frame
from split_window_data import _safe_read_frame, build_dataset_from_cache


MODELS = ("published_lqmm", "calibrated_published_lqmm")


def r_environment(r_tmp: Path) -> dict:
    return dict(os.environ, TMPDIR=os.path.relpath(r_tmp.resolve(), Path.cwd()),
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")


def random_intercept_blp(
    residuals: np.ndarray, variance: float, residual_mean: float, residual_variance: float
) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Index residuals must be a nonempty finite vector")
    if variance < 0 or residual_variance <= 0:
        raise ValueError("Invalid LQMM variances")
    return float(variance * np.sum(residuals - residual_mean) / (residual_variance + residuals.size * variance))


def ensure_matching_configuration(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"Saved LQMM configuration differs from the requested analysis: {path}")
    return True


def fit_parameters(design: pd.DataFrame, work_dir: Path, tau: float, nodes: int, resume: bool,
                   partition_configuration: dict) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = work_dir / "parameters.csv"
    model_path = work_dir / "model.rds"
    context_path = work_dir / "fitting_configuration.json"
    if resume:
        ensure_matching_configuration(context_path, partition_configuration)
    design_path = work_dir / "fitting_design.csv"
    design.loc[:, ["y", "stay_index", "age_z", "male", "emergency_or_urgent"]].to_csv(design_path, index=False)
    r_tmp = work_dir / "r_tmp"
    r_tmp.mkdir(exist_ok=True)
    env = r_environment(r_tmp)
    if not resume or not parameter_path.exists():
        command = ["Rscript", str(Path(__file__).with_name("fit_published_quantile_comparator.R")),
                   str(design_path), str(tau), str(nodes), str(parameter_path), str(model_path)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (work_dir / "fit.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    validation = subprocess.run([
        "Rscript", str(Path(__file__).with_name("validate_published_quantile_fit.R")),
        str(design_path), str(tau), str(nodes), str(parameter_path), str(model_path), str(work_dir / "current_design_validation.csv"),
    ], env=env, capture_output=True, text=True)
    (work_dir / "validation.log").write_text(validation.stdout + validation.stderr, encoding="utf-8")
    validation.check_returncode()
    design_path.unlink()
    write_json(context_path, partition_configuration)
    parameters = pd.read_csv(parameter_path).set_index("term")["estimate"].to_dict()
    if not np.isfinite(list(parameters.values())).all():
        raise ValueError("Nonfinite fitted LQMM parameters")
    if int(parameters["quadrature_nodes"]) != nodes:
        raise ValueError("Cached LQMM quadrature differs from requested nodes")
    return parameters


def prediction_records(design: pd.DataFrame, parameters: dict, lookup: dict, tau: float, index_hours: float) -> pd.DataFrame:
    beta = np.asarray([parameters[name] for name in X_PREDICTORS], dtype=float)
    fitted = design.loc[:, X_PREDICTORS].to_numpy(dtype=float) @ beta
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    rows = []
    for start, end in contiguous_stay_slices(design):
        index, late = strict_split_indices(t_all[start:end], index_hours)
        if index.size == 0 or late.size == 0:
            continue
        blp = random_intercept_blp(
            y_all[start:end][index] - fitted[start:end][index],
            parameters["random_intercept_variance"], parameters["residual_mean"], parameters["residual_variance"],
        )
        stay_id = int(design["stay_id"].iat[start])
        prediction = float(fitted[start] + blp)
        rows.append({
            "global_stay_index": int(lookup[stay_id]), "index_obs": int(index.size), "late_obs": int(late.size),
            "prediction_published_lqmm": prediction, "lqmm_blp": blp,
            "later_values": np.asarray(y_all[start:end][late], dtype=float),
        })
    return pd.DataFrame(rows)


def fit_calibration(records: pd.DataFrame, work_dir: Path, tau: float, resume: bool) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = work_dir / "calibration.json"
    if resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("solver") == "br" and metadata.get("tau") == tau:
            return metadata
    frame, weights = calibration_long_frame(records, "prediction_published_lqmm")
    frame["fit_weight"] = weights
    design_path = work_dir / "calibration_design.csv"
    frame.loc[:, ["y", "cal_intercept", "cal_predictor", "fit_weight"]].to_csv(design_path, index=False)
    coefficient_path = work_dir / "coefficients.csv"
    r_tmp = work_dir / "r_tmp"
    r_tmp.mkdir(exist_ok=True)
    command = ["Rscript", str(Path(__file__).with_name("fit_published_quantile_comparator.R")),
               "--calibrate", str(design_path), str(tau), str(coefficient_path)]
    result = subprocess.run(command, env=r_environment(r_tmp),
                            capture_output=True, text=True)
    (work_dir / "fit.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    design_path.unlink()
    coefficients = pd.read_csv(coefficient_path)
    values = coefficients.set_index("term")["estimate"]
    intercept, slope = float(values["cal_intercept"]), float(values["cal_predictor"])
    losses = [float(np.mean(check_loss(np.asarray(row.later_values) - (intercept + slope * row.prediction_published_lqmm), tau)))
              for row in records.itertuples()]
    metadata = {
        "predictor_column": "prediction_published_lqmm", "tau": tau, "intercept": intercept, "slope": slope,
        "objective_weighting": "equal total weight per tuning stay", "solver": "br",
        "dual_stationarity": float(coefficients.dual_stationarity.iloc[0]),
        "relative_objective_gap": float(coefficients.relative_objective_gap.iloc[0]),
        "tuning_mean_stay_level_check_loss": float(np.mean(losses)),
        "tuning_se_stay_level_check_loss": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
    }
    write_json(metadata_path, metadata)
    return metadata


def score_records(records: pd.DataFrame, calibration: dict, tau: float) -> pd.DataFrame:
    out = records.copy()
    out["prediction_calibrated_published_lqmm"] = (
        calibration["intercept"] + calibration["slope"] * out["prediction_published_lqmm"]
    )
    for model in MODELS:
        values = [np.asarray(row.later_values) for row in out.itertuples()]
        predictions = out[f"prediction_{model}"].to_numpy(dtype=float)
        out[f"loss_{model}"] = [float(np.mean(check_loss(y - p, tau))) for y, p in zip(values, predictions)]
        out[f"p_lt_{model}"] = [float(np.mean(y < p)) for y, p in zip(values, predictions)]
        out[f"p_le_{model}"] = [float(np.mean(y <= p)) for y, p in zip(values, predictions)]
    return out


def run_partition(dataset: dict, stays: pd.DataFrame, fit_idx: np.ndarray, tune_idx: np.ndarray,
                  assess_idx: np.ndarray, output_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    start = time.monotonic()
    if min(len(fit_idx), len(tune_idx), len(assess_idx)) == 0:
        raise ValueError("Each LQMM partition must contain stays")
    assigned = np.concatenate([fit_idx, tune_idx, assess_idx])
    if len(np.unique(assigned)) != len(assigned):
        raise ValueError("LQMM fitting, tuning and assessment partitions overlap")
    standardized, age_info = apply_training_age_standardization(dataset, stays, fit_idx)
    partition_configuration = {
        "analysis": args.analysis_configuration, "age_standardization": age_info,
        "fitting_global_indices": np.asarray(fit_idx, dtype=int).tolist(),
        "tuning_global_indices": np.asarray(tune_idx, dtype=int).tolist(),
        "assessment_global_indices": np.asarray(assess_idx, dtype=int).tolist(),
    }
    fit_design, _ = design_frame(data_utils.subset_cluster_data(standardized, fit_idx))
    parameters = fit_parameters(fit_design, output_dir / "lqmm", args.tau, args.nodes, not args.no_resume,
                                partition_configuration)
    del fit_design
    lookup = global_index_lookup(dataset)
    tuning_design, _ = design_frame(data_utils.subset_cluster_data(standardized, tune_idx))
    tuning = prediction_records(tuning_design, parameters, lookup, args.tau, args.index_hours)
    del tuning_design
    r_tmp = output_dir / "r_tmp"
    r_tmp.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = os.path.relpath(r_tmp.resolve(), Path.cwd())
    calibration = fit_calibration(tuning, output_dir / "calibration", args.tau,
                                  not args.no_resume and args.configuration_previously_saved)
    assessment_design, _ = design_frame(data_utils.subset_cluster_data(standardized, assess_idx))
    assessment = prediction_records(assessment_design, parameters, lookup, args.tau, args.index_hours)
    assessment = score_records(assessment, calibration, args.tau)
    tuning = score_records(tuning, calibration, args.tau)
    tuning.drop(columns="later_values").to_csv(output_dir / "tuning_metrics.csv", index=False)
    metadata = {
        "fit_stays": len(fit_idx), "tuning_stays": len(tune_idx), "assessment_stays": len(assess_idx),
        "age_standardization": age_info, "parameters": parameters, "calibration": calibration,
        "tuning_raw_mean_loss": float(tuning.loss_published_lqmm.mean()),
        "runtime_seconds": time.monotonic() - start,
    }
    write_json(output_dir / "partition_metadata.json", metadata)
    return assessment, metadata


def aggregate(records: pd.DataFrame, scope: str) -> list[dict]:
    return [{"scope": scope, "model": model, "n_stays": len(records),
             "mean_stay_level_check_loss": float(records[f"loss_{model}"].mean()),
             "se_stay_level_check_loss": float(records[f"loss_{model}"].std(ddof=1) / np.sqrt(len(records))),
             "p_lt": float(records[f"p_lt_{model}"].mean()), "p_le": float(records[f"p_le_{model}"].mean())}
            for model in MODELS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Published Gaussian random-intercept LQMM comparison")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--obs-cache", type=Path, default=root / "recovery_20260822/gate1_approved_run/cache/mimic_map_observations.parquet")
    parser.add_argument("--stays-cache", type=Path, default=root / "recovery_20260822/gate1_approved_run/cache/mimic_map_stays.parquet")
    parser.add_argument("--patients-file", type=Path, default=root / "data/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--index-hours", type=float, default=12.0)
    parser.add_argument("--nodes", type=int, default=321)
    parser.add_argument("--pilot-fit-stays", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_metadata = json.loads((args.obs_cache.parent / "mimic_map_full_cache_metadata.json").read_text(encoding="utf-8"))
    if cache_metadata["cache_schema_version"] != 4 or cache_metadata["age_definition"] != "icu_proxy":
        raise ValueError("The comparison requires schema 4 and ICU proxy age")
    args.analysis_configuration = {
        "seed": args.seed, "tau": args.tau, "index_hours": args.index_hours, "nodes": args.nodes,
        "pilot_fit_stays": args.pilot_fit_stays, "outer_folds": 5,
        "predictors": X_PREDICTORS, "age_definition": cache_metadata["age_definition"],
        "cache_schema_version": cache_metadata["cache_schema_version"],
        "analysis_hours": 24.0, "qualification": [8, 4, 1],
        "temporal_assignment": "aligned_full_containment",
        "temporal_periods": ["2008 - 2013", "2014 - 2016", "2017 - 2022"],
        "source_files": {str(path.resolve()): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                         for path in (args.obs_cache, args.stays_cache, args.patients_file)},
    }
    configuration_path = args.output_dir / "analysis_configuration.json"
    args.configuration_previously_saved = ensure_matching_configuration(configuration_path, args.analysis_configuration)
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, _ = build_dataset_from_cache(obs, stays, fit_stays=0, seed=args.seed, analysis_hours=24.0)
    del obs
    n_stays = len(dataset["cluster_ids"])
    fold_ids = balanced_fold_ids(n_stays, 5, args.seed)
    if args.pilot_fit_stays:
        train = np.flatnonzero(fold_ids != 0)
        fit_idx, tune_idx = inner_fit_tune_indices(train, 0, args.seed)
        fit_idx = fit_idx[:args.pilot_fit_stays]
        assessment, metadata = run_partition(dataset, stays, fit_idx, tune_idx[:200], np.flatnonzero(fold_ids == 0)[:200], args.output_dir / "pilot", args)
        write_json(args.output_dir / "pilot_summary.json", {"metadata": metadata, "models": aggregate(assessment, "pilot")})
        write_json(configuration_path, args.analysis_configuration)
        return
    nested = []
    partitions = []
    summaries = []
    for fold in range(5):
        fit_idx, tune_idx = inner_fit_tune_indices(np.flatnonzero(fold_ids != fold), fold, args.seed)
        assess_idx = np.flatnonzero(fold_ids == fold)
        print(f"LQMM nested fold {fold + 1}: fit={len(fit_idx)}, tuning={len(tune_idx)}, assessment={len(assess_idx)}", flush=True)
        records, metadata = run_partition(dataset, stays, fit_idx, tune_idx, assess_idx, args.output_dir / f"fold_{fold + 1}", args)
        records["outer_fold"] = fold + 1
        records.drop(columns="later_values").to_csv(args.output_dir / f"fold_{fold + 1}/assessment_metrics.csv", index=False)
        nested.append(records.drop(columns="later_values"))
        partitions.append(metadata)
    nested_records = pd.concat(nested, ignore_index=True).sort_values("global_stay_index")
    if len(nested_records) != n_stays or nested_records.global_stay_index.nunique() != n_stays:
        raise RuntimeError("Each eligible stay must appear once in nested assessment")
    nested_records.to_csv(args.output_dir / "nested_outer_assessment_metrics.csv", index=False)
    summaries.extend(aggregate(nested_records, "nested_5fold_internal_crossfit"))
    patients = pd.read_csv(args.patients_file, usecols=["subject_id", "anchor_year", "anchor_year_group"])
    fit_idx, tune_idx, assess_idx, periods = temporal_indices(
        dataset, stays, patients, ("2008 - 2013",), ("2014 - 2016",), ("2017 - 2022",), "aligned_full_containment"
    )
    print(f"LQMM temporal: fit={len(fit_idx)}, tuning={len(tune_idx)}, assessment={len(assess_idx)}", flush=True)
    records, temporal_metadata = run_partition(dataset, stays, fit_idx, tune_idx, assess_idx, args.output_dir / "temporal", args)
    records["anchor_year_group"] = periods[records.global_stay_index.to_numpy(dtype=int)]
    records.drop(columns="later_values").to_csv(args.output_dir / "temporal_assessment_metrics.csv", index=False)
    summaries.extend(aggregate(records, "temporal_assessment"))
    pd.DataFrame(summaries).to_csv(args.output_dir / "published_comparator_model_summary.csv", index=False)
    write_json(args.output_dir / "published_comparator_results.json", {
        "method": "Gaussian random-intercept linear quantile mixed model; lqmm 1.5.8",
        "references": ["https://doi.org/10.18637/jss.v057.i13", "https://doi.org/10.1007/s11222-013-9381-9"],
        "fixed_predictors": X_PREDICTORS, "tau": args.tau, "quadrature_nodes": args.nodes,
        "new_stay_prediction": "JSS Equation 7 estimated BLP using index observations and fitting parameters; later records used for assessment only",
        "calibration": "Stay-equal affine check-loss calibration fitted on inner tuning stays",
        "data_summary": data_summary, "nested_partitions": partitions,
        "temporal_partition": temporal_metadata, "model_summary": summaries,
    })
    write_json(configuration_path, args.analysis_configuration)


if __name__ == "__main__":
    main()
