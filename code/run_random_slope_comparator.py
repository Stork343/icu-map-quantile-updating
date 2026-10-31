from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_split_window_hard_result_tables import X_PREDICTORS, contiguous_stay_slices, strict_split_indices
from run_published_quantile_comparator import r_environment
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import balanced_fold_ids, global_index_lookup, inner_fit_tune_indices, write_json
from run_temporal_validation import temporal_indices
from split_window_analysis_core import check_loss, design_frame
from split_window_data import subset_cluster_data


HORIZONS = (4.0, 8.0, 12.0)
FIXED_NAMES = (*X_PREDICTORS, "time_u")


def random_slope_blp(residual: np.ndarray, design: np.ndarray, covariance: np.ndarray,
                     residual_mean: float, residual_variance: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    design = np.asarray(design, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if residual.ndim != 1 or design.shape != (residual.size, 2) or covariance.shape != (2, 2):
        raise ValueError("Random-slope BLP dimensions differ")
    if not np.all(np.isfinite([residual_mean, residual_variance])):
        raise ValueError("Random-slope moments must be finite")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("Random-slope covariance must be symmetric")
    if residual.size == 0 or residual_variance <= 0 or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Invalid random-slope moments")
    if not all(np.all(np.isfinite(value)) for value in (residual, design, covariance)):
        raise ValueError("Nonfinite random-slope BLP input")
    return np.linalg.solve(residual_variance * np.eye(2) + covariance @ (design.T @ design),
                           covariance @ design.T @ (residual - residual_mean))


def observation_predictions(design: pd.DataFrame, parameters: dict, lookup: dict,
                            index_hours: float) -> pd.DataFrame:
    time_u = (design["time_hours"].to_numpy(dtype=float) - index_hours) / index_hours
    x = np.column_stack([design.loc[:, X_PREDICTORS].to_numpy(dtype=float), time_u])
    beta = np.asarray([parameters[name] for name in FIXED_NAMES])
    covariance = np.array([[parameters["variance_level"], parameters["covariance_level_slope"]],
                           [parameters["covariance_level_slope"], parameters["variance_slope"]]])
    y = design["y"].to_numpy(dtype=float)
    t = design["time_hours"].to_numpy(dtype=float)
    population = x @ beta
    frames = []
    for start, end in contiguous_stay_slices(design):
        early, later = strict_split_indices(t[start:end], index_hours)
        if early.size < 4 or later.size < 1:
            raise ValueError("The fixed clinical qualification changed")
        z = np.column_stack([np.ones(early.size), time_u[start:end][early]])
        coefficients = random_slope_blp(y[start:end][early] - population[start:end][early], z,
                                         covariance, parameters["residual_mean"], parameters["residual_variance"])
        later_z = np.column_stack([np.ones(later.size), time_u[start:end][later]])
        stay_id = int(design["stay_id"].iat[start])
        frames.append(pd.DataFrame({
            "global_stay_index": int(lookup[stay_id]),
            "time_hours": t[start:end][later], "y": y[start:end][later],
            "prediction": population[start:end][later] + later_z @ coefficients,
        }))
    return pd.concat(frames, ignore_index=True)


def horizon_records(observations: pd.DataFrame, horizon: float, index_hours: float) -> pd.DataFrame:
    frame = observations.loc[observations["time_hours"] <= index_hours + horizon].copy()
    frame["weight"] = 1.0 / frame.groupby("global_stay_index")["y"].transform("size")
    return frame


def fit_affine(frame: pd.DataFrame, work: Path, tau: float) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    design = pd.DataFrame({"y": frame["y"], "cal_intercept": 1.0,
                           "cal_predictor": frame["prediction"], "fit_weight": frame["weight"]})
    source, output = work / "calibration.csv", work / "coefficients.csv"
    design.to_csv(source, index=False)
    result = subprocess.run(["Rscript", str(Path(__file__).with_name("fit_published_quantile_comparator.R")),
                             "--calibrate", str(source), str(tau), str(output)],
                            env=r_environment(work), capture_output=True, text=True)
    (work / "calibration.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    coefficients = pd.read_csv(output)
    values = coefficients.set_index("term")["estimate"]
    source.unlink()
    return {"intercept": float(values["cal_intercept"]), "slope": float(values["cal_predictor"]),
            "stationarity": float(coefficients.dual_stationarity.iloc[0]),
            "relative_gap": float(coefficients.relative_objective_gap.iloc[0])}


def score_frame(frame: pd.DataFrame, calibration: dict, tau: float) -> pd.DataFrame:
    data = {"global_stay_index": frame["global_stay_index"].to_numpy(),
            "late_obs": np.ones(len(frame), dtype=int)}
    for name, prediction in (
        ("random_slope_lqmm", frame["prediction"].to_numpy()),
        ("calibrated_random_slope_lqmm", calibration["intercept"] + calibration["slope"] * frame["prediction"].to_numpy()),
    ):
        y = frame["y"].to_numpy()
        data[f"loss_{name}"] = check_loss(y - prediction, tau)
        data[f"p_lt_{name}"] = (y < prediction).astype(float)
        data[f"p_le_{name}"] = (y <= prediction).astype(float)
    values = pd.DataFrame(data)
    metric_columns = [name for name in values if name not in ("global_stay_index", "late_obs")]
    result = values.groupby("global_stay_index", sort=True)[metric_columns].mean()
    result["late_obs"] = values.groupby("global_stay_index")["late_obs"].sum()
    return result.reset_index()


def run_partition(dataset: dict, stays: pd.DataFrame, fit: np.ndarray, tune: np.ndarray,
                  assess: np.ndarray, directory: Path, args: argparse.Namespace) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    specification = {"fitting": fit.tolist(), "tuning": tune.tolist(), "assessment": assess.tolist(),
                     "nodes": args.nodes, "tau": args.tau, "index_hours": args.index_hours,
                     "fixed": FIXED_NAMES, "covariance": "pdSymm",
                     "quadrature_initialization": args.quadrature_stages}
    config_path = directory / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(specification)):
        raise ValueError("Random-slope checkpoint configuration changed")
    write_json(config_path, specification)
    standardized, age_info = apply_training_age_standardization(dataset, stays, fit)
    parameter_path = directory / "parameters.csv"
    r_tmp = directory / "r_tmp"
    r_tmp.mkdir(exist_ok=True)
    if not parameter_path.exists():
        fitting, _ = design_frame(subset_cluster_data(standardized, fit))
        fitting["time_u"] = (fitting["time_hours"] - args.index_hours) / args.index_hours
        design_path = directory / "fitting_design.csv"
        fitting[["y", "stay_index", "age_z", "male", "emergency_or_urgent", "time_u"]].to_csv(design_path, index=False)
        command = ["Rscript", str(Path(__file__).with_name("fit_random_slope_lqmm.R")), str(design_path),
                   str(args.tau), str(args.nodes), str(parameter_path), str(directory / "model.rds"),
                   ",".join(str(value) for value in args.quadrature_stages)]
        print(f"Fitting {directory.name}, nodes={args.nodes}^2", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, env=r_environment(r_tmp), capture_output=True, text=True)
        (directory / "fit.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        design_path.unlink()
        del fitting
        print(f"Fitted {directory.name} in {time.perf_counter()-started:.1f}s", flush=True)
    parameters = pd.read_csv(parameter_path).set_index("term")["estimate"].to_dict()
    if int(parameters["nodes_per_dimension"]) != args.nodes or len(parameters) != 16:
        raise RuntimeError("Random-slope parameter checkpoint differs")
    current_fitting, _ = design_frame(subset_cluster_data(standardized, fit))
    current_fitting["time_u"] = (current_fitting["time_hours"] - args.index_hours) / args.index_hours
    current_path = directory / "current_fitting_design.csv"
    current_fitting[["y", "stay_index", "age_z", "male", "emergency_or_urgent", "time_u"]].to_csv(current_path, index=False)
    validation_path = directory / "current_design_validation.csv"
    validation = subprocess.run(["Rscript", str(Path(__file__).with_name("validate_random_slope_fit.R")),
        str(current_path), str(args.tau), str(args.nodes), str(parameter_path),
        str(directory / "model.rds"), str(validation_path)],
        env=r_environment(r_tmp), capture_output=True, text=True)
    (directory / "current_design_validation.log").write_text(validation.stdout + validation.stderr)
    validation.check_returncode()
    fitting_validation = pd.read_csv(validation_path).iloc[0].to_dict()
    current_path.unlink()
    del current_fitting
    lookup = global_index_lookup(dataset)
    tuning, _ = design_frame(subset_cluster_data(standardized, tune))
    assessment, _ = design_frame(subset_cluster_data(standardized, assess))
    tuning_observations = observation_predictions(tuning, parameters, lookup, args.index_hours)
    assessment_observations = observation_predictions(assessment, parameters, lookup, args.index_hours)
    metadata = {"age_standardization": age_info, "parameters": parameters,
                "fitting_validation": fitting_validation, "horizons": {}}
    for horizon in HORIZONS:
        local_tuning = horizon_records(tuning_observations, horizon, args.index_hours)
        local_assessment = horizon_records(assessment_observations, horizon, args.index_hours)
        calibration = fit_affine(local_tuning, directory / f"horizon_{int(horizon)}", args.tau)
        scored = score_frame(local_assessment, calibration, args.tau)
        scored.to_csv(directory / f"assessment_horizon_{int(horizon)}.csv", index=False)
        tuning_scored = score_frame(local_tuning, calibration, args.tau)
        tuning_scored.to_csv(directory / f"tuning_horizon_{int(horizon)}.csv", index=False)
        metadata["horizons"][str(int(horizon))] = {"calibration": calibration,
            "assessment_stays": len(scored), "tuning_stays": len(tuning_scored)}
    write_json(directory / "results.json", metadata)
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work/adaptive/random_slope")
    parser.add_argument("--nodes", type=int, default=31)
    parser.add_argument("--quadrature-stages", type=int, nargs="+")
    parser.add_argument("--scope", choices=("all", "nested", "temporal"), default="all")
    parser.add_argument("--outer-fold", type=int, choices=range(1, 6))
    parser.add_argument("--dataset-cache", type=Path,
                        default=root / ".review-work/strengthening/expanded_family/local_dataset.pkl")
    parser.add_argument("--patients-path", type=Path,
                        default=root / "statistics-in-medicine-paper/data/mimic-iv-3.1/hosp/patients.csv.gz")
    args = parser.parse_args()
    args.tau, args.index_hours = 0.1, 12.0
    if args.quadrature_stages is None:
        args.quadrature_stages = [value for value in dict.fromkeys((7, 15, args.nodes)) if value <= args.nodes]
    if (not args.quadrature_stages or args.quadrature_stages[-1] != args.nodes
            or min(args.quadrature_stages) < 7 or np.any(np.diff(args.quadrature_stages) <= 0)):
        raise ValueError("Quadrature stages must increase to the requested final order")
    if args.nodes < 7:
        raise ValueError("Random-slope quadrature requires at least seven nodes per dimension")
    if args.outer_fold is not None and args.scope != "nested":
        raise ValueError("A numbered outer fold requires nested scope")
    with args.dataset_cache.open("rb") as handle:
        cached = pickle.load(handle)
    dataset, stays = cached["dataset"], cached["stays"]
    if len(dataset["cluster_ids"]) != 60310:
        raise RuntimeError("The fixed cohort changed")
    folds = balanced_fold_ids(60310, 5, 20260522)
    if args.scope in ("all", "nested"):
        for fold in range(5):
            if args.outer_fold is not None and fold + 1 != args.outer_fold:
                continue
            fit, tune = inner_fit_tune_indices(np.flatnonzero(folds != fold), fold, 20260522)
            assess = np.flatnonzero(folds == fold)
            run_partition(dataset, stays, fit, tune, assess, args.output_dir / f"fold_{fold+1}", args)
    if args.scope in ("all", "temporal"):
        patients = pd.read_csv(args.patients_path,
                               usecols=["subject_id", "anchor_year", "anchor_year_group"])
        fit, tune, assess, _ = temporal_indices(dataset, stays, patients, ("2008 - 2013",),
                                               ("2014 - 2016",), ("2017 - 2022",), "aligned_full_containment")
        run_partition(dataset, stays, fit, tune, assess, args.output_dir / "temporal", args)
    print("Random-slope LQMM partitions and horizons complete", flush=True)


if __name__ == "__main__":
    main()
