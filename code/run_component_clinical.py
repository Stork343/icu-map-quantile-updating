from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from component_clinical_data import (
    BASELINE_MODELS, combined_records, load_clinical_cache, load_partition, partition_directory,
)
from component_learning_selection import DIAGONAL_GRID, fit_selection, predict_selection
from expanded_family_core import candidate_specs, later_prediction
from run_adaptive_clinical import HORIZONS, prediction_metrics
from run_random_slope_comparator import fit_affine as fit_certified_affine
from run_submission_validation_extensions import write_json
from run_temporal_validation import paired_interval, temporal_indices


VERSION = "component_clinical_20261022_v1"
NEW_MODELS = ("continuous_direct", "continuous_affine", "selected_component_direct", "selected_component_affine",
              "constant_diagonal_direct", "constant_diagonal_affine")


def inherited_directory(root: Path, name: str) -> Path:
    return root / "temporal" if name == "temporal" else root / "nested" / name


def inherited_constant_losses(partition, root: Path, work: Path) -> np.ndarray:
    path = work / "constant_tuning_losses.npz"
    layouts = partition.tuning_layouts
    identities = np.concatenate([layouts[h].global_indices for h in HORIZONS])
    horizons = np.concatenate([np.full(layouts[h].n_stays, h) for h in HORIZONS])
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["identities"], identities)
            np.testing.assert_array_equal(saved["horizons"], horizons)
            np.testing.assert_array_equal(saved["penalty_grid"], DIAGONAL_GRID)
            return saved["losses"]
    directory = inherited_directory(root, partition.name)
    grid = pd.read_csv(directory / "tuning_grid.csv")
    vector_grid = grid.loc[grid["basis"] == "level_slope"]
    if len(vector_grid) != 48 or not (vector_grid["max_projected_kkt_violation"] <= 5e-7).all():
        raise ValueError("The retained diagonal tuning grid is not fully certified")
    if not ((vector_grid["max_duality_gap"] <= 1e-7)
            | (vector_grid["max_relative_duality_gap"] <= 1e-8)).all():
        raise ValueError("A retained diagonal tuning profile failed its objective certificate")
    specs = [spec for spec in candidate_specs() if spec["basis"] == "level_slope"]
    losses = []
    for horizon in HORIZONS:
        layout = layouts[horizon]
        positions = pd.Index(partition.tuning_indices).get_indexer(layout.global_indices)
        if np.any(positions < 0):
            raise ValueError("A scored record is absent from its original tuning population")
        local = np.empty((layout.n_stays, 48))
        for index, spec in enumerate(specs):
            coefficients = np.load(directory / "tuning_profiles" / f"{spec['key']}.npy", allow_pickle=False)
            if coefficients.shape != (len(partition.tuning_indices), 2):
                raise ValueError("Retained tuning coefficients use another cohort")
            prediction = later_prediction(layout, coefficients[positions])
            local[:, index] = prediction_metrics(layout, prediction)["loss"]
        if horizon == 12.0:
            expected = grid.set_index("key").loc[[spec["key"] for spec in specs], "raw_tuning_loss"].to_numpy()
            np.testing.assert_allclose(local.mean(axis=0), expected, rtol=0, atol=1e-8)
        losses.append(local)
    result = np.concatenate(losses)
    np.savez_compressed(path, losses=result, identities=identities, horizons=horizons, penalty_grid=DIAGONAL_GRID)
    return result


def result_predictions(partition, evaluated: dict, phase: str) -> dict:
    layouts = partition.tuning_layouts if phase == "tuning" else partition.assessment_layouts
    result, start = {}, 0
    for horizon in HORIZONS:
        layout = layouts[horizon]
        stop = start + layout.n_stays
        result[horizon] = {name: later_prediction(layout, value["coefficients"][start:stop])
                           for name, value in evaluated.items()}
        start = stop
    if any(len(value["loss"]) != start for value in evaluated.values()):
        raise ValueError("Continuous profile outputs do not match the complete horizon record order")
    return result


def certified_calibrations(layout, predictions: dict, work: Path) -> dict:
    result = {}
    for name, values in predictions.items():
        frame = pd.DataFrame({"y": layout.later_values, "prediction": values,
                              "weight": 1.0 / layout.later_counts[layout.later_stay]})
        result[name] = fit_certified_affine(frame, work / name, .1)
    return result


def metric_summary(frame: pd.DataFrame, scope: str, horizon: float) -> pd.DataFrame:
    rows = []
    models = [column.removeprefix("loss_") for column in frame if column.startswith("loss_")]
    for model in models:
        lower, upper = frame[[f"p_lt_{model}", f"p_le_{model}"]].mean()
        rows.append({"scope": scope, "horizon_hours": horizon, "model": model,
            "n_stays": len(frame), "mean_loss": frame[f"loss_{model}"].mean(),
            "p_lt": lower, "p_le": upper, "bracket_violation": max(lower - .1, .1 - upper, 0)})
    return pd.DataFrame(rows)


def evaluate_partition(cached: dict, name: str, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    partition = load_partition(cached, args.source_clinical, name)
    if name == "temporal":
        patients = pd.read_csv(args.patients_file, usecols=["subject_id", "anchor_year", "anchor_year_group"])
        expected = temporal_indices(cached["dataset"], cached["stays"], patients,
            ("2008 - 2013",), ("2014 - 2016",), ("2017 - 2022",), "aligned_full_containment")[:3]
        for actual, reference in zip((partition.fit_indices, partition.tuning_indices, partition.assessment_indices), expected):
            np.testing.assert_array_equal(actual, reference)
    work = args.output / name
    work.mkdir(parents=True, exist_ok=True)
    configuration = {"version": VERSION, "partition": name, "max_iter": args.max_iter, "deterministic_starts": args.starts,
        "fit_indices": partition.fit_indices.tolist(), "tuning_indices": partition.tuning_indices.tolist(),
        "assessment_indices": partition.assessment_indices.tolist(),
        "population_coefficients": partition.population_coefficients.tolist(),
        "age_standardization": partition.age_standardization, "seed": 20261022,
        "horizons": list(HORIZONS), "ridge_grid": [0, .001, .01, .1],
        "validation": "three stay grouped folds, equal horizon and stay weights"}
    configuration_path = work / "private_configuration.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text()) != configuration:
        raise ValueError("The completed clinical learning configuration differs")
    write_json(configuration_path, configuration)
    completion_path = work / "complete.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text())
        if existing["version"] != VERSION or existing["partition"] != name or existing["status"] != "complete":
            raise ValueError("Completed clinical learning metadata differs from its partition")
        for horizon in HORIZONS:
            expected = partition.assessment_layouts[horizon].global_indices
            frame = pd.read_csv(work / f"assessment_h{int(horizon)}.csv")
            np.testing.assert_array_equal(frame["global_stay_index"], expected)
        print(f"Verified completed clinical learning partition {name}", flush=True)
        return
    tuning_stays, tuning_horizons, _, identities = combined_records(partition.tuning_layouts)
    losses = inherited_constant_losses(partition, args.inherited_family, work)
    fit_path = work / "fitted_selection.json"
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text())
        if (fitted["max_iter"] != args.max_iter or fitted["deterministic_starts"] != args.starts
                or fitted["records"] != len(tuning_stays)):
            raise ValueError("The fitted clinical map checkpoint differs")
    else:
        fitted = fit_selection(tuning_stays, tuning_horizons, identities, 20261022,
            precomputed_constant_losses=losses, max_iter=args.max_iter,
            deterministic_starts=args.starts, verbose=True, diagnostic_directory=work / "solver_diagnostics")
        write_json(fit_path, fitted)
    tuning = predict_selection(tuning_stays, tuning_horizons, fitted)
    tuning_predictions = result_predictions(partition, tuning, "tuning")
    assessment_stays, assessment_horizons, _, _ = combined_records(partition.assessment_layouts)
    assessment = predict_selection(assessment_stays, assessment_horizons, fitted)
    assessment_predictions = result_predictions(partition, assessment, "assessment")
    calibrations, penalty_rows = {}, []
    record_start = 0
    for horizon in HORIZONS:
        layout = partition.tuning_layouts[horizon]
        fit_predictions = tuning_predictions[horizon]
        calibrations[horizon] = certified_calibrations(layout, fit_predictions, work / f"calibration_h{int(horizon)}")
        current = partition.assessment_layouts[horizon]
        frame = partition.baselines[horizon].copy()
        for source_model, label in (("continuous", "continuous"), ("selected", "selected_component"), ("constant", "constant_diagonal")):
            base = assessment_predictions[horizon][source_model]
            calibrated = calibrations[horizon][source_model]["intercept"] + calibrations[horizon][source_model]["slope"] * base
            for suffix, prediction in (("direct", base), ("affine", calibrated)):
                metrics = prediction_metrics(current, prediction)
                for metric, values in metrics.items():
                    frame[f"{metric}_{label}_{suffix}"] = values
            penalties = assessment[source_model]["penalties"][record_start:record_start + current.n_stays]
            for coordinate, axis in enumerate(("level", "slope")):
                values = penalties[:, coordinate]
                penalty_rows.append({"horizon_hours": horizon, "model": label, "component": axis,
                    "mean": float(values.mean()), "minimum": float(values.min()), "q10": float(np.quantile(values, .1)),
                    "median": float(np.median(values)), "q90": float(np.quantile(values, .9)), "maximum": float(values.max())})
        if not np.isfinite(frame.to_numpy()).all() or frame["global_stay_index"].duplicated().any():
            raise ValueError("Nonfinite or duplicated clinical learning assessment record")
        frame.to_csv(work / f"assessment_h{int(horizon)}.csv", index=False)
        record_start += current.n_stays
    write_json(work / "calibration_coefficients.json", {str(int(h)): values for h, values in calibrations.items()})
    pd.DataFrame(penalty_rows).to_csv(work / "assessment_penalty_summary.csv", index=False)
    write_json(completion_path, {"version": VERSION, "partition": name, "status": "complete",
        "selected_kind": fitted["selected_kind"], "best_continuous_eta": fitted["best_continuous_eta"],
        "tuning_stay_counts": {str(int(h)): partition.tuning_layouts[h].n_stays for h in HORIZONS},
        "assessment_stay_counts": {str(int(h)): partition.assessment_layouts[h].n_stays for h in HORIZONS},
        "tuning_diagnostics": {key: values["diagnostics"] for key, values in tuning.items()},
        "assessment_diagnostics": {key: values["diagnostics"] for key, values in assessment.items()},
        "seconds": time.perf_counter() - started})
    print(f"Completed clinical learning partition {name}: selected={fitted['selected_kind']} eta={fitted['best_continuous_eta']}", flush=True)


def export_aggregate(args: argparse.Namespace, partitions: list[str]) -> None:
    if set(partitions) != {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "temporal"}:
        return
    for name in partitions:
        if not (args.output / name / "complete.json").is_file():
            return
        completed = json.loads((args.output / name / "complete.json").read_text())
        if completed["version"] != VERSION or completed["partition"] != name or completed["status"] != "complete":
            raise ValueError("A clinical export partition has inconsistent completion metadata")
    output = args.public_output
    output.mkdir(parents=True, exist_ok=True)
    summaries, temporal_pairs, parameter_rows, cv_rows, calibration_rows, penalty_rows, solver_rows = [], [], [], [], [], [], []
    optimization_rows, public_fits = [], {}
    for horizon in HORIZONS:
        nested = []
        for name in partitions:
            work = args.output / name
            frame = pd.read_csv(work / f"assessment_h{int(horizon)}.csv")
            if name == "temporal":
                summaries.append(metric_summary(frame, "temporal_2017_2022", horizon))
                for first in NEW_MODELS:
                    for second in BASELINE_MODELS:
                        interval = paired_interval(frame[f"loss_{first}"].to_numpy(), frame[f"loss_{second}"].to_numpy())
                        temporal_pairs.append({"scope": "temporal_2017_2022", "horizon_hours": horizon,
                            "first": first, "second": second, "n_stays": len(frame),
                            "interval_type": "pointwise paired normal 95 percent", **interval})
                for first, second in (("continuous_direct", "constant_diagonal_direct"),
                                      ("selected_component_direct", "constant_diagonal_direct"),
                                      ("continuous_affine", "constant_diagonal_affine"),
                                      ("selected_component_affine", "constant_diagonal_affine")):
                    temporal_pairs.append({"scope": "temporal_2017_2022", "horizon_hours": horizon,
                        "first": first, "second": second, "interval_type": "pointwise paired normal 95 percent",
                        **paired_interval(frame[f"loss_{first}"].to_numpy(), frame[f"loss_{second}"].to_numpy())})
            else:
                nested.append(frame)
                summaries.append(metric_summary(frame, name, horizon))
        pooled = pd.concat(nested, ignore_index=True)
        if pooled["global_stay_index"].duplicated().any():
            raise ValueError("Nested learning assessment records overlap")
        expected = {4.: 59969, 8.: 60241, 12.: 60310}[horizon]
        if len(pooled) != expected:
            raise ValueError("The complete nested learning cohort differs")
        summaries.append(metric_summary(pooled, "nested_pooled", horizon))
    for name in partitions:
        work = args.output / name
        fitted = json.loads((work / "fitted_selection.json").read_text())
        completed = json.loads((work / "complete.json").read_text())
        public_fits[name] = fitted
        for row in fitted["validation_rows"]:
            if row["candidate"] == "constant":
                continue
            current = row["fit"]
            phase = f"inner_fold_{row['fold']}_eta_{row['ridge_eta']:g}"
            optimization_rows.append({"partition": name, "phase": phase,
                "ridge_eta": row["ridge_eta"], "optimizer_success": current["optimizer_success"],
                "objective": current["objective"], "empirical_loss": current["empirical_loss"],
                "gradient_norm": current["gradient_norm"], "projected_gradient_norm": current["projected_gradient_norm"],
                "evaluations": current["evaluations"], "seconds": current["seconds"]})
            solver_rows.append({"partition": name, "phase": phase, "model": "continuous",
                                **current["all_evaluation_diagnostics"]})
        current = fitted["best_continuous"]
        optimization_rows.append({"partition": name, "phase": "full_tuning", "ridge_eta": fitted["best_continuous_eta"],
            "optimizer_success": current["optimizer_success"], "objective": current["objective"],
            "empirical_loss": current["empirical_loss"], "gradient_norm": current["gradient_norm"],
            "projected_gradient_norm": current["projected_gradient_norm"],
            "evaluations": current["evaluations"], "seconds": current["seconds"]})
        solver_rows.append({"partition": name, "phase": "full_tuning", "model": "continuous",
                            **current["all_evaluation_diagnostics"]})
        for coordinate, axis in enumerate(("level", "slope")):
            for feature, value in zip(("horizon_4", "horizon_8", "horizon_12", "count", "time_spread", "q10_change"),
                                      fitted["best_continuous"]["params"][coordinate]):
                parameter_rows.append({"partition": name, "component": axis, "feature": feature,
                    "estimate": value, "best_continuous_eta": fitted["best_continuous_eta"],
                    "selected_kind": fitted["selected_kind"]})
        cv_rows.extend({"partition": name, **row} for row in fitted["validation_scores"])
        for horizon, values in json.loads((work / "calibration_coefficients.json").read_text()).items():
            calibration_rows.extend({"partition": name, "horizon_hours": int(horizon), "model": model, **coefficients}
                                    for model, coefficients in values.items())
        penalty_rows.extend(pd.read_csv(work / "assessment_penalty_summary.csv").assign(partition=name).to_dict("records"))
        for phase in ("tuning", "assessment"):
            solver_rows.extend({"partition": name, "phase": phase, "model": model, **diagnostics}
                               for model, diagnostics in completed[f"{phase}_diagnostics"].items() if model != "selected")
    frames = {"clinical_model_summary.csv": pd.concat(summaries, ignore_index=True),
              "clinical_temporal_paired.csv": pd.DataFrame(temporal_pairs),
              "clinical_function_parameters.csv": pd.DataFrame(parameter_rows),
              "clinical_cv_scores.csv": pd.DataFrame(cv_rows),
              "clinical_affine_coefficients.csv": pd.DataFrame(calibration_rows),
              "clinical_penalty_summary.csv": pd.DataFrame(penalty_rows),
              "clinical_optimization_diagnostics.csv": pd.DataFrame(optimization_rows),
              "clinical_solver_diagnostics.csv": pd.DataFrame(solver_rows)}
    for name, frame in frames.items():
        if any("stay_index" in column or column in ("subject_id", "stay_id", "hadm_id") for column in frame.columns):
            raise ValueError("An individual clinical identifier reached an aggregate export")
        frame.to_csv(output / name, index=False)
    encoded_fits = json.dumps(public_fits, allow_nan=False)
    for value in ("global_stay_index", "subject_id", "hadm_id", "fit_indices", "assessment_indices", "/Users/"):
        if value in encoded_fits:
            raise ValueError("A private clinical field reached the aggregate learned function export")
    write_json(output / "clinical_learning_fits.json", public_fits)
    write_json(output / "clinical_public_schema.json", {
        "csv_fields": {name: list(frame.columns) for name, frame in frames.items()},
        "nullable_fields": {"clinical_cv_scores.csv": ["ridge_eta"]},
        "public_json": ["clinical_learning_fits.json", "clinical_verification.json", "clinical_public_schema.json"]})
    write_json(output / "clinical_verification.json", {"status": "complete", "partitions": 6,
        "horizon_comparisons": 18, "horizons": list(HORIZONS), "learning_uses_assessment": False,
        "same_stay_inner_folds_across_horizons": True, "public_identifier_columns": 0,
        "selected_predictions_reuse_a_fitted_branch": True,
        "nested_coverage": {"4": 59969, "8": 60241, "12": 60310},
        "profile_diagnostics": solver_rows})


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, default=root / ".review-work/strengthening/expanded_family/local_dataset.pkl")
    parser.add_argument("--source-clinical", type=Path, default=root / ".review-work/adaptive/clinical")
    parser.add_argument("--inherited-family", type=Path, default=root / ".review-work/strengthening/expanded_family")
    parser.add_argument("--source-cache", type=Path, default=project / "recovery_20260822/gate1_approved_run/cache")
    parser.add_argument("--patients-file", type=Path, default=project / "data/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output", type=Path, default=root / ".review-work/component_learning/clinical")
    parser.add_argument("--public-output", type=Path, default=root / ".review-work/component_learning/public/clinical")
    parser.add_argument("--scope", choices=("all", "nested", "temporal"), default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=(1, 2, 3, 4, 5))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--starts", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if args.max_iter < 1 or any(fold not in range(1, 6) for fold in args.folds):
        raise ValueError("Invalid clinical learning computation settings")
    names = [f"fold_{number}" for number in args.folds] if args.scope != "temporal" else []
    if args.scope in ("all", "temporal"):
        names.append("temporal")
    cached = load_clinical_cache(args.dataset_cache, args.source_cache)
    for name in names:
        evaluate_partition(cached, name, args)
    export_aggregate(args, [f"fold_{number}" for number in range(1, 6)] + ["temporal"])


if __name__ == "__main__":
    main()
