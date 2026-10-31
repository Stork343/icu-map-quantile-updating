from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm

from component_clinical_data import ClinicalPartition, combined_records, load_clinical_cache, verify_nested_indices
from component_learning_selection import jsonable, predict_selection
from component_profile_learning import make_profile_batch
from continuous_constant_control import fit_continuous_constant_control, predict_continuous_constant_control
from run_adaptive_clinical import HORIZONS, make_layout, prediction_metrics
from run_component_clinical import metric_summary, result_predictions
from run_random_slope_comparator import fit_affine
from run_split_window_mixed_effects_analysis import apply_training_age_standardization


VERSION = "continuous_constant_clinical_20261031_v1"
PARTITIONS = tuple(f"fold_{number}" for number in range(1, 6)) + ("temporal",)
CONTRASTS = (
    ("continuous_direct", "continuous_constant_direct"),
    ("selected_component_direct", "selected_constant_direct"),
    ("continuous_affine", "continuous_constant_affine"),
    ("selected_component_affine", "selected_constant_affine"),
    ("continuous_constant_direct", "constant_diagonal_direct"),
    ("selected_constant_direct", "constant_diagonal_direct"),
    ("continuous_constant_affine", "constant_diagonal_affine"),
    ("selected_constant_affine", "constant_diagonal_affine"),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2) + "\n")


def prepare_partition(cached, source, name, reference_fits):
    location = source / name
    config = json.loads((location / "private_configuration.json").read_text())
    reference = reference_fits[name]
    if (config["partition"] != name or config["max_iter"] != 100
            or config["deterministic_starts"] != 2 or config["seed"] != 20261022):
        raise ValueError("The clinical source configuration differs from the fixed design")
    if (reference["seed"] != config["seed"] or reference["max_iter"] != 100
            or reference["deterministic_starts"] != 2 or reference["tau"] != .1):
        raise ValueError("The frozen history fit has a different tuning configuration")
    fit, tune, assess = [np.asarray(config[key], dtype=np.int64)
                        for key in ("fit_indices", "tuning_indices", "assessment_indices")]
    if len(np.unique(np.concatenate((fit, tune, assess)))) != len(fit) + len(tune) + len(assess):
        raise ValueError("Clinical allocations overlap")
    verify_nested_indices(name, fit, tune, assess)
    standardized, age = apply_training_age_standardization(cached["dataset"], cached["stays"], fit)
    if set(age) != set(config["age_standardization"]):
        raise ValueError("The source age transformation has different fields")
    for key, value in age.items():
        np.testing.assert_allclose(value, config["age_standardization"][key], rtol=0, atol=1e-12)
    beta = np.asarray(config["population_coefficients"], dtype=float)
    if beta.shape != (4,) or not np.isfinite(beta).all():
        raise ValueError("The retained population coefficient vector is invalid")
    tuning, assessment, originals = {}, {}, {}
    for horizon in HORIZONS:
        tuning[horizon], _ = make_layout(standardized, tune, beta, horizon)
        assessment[horizon], _ = make_layout(standardized, assess, beta, horizon)
        frame = pd.read_csv(location / f"assessment_h{int(horizon)}.csv")
        np.testing.assert_array_equal(frame.global_stay_index, assessment[horizon].global_indices)
        np.testing.assert_array_equal(frame.late_obs, assessment[horizon].later_counts)
        if frame.global_stay_index.duplicated().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("The saved clinical assessment rows are invalid")
        originals[horizon] = frame
    return ClinicalPartition(name, fit, tune, assess, beta, age, tuning, assessment, originals), config, reference


def replay_original(partition, calibrations, reference, records, horizons):
    result = predict_selection(records, horizons, reference)
    predictions = result_predictions(partition, result, "assessment")
    errors = []
    for horizon in HORIZONS:
        layout = partition.assessment_layouts[horizon]
        frame = partition.baselines[horizon]
        for model, label in (("continuous", "continuous"), ("constant", "constant_diagonal"),
                             ("selected", "selected_component")):
            base = predictions[horizon][model]
            cal = calibrations[str(int(horizon))][model]
            for suffix, forecast in (("direct", base), ("affine", cal["intercept"] + cal["slope"] * base)):
                observed = prediction_metrics(layout, forecast)
                for metric, values in observed.items():
                    stored = frame[f"{metric}_{label}_{suffix}"].to_numpy()
                    error = float(np.max(np.abs(values - stored)))
                    np.testing.assert_allclose(values, stored, rtol=0, atol=1e-8)
                    errors.append(error)
    return result, max(errors)


def penalty_summaries(partition, batch, predictions, fitted):
    counts = np.diff(batch.index_indptr)
    rows = []
    for model, prediction in predictions.items():
        for horizon in HORIZONS:
            chosen = batch.horizons == horizon
            for coordinate, component in enumerate(("level", "slope")):
                values = prediction["penalties"][chosen, coordinate]
                quantiles = np.quantile(values, [.05, .25, .5, .75, .95])
                row = {"partition": partition.name, "horizon_hours": horizon, "model": model,
                       "component": component, "n_stays": len(values),
                       **dict(zip(("q05", "q25", "median", "q75", "q95"), quantiles))}
                local_counts = counts[chosen]
                for label, mask in (("count_below13", local_counts < 13), ("count_atleast13", local_counts >= 13)):
                    if not np.any(mask):
                        raise ValueError("A fixed early observation count group is empty")
                    row[f"{label}_stays"] = int(mask.sum())
                    row[f"{label}_median"] = float(np.median(values[mask]))
                rows.append(row)
    params = []
    for coordinate, component in enumerate(("level", "slope")):
        for feature, value in zip(("horizon_4", "horizon_8", "horizon_12", "count", "time_spread", "q10_change"),
                                  fitted["best_continuous"]["params"][coordinate]):
            params.append({"partition": partition.name, "component": component, "feature": feature, "estimate": value})
    return pd.DataFrame(rows), pd.DataFrame(params)


def evaluate_partition(cached, name, args):
    started = time.perf_counter()
    work = args.output / name
    work.mkdir(parents=True, exist_ok=True)
    partition, source_config, reference = prepare_partition(cached, args.source_clinical, name, args.reference_fits)
    config = {"version": VERSION, "partition": name, "source_configuration": source_config,
              "protocol": Path(__file__).with_name("CONTINUOUS_CONSTANT_PROTOCOL.md").read_text()}
    config_path = work / "private_configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("The continuous constant clinical checkpoint design differs")
    write_json(config_path, config)
    marker = work / "complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["status"] != "complete" or previous["version"] != VERSION:
            raise ValueError("The clinical control completion marker is invalid")
        for horizon in HORIZONS:
            frame = pd.read_csv(work / f"assessment_h{int(horizon)}.csv")
            np.testing.assert_array_equal(frame.global_stay_index, partition.assessment_layouts[horizon].global_indices)
        print(f"Verified completed clinical control partition {name}", flush=True)
        return
    tune_stays, tune_horizons, tune_weights, tune_ids = combined_records(partition.tuning_layouts)
    tuning_batch = make_profile_batch(tune_stays, tune_horizons, tune_weights)
    print(f"Fitting clinical continuous constant {name}: records={tuning_batch.n_records}", flush=True)
    fit_path = work / "fitted_control.json"
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text())
        if fitted["seed"] != reference["seed"] or fitted["records"] != tuning_batch.n_records:
            raise ValueError("The clinical control fit differs from its source selection")
    else:
        fitted = fit_continuous_constant_control(tuning_batch, tune_horizons, tune_ids, reference,
                                                diagnostic_directory=work / "solver_diagnostics")
        write_json(fit_path, fitted)
    tuning = predict_continuous_constant_control(tuning_batch, fitted)
    tuning_predictions = result_predictions(partition, tuning, "tuning")
    assess_stays, assess_horizons, assess_weights, _ = combined_records(partition.assessment_layouts)
    assessment_batch = make_profile_batch(assess_stays, assess_horizons, assess_weights)
    original_calibrations = args.reference_calibrations[name]
    original, replay_error = replay_original(partition, original_calibrations, reference, assess_stays, assess_horizons)
    assessment = predict_continuous_constant_control(assessment_batch, fitted)
    assessment_predictions = result_predictions(partition, assessment, "assessment")
    calibrations = {}
    maximum_decomposition_error = 0.0
    for horizon in HORIZONS:
        layout = partition.tuning_layouts[horizon]
        direct = tuning_predictions[horizon]["continuous_constant"]
        design = pd.DataFrame({"y": layout.later_values, "prediction": direct,
                               "weight": 1.0 / layout.later_counts[layout.later_stay]})
        continuous_calibration = fit_affine(design, work / f"calibration_h{int(horizon)}", .1)
        selected_calibration = (original_calibrations[str(int(horizon))]["constant"]
                                if fitted["selected_kind"] == "grid" else continuous_calibration)
        calibrations[str(int(horizon))] = {"continuous_constant": continuous_calibration,
                                           "selected_constant": selected_calibration}
        current = partition.assessment_layouts[horizon]
        frame = partition.baselines[horizon].copy()
        for model in ("continuous_constant", "selected_constant"):
            base = assessment_predictions[horizon][model]
            cal = calibrations[str(int(horizon))][model]
            for suffix, forecast in (("direct", base), ("affine", cal["intercept"] + cal["slope"] * base)):
                for metric, values in prediction_metrics(current, forecast).items():
                    if model == "selected_constant" and fitted["selected_kind"] == "grid":
                        stored = frame[f"{metric}_constant_diagonal_{suffix}"].to_numpy()
                        np.testing.assert_allclose(values, stored, rtol=0, atol=1e-8)
                        frame[f"{metric}_{model}_{suffix}"] = stored
                    else:
                        frame[f"{metric}_{model}_{suffix}"] = values
        for suffix in ("direct", "affine"):
            for history, constant in (("continuous", "continuous_constant"), ("selected_component", "selected_constant")):
                history_grid = frame[f"loss_{history}_{suffix}"] - frame[f"loss_constant_diagonal_{suffix}"]
                history_constant = frame[f"loss_{history}_{suffix}"] - frame[f"loss_{constant}_{suffix}"]
                constant_grid = frame[f"loss_{constant}_{suffix}"] - frame[f"loss_constant_diagonal_{suffix}"]
                np.testing.assert_allclose(history_constant + constant_grid, history_grid, rtol=0, atol=1e-12)
                maximum_decomposition_error = max(maximum_decomposition_error,
                    float(np.max(np.abs(history_constant + constant_grid - history_grid))))
                frame[f"difference_{history}_minus_{constant}_{suffix}"] = history_constant
                frame[f"difference_{constant}_minus_grid_{suffix}"] = constant_grid
        if not np.isfinite(frame.to_numpy()).all() or frame.global_stay_index.duplicated().any():
            raise ValueError("The new clinical assessment output is invalid")
        frame.to_csv(work / f"assessment_h{int(horizon)}.csv", index=False)
    write_json(work / "calibration_coefficients.json", calibrations)
    penalty_predictions = {"history_continuous": original["continuous"],
                           "history_selected": original["selected"], **assessment}
    penalties, params = penalty_summaries(partition, assessment_batch, penalty_predictions, reference)
    penalties.to_csv(work / "penalty_summary.csv", index=False)
    params.to_csv(work / "history_parameters.csv", index=False)
    write_json(marker, {"status": "complete", "version": VERSION, "partition": name,
        "selected_kind": fitted["selected_kind"], "history_selected_kind": reference["selected_kind"],
        "maximum_original_metric_replay_error": replay_error,
        "maximum_loss_decomposition_error": maximum_decomposition_error,
        "history_coefficients_fixed_at_zero": bool(np.all(np.asarray(fitted["continuous_fit"]["params"])[:, 3:] == 0)),
        "tuning_diagnostics": {model: value["diagnostics"] for model, value in tuning.items()},
        "assessment_diagnostics": {model: value["diagnostics"] for model, value in assessment.items()},
        "seconds": time.perf_counter() - started})
    print(f"Completed clinical control {name}: selected={fitted['selected_kind']} seconds={time.perf_counter()-started:.1f}", flush=True)


def export_aggregate(args):
    summaries, pairs, nested_rows, omissions, penalties, parameters, fits = [], [], [], [], [], [], []
    optimizer_rows, solver_rows, validation_rows = [], [], []
    affine_rows = []
    frames = {}
    for name in PARTITIONS:
        work = args.output / name
        complete = json.loads((work / "complete.json").read_text())
        if complete["status"] != "complete" or complete["version"] != VERSION:
            raise RuntimeError("A clinical control partition is incomplete")
        fit = json.loads((work / "fitted_control.json").read_text())
        fits.append({"partition": name, "selected_kind": fit["selected_kind"],
                     "grid_validation_score": fit["grid_validation_score"],
                     "continuous_validation_score": fit["continuous_validation_score"],
                     "continuous_parameters": fit["continuous_fit"]["params"],
                     "optimizer_success": fit["continuous_fit"]["optimizer_success"],
                     "projected_gradient_norm": fit["continuous_fit"]["projected_gradient_norm"],
                     "max_iter": fit["max_iter"], "deterministic_starts": fit["deterministic_starts"]})
        optimization_fits = [(f"inner_fold_{row['fold']}", row["fit"]) for row in fit["validation_rows"]]
        optimization_fits.append(("full_tuning", fit["continuous_fit"]))
        for phase, current_fit in optimization_fits:
            optimizer_rows.append({"partition": name, "phase": phase,
                "optimizer_success": current_fit["optimizer_success"],
                "objective": current_fit["objective"], "empirical_loss": current_fit["empirical_loss"],
                "projected_gradient_norm": current_fit["projected_gradient_norm"],
                "evaluations": current_fit["evaluations"], "seconds": current_fit["seconds"],
                "history_coefficient_maximum": float(np.max(np.abs(np.asarray(current_fit["params"])[:, 3:]))),
                "start_1_iterations": current_fit["starts"][0]["iterations"],
                "start_2_iterations": current_fit["starts"][1]["iterations"]})
            solver_rows.append({"partition": name, "phase": phase,
                                **current_fit["all_evaluation_diagnostics"]})
        for row in fit["validation_rows"]:
            validation_rows.append({"partition": name, "fold": row["fold"],
                "training_records": row["training_records"], "validation_records": row["validation_records"],
                "grid_validation_loss": row["grid_validation_loss"],
                "continuous_validation_loss": row["continuous_validation_loss"],
                "grid_validation_contribution": row["grid_validation_contribution"],
                "continuous_validation_contribution": row["continuous_validation_contribution"]})
        penalties.append(pd.read_csv(work / "penalty_summary.csv"))
        parameters.append(pd.read_csv(work / "history_parameters.csv"))
        calibration = json.loads((work / "calibration_coefficients.json").read_text())
        for horizon, values in calibration.items():
            affine_rows.extend({"partition": name, "horizon_hours": int(horizon), "model": model, **coefficients}
                               for model, coefficients in values.items())
        for horizon in HORIZONS:
            frame = pd.read_csv(work / f"assessment_h{int(horizon)}.csv")
            if fit["selected_kind"] == "grid":
                for suffix in ("direct", "affine"):
                    for metric in ("loss", "p_lt", "p_le"):
                        stored = frame[f"{metric}_constant_diagonal_{suffix}"].to_numpy()
                        np.testing.assert_allclose(frame[f"{metric}_selected_constant_{suffix}"], stored, rtol=0, atol=1e-8)
                        frame[f"{metric}_selected_constant_{suffix}"] = stored
            frames[name, horizon] = frame
            scope = "temporal_2017_2022" if name == "temporal" else name
            summaries.append(metric_summary(frame, scope, horizon))
    family_size = len(CONTRASTS) * len(HORIZONS)
    critical = float(norm.ppf(1 - .05 / (2 * family_size)))
    for horizon in HORIZONS:
        nested = [frames[name, horizon] for name in PARTITIONS if name != "temporal"]
        pooled = pd.concat(nested, ignore_index=True)
        if pooled.global_stay_index.duplicated().any() or len(pooled) != {4: 59969, 8: 60241, 12: 60310}[horizon]:
            raise ValueError("The combined nested clinical cohort differs")
        summaries.append(metric_summary(pooled, "nested_pooled", horizon))
        temporal = frames["temporal", horizon]
        for first, second in CONTRASTS:
            difference = temporal[f"loss_{first}"].to_numpy() - temporal[f"loss_{second}"].to_numpy()
            mean, se = float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if se == 0 and np.any(difference != 0):
                raise RuntimeError("A nonzero temporal contrast has zero estimated variance")
            identity = se == 0
            pairs.append({"scope": "temporal_2017_2022", "horizon_hours": horizon,
                "first": first, "second": second, "n_stays": len(difference), "mean_difference": mean,
                "se": se, "ci95_low": mean-1.96*se, "ci95_high": mean+1.96*se,
                "simultaneous_ci95_low": mean-critical*se, "simultaneous_ci95_high": mean+critical*se,
                "raw_p_value": 1.0 if identity else float(2 * norm.sf(abs(mean / se))),
                "interval_type": "identical frozen rules" if identity else "paired frozen rule normal",
                "comparison_family_size": family_size, "critical_value": critical,
                "benefit_fraction": float(np.mean(difference < 0))})
            local = []
            for name in PARTITIONS:
                if name == "temporal":
                    continue
                values = frames[name, horizon][f"loss_{first}"].to_numpy() - frames[name, horizon][f"loss_{second}"].to_numpy()
                local.append(values)
                nested_rows.append({"partition": name, "horizon_hours": horizon, "first": first, "second": second,
                    "n_stays": len(values), "mean_difference": float(values.mean()),
                    "benefit_fraction": float(np.mean(values < 0)), "scope": "descriptive saved outer-fold forecasts"})
            combined = np.concatenate(local)
            nested_rows.append({"partition": "nested_pooled", "horizon_hours": horizon, "first": first, "second": second,
                "n_stays": len(combined), "mean_difference": float(combined.mean()),
                "benefit_fraction": float(np.mean(combined < 0)), "scope": "descriptive saved outer-fold forecasts"})
            for omitted, name in enumerate(PARTITIONS[:-1]):
                retained = np.concatenate([value for index, value in enumerate(local) if index != omitted])
                omissions.append({"omitted_fold": name, "horizon_hours": horizon, "first": first, "second": second,
                    "n_stays": len(retained), "mean_difference": float(retained.mean()),
                    "scope": "saved forecasts without refitting"})
    temporal = pd.DataFrame(pairs)
    if len(temporal) != family_size:
        raise ValueError("The temporal contrast family is incomplete")
    order = np.argsort(temporal.raw_p_value.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate((family_size-np.arange(family_size)) * temporal.raw_p_value.to_numpy()[order]))
    temporal["holm_p_value"] = np.nan
    temporal.loc[order, "holm_p_value"] = adjusted
    temporal["comparison_family"] = "continuous_constant_temporal"
    output = args.public_output
    output.mkdir(parents=True, exist_ok=True)
    exports = {"clinical_model_summary.csv": pd.concat(summaries, ignore_index=True),
               "clinical_temporal_paired.csv": temporal, "clinical_nested_pairs.csv": pd.DataFrame(nested_rows),
               "clinical_nested_omission.csv": pd.DataFrame(omissions),
               "clinical_penalty_summary.csv": pd.concat(penalties, ignore_index=True),
               "clinical_history_parameters.csv": pd.concat(parameters, ignore_index=True),
               "clinical_affine_coefficients.csv": pd.DataFrame(affine_rows),
               "clinical_optimizer_diagnostics.csv": pd.DataFrame(optimizer_rows),
               "clinical_solver_diagnostics.csv": pd.DataFrame(solver_rows),
               "clinical_validation_scores.csv": pd.DataFrame(validation_rows)}
    for name, frame in exports.items():
        if {"global_stay_index", "subject_id", "stay_id", "hadm_id"}.intersection(frame):
            raise RuntimeError("A public clinical output contains an individual identifier")
        frame.to_csv(output / name, index=False)
    write_json(output / "clinical_fits.json", fits)
    write_json(output / "completion_manifest.json", {"status": "complete", "version": VERSION,
        "partitions": list(PARTITIONS), "forecast_horizons": list(HORIZONS),
        "temporal_contrasts": family_size, "public_identifier_columns": 0,
        "nested_uncertainty": "descriptive fold means and saved-forecast omissions",
        "files": list(exports) + ["clinical_fits.json"]})
    print(temporal[["horizon_hours", "first", "second", "mean_difference", "simultaneous_ci95_low", "simultaneous_ci95_high", "holm_p_value"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--source-clinical", type=Path, required=True)
    parser.add_argument("--source-fits", type=Path, required=True)
    parser.add_argument("--source-affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--partitions", default=",".join(PARTITIONS))
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    partitions = args.partitions.split(",")
    if len(set(partitions)) != len(partitions) or any(name not in PARTITIONS for name in partitions):
        raise ValueError("Invalid clinical control partition selection")
    if args.export_only:
        export_aggregate(args)
        return
    args.reference_fits = json.loads(args.source_fits.read_text())
    if set(args.reference_fits) != set(PARTITIONS):
        raise ValueError("The frozen history snapshot has different clinical partitions")
    affine = pd.read_csv(args.source_affine, float_precision="round_trip")
    if len(affine) != 54 or affine.duplicated(["partition", "horizon_hours", "model"]).any():
        raise ValueError("The frozen clinical affine snapshot is incomplete")
    args.reference_calibrations = {name: {str(int(horizon)): {} for horizon in HORIZONS} for name in PARTITIONS}
    for row in affine.itertuples(index=False):
        args.reference_calibrations[row.partition][str(int(row.horizon_hours))][row.model] = {
            key: float(getattr(row, key)) for key in ("intercept", "slope", "stationarity", "relative_gap")}
    cached = load_clinical_cache(args.dataset_cache)
    for name in partitions:
        evaluate_partition(cached, name, args)


if __name__ == "__main__":
    main()
