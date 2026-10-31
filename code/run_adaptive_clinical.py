from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import (
    KAPPA, POLISH_SETTINGS, QP_SETTINGS, ProfileStay, adaptive_candidate_specs, apply_stump_selector,
    check_loss, empirical_quantile, fit_profile, fit_stump_selector, index_features,
    solve_spd_profile,
)
from expanded_family_core import StayLayout, candidate_specs, later_prediction, profile_coefficients, stay_mean
from run_expanded_family_validation import verify_population_source
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import balanced_fold_ids, inner_fit_tune_indices, write_json
from run_temporal_validation import paired_interval, temporal_indices


HORIZONS = (4.0, 8.0, 12.0)
SEED = 20260522
TAU = 0.1
SUMMARY_COLUMNS = {"summary_last": "index_last", "summary_min": "index_min",
                   "summary_mean": "index_mean", "summary_q10": "index_q10",
                   "summary_below65": "index_below65_fraction"}
_WORK_STAYS: list[ProfileStay] = []
_WORK_HORIZON = 0.0


def make_layout(dataset: dict[str, object], indices: np.ndarray, beta: np.ndarray,
                horizon: float) -> tuple[StayLayout, np.ndarray]:
    global_indices, residuals, times, values, bases, populations, counts, summaries = ([] for _ in range(8))
    positions = []
    x = np.asarray(dataset["X"], dtype=float)
    for position, index in enumerate(indices):
        y = np.asarray(dataset["y_list"][int(index)], dtype=float)
        t = np.asarray(dataset["t_list"][int(index)], dtype=float)
        early = t <= 12.0
        late = (t > 12.0) & (t <= 12.0 + horizon)
        if np.sum(early) < 4 or y.size < 8:
            raise RuntimeError("The fixed cohort does not satisfy its index eligibility")
        if not np.any(late):
            continue
        common = float(x[int(index)] @ beta)
        global_indices.append(int(index))
        positions.append(position)
        residuals.append(y[early] - common)
        times.append(t[early])
        values.append(y[late])
        bases.append((t[late] - 12.0) / 12.0)
        populations.append(np.full(int(np.sum(late)), common))
        counts.append(int(np.sum(late)))
        summaries.append({"index_last": float(y[early][-1]), "index_min": float(np.min(y[early])),
                          "index_mean": float(np.mean(y[early])), "index_q10": empirical_quantile(y[early]),
                          "index_below65_fraction": float(np.mean(y[early] < 65.0))})
    count_array = np.asarray(counts, dtype=np.int64)
    if count_array.size == 0:
        raise RuntimeError("No stay has later observations for the prespecified horizon")
    return StayLayout(np.asarray(global_indices, dtype=np.int64), residuals, times,
                      np.concatenate(values), np.concatenate(bases), np.concatenate(populations),
                      np.repeat(np.arange(count_array.size), count_array),
                      np.r_[0, np.cumsum(count_array)[:-1]], count_array, pd.DataFrame(summaries)), np.asarray(positions)


def layout_stays(layout: StayLayout) -> list[ProfileStay]:
    stays = []
    for index, start in enumerate(layout.later_starts):
        end = int(start + layout.later_counts[index])
        selected = slice(int(start), end)
        stays.append(ProfileStay(layout.residual_index[index], layout.time_index[index],
                                 layout.later_values[selected], 12.0 + 12.0 * layout.later_basis[selected],
                                 layout.later_population[selected]))
    return stays


def fitting_features(dataset: dict[str, object], indices: np.ndarray, beta: np.ndarray) -> np.ndarray:
    result = []
    x = np.asarray(dataset["X"], dtype=float)
    for index in indices:
        y = np.asarray(dataset["y_list"][int(index)], dtype=float)
        times = np.asarray(dataset["t_list"][int(index)], dtype=float)
        early = times <= 12.0
        stay = ProfileStay(y[early] - float(x[int(index)] @ beta), times[early],
                           np.empty(0), np.empty(0), np.empty(0))
        result.append(index_features(stay))
    return np.asarray(result)


def initialize_worker(stays: list[ProfileStay], horizon: float) -> None:
    global _WORK_STAYS, _WORK_HORIZON
    _WORK_STAYS, _WORK_HORIZON = stays, horizon


def fit_candidate_worker(payload: tuple[dict[str, object], str]) -> dict[str, object]:
    spec, directory = payload
    work = Path(directory)
    path = work / f"{spec['key']}.npz"
    metadata_path = work / f"{spec['key']}.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["spec"] != spec or metadata["horizon"] != _WORK_HORIZON
                or metadata["n_stays"] != len(_WORK_STAYS) or metadata["solver_settings"] != QP_SETTINGS):
            raise RuntimeError("Candidate checkpoint configuration differs")
        with np.load(path, allow_pickle=False) as saved:
            if saved["coefficients"].shape != (len(_WORK_STAYS), 2):
                raise RuntimeError("Candidate checkpoint has invalid coefficient dimensions")
        return metadata
    started = time.perf_counter()
    coefficients = np.empty((len(_WORK_STAYS), 2))
    maxima = {"duality_gap": 0.0, "relative_duality_gap": 0.0,
              "projected_kkt_violation": 0.0, "stationarity_violation": 0.0}
    iterations = 0
    for index, stay in enumerate(_WORK_STAYS):
        coefficients[index], diagnostics = fit_profile(stay, _WORK_HORIZON, spec,
                                                        diagnostic_path=work / "solver_failures" / f"{spec['key']}_{index}.npz")
        for name in maxima:
            maxima[name] = max(maxima[name], float(diagnostics[name]))
        iterations += int(diagnostics["iterations"])
    temporary = work / f"{spec['key']}.partial.npz"
    np.savez(temporary, coefficients=coefficients)
    temporary.replace(path)
    metadata = {"spec": spec, "horizon": _WORK_HORIZON, "n_stays": len(_WORK_STAYS),
                "solver_settings": QP_SETTINGS, "maxima": maxima,
                "iterations": iterations, "seconds": time.perf_counter() - started}
    write_json(metadata_path, metadata)
    return metadata


def calibration_batch(layout: StayLayout, predictions: dict[str, np.ndarray], work: Path) -> dict[str, dict[str, float]]:
    work.mkdir(parents=True, exist_ok=True)
    coefficients_path = work / "coefficients.csv"
    protocol_path = work / "prediction_protocol.json"
    protocol = {"version": 2, "scalar_zero": "exact empirical inverse-CDF q10"}
    if not coefficients_path.exists() or not protocol_path.exists():
        frame = pd.DataFrame({"y": layout.later_values,
                              "weight": 1.0 / layout.later_counts[layout.later_stay], **predictions})
        input_path = work / "input.csv"
        frame.to_csv(input_path, index=False)
        environment = os.environ.copy()
        environment["TMPDIR"] = "."
        script = Path(__file__).with_name("fit_expanded_family_calibrations.R")
        subprocess.run(["Rscript", str(script.resolve()), str(input_path.resolve()), str(TAU),
                        str(coefficients_path.resolve())], env=environment, cwd=work, check=True)
        input_path.unlink()
        write_json(protocol_path, protocol)
    elif json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Calibration prediction protocol differs")
    fitted = pd.read_csv(coefficients_path)
    if set(fitted["candidate"]) != set(predictions) or fitted["candidate"].duplicated().any():
        raise RuntimeError("Calibration checkpoint does not match the prediction dictionary")
    result = {}
    for row in fitted.itertuples(index=False):
        if not np.all(np.isfinite([row.intercept, row.slope])):
            raise RuntimeError("Nonfinite affine calibration coefficients")
        result[row.candidate] = {"intercept": float(row.intercept), "slope": float(row.slope)}
    return result


def prediction_metrics(layout: StayLayout, prediction: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != layout.later_values.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError("Invalid clinical prediction")
    return {"loss": stay_mean(layout, check_loss(layout.later_values - prediction, TAU)),
            "p_lt": stay_mean(layout, (layout.later_values < prediction).astype(float)),
            "p_le": stay_mean(layout, (layout.later_values <= prediction).astype(float))}


def rule_id(key: str, calibrated: bool) -> str:
    return f"{key}__{'affine' if calibrated else 'direct'}"


def collect_rules(predictions: dict[str, np.ndarray], calibrations: dict[str, dict[str, float]],
                  layout: StayLayout) -> tuple[list[dict[str, object]], np.ndarray]:
    rules, losses = [], []
    for key, base in predictions.items():
        for calibrated in (False, True):
            if key.startswith("summary_") and (not calibrated or key == "summary_q10"):
                continue
            prediction = base
            if calibrated:
                calibration = calibrations[key]
                prediction = calibration["intercept"] + calibration["slope"] * base
            values = prediction_metrics(layout, prediction)["loss"]
            rules.append({"id": rule_id(key, calibrated), "key": key, "calibrated": calibrated,
                          "tuning_loss": float(np.mean(values))})
            losses.append(values)
    return rules, np.column_stack(losses)


def select_groups(rules: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    specs = {str(spec["key"]): spec for spec in adaptive_candidate_specs()}
    original = [rule for rule in rules if not str(rule["key"]).startswith("adaptive_")]
    selected = {"original_selected": min(original, key=lambda row: float(row["tuning_loss"])),
                "expanded_selected": min(rules, key=lambda row: float(row["tuning_loss"]))}
    for name, key in (("population", "population"), ("raw_q10", "scalar_0"), ("calibrated_q10", "scalar_0")):
        calibrated = name == "calibrated_q10"
        selected[name] = next(rule for rule in rules if rule["key"] == key and rule["calibrated"] == calibrated)
    for group in ("scalar", "vector", "geometry", "history_geometry", "adaptive"):
        for calibrated in (False, True):
            candidates = []
            for rule in rules:
                key = str(rule["key"])
                if rule["calibrated"] != calibrated:
                    continue
                if group in ("scalar", "vector"):
                    accepted = key.startswith(group + "_")
                elif key in specs:
                    spec = specs[key]
                    accepted = (group == "adaptive" or (group == "geometry" and spec["gamma"] == 0.0)
                                or (group == "history_geometry" and spec["theta"] == 0.0))
                else:
                    accepted = False
                if accepted:
                    candidates.append(rule)
            if not candidates:
                raise RuntimeError("A prespecified candidate group is empty")
            selected[f"{group}_{'affine' if calibrated else 'direct'}"] = min(candidates, key=lambda row: float(row["tuning_loss"]))
    return selected


def build_tuning_predictions(layout: StayLayout, positions: np.ndarray, inherited: Path,
                             work: Path, workers: int, horizon: float) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    predictions = {}
    inherited_rows = layout.n_stays + int(positions.max() + 1 - layout.n_stays)
    for spec in candidate_specs():
        path = inherited / "tuning_profiles" / f"{spec['key']}.npy"
        coefficients = np.load(path, allow_pickle=False)
        if coefficients.ndim != 2 or coefficients.shape[1] != 2 or coefficients.shape[0] < inherited_rows:
            raise RuntimeError("Inherited tuning coefficient dimensions differ from the fixed partition")
        predictions[str(spec["key"])] = later_prediction(layout, coefficients[positions])
    q10 = layout.summaries["index_q10"].to_numpy()[layout.later_stay]
    if not np.allclose(predictions["scalar_0"], q10, rtol=0.0, atol=1e-9):
        raise RuntimeError("Inherited candidate order does not reproduce the exact q10 boundary")
    predictions["scalar_0"] = q10
    coefficient_work = work / "tuning_profiles"
    coefficient_work.mkdir(parents=True, exist_ok=True)
    profile_stays = layout_stays(layout)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=initialize_worker, initargs=(profile_stays, horizon)) as executor:
        diagnostics = []
        for result in executor.map(fit_candidate_worker, [(spec, str(coefficient_work)) for spec in adaptive_candidate_specs()]):
            diagnostics.append(result)
            print(f"{work.name} {result['spec']['key']} certified {result['n_stays']} stays in {result['seconds']:.1f}s", flush=True)
    for spec in adaptive_candidate_specs():
        with np.load(coefficient_work / f"{spec['key']}.npz", allow_pickle=False) as saved:
            predictions[str(spec["key"])] = later_prediction(layout, saved["coefficients"])
    for key, column in SUMMARY_COLUMNS.items():
        predictions[key] = layout.summaries[column].to_numpy()[layout.later_stay]
    return predictions, diagnostics


def assess_selected(layout: StayLayout, selected: dict[str, dict[str, object]], selector: dict[str, object],
                    original_rules: list[dict[str, object]], calibrations: dict[str, dict[str, float]],
                    horizon: float, work: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    needs = {str(rule["id"]): rule for rule in selected.values()}
    needs.update({str(original_rules[int(selector[name])]["id"]): original_rules[int(selector[name])]
                  for name in ("left_rule", "right_rule")})
    specs = {str(spec["key"]): spec for spec in candidate_specs() + adaptive_candidate_specs()}
    predictions, diagnostics = {}, {}
    private_coefficients = work / "assessment_coefficients"
    private_coefficients.mkdir(exist_ok=True)
    for key in dict.fromkeys(str(rule["key"]) for rule in needs.values()):
        if key == "scalar_0":
            predictions[key] = layout.summaries["index_q10"].to_numpy()[layout.later_stay]
            continue
        if key in SUMMARY_COLUMNS:
            predictions[key] = layout.summaries[SUMMARY_COLUMNS[key]].to_numpy()[layout.later_stay]
            continue
        path = private_coefficients / f"{key}.npz"
        metadata_path = private_coefficients / f"{key}.json"
        if path.exists() and metadata_path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["global_indices"], layout.global_indices):
                    raise RuntimeError("Assessment coefficient checkpoint uses different stays")
                coefficients = saved["coefficients"]
            info = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif key.startswith("adaptive_"):
            coefficients = np.empty((layout.n_stays, 2))
            info = {name: 0.0 for name in ("duality_gap", "relative_duality_gap", "projected_kkt_violation", "stationarity_violation")}
            for index, stay in enumerate(layout_stays(layout)):
                coefficients[index], current = fit_profile(stay, horizon, specs[key])
                for name in info:
                    info[name] = max(float(info[name]), float(current[name]))
            np.savez(path, coefficients=coefficients, global_indices=layout.global_indices)
            write_json(metadata_path, info)
        elif specs[key]["basis"] == "level_slope":
            coefficients = np.empty((layout.n_stays, 2))
            info = {name: 0.0 for name in ("duality_gap", "relative_duality_gap", "projected_kkt_violation", "stationarity_violation")}
            penalties = np.diag([float(specs[key]["lambda_level"]), float(specs[key]["lambda_slope"])])
            for index, stay in enumerate(layout_stays(layout)):
                coefficients[index], current = solve_spd_profile(stay.index_residual, stay._index_design, TAU, penalties)
                for name in info:
                    info[name] = max(float(info[name]), float(current[name]))
            np.savez(path, coefficients=coefficients, global_indices=layout.global_indices)
            write_json(metadata_path, info)
        else:
            coefficients, info = profile_coefficients(layout, specs[key], TAU, 12.0)
            np.savez(path, coefficients=coefficients, global_indices=layout.global_indices)
            write_json(metadata_path, info)
        predictions[key] = later_prediction(layout, coefficients)
        diagnostics[key] = info
    metrics = {}
    for identity, rule in needs.items():
        prediction = predictions[str(rule["key"])]
        if rule["calibrated"]:
            calibration = calibrations[str(rule["key"])]
            prediction = calibration["intercept"] + calibration["slope"] * prediction
        metrics[identity] = prediction_metrics(layout, prediction)
    frame = pd.DataFrame({"global_stay_index": layout.global_indices, "late_obs": layout.later_counts})
    for model, rule in selected.items():
        for metric, values in metrics[str(rule["id"])].items():
            frame[f"{metric}_{model}"] = values
    chosen = apply_stump_selector(selector, np.asarray([index_features(stay) for stay in layout_stays(layout)]))
    for metric in ("loss", "p_lt", "p_le"):
        result = np.empty(layout.n_stays)
        for leaf in np.unique(chosen):
            identity = str(original_rules[int(leaf)]["id"])
            mask = chosen == leaf
            result[mask] = metrics[identity][metric][mask]
        frame[f"{metric}_stump_original"] = result
    return frame, diagnostics


def aggregate(frame: pd.DataFrame, scope: str, horizon: float) -> pd.DataFrame:
    models = [column.removeprefix("loss_") for column in frame if column.startswith("loss_")]
    rows = []
    for model in models:
        low, high = float(frame[f"p_lt_{model}"].mean()), float(frame[f"p_le_{model}"].mean())
        rows.append({"scope": scope, "horizon_hours": horizon, "model": model, "n_stays": frame.shape[0],
                     "mean_loss": float(frame[f"loss_{model}"].mean()), "p_lt": low, "p_le": high,
                     "bracket_violation": max(low - TAU, TAU - high, 0.0)})
    return pd.DataFrame(rows)


def evaluate_partition(dataset: dict[str, object], stays: pd.DataFrame, fit: np.ndarray, tune: np.ndarray,
                       assessment: np.ndarray, source_path: Path, inherited: Path, work: Path,
                       horizon: float, workers: int, temporal: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    work.mkdir(parents=True, exist_ok=True)
    if np.unique(np.concatenate([fit, tune, assessment])).size != fit.size + tune.size + assessment.size:
        raise RuntimeError("Clinical fitting, tuning, and assessment samples overlap")
    standardized, age_info = apply_training_age_standardization(dataset, stays, fit)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    beta = verify_population_source(source, fit, tune, assessment, age_info, temporal, TAU, 12.0)
    inherited_metadata = json.loads((inherited / "partition_results.json").read_text(encoding="utf-8"))
    same_coefficients = np.allclose(inherited_metadata["population_coefficients"], beta, rtol=1e-12, atol=1e-12)
    same_age = all(np.isclose(float(inherited_metadata["age_standardization"][key]), value, rtol=1e-12, atol=1e-12)
                   for key, value in age_info.items())
    if not same_coefficients or not same_age:
        raise RuntimeError("Inherited common-family candidates use a different population fit")
    tuning, positions = make_layout(standardized, tune, beta, horizon)
    assessment_layout, _ = make_layout(standardized, assessment, beta, horizon)
    config = {"version": 4, "horizon": horizon, "tau": TAU, "index_hours": 12.0,
              "fit_indices": fit.tolist(), "tune_indices": tune.tolist(), "assessment_indices": assessment.tolist(),
              "scored_tuning_indices": tuning.global_indices.tolist(), "scored_assessment_indices": assessment_layout.global_indices.tolist(),
              "age_standardization": age_info, "population_coefficients": beta.tolist(),
              "adaptive_grid": adaptive_candidate_specs(), "solver_settings": QP_SETTINGS,
              "solver_formulation": "clarabel_spd_primal_qp_active_set_certified", "polish_settings": POLISH_SETTINGS}
    checkpoint = work / "private_partition_config.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing partition configuration differs from the frozen design")
    write_json(checkpoint, config)
    completed = work / "partition_results.json"
    records_path = work / "assessment_metrics.csv"
    if completed.exists() and records_path.exists():
        completed_metadata = json.loads(completed.read_text(encoding="utf-8"))
        if completed_metadata.get("prediction_protocol_version") == 2:
            frame = pd.read_csv(records_path)
            if not np.array_equal(frame["global_stay_index"].to_numpy(), assessment_layout.global_indices):
                raise RuntimeError("Completed checkpoint assessment order differs")
            return frame, completed_metadata
    started = time.perf_counter()
    predictions, tuning_diagnostics = build_tuning_predictions(tuning, positions, inherited, work, workers, horizon)
    calibrations = calibration_batch(tuning, predictions, work / "calibration")
    rules, loss_matrix = collect_rules(predictions, calibrations, tuning)
    selected = select_groups(rules)
    original_positions = [index for index, rule in enumerate(rules) if not str(rule["key"]).startswith("adaptive_")]
    original_rules = [rules[index] for index in original_positions]
    tuning_features = np.asarray([index_features(stay) for stay in layout_stays(tuning)])
    selector = fit_stump_selector(fitting_features(standardized, fit, beta), tuning_features,
                                  loss_matrix[:, original_positions], [str(rule["id"]) for rule in original_rules])
    write_json(work / "frozen_rules.json", {"selected": selected, "selector": selector,
                                             "calibrations": calibrations, "all_tuning_rules": rules})
    del predictions, loss_matrix
    frame, assessment_diagnostics = assess_selected(assessment_layout, selected, selector, original_rules,
                                                   calibrations, horizon, work)
    metadata = {"prediction_protocol_version": 2, "horizon_hours": horizon, "fit_stays": int(fit.size), "source_tuning_stays": int(tune.size),
                "source_assessment_stays": int(assessment.size), "tuning_stays": tuning.n_stays,
                "assessment_stays": assessment_layout.n_stays,
                "excluded_no_horizon_tuning_record": int(tune.size - tuning.n_stays),
                "excluded_no_horizon_assessment_record": int(assessment.size - assessment_layout.n_stays),
                "population_coefficients": beta.tolist(), "age_standardization": age_info,
                "selected_rules": selected, "selector": selector, "calibrations": calibrations,
                "all_tuning_rules": rules, "tuning_solver_diagnostics": tuning_diagnostics,
                "assessment_solver_diagnostics": assessment_diagnostics,
                "seconds": time.perf_counter() - started}
    frame.to_csv(records_path, index=False)
    write_json(completed, metadata)
    print(f"Completed {work}: expanded={selected['expanded_selected']['id']} n={frame.shape[0]}", flush=True)
    return frame, metadata


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    root = repo.parent
    recovery = repo / "recovery_20260822" / "gate1_approved_run"
    parser = argparse.ArgumentParser(description="Evaluate horizon and history geometry quantile profiles")
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work" / "adaptive" / "clinical")
    parser.add_argument("--inherited-family-dir", type=Path, default=root / ".review-work" / "strengthening" / "expanded_family")
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--source-cache-dir", type=Path, default=recovery / "cache")
    parser.add_argument("--patients-file", type=Path, default=repo / "data" / "mimic-iv-3.1" / "hosp" / "patients.csv.gz")
    parser.add_argument("--nested-population-dir", type=Path, default=recovery / "validation_extensions")
    parser.add_argument("--temporal-population-file", type=Path, default=recovery / "temporal" / "temporal_validation_results.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scope", choices=("all", "nested", "temporal"), default="all")
    parser.add_argument("--horizons", type=float, nargs="+", default=HORIZONS)
    parser.add_argument("--folds", type=int, nargs="+", default=(1, 2, 3, 4, 5))
    args = parser.parse_args()
    if args.workers < 1 or any(horizon not in HORIZONS for horizon in args.horizons) or any(fold not in range(1, 6) for fold in args.folds):
        raise ValueError("Invalid worker count, horizon, or fold")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inherited_root = args.inherited_family_dir
    cache_path = args.dataset_cache if args.dataset_cache is not None else inherited_root / "local_dataset.pkl"
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    for source, signature in cached["source_signature"].items():
        stat = (args.source_cache_dir / Path(source).name).stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != signature:
            raise RuntimeError("The cached clinical observations changed")
    dataset, stays = cached["dataset"], cached["stays"]
    if len(dataset["cluster_ids"]) != 60310 or int(cached["data_summary"]["fit_stays"]) != 60310:
        raise RuntimeError("The clinical source cohort must contain exactly 60310 stays")
    configuration = {"version": 4, "tau": TAU, "index_hours": 12.0, "source_cohort": 60310,
                     "horizons": HORIZONS, "seed": SEED, "kappa": KAPPA, "adaptive_grid": adaptive_candidate_specs(),
                     "z": "clip((q10(last observation-rank half residuals)-q10(first half residuals))/10,-1,1)",
                     "halves": "first floor(m/2), remaining observations in chronological order",
                     "loss": "ordinary check loss, later observation mean within stay, equal weight across stays",
                     "calibration": "ordinary unconstrained affine quantile regression on horizon-specific tuning outcomes",
                     "forecast_clipping": False, "horizon_eligibility": "fixed source cohort and partitions, at least one record in (12,12+D]",
                     "common_family": candidate_specs(), "stump": "training quartiles of z,log(m),variance((t-12)/12); tuning leaf rules from original family",
                     "solver_settings": QP_SETTINGS, "cache_source_signature": cached["source_signature"],
                     "solver_formulation": "clarabel_spd_primal_qp_active_set_certified", "polish_settings": POLISH_SETTINGS}
    config_path = args.output_dir / "analysis_config.json"
    canonical = json.loads(json.dumps(configuration))
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != canonical:
        raise RuntimeError("The existing adaptive clinical configuration differs")
    write_json(config_path, canonical)
    aggregate_dir = args.output_dir / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    folds = balanced_fold_ids(60310, 5, SEED)
    temporal = None
    if args.scope in ("all", "temporal"):
        patients = pd.read_csv(args.patients_file,
                               usecols=["subject_id", "anchor_year", "anchor_year_group"])
        temporal = temporal_indices(dataset, stays, patients, ("2008 - 2013",), ("2014 - 2016",),
                                    ("2017 - 2022",), "aligned_full_containment")[:3]
    for horizon in args.horizons:
        horizon_dir = args.output_dir / f"horizon_{int(horizon)}"
        summaries, metadata, frames = [], [], []
        if args.scope in ("all", "nested"):
            for number in args.folds:
                fold = number - 1
                fit, tune = inner_fit_tune_indices(np.flatnonzero(folds != fold), fold, SEED)
                assessment = np.flatnonzero(folds == fold)
                source = args.nested_population_dir / "work" / "nested_crossfit" / f"fold_{number}" / "fold_metadata.json"
                frame, info = evaluate_partition(dataset, stays, fit, tune, assessment, source,
                                                   inherited_root / "nested" / f"fold_{number}",
                                                   horizon_dir / "nested" / f"fold_{number}", horizon, args.workers, False)
                frame["outer_fold"] = number
                frames.append(frame)
                metadata.append(info)
                summaries.append(aggregate(frame, f"nested_fold_{number}", horizon))
            combined = pd.concat(frames, ignore_index=True)
            if combined["global_stay_index"].duplicated().any():
                raise RuntimeError("An eligible stay received multiple outer assessment predictions")
            if len(args.folds) == 5:
                expected = sum(np.any((np.asarray(times) > 12.0) & (np.asarray(times) <= 12.0 + horizon))
                               for times in dataset["t_list"])
                if combined.shape[0] != expected:
                    raise RuntimeError("Nested assessment does not cover every horizon-qualified source stay")
            combined.to_csv(horizon_dir / "nested_assessment_metrics.csv", index=False)
            scope = "nested_pooled" if len(args.folds) == 5 else "nested_partial"
            summaries.insert(0, aggregate(combined, scope, horizon))
            pd.concat(summaries, ignore_index=True).to_csv(aggregate_dir / f"nested_horizon_{int(horizon)}.csv", index=False)
            write_json(aggregate_dir / f"nested_horizon_{int(horizon)}.json", metadata)
        if temporal is not None:
            fit, tune, assessment = temporal
            frame, info = evaluate_partition(dataset, stays, fit, tune, assessment,
                                               args.temporal_population_file,
                                               inherited_root / "temporal", horizon_dir / "temporal", horizon, args.workers, True)
            aggregate(frame, "temporal_2017_2022", horizon).to_csv(aggregate_dir / f"temporal_horizon_{int(horizon)}.csv", index=False)
            comparisons = []
            for reference in ("original_selected", "calibrated_q10", "stump_original"):
                for model in ("expanded_selected", "adaptive_direct", "adaptive_affine", "geometry_direct", "history_geometry_direct"):
                    comparisons.append({"horizon_hours": horizon, "candidate": model, "reference": reference,
                                        **paired_interval(frame[f"loss_{model}"].to_numpy(), frame[f"loss_{reference}"].to_numpy())})
            pd.DataFrame(comparisons).to_csv(aggregate_dir / f"temporal_paired_horizon_{int(horizon)}.csv", index=False)
            write_json(aggregate_dir / f"temporal_horizon_{int(horizon)}.json", info)


if __name__ == "__main__":
    main()
