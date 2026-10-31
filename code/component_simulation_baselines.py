from __future__ import annotations

import numpy as np

from adaptive_profile_core import (
    apply_stump_selector, check_loss, empirical_quantile,
    fit_affine_calibration, fit_stump_selector, index_features,
)
from component_profile_learning import evaluate_component_penalties, make_profile_batch
from expanded_family_core import candidate_specs
from run_adaptive_mixed_simulation import HORIZONS, SUMMARY_NAMES, summary_prediction
from split_window_analysis_core import profiled_intercept


def combined_records(stays: list, identities: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    if len(stays) != len(identities):
        raise ValueError("Simulation stay identities differ from the records")
    return (stays * len(HORIZONS), np.repeat(HORIZONS, len(stays)),
            np.tile(np.asarray(identities, dtype=np.int64), len(HORIZONS)))


def actual_response(stays: list, horizons: np.ndarray) -> np.ndarray:
    return np.concatenate([stay.later_y[stay.later_time <= stay.index_hours + horizon]
                           for stay, horizon in zip(stays, horizons)])


def score_predictions(batch, predictions: np.ndarray, response: np.ndarray) -> dict:
    prediction = np.asarray(predictions, dtype=float)
    y = np.asarray(response, dtype=float)
    if prediction.shape != y.shape or y.shape != batch.later_residual.shape or not np.isfinite(prediction).all() or not np.isfinite(y).all():
        raise ValueError("Simulation predictions must match all later observations")
    counts = np.diff(batch.later_indptr)
    losses = check_loss(y - prediction, .1)
    return {"loss": np.add.reduceat(losses, batch.later_indptr[:-1]) / counts,
            "p_lt": np.add.reduceat((y < prediction).astype(float), batch.later_indptr[:-1]) / counts,
            "p_le": np.add.reduceat((y <= prediction).astype(float), batch.later_indptr[:-1]) / counts}


def raw_predictions(stays: list, batch, key: str, diagnostic_directory=None) -> tuple[np.ndarray, dict | None]:
    specs = {str(row["key"]): row for row in candidate_specs()}
    counts = np.diff(batch.later_indptr)
    if key == "population":
        return batch.later_population.copy(), None
    if key.startswith("summary_"):
        name = key.removeprefix("summary_")
        return np.repeat([summary_prediction(stay, name) for stay in stays], counts), None
    spec = specs[key]
    if spec["basis"] == "level_slope":
        result = evaluate_component_penalties(batch, np.asarray([spec["lambda_level"], spec["lambda_slope"]]),
                                             diagnostic_directory=diagnostic_directory)
        return result["flat_predictions"], result["diagnostics"]
    if spec["basis"] != "level":
        raise ValueError("Unknown original prediction basis")
    if float(spec["lambda_level"]) == 0:
        return np.repeat([empirical_quantile(stay.index_y) for stay in stays], counts), None
    offsets = [profiled_intercept(stay.index_residual, .1, float(spec["lambda_level"])) for stay in stays]
    return batch.later_population + np.repeat(offsets, counts), None


def horizon_flat_mask(batch, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    records = batch.horizons == horizon
    return records, np.repeat(records, np.diff(batch.later_indptr))


def fit_baselines(stays: list, horizons: np.ndarray, fitting_stays: list) -> dict:
    batch = make_profile_batch(stays, horizons)
    specs = [{**row, "family": "original"} for row in candidate_specs()]
    specs.extend({"key": f"summary_{name}", "basis": "summary", "summary": name, "family": "original"}
                 for name in SUMMARY_NAMES)
    y = batch.later_residual + batch.later_population
    raw, diagnostics = {}, []
    constant_losses = []
    for spec in specs:
        prediction, diagnostic = raw_predictions(stays, batch, str(spec["key"]))
        raw[str(spec["key"])] = prediction
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        if spec["basis"] == "level_slope":
            constant_losses.append(score_predictions(batch, prediction, y)["loss"])
    fitting_features = np.asarray([index_features(stay) for stay in fitting_stays])
    result = {}
    for horizon in HORIZONS:
        record_mask, observation_mask = horizon_flat_mask(batch, horizon)
        observation_weight = np.repeat(1.0 / np.diff(batch.later_indptr)[record_mask],
                                        np.diff(batch.later_indptr)[record_mask])
        rules, columns = [], []
        for spec in specs:
            prediction = raw[str(spec["key"])]
            if spec["basis"] != "summary":
                metrics = score_predictions(batch, prediction, y)
                rules.append({"name": str(spec["key"]), "spec": spec, "intercept": 0., "slope": 1.,
                              "calibrated": False, "tuning_loss": float(metrics["loss"][record_mask].mean())})
                columns.append(metrics["loss"][record_mask])
            calibration = fit_affine_calibration(y[observation_mask], prediction[observation_mask], observation_weight)
            metrics = score_predictions(batch, calibration["intercept"] + calibration["slope"] * prediction, y)
            rules.append({"name": str(spec["key"]) + "__affine", "spec": spec,
                "intercept": calibration["intercept"], "slope": calibration["slope"], "calibrated": True,
                "tuning_loss": float(metrics["loss"][record_mask].mean()), "calibration_certificate": calibration})
            columns.append(metrics["loss"][record_mask])
        if len(rules) != 116:
            raise RuntimeError("The original comparison must contain all 116 rules")
        selected = {"original_selected": min(rules, key=lambda row: row["tuning_loss"]),
                    "constant_direct": min([row for row in rules if row["spec"]["basis"] == "level_slope"
                                             and not row["calibrated"]], key=lambda row: row["tuning_loss"]),
                    "vector_affine": min([row for row in rules if row["spec"]["basis"] == "level_slope"
                                           and row["calibrated"]], key=lambda row: row["tuning_loss"]),
                    "calibrated_q10": next(row for row in rules if row["name"] == "scalar_0__affine"),
                    "population": next(row for row in rules if row["name"] == "population")}
        features = np.asarray([index_features(stay) for stay, keep in zip(stays, record_mask) if keep])
        stump = fit_stump_selector(fitting_features, features, np.column_stack(columns), [row["name"] for row in rules])
        result[str(int(horizon))] = {"rules": rules, "selected": selected, "stump": stump}
    return {"horizons": result, "constant_loss_matrix": np.column_stack(constant_losses),
            "numerical_certificates": diagnostics}


def predict_baselines(stays: list, horizons: np.ndarray, fitted: dict, diagnostic_directory=None) -> tuple[dict[str, dict], object]:
    batch = make_profile_batch(stays, horizons)
    outputs = {name: np.empty(len(batch.later_residual)) for name in
               ("original_selected", "constant_direct", "vector_affine", "calibrated_q10", "population", "stump_original")}
    cache, diagnostics = {}, []
    for horizon in HORIZONS:
        _, observation_mask = horizon_flat_mask(batch, horizon)
        local = fitted["horizons"][str(int(horizon))]
        by_name = {rule["name"]: rule for rule in local["rules"]}
        needs = dict(local["selected"])
        selector = local["stump"]
        needs["left"] = by_name[selector["left_rule_name"]]
        needs["right"] = by_name[selector["right_rule_name"]]
        for rule in needs.values():
            key = str(rule["spec"]["key"])
            if key not in cache:
                cache[key], diagnostic = raw_predictions(stays, batch, key, diagnostic_directory)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
        for name, rule in local["selected"].items():
            prediction = rule["intercept"] + rule["slope"] * cache[str(rule["spec"]["key"])]
            outputs[name][observation_mask] = prediction[observation_mask]
        assignments = apply_stump_selector(selector, np.asarray([index_features(stay) for stay in stays]))
        assignment_flat = np.repeat(assignments, np.diff(batch.later_indptr))
        for index in np.unique(assignments):
            rule = local["rules"][int(index)]
            key = str(rule["spec"]["key"])
            if key not in cache:
                cache[key], diagnostic = raw_predictions(stays, batch, key, diagnostic_directory)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
            mask = observation_mask & (assignment_flat == index)
            outputs["stump_original"][mask] = rule["intercept"] + rule["slope"] * cache[key][mask]
    response = actual_response(stays, horizons)
    return {name: {**score_predictions(batch, prediction, response), "flat_predictions": prediction}
            for name, prediction in outputs.items()}, batch, diagnostics
