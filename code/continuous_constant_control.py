from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from component_learning_selection import (
    constant_penalties, grouped_folds, horizon_weights, jsonable, weighted_subset,
)
from component_profile_learning import ProfileBatch, evaluate_component_map, fit_component_map, solve_component_batch


VERSION = "continuous_constant_control_20261031_v1"


def fit_continuous_constant_control(batch: ProfileBatch, horizons: np.ndarray,
                                    stay_ids: np.ndarray, reference_selection: dict,
                                    diagnostic_directory: Path | None = None) -> dict:
    started = time.perf_counter()
    horizons = np.asarray(horizons, dtype=float)
    identities = np.asarray(stay_ids)
    if len(horizons) != batch.n_records or identities.shape != horizons.shape:
        raise ValueError("Control identities and horizons must match the tuning batch")
    np.testing.assert_array_equal(batch.horizons, horizons)
    np.testing.assert_allclose(batch.weights, horizon_weights(horizons), rtol=0, atol=1e-14)
    if (reference_selection["records"] != batch.n_records
            or reference_selection["unique_stays"] != len(np.unique(identities))
            or reference_selection["inner_folds"] != 3):
        raise ValueError("The original selection does not describe these tuning records")
    if reference_selection["max_iter"] != 100 or reference_selection["deterministic_starts"] != 2:
        raise ValueError("The control requires the original 100 iteration and two start budget")
    assignments = grouped_folds(identities, int(reference_selection["seed"]))
    grid_rows = [row for row in reference_selection["validation_rows"] if row["candidate"] == "constant"]
    if len(grid_rows) != 3 or {row["fold"] for row in grid_rows} != {1, 2, 3}:
        raise ValueError("Original constant validation records are incomplete")
    rows = []
    for fold in range(3):
        training_indices = np.flatnonzero(assignments != fold)
        heldout_indices = np.flatnonzero(assignments == fold)
        training = weighted_subset(batch, training_indices)
        heldout = weighted_subset(batch, heldout_indices)
        original = next(row for row in grid_rows if row["fold"] == fold + 1)
        if original["training_records"] != len(training_indices) or original["validation_records"] != len(heldout_indices):
            raise RuntimeError("The original validation allocation differs")
        initial = np.asarray(original["initial_penalties"], dtype=float)
        grid_prediction = solve_component_batch(heldout, constant_penalties(heldout.horizons, initial), reference_selection["tau"])
        grid_score = float(heldout.weights @ grid_prediction["loss"])
        grid_contribution = float(batch.weights[heldout_indices] @ grid_prediction["loss"])
        np.testing.assert_allclose(grid_score, original["validation_loss"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(grid_contribution, original["validation_contribution"], rtol=0, atol=1e-8)
        fitted = fit_component_map(training, initial_penalties=initial, ridge_eta=0.0,
            max_iter=100, deterministic_starts=2, tau=reference_selection["tau"], history_dependent=False,
            diagnostic_directory=None if diagnostic_directory is None else diagnostic_directory / f"cv_fold_{fold+1}")
        if np.any(np.asarray(fitted["params"])[:, 3:] != 0):
            raise RuntimeError("The continuous constant fit used historical features")
        prediction = evaluate_component_map(heldout, np.asarray(fitted["params"]), reference_selection["tau"])
        rows.append({"fold": fold + 1, "training_records": len(training_indices),
            "validation_records": len(heldout_indices), "initial_penalties": initial,
            "grid_validation_loss": grid_score, "grid_validation_contribution": grid_contribution,
            "continuous_validation_loss": float(heldout.weights @ prediction["loss"]),
            "continuous_validation_contribution": float(batch.weights[heldout_indices] @ prediction["loss"]),
            "fit": fitted, "grid_validation_diagnostics": grid_prediction["diagnostics"],
            "continuous_validation_diagnostics": prediction["diagnostics"]})
    grid_score = float(sum(row["grid_validation_contribution"] for row in rows))
    continuous_score = float(sum(row["continuous_validation_contribution"] for row in rows))
    selected_kind = "grid" if grid_score <= continuous_score + 1e-12 else "continuous_constant"
    initial = np.asarray(reference_selection["constant_penalties"], dtype=float)
    fitted = fit_component_map(batch, initial_penalties=initial, ridge_eta=0.0,
        max_iter=100, deterministic_starts=2, tau=reference_selection["tau"], history_dependent=False,
        diagnostic_directory=None if diagnostic_directory is None else diagnostic_directory / "full_tuning")
    if np.any(np.asarray(fitted["params"])[:, 3:] != 0):
        raise RuntimeError("The final continuous constant fit used historical features")
    return jsonable({"version": VERSION, "tau": reference_selection["tau"],
        "seed": int(reference_selection["seed"]), "records": batch.n_records,
        "unique_stays": len(np.unique(identities)), "assignments": assignments,
        "selected_kind": selected_kind, "continuous_fit": fitted,
        "grid_penalties": initial, "validation_rows": rows,
        "grid_validation_score": grid_score, "continuous_validation_score": continuous_score,
        "max_iter": 100, "deterministic_starts": 2, "history_coefficients_fixed_at_zero": True,
        "seconds": time.perf_counter() - started})


def predict_continuous_constant_control(batch: ProfileBatch, fitted: dict) -> dict:
    if fitted["version"] != VERSION:
        raise ValueError("Unknown continuous constant control protocol")
    params = np.asarray(fitted["continuous_fit"]["params"], dtype=float)
    if params.shape != (2, 6) or np.any(params[:, 3:] != 0):
        raise ValueError("The constant control must have exactly zero history coefficients")
    continuous = evaluate_component_map(batch, params, fitted["tau"])
    for horizon in np.unique(batch.horizons):
        values = continuous["penalties"][batch.horizons == horizon]
        np.testing.assert_array_equal(values, np.broadcast_to(values[0], values.shape))
    continuous["diagnostics"]["constant_penalties_within_horizon"] = True
    grid = solve_component_batch(batch, constant_penalties(batch.horizons, np.asarray(fitted["grid_penalties"])), fitted["tau"])
    selected = grid if fitted["selected_kind"] == "grid" else continuous
    return {"continuous_constant": continuous, "selected_constant": selected, "grid_constant": grid}
