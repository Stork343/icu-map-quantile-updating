from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from component_profile_learning import (
    ProfileBatch, evaluate_component_map, fit_component_map,
    make_profile_batch, solve_component_batch, subset_profile_batch,
)
from expanded_family_core import candidate_specs


HORIZONS = (4.0, 8.0, 12.0)
RIDGE_GRID = (0.0, 0.001, 0.01, 0.1)
DIAGONAL_GRID = np.asarray([[row["lambda_level"], row["lambda_slope"]]
                            for row in candidate_specs() if row["basis"] == "level_slope"], dtype=float)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def horizon_weights(horizons: np.ndarray, require_all: bool = True) -> np.ndarray:
    horizons = np.asarray(horizons, dtype=float)
    unique = np.unique(horizons)
    if not np.isfinite(horizons).all() or not set(unique).issubset(HORIZONS) or not len(horizons):
        raise ValueError("Learning records require finite prespecified forecast horizons")
    if require_all and set(unique) != set(HORIZONS):
        raise ValueError("Joint component learning requires all three forecast horizons")
    weights = np.empty(len(horizons), dtype=float)
    for horizon in unique:
        mask = horizons == horizon
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return weights


def grouped_folds(stay_ids: np.ndarray, seed: int, folds: int = 3) -> np.ndarray:
    identities = np.asarray(stay_ids)
    unique = np.unique(identities)
    if identities.ndim != 1 or len(unique) < folds or folds != 3:
        raise ValueError("The learning protocol requires three nonempty stay folds")
    order = np.random.default_rng(seed).permutation(len(unique))
    assignment = np.empty(len(unique), dtype=np.int64)
    assignment[order] = np.arange(len(unique)) % folds
    return assignment[np.searchsorted(unique, identities)]


def weighted_subset(batch: ProfileBatch, selected: np.ndarray) -> ProfileBatch:
    subset = subset_profile_batch(batch, selected)
    return ProfileBatch(subset.index_residual, subset.index_time, subset.index_indptr,
        subset.later_residual, subset.later_time, subset.later_population, subset.later_indptr,
        subset.features, subset.horizons, horizon_weights(subset.horizons))


def constant_loss_matrix(batch: ProfileBatch, tau: float = .1) -> tuple[np.ndarray, dict]:
    losses = np.empty((batch.n_records, len(DIAGONAL_GRID)))
    diagnostics = []
    dual = None
    for index, penalties in enumerate(DIAGONAL_GRID):
        values = np.broadcast_to(penalties, (batch.n_records, 2)).copy()
        result = solve_component_batch(batch, values, tau, dual)
        dual = result["dual"]
        losses[:, index] = result["loss"]
        diagnostics.append(result["diagnostics"])
    return losses, {"grid_matrices": len(DIAGONAL_GRID),
        "profiles": sum(row["profiles"] for row in diagnostics),
        "max_duality_gap": max(row["max_duality_gap"] for row in diagnostics),
        "max_projected_kkt_violation": max(row["max_projected_kkt_violation"] for row in diagnostics),
        "max_stationarity_violation": max(row["max_stationarity_violation"] for row in diagnostics),
        "all_certified": all(row["all_certified"] for row in diagnostics)}


def choose_constant(losses: np.ndarray, horizons: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    losses = np.asarray(losses, dtype=float)
    if losses.shape != (len(horizons), 48) or not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("The constant branch requires finite stay losses for all 48 diagonal matrices")
    indices, penalties = [], []
    for horizon in HORIZONS:
        rows = np.asarray(selected)[horizons[selected] == horizon]
        if not len(rows):
            raise ValueError("Constant branch training misses a forecast horizon")
        best = int(np.argmin(losses[rows].mean(axis=0)))
        indices.append(best)
        penalties.append(DIAGONAL_GRID[best])
    return np.asarray(indices, dtype=np.int64), np.asarray(penalties)


def constant_penalties(horizons: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    penalties = np.asarray(penalties, dtype=float)
    if penalties.shape != (3, 2) or not np.isfinite(penalties).all() or np.any(penalties <= 0):
        raise ValueError("The constant branch requires one positive penalty pair at each horizon")
    result = np.empty((len(horizons), 2))
    for index, horizon in enumerate(HORIZONS):
        result[horizons == horizon] = penalties[index]
    return result


def fit_selection(stays: list, horizons: np.ndarray, stay_ids: np.ndarray, seed: int,
                  precomputed_constant_losses: np.ndarray | None = None,
                  max_iter: int = 100, deterministic_starts: int = 2, tau: float = .1,
                  verbose: bool = False, diagnostic_directory: Path | None = None) -> dict:
    started = time.perf_counter()
    horizons = np.asarray(horizons, dtype=float)
    identities = np.asarray(stay_ids)
    if len(stays) != len(horizons) or len(stays) != len(identities):
        raise ValueError("Stay identities and forecast records must match")
    weights = horizon_weights(horizons)
    batch = make_profile_batch(stays, horizons, weights)
    assignments = grouped_folds(identities, seed)
    if precomputed_constant_losses is None:
        losses, grid_diagnostics = constant_loss_matrix(batch, tau)
        grid_source = "certified component batch profiles"
    else:
        losses = np.asarray(precomputed_constant_losses, dtype=float)
        if losses.shape != (len(stays), 48) or not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError("The supplied constant profile losses differ from the tuning records")
        grid_diagnostics = None
        grid_source = "retained certified constant profiles"
    rows = []
    for fold in range(3):
        train = np.flatnonzero(assignments != fold)
        validation = np.flatnonzero(assignments == fold)
        training = weighted_subset(batch, train)
        heldout = weighted_subset(batch, validation)
        selected_indices, initial = choose_constant(losses, horizons, train)
        constant_predictions = np.empty(len(validation), dtype=float)
        for index, horizon in enumerate(HORIZONS):
            mask = horizons[validation] == horizon
            constant_predictions[mask] = losses[validation[mask], selected_indices[index]]
        constant_validation = float(heldout.weights @ constant_predictions)
        rows.append({"fold": fold + 1, "candidate": "constant", "ridge_eta": None,
            "validation_loss": constant_validation, "training_records": len(train),
            "validation_records": len(validation), "initial_penalties": initial.tolist(),
            "validation_contribution": float(batch.weights[validation] @ constant_predictions)})
        for eta in RIDGE_GRID:
            fitted = fit_component_map(training, initial_penalties=initial, ridge_eta=eta,
                max_iter=max_iter, deterministic_starts=deterministic_starts, tau=tau,
                diagnostic_directory=None if diagnostic_directory is None else
                diagnostic_directory / f"cv_fold_{fold+1}_eta_{eta:g}")
            evaluated = evaluate_component_map(heldout, fitted["params"], tau)
            validation_loss = float(heldout.weights @ evaluated["loss"])
            rows.append({"fold": fold + 1, "candidate": f"continuous_{eta:g}", "ridge_eta": eta,
                "validation_loss": validation_loss, "training_records": len(train),
                "validation_records": len(validation), "initial_penalties": initial.tolist(),
                "validation_contribution": float(batch.weights[validation] @ evaluated["loss"]),
                "fit": jsonable(fitted), "validation_diagnostics": evaluated["diagnostics"]})
            if verbose:
                print(f"Component learning inner fold {fold+1}/3 eta={eta:g} validation loss={validation_loss:.10f}", flush=True)
    scores = []
    for candidate in ["constant"] + [f"continuous_{eta:g}" for eta in RIDGE_GRID]:
        selected = [row for row in rows if row["candidate"] == candidate]
        if len(selected) != 3:
            raise RuntimeError("An inner validation candidate is incomplete")
        scores.append({"candidate": candidate, "validation_loss": float(np.sum([row["validation_contribution"] for row in selected])),
                       "ridge_eta": selected[0]["ridge_eta"]})
    continuous = min([row for row in scores if row["ridge_eta"] is not None], key=lambda row: row["validation_loss"])
    constant_score = scores[0]["validation_loss"]
    selected_kind = "constant" if constant_score <= continuous["validation_loss"] + 1e-12 else "continuous"
    selected_indices, initial = choose_constant(losses, horizons, np.arange(len(stays)))
    fitted = fit_component_map(batch, initial_penalties=initial, ridge_eta=float(continuous["ridge_eta"]),
        max_iter=max_iter, deterministic_starts=deterministic_starts, tau=tau,
        diagnostic_directory=None if diagnostic_directory is None else diagnostic_directory / "full_tuning")
    verified_constant = solve_component_batch(batch, constant_penalties(horizons, initial), tau)
    expected = np.empty(len(stays))
    for index, horizon in enumerate(HORIZONS):
        mask = horizons == horizon
        expected[mask] = losses[mask, selected_indices[index]]
    maximum_replay_error = float(np.max(np.abs(expected - verified_constant["loss"])))
    if maximum_replay_error > 1e-7:
        raise RuntimeError(f"Constant profile score replay differs by {maximum_replay_error}")
    return jsonable({"version": "component_selection_20261022_v1", "seed": int(seed),
        "tau": tau, "horizons": HORIZONS, "ridge_grid": RIDGE_GRID, "inner_folds": 3,
        "records": len(stays), "unique_stays": len(np.unique(identities)),
        "validation_rows": rows, "validation_scores": scores,
        "selected_kind": selected_kind, "best_continuous_eta": continuous["ridge_eta"],
        "best_continuous": fitted, "constant_penalties": initial,
        "constant_grid_indices": selected_indices, "constant_grid_source": grid_source,
        "constant_grid_diagnostics": grid_diagnostics,
        "constant_replay_maximum_loss_difference": maximum_replay_error,
        "constant_refit_diagnostics": verified_constant["diagnostics"],
        "max_iter": max_iter, "deterministic_starts": deterministic_starts,
        "seconds": time.perf_counter() - started})


def predict_selection(stays: list, horizons: np.ndarray, fitted: dict,
                      weights: np.ndarray | None = None) -> dict:
    horizons = np.asarray(horizons, dtype=float)
    if weights is None:
        weights = horizon_weights(horizons, require_all=False)
    batch = make_profile_batch(stays, horizons, weights)
    continuous = evaluate_component_map(batch, np.asarray(fitted["best_continuous"]["params"], dtype=float), fitted["tau"])
    constant = solve_component_batch(batch, constant_penalties(horizons, np.asarray(fitted["constant_penalties"])), fitted["tau"])
    return {"continuous": continuous, "constant": constant,
            "selected": constant if fitted["selected_kind"] == "constant" else continuous}
