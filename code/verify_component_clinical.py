from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_clinical_data import combined_records, load_clinical_cache, load_partition
from component_learning_selection import DIAGONAL_GRID
from component_profile_learning import evaluate_component_map, make_profile_batch, solve_component_batch


HORIZONS = (4., 8., 12.)
MODELS = ("continuous_direct", "continuous_affine", "selected_component_direct", "selected_component_affine",
          "constant_diagonal_direct", "constant_diagonal_affine")
OLD_MODELS = ("population", "calibrated_q10", "original_selected", "vector_direct", "vector_affine", "stump_original")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def weights(horizons):
    result = np.zeros(len(horizons))
    for horizon in HORIZONS:
        mask = horizons == horizon
        require(mask.any(), "An evaluation is missing a forecast horizon")
        result[mask] = 1.0 / (3 * int(mask.sum()))
    return result


def normal_interval(first, second):
    difference = np.asarray(first) - np.asarray(second)
    standard_error = np.std(difference, ddof=1) / np.sqrt(len(difference))
    return {"n_stays": len(difference), "mean_difference": difference.mean(), "se": standard_error,
            "ci95_low": difference.mean() - 1.96 * standard_error,
            "ci95_high": difference.mean() + 1.96 * standard_error,
            "candidate_better_fraction": np.mean(difference < 0)}


def verify_partition(cached, source, output, name):
    partition = load_partition(cached, source, name)
    directory = output / name
    completion = json.loads((directory / "complete.json").read_text())
    require(completion["partition"] == name and completion["status"] == "complete", "Incorrect clinical completion identity")
    fit = json.loads((directory / "fitted_selection.json").read_text())
    require(fit["inner_folds"] == 3 and fit["ridge_grid"] == [0., .001, .01, .1], "Incorrect learning selection protocol")
    params = np.asarray(fit["best_continuous"]["params"])
    require(params.shape == (2, 6) and np.isfinite(params).all(), "Invalid learned function parameters")
    require((np.abs(params[:, :3]) <= 12 + 1e-12).all() and (np.abs(params[:, 3:]) <= 4 + 1e-12).all(), "Function parameters leave the prespecified domain")
    stays, horizons, _, identities = combined_records(partition.tuning_layouts)
    unique = np.unique(identities)
    order = np.random.default_rng(fit["seed"]).permutation(len(unique))
    labels = np.empty(len(unique), dtype=int)
    labels[order] = np.arange(len(unique)) % 3
    folds = labels[np.searchsorted(unique, identities)]
    full_weights = weights(horizons)
    with np.load(directory / "constant_tuning_losses.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["identities"], identities)
        np.testing.assert_array_equal(saved["horizons"], horizons)
        np.testing.assert_array_equal(saved["penalty_grid"], DIAGONAL_GRID)
        grid_loss = saved["losses"]
    contribution = {}
    fits_checked = 0
    profile_evaluations = 0
    for row in fit["validation_rows"]:
        fold = row["fold"] - 1
        training, heldout = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        expected_penalties, indices = [], []
        for horizon in HORIZONS:
            local = training[horizons[training] == horizon]
            best = int(np.argmin(grid_loss[local].mean(axis=0)))
            indices.append(best)
            expected_penalties.append(DIAGONAL_GRID[best])
        np.testing.assert_array_equal(row["initial_penalties"], expected_penalties)
        if row["candidate"] == "constant":
            losses = np.empty(len(heldout))
            for index, horizon in enumerate(HORIZONS):
                mask = horizons[heldout] == horizon
                losses[mask] = grid_loss[heldout[mask], indices[index]]
        else:
            current = row["fit"]
            for key in ("all_evaluation_diagnostics", "gradient_norm", "projected_gradient_norm"):
                require(key in current, "A validation fit lacks the frozen numerical diagnostics")
            require(current["profile_certificates_passed"] is True and current["global_optimum_certified"] is False,
                    "The returned rule misstates its optimization guarantee")
            batch = make_profile_batch([stays[index] for index in heldout], horizons[heldout], weights(horizons[heldout]))
            evaluated = evaluate_component_map(batch, np.asarray(current["params"]))
            losses = evaluated["loss"]
            fits_checked += 1
            profile_evaluations += current["all_evaluation_diagnostics"]["profiles"]
        value = float(full_weights[heldout] @ losses)
        np.testing.assert_allclose(row["validation_contribution"], value, rtol=0, atol=2e-10)
        np.testing.assert_allclose(row["validation_loss"], weights(horizons[heldout]) @ losses, rtol=0, atol=2e-10)
        contribution[row["candidate"]] = contribution.get(row["candidate"], 0.0) + value
    for row in fit["validation_scores"]:
        np.testing.assert_allclose(row["validation_loss"], contribution[row["candidate"]], rtol=0, atol=2e-10)
    best_continuous = min((row for row in fit["validation_scores"] if row["ridge_eta"] is not None), key=lambda row: row["validation_loss"])
    require(fit["best_continuous_eta"] == best_continuous["ridge_eta"], "Incorrect continuous complexity selection")
    constant_loss = next(row["validation_loss"] for row in fit["validation_scores"] if row["candidate"] == "constant")
    selected = "constant" if constant_loss <= best_continuous["validation_loss"] + 1e-12 else "continuous"
    require(fit["selected_kind"] == selected, "Incorrect constant or continuous branch selection")
    current = fit["best_continuous"]
    require(current["profile_certificates_passed"] is True and current["global_optimum_certified"] is False,
            "Incorrect full tuning profile certificate metadata")
    profile_evaluations += current["all_evaluation_diagnostics"]["profiles"]
    fits_checked += 1
    assessment_stays, assessment_horizons, _, _ = combined_records(partition.assessment_layouts)
    batch = make_profile_batch(assessment_stays, assessment_horizons, weights(assessment_horizons))
    continuous = evaluate_component_map(batch, params)
    constants = np.empty((len(assessment_stays), 2))
    for index, horizon in enumerate(HORIZONS):
        constants[assessment_horizons == horizon] = fit["constant_penalties"][index]
    constant = solve_component_batch(batch, constants)
    evaluations = {"continuous": continuous, "constant_diagonal": constant,
                   "selected_component": constant if selected == "constant" else continuous}
    calibrations = json.loads((directory / "calibration_coefficients.json").read_text())
    frames, rows, offset = {}, [], 0
    for horizon in HORIZONS:
        layout = partition.assessment_layouts[horizon]
        frame = pd.read_csv(directory / f"assessment_h{int(horizon)}.csv")
        expected = partition.baselines[horizon]
        np.testing.assert_array_equal(frame["global_stay_index"], expected["global_stay_index"])
        np.testing.assert_array_equal(frame["late_obs"], expected["late_obs"])
        for model in OLD_MODELS:
            for metric in ("loss", "p_lt", "p_le"):
                np.testing.assert_allclose(frame[f"{metric}_{model}"], expected[f"{metric}_{model}"], rtol=0, atol=2e-10)
        for label, values in evaluations.items():
            coefficients = values["coefficients"][offset:offset + layout.n_stays]
            raw = layout.later_population + coefficients[layout.later_stay, 0] + coefficients[layout.later_stay, 1] * layout.later_basis
            calibration_key = {"continuous": "continuous", "constant_diagonal": "constant", "selected_component": "selected"}[label]
            calibration = calibrations[str(int(horizon))][calibration_key]
            require(calibration["stationarity"] <= 1e-6 and calibration["relative_gap"] <= 1e-7, "An affine fit lacks its numerical certificate")
            calibrated = calibration["intercept"] + calibration["slope"] * raw
            for suffix, prediction in (("direct", raw), ("affine", calibrated)):
                residual = layout.later_values - prediction
                point_loss = residual * (.1 - (residual < 0))
                for metric, point in (("loss", point_loss), ("p_lt", layout.later_values < prediction), ("p_le", layout.later_values <= prediction)):
                    scored = np.add.reduceat(np.asarray(point, dtype=float), layout.later_starts) / layout.later_counts
                    np.testing.assert_allclose(frame[f"{metric}_{label}_{suffix}"], scored, rtol=0, atol=2e-10)
        frames[horizon] = frame
        for model in OLD_MODELS + MODELS:
            lower, upper = frame[[f"p_lt_{model}", f"p_le_{model}"]].mean()
            rows.append({"scope": "temporal_2017_2022" if name == "temporal" else name,
                "horizon_hours": horizon, "model": model, "n_stays": len(frame),
                "mean_loss": frame[f"loss_{model}"].mean(), "p_lt": lower, "p_le": upper,
                "bracket_violation": max(lower - .1, .1 - upper, 0)})
        offset += layout.n_stays
    return frames, rows, {"partition": name, "fits_checked": fits_checked,
        "optimization_profile_evaluations_checked": profile_evaluations,
        "assessment_records_checked": len(assessment_stays), "selected_kind": selected}


def main():
    project = Path(__file__).resolve().parents[1]
    root = project.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, default=root / ".review-work/strengthening/expanded_family/local_dataset.pkl")
    parser.add_argument("--source", type=Path, default=root / ".review-work/adaptive/clinical")
    parser.add_argument("--private", type=Path, default=root / ".review-work/component_learning/clinical")
    parser.add_argument("--public", type=Path, default=root / ".review-work/component_learning/public/clinical")
    args = parser.parse_args()
    cached = load_clinical_cache(args.dataset_cache)
    output, rebuilt, reports = {}, [], []
    names = [f"fold_{index}" for index in range(1, 6)] + ["temporal"]
    for name in names:
        frames, rows, report = verify_partition(cached, args.source, args.private, name)
        output[name] = frames
        rebuilt.extend(rows)
        reports.append(report)
    for horizon in HORIZONS:
        combined = pd.concat([output[name][horizon] for name in names if name != "temporal"], ignore_index=True)
        require(not combined["global_stay_index"].duplicated().any(), "Nested assessment stays overlap")
        require(len(combined) == {4.: 59969, 8.: 60241, 12.: 60310}[horizon], "Nested assessment coverage differs")
        for model in OLD_MODELS + MODELS:
            lower, upper = combined[[f"p_lt_{model}", f"p_le_{model}"]].mean()
            rebuilt.append({"scope": "nested_pooled", "horizon_hours": horizon, "model": model,
                "n_stays": len(combined), "mean_loss": combined[f"loss_{model}"].mean(),
                "p_lt": lower, "p_le": upper, "bracket_violation": max(lower - .1, .1 - upper, 0)})
    actual = pd.read_csv(args.public / "clinical_model_summary.csv").set_index(["scope", "horizon_hours", "model"]).sort_index()
    expected = pd.DataFrame(rebuilt).set_index(["scope", "horizon_hours", "model"]).sort_index()
    require(actual.index.equals(expected.index), "Public clinical model summary grid differs")
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=0, atol=2e-10)
    pairs = pd.read_csv(args.public / "clinical_temporal_paired.csv")
    require(len(pairs) == 120 and not pairs.duplicated(["horizon_hours", "first", "second"]).any(), "Temporal comparison grid differs")
    for row in pairs.to_dict("records"):
        frame = output["temporal"][row["horizon_hours"]]
        interval = normal_interval(frame[f"loss_{row['first']}"], frame[f"loss_{row['second']}"])
        for key, value in interval.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=2e-10)
    report = {"status": "passed", "clinical_partitions": 6, "horizon_comparisons": 18,
        "model_summary_cells": len(actual), "temporal_paired_comparisons": len(pairs),
        "inner_validation_and_selection_replayed": True, "assessment_predictions_replayed": True,
        "original_assessment_membership_and_scores_matched": True, "public_identifier_columns": 0,
        "partition_checks": reports}
    (args.public / "clinical_independent_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
