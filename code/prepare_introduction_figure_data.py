from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import empirical_quantile
from component_learning_selection import jsonable, predict_selection
from component_profile_learning import evaluate_component_map, make_profile_batch, penalty_values
from component_simulation_baselines import actual_response, combined_records, score_predictions
from run_adaptive_mixed_simulation import build_stays, simulate_dataset, split_indices, write_json


SETTING = "informative_recovery"
HORIZONS = (4.0, 8.0, 12.0)
FEATURES = ("horizon_4", "horizon_8", "horizon_12", "count", "time_spread", "q10_change")
MINIMUM_PENALTY_RATIO = 2.0
CONTOUR_LEVEL = 1.0


def early_features(data, indices: np.ndarray, beta: np.ndarray, horizon: float) -> np.ndarray:
    rows = []
    for index in indices:
        times = (data.index_time[index] - 12.0) / 12.0
        residuals = data.index_y[index] - float(data.x[index] @ beta)
        middle = len(times) // 2
        if middle == 0:
            raise ValueError("Every illustrated history requires two nonempty rank halves")
        history = [
            np.clip(np.log(len(times) / 12.0) / np.log(4.0), -1.0, 1.0),
            np.clip(12.0 * np.var(times) - 1.0, -1.0, 1.0),
            np.clip((empirical_quantile(residuals[middle:])
                     - empirical_quantile(residuals[:middle])) / 10.0, -1.0, 1.0),
        ]
        horizon_features = [max((8.0 - horizon) / 4.0, 0.0),
                            1.0 - abs(horizon - 8.0) / 4.0,
                            max((horizon - 8.0) / 4.0, 0.0)]
        rows.append([*horizon_features, *history])
    return np.asarray(rows)


def select_source(source: Path):
    candidates = sorted((source / "replicates" / SETTING).glob("rep_*/frozen.json"))
    if not candidates:
        raise FileNotFoundError("No frozen informative recovery replicates were found")
    inspected = []
    for path in candidates:
        frozen = json.loads(path.read_text())
        completed = json.loads((path.parent / "complete.json").read_text())
        if completed["status"] != "complete" or completed["setting"] != SETTING:
            raise ValueError("The frozen simulation replicate is incomplete")
        learning = frozen["learning"]
        if learning["selected_kind"] != "continuous":
            inspected.append({"replicate": completed["replicate"],
                              "selected_kind": learning["selected_kind"]})
            continue
        np.testing.assert_array_equal(frozen["derived_seeds"], completed["derived_seeds"])
        data = simulate_dataset(SETTING, 600, frozen["derived_seeds"][0])
        fitting, tuning, assessment = split_indices(600, frozen["derived_seeds"][1])
        beta = np.asarray(frozen["population_coefficients"], dtype=float)
        params = np.asarray(learning["best_continuous"]["params"], dtype=float)
        features = early_features(data, tuning, beta, 12.0)
        penalties = penalty_values(params, features)[0]
        ratios = penalties.max(axis=0) / penalties.min(axis=0)
        inspected.append({"replicate": completed["replicate"], "selected_kind": "continuous",
                          "tuning_penalty_minima": penalties.min(axis=0).tolist(),
                          "tuning_penalty_maxima": penalties.max(axis=0).tolist(),
                          "tuning_penalty_maximum_to_minimum_ratios": ratios.tolist()})
        if np.all(ratios > MINIMUM_PENALTY_RATIO):
            return (path.parent, frozen, completed, data, fitting, tuning, assessment, inspected)
    raise ValueError("No continuous frozen replicate meets the specified penalty variation criterion")


def prepare(source: Path, output: Path) -> dict:
    (directory, frozen, completed, data, fitting, tuning,
     assessment, inspected) = select_source(source)
    if len(set(fitting) & set(tuning)) or len(set(fitting) & set(assessment)) or len(set(tuning) & set(assessment)):
        raise ValueError("The simulation sample allocations overlap")
    beta = np.asarray(frozen["population_coefficients"], dtype=float)
    learning = frozen["learning"]
    params = np.asarray(learning["best_continuous"]["params"], dtype=float)
    if learning["tau"] != 0.10:
        raise ValueError("The introduction illustration requires the saved 0.10 quantile rule")

    # 选例使用 early history 和冻结函数，按相同比例并列时选取最小 index。
    assessment = np.sort(assessment)
    candidate_features = early_features(data, assessment, beta, 12.0)
    candidate_penalties = penalty_values(params, candidate_features)[0]
    ratio = candidate_penalties[:, 1] / candidate_penalties[:, 0]
    positions = (int(np.argmin(ratio)), int(np.argmax(ratio)))
    if positions[0] == positions[1]:
        raise ValueError("The history specific penalty geometries are identical")
    selected_indices = np.asarray([assessment[position] for position in positions], dtype=np.int64)
    selected_stays = build_stays(data, selected_indices, beta)

    # 完整重现 assessment 上保存的损失与两个尾部概率。
    complete_records, complete_horizons, complete_ids = combined_records(
        build_stays(data, assessment, beta), assessment)
    complete_batch = make_profile_batch(complete_records, complete_horizons)
    evaluated = predict_selection(complete_records, complete_horizons, learning)
    retained = pd.read_parquet(directory / "assessment_metrics.parquet")
    response = actual_response(complete_records, complete_horizons)
    replay_errors = {}
    for name in ("continuous", "constant", "selected"):
        scores = score_predictions(complete_batch, evaluated[name]["flat_predictions"], response)
        stored = retained.loc[retained.method == f"{name}_direct"].set_index(
            ["horizon_hours", "synthetic_stay_index"])
        keys = pd.MultiIndex.from_arrays([complete_horizons, complete_ids], names=stored.index.names)
        for metric in ("loss", "p_lt", "p_le"):
            expected = stored.loc[keys, metric].to_numpy()
            np.testing.assert_allclose(scores[metric], expected, rtol=0, atol=1e-12)
            replay_errors[f"{name}_{metric}"] = float(np.max(np.abs(scores[metric] - expected)))

    requested_horizons = np.tile(HORIZONS, len(selected_indices))
    records = [stay for stay in selected_stays for _ in HORIZONS]
    requested = make_profile_batch(records, requested_horizons)
    fitted = evaluate_component_map(requested, params, learning["tau"])
    np.testing.assert_allclose(
        requested.features,
        np.asarray([early_features(data, np.asarray([index]), beta, horizon)[0]
                    for index in selected_indices for horizon in HORIZONS]), rtol=0, atol=1e-14)
    if not fitted["diagnostics"]["all_certified"]:
        raise ValueError("The illustrated individual profiles did not pass numerical certification")

    observation_rows, profile_rows, curve_rows, contour_rows, examples = [], [], [], [], []
    curve_replay_errors = []
    for example_number, (label, stay, index) in enumerate(zip(("A", "B"), selected_stays, selected_indices)):
        population = float(data.x[index] @ beta)
        for window, times, values in (("index", stay.index_time, stay.index_y),
                                      ("later", stay.later_time, stay.later_y)):
            for time, value in zip(times, values):
                observation_rows.append({"example": label, "synthetic_stay_index": int(index),
                    "window": window, "time_hours": time, "map_mmhg": value,
                    "population_mmhg": population})
        for horizon_number, horizon in enumerate(HORIZONS):
            number = example_number * len(HORIZONS) + horizon_number
            level, slope = fitted["coefficients"][number]
            lambda_level, lambda_slope = fitted["penalties"][number]
            radii = np.sqrt(CONTOUR_LEVEL / np.asarray([lambda_level, lambda_slope]))
            profile_rows.append({"example": label, "synthetic_stay_index": int(index),
                "horizon_hours": horizon, "lambda_level": lambda_level, "lambda_slope": lambda_slope,
                "level_coefficient": level, "slope_coefficient": slope,
                "contour_level": CONTOUR_LEVEL, "ellipse_level_radius": radii[0],
                "ellipse_slope_radius": radii[1],
                **dict(zip(FEATURES, requested.features[number]))})
            record_index = np.flatnonzero((complete_ids == index) & (complete_horizons == horizon))
            if len(record_index) != 1:
                raise ValueError("An illustration does not identify one saved assessment horizon record")
            record_number = int(record_index[0])
            first, stop = complete_batch.later_indptr[record_number:record_number + 2]
            reproduced = population + level + slope * complete_batch.later_time[first:stop]
            expected = evaluated["continuous"]["flat_predictions"][first:stop]
            np.testing.assert_allclose(reproduced, expected, rtol=0, atol=1e-8)
            curve_replay_errors.append(float(np.max(np.abs(reproduced - expected))))
            for time in np.arange(12.0, 12.0 + horizon + 0.125, 0.25):
                slope_contribution = slope * (time - 12.0) / 12.0
                curve_rows.append({"example": label, "synthetic_stay_index": int(index),
                    "horizon_hours": horizon, "time_hours": time, "population_mmhg": population,
                    "level_contribution_mmhg": level, "slope_contribution_mmhg": slope_contribution,
                    "level_prediction_mmhg": population + level,
                    "direct_prediction_mmhg": population + level + slope_contribution})
            for angle in np.linspace(0.0, 2.0 * np.pi, 181):
                point = radii * np.asarray([np.cos(angle), np.sin(angle)])
                penalty = lambda_level * point[0] ** 2 + lambda_slope * point[1] ** 2
                np.testing.assert_allclose(penalty, CONTOUR_LEVEL, rtol=0, atol=1e-14)
                contour_rows.append({"example": label, "synthetic_stay_index": int(index),
                    "horizon_hours": horizon, "contour_level": CONTOUR_LEVEL,
                    "angle_radians": angle, "u_level": point[0], "u_slope": point[1],
                    "penalty_value": penalty})
        examples.append({"example": label, "synthetic_stay_index": int(index),
            "index_count": len(stay.index_y), "index_coverage_hours": float(data.coverage[index]),
            "index_observed_span_hours": float(np.ptp(stay.index_time)),
            "population_mmhg": population, "index_q10_mmhg": empirical_quantile(stay.index_y),
            "selection_penalty_ratio_at_12_hours": float(ratio[positions[example_number]]),
            "x": data.x[index].tolist(), "index_time": stay.index_time.tolist(),
            "index_y": stay.index_y.tolist()})

    output.mkdir(parents=True, exist_ok=True)
    table_rows = {"introduction_observations.csv": observation_rows,
                  "introduction_profiles.csv": profile_rows, "introduction_forecast.csv": curve_rows,
                  "introduction_contours.csv": contour_rows}
    for filename, rows in table_rows.items():
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    pd.DataFrame([{"component": component, "feature": feature, "estimate": value}
                  for component, values in zip(("level", "slope"), params)
                  for feature, value in zip(FEATURES, values)]).to_csv(
        output / "introduction_parameters.csv", index=False)
    manifest = {"illustration_type": "Simulation mechanism illustration", "setting": SETTING,
        "replicate": completed["replicate"], "master_seed": completed["seed"],
        "seed_sequence_tuple": [completed["seed"], 0, completed["replicate"]],
        "derived_seeds": frozen["derived_seeds"],
        "source_directory": str(directory), "tau": learning["tau"], "landmark_hours": 12.0,
        "horizons_hours": HORIZONS, "allocation_counts": {"F": len(fitting), "T": len(tuning), "A": len(assessment)},
        "replicate_selection_rule": "First continuous frozen replicate in numerical order whose two "
            "penalty functions each have maximum/minimum ratio greater than 2 over tuning early histories at D=12",
        "replicate_selection_inputs": ["frozen function parameters", "tuning early history features"],
        "replicate_selection_audit": inspected,
        "record_selection_rule": "Assessment histories with minimum and maximum lambda_slope/lambda_level "
            "at D=12; numerical ties choose the smallest synthetic stay index",
        "record_selection_inputs": ["assessment membership", "early history", "frozen function parameters"],
        "population_coefficients": beta.tolist(), "component_parameters": params.tolist(),
        "selected_kind": learning["selected_kind"], "best_continuous_eta": learning["best_continuous_eta"],
        "examples": examples, "profile_diagnostics": fitted["diagnostics"],
        "assessment_replay_maximum_errors": replay_errors,
        "example_curve_replay_maximum_error": max(curve_replay_errors),
        "contour_definition": "lambda_level*u_level^2+lambda_slope*u_slope^2=1",
        "contour_semantics": "Unit level sets of the quadratic penalty in the individual check loss objective",
        "contour_level": CONTOUR_LEVEL,
        "curve_definition": "At fixed requested horizon D, q(t;D)=x beta+u0(D)+u1(D)*(t-12)/12 for 12<=t<=12+D",
        "forecast_performance_claim": "none",
        "table_row_counts": {filename: len(rows) for filename, rows in table_rows.items()},
        "plot_table_names": [*table_rows, "introduction_parameters.csv"]}
    write_json(output / "introduction_data.json", jsonable(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = prepare(arguments.source.resolve(strict=True), arguments.output)
    print(json.dumps({"output": str(arguments.output), "replicate": manifest["replicate"],
        "examples": [row["synthetic_stay_index"] for row in manifest["examples"]],
        "profile_diagnostics": manifest["profile_diagnostics"],
        "maximum_assessment_replay_error": max(manifest["assessment_replay_maximum_errors"].values()),
        "maximum_curve_replay_error": manifest["example_curve_replay_maximum_error"]}, indent=2))


if __name__ == "__main__":
    main()
