from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import empirical_quantile
from component_learning_selection import jsonable, predict_selection
from component_profile_learning import evaluate_component_map, make_profile_batch
from component_simulation_baselines import actual_response, combined_records, score_predictions
from run_adaptive_mixed_simulation import build_stays, simulate_dataset, split_indices, write_json


SETTING = "informative_recovery"
REPLICATE = 1
HORIZONS = (4.0, 8.0, 12.0)
FEATURES = ("horizon_4", "horizon_8", "horizon_12", "count", "time_spread", "q10_change")


def prepare(source: Path, output: Path) -> dict:
    directory = source / "replicates" / SETTING / f"rep_{REPLICATE:04d}"
    frozen = json.loads((directory / "frozen.json").read_text())
    completed = json.loads((directory / "complete.json").read_text())
    if completed["status"] != "complete" or completed["replicate"] != REPLICATE:
        raise ValueError("The requested frozen simulation replicate is incomplete")
    seeds = frozen["derived_seeds"]
    np.testing.assert_array_equal(seeds, completed["derived_seeds"])
    data = simulate_dataset(SETTING, 600, seeds[0])
    fitting, tuning, assessment = split_indices(600, seeds[1])
    if len(set(fitting) & set(tuning)) or len(set(fitting) & set(assessment)) or len(set(tuning) & set(assessment)):
        raise ValueError("The simulation sample allocations overlap")

    # 选择只读取 assessment 分配、index 数量及覆盖时间。
    designs = (("A", 6, 4.0), ("B", 16, 12.0))
    selections = []
    for label, count, coverage in designs:
        candidates = sorted(int(i) for i in assessment
                            if len(data.index_y[i]) == count and data.coverage[i] == coverage)
        if not candidates:
            raise ValueError("The fixed early sampling design has no assessment stay")
        selections.append({"example": label, "synthetic_stay_index": candidates[0],
                           "index_count": count, "index_coverage_hours": coverage,
                           "eligible_examples": len(candidates)})
    indices = np.asarray([row["synthetic_stay_index"] for row in selections], dtype=np.int64)
    beta = np.asarray(frozen["population_coefficients"], dtype=float)
    learning = frozen["learning"]
    params = np.asarray(learning["best_continuous"]["params"], dtype=float)
    selected = build_stays(data, indices, beta)

    # 全部 assessment 预测与保存的逐条结果进行数值比较。
    complete_records, complete_horizons, complete_ids = combined_records(
        build_stays(data, assessment, beta), assessment)
    batch = make_profile_batch(complete_records, complete_horizons)
    evaluated = predict_selection(complete_records, complete_horizons, learning)
    retained = pd.read_parquet(directory / "assessment_metrics.parquet")
    response = actual_response(complete_records, complete_horizons)
    replay_errors = {}
    for name in ("continuous", "constant", "selected"):
        scores = score_predictions(batch, evaluated[name]["flat_predictions"], response)
        stored = retained.loc[retained.method == f"{name}_direct"].set_index(
            ["horizon_hours", "synthetic_stay_index"])
        keys = pd.MultiIndex.from_arrays([complete_horizons, complete_ids], names=stored.index.names)
        for metric in ("loss", "p_lt", "p_le"):
            expected = stored.loc[keys, metric].to_numpy()
            np.testing.assert_allclose(scores[metric], expected, rtol=0, atol=1e-12)
            replay_errors[f"{name}_{metric}"] = float(np.max(np.abs(scores[metric] - expected)))

    records, observation_rows, curve_rows, profile_rows = [], [], [], []
    curve_replay_errors = []
    horizon_grid = np.linspace(4.0, 12.0, 81)
    for selection, stay, index in zip(selections, selected, indices):
        label = selection["example"]
        population = float(data.x[index] @ beta)
        requested = make_profile_batch([stay] * len(horizon_grid), horizon_grid)
        fitted = evaluate_component_map(requested, params, learning["tau"])
        for window, times, values in (("index", stay.index_time, stay.index_y),
                                      ("later", stay.later_time, stay.later_y)):
            for time, value in zip(times, values):
                observation_rows.append({"example": label, "synthetic_stay_index": int(index),
                    "window": window, "time_hours": time, "map_mmhg": value,
                    "population_mmhg": population})
        for number, horizon in enumerate(horizon_grid):
            level, slope = fitted["coefficients"][number]
            lambda_level, lambda_slope = fitted["penalties"][number]
            profile_rows.append({"example": label, "synthetic_stay_index": int(index),
                "horizon_hours": horizon, "lambda_level": lambda_level, "lambda_slope": lambda_slope,
                "level_coefficient": level, "slope_coefficient": slope,
                **dict(zip(FEATURES, requested.features[number]))})
            if not any(np.isclose(horizon, value, rtol=0, atol=1e-12) for value in HORIZONS):
                continue
            record_index = np.flatnonzero((complete_ids == index) & (complete_horizons == horizon))
            if len(record_index) != 1:
                raise ValueError("An example does not identify one assessment horizon record")
            record_number = int(record_index[0])
            first, stop = batch.later_indptr[record_number:record_number + 2]
            reproduced = population + level + slope * batch.later_time[first:stop]
            expected = evaluated["continuous"]["flat_predictions"][first:stop]
            np.testing.assert_allclose(reproduced, expected, rtol=0, atol=1e-8)
            curve_replay_errors.append(float(np.max(np.abs(reproduced - expected))))
            for time in np.arange(0.0, 12.0 + horizon + 0.125, 0.25):
                slope_contribution = slope * (time - 12.0) / 12.0
                curve_rows.append({"example": label, "synthetic_stay_index": int(index),
                    "horizon_hours": horizon, "time_hours": time,
                    "population_mmhg": population, "level_contribution_mmhg": level,
                    "slope_contribution_mmhg": slope_contribution,
                    "level_prediction_mmhg": population + level,
                    "direct_prediction_mmhg": population + level + slope_contribution})
        knots = [row for row in profile_rows if row["example"] == label and row["horizon_hours"] in HORIZONS]
        records.append({**selection, "x": data.x[index].tolist(), "population_mmhg": population,
            "index_q10_mmhg": empirical_quantile(stay.index_y),
            "index_time": stay.index_time.tolist(), "index_y": stay.index_y.tolist(),
            "later_time": stay.later_time.tolist(), "later_y": stay.later_y.tolist(),
            "knots": knots, "profile_diagnostics": fitted["diagnostics"]})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observation_rows).to_csv(output / "method_figure_observations.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "method_figure_horizon_profiles.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "method_figure_quantile_curves.csv", index=False)
    parameter_rows = [{"component": component, "feature": feature, "estimate": value}
                      for component, values in zip(("level", "slope"), params)
                      for feature, value in zip(FEATURES, values)]
    pd.DataFrame(parameter_rows).to_csv(output / "method_figure_parameters.csv", index=False)
    manifest = {"illustration_type": "Simulation illustration", "setting": SETTING,
        "replicate": REPLICATE, "master_seed": completed["seed"],
        "seed_sequence_tuple": [completed["seed"], 0, REPLICATE], "derived_seeds": seeds,
        "source_directory": str(directory), "tau": learning["tau"], "landmark_hours": 12.0,
        "allocation_counts": {"F": len(fitting), "T": len(tuning), "A": len(assessment)},
        "selection_rule": "Smallest assessment synthetic index with count=6 and coverage=4 hours for A; "
                          "smallest assessment synthetic index with count=16 and coverage=12 hours for B",
        "selection_inputs": ["assessment membership", "index count", "index coverage", "synthetic index"],
        "population_coefficients": beta.tolist(), "component_parameters": params.tolist(),
        "constant_penalties": learning["constant_penalties"], "selected_kind": learning["selected_kind"],
        "best_continuous_eta": learning["best_continuous_eta"],
        "examples": records, "assessment_replay_maximum_errors": replay_errors,
        "example_curve_replay_maximum_error": max(curve_replay_errors),
        "assessment_profile_diagnostics": {name: value["diagnostics"] for name, value in evaluated.items()},
        "curve_definition": "For each requested horizon D, q(t;D)=x beta+b0(D)+b1(D)(t-12)/12. "
                            "The requested horizon is fixed along each curve.",
        "horizon_grid": {"start": 4.0, "end": 12.0, "step": 0.1},
        "plot_table_names": ["method_figure_observations.csv", "method_figure_horizon_profiles.csv",
                             "method_figure_quantile_curves.csv", "method_figure_parameters.csv"]}
    write_json(output / "method_figure_data.json", jsonable(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = prepare(arguments.source, arguments.output)
    print(json.dumps({"output": str(arguments.output),
        "examples": [row["synthetic_stay_index"] for row in manifest["examples"]],
        "maximum_replay_error": max(manifest["assessment_replay_maximum_errors"].values())}, indent=2))


if __name__ == "__main__":
    main()
