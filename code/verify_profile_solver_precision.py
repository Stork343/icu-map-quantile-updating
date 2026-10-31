from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import QP_SETTINGS, POLISH_SETTINGS, check_loss, solve_spd_profile
from expanded_family_core import build_layout, later_prediction, profile_coefficients, stay_mean
from run_adaptive_clinical import layout_stays, make_layout
from run_submission_validation_extensions import global_index_lookup, write_json
from split_window_analysis_core import design_frame
from split_window_data import subset_cluster_data


def summarize(layout, prediction: np.ndarray, solver: str, tolerance: float) -> dict[str, object]:
    lower = float(np.mean(stay_mean(layout, (layout.later_values < prediction - tolerance).astype(float))))
    upper = float(np.mean(stay_mean(layout, (layout.later_values <= prediction + tolerance).astype(float))))
    return {"solver": solver, "horizon_hours": 12.0, "partition": "temporal", "tau": 0.1,
            "lambda_level": 0.03, "lambda_slope": 0.1, "n_stays": layout.n_stays,
            "n_observations": int(layout.later_values.size), "absolute_tolerance_mmhg": tolerance,
            "strict_rate": lower, "weak_rate": upper,
            "bracket_violation": max(lower - 0.1, 0.1 - upper, 0.0),
            "mean_check_loss": float(np.mean(stay_mean(layout, check_loss(layout.later_values - prediction, 0.1))))}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare certified solvers for one fixed temporal profile")
    parser.add_argument("--dataset-cache", type=Path, default=root / ".review-work" / "strengthening" / "expanded_family" / "local_dataset.pkl")
    parser.add_argument("--clinical-partition", type=Path, default=root / ".review-work" / "adaptive" / "clinical" / "horizon_12" / "temporal")
    parser.add_argument("--inherited-partition", type=Path, default=root / ".review-work" / "strengthening" / "expanded_family" / "temporal")
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work" / "adaptive" / "clinical" / "diagnostics" / "solver_precision")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.dataset_cache.open("rb") as handle:
        cached = pickle.load(handle)
    dataset = dict(cached["dataset"])
    metadata = json.loads((args.clinical_partition / "partition_results.json").read_text())
    inherited_metadata = json.loads((args.inherited_partition / "partition_results.json").read_text())
    if (metadata["population_coefficients"] != inherited_metadata["population_coefficients"]
            or metadata["age_standardization"] != inherited_metadata["age_standardization"]):
        raise RuntimeError("Solver comparison requires exactly matching population fits and age scaling")
    assignment = json.loads((args.clinical_partition / "private_partition_config.json").read_text())
    assessment = np.asarray(assignment["assessment_indices"], dtype=int)
    beta = np.asarray(metadata["population_coefficients"])
    age = metadata["age_standardization"]
    ages = cached["stays"].set_index("stay_id")["age"].reindex(dataset["cluster_ids"]).to_numpy(dtype=float)
    x = np.asarray(dataset["X"], dtype=float).copy()
    x[:, 1] = (ages - age["age_mean_training_split"]) / age["age_sd_training_split"]
    dataset["X"] = x
    dataset["X_long"] = np.repeat(x, [len(values) for values in dataset["y_list"]], axis=0)
    design, _ = design_frame(subset_cluster_data(dataset, assessment))
    layout = build_layout(design, beta, global_index_lookup(dataset), 12.0)
    current_layout, _ = make_layout(dataset, assessment, beta, 12.0)
    if not np.array_equal(layout.global_indices, current_layout.global_indices):
        raise RuntimeError("Original and current temporal layouts use different stays")
    spec = {"key": "vector_0_1", "basis": "level_slope", "lambda_level": 0.03, "lambda_slope": 0.1}
    osqp_path = args.output_dir / "private_osqp_coefficients.npz"
    if osqp_path.exists():
        with np.load(osqp_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["global_indices"], layout.global_indices):
                raise RuntimeError("OSQP precision checkpoint contains different stays")
            osqp_coefficients = saved["coefficients"]
        osqp_info = json.loads((args.output_dir / "private_osqp_diagnostics.json").read_text())
    else:
        osqp_coefficients, osqp_info = profile_coefficients(layout, spec, 0.1, 12.0)
        np.savez(osqp_path, coefficients=osqp_coefficients, global_indices=layout.global_indices)
        write_json(args.output_dir / "private_osqp_diagnostics.json", osqp_info)
    clarabel_coefficients = np.empty((layout.n_stays, 2))
    maxima = {name: 0.0 for name in ("duality_gap", "relative_duality_gap", "projected_kkt_violation", "stationarity_violation")}
    for position, stay in enumerate(layout_stays(layout)):
        clarabel_coefficients[position], info = solve_spd_profile(stay.index_residual, stay._index_design,
                                                                 0.1, np.diag([0.03, 0.1]))
        for name in maxima:
            maxima[name] = max(maxima[name], float(info[name]))
    with np.load(args.clinical_partition / "assessment_coefficients" / "vector_0_1.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["global_indices"], layout.global_indices):
            raise RuntimeError("Current Clarabel coefficient order differs")
        cached_clarabel = saved["coefficients"]
    predictions = {"osqp_original_layout": later_prediction(layout, osqp_coefficients),
                   "clarabel_original_layout": later_prediction(layout, clarabel_coefficients),
                   "clarabel_current_cache": later_prediction(current_layout, cached_clarabel)}
    rows = [summarize(layout, prediction, name, tolerance) for name, prediction in predictions.items()
            for tolerance in (0.0, 1e-10, 1e-8)]
    pd.DataFrame(rows).to_csv(args.output_dir / "profile_solver_precision.csv", index=False)
    inherited = pd.read_csv(args.inherited_partition / "assessment_metrics.csv").set_index("global_stay_index").loc[layout.global_indices]
    original_lower, original_upper = (float(inherited[f"{prefix}_level_with_slope"].mean()) for prefix in ("p_lt", "p_le"))
    osqp_ordinary = rows[0]
    result = {"fixed_profile": {"tau": 0.1, "horizon_hours": 12.0, "lambda_level": 0.03, "lambda_slope": 0.1},
              "ordinary_and_tolerance_brackets": rows, "clarabel_certificates": maxima,
              "osqp_certificates": osqp_info, "clarabel_solver_settings": QP_SETTINGS,
              "clarabel_polish_settings": POLISH_SETTINGS,
              "max_population_layout_difference": float(np.max(np.abs(layout.later_population - current_layout.later_population))),
              "max_index_residual_layout_difference": max(float(np.max(np.abs(a - b))) for a, b in zip(layout.residual_index, current_layout.residual_index)),
              "max_prediction_difference_same_layout": float(np.max(np.abs(predictions["osqp_original_layout"] - predictions["clarabel_original_layout"]))),
              "max_prediction_difference_current_cache": float(np.max(np.abs(predictions["osqp_original_layout"] - predictions["clarabel_current_cache"]))),
              "max_clarabel_prediction_difference_between_layouts": float(np.max(np.abs(predictions["clarabel_original_layout"] - predictions["clarabel_current_cache"]))),
              "inherited_ordinary_bracket": [original_lower, original_upper],
              "osqp_reproduces_inherited_ordinary_bracket": bool(abs(original_lower - osqp_ordinary["strict_rate"]) < 1e-12
                                                                 and abs(original_upper - osqp_ordinary["weak_rate"]) < 1e-12)}
    write_json(args.output_dir / "profile_solver_precision.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
