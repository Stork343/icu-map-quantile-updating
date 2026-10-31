from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from expanded_family_core import later_prediction
from run_adaptive_clinical import aggregate, make_layout, prediction_metrics
from run_submission_validation_extensions import write_json
from run_temporal_validation import paired_interval
from summarize_adaptive_clinical import join_random_slope


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export frozen 61-node temporal random-slope sensitivity")
    parser.add_argument("--dataset-cache", type=Path, default=root / ".review-work/strengthening/expanded_family/local_dataset.pkl")
    parser.add_argument("--clinical-dir", type=Path, default=root / ".review-work/adaptive/clinical")
    parser.add_argument("--random-slope-61-dir", type=Path, default=root / ".review-work/adaptive/random_slope_precision_61")
    parser.add_argument("--tuning-precision", type=Path, default=root / ".review-work/adaptive/precision/temporal/quadrature_31_to_61.csv")
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work/adaptive/public/clinical")
    args = parser.parse_args()
    with args.dataset_cache.open("rb") as handle:
        cached = pickle.load(handle)
    dataset = dict(cached["dataset"])
    fixed_dir = args.clinical_dir / "horizon_12" / "temporal"
    metadata = json.loads((fixed_dir / "partition_results.json").read_text())
    assignment = json.loads((fixed_dir / "private_partition_config.json").read_text())
    assessment = np.asarray(assignment["assessment_indices"], dtype=int)
    beta = np.asarray(metadata["population_coefficients"])
    age = metadata["age_standardization"]
    ages = cached["stays"].set_index("stay_id")["age"].reindex(dataset["cluster_ids"]).to_numpy(dtype=float)
    x = np.asarray(dataset["X"], dtype=float).copy()
    x[:, 1] = (ages - age["age_mean_training_split"]) / age["age_sd_training_split"]
    dataset["X"] = x
    with np.load(fixed_dir / "assessment_coefficients" / "vector_0_1.npz", allow_pickle=False) as saved:
        coefficient_indices = saved["global_indices"]
        coefficients = saved["coefficients"]
    if not np.array_equal(coefficient_indices, assessment):
        raise RuntimeError("The fixed diagonal profile cache differs from the frozen temporal cohort")
    coefficient_lookup = pd.Series(np.arange(assessment.size), index=assessment)
    reference_metadata = json.loads((args.random_slope_61_dir / "temporal" / "results.json").read_text())
    validation = reference_metadata["fitting_validation"]
    if not validation["validation_passed"] or validation["nodes_per_dimension"] != 61 or validation["quadrature_nodes"] != 3721:
        raise RuntimeError("The temporal sensitivity requires the verified 61-node fit")
    model_names = {"random_slope_lqmm": "random_slope_lqmm_61",
                   "calibrated_random_slope_lqmm": "calibrated_random_slope_lqmm_61"}
    summaries, comparisons, calibrations, cohort_checks = [], [], [], []
    for horizon in (4.0, 8.0, 12.0):
        layout, _ = make_layout(dataset, assessment, beta, horizon)
        selected_coefficients = coefficients[coefficient_lookup.loc[layout.global_indices].to_numpy(dtype=int)]
        prediction = later_prediction(layout, selected_coefficients)
        frame = pd.DataFrame({"global_stay_index": layout.global_indices, "late_obs": layout.later_counts})
        for name, values in prediction_metrics(layout, prediction).items():
            frame[f"{name}_fixed_diagonal_profile"] = values
        if horizon == 12.0:
            frozen = pd.read_csv(fixed_dir / "assessment_metrics.csv").set_index("global_stay_index").loc[layout.global_indices]
            if not np.allclose(frame["loss_fixed_diagonal_profile"], frozen["loss_vector_direct"], rtol=1e-12, atol=1e-12):
                raise RuntimeError("The fixed diagonal profile does not reproduce its frozen D12 loss")
        frame = join_random_slope(frame, args.random_slope_61_dir / "temporal" / f"assessment_horizon_{int(horizon)}.csv")
        summary = aggregate(frame, "temporal_2017_2022_sensitivity_61", horizon)
        summary["model"] = summary["model"].replace(model_names)
        summary["profile_level_penalty"] = 0.03
        summary["profile_slope_penalty"] = 0.1
        summaries.append(summary)
        for reference, label in model_names.items():
            left = frame["loss_fixed_diagonal_profile"].to_numpy(dtype=float)
            right = frame[f"loss_{reference}"].to_numpy(dtype=float)
            comparisons.append({"scope": "temporal_2017_2022_sensitivity_61", "horizon_hours": horizon,
                                "candidate": "fixed_diagonal_profile", "reference": label,
                                "profile_level_penalty": 0.03, "profile_slope_penalty": 0.1,
                                **paired_interval(left, right), "interval_type": "pointwise paired normal 95 percent"})
        calibration = reference_metadata["horizons"][str(int(horizon))]["calibration"]
        calibrations.append({"horizon_hours": horizon, "intercept": calibration["intercept"],
                             "slope": calibration["slope"], "stationarity": calibration["stationarity"],
                             "relative_gap": calibration["relative_gap"]})
        cohort_checks.append({"horizon_hours": horizon, "n_stays": frame.shape[0],
                              "n_later_observations": int(frame["late_obs"].sum()),
                              "exact_identifiers_and_later_counts_match": True})
    tuning = pd.read_csv(args.tuning_precision)
    if (tuning.shape[0] != 6 or set(tuning["nodes_first"]) != {31} or set(tuning["nodes_second"]) != {61}
            or set(tuning["horizon_hours"]) != {4, 8, 12} or set(tuning["method"]) != {"direct", "affine"}):
        raise RuntimeError("The 31-to-61 tuning precision table is incomplete")
    parameters = pd.DataFrame([{"term": name, "estimate": value} for name, value in reference_metadata["parameters"].items()])
    tables = {"clinical_quadrature61_temporal_model_summary.csv": pd.concat(summaries, ignore_index=True),
              "clinical_quadrature61_temporal_paired_comparisons.csv": pd.DataFrame(comparisons),
              "clinical_quadrature31_to61_tuning_precision.csv": tuning,
              "clinical_quadrature61_parameters.csv": parameters,
              "clinical_quadrature61_calibrations.csv": pd.DataFrame(calibrations)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if any(column in ("global_stay_index", "stay_id", "subject_id", "hadm_id") for column in table.columns):
            raise RuntimeError("An identifier column entered the public quadrature sensitivity")
        table.to_csv(args.output_dir / name, index=False)
    public = {"design": {"main_quadrature_nodes_per_dimension": 31, "sensitivity_nodes_per_dimension": 61,
                          "tau": 0.1, "landmark_hours": 12, "horizon_hours": [4, 8, 12],
                          "fixed_profile_penalty": [0.03, 0.1], "all_main_tuning_and_selections_unchanged": True,
                          "score_weighting": "mean within each stay, then equal weight across stays",
                          "tuning_precision_weighting": "mean observation change within each tuning stay, then equal weight across tuning stays"},
              "cohort_checks": cohort_checks, "fitting_validation": validation,
              "aggregate_tables": {name.removesuffix(".csv"): table.astype(object).where(pd.notna(table), None).to_dict(orient="records")
                                   for name, table in tables.items()}}
    json.dumps(public, allow_nan=False)
    write_json(args.output_dir / "clinical_quadrature_sensitivity_results.json", public)
    write_json(args.output_dir / "clinical_quadrature_sensitivity_schema.json", {
        "csv_fields": {name: list(table.columns) for name, table in tables.items()},
        "json_top_level_fields": list(public), "json_design_fields": list(public["design"])})
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
