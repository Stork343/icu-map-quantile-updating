from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import adaptive_candidate_specs
from expanded_family_core import candidate_specs
from run_adaptive_clinical import HORIZONS, aggregate
from run_submission_validation_extensions import write_json
from run_temporal_validation import paired_interval


def verify_metrics(frame: pd.DataFrame) -> None:
    if frame.empty or frame["global_stay_index"].duplicated().any():
        raise RuntimeError("Clinical assessment needs one row for each eligible stay")
    metrics = [column for column in frame if column.startswith(("loss_", "p_lt_", "p_le_"))]
    if not metrics or not np.all(np.isfinite(frame[metrics].to_numpy())):
        raise RuntimeError("Assessment metrics must all be finite")
    for column in metrics:
        values = frame[column].to_numpy()
        if np.any(values < -1e-12):
            raise RuntimeError("Loss and tail frequency metrics must be nonnegative")
        if column.startswith(("p_lt_", "p_le_")) and np.any(values > 1.0 + 1e-12):
            raise RuntimeError("Tail frequencies must not exceed one")
    for column in metrics:
        if column.startswith("p_lt_"):
            model = column.removeprefix("p_lt_")
            if np.any(frame[column].to_numpy() > frame[f"p_le_{model}"].to_numpy() + 1e-12):
                raise RuntimeError("Strict tail frequency exceeds weak tail frequency")


def join_random_slope(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    reference = pd.read_csv(path)
    required = ["global_stay_index", "late_obs"] + [f"{metric}_{model}" for model in
                ("random_slope_lqmm", "calibrated_random_slope_lqmm") for metric in ("loss", "p_lt", "p_le")]
    if any(column not in reference for column in required):
        raise RuntimeError("Random slope assessment columns differ from the declared interface")
    if reference["global_stay_index"].duplicated().any():
        raise RuntimeError("Random slope assessment contains repeated stays")
    if set(frame["global_stay_index"]) != set(reference["global_stay_index"]):
        raise RuntimeError("Random slope and profile assessment cohorts differ")
    aligned = reference.set_index("global_stay_index").loc[frame["global_stay_index"]]
    if "late_obs" not in frame or not np.array_equal(frame["late_obs"].to_numpy(), aligned["late_obs"].to_numpy()):
        raise RuntimeError("Random slope and profile later observation counts differ within stay")
    joined = frame.merge(reference[required].drop(columns="late_obs"), on="global_stay_index", validate="one_to_one", sort=False)
    if joined.shape[0] != frame.shape[0]:
        raise RuntimeError("A stay was lost while merging the clinical comparisons")
    verify_metrics(joined)
    return joined


def paired_rows(frame: pd.DataFrame, horizon: float) -> list[dict[str, object]]:
    candidates = tuple(column.removeprefix("loss_") for column in frame.columns if column.startswith("loss_"))
    references = ("calibrated_q10", "original_selected", "stump_original",
                  "random_slope_lqmm", "calibrated_random_slope_lqmm")
    rows = []
    for candidate in candidates:
        for reference in references:
            if candidate == reference:
                continue
            left, right = (frame[f"loss_{model}"].to_numpy(dtype=float) for model in (candidate, reference))
            rows.append({"scope": "temporal_2017_2022", "horizon_hours": horizon,
                         "candidate": candidate, "reference": reference, **paired_interval(left, right),
                         "interval_type": "pointwise paired normal 95 percent",
                         "relative_loss_reduction_percent": 100.0 * (1.0 - float(np.mean(left)) / float(np.mean(right)))})
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Verify and combine the complete adaptive clinical comparison")
    parser.add_argument("--clinical-dir", type=Path, default=root / ".review-work" / "adaptive" / "clinical")
    parser.add_argument("--random-slope-dir", type=Path, default=root / ".review-work" / "adaptive" / "random_slope")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel-validation-report", type=Path)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir is not None else args.clinical_dir / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, selections, solver_rows, cohort_rows = [], [], [], [], []
    tuning_rows, calibration_rows, selector_rows = [], [], []
    random_parameter_rows, random_validation_rows = [], []
    population_rows = []
    expected_counts = {4: 59969, 8: 60241, 12: 60310}
    candidate_grid = {str(spec["key"]): {**spec, "basis": spec.get("basis", "level_slope")}
                      for spec in candidate_specs() + adaptive_candidate_specs()}
    for horizon in HORIZONS:
        horizon_dir = args.clinical_dir / f"horizon_{int(horizon)}"
        nested = []
        for partition in [f"fold_{number}" for number in range(1, 6)] + ["temporal"]:
            work = horizon_dir / ("temporal" if partition == "temporal" else f"nested/{partition}")
            metadata = json.loads((work / "partition_results.json").read_text(encoding="utf-8"))
            if len(metadata["all_tuning_rules"]) != 200:
                raise RuntimeError("The frozen clinical tuning dictionary must contain 200 canonical rules")
            frame = pd.read_csv(work / "assessment_metrics.csv")
            verify_metrics(frame)
            frame = join_random_slope(frame, args.random_slope_dir / partition / f"assessment_horizon_{int(horizon)}.csv")
            reference_info = json.loads((args.random_slope_dir / partition / "results.json").read_text(encoding="utf-8"))
            validation = reference_info["fitting_validation"]
            if (not validation["validation_passed"] or not validation["group_membership_matches"]
                    or not validation["dimensions_match"] or validation["nodes_per_dimension"] != 31
                    or validation["quadrature_nodes"] != 961 or validation["n_fitting_stays"] != metadata["fit_stays"]):
                raise RuntimeError("Random-slope fitting input validation is incomplete")
            reference_horizon = reference_info["horizons"][str(int(horizon))]
            if (reference_horizon["assessment_stays"] != metadata["assessment_stays"]
                    or reference_horizon["tuning_stays"] != metadata["tuning_stays"]):
                raise RuntimeError("Random-slope horizon sample allocations differ")
            reference_calibration = reference_horizon["calibration"]
            calibration_rows.append({"horizon_hours": horizon, "partition": partition,
                                     "base_candidate": "random_slope_lqmm",
                                     "intercept": reference_calibration["intercept"],
                                     "slope": reference_calibration["slope"],
                                     "calibration_stationarity": reference_calibration["stationarity"],
                                     "calibration_relative_gap": reference_calibration["relative_gap"]})
            if horizon == HORIZONS[0]:
                population_terms = ("x_intercept", "age_z", "male", "emergency_or_urgent")
                population_rows.extend({"partition": partition, "term": term, "estimate": estimate}
                                       for term, estimate in zip(population_terms, metadata["population_coefficients"]))
                population_rows.extend({"partition": partition, "term": term, "estimate": estimate}
                                       for term, estimate in metadata["age_standardization"].items())
                for term, estimate in reference_info["parameters"].items():
                    if not np.isfinite(float(estimate)):
                        raise RuntimeError("Nonfinite public random-slope parameter")
                    random_parameter_rows.append({"partition": partition, "term": term, "estimate": estimate})
                allowed_validation_fields = ("n_fitting_stays", "n_observations", "tau", "nodes_per_dimension", "quadrature_nodes",
                                             "maximum_outcome_difference", "maximum_covariate_difference", "maximum_random_design_difference",
                                             "maximum_relative_parameter_difference", "group_membership_matches", "dimensions_match",
                                             "lower_iterations", "upper_iterations", "log_likelihood", "validation_passed")
                random_validation_rows.append({"partition": partition,
                                               **{name: validation[name] for name in allowed_validation_fields}})
            if frame.shape[0] != metadata["assessment_stays"]:
                raise RuntimeError("Assessment count differs from the completed partition metadata")
            scope = "temporal_2017_2022" if partition == "temporal" else f"nested_{partition}"
            summaries.append(aggregate(frame, scope, horizon))
            cohort_rows.append({"horizon_hours": horizon, "partition": partition,
                                "assessment_later_observations": int(frame["late_obs"].sum()),
                                "random_slope_later_count_mismatches": 0,
                                **{name: metadata[name] for name in ("fit_stays", "source_tuning_stays", "source_assessment_stays",
                                                                   "tuning_stays", "assessment_stays", "excluded_no_horizon_tuning_record",
                                                                   "excluded_no_horizon_assessment_record")}})
            for model, rule in metadata["selected_rules"].items():
                calibration = metadata["calibrations"][str(rule["key"])]
                spec = candidate_grid.get(str(rule["key"]), {"basis": "index_summary"})
                selections.append({"horizon_hours": horizon, "partition": partition, "model": model,
                                   "rule_id": rule["id"], "base_candidate": rule["key"],
                                   "calibrated": bool(rule["calibrated"]), "tuning_loss": rule["tuning_loss"],
                                   "applied_intercept": calibration["intercept"] if rule["calibrated"] else 0.0,
                                   "applied_slope": calibration["slope"] if rule["calibrated"] else 1.0,
                                   **{name: spec.get(name) for name in ("basis", "lambda_level", "lambda_slope", "lambda_base", "gamma", "theta")}})
            for key, calibration in metadata["calibrations"].items():
                calibration_rows.append({"horizon_hours": horizon, "partition": partition,
                                         "base_candidate": key, "intercept": calibration["intercept"],
                                         "slope": calibration["slope"]})
            for rule in metadata["all_tuning_rules"]:
                calibration = metadata["calibrations"][str(rule["key"])]
                tuning_rows.append({"horizon_hours": horizon, "partition": partition,
                                    "rule_id": rule["id"], "base_candidate": rule["key"],
                                    "calibrated": bool(rule["calibrated"]), "tuning_loss": rule["tuning_loss"],
                                    "applied_intercept": calibration["intercept"] if rule["calibrated"] else 0.0,
                                    "applied_slope": calibration["slope"] if rule["calibrated"] else 1.0})
            selector = metadata["selector"]
            selector_rows.append({"horizon_hours": horizon, "partition": partition,
                                  "feature_index": selector["feature"], "threshold": selector["threshold"],
                                  "left_rule": selector["left_rule_name"], "right_rule": selector["right_rule_name"],
                                  "tuning_loss": selector["tuning_loss"],
                                  "training_quartile_cutpoints": selector["training_quartile_cutpoints"]})
            for row in metadata["tuning_solver_diagnostics"]:
                if (row["maxima"]["projected_kkt_violation"] > 5e-7
                        or row["maxima"]["stationarity_violation"] > 1e-10):
                    raise RuntimeError("A completed tuning profile violates the acceptance thresholds")
                solver_rows.append({"horizon_hours": horizon, "partition": partition,
                                    "phase": "tuning", "candidate": row["spec"]["key"],
                                    "n_stays": row["n_stays"], **row["maxima"],
                                    "kkt_limit": 5e-7, "stationarity_limit": 1e-10,
                                    "all_solutions_certified": True})
            for key, info in metadata["assessment_solver_diagnostics"].items():
                gap = info.get("duality_gap", info.get("max_duality_gap"))
                relative_gap = info.get("relative_duality_gap", info.get("max_relative_duality_gap"))
                kkt = info.get("projected_kkt_violation", info.get("max_projected_kkt_violation"))
                stationarity = info.get("stationarity_violation")
                if kkt is not None and float(kkt) > 5e-7:
                    raise RuntimeError("Assessment profile KKT certificate exceeds its declared limit")
                if stationarity is not None and float(stationarity) > 1e-10:
                    raise RuntimeError("Assessment profile stationarity certificate exceeds its declared limit")
                solver_rows.append({"horizon_hours": horizon, "partition": partition, "phase": "assessment",
                                    "candidate": key, "n_stays": metadata["assessment_stays"],
                                    "duality_gap": gap, "relative_duality_gap": relative_gap,
                                    "projected_kkt_violation": kkt, "stationarity_violation": stationarity,
                                    "kkt_limit": 5e-7, "stationarity_limit": 1e-10,
                                    "all_solutions_certified": True})
            if partition == "temporal":
                comparisons.extend(paired_rows(frame, horizon))
            else:
                nested.append(frame)
        combined = pd.concat(nested, ignore_index=True)
        verify_metrics(combined)
        if combined.shape[0] != expected_counts[int(horizon)]:
            raise RuntimeError("Pooled nested cohort does not reproduce the horizon-qualified source count")
        summaries.append(aggregate(combined, "nested_pooled", horizon))
    selector_frame = pd.DataFrame(selector_rows)
    selector_frame["training_quartile_cutpoints"] = selector_frame["training_quartile_cutpoints"].map(json.dumps)
    frames = {
        "clinical_all_model_summary.csv": pd.concat(summaries, ignore_index=True),
        "clinical_temporal_paired_comparisons.csv": pd.DataFrame(comparisons),
        "clinical_selected_rules.csv": pd.DataFrame(selections),
        "clinical_solver_diagnostics.csv": pd.DataFrame(solver_rows),
        "clinical_horizon_cohorts.csv": pd.DataFrame(cohort_rows),
        "clinical_all_tuning_rules.csv": pd.DataFrame(tuning_rows),
        "clinical_affine_calibrations.csv": pd.DataFrame(calibration_rows),
        "clinical_stump_rules.csv": selector_frame,
        "clinical_random_slope_parameters.csv": pd.DataFrame(random_parameter_rows),
        "clinical_random_slope_fitting_validation.csv": pd.DataFrame(random_validation_rows),
        "clinical_population_parameters.csv": pd.DataFrame(population_rows),
    }
    sensitivity = pd.read_csv(args.clinical_dir / "diagnostics" / "affine_q10_tie_diagnostics.csv")
    if (sensitivity.shape[0] != 36 or sensitivity.duplicated(["partition", "model", "horizon_hours"]).any()
            or set(sensitivity["model"]) != {"calibrated_q10", "vector_direct"}
            or set(sensitivity["horizon_hours"]) != set(HORIZONS)
            or not np.all(sensitivity["absolute_tolerance"] == 1e-10)):
        raise RuntimeError("The fixed rounding precision sensitivity must cover both rules in all 18 partitions")
    if (np.any(sensitivity["tolerance_strict_rate"] > sensitivity["original_strict_rate"] + 1e-12)
            or np.any(sensitivity["tolerance_weak_rate"] < sensitivity["original_weak_rate"] - 1e-12)):
        raise RuntimeError("A numerical tolerance bracket does not contain its original bracket")
    frames["clinical_rounding_precision_sensitivity.csv"] = sensitivity
    precision_dir = args.clinical_dir / "diagnostics" / "solver_precision"
    precision_frame = pd.read_csv(precision_dir / "profile_solver_precision.csv")
    if precision_frame.shape[0] != 9 or set(precision_frame["absolute_tolerance_mmhg"]) != {0.0, 1e-10, 1e-8}:
        raise RuntimeError("The fixed-profile solver precision comparison is incomplete")
    precision_result = json.loads((precision_dir / "profile_solver_precision.json").read_text(encoding="utf-8"))
    frames["clinical_fixed_profile_solver_precision.csv"] = precision_frame
    parallel_path = args.parallel_validation_report if args.parallel_validation_report is not None else args.clinical_dir.parent / "parallel_lqmm_validation.json"
    parallel_report = json.loads(parallel_path.read_text(encoding="utf-8"))
    difference_fields = ("max_theta_difference", "scale_difference", "log_likelihood_difference", "max_blp_difference")
    subset = parallel_report["real_clinical_subset_comparison"]
    if (parallel_report.get("equivalence_passed") is not True or parallel_report["equivalence_tolerance"] != 1e-7
            or len(parallel_report["official_namespace_likelihood_checks"]) < 5
            or any(float(subset[name]) > 1e-7 or not np.isfinite(float(subset[name])) for name in difference_fields)
            or any(float(row["absolute_difference"]) > 1e-7 for row in parallel_report["official_namespace_likelihood_checks"])):
        raise RuntimeError("Official and parallel lqmm likelihood equivalence has not passed the declared checks")
    parallel_fields = ("lqmm_version", "full_data_stays", "full_data_observations", "nodes_per_dimension", "quadrature_nodes",
                       "workers", "official_namespace_likelihood_checks", "real_clinical_subset_comparison",
                       "equivalence_tolerance", "equivalence_passed", "unchanged_control")
    public_parallel = {name: parallel_report[name] for name in parallel_fields}
    for name, frame in frames.items():
        if any("stay_index" in column or column in ("stay_id", "subject_id", "hadm_id") for column in frame.columns):
            raise RuntimeError("An identifier column entered the public aggregate export")
        frame.to_csv(output / name, index=False)
    verification = {
        "status": "passed", "horizons": list(HORIZONS), "completed_partitions": 18,
        "complete_nested_coverage": expected_counts, "random_slope_exact_cohort_match": True,
        "random_slope_late_observation_counts_match": True,
        "random_slope_actual_fitting_inputs_verified": True,
        "all_metrics_finite_and_within_bounds": True,
        "public_files_contain_no_stay_identifiers": True,
        "max_adaptive_tuning_kkt": max(row["projected_kkt_violation"] for row in solver_rows if row["phase"] == "tuning"),
        "max_adaptive_tuning_stationarity": max(row["stationarity_violation"] for row in solver_rows if row["phase"] == "tuning"),
        "all_tuning_rules": len(tuning_rows),
        "rounding_precision_sensitivity_rows": int(sensitivity.shape[0]),
        "parallel_likelihood_equivalence_verified": True,
    }
    write_json(output / "clinical_verification.json", verification)
    public_tables = {name.removesuffix(".csv"): frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
                     for name, frame in frames.items()}
    json.dumps(public_tables, allow_nan=False)
    public = {"design": {"tau": 0.1, "landmark_hours": 12.0, "horizons": list(HORIZONS),
                          "source_cohort": 60310, "adaptive_base_candidates": 42,
                          "original_canonical_rules": 116, "expanded_canonical_rules": 200,
                          "forecast_clipping": False, "kappa": 1.0 / 16.0,
                          "candidate_grid": list(candidate_grid.values()),
                          "selector_features": ["rank_half_q10_change", "log_index_count", "normalized_time_variance"]},
              "verification": verification,
              "aggregate_tables": public_tables}
    write_json(output / "clinical_public_results.json", public)
    write_json(output / "clinical_fixed_profile_solver_precision.json", precision_result)
    write_json(output / "clinical_parallel_likelihood_validation.json", public_parallel)
    write_json(output / "clinical_public_schema.json", {
        "csv_fields": {name: list(frame.columns) for name, frame in frames.items()},
        "json_top_level_fields": {"clinical_public_results.json": list(public),
                                  "clinical_verification.json": list(verification),
                                  "clinical_fixed_profile_solver_precision.json": list(precision_result),
                                  "clinical_parallel_likelihood_validation.json": list(public_parallel)},
        "json_design_fields": list(public["design"]),
        "json_aggregate_table_fields_follow_csv_fields": True,
        "public_content": "Aggregate cohort counts, frozen rule specifications, calibration coefficients, risk summaries and numerical certificates",
    })


if __name__ == "__main__":
    main()
