from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import osqp

from expanded_family_core import aggregate_metrics
from expanded_profile_qp import QP_SETTINGS, solve_profile_qp
from run_submission_validation_extensions import CALIBRATED_SUMMARIES, write_json
from run_temporal_validation import paired_interval


def complete_candidate_frame(grid: pd.DataFrame, calibrations: dict[str, object], scope: str) -> pd.DataFrame:
    rows = []
    for model in CALIBRATED_SUMMARIES:
        calibration = calibrations[model]
        rows.append({"key": model, "basis": "clinical_summary", "lambda_level": None, "lambda_slope": None,
                     "raw_tuning_loss": None, "affine_tuning_loss": calibration["tuning_loss"],
                     "affine_intercept": calibration["intercept"], "affine_slope": calibration["slope"]})
    return pd.DataFrame([*grid.to_dict(orient="records"), *rows]).assign(scope=scope)


def verify_solver_grid(path: Path) -> dict[str, object]:
    grid = json.loads(path.read_text(encoding="utf-8"))
    maximum_iterations = 0
    total_profiles = 0
    for row in grid:
        if row["basis"] != "level_slope":
            continue
        if row["solver_method"] != "osqp_fenchel_dual_qp_polished" or row["solver_settings"] != QP_SETTINGS:
            raise RuntimeError("Candidate QPs use different numerical solver settings")
        if sum(row["solver_status_counts"].values()) != int(row["n_stays"]) or sum(row["solver_polishing_status_counts"].values()) != int(row["n_stays"]):
            raise RuntimeError("Candidate QP status counts do not cover every tuning stay")
        if float(row["max_projected_kkt_violation"]) > 5e-7 or int(row["solver_iterations_max"]) > QP_SETTINGS["max_iter"]:
            raise RuntimeError("Candidate QP numerical limits are inconsistent")
        maximum_iterations = max(maximum_iterations, int(row["solver_iterations_max"]))
        total_profiles += int(row["n_stays"])
    return {"all_candidate_solver_settings_verified": True, "tuning_vector_profiles": total_profiles,
            "maximum_iterations": maximum_iterations}


def verify_existing_components(records: pd.DataFrame, legacy: pd.DataFrame) -> dict[str, float]:
    merged = records.merge(legacy, left_on="global_stay_index", right_on="stay_index", suffixes=("_new", "_old"), validate="one_to_one")
    if merged.shape[0] != records.shape[0]:
        raise RuntimeError("Frozen original predictions do not cover the new nested assessment cohort")
    np.testing.assert_array_equal(merged["outer_fold_new"], merged["outer_fold_old"])
    result = {}
    for new_name, old_name in {"population": "population", "scalar_profile": "primary_level_update", "raw_q10": "raw_q10", **{name: name for name in CALIBRATED_SUMMARIES}}.items():
        difference = merged[f"loss_{new_name}_new" if f"loss_{new_name}_new" in merged else f"loss_{new_name}"] - merged[f"loss_{old_name}_old" if f"loss_{old_name}_old" in merged else f"loss_{old_name}"]
        maximum = float(np.max(np.abs(difference)))
        if maximum > 1e-8:
            raise RuntimeError(f"Existing candidate changed its assessment losses: {new_name}")
        result[new_name] = maximum
    return result


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Verify complete-family aggregate results and population-fit provenance")
    parser.add_argument("--analysis-dir", type=Path, default=repo.parent / ".review-work" / "strengthening" / "expanded_family")
    parser.add_argument("--nested-results-dir", type=Path, default=repo / "recovery_20260822" / "gate1_approved_run" / "validation_extensions")
    parser.add_argument("--temporal-results-file", type=Path, default=repo / "recovery_20260822" / "gate1_approved_run" / "temporal" / "temporal_validation_results.json")
    args = parser.parse_args()
    config = json.loads((args.analysis_dir / "analysis_config.json").read_text(encoding="utf-8"))
    legacy = json.loads((args.nested_results_dir / "submission_validation_extensions_results.json").read_text(encoding="utf-8"))
    for key in ("seed", "age_definition", "tau", "index_hours", "outer_folds"):
        if legacy["settings"][key] != config[key]:
            raise RuntimeError(f"Original analysis configuration differs for {key}")
    verification, candidate_frames, partition_frames = {}, [], []
    real_solver_checks = []
    for path in sorted(args.analysis_dir.glob("nested/fold_*/tuning_profiles/solver_diagnostics/*.npz")):
        inputs = np.load(path, allow_pickle=False)
        _, diagnostics = solve_profile_qp(inputs["residual"], inputs["design"], float(inputs["tau"]), inputs["penalties"])
        real_solver_checks.append({"input": str(path.relative_to(args.analysis_dir)), "certified": diagnostics["success"],
                                   "duality_gap": diagnostics["duality_gap"], "projected_kkt_violation": diagnostics["projected_kkt_violation"],
                                   "solver_status": diagnostics["status"], "polishing_status": diagnostics["polishing_status"]})
    verification["local_real_solver_regression"] = real_solver_checks
    for fold in range(1, 6):
        directory = args.analysis_dir / "nested" / f"fold_{fold}"
        source_dir = args.nested_results_dir / "work" / "nested_crossfit" / f"fold_{fold}"
        source = json.loads((source_dir / "fold_metadata.json").read_text(encoding="utf-8"))
        coefficients = json.loads((source_dir / "population" / "population_coefficients.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "partition_results.json").read_text(encoding="utf-8"))
        if int(source["outer_fold"]) != fold or coefficients["tau"] != config["tau"]:
            raise RuntimeError("Fold or quantile source is inconsistent")
        if coefficients["predictors"] != ["x_intercept", "age_z", "male", "emergency_or_urgent"]:
            raise RuntimeError("Source population covariates differ")
        np.testing.assert_array_equal(coefficients["coefficients"], metadata["population_coefficients"])
        np.testing.assert_array_equal(source["population_coefficients"], metadata["population_coefficients"])
        for name in ("age_mean_training_split", "age_sd_training_split"):
            np.testing.assert_allclose(source["age_standardization"][name], metadata["age_standardization"][name], rtol=1e-12, atol=1e-12)
        grid = pd.read_csv(directory / "tuning_grid.csv")
        if grid.shape[0] != 56 or (grid["basis"] == "level_slope").sum() != 48:
            raise RuntimeError("The complete fixed tuning grid was not evaluated")
        rules = metadata["selected_rules"]
        if float(rules["selected_family"]["tuning_loss"]) != min(float(rule["tuning_loss"]) for model, rule in rules.items() if model != "selected_family"):
            raise RuntimeError("The selected family rule does not minimize tuning loss")
        metrics = pd.read_csv(directory / "assessment_metrics.csv")
        metrics["outer_fold"] = fold
        np.testing.assert_allclose(metrics["loss_selected_family"], metrics[f"loss_{rules['selected_family']['selected_model']}"], atol=1e-12)
        if rules["calibrated_profile"]["key"] == "scalar_0":
            for metric in ("loss", "p_lt", "p_le", "prediction_mean"):
                np.testing.assert_allclose(metrics[f"{metric}_calibrated_profile"], metrics[f"{metric}_calibrated_q10"], atol=1e-12)
        candidate_frames.append(complete_candidate_frame(grid, metadata["calibrations"], f"nested_fold_{fold}"))
        partition_frames.append(metrics)
        verification[f"fold_{fold}"] = {"population_reuse_verified": True, "complete_grid_rows": grid.shape[0],
                                          "selected_model": rules["selected_family"]["selected_model"],
                                          **verify_solver_grid(directory / "tuning_grid.json")}
    nested = pd.concat(partition_frames, ignore_index=True).sort_values("global_stay_index")
    if nested.shape[0] != 60310 or nested["global_stay_index"].nunique() != 60310:
        raise RuntimeError("Nested assessment records are not exhaustive and unique")
    verification["existing_component_maximum_loss_difference"] = verify_existing_components(
        nested, pd.read_csv(args.nested_results_dir / "nested_crossfit_stay_level_predictions.csv"))
    aggregate_dir = args.analysis_dir / "aggregate"
    summary = pd.read_csv(aggregate_dir / "nested_model_summary.csv")
    if any("ci95" in name for name in summary):
        raise RuntimeError("Nested risk results contain a fixed-rule confidence interval")
    expected = aggregate_metrics(nested, "nested_pooled", .1).set_index("model")
    observed = summary.loc[summary["scope"] == "nested_pooled"].set_index("model")
    np.testing.assert_allclose(expected["mean_stay_level_check_loss"], observed["mean_stay_level_check_loss"], atol=1e-12)
    temporal_metadata = json.loads((args.analysis_dir / "temporal" / "partition_results.json").read_text(encoding="utf-8"))
    source = json.loads(args.temporal_results_file.read_text(encoding="utf-8"))
    if source["design"]["assignment_rule"] != "aligned_full_containment":
        raise RuntimeError("Temporal source uses a different time assignment")
    np.testing.assert_array_equal(source["population_coefficients"], temporal_metadata["population_coefficients"])
    for key in ("fit_stays", "tuning_stays", "assessment_stays"):
        if int(source["design"][key]) != int(temporal_metadata[key]):
            raise RuntimeError(f"Temporal source differs for {key}")
    temporal = pd.read_csv(args.analysis_dir / "temporal_assessment_metrics.csv")
    if temporal.shape[0] != 22005 or temporal["global_stay_index"].nunique() != 22005:
        raise RuntimeError("Temporal assessment records differ from the fixed cohort")
    comparisons = pd.read_csv(aggregate_dir / "temporal_paired_comparisons.csv")
    for row in comparisons.itertuples(index=False):
        interval = paired_interval(temporal[f"loss_{row.candidate}"].to_numpy(), temporal[f"loss_{row.reference}"].to_numpy())
        np.testing.assert_allclose([row.mean_difference, row.ci95_low, row.ci95_high],
                                   [interval["mean_difference"], interval["ci95_low"], interval["ci95_high"]], atol=1e-12)
    candidate_frames.append(complete_candidate_frame(pd.read_csv(args.analysis_dir / "temporal" / "tuning_grid.csv"),
                                                    temporal_metadata["calibrations"], "temporal_2017_2022"))
    pd.concat(candidate_frames, ignore_index=True).to_csv(aggregate_dir / "complete_tuning_grid.csv", index=False)
    verification["temporal"] = {"population_reuse_verified": True, "n_assessment_stays": temporal.shape[0],
                                  "paired_normal_intervals_verified": comparisons.shape[0],
                                  **verify_solver_grid(args.analysis_dir / "temporal" / "tuning_grid.json")}
    verification["public_aggregate_fields"] = True
    for path in aggregate_dir.glob("*.csv"):
        columns = pd.read_csv(path, nrows=0).columns
        if any(column in {"stay_id", "subject_id", "global_stay_index", "stay_index"} for column in columns):
            raise RuntimeError(f"Public aggregate file contains an individual-record field: {path.name}")
    aggregate_json = aggregate_dir / "expanded_family_results_all.json"
    payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
    for offset, partition in enumerate(payload["nested_partitions"], start=1):
        partition["population_fit_source"] = f"work/nested_crossfit/fold_{offset}/fold_metadata.json"
    payload["temporal_partition"]["population_fit_source"] = args.temporal_results_file.name
    write_json(aggregate_json, payload)
    write_json(aggregate_dir / "validation_checks.json", verification)
    r_environment = os.environ.copy()
    r_environment["TMPDIR"] = "."
    r_versions = subprocess.run(["Rscript", "-e", "cat(paste(getRversion(),as.character(packageVersion('quantreg')),as.character(packageVersion('data.table')),sep='|'))"],
                                cwd=args.analysis_dir.resolve(), env=r_environment, check=True, capture_output=True, text=True).stdout.strip().split("|")
    runtime = {"python": platform.python_version(), "python_executable": sys.executable,
               "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "osqp": osqp.__version__,
               "R": r_versions[0], "quantreg": r_versions[1], "data.table": r_versions[2], "osqp_settings": QP_SETTINGS,
               "partition_compute_seconds": sum(float(item["runtime_seconds"]) for item in payload["nested_partitions"]) + float(payload["temporal_partition"]["runtime_seconds"])}
    write_json(aggregate_dir / "runtime_manifest.json", runtime)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
