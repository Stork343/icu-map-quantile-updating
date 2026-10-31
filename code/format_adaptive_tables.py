import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
HORIZONS = (4, 8, 12)
SETTINGS = (
    "informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress",
)
SIMULATION_METHODS = (
    "population", "calibrated_q10", "original_selected", "geometry_selected",
    "conditioned_selected", "adaptive_selected", "expanded_selected", "stump_original",
)
SIMULATION_CONTRASTS = (
    ("expanded_selected", "original_selected"),
    ("expanded_selected", "stump_original"),
    ("expanded_selected", "calibrated_q10"),
    ("adaptive_selected", "original_selected"),
    ("adaptive_selected", "geometry_selected"),
    ("adaptive_selected", "conditioned_selected"),
    ("original_selected", "calibrated_q10"),
)
CLINICAL_MODELS = (
    "population", "raw_q10", "calibrated_q10", "original_selected", "expanded_selected",
    "scalar_direct", "scalar_affine", "vector_direct", "vector_affine",
    "geometry_direct", "geometry_affine", "history_geometry_direct", "history_geometry_affine",
    "adaptive_direct", "adaptive_affine", "stump_original", "random_slope_lqmm",
    "calibrated_random_slope_lqmm",
)
MAIN_CLINICAL_MODELS = (
    "population", "calibrated_q10", "vector_direct", "random_slope_lqmm", "calibrated_random_slope_lqmm",
    "stump_original", "expanded_selected",
)
CLINICAL_SCOPES = ("nested_pooled", "temporal_2017_2022")
MAIN_COMPARISONS = {
    "original_selected": "expanded_original",
    "calibrated_q10": "expanded_calibrated_q10",
    "stump_original": "expanded_stump",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_columns(frame, columns, source):
    missing = set(columns).difference(frame.columns)
    require(not missing, f"Missing columns in {source}: {sorted(missing)}")


def numeric(frame, columns, source):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    require(np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all(), f"Nonfinite values in {source}")


def integer_column(frame, column, source):
    values = pd.to_numeric(frame[column], errors="raise")
    require(np.isfinite(values.to_numpy(dtype=float)).all(), f"Nonfinite {column} in {source}")
    require((values == np.floor(values)).all(), f"Noninteger {column} in {source}")
    frame[column] = values.astype("int64")


def exact_identifiers(frame, column, expected, source):
    require(not frame[column].isna().any(), f"Missing {column} in {source}")
    observed = set(frame[column])
    require(observed == set(expected), f"Unexpected {column} in {source}: {sorted(observed)}")


def unique_rows(frame, columns, source):
    require(not frame.duplicated(list(columns)).any(), f"Duplicate rows in {source} for {columns}")


def close_values(actual, expected, columns, keys, source):
    joined = actual.merge(expected, on=list(keys), how="outer", suffixes=("_actual", "_expected"), indicator=True, validate="one_to_one")
    require((joined["_merge"] == "both").all(), f"Summary keys differ in {source}")
    for column in columns:
        left = joined[f"{column}_actual"].to_numpy(dtype=float)
        right = joined[f"{column}_expected"].to_numpy(dtype=float)
        require(np.allclose(left, right, rtol=1e-8, atol=1e-10), f"Summary mismatch in {source}: {column}")


def check_brackets(frame, lower, upper, violation, source):
    require(((frame[lower] >= 0) & (frame[upper] <= 1)).all(), f"Probability outside [0,1] in {source}")
    require((frame[lower] <= frame[upper] + 1e-12).all(), f"Reversed probability brackets in {source}")
    expected = np.maximum.reduce((frame[lower].to_numpy() - 0.10, 0.10 - frame[upper].to_numpy(), np.zeros(len(frame))))
    require(np.allclose(frame[violation], expected, rtol=1e-8, atol=1e-10), f"Bracket violation mismatch in {source}")


def mcse(values):
    return values.std(ddof=1) / math.sqrt(len(values))


def ordered(frame, columns, categories):
    result = frame.copy()
    for column, values in categories.items():
        result[column] = pd.Categorical(result[column], categories=values, ordered=True)
    return result.sort_values(list(columns)).reset_index(drop=True)


def validate_simulation(directory, settings=SETTINGS, replicate_ids=tuple(range(1, 201))):
    directory = Path(directory)
    paths = {name: directory / f"{name}.csv" for name in (
        "methods_replicates", "paired_replicates", "method_summary", "paired_summary",
    )}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    methods, pairs = frames["methods_replicates"], frames["paired_replicates"]
    summary, paired_summary = frames["method_summary"], frames["paired_summary"]
    method_keys = ("setting", "horizon", "method", "scope")
    pair_keys = ("setting", "horizon", "first", "second", "scope")
    require_columns(methods, (*method_keys, "replicate", "n_stays", "loss", "p_lt", "p_le", "violation", "within_sample_se"), paths["methods_replicates"])
    require_columns(pairs, (*pair_keys, "replicate", "difference", "within_sample_se"), paths["paired_replicates"])
    require_columns(summary, (*method_keys, "replicates", "loss", "loss_mcse", "p_lt", "p_lt_mcse", "p_le", "p_le_mcse", "violation", "violation_mcse", "mean_bracket_violation"), paths["method_summary"])
    require_columns(paired_summary, (*pair_keys, "replicates", "mean_difference", "mcse", "mean_within_sample_se", "mc_lower", "mc_upper"), paths["paired_summary"])
    for name, frame in frames.items():
        integer_column(frame, "horizon", paths[name])
        exact_identifiers(frame, "setting", settings, paths[name])
        exact_identifiers(frame, "horizon", HORIZONS, paths[name])
        exact_identifiers(frame, "scope", ("assessment", "reference"), paths[name])
    for name, frame in (("methods_replicates", methods), ("paired_replicates", pairs)):
        integer_column(frame, "replicate", paths[name])
    numeric(methods, ("n_stays", "loss", "p_lt", "p_le", "violation", "within_sample_se"), paths["methods_replicates"])
    numeric(pairs, ("difference", "within_sample_se"), paths["paired_replicates"])
    method_metrics = ("replicates", "loss", "loss_mcse", "p_lt", "p_lt_mcse", "p_le", "p_le_mcse", "violation", "violation_mcse", "mean_bracket_violation")
    pair_metrics = ("replicates", "mean_difference", "mcse", "mean_within_sample_se", "mc_lower", "mc_upper")
    numeric(summary, method_metrics, paths["method_summary"])
    numeric(paired_summary, pair_metrics, paths["paired_summary"])
    exact_identifiers(methods, "method", SIMULATION_METHODS, paths["methods_replicates"])
    exact_identifiers(summary, "method", SIMULATION_METHODS, paths["method_summary"])
    unique_rows(methods, (*method_keys, "replicate"), paths["methods_replicates"])
    unique_rows(pairs, (*pair_keys, "replicate"), paths["paired_replicates"])
    unique_rows(summary, method_keys, paths["method_summary"])
    unique_rows(paired_summary, pair_keys, paths["paired_summary"])
    expected_ids = set(replicate_ids)
    require(len(expected_ids) >= 2, "At least two independent replicates are required")
    for name, frame, keys in (("methods_replicates", methods, method_keys), ("paired_replicates", pairs, pair_keys)):
        for key, group in frame.groupby(list(keys), observed=True):
            require(set(group["replicate"]) == expected_ids, f"Incomplete replicate identifiers in {paths[name]}: {key}")
    expected_method_rows = len(settings) * len(HORIZONS) * len(SIMULATION_METHODS) * 2
    require(len(summary) == expected_method_rows, "Incomplete simulation summary design")
    require(len(methods) == expected_method_rows * len(expected_ids), "Incomplete simulation replicate design")
    for key, group in pairs.groupby(["setting", "horizon", "scope"], observed=True):
        observed_contrasts = set(group[["first", "second"]].itertuples(index=False, name=None))
        require(observed_contrasts == set(SIMULATION_CONTRASTS), f"Unexpected paired design: {key}")
    require(len(paired_summary) == len(settings) * len(HORIZONS) * len(SIMULATION_CONTRASTS) * 2, "Incomplete paired summary design")
    pair_record_keys = ["setting", "horizon", "scope", "replicate"]
    method_losses = methods[pair_record_keys + ["method", "loss"]]
    linked_pairs = pairs.merge(
        method_losses.rename(columns={"method": "first", "loss": "first_loss"}),
        on=pair_record_keys + ["first"], how="left", validate="many_to_one", indicator=True,
    )
    require((linked_pairs["_merge"] == "both").all(), "A paired first rule lacks its scored replicate")
    linked_pairs = linked_pairs.drop(columns="_merge").merge(
        method_losses.rename(columns={"method": "second", "loss": "second_loss"}),
        on=pair_record_keys + ["second"], how="left", validate="many_to_one", indicator=True,
    )
    require((linked_pairs["_merge"] == "both").all(), "A paired second rule lacks its scored replicate")
    require(np.allclose(linked_pairs["difference"], linked_pairs["first_loss"] - linked_pairs["second_loss"], rtol=1e-8, atol=1e-10), "Paired differences disagree with their signed replicate losses")
    expected_stays = methods["scope"].map({"assessment": 120, "reference": 2000})
    require((methods["n_stays"] == expected_stays).all(), "Incorrect assessment or independent reference sample size")
    require((methods["loss"] >= 0).all() and (methods["within_sample_se"] >= 0).all(), "Negative simulation loss or standard error")
    require((pairs["within_sample_se"] >= 0).all(), "Negative paired reference standard error")
    check_brackets(methods, "p_lt", "p_le", "violation", paths["methods_replicates"])
    expected_summary = methods.groupby(list(method_keys), as_index=False, observed=True).agg(
        replicates=("replicate", "size"), loss=("loss", "mean"), loss_mcse=("loss", mcse),
        p_lt=("p_lt", "mean"), p_lt_mcse=("p_lt", mcse), p_le=("p_le", "mean"), p_le_mcse=("p_le", mcse),
        violation=("violation", "mean"), violation_mcse=("violation", mcse),
    )
    expected_summary["mean_bracket_violation"] = np.maximum.reduce((
        expected_summary["p_lt"].to_numpy() - 0.10,
        0.10 - expected_summary["p_le"].to_numpy(), np.zeros(len(expected_summary)),
    ))
    close_values(summary, expected_summary, method_metrics, method_keys, paths["method_summary"])
    expected_pairs = pairs.groupby(list(pair_keys), as_index=False, observed=True).agg(
        replicates=("replicate", "size"), mean_difference=("difference", "mean"), mcse=("difference", mcse),
        mean_within_sample_se=("within_sample_se", "mean"),
    )
    expected_pairs["mc_lower"] = expected_pairs["mean_difference"] - 1.96 * expected_pairs["mcse"]
    expected_pairs["mc_upper"] = expected_pairs["mean_difference"] + 1.96 * expected_pairs["mcse"]
    close_values(paired_summary, expected_pairs, pair_metrics, pair_keys, paths["paired_summary"])
    return summary, paired_summary


def simulation_tables(summary, paired_summary, settings=SETTINGS):
    mixed_rows = []
    for setting in settings:
        for horizon in HORIZONS:
            row = {"setting": setting, "horizon_hours": horizon}
            for second, prefix in MAIN_COMPARISONS.items():
                selected = paired_summary.loc[(paired_summary["setting"] == setting) & (paired_summary["horizon"] == horizon) & (paired_summary["first"] == "expanded_selected") & (paired_summary["second"] == second) & (paired_summary["scope"] == "reference")]
                require(len(selected) == 1, f"Missing reference paired comparison: {setting}, {horizon}, {second}")
                row[prefix + "_mean"] = float(selected.iloc[0]["mean_difference"])
                row[prefix + "_mcse"] = float(selected.iloc[0]["mcse"])
            mixed_rows.append(row)
    outputs = {"adaptive_mixed_comparisons.csv": pd.DataFrame(mixed_rows)}
    keys = ["setting", "horizon", "method"]
    assessment = summary.loc[summary["scope"] == "assessment", keys + ["loss", "loss_mcse"]].rename(columns={"loss": "assessment_loss", "loss_mcse": "assessment_mcse"})
    reference = summary.loc[summary["scope"] == "reference", keys + ["loss", "loss_mcse", "violation", "mean_bracket_violation", "replicates"]].rename(columns={"loss": "reference_loss", "loss_mcse": "reference_mcse", "violation": "replicate_violation"})
    complete = assessment.merge(reference, on=keys, validate="one_to_one")
    for horizon in HORIZONS:
        table = complete.loc[complete["horizon"] == horizon].drop(columns="horizon")
        table = ordered(table, ("setting", "method"), {"setting": settings, "method": SIMULATION_METHODS})
        outputs[f"adaptive_simulation_d{horizon}.csv"] = table
    return outputs


def validate_clinical(frame, models=CLINICAL_MODELS, horizons=HORIZONS):
    frame = frame.copy()
    source = "clinical score summary"
    columns = ("scope", "horizon_hours", "model", "n_stays", "mean_loss", "p_lt", "p_le", "bracket_violation")
    require_columns(frame, columns, source)
    frame = frame.loc[frame["scope"].isin(CLINICAL_SCOPES), list(columns)].copy()
    integer_column(frame, "horizon_hours", source)
    integer_column(frame, "n_stays", source)
    numeric(frame, ("mean_loss", "p_lt", "p_le", "bracket_violation"), source)
    exact_identifiers(frame, "scope", CLINICAL_SCOPES, source)
    exact_identifiers(frame, "model", models, source)
    exact_identifiers(frame, "horizon_hours", horizons, source)
    unique_rows(frame, ("scope", "horizon_hours", "model"), source)
    require(len(frame) == len(CLINICAL_SCOPES) * len(horizons) * len(models), "Incomplete clinical model grid")
    require((frame["n_stays"] > 0).all() and (frame["mean_loss"] >= 0).all(), "Invalid clinical counts or losses")
    for key, group in frame.groupby(["scope", "horizon_hours"], observed=True):
        require(group["n_stays"].nunique() == 1, f"Clinical models use different cohorts: {key}")
    check_brackets(frame, "p_lt", "p_le", "bracket_violation", source)
    return frame


def clinical_tables(frame, models=CLINICAL_MODELS, main_models=MAIN_CLINICAL_MODELS):
    outputs = {}
    for horizon in HORIZONS:
        selected = frame.loc[frame["horizon_hours"] == horizon].drop(columns="horizon_hours")
        if selected.empty:
            continue
        outputs[f"adaptive_clinical_d{horizon}.csv"] = ordered(selected, ("scope", "model"), {"scope": CLINICAL_SCOPES, "model": models})
    rows = []
    for scope in CLINICAL_SCOPES:
        for model in main_models:
            row = {"scope": scope, "model": model}
            for horizon in HORIZONS:
                selected = frame.loc[(frame["scope"] == scope) & (frame["model"] == model) & (frame["horizon_hours"] == horizon)]
                require(len(selected) == 1, f"Missing clinical panel cell: {scope}, {model}, {horizon}")
                record = selected.iloc[0]
                row[f"n_d{horizon}"] = int(record["n_stays"])
                row[f"loss_d{horizon}"] = float(record["mean_loss"])
                row[f"violation_d{horizon}"] = float(record["bracket_violation"])
            rows.append(row)
    column_order = ["scope", "model", "n_d4", "n_d8", "n_d12", "loss_d4", "violation_d4", "loss_d8", "violation_d8", "loss_d12", "violation_d12"]
    outputs["adaptive_clinical_horizons.csv"] = pd.DataFrame(rows).reindex(columns=column_order)
    return outputs


def validate_numerics(sensitivity, precision):
    sensitivity = sensitivity.copy()
    precision = precision.copy()
    sensitivity_columns = (
        "partition", "model", "horizon_hours", "n_stays", "absolute_tolerance",
        "original_strict_rate", "original_weak_rate", "tolerance_strict_rate", "tolerance_weak_rate",
        "original_bracket_violation", "tolerance_bracket_violation", "original_mean_check_loss",
        "max_loss_reproduction_difference",
    )
    precision_columns = (
        "solver", "horizon_hours", "partition", "tau", "lambda_level", "lambda_slope",
        "n_stays", "n_observations", "absolute_tolerance_mmhg", "strict_rate", "weak_rate",
        "bracket_violation", "mean_check_loss",
    )
    require_columns(sensitivity, sensitivity_columns, "rounding sensitivity")
    require_columns(precision, precision_columns, "fixed-profile numerical comparison")
    for frame, source in ((sensitivity, "rounding sensitivity"), (precision, "fixed-profile numerical comparison")):
        integer_column(frame, "horizon_hours", source)
        integer_column(frame, "n_stays", source)
        require((frame["n_stays"] > 0).all(), f"Invalid cohort count in {source}")
    numeric(sensitivity, sensitivity_columns[4:], "rounding sensitivity")
    numeric(precision, precision_columns[3:], "fixed-profile numerical comparison")
    exact_identifiers(sensitivity, "partition", tuple(f"fold_{index}" for index in range(1, 6)) + ("temporal",), "rounding sensitivity")
    exact_identifiers(sensitivity, "model", ("calibrated_q10", "vector_direct"), "rounding sensitivity")
    exact_identifiers(sensitivity, "horizon_hours", HORIZONS, "rounding sensitivity")
    unique_rows(sensitivity, ("partition", "model", "horizon_hours"), "rounding sensitivity")
    require(len(sensitivity) == 36, "Incomplete 36-partition rounding sensitivity")
    require((sensitivity["absolute_tolerance"] == 1e-10).all(), "Unexpected rounding sensitivity tolerance")
    check_brackets(sensitivity, "original_strict_rate", "original_weak_rate", "original_bracket_violation", "ordinary sensitivity bracket")
    check_brackets(sensitivity, "tolerance_strict_rate", "tolerance_weak_rate", "tolerance_bracket_violation", "tolerance sensitivity bracket")
    require((sensitivity["tolerance_strict_rate"] <= sensitivity["original_strict_rate"] + 1e-12).all(), "Tolerance lower endpoint does not contain the ordinary bracket")
    require((sensitivity["tolerance_weak_rate"] >= sensitivity["original_weak_rate"] - 1e-12).all(), "Tolerance upper endpoint does not contain the ordinary bracket")
    require((sensitivity["max_loss_reproduction_difference"] >= 0).all(), "Negative loss reproduction error")
    require((sensitivity["max_loss_reproduction_difference"] <= 1e-10).all(), "Frozen losses fail the numerical reproduction threshold")
    solvers = ("osqp_original_layout", "clarabel_original_layout", "clarabel_current_cache")
    exact_identifiers(precision, "solver", solvers, "fixed-profile numerical comparison")
    exact_identifiers(precision, "partition", ("temporal",), "fixed-profile numerical comparison")
    exact_identifiers(precision, "horizon_hours", (12,), "fixed-profile numerical comparison")
    exact_identifiers(precision, "absolute_tolerance_mmhg", (0.0, 1e-10, 1e-8), "fixed-profile numerical comparison")
    unique_rows(precision, ("solver", "absolute_tolerance_mmhg"), "fixed-profile numerical comparison")
    require(len(precision) == 9, "Incomplete nine-row fixed-profile comparison")
    require(np.allclose(precision["tau"], 0.1) and np.allclose(precision["lambda_level"], 0.03) and np.allclose(precision["lambda_slope"], 0.10), "The numerical comparison changes the fixed profile")
    require(precision["n_stays"].nunique() == 1 and precision["n_observations"].nunique() == 1, "Numerical solvers use different assessment records")
    check_brackets(precision, "strict_rate", "weak_rate", "bracket_violation", "fixed-profile numerical bracket")
    for solver, group in precision.groupby("solver", observed=True):
        require(np.ptp(group["mean_check_loss"].to_numpy()) <= 1e-12, f"Tolerance alters the scored forecast for {solver}")
        tolerance_order = group.sort_values("absolute_tolerance_mmhg")
        require((np.diff(tolerance_order["strict_rate"].to_numpy()) <= 1e-12).all(), f"Tolerance lower endpoints are not nested for {solver}")
        require((np.diff(tolerance_order["weak_rate"].to_numpy()) >= -1e-12).all(), f"Tolerance upper endpoints are not nested for {solver}")
    precision = ordered(precision, ("solver", "absolute_tolerance_mmhg"), {"solver": solvers})
    return precision.reindex(columns=precision_columns)


def export_tables(outputs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        require(filename.endswith(".csv"), f"Invalid table data filename: {filename}")
        require(not frame.isna().any().any(), f"Missing table values in {filename}")
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.17g", lineterminator="\n")


def main():
    parser = argparse.ArgumentParser(description="Validate complete adaptive studies and export numeric CSV table data")
    parser.add_argument("--simulation-dir", type=Path, default=WORKSPACE_DIR / ".review-work/adaptive/public/simulation")
    parser.add_argument("--clinical-dir", type=Path, default=WORKSPACE_DIR / ".review-work/adaptive/public/clinical")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "manuscript/tables")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE_DIR / ".review-work/adaptive/table-data-validation.json")
    args = parser.parse_args()
    summary, pairs = validate_simulation(args.simulation_dir)
    clinical = validate_clinical(pd.read_csv(args.clinical_dir / "clinical_all_model_summary.csv"))
    numerical = validate_numerics(
        pd.read_csv(args.clinical_dir / "clinical_rounding_precision_sensitivity.csv"),
        pd.read_csv(args.clinical_dir / "clinical_fixed_profile_solver_precision.csv"),
    )
    outputs = simulation_tables(summary, pairs)
    outputs.update(clinical_tables(clinical))
    outputs["adaptive_clinical_numerics.csv"] = numerical
    require(len(outputs) == 9, "Incorrect final table count")
    export_tables(outputs, args.output_dir)
    report = {
        "status": "complete", "replicates_per_setting": 200,
        "simulation_source": str(args.simulation_dir.resolve()),
        "clinical_source": str(args.clinical_dir.resolve()),
        "table_output": str(args.output_dir.resolve()),
        "simulation_settings": list(SETTINGS), "horizons": list(HORIZONS),
        "simulation_summary_rows": len(summary), "paired_summary_rows": len(pairs),
        "clinical_summary_rows": len(clinical),
        "reference_stays_per_replicate": 2000,
        "simulation_calibration_scope": "reference",
        "rounding_sensitivity_rows": 36, "fixed_profile_numerical_rows": len(numerical),
        "tables": {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in outputs.items()},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
