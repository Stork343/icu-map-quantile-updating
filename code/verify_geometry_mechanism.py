from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_profile_core import adaptive_candidate_specs
from expanded_family_core import candidate_specs
from run_adaptive_mixed_simulation import simulate_dataset


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")
HORIZONS = (4.0, 8.0, 12.0)
COVERAGES = (4.0, 8.0, 12.0)
SCOPES = ("assessment", "reference")
MODES = ("joint", "direct")
METHODS = ("full_geometry", "horizon_geometry", "diagonal_profile")
CONTRASTS = (("full_geometry", "diagonal_profile"), ("horizon_geometry", "diagonal_profile"),
             ("full_geometry", "horizon_geometry"))
GROUP_KEYS = ["scope", "selection_mode", "horizon_hours", "coverage_hours"]
METHOD_KEYS = GROUP_KEYS + ["method"]
PAIR_KEYS = GROUP_KEYS + ["first", "second"]
FAMILY_SIZE = 432
VERSION = "geometry_mechanism_20261022_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str],
                   columns: list[str], tolerance: float = 2e-11) -> None:
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(),
            "Duplicate scientific result keys")
    left, right = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    require(left.index.equals(right.index), "Scientific result grids differ")
    np.testing.assert_allclose(left[columns].to_numpy(dtype=float), right[columns].to_numpy(dtype=float),
                               rtol=0, atol=tolerance)


def reconstruct_records(stays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifiers = ["scope", "selection_mode", "horizon_hours", "method", "synthetic_stay_index"]
    require(not stays.duplicated(identifiers).any(), "Repeated stay score")
    require(set(stays["scope"]) == set(SCOPES) and set(stays["selection_mode"]) == set(MODES),
            "Missing evaluation population or selection mode")
    require(set(stays["horizon_hours"]) == set(HORIZONS) and set(stays["method"]) == set(METHODS),
            "Missing forecast horizon or prediction method")
    require(np.isfinite(stays.select_dtypes(include=np.number).to_numpy()).all(), "Nonfinite stay score")
    require((stays["loss"] >= 0).all(), "Negative quantile loss")
    require(((stays["p_lt"] >= 0) & (stays["p_le"] <= 1)
             & (stays["p_lt"] <= stays["p_le"] + 1e-13)).all(), "Invalid probability mass bracket")
    require(set(stays["coverage_hours"]) == set(COVERAGES), "Missing early coverage interval")
    require(((stays["actual_span_hours"] >= 0)
             & (stays["actual_span_hours"] <= stays["coverage_hours"])).all(), "Invalid observed time span")
    metadata = ["coverage_hours", "index_count", "actual_span_hours", "normalized_time_variance"]
    counts = stays.groupby(["scope", "synthetic_stay_index"])[metadata].nunique()
    require((counts == 1).all().all(), "A stay changes its early information between prediction rules")
    expected_groups = set(itertools.product(SCOPES, MODES, HORIZONS, COVERAGES))
    require(set(stays.groupby(GROUP_KEYS).groups) == expected_groups, "Incomplete coverage and forecast grid")
    method_rows, pair_rows = [], []
    for key, values in stays.groupby(GROUP_KEYS, sort=False):
        group = dict(zip(GROUP_KEYS, key))
        loss = values.pivot(index="synthetic_stay_index", columns="method", values="loss")
        require(set(loss.columns) == set(METHODS) and not loss.isna().any().any(), "Unpaired rule evaluation")
        require(len(loss) >= 2, "Coverage group has insufficient stays")
        for name in METHODS:
            records = values.loc[values["method"] == name]
            require(len(records) == len(loss), "Coverage group denominators differ between methods")
            lower, upper = records[["p_lt", "p_le"]].mean()
            method_rows.append({**group, "method": name, "n_stays": len(records),
                "mean_loss": records["loss"].mean(), "p_lt": lower, "p_le": upper,
                "bracket_violation": max(lower - .1, .1 - upper, 0),
                "mean_index_count": records["index_count"].mean(),
                "mean_actual_span_hours": records["actual_span_hours"].mean(),
                "mean_time_variance": records["normalized_time_variance"].mean()})
        for first, second in CONTRASTS:
            differences = loss[first].to_numpy() - loss[second].to_numpy()
            pair_rows.append({**group, "first": first, "second": second, "n_stays": len(differences),
                "mean_difference": float(np.mean(differences)),
                "within_sample_se": float(np.std(differences, ddof=1) / np.sqrt(len(differences)))})
    return pd.DataFrame(method_rows), pd.DataFrame(pair_rows)


def verify_selections(selected: pd.DataFrame, tuning: pd.DataFrame, calibration: pd.DataFrame,
                      source_directory: Path) -> None:
    keys = ["horizon_hours", "selection_mode", "method"]
    expected = set(itertools.product(HORIZONS, MODES, METHODS))
    require(set(map(tuple, selected[keys].to_numpy())) == expected and len(selected) == len(expected),
            "Missing or duplicate frozen rule")
    adaptive = {str(row["key"]): row for row in adaptive_candidate_specs()}
    diagonal = {str(row["key"]): row for row in candidate_specs() if row["basis"] == "level_slope"}
    for horizon in HORIZONS:
        local = tuning.loc[tuning["horizon"] == horizon].copy()
        require(local["name"].nunique() == 200 and len(local) == 200, "Saved tuning menu is incomplete")
        local["base_key"] = local["name"].str.removesuffix("__affine")
        frozen = json.loads((source_directory / f"frozen_h{int(horizon)}.json").read_text())["rules"]
        for mode, method in itertools.product(MODES, METHODS):
            pool = local if mode == "joint" else local.loc[~local["name"].str.endswith("__affine")]
            if method == "diagonal_profile":
                pool = pool.loc[pool["base_key"].isin(diagonal)]
                count = 96 if mode == "joint" else 48
            else:
                names = [key for key, spec in adaptive.items()
                         if method == "full_geometry" or spec["gamma"] == 0]
                pool = pool.loc[pool["base_key"].isin(names)]
                count = (84 if method == "full_geometry" else 12) if mode == "joint" else (42 if method == "full_geometry" else 6)
            require(len(pool) == count, "Penalty class contains unexpected candidates")
            best = pool.loc[pool["tuning_loss"].idxmin()]
            chosen = selected.loc[(selected["horizon_hours"] == horizon)
                & (selected["selection_mode"] == mode) & (selected["method"] == method)].iloc[0]
            require(chosen["rule_name"] == best["name"], "Rule was not selected from all tuning stays")
            np.testing.assert_allclose(chosen["tuning_loss"], best["tuning_loss"], rtol=0, atol=1e-12)
            calibrated = best["name"].endswith("__affine")
            require(bool(chosen["calibrated"]) == calibrated, "Saved calibration flag differs")
            if calibrated:
                coefficients = calibration.loc[(calibration["horizon"] == horizon)
                    & (calibration["name"] == best["name"])]
                require(len(coefficients) == 1, "Missing shared affine calibration")
                expected_coefficients = coefficients.iloc[0][["intercept", "slope"]].to_numpy(dtype=float)
            else:
                expected_coefficients = np.asarray([0., 1.])
            np.testing.assert_allclose(chosen[["intercept", "slope"]].to_numpy(dtype=float),
                                       expected_coefficients, rtol=0, atol=1e-11)
            if method == "diagonal_profile":
                require(np.isnan(chosen["gamma"]), "Diagonal rule contains a history geometry term")
                spec = diagonal[best["base_key"]]
                np.testing.assert_allclose(chosen[["lambda_level", "lambda_slope"]].to_numpy(dtype=float),
                                           [spec["lambda_level"], spec["lambda_slope"]], rtol=0, atol=1e-13)
            else:
                spec = adaptive[best["base_key"]]
                np.testing.assert_allclose(chosen[["lambda_base", "gamma"]].to_numpy(dtype=float),
                                           [spec["lambda_base"], spec["gamma"]], rtol=0, atol=1e-13)
                if mode == "joint":
                    original = frozen["adaptive_selected" if method == "full_geometry" else "geometry_selected"]
                    require(best["name"] == original["name"], "Original frozen rule changed")


def verify_replicate(payload: tuple[str, str, str, int]) -> dict:
    output_root, source_root, setting, number = payload
    directory = Path(output_root) / "replicates" / setting / f"rep_{number:04d}"
    source = Path(source_root) / "replicates" / setting / f"rep_{number:04d}"
    completion = json.loads((directory / "complete.json").read_text())
    require(completion["status"] == "complete" and completion["version"] == VERSION,
            "Mechanism completion metadata differs")
    require(completion["setting"] == setting and completion["replicate"] == number, "Replicate identity differs")
    started = json.loads((source / "started.json").read_text())
    original_completion = json.loads((source / "complete.json").read_text())
    require(original_completion["status"] == "complete" and original_completion["derived_seeds"] == started["derived_seeds"],
            "Original simulation has no matching completion")
    seeds = started["derived_seeds"]
    require(completion["derived_seeds"] == seeds, "Replicate seeds differ from original simulation")
    sequence = np.random.SeedSequence([20261021, SETTINGS.index(setting), number])
    expected_seeds = dict(zip(("dataset", "split", "reference"),
                              [int(child.generate_state(1)[0]) for child in sequence.spawn(3)]))
    require(seeds == expected_seeds, "Derived seeds differ from the original frozen seed rule")
    require(completion["assessment_stays"] == 120 and completion["reference_stays"] == 2000,
            "Evaluation population sizes differ")
    error = float(completion["maximum_original_replay_error"])
    require(np.isfinite(error) and 0 <= error <= 1e-9, "Original risk replay exceeded tolerance")
    diagnostic = completion["diagnostics"]
    require(diagnostic["qp_fits"] > 0 and np.isfinite(list(diagnostic.values())).all(), "Invalid numerical certificates")
    require(diagnostic["max_projected_kkt_violation"] <= 5e-7
            and diagnostic["max_stationarity_violation"] <= 1e-10, "Profile optimality certificate failed")
    require(diagnostic["max_duality_gap"] <= 1e-7 or diagnostic["max_relative_duality_gap"] <= 1e-8,
            "Profile objective gap exceeded tolerance")
    selected = pd.read_csv(directory / "selections.csv")
    verify_selections(selected, pd.read_csv(source / "tuning.csv"), pd.read_csv(source / "calibration.csv"), source)
    coefficients = pd.read_csv(source / "population_coefficients.csv")
    require(len(coefficients) == 4 and set(coefficients["solver"]) == {"br"}, "Unexpected population fitting method")
    require(np.isfinite(coefficients.select_dtypes(include=np.number).to_numpy()).all(), "Nonfinite population fitting certificate")
    require(coefficients["relative_objective_gap"].max() <= 1e-7
            and coefficients["dual_stationarity"].max() <= 1e-6, "Population fitting certificate failed")
    stays = pd.read_parquet(directory / "stay_metrics.parquet")
    original_stays = pd.read_parquet(source / "assessment_stay_metrics.parquet")
    original_methods = pd.read_csv(source / "methods.csv")
    replay_error = 0.
    for horizon in HORIZONS:
        frozen = json.loads((source / f"frozen_h{int(horizon)}.json").read_text())["rules"]
        for method in METHODS:
            original_method = {"full_geometry": "adaptive_selected", "horizon_geometry": "geometry_selected"}.get(method)
            if method == "diagonal_profile":
                name = selected.loc[(selected["horizon_hours"] == horizon)
                    & (selected["selection_mode"] == "joint") & (selected["method"] == method), "rule_name"].iat[0]
                if name == frozen["original_selected"]["name"]:
                    original_method = "original_selected"
            if original_method is None:
                continue
            for scope in SCOPES:
                current = stays.loc[(stays["selection_mode"] == "joint")
                    & (stays["scope"] == scope) & (stays["horizon_hours"] == horizon) & (stays["method"] == method)]
                if scope == "assessment":
                    previous = original_stays.loc[(original_stays["horizon"] == horizon)
                        & (original_stays["method"] == original_method)].set_index("synthetic_stay_index").sort_index()
                    current = current.set_index("synthetic_stay_index").sort_index()
                    require(current.index.equals(previous.index), "Replayed assessment membership differs")
                    replay_error = max(replay_error, float(np.max(np.abs(
                        current[["loss", "p_lt", "p_le"]].to_numpy() - previous[["loss", "p_lt", "p_le"]].to_numpy()))))
                else:
                    previous = original_methods.loc[(original_methods["horizon"] == horizon)
                        & (original_methods["scope"] == scope) & (original_methods["method"] == original_method)]
                    require(len(previous) == 1, "Original reference summary is missing or repeated")
                    replay_error = max(replay_error, float(np.max(np.abs(
                        current[["loss", "p_lt", "p_le"]].mean().to_numpy()
                        - previous.iloc[0][["loss", "p_lt", "p_le"]].to_numpy(dtype=float)))))
    require(replay_error <= 1e-9, "Independent original risk replay exceeded tolerance")
    np.testing.assert_allclose(replay_error, error, rtol=0, atol=2e-12)
    data = simulate_dataset(setting, 600, seeds["dataset"])
    reference = simulate_dataset(setting, 2000, seeds["reference"])
    assessment = np.random.default_rng(seeds["split"]).permutation(600)[480:]
    metadata = {"assessment": (data, assessment), "reference": (reference, np.arange(2000))}
    for scope, (population, indices) in metadata.items():
        local = stays.loc[stays["scope"] == scope]
        expected_rows = len(indices) * len(MODES) * len(HORIZONS) * len(METHODS)
        require(len(local) == expected_rows, "Evaluation records are missing or duplicated")
        for _, group in local.groupby(["selection_mode", "horizon_hours", "method"]):
            observed = group.set_index("synthetic_stay_index").sort_index()
            ordered = np.sort(indices)
            np.testing.assert_array_equal(observed.index.to_numpy(), ordered)
            np.testing.assert_array_equal(observed["coverage_hours"].to_numpy(), population.coverage[ordered])
            np.testing.assert_array_equal(observed["index_count"].to_numpy(),
                                          [len(population.index_time[index]) for index in ordered])
            np.testing.assert_allclose(observed["actual_span_hours"].to_numpy(),
                [np.ptp(population.index_time[index]) for index in ordered], rtol=0, atol=1e-13)
            np.testing.assert_allclose(observed["normalized_time_variance"].to_numpy(),
                [np.var((population.index_time[index] - 12) / 12) for index in ordered], rtol=0, atol=1e-13)
    methods, paired = reconstruct_records(stays)
    require(len(methods) == 108 and len(paired) == 108, "Incomplete replicate contrast grid")
    compare_frames(pd.read_csv(directory / "methods.csv"), methods, METHOD_KEYS,
        ["n_stays", "mean_loss", "p_lt", "p_le", "bracket_violation", "mean_index_count",
         "mean_actual_span_hours", "mean_time_variance"])
    compare_frames(pd.read_csv(directory / "paired.csv"), paired, PAIR_KEYS,
                   ["n_stays", "mean_difference", "within_sample_se"])
    return {"methods": methods.assign(setting=setting, replicate=number),
            "paired": paired.assign(setting=setting, replicate=number),
            "selections": selected.assign(setting=setting, replicate=number),
            "qp_fits": diagnostic["qp_fits"], "replay_error": replay_error, "stay_scores": len(stays)}


def reconstruct_intervals(paired: pd.DataFrame, replicates: int) -> pd.DataFrame:
    keys = ["setting"] + PAIR_KEYS
    results = []
    for key, frame in paired.groupby(keys, sort=False):
        require(len(frame) == replicates and set(frame["replicate"]) == set(range(1, replicates + 1)),
                "Interval calculation omitted an independent replicate")
        differences = frame["mean_difference"].to_numpy(dtype=float)
        mean = float(np.mean(differences))
        mcse = float(np.std(differences, ddof=1) / np.sqrt(replicates))
        halfwidth = float(t.ppf(.975, replicates - 1) * mcse)
        family_halfwidth = float(t.ppf(1 - .05 / (2 * FAMILY_SIZE), replicates - 1) * mcse)
        results.append({**dict(zip(keys, key)), "replicates": replicates, "mean_difference": mean,
            "mcse": mcse, "pointwise_low": mean - halfwidth, "pointwise_high": mean + halfwidth,
            "simultaneous_low": mean - family_halfwidth, "simultaneous_high": mean + family_halfwidth,
            "mean_stays_per_replicate": float(frame["n_stays"].mean()),
            "minimum_stays_per_replicate": int(frame["n_stays"].min()),
            "total_stay_evaluations": int(frame["n_stays"].sum())})
    return pd.DataFrame(results)


def verify(output: Path, source: Path, workers: int = 1) -> dict:
    design = json.loads((output / "design.json").read_text())
    replicates = int(design["replicates_per_setting"])
    require(design["version"] == VERSION and replicates in range(2, 201), "Unexpected mechanism protocol")
    for field, values in (("settings", SETTINGS), ("coverage_hours", COVERAGES), ("horizon_hours", HORIZONS),
                          ("selection_modes", MODES), ("scopes", SCOPES), ("methods", METHODS)):
        require(design[field] == list(values), f"Unexpected protocol field {field}")
    require(design["simultaneous_comparisons"] == FAMILY_SIZE
            and design["selection_uses_coverage_groups"] is False, "Unexpected inference or selection protocol")
    protocol = Path(__file__).with_name("GEOMETRY_MECHANISM_PROTOCOL.md").read_text()
    require(design["protocol"] == protocol, "Saved scientific protocol differs from the supplied protocol")
    jobs = [(str(output), str(source), setting, number)
            for setting in SETTINGS for number in range(1, replicates + 1)]
    if workers == 1:
        results = [verify_replicate(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(verify_replicate, jobs))
    methods = pd.concat([result["methods"] for result in results], ignore_index=True)
    paired = pd.concat([result["paired"] for result in results], ignore_index=True)
    selected = pd.concat([result["selections"] for result in results], ignore_index=True)
    for name, rebuilt, keys, columns in (
        ("methods", methods, ["setting", "replicate"] + METHOD_KEYS,
         ["n_stays", "mean_loss", "p_lt", "p_le", "bracket_violation", "mean_index_count",
          "mean_actual_span_hours", "mean_time_variance"]),
        ("paired", paired, ["setting", "replicate"] + PAIR_KEYS,
         ["n_stays", "mean_difference", "within_sample_se"]),
        ("selections", selected, ["setting", "replicate", "horizon_hours", "selection_mode", "method"],
         ["intercept", "slope", "tuning_loss"]),
    ):
        compare_frames(pd.read_csv(output / f"{name}_replicates.csv"), rebuilt, keys, columns)
    intervals = reconstruct_intervals(paired, replicates)
    require(len(intervals) == FAMILY_SIZE, "Missing planned interval")
    compare_frames(pd.read_csv(output / "paired_summary.csv"), intervals, ["setting"] + PAIR_KEYS,
        ["replicates", "mean_difference", "mcse", "pointwise_low", "pointwise_high", "simultaneous_low",
         "simultaneous_high", "mean_stays_per_replicate", "minimum_stays_per_replicate", "total_stay_evaluations"])
    method_keys = ["setting"] + METHOD_KEYS
    method_summary = methods.groupby(method_keys, sort=False).agg(
        replicates=("mean_loss", "size"), mean_loss=("mean_loss", "mean"),
        p_lt=("p_lt", "mean"), p_le=("p_le", "mean"), mean_index_count=("mean_index_count", "mean"),
        mean_actual_span_hours=("mean_actual_span_hours", "mean"), mean_time_variance=("mean_time_variance", "mean")).reset_index()
    compare_frames(pd.read_csv(output / "method_summary.csv"), method_summary, method_keys,
        ["replicates", "mean_loss", "p_lt", "p_le", "mean_index_count", "mean_actual_span_hours", "mean_time_variance"])
    return {"status": "passed" if replicates == 200 else "partial_design_check",
        "replicates_checked": len(jobs), "planned_comparisons_checked": len(intervals),
        "stay_scores_checked": sum(result["stay_scores"] for result in results),
        "qp_fits_checked": sum(result["qp_fits"] for result in results),
        "maximum_original_replay_error": max(result["replay_error"] for result in results),
        "frozen_selection_and_calibration_checked": True, "all_coverage_groups_reported": True,
        "pointwise_and_simultaneous_intervals_checked": True, "visual_inspection": False}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=root / ".review-work/adaptive/public/simulation")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    require(args.workers > 0, "Worker count must be positive")
    report = verify(args.output, args.source, args.workers)
    (args.output / "independent_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
