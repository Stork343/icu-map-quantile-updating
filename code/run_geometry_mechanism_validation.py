from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_profile_core import adaptive_candidate_specs
from expanded_family_core import candidate_specs
from run_adaptive_mixed_simulation import (
    HORIZONS, SETTINGS, base_predictions, build_stays, evaluate_matrix,
    simulate_dataset, split_indices, write_json,
)


COVERAGES = (4.0, 8.0, 12.0)
MODES = ("joint", "direct")
SCOPES = ("assessment", "reference")
METHODS = ("full_geometry", "horizon_geometry", "diagonal_profile")
CONTRASTS = (("full_geometry", "diagonal_profile"),
             ("horizon_geometry", "diagonal_profile"),
             ("full_geometry", "horizon_geometry"))
VERSION = "geometry_mechanism_20261022_v1"
REPLAY_TOLERANCE = 1e-9
SIMULTANEOUS_COMPARISONS = 432


def configuration(replicates: int) -> dict:
    return {"version": VERSION, "settings": list(SETTINGS), "coverage_hours": list(COVERAGES),
            "horizon_hours": list(HORIZONS), "selection_modes": list(MODES), "scopes": list(SCOPES),
            "methods": list(METHODS), "contrasts": [list(pair) for pair in CONTRASTS],
            "replicates_per_setting": replicates, "full_replicates_per_setting": 200,
            "source_seed": 20261021, "new_simulation_settings": False,
            "selection_uses_coverage_groups": False, "reference_stays": 2000,
            "replay_tolerance": REPLAY_TOLERANCE, "simultaneous_comparisons": SIMULTANEOUS_COMPARISONS,
            "joint_candidates": {"full_geometry": 84, "horizon_geometry": 12, "diagonal_profile": 96},
            "direct_candidates": {"full_geometry": 42, "horizon_geometry": 6, "diagonal_profile": 48}}


def empty_diagnostics() -> dict:
    return {"qp_fits": 0, "iterations_total": 0, "max_duality_gap": 0.0,
            "max_relative_duality_gap": 0.0, "max_projected_kkt_violation": 0.0,
            "max_stationarity_violation": 0.0}


def choose_rules(tuning: pd.DataFrame, calibration: pd.DataFrame, horizon: float,
                 frozen: dict) -> tuple[dict, list[dict]]:
    specs = {str(row["key"]): row for row in candidate_specs()}
    specs.update({str(row["key"]): {**row, "basis": "adaptive"}
                  for row in adaptive_candidate_specs()})
    candidates = []
    for row in tuning.loc[tuning["horizon"] == horizon].to_dict("records"):
        key = row["name"].removesuffix("__affine")
        if key not in specs:
            continue
        spec = specs[key]
        if spec["basis"] not in ("adaptive", "level_slope"):
            continue
        calibrated = row["name"].endswith("__affine")
        if bool(row["calibrated"]) != calibrated or not np.isfinite(row["tuning_loss"]):
            raise RuntimeError("Invalid saved tuning rule")
        coefficients = {"intercept": 0.0, "slope": 1.0}
        if calibrated:
            match = calibration.loc[(calibration["horizon"] == horizon)
                                    & (calibration["name"] == row["name"])]
            if len(match) != 1:
                raise RuntimeError("Affine rule has no unique saved calibration")
            coefficients = match.iloc[0][["intercept", "slope"]].to_dict()
        candidates.append({**row, "spec": spec, **coefficients})
    selected, selection_rows = {}, []
    for mode in MODES:
        pool = [row for row in candidates if mode == "joint" or not row["calibrated"]]
        menus = {
            "full_geometry": [row for row in pool if row["spec"]["basis"] == "adaptive"],
            "horizon_geometry": [row for row in pool if row["spec"]["basis"] == "adaptive"
                                 and float(row["spec"]["gamma"]) == 0],
            "diagonal_profile": [row for row in pool if row["spec"]["basis"] == "level_slope"],
        }
        expected = (84, 12, 96) if mode == "joint" else (42, 6, 48)
        if tuple(len(menus[name]) for name in METHODS) != expected:
            raise RuntimeError("The penalty class menu differs from the protocol")
        for name in METHODS:
            rule = min(menus[name], key=lambda value: value["tuning_loss"])
            selected[(mode, name)] = rule
            if mode == "joint" and name != "diagonal_profile":
                old = frozen["adaptive_selected" if name == "full_geometry" else "geometry_selected"]
                if old["name"] != rule["name"]:
                    raise RuntimeError("Saved selection does not match complete tuning losses")
                np.testing.assert_allclose([rule["intercept"], rule["slope"], rule["tuning_loss"]],
                                           [old["intercept"], old["slope"], old["tuning_loss"]],
                                           rtol=0, atol=1e-11)
            selection_rows.append({"horizon_hours": horizon, "selection_mode": mode, "method": name,
                "rule_name": rule["name"], "calibrated": rule["calibrated"],
                "intercept": rule["intercept"], "slope": rule["slope"],
                "tuning_loss": rule["tuning_loss"], "gamma": rule["spec"].get("gamma", np.nan),
                "lambda_base": rule["spec"].get("lambda_base", np.nan),
                "lambda_level": rule["spec"].get("lambda_level", np.nan),
                "lambda_slope": rule["spec"].get("lambda_slope", np.nan)})
    return selected, selection_rows


def summarize_stay_records(stays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scope", "selection_mode", "horizon_hours", "coverage_hours"]
    methods, pairs = [], []
    for key, frame in stays.groupby(keys, sort=False):
        group = dict(zip(keys, key))
        pivot = frame.pivot(index="synthetic_stay_index", columns="method", values="loss")
        if set(pivot.columns) != set(METHODS) or pivot.isna().any().any() or len(pivot) < 2:
            raise RuntimeError("Incomplete paired coverage group")
        for name, values in frame.groupby("method", sort=False):
            lower, upper = values[["p_lt", "p_le"]].mean()
            methods.append({**group, "method": name, "n_stays": len(values),
                "mean_loss": values["loss"].mean(), "p_lt": lower, "p_le": upper,
                "bracket_violation": max(lower - .1, .1 - upper, 0),
                "mean_index_count": values["index_count"].mean(),
                "mean_actual_span_hours": values["actual_span_hours"].mean(),
                "mean_time_variance": values["normalized_time_variance"].mean()})
        for first, second in CONTRASTS:
            differences = pivot[first] - pivot[second]
            pairs.append({**group, "first": first, "second": second, "n_stays": len(pivot),
                          "mean_difference": differences.mean(),
                          "within_sample_se": differences.std(ddof=1) / np.sqrt(len(pivot))})
    return pd.DataFrame(methods), pd.DataFrame(pairs)


def process_replicate(payload: dict) -> dict:
    source = Path(payload["source"]) / "replicates" / payload["setting"] / f"rep_{payload['replicate']:04d}"
    destination = Path(payload["output"]) / "replicates" / payload["setting"] / f"rep_{payload['replicate']:04d}"
    expected = {key: payload[key] for key in ("setting", "replicate")}
    expected["version"] = VERSION
    completion_path = destination / "complete.json"
    if completion_path.exists():
        previous = json.loads(completion_path.read_text())
        if any(previous[key] != value for key, value in expected.items()):
            raise RuntimeError("Mechanism checkpoint configuration differs")
        for name in ("stay_metrics.parquet", "methods.csv", "paired.csv", "selections.csv"):
            if not (destination / name).is_file():
                raise FileNotFoundError(destination / name)
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    started = json.loads((source / "started.json").read_text())
    original_completion = json.loads((source / "complete.json").read_text())
    if (started["seed"] != 20261021 or started["n_stays"] != 600 or started["reference_stays"] != 2000
            or started["setting"] != payload["setting"] or started["replicate"] != payload["replicate"]
            or original_completion["status"] != "complete"):
        raise RuntimeError("Source simulation configuration differs")
    seeds = started["derived_seeds"]
    data = simulate_dataset(payload["setting"], 600, seeds["dataset"])
    _, _, assessment = split_indices(600, seeds["split"])
    reference_data = simulate_dataset(payload["setting"], 2000, seeds["reference"])
    parameters = pd.read_csv(source / "population_coefficients.csv").set_index("term")["estimate"]
    beta = parameters.loc[["intercept", "age_z", "male", "urgent"]].to_numpy()
    populations = {"assessment": (data, assessment), "reference": (reference_data, np.arange(2000))}
    layouts = {scope: build_stays(values, indices, beta) for scope, (values, indices) in populations.items()}
    caches = {scope: {} for scope in SCOPES}
    tuning, calibration = (pd.read_csv(source / name) for name in ("tuning.csv", "calibration.csv"))
    source_stays = pd.read_parquet(source / "assessment_stay_metrics.parquet")
    source_methods = pd.read_csv(source / "methods.csv")
    diagnostics, record_frames, selections = empty_diagnostics(), [], []
    maximum_replay_error = 0.0
    for horizon in HORIZONS:
        frozen = json.loads((source / f"frozen_h{int(horizon)}.json").read_text())["rules"]
        rules, local_selections = choose_rules(tuning, calibration, horizon, frozen)
        selections.extend(local_selections)
        for scope in SCOPES:
            values, indices = populations[scope]
            for (mode, name), rule in rules.items():
                prediction = base_predictions(layouts[scope], rule["spec"], horizon, caches[scope], diagnostics)
                metrics = evaluate_matrix(layouts[scope], rule["intercept"] + rule["slope"] * prediction, horizon)
                if mode == "joint":
                    source_name = {"full_geometry": "adaptive_selected", "horizon_geometry": "geometry_selected"}.get(name)
                    if name == "diagonal_profile" and rule["name"] == frozen["original_selected"]["name"]:
                        source_name = "original_selected"
                    if source_name is not None:
                        if scope == "assessment":
                            saved = source_stays.loc[(source_stays["horizon"] == horizon)
                                                     & (source_stays["method"] == source_name)].set_index("synthetic_stay_index").loc[indices]
                            for metric in ("loss", "p_lt", "p_le"):
                                error = float(np.max(np.abs(metrics[metric] - saved[metric].to_numpy())))
                                maximum_replay_error = max(maximum_replay_error, error)
                        else:
                            saved = source_methods.loc[(source_methods["horizon"] == horizon)
                                & (source_methods["method"] == source_name) & (source_methods["scope"] == "reference")]
                            if len(saved) != 1:
                                raise RuntimeError("Source reference score is not unique")
                            for metric in ("loss", "p_lt", "p_le"):
                                maximum_replay_error = max(maximum_replay_error,
                                    abs(float(metrics[metric].mean()) - float(saved.iloc[0][metric])))
                        if maximum_replay_error > REPLAY_TOLERANCE:
                            raise RuntimeError(f"Original simulation replay differs by {maximum_replay_error}")
                record_frames.append(pd.DataFrame({"scope": scope, "selection_mode": mode,
                    "horizon_hours": horizon, "method": name, "synthetic_stay_index": indices,
                    "coverage_hours": values.coverage[indices],
                    "index_count": [len(values.index_time[i]) for i in indices],
                    "actual_span_hours": [np.ptp(values.index_time[i]) for i in indices],
                    "normalized_time_variance": [np.var((values.index_time[i] - 12) / 12) for i in indices],
                    **metrics}))
    stays = pd.concat(record_frames, ignore_index=True)
    if not np.isfinite(stays.select_dtypes(include=np.number).to_numpy()).all():
        raise RuntimeError("Nonfinite mechanism records")
    methods, paired = summarize_stay_records(stays)
    if len(methods) != 108 or len(paired) != 108:
        raise RuntimeError("The complete coverage grid is missing")
    stays.to_parquet(destination / "stay_metrics.parquet", index=False)
    for name, frame in (("methods", methods), ("paired", paired), ("selections", pd.DataFrame(selections))):
        frame.to_csv(destination / f"{name}.csv", index=False)
    result = {**expected, "status": "complete", "derived_seeds": seeds,
              "assessment_stays": 120, "reference_stays": 2000,
              "maximum_original_replay_error": maximum_replay_error, "diagnostics": diagnostics}
    write_json(completion_path, result)
    return result


def summarize_paired(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    keys = ["setting", "scope", "selection_mode", "horizon_hours", "coverage_hours", "first", "second"]
    rows = []
    for key, values in frame.groupby(keys, sort=False):
        if len(values) != replicates or set(values["replicate"]) != set(range(1, replicates + 1)):
            raise RuntimeError("Paired summaries must retain every independent replicate")
        result = dict(zip(keys, key))
        mean = float(values["mean_difference"].mean())
        mcse = float(values["mean_difference"].std(ddof=1) / np.sqrt(replicates))
        pointwise = t.ppf(.975, replicates - 1) * mcse
        simultaneous = t.ppf(1 - .05 / (2 * SIMULTANEOUS_COMPARISONS), replicates - 1) * mcse
        rows.append({**result, "replicates": replicates, "mean_difference": mean, "mcse": mcse,
            "pointwise_low": mean - pointwise, "pointwise_high": mean + pointwise,
            "simultaneous_low": mean - simultaneous, "simultaneous_high": mean + simultaneous,
            "mean_stays_per_replicate": values["n_stays"].mean(),
            "minimum_stays_per_replicate": int(values["n_stays"].min()),
            "total_stay_evaluations": int(values["n_stays"].sum())})
    return pd.DataFrame(rows)


def aggregate(output: Path, jobs: list[dict], replicates: int) -> None:
    files = {name: [] for name in ("paired", "methods", "selections")}
    completions = []
    for job in jobs:
        directory = output / "replicates" / job["setting"] / f"rep_{job['replicate']:04d}"
        completion = json.loads((directory / "complete.json").read_text())
        if completion["status"] != "complete":
            raise RuntimeError("Incomplete mechanism replicate")
        completions.append(completion)
        for name in files:
            files[name].append(pd.read_csv(directory / f"{name}.csv").assign(
                setting=job["setting"], replicate=job["replicate"]))
    frames = {name: pd.concat(values, ignore_index=True) for name, values in files.items()}
    summary = summarize_paired(frames["paired"], replicates)
    if len(summary) != SIMULTANEOUS_COMPARISONS:
        raise RuntimeError("The summary must contain all 432 planned comparisons")
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}_replicates.csv", index=False)
    summary.to_csv(output / "paired_summary.csv", index=False, float_format="%.17g")
    method_keys = ["setting", "scope", "selection_mode", "horizon_hours", "coverage_hours", "method"]
    methods = frames["methods"].groupby(method_keys, sort=False).agg(
        replicates=("mean_loss", "size"), mean_loss=("mean_loss", "mean"),
        p_lt=("p_lt", "mean"), p_le=("p_le", "mean"),
        mean_index_count=("mean_index_count", "mean"),
        mean_actual_span_hours=("mean_actual_span_hours", "mean"),
        mean_time_variance=("mean_time_variance", "mean")).reset_index()
    methods.to_csv(output / "method_summary.csv", index=False)
    write_json(output / "verification.json", {"status": "complete" if replicates == 200 else "partial_design_check",
        "replicates_checked": len(completions), "planned_comparisons": SIMULTANEOUS_COMPARISONS,
        "qp_fits_checked": sum(row["diagnostics"]["qp_fits"] for row in completions),
        "maximum_original_replay_error": max(row["maximum_original_replay_error"] for row in completions),
        "visual_inspection": False, "all_coverage_groups_reported": True,
        "coverage_specific_tuning": False, "source_simulation_seed": 20261021})


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / ".review-work/adaptive/public/simulation")
    parser.add_argument("--output", type=Path, default=root / ".review-work/mechanism/public")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 2 <= args.replicates <= 200 or args.workers < 1:
        raise ValueError("The study requires 2 through 200 replicates and a positive worker count")
    args.output.mkdir(parents=True, exist_ok=True)
    design = configuration(args.replicates)
    design["protocol"] = Path(__file__).with_name("GEOMETRY_MECHANISM_PROTOCOL.md").read_text()
    path = args.output / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise RuntimeError("The saved scientific protocol differs")
    write_json(path, design)
    jobs = [{"source": str(args.source.resolve()), "output": str(args.output.resolve()),
             "setting": setting, "replicate": number}
            for setting in SETTINGS for number in range(1, args.replicates + 1)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        remaining = iter(jobs)
        futures = {pool.submit(process_replicate, job) for job in
                   [next(remaining) for _ in range(min(args.workers, len(jobs)))]}
        count = 0
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                count += 1
                if count % 20 == 0 or count == len(jobs):
                    print(f"Completed {count}/{len(jobs)} mechanism replicates; last {result['setting']} {result['replicate']}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
                job = next(remaining, None)
                if job is not None:
                    futures.add(pool.submit(process_replicate, job))
    aggregate(args.output, jobs, args.replicates)
    print("Complete geometry mechanism evaluation verified", flush=True)


if __name__ == "__main__":
    main()
