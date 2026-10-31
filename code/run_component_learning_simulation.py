from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numba
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_profile_core import fit_affine_calibration
from component_learning_selection import constant_loss_matrix, fit_selection, jsonable, predict_selection
from component_simulation_baselines import (
    actual_response, combined_records, fit_baselines, horizon_flat_mask, predict_baselines, score_predictions,
)
from component_profile_learning import make_profile_batch
from run_adaptive_mixed_simulation import (
    HORIZONS, SETTINGS, build_stays, fit_population, simulate_dataset, split_indices, write_json,
)


SEED = 20261022
VERSION = "continuous_components_20261022_v1"
METHODS = ("continuous_direct", "continuous_affine", "selected_direct", "selected_affine",
           "constant_direct", "constant_affine", "vector_affine", "original_selected",
           "stump_original", "calibrated_q10", "population")
CONTRASTS = (("continuous_direct", "constant_direct"), ("selected_direct", "constant_direct"),
             ("continuous_affine", "constant_affine"), ("selected_affine", "constant_affine"),
             ("selected_affine", "stump_original"), ("selected_affine", "original_selected"),
             ("continuous_affine", "original_selected"), ("selected_affine", "calibrated_q10"))
COMPARISON_COUNT = len(SETTINGS) * len(HORIZONS) * 2 * len(CONTRASTS)


def configuration(replicates: int) -> dict:
    return {"version": VERSION, "master_seed": SEED, "settings": list(SETTINGS),
        "replicates_per_setting": replicates, "full_replicates_per_setting": 200,
        "n_stays": 600, "reference_stays": 2000, "horizons": list(HORIZONS),
        "methods": list(METHODS), "contrasts": [list(pair) for pair in CONTRASTS],
        "comparison_count": COMPARISON_COUNT, "max_iter": 100, "deterministic_starts": 2,
        "protocol": Path(__file__).with_name("COMPONENT_LEARNING_PROTOCOL.md").read_text()}


def fit_post_calibrations(batch, predictions: dict) -> dict:
    response = batch.later_residual + batch.later_population
    coefficients = {}
    for model in ("continuous", "constant"):
        for horizon in HORIZONS:
            records, observations = horizon_flat_mask(batch, horizon)
            sizes = np.diff(batch.later_indptr)[records]
            weights = np.repeat(1.0 / sizes, sizes)
            coefficients[f"{model}_{int(horizon)}"] = fit_affine_calibration(
                response[observations], predictions[model]["flat_predictions"][observations], weights)
    return coefficients


def combine_predictions(batch, learned: dict, baseline: dict, calibrations: dict,
                        selected_kind: str, raw_response: np.ndarray) -> dict:
    result = dict(baseline)
    for model in ("continuous", "constant"):
        result[f"{model}_direct"] = {**learned[model],
            **score_predictions(batch, learned[model]["flat_predictions"], raw_response)}
        calibrated = np.empty(len(batch.later_residual))
        for horizon in HORIZONS:
            _, observations = horizon_flat_mask(batch, horizon)
            coefficients = calibrations[f"{model}_{int(horizon)}"]
            calibrated[observations] = coefficients["intercept"] + coefficients["slope"] * learned[model]["flat_predictions"][observations]
        result[f"{model}_affine"] = {**score_predictions(batch, calibrated, raw_response), "flat_predictions": calibrated}
    result["selected_direct"] = result[f"{selected_kind}_direct"]
    result["selected_affine"] = result[f"{selected_kind}_affine"]
    if set(result) != set(METHODS):
        raise RuntimeError("The complete component comparison is missing a method")
    for name, values in result.items():
        for metric in ("loss", "p_lt", "p_le"):
            if values[metric].shape != (batch.n_records,) or not np.isfinite(values[metric]).all():
                raise RuntimeError(f"Invalid {name} {metric} score")
    return result


def profile_diagnostics(fitted: dict, baselines: dict, evaluations: list[dict]) -> dict:
    records = list(baselines["numerical_certificates"]) + evaluations
    for row in fitted["validation_rows"]:
        if row["candidate"] != "constant":
            records.append(row["fit"]["all_evaluation_diagnostics"])
            records.append(row["validation_diagnostics"])
    records.append(fitted["best_continuous"]["all_evaluation_diagnostics"])
    records.append(fitted["constant_refit_diagnostics"])
    return {"recorded_profile_solves": int(sum(row["profiles"] for row in records)),
        "all_certified": all(row["all_certified"] for row in records),
        "max_duality_gap": max(row["max_duality_gap"] for row in records),
        "max_projected_kkt_violation": max(row["max_projected_kkt_violation"] for row in records),
        "max_stationarity_violation": max(row["max_stationarity_violation"] for row in records)}


def replicate_job(payload: dict) -> dict:
    directory = Path(payload["output"]) / "replicates" / payload["setting"] / f"rep_{payload['replicate']:04d}"
    expected = {"version": VERSION, "setting": payload["setting"], "replicate": payload["replicate"], "seed": SEED}
    completion_path = directory / "complete.json"
    previous_completion = json.loads(completion_path.read_text()) if completion_path.is_file() else None
    if completion_path.exists() and not payload.get("rescore_frozen", False):
        completed = json.loads(completion_path.read_text())
        if any(completed[key] != value for key, value in expected.items()):
            raise RuntimeError("The completed component replicate has a different protocol")
        for name in ("frozen.json", "population_coefficients.csv", "methods.csv", "paired.csv", "assessment_metrics.parquet"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        return completed
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sequence = np.random.SeedSequence([SEED, SETTINGS.index(payload["setting"]), payload["replicate"]])
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(4)]
    data = simulate_dataset(payload["setting"], 600, seeds[0])
    fitting, tuning, assessment = split_indices(600, seeds[1])
    frozen_path = directory / "frozen.json"
    resumed_frozen_fit = frozen_path.is_file()
    if resumed_frozen_fit:
        frozen = json.loads(frozen_path.read_text())
        if frozen["derived_seeds"] != seeds:
            raise RuntimeError("The frozen learning data seeds differ")
        beta = np.asarray(frozen["population_coefficients"], dtype=float)
        source = pd.read_csv(directory / "population_coefficients.csv").set_index("term")["estimate"]
        np.testing.assert_allclose(beta, source.loc[["intercept", "age_z", "male", "urgent"]], rtol=0, atol=1e-12)
        tune_stays = build_stays(data, tuning, beta)
        tuning_records, tuning_horizons, tuning_ids = combined_records(tune_stays, tuning)
        tuning_batch = make_profile_batch(tuning_records, tuning_horizons)
        _, grid_diagnostics = constant_loss_matrix(tuning_batch)
        baselines = {"horizons": frozen["baseline_horizons"], "numerical_certificates": [grid_diagnostics]}
        fitted, calibrations = frozen["learning"], frozen["post_calibrations"]
    else:
        beta = fit_population(data, fitting, directory)
        fit_stays = build_stays(data, fitting, beta)
        tune_stays = build_stays(data, tuning, beta)
        tuning_records, tuning_horizons, tuning_ids = combined_records(tune_stays, tuning)
        baselines = fit_baselines(tuning_records, tuning_horizons, fit_stays)
        fitted = fit_selection(tuning_records, tuning_horizons, tuning_ids, seeds[3],
            precomputed_constant_losses=baselines["constant_loss_matrix"], max_iter=100,
            deterministic_starts=2, verbose=False)
        tuning_batch = make_profile_batch(tuning_records, tuning_horizons)
    tuning_predictions = predict_selection(tuning_records, tuning_horizons, fitted)
    if not resumed_frozen_fit:
        calibrations = fit_post_calibrations(tuning_batch, tuning_predictions)
        frozen = {"learning": fitted, "baseline_horizons": baselines["horizons"],
                  "post_calibrations": calibrations, "derived_seeds": seeds,
                  "population_coefficients": beta.tolist()}
        write_json(frozen_path, jsonable(frozen))
    reference = simulate_dataset(payload["setting"], 2000, seeds[2])
    populations = {"assessment": (data, assessment), "reference": (reference, np.arange(2000))}
    method_rows, paired_rows, assessment_frames, evaluation_diagnostics = [], [], [], []
    evaluation_diagnostics.extend([tuning_predictions[name]["diagnostics"] for name in ("continuous", "constant")])
    maximum_constant_error = 0.0
    for scope, (population, identities) in populations.items():
        stays = build_stays(population, identities, beta)
        records, horizons, record_ids = combined_records(stays, identities)
        baseline, batch, certificates = predict_baselines(records, horizons, baselines,
            diagnostic_directory=directory / "numerical_diagnostics" / scope)
        predictions = predict_selection(records, horizons, fitted)
        evaluation_diagnostics.extend(certificates)
        evaluation_diagnostics.extend([predictions[name]["diagnostics"] for name in ("continuous", "constant")])
        maximum_constant_error = max(maximum_constant_error,
            float(np.max(np.abs(predictions["constant"]["loss"] - baseline["constant_direct"]["loss"]))))
        if maximum_constant_error > 1e-7:
            raise RuntimeError("The two independent constant prediction calculations differ")
        scores = combine_predictions(batch, predictions, baseline, calibrations, fitted["selected_kind"],
                                     actual_response(records, horizons))
        for horizon in HORIZONS:
            selected = horizons == horizon
            for name in METHODS:
                values = scores[name]
                loss, lower, upper = [float(values[metric][selected].mean()) for metric in ("loss", "p_lt", "p_le")]
                method_rows.append({"scope": scope, "horizon_hours": horizon, "method": name,
                    "n_stays": int(selected.sum()), "mean_loss": loss, "p_lt": lower, "p_le": upper,
                    "bracket_violation": max(lower - .1, .1 - upper, 0),
                    "within_sample_se": float(values["loss"][selected].std(ddof=1) / np.sqrt(selected.sum()))})
                if scope == "assessment":
                    assessment_frames.append(pd.DataFrame({"horizon_hours": horizon, "method": name,
                        "synthetic_stay_index": record_ids[selected],
                        **{metric: values[metric][selected] for metric in ("loss", "p_lt", "p_le")}}))
            for first, second in CONTRASTS:
                differences = scores[first]["loss"][selected] - scores[second]["loss"][selected]
                paired_rows.append({"scope": scope, "horizon_hours": horizon, "first": first, "second": second,
                    "n_stays": int(selected.sum()), "mean_difference": float(differences.mean()),
                    "within_sample_se": float(differences.std(ddof=1) / np.sqrt(selected.sum()))})
    pd.DataFrame(method_rows).to_csv(directory / "methods.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(directory / "paired.csv", index=False)
    pd.concat(assessment_frames, ignore_index=True).to_parquet(directory / "assessment_metrics.parquet", index=False)
    diagnostics = profile_diagnostics(fitted, baselines, evaluation_diagnostics)
    if not diagnostics["all_certified"]:
        raise RuntimeError("An uncertified profile entered the simulation")
    completed = {**expected, "status": "complete", "derived_seeds": seeds,
        "selected_kind": fitted["selected_kind"], "ridge_eta": fitted["best_continuous_eta"],
        "maximum_constant_replay_error": maximum_constant_error, "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started, "resumed_frozen_fit": resumed_frozen_fit,
        "reporting_comparisons": "retained raw outcomes versus returned numerical forecasts",
        "original_learning_run_seconds": (previous_completion.get("original_learning_run_seconds", previous_completion["runtime_seconds"])
                                           if previous_completion is not None else time.perf_counter() - started)}
    write_json(completion_path, jsonable(completed))
    return completed


def aggregate(output: Path, replicates: int) -> None:
    frames = {name: [] for name in ("methods", "paired")}
    completions = []
    for setting in SETTINGS:
        for number in range(1, replicates + 1):
            directory = output / "replicates" / setting / f"rep_{number:04d}"
            completed = json.loads((directory / "complete.json").read_text())
            if completed["status"] != "complete":
                raise RuntimeError("Missing completed component replicate")
            completions.append(completed)
            for name in frames:
                frames[name].append(pd.read_csv(directory / f"{name}.csv").assign(setting=setting, replicate=number))
    values = {name: pd.concat(rows, ignore_index=True) for name, rows in frames.items()}
    for name, frame in values.items():
        frame.to_csv(output / f"{name}_replicates.csv", index=False)
    methods = []
    for key, frame in values["methods"].groupby(["setting", "scope", "horizon_hours", "method"], sort=False):
        if len(frame) != replicates or frame.replicate.nunique() != replicates:
            raise RuntimeError("Method summaries omitted independent replicates")
        methods.append({**dict(zip(("setting", "scope", "horizon_hours", "method"), key)),
            "replicates": replicates, "mean_loss": frame.mean_loss.mean(),
            "mcse": frame.mean_loss.std(ddof=1) / np.sqrt(replicates),
            "p_lt": frame.p_lt.mean(), "p_le": frame.p_le.mean(),
            "mean_bracket_violation": max(frame.p_lt.mean() - .1, .1 - frame.p_le.mean(), 0),
            "mean_replicate_violation": frame.bracket_violation.mean()})
    pairs = []
    for key, frame in values["paired"].groupby(["setting", "scope", "horizon_hours", "first", "second"], sort=False):
        if len(frame) != replicates or frame.replicate.nunique() != replicates:
            raise RuntimeError("Paired summaries omitted independent replicates")
        mean, mcse = frame.mean_difference.mean(), frame.mean_difference.std(ddof=1) / np.sqrt(replicates)
        point, simultaneous = t.ppf(.975, replicates - 1) * mcse, t.ppf(1 - .05 / (2 * COMPARISON_COUNT), replicates - 1) * mcse
        pairs.append({**dict(zip(("setting", "scope", "horizon_hours", "first", "second"), key)),
            "replicates": replicates, "mean_difference": mean, "mcse": mcse,
            "pointwise_low": mean - point, "pointwise_high": mean + point,
            "simultaneous_low": mean - simultaneous, "simultaneous_high": mean + simultaneous})
    pd.DataFrame(methods).to_csv(output / "method_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(pairs).to_csv(output / "paired_summary.csv", index=False, float_format="%.17g")
    selections = pd.DataFrame([{key: row[key] for key in ("setting", "replicate", "selected_kind", "ridge_eta")}
                               for row in completions])
    selections.to_csv(output / "selection_replicates.csv", index=False)
    selections.groupby(["setting", "selected_kind", "ridge_eta"]).size().rename("count").reset_index().to_csv(output / "selection_frequencies.csv", index=False)
    write_json(output / "verification.json", {"status": "complete" if replicates == 200 else "partial_design_check",
        "replicates_checked": len(completions), "planned_comparisons": COMPARISON_COUNT,
        "all_profiles_certified": all(row["diagnostics"]["all_certified"] for row in completions),
        "recorded_profile_solves": sum(row["diagnostics"]["recorded_profile_solves"] for row in completions),
        "maximum_constant_replay_error": max(row["maximum_constant_replay_error"] for row in completions),
        "assessment_used_for_learning": False, "fresh_master_seed": SEED, "visual_inspection": False})


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / ".review-work/component_learning/public/simulation")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rescore-frozen", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.replicates <= 200 or args.workers < 1:
        raise ValueError("The simulation requires 2 through 200 replicates and positive workers")
    args.output.mkdir(parents=True, exist_ok=True)
    design = configuration(args.replicates)
    path = args.output / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise RuntimeError("The component simulation protocol differs")
    write_json(path, design)
    write_json(args.output / "environment.json", {"python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "numba": numba.__version__, "numba_threads": numba.get_num_threads()})
    jobs = [{"output": str(args.output.resolve()), "setting": setting, "replicate": number,
             "rescore_frozen": args.rescore_frozen}
            for setting in SETTINGS for number in range(1, args.replicates + 1)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        remaining = iter(jobs)
        futures = {pool.submit(replicate_job, job) for job in
                   [next(remaining) for _ in range(min(args.workers, len(jobs)))]}
        count = 0
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                if future.exception() is not None:
                    write_json(args.output / "numerical_attempt.json", {
                        "version": VERSION, "error": str(future.exception()),
                        "status": "unresolved", "completed_replicates": count})
                completed = future.result()
                count += 1
                if count % 20 == 0 or count == len(jobs):
                    print(f"Completed {count}/{len(jobs)} fresh component replicates; last {completed['setting']} {completed['replicate']}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
                job = next(remaining, None)
                if job is not None:
                    futures.add(pool.submit(replicate_job, job))
    aggregate(args.output, args.replicates)
    print("Fresh component learning simulation completed", flush=True)


if __name__ == "__main__":
    main()
