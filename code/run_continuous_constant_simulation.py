from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

from adaptive_profile_core import fit_affine_calibration
from component_learning_selection import jsonable, predict_selection
from component_profile_learning import make_profile_batch, penalty_values
from component_simulation_baselines import actual_response, combined_records, horizon_flat_mask, score_predictions
from continuous_constant_control import VERSION, fit_continuous_constant_control, predict_continuous_constant_control
from run_adaptive_mixed_simulation import HORIZONS, SETTINGS, build_stays, simulate_dataset, split_indices, write_json


CONTRASTS = (("continuous_direct", "continuous_constant_direct"),
             ("selected_direct", "selected_constant_direct"),
             ("continuous_affine", "continuous_constant_affine"),
             ("selected_affine", "selected_constant_affine"),
             ("continuous_constant_direct", "constant_direct"),
             ("selected_constant_direct", "constant_direct"),
             ("continuous_constant_affine", "constant_affine"),
             ("selected_constant_affine", "constant_affine"))
COMPARISON_COUNT = len(CONTRASTS) * len(SETTINGS) * len(HORIZONS) * 2


def calibrations(batch, predictions):
    response = batch.later_residual + batch.later_population
    result = {}
    for model in ("continuous_constant", "selected_constant"):
        for horizon in HORIZONS:
            records, observations = horizon_flat_mask(batch, horizon)
            sizes = np.diff(batch.later_indptr)[records]
            result[f"{model}_{int(horizon)}"] = fit_affine_calibration(response[observations],
                predictions[model]["flat_predictions"][observations], np.repeat(1.0 / sizes, sizes))
    return result


def score_all(batch, records, horizons, old_predictions, control_predictions, original, added):
    response = actual_response(records, horizons)
    output = {}
    for model in ("continuous", "constant", "continuous_constant", "selected_constant"):
        prediction = (old_predictions if model in ("continuous", "constant") else control_predictions)[model]
        output[f"{model}_direct"] = score_predictions(batch, prediction["flat_predictions"], response)
        affine = np.empty(len(response))
        for horizon in HORIZONS:
            _, selected = horizon_flat_mask(batch, horizon)
            coefficients = (original if model in ("continuous", "constant") else added)[f"{model}_{int(horizon)}"]
            affine[selected] = coefficients["intercept"] + coefficients["slope"] * prediction["flat_predictions"][selected]
        output[f"{model}_affine"] = score_predictions(batch, affine, response)
    for scale in ("direct", "affine"):
        output[f"selected_{scale}"] = output[f"{old_predictions['selected_kind']}_{scale}"]
    return output


def run_replicate(payload):
    source = Path(payload["source"]) / "replicates" / payload["setting"] / f"rep_{payload['replicate']:04d}"
    output = Path(payload["output"]) / "replicates" / payload["setting"] / f"rep_{payload['replicate']:04d}"
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete.json"
    identity = {"version": VERSION, "setting": payload["setting"], "replicate": payload["replicate"]}
    if complete.exists():
        retained = json.loads(complete.read_text())
        if any(retained[key] != value for key, value in identity.items()):
            raise RuntimeError("The control checkpoint has a different protocol")
        for name in ("frozen.json", "methods.csv", "paired.csv", "assessment_metrics.parquet"):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return retained
    started = time.perf_counter()
    original = json.loads((source / "frozen.json").read_text())
    seeds = original["derived_seeds"]
    sequence = np.random.SeedSequence([20261022, SETTINGS.index(payload["setting"]), payload["replicate"]])
    expected = [int(child.generate_state(1)[0]) for child in sequence.spawn(4)]
    if seeds != expected:
        raise RuntimeError("The original data seeds differ from the fixed simulation design")
    beta = np.asarray(original["population_coefficients"], dtype=float)
    data = simulate_dataset(payload["setting"], 600, seeds[0])
    fitting, tuning, assessment = split_indices(600, seeds[1])
    if len(fitting) != 360 or len(tuning) != 120 or len(assessment) != 120:
        raise RuntimeError("The original fitting allocation differs")
    tune_records, tune_horizons, tune_ids = combined_records(build_stays(data, tuning, beta), tuning)
    tuning_batch = make_profile_batch(tune_records, tune_horizons)
    frozen_path = output / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen["identity"] != identity or frozen["derived_seeds"] != seeds:
            raise RuntimeError("The frozen constant control uses different inputs")
        fitted, affine = frozen["control"], frozen["post_calibrations"]
    else:
        fitted = fit_continuous_constant_control(tuning_batch, tune_horizons, tune_ids,
            original["learning"], diagnostic_directory=output / "numerical_diagnostics/tuning")
        affine = calibrations(tuning_batch, predict_continuous_constant_control(tuning_batch, fitted))
        frozen = {"identity": identity, "derived_seeds": seeds, "population_coefficients": beta,
                  "control": fitted, "post_calibrations": affine,
                  "source_learning_seed": original["learning"]["seed"]}
        write_json(frozen_path, jsonable(frozen))
    expected_methods = pd.read_csv(source / "methods.csv")
    expected_assessment = pd.read_parquet(source / "assessment_metrics.parquet")
    reference = simulate_dataset(payload["setting"], 2000, seeds[2])
    methods, paired, assessment_frames, checks = [], [], [], []
    maximum_replay_error, maximum_probability_error = 0.0, 0.0
    for scope, population, indices in (("assessment", data, assessment), ("reference", reference, np.arange(2000))):
        records, horizons, identities = combined_records(build_stays(population, indices, beta), indices)
        batch = make_profile_batch(records, horizons)
        old = predict_selection(records, horizons, original["learning"])
        old["selected_kind"] = original["learning"]["selected_kind"]
        added = predict_continuous_constant_control(batch, fitted)
        scores = score_all(batch, records, horizons, old, added, original["post_calibrations"], affine)
        checks.extend(values["diagnostics"] for key, values in old.items() if key != "selected_kind")
        checks.extend(values["diagnostics"] for values in added.values())
        for horizon in HORIZONS:
            selected = horizons == horizon
            for model, values in scores.items():
                means = [float(values[key][selected].mean()) for key in ("loss", "p_lt", "p_le")]
                if model in {"continuous_direct", "continuous_affine", "selected_direct", "selected_affine", "constant_direct", "constant_affine"}:
                    row = expected_methods[(expected_methods.scope == scope) & (expected_methods.horizon_hours == horizon) & (expected_methods.method == model)]
                    if len(row) != 1:
                        raise RuntimeError("A frozen history comparison row is missing")
                    expected_loss = float(row.mean_loss.iloc[0])
                    maximum_replay_error = max(maximum_replay_error, abs(means[0] - expected_loss))
                    np.testing.assert_allclose(means[0], expected_loss, rtol=0, atol=1e-8)
                    for metric, observed in zip(("p_lt", "p_le"), means[1:]):
                        probability = float(row[metric].iloc[0])
                        maximum_probability_error = max(maximum_probability_error, abs(observed - probability))
                        np.testing.assert_allclose(observed, probability, rtol=0, atol=2e-12)
                    if scope == "assessment":
                        saved = expected_assessment[(expected_assessment.horizon_hours == horizon) & (expected_assessment.method == model)]
                        np.testing.assert_array_equal(saved.synthetic_stay_index, identities[selected])
                        np.testing.assert_allclose(saved.loss, values["loss"][selected], rtol=0, atol=1e-8)
                        for metric in ("p_lt", "p_le"):
                            np.testing.assert_allclose(saved[metric], values[metric][selected], rtol=0, atol=2e-12)
                methods.append({"scope": scope, "horizon_hours": horizon, "method": model,
                    "n_stays": int(selected.sum()), "mean_loss": means[0], "p_lt": means[1], "p_le": means[2],
                    "bracket_violation": max(means[1] - .1, .1 - means[2], 0),
                    "within_sample_se": float(values["loss"][selected].std(ddof=1) / np.sqrt(selected.sum()))})
                if scope == "assessment":
                    assessment_frames.append(pd.DataFrame({"horizon_hours": horizon, "method": model,
                        "synthetic_stay_index": identities[selected], **{key: values[key][selected] for key in ("loss", "p_lt", "p_le")}}))
            for first, second in CONTRASTS:
                differences = scores[first]["loss"][selected] - scores[second]["loss"][selected]
                paired.append({"scope": scope, "horizon_hours": horizon, "first": first, "second": second,
                    "n_stays": int(selected.sum()), "mean_difference": float(differences.mean()),
                    "within_sample_se": float(differences.std(ddof=1) / np.sqrt(selected.sum())),
                    "gain_proportion": float(np.mean(differences < 0))})
    if not all(row["all_certified"] for row in checks):
        raise RuntimeError("An uncertified profile entered a control comparison")
    pd.DataFrame(methods).to_csv(output / "methods.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired.csv", index=False)
    pd.concat(assessment_frames, ignore_index=True).to_parquet(output / "assessment_metrics.parquet", index=False)
    completed = {**identity, "status": "complete", "derived_seeds": seeds,
        "selected_kind": fitted["selected_kind"], "maximum_history_replay_error": maximum_replay_error,
        "maximum_history_probability_error": maximum_probability_error,
        "all_profiles_certified": True, "zero_history_coefficients": True,
        "constant_penalties_within_horizon": True,
        "optimizer_success": fitted["continuous_fit"]["optimizer_success"],
        "optimizer_evaluations": sum(row["fit"]["evaluations"] for row in fitted["validation_rows"]) + fitted["continuous_fit"]["evaluations"],
        "runtime_seconds": time.perf_counter() - started}
    write_json(complete, completed)
    return completed


def aggregate(output, replicates):
    frames = {name: [] for name in ("methods", "paired")}
    completions, optimizer_rows, validation_rows, penalty_rows = [], [], [], []
    for setting in SETTINGS:
        for number in range(1, replicates + 1):
            folder = output / "replicates" / setting / f"rep_{number:04d}"
            row = json.loads((folder / "complete.json").read_text())
            if row["status"] != "complete":
                raise RuntimeError("An independent replicate is incomplete")
            completions.append(row)
            frozen = json.loads((folder / "frozen.json").read_text())["control"]
            fits = [(f"fold_{record['fold']}", record["fit"]) for record in frozen["validation_rows"]]
            fits.append(("full_tuning", frozen["continuous_fit"]))
            for stage, fitted in fits:
                optimizer_rows.append({"setting": setting, "replicate": number, "stage": stage,
                    "optimizer_success": fitted["optimizer_success"], "evaluations": fitted["evaluations"],
                    "projected_gradient_norm": fitted["projected_gradient_norm"],
                    "profile_solves": fitted["all_evaluation_diagnostics"]["profiles"],
                    "max_duality_gap": fitted["all_evaluation_diagnostics"]["max_duality_gap"],
                    "max_projected_kkt_violation": fitted["all_evaluation_diagnostics"]["max_projected_kkt_violation"],
                    "all_profiles_certified": fitted["all_evaluation_diagnostics"]["all_certified"]})
            validation_rows.append({"setting": setting, "replicate": number,
                "selected_kind": frozen["selected_kind"], "grid_validation_score": frozen["grid_validation_score"],
                "continuous_validation_score": frozen["continuous_validation_score"]})
            horizon_features = np.c_[np.eye(3), np.zeros((3, 3))]
            penalties = penalty_values(np.asarray(frozen["continuous_fit"]["params"]), horizon_features)[0]
            for index, horizon in enumerate(HORIZONS):
                penalty_rows.append({"setting": setting, "replicate": number, "horizon_hours": horizon,
                    "lambda_level": penalties[index, 0], "lambda_slope": penalties[index, 1],
                    "selected_kind": frozen["selected_kind"]})
            for name in frames:
                frames[name].append(pd.read_csv(folder / f"{name}.csv").assign(setting=setting, replicate=number))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}_replicates.csv", index=False)
    methods, paired = [], []
    for key, frame in tables["methods"].groupby(["setting", "scope", "horizon_hours", "method"], sort=False):
        if len(frame) != replicates or frame.replicate.nunique() != replicates:
            raise RuntimeError("Method aggregation omits a replicate")
        methods.append({**dict(zip(("setting", "scope", "horizon_hours", "method"), key)), "replicates": replicates,
            "mean_loss": float(frame.mean_loss.mean()), "mcse": float(frame.mean_loss.std(ddof=1) / np.sqrt(replicates)),
            "p_lt": float(frame.p_lt.mean()), "p_le": float(frame.p_le.mean())})
    for key, frame in tables["paired"].groupby(["setting", "scope", "horizon_hours", "first", "second"], sort=False):
        if len(frame) != replicates or frame.replicate.nunique() != replicates:
            raise RuntimeError("Paired aggregation omits a replicate")
        differences = frame.mean_difference.to_numpy()
        mean, se = differences.mean(), differences.std(ddof=1) / np.sqrt(replicates)
        point = t.ppf(.975, replicates - 1) * se
        simultaneous = t.ppf(1 - .05 / (2 * COMPARISON_COUNT), replicates - 1) * se
        paired.append({**dict(zip(("setting", "scope", "horizon_hours", "first", "second"), key)),
            "replicates": replicates, "mean_difference": mean, "mcse": se,
            "pointwise_low": mean - point, "pointwise_high": mean + point,
            "simultaneous_low": mean - simultaneous, "simultaneous_high": mean + simultaneous,
            "gain_frequency": float(np.mean(differences < 0)),
            "leave_one_out_maximum_mean": float(((differences.sum() - differences) / (replicates - 1)).max())})
    pd.DataFrame(methods).to_csv(output / "method_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(output / "paired_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(completions).to_csv(output / "replicate_manifest.csv", index=False)
    optimizer_frame = pd.DataFrame(optimizer_rows)
    optimizer_frame.to_csv(output / "optimizer_replicates.csv", index=False)
    optimizer_frame.groupby(["setting", "stage"]).agg(
        fits=("replicate", "size"), successful=("optimizer_success", "sum"),
        total_evaluations=("evaluations", "sum"), total_profiles=("profile_solves", "sum"),
        maximum_projected_gradient=("projected_gradient_norm", "max"),
        maximum_duality_gap=("max_duality_gap", "max"),
        maximum_projected_kkt=("max_projected_kkt_violation", "max")).reset_index().to_csv(output / "optimizer_summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output / "validation_scores.csv", index=False)
    pd.DataFrame(penalty_rows).to_csv(output / "penalty_replicates.csv", index=False)
    write_json(output / "verification.json", {"status": "complete" if replicates == 200 else "partial_design_check",
        "completed_replicates": len(completions), "planned_replicates": 800,
        "paired_comparisons": len(paired), "comparison_count": COMPARISON_COUNT,
        "all_profiles_certified": all(row["all_profiles_certified"] for row in completions),
        "maximum_history_replay_error": max(row["maximum_history_replay_error"] for row in completions),
        "maximum_history_probability_error": max(row["maximum_history_probability_error"] for row in completions),
        "constant_penalties_within_horizon": all(row["constant_penalties_within_horizon"] for row in completions),
        "successful_full_tuning_fits": sum(row["optimizer_success"] for row in completions),
        "all_history_coefficients_zero": all(row["zero_history_coefficients"] for row in completions)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 2 <= args.replicates <= 200:
        raise ValueError("Invalid control simulation worker or replicate count")
    args.output.mkdir(parents=True, exist_ok=True)
    design = {"version": VERSION, "replicates_per_setting": args.replicates, "settings": list(SETTINGS),
              "horizons": list(HORIZONS), "contrasts": [list(pair) for pair in CONTRASTS],
              "comparison_count": COMPARISON_COUNT, "source": str(args.source.resolve()),
              "max_iter": 100, "deterministic_starts": 2,
              "protocol": Path(__file__).with_name("CONTINUOUS_CONSTANT_PROTOCOL.md").read_text()}
    path = args.output / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise RuntimeError("The saved control design differs")
    write_json(path, design)
    write_json(args.output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__})
    if args.summarize_only:
        aggregate(args.output, args.replicates)
        return
    jobs = [{"source": str(args.source.resolve()), "output": str(args.output.resolve()), "setting": setting, "replicate": number}
            for setting in SETTINGS for number in range(1, args.replicates + 1)]
    if args.benchmark:
        print(json.dumps(run_replicate(jobs[0])), flush=True)
        return
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        remaining = iter(jobs)
        futures = {pool.submit(run_replicate, next(remaining)) for _ in range(min(args.workers, len(jobs)))}
        count = 0
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                count += 1
                if count % 20 == 0 or count == len(jobs):
                    print(f"Completed {count}/{len(jobs)} continuous constant replicates in {time.perf_counter()-started:.1f}s; last {result['setting']} {result['replicate']}", flush=True)
                job = next(remaining, None)
                if job is not None:
                    futures.add(pool.submit(run_replicate, job))
    aggregate(args.output, args.replicates)


if __name__ == "__main__":
    main()
