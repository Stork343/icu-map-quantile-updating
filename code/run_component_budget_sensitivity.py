from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
import os
import sys
import tempfile
import time


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--output", type=Path, default=Path.cwd() / "local_runs/component_budget")
BOOTSTRAP.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
CONFIGURATION, _ = BOOTSTRAP.parse_known_args()
OUTPUT = CONFIGURATION.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE = OUTPUT / "runtime"
CACHE.mkdir(exist_ok=True)
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TMPDIR"] = str(CACHE)
tempfile.tempdir = str(CACHE)
sys.path.insert(0, str(CONFIGURATION.code_dir.resolve()))

import numpy as np
import pandas as pd

from component_learning_selection import constant_loss_matrix, fit_selection, jsonable, predict_selection
from component_profile_learning import fit_component_map, make_profile_batch
from component_simulation_baselines import actual_response, combined_records, horizon_flat_mask, score_predictions
from run_adaptive_mixed_simulation import HORIZONS, build_stays, simulate_dataset
from run_component_learning_simulation import fit_post_calibrations


BUDGETS = (300, 500)


def evaluate(stays, ids, learning, calibrations, scope):
    records, horizons, identities = combined_records(stays, ids)
    batch = make_profile_batch(records, horizons)
    raw = actual_response(records, horizons)
    predictions = predict_selection(records, horizons, learning)
    outputs = {}
    for name in ("continuous", "constant"):
        prediction = predictions[name]["flat_predictions"]
        outputs[f"{name}_direct"] = score_predictions(batch, prediction, raw)
        affine = np.empty(len(raw))
        for horizon in HORIZONS:
            mask, observations = horizon_flat_mask(batch, horizon)
            coefficients = calibrations[f"{name}_{int(horizon)}"]
            affine[observations] = coefficients["intercept"] + coefficients["slope"] * prediction[observations]
        outputs[f"{name}_affine"] = score_predictions(batch, affine, raw)
    for form in ("direct", "affine"):
        outputs[f"selected_{form}"] = outputs[f"{learning['selected_kind']}_{form}"]
    rows = []
    for horizon in HORIZONS:
        mask = horizons == horizon
        for name, scores in outputs.items():
            rows.append({"scope": scope, "horizon_hours": horizon, "method": name,
                         "mean_loss": float(scores["loss"][mask].mean()),
                         "p_lt": float(scores["p_lt"][mask].mean()),
                         "p_le": float(scores["p_le"][mask].mean())})
    return rows


def refit_job(job):
    setting, number = job["setting"], int(job["replicate"])
    destination = OUTPUT / "budget_results" / setting / f"rep_{number:04d}.json"
    if destination.is_file():
        previous = json.loads(destination.read_text())
        assert previous["status"] == "complete" and previous["budgets"] == list(BUDGETS)
        assert previous["setting"] == setting and previous["replicate"] == number
        assert previous["redo_cv"] == bool(job["redo_cv"])
        source_frozen = json.loads((Path(job["source_directory"]) / "replicates" / setting
                                    / f"rep_{number:04d}/frozen.json").read_text())["learning"]
        assert previous["original_selected_kind"] == source_frozen["selected_kind"]
        assert previous["original_eta"] == source_frozen["best_continuous_eta"]
        assert previous["original_fullfit_objective"] == source_frozen["best_continuous"]["objective"]
        return {**previous, "reused_completed_result": True}
    started = time.perf_counter()
    directory = Path(job["source_directory"]) / "replicates" / setting / f"rep_{number:04d}"
    frozen = json.loads((directory / "frozen.json").read_text())
    learning = frozen["learning"]
    seeds = frozen["derived_seeds"]
    beta = np.asarray(frozen["population_coefficients"])
    data = simulate_dataset(setting, 600, seeds[0])
    order = np.random.default_rng(seeds[1]).permutation(600)
    tuning_ids, assessment_ids = order[360:480], order[480:]
    tuning_stays = build_stays(data, tuning_ids, beta)
    records, horizons, record_ids = combined_records(tuning_stays, tuning_ids)
    batch = make_profile_batch(records, horizons)
    assessment_stays = build_stays(data, assessment_ids, beta)
    reference = simulate_dataset(setting, 2000, seeds[2])
    reference_ids = np.arange(2000)
    reference_stays = build_stays(reference, reference_ids, beta)
    populations = (("assessment", assessment_stays, assessment_ids), ("reference", reference_stays, reference_ids))
    originals = pd.read_csv(directory / "methods.csv").set_index(["scope", "horizon_hours", "method"])
    original_replay = []
    for scope, stays, ids in populations:
        original_replay.extend(evaluate(stays, ids, learning, frozen["post_calibrations"], scope))
    replay = pd.DataFrame(original_replay).set_index(["scope", "horizon_hours", "method"]).sort_index()
    np.testing.assert_allclose(replay.mean_loss, originals.loc[replay.index].mean_loss, rtol=0, atol=1e-7)
    losses = constant_loss_matrix(batch)[0] if job["redo_cv"] else None
    results = []
    for budget in BUDGETS:
        fit_started = time.perf_counter()
        fitted = fit_component_map(batch, initial_penalties=np.asarray(learning["constant_penalties"]),
                                   ridge_eta=learning["best_continuous_eta"], max_iter=budget, deterministic_starts=2)
        assert fitted["profile_certificates_passed"] and fitted["all_evaluation_diagnostics"]["all_certified"]
        assert fitted["objective"] <= learning["best_continuous"]["objective"] + 1e-6
        updated = deepcopy(learning)
        updated["best_continuous"] = jsonable(fitted)
        updated["max_iter"] = budget
        candidates = [("fullfit_fixed_cv", updated)]
        if job["redo_cv"]:
            updated_cv = fit_selection(records, horizons, record_ids, seeds[3],
                                       precomputed_constant_losses=losses, max_iter=budget, deterministic_starts=2)
            candidates.append(("full_cv_refit", updated_cv))
        for mode, candidate in candidates:
            predictions = predict_selection(records, horizons, candidate)
            calibrations = fit_post_calibrations(batch, predictions)
            risk = []
            for scope, stays, ids in populations:
                risk.extend(evaluate(stays, ids, candidate, calibrations, scope))
            candidate_frame = pd.DataFrame(risk).set_index(["scope", "horizon_hours", "method"])
            constant = candidate_frame.loc[candidate_frame.index.get_level_values("method").str.startswith("constant_")]
            np.testing.assert_allclose(constant.mean_loss, originals.loc[constant.index].mean_loss, rtol=0, atol=1e-7)
            results.append({"budget": budget, "mode": mode, "selected_kind": candidate["selected_kind"],
                "eta": candidate["best_continuous_eta"],
                "selected_kind_changed": candidate["selected_kind"] != learning["selected_kind"],
                "eta_changed": candidate["best_continuous_eta"] != learning["best_continuous_eta"],
                "fullfit_objective": candidate["best_continuous"]["objective"],
                "fullfit_objective_change": candidate["best_continuous"]["objective"] - learning["best_continuous"]["objective"],
                "fullfit_optimizer_success": candidate["best_continuous"]["optimizer_success"],
                "fullfit_projected_gradient_norm": candidate["best_continuous"]["projected_gradient_norm"],
                "validation_scores": candidate["validation_scores"],
                "params": candidate["best_continuous"]["params"], "risk": risk,
                "all_profiles_certified": candidate["best_continuous"]["all_evaluation_diagnostics"]["all_certified"]})
        print(f"Budget {budget} finished {setting} {number}; elapsed {time.perf_counter()-fit_started:.2f}s", flush=True)
    report = {"status": "complete", "setting": setting, "replicate": number, "budgets": list(BUDGETS),
              "selection_reason": job["selection_reason"], "redo_cv": job["redo_cv"],
              "original_selected_kind": learning["selected_kind"], "original_eta": learning["best_continuous_eta"],
              "original_fullfit_objective": learning["best_continuous"]["objective"],
              "original_fullfit_optimizer_success": learning["best_continuous"]["optimizer_success"],
              "original_replay_maximum_loss_difference": float(np.max(np.abs(replay.mean_loss - originals.loc[replay.index].mean_loss))),
              "seconds": time.perf_counter() - started, "reused_completed_result": False, "results": results}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    design = json.loads((args.source / "design.json").read_text())
    assert design["master_seed"] == 20261022 and design["replicates_per_setting"] == 200
    assert design["version"] == "continuous_components_20261022_v1"
    jobs = pd.read_csv(args.jobs)
    assert len(jobs) == 211 and jobs.redo_cv.sum() == 15
    assert not jobs.duplicated(["setting", "replicate"]).any()
    jobs["source_directory"] = str(args.source.resolve())
    manifest = {"status": "running", "all_jobs": 211, "launched_jobs": len(jobs),
                "complete_cv_controls": int(jobs.redo_cv.sum()), "workers": args.workers,
                "budgets": list(BUDGETS), "assessment_or_reference_used_for_job_selection": False,
                "cv_control_rule": "First replicate in each setting, original fullfit termination status, and originally selected branch stratum.",
                "interpretation": "Optimization budget sensitivity on the original fitting/tuning/evaluation records; no additional independent simulation datasets."}
    (OUTPUT / "budget_execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(refit_job, job) for job in jobs.to_dict("records")]
        results = []
        for completed in as_completed(pending):
            results.append(completed.result())
            print(f"Complete {len(results)}/{len(jobs)}; wall seconds {time.perf_counter()-started:.1f}", flush=True)
    manifest["status"] = "complete"
    manifest["reused_result_records"] = sum(row.get("reused_completed_result", False) for row in results)
    manifest["computed_result_records"] = len(results) - manifest["reused_result_records"]
    manifest["seconds"] = time.perf_counter() - started
    (OUTPUT / "budget_execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
