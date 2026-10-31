from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import clarabel
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from adaptive_profile_core import (
    KAPPA,
    POLISH_SETTINGS,
    QP_SETTINGS,
    ProfileStay,
    adaptive_candidate_specs,
    apply_stump_selector,
    check_loss,
    empirical_quantile,
    fit_affine_calibration,
    fit_profile,
    fit_stump_selector,
    index_features,
    solve_spd_profile,
)
from expanded_family_core import candidate_specs
from split_window_analysis_core import profiled_intercept


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")
TYPE_NAMES = ("persistent_level", "transient_recovering", "persistent_shape")
TYPE_PROBABILITIES = (0.35, 0.35, 0.30)
METHODS = (
    "population", "calibrated_q10", "original_selected", "geometry_selected",
    "conditioned_selected", "adaptive_selected", "expanded_selected", "stump_original",
)
CONTRASTS = (
    ("expanded_selected", "original_selected"),
    ("expanded_selected", "stump_original"),
    ("expanded_selected", "calibrated_q10"),
    ("adaptive_selected", "original_selected"),
    ("adaptive_selected", "geometry_selected"),
    ("adaptive_selected", "conditioned_selected"),
    ("original_selected", "calibrated_q10"),
)
TAU = 0.10
HORIZONS = (4.0, 8.0, 12.0)
BASE_BETA = np.asarray([72.0, -2.0, 1.0, -1.8])
SUMMARY_NAMES = ("last", "minimum", "mean", "below65")


@dataclass
class MixedDataset:
    x: np.ndarray
    index_y: list[np.ndarray]
    index_time: list[np.ndarray]
    later_y: np.ndarray
    later_time: np.ndarray
    latent_type: np.ndarray
    latent_level: np.ndarray
    latent_slope: np.ndarray
    coverage: np.ndarray


@dataclass(frozen=True)
class SimulationStay(ProfileStay):
    index_y: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        values = np.asarray(self.index_y, dtype=float)
        if values.shape != self.index_residual.shape or not np.all(np.isfinite(values)):
            raise ValueError("Original index outcomes must match the residual record")
        object.__setattr__(self, "index_y", values)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def saved_job(payload: dict) -> dict:
    result = {key: value for key, value in payload.items() if key != "work"}
    result["work"] = str(Path("replicates") / payload["setting"] / f"rep_{payload['replicate']:04d}")
    return result


def design_configuration() -> dict[str, object]:
    return {
        "version": "prespecified_20261021_v1", "master_seed": 20261021, "tau": TAU,
        "n_stays": 600, "replicates_per_setting": 200, "reference_stays": 2000,
        "fit_tune_assessment_fractions": [0.6, 0.2, 0.2], "landmark_hours": 12,
        "horizons_hours": list(HORIZONS), "settings": list(SETTINGS),
        "latent_type_names": list(TYPE_NAMES), "latent_type_probabilities": list(TYPE_PROBABILITIES),
        "baseline_beta": BASE_BETA.tolist(), "male_probability": 0.55, "urgent_probability": 0.65,
        "age_distribution": "standard_normal", "residual_scale": "4.5+0.5*abs(age_z)+0.9*urgent",
        "residual_marginal": "normal minus its 0.10 quantile", "latent_grid_hours": 0.25,
        "persistent_level": {"level_sd": 5.5, "slope": 0},
        "persistent_shape": {"level_sd": 3.5, "slope_sd": 5},
        "transient_recovering": {"index_level": "-5-abs(N(0,3^2))", "slope_mean": 8, "slope_sd": 2,
                                 "later_signal": 0},
        "weaker_recovery_change": {"transient_slope_mean": 3, "transient_slope_sd": 4},
        "information_limit": {"shared_index_level_sd": 5.5, "shared_index_slope_sd": 4,
                              "later_signals_by_type": ["b", "0", "b+s*u"]},
        "serial_integer_stress": {"ar1_rho_per_15_minutes": 0.82, "round_outcomes_to_integer": True},
        "index_count_values": [6, 10, 16], "index_count_probabilities": [0.3, 0.4, 0.3],
        "index_coverage_hours": [4, 8, 12], "index_coverage_probabilities": [0.3, 0.4, 0.3],
        "index_sampling": "distinct uniform grid positions in coverage interval, always including 12h",
        "later_sampling": "four distinct uniform grid positions in each of (12,16],(16,20],(20,24]",
        "generated_outcome_bounds": [20, 200], "forecast_clipping": False,
        "calibration": "ordinary stay-weighted affine check loss, unrestricted intercept and slope",
        "fitting": "ordinary observation-weighted R quantreg br with primal-dual certificate",
        "original_base_candidates": candidate_specs(), "adaptive_base_candidates": adaptive_candidate_specs(),
        "additional_calibrated_summaries": list(SUMMARY_NAMES),
        "q10_alias": "scalar_0 and scalar_0__affine represent raw and calibrated q10",
        "kappa": KAPPA, "qp_settings": QP_SETTINGS, "profile_polish_settings": POLISH_SETTINGS,
        "stump_features": ["rank-half q10 change", "log index count", "normalized time variance"],
        "stump_cutpoints": "fitting-sample feature quartiles", "stump_leaf_pool": "complete original family",
        "reference_target": "risk of each selected frozen history-measurable rule; no oracle regret is reported",
    }


def simulate_dataset(setting: str, n_stays: int, seed: int) -> MixedDataset:
    if setting not in SETTINGS or n_stays < 20:
        raise ValueError("Unknown setting or insufficient number of stays")
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n_stays), rng.normal(size=n_stays),
                         rng.binomial(1, 0.55, n_stays), rng.binomial(1, 0.65, n_stays)])
    latent_type = rng.choice(3, n_stays, p=TYPE_PROBABILITIES)
    # 同一个总体中同时包含三种持续类型，观测数量与覆盖时间独立变化。
    if setting == "information_limit":
        level, slope = rng.normal(0, 5.5, n_stays), rng.normal(0, 4, n_stays)
    else:
        level, slope = rng.normal(0, 5.5, n_stays), np.zeros(n_stays)
        shape = latent_type == 2
        level[shape], slope[shape] = rng.normal(0, 3.5, shape.sum()), rng.normal(0, 5, shape.sum())
        transient = latent_type == 1
        level[transient] = -5 - np.abs(rng.normal(0, 3, transient.sum()))
        mean_slope, sd_slope = (3, 4) if setting == "weaker_recovery" else (8, 2)
        slope[transient] = rng.normal(mean_slope, sd_slope, transient.sum())
    counts = rng.choice([6, 10, 16], n_stays, p=[0.3, 0.4, 0.3])
    coverage = rng.choice([4.0, 8.0, 12.0], n_stays, p=[0.3, 0.4, 0.3])
    grid = np.arange(97) / 4.0
    index_y, index_time, later_y, later_time = [], [], [], []
    for i in range(n_stays):
        candidate = np.flatnonzero((grid >= 12 - coverage[i]) & (grid < 12))
        early = np.sort(np.r_[rng.choice(candidate, counts[i] - 1, replace=False), 48])
        future = np.concatenate([np.sort(rng.choice(np.arange(start, start + 16), 4, replace=False))
                                 for start in (49, 65, 81)])
        innovation = rng.normal(size=grid.size)
        if setting == "serial_integer_stress":
            error = innovation.copy()
            for j in range(1, grid.size):
                error[j] = 0.82 * error[j - 1] + np.sqrt(1 - 0.82**2) * innovation[j]
        else:
            error = innovation
        u = (grid - 12) / 12
        early_signal = level[i] + slope[i] * u
        if latent_type[i] == 0:
            later_signal = np.full(grid.size, level[i])
        elif latent_type[i] == 1:
            later_signal = np.zeros(grid.size)
        else:
            later_signal = level[i] + slope[i] * u
        signal = np.where(grid <= 12, early_signal, later_signal)
        scale = 4.5 + 0.5 * abs(x[i, 1]) + 0.9 * x[i, 3]
        y = np.clip(x[i] @ BASE_BETA + signal + scale * (error - norm.ppf(TAU)), 20, 200)
        if setting == "serial_integer_stress":
            y = np.rint(y)
        index_y.append(y[early])
        index_time.append(grid[early])
        later_y.append(y[future])
        later_time.append(grid[future])
    return MixedDataset(x, index_y, index_time, np.asarray(later_y), np.asarray(later_time),
                        latent_type, level, slope, coverage)


def split_indices(n_stays: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_stays)
    fitting, tuning = int(0.6 * n_stays), int(0.2 * n_stays)
    return order[:fitting], order[fitting:fitting + tuning], order[fitting + tuning:]


def save_synthetic(data: MixedDataset, path: Path, fit: np.ndarray, tune: np.ndarray, assessment: np.ndarray) -> None:
    width = max(map(len, data.index_y))
    y, times = np.full((len(data.index_y), width), np.nan), np.full((len(data.index_y), width), np.nan)
    for i, (values, observed) in enumerate(zip(data.index_y, data.index_time)):
        y[i, :values.size], times[i, :values.size] = values, observed
    np.savez_compressed(path, x=data.x, index_y=y, index_time=times, later_y=data.later_y,
                        later_time=data.later_time, latent_type=data.latent_type,
                        latent_level=data.latent_level, latent_slope=data.latent_slope,
                        coverage=data.coverage, fitting_indices=fit, tuning_indices=tune,
                        assessment_indices=assessment)


def fit_population(data: MixedDataset, indices: np.ndarray, work: Path) -> np.ndarray:
    frames = []
    for i in indices:
        y = np.r_[data.index_y[i], data.later_y[i]]
        x = np.repeat(data.x[i][None, :], y.size, axis=0)
        frames.append(pd.DataFrame({"group_id": 0, "y": y, "intercept": x[:, 0],
                                    "age_z": x[:, 1], "male": x[:, 2], "urgent": x[:, 3]}))
    design_path, coefficient_path = work / "fitting_design.csv", work / "population_coefficients.csv"
    pd.concat(frames, ignore_index=True).to_csv(design_path, index=False)
    runtime = work / "r_runtime"
    runtime.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = runtime.name
    with (work / "r_quantreg.log").open("w", encoding="utf-8") as log:
        subprocess.run(["Rscript", str(CODE_DIR / "fit_grouped_quantile_inference.R"),
                        str(design_path.resolve()), str(TAU), str(coefficient_path.resolve())],
                       cwd=work, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    coefficients = pd.read_csv(coefficient_path)
    if set(coefficients["term"]) != {"intercept", "age_z", "male", "urgent"}:
        raise RuntimeError("R quantile fitting returned unexpected coefficients")
    beta = coefficients.set_index("term").loc[["intercept", "age_z", "male", "urgent"], "estimate"].to_numpy()
    design_path.unlink()
    return beta


def build_stays(data: MixedDataset, indices: np.ndarray, beta: np.ndarray) -> list[ProfileStay]:
    return [SimulationStay(data.index_y[i] - data.x[i] @ beta, data.index_time[i], data.later_y[i],
                           data.later_time[i], np.full(12, data.x[i] @ beta), index_y=data.index_y[i]) for i in indices]


def summary_prediction(stay: ProfileStay, name: str) -> float:
    y = stay.index_y
    summaries = {"last": y[-1], "minimum": np.min(y), "mean": np.mean(y), "below65": np.mean(y < 65)}
    return float(summaries[name])


def update_diagnostics(aggregate: dict[str, float | int], result: dict[str, object]) -> None:
    aggregate["qp_fits"] += 1
    for name in ("duality_gap", "relative_duality_gap", "projected_kkt_violation", "stationarity_violation"):
        aggregate[f"max_{name}"] = max(float(aggregate[f"max_{name}"]), float(result[name]))
    aggregate["iterations_total"] += int(result["iterations"])


def coefficient(stay: ProfileStay, spec: dict[str, object], horizon: float,
                diagnostics: dict[str, float | int]) -> np.ndarray:
    basis = spec["basis"]
    if basis == "none":
        return np.zeros(2)
    if basis == "level":
        return np.asarray([profiled_intercept(stay.index_residual, TAU, float(spec["lambda_level"])), 0.0])
    if basis == "adaptive":
        answer, certificate = fit_profile(stay, horizon, spec, TAU)
    elif basis == "level_slope":
        design = np.column_stack([np.ones(stay.index_time.size), (stay.index_time - 12) / 12])
        penalty = np.diag([float(spec["lambda_level"]), float(spec["lambda_slope"])])
        answer, certificate = solve_spd_profile(stay.index_residual, design, TAU, penalty)
    else:
        raise ValueError(f"Unknown basis {basis}")
    update_diagnostics(diagnostics, certificate)
    return answer


def base_predictions(stays: list[ProfileStay], spec: dict[str, object], horizon: float,
                     cache: dict[tuple[str, float], np.ndarray], diagnostics: dict[str, float | int]) -> np.ndarray:
    key = (str(spec["key"]), horizon if spec["basis"] == "adaptive" else 0.0)
    if key not in cache:
        if spec["basis"] == "summary":
            cache[key] = np.asarray([np.full(12, summary_prediction(stay, str(spec["summary"]))) for stay in stays])
        elif spec["basis"] == "level" and float(spec["lambda_level"]) == 0:
            cache[key] = np.asarray([np.full(12, empirical_quantile(stay.index_y, TAU)) for stay in stays])
        else:
            coefficients = np.asarray([coefficient(stay, spec, horizon, diagnostics) for stay in stays])
            cache[key] = np.asarray([stay.later_population + value[0] + value[1] * (stay.later_time - 12) / 12
                                     for stay, value in zip(stays, coefficients)])
    return cache[key]


def evaluate_matrix(stays: list[ProfileStay], predictions: np.ndarray, horizon: float) -> dict[str, np.ndarray]:
    y = np.asarray([stay.later_y for stay in stays])
    selected = np.asarray([stay.later_time <= 12 + horizon for stay in stays])
    counts = selected.sum(axis=1)
    if not np.all(counts == int(horizon)) or predictions.shape != y.shape or not np.all(np.isfinite(predictions)):
        raise RuntimeError("Future opportunities or forecast dimensions violate the fixed design")
    return {"loss": np.sum(check_loss(y - predictions, TAU) * selected, axis=1) / counts,
            "p_lt": np.sum((y < predictions) * selected, axis=1) / counts,
            "p_le": np.sum((y <= predictions) * selected, axis=1) / counts}


def tuning_rules(stays: list[ProfileStay], horizon: float, cache: dict,
                 diagnostics: dict) -> tuple[list[dict], np.ndarray, list[dict]]:
    old = [{**spec, "family": "original"} for spec in candidate_specs()]
    adaptive = [{**spec, "basis": "adaptive", "family": "adaptive"} for spec in adaptive_candidate_specs()]
    summaries = [{"key": f"summary_{name}", "basis": "summary", "summary": name, "family": "original"}
                 for name in SUMMARY_NAMES]
    specs = old + adaptive + summaries
    rule_rows, loss_columns, calibration_rows = [], [], []
    y = np.asarray([stay.later_y for stay in stays])
    selected = np.asarray([stay.later_time <= 12 + horizon for stay in stays])
    weight = np.repeat(1 / horizon, int(selected.sum()))
    for spec in specs:
        prediction = base_predictions(stays, spec, horizon, cache, diagnostics)
        if spec["basis"] != "summary":
            metric = evaluate_matrix(stays, prediction, horizon)
            rule_rows.append({"name": str(spec["key"]), "spec": spec, "intercept": 0.0, "slope": 1.0,
                              "calibrated": False, "tuning_loss": float(metric["loss"].mean())})
            loss_columns.append(metric["loss"])
        calibration = fit_affine_calibration(y[selected], prediction[selected], weight, TAU)
        calibrated = calibration["intercept"] + calibration["slope"] * prediction
        metric = evaluate_matrix(stays, calibrated, horizon)
        name = str(spec["key"]) + "__affine"
        rule_rows.append({"name": name, "spec": spec, "intercept": calibration["intercept"],
                          "slope": calibration["slope"], "calibrated": True,
                          "tuning_loss": float(metric["loss"].mean())})
        loss_columns.append(metric["loss"])
        calibration_rows.append({"horizon": horizon, "name": name, **calibration})
    if len(rule_rows) != 200 or len({row["name"] for row in rule_rows}) != 200:
        raise RuntimeError("The complete tuning menu must contain 200 canonical rules")
    return rule_rows, np.column_stack(loss_columns), calibration_rows


def select_methods(rules: list[dict], tuning_loss: np.ndarray, fitting_features: np.ndarray,
                   tuning_features: np.ndarray) -> tuple[dict[str, dict], dict]:
    old_indices = [i for i, rule in enumerate(rules) if rule["spec"]["family"] == "original"]
    old = [rules[i] for i in old_indices]
    adaptive = [rule for rule in rules if rule["spec"]["family"] == "adaptive"]
    selected = {
        "population": next(rule for rule in rules if rule["name"] == "population"),
        "calibrated_q10": next(rule for rule in rules if rule["name"] == "scalar_0__affine"),
        "original_selected": min(old, key=lambda rule: rule["tuning_loss"]),
        "geometry_selected": min([rule for rule in adaptive if float(rule["spec"]["gamma"]) == 0],
                                  key=lambda rule: rule["tuning_loss"]),
        "conditioned_selected": min([rule for rule in adaptive if float(rule["spec"]["theta"]) == 0],
                                     key=lambda rule: rule["tuning_loss"]),
        "adaptive_selected": min(adaptive, key=lambda rule: rule["tuning_loss"]),
        "expanded_selected": min(rules, key=lambda rule: rule["tuning_loss"]),
    }
    stump = fit_stump_selector(fitting_features, tuning_features, tuning_loss[:, old_indices],
                              [rule["name"] for rule in old])
    return selected, stump


def assess_selected(stays: list[ProfileStay], horizon: float, selected: dict[str, dict], stump: dict,
                    all_rules: list[dict], cache: dict, diagnostics: dict) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    by_name = {rule["name"]: rule for rule in all_rules}
    for method, rule in selected.items():
        raw = base_predictions(stays, rule["spec"], horizon, cache, diagnostics)
        output[method] = evaluate_matrix(stays, rule["intercept"] + rule["slope"] * raw, horizon)
    assignment = apply_stump_selector(stump, np.asarray([index_features(stay) for stay in stays]))
    prediction = np.zeros((len(stays), 12))
    for index in np.unique(assignment):
        rule = by_name[stump["rule_names"][int(index)]]
        raw = base_predictions(stays, rule["spec"], horizon, cache, diagnostics)
        rows = assignment == index
        prediction[rows] = rule["intercept"] + rule["slope"] * raw[rows]
    output["stump_original"] = evaluate_matrix(stays, prediction, horizon)
    return output


def replicate_job(payload: dict) -> dict:
    started = time.perf_counter()
    work = Path(payload["work"])
    work.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence([payload["seed"], payload["setting_index"], payload["replicate"]])
    seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(3)]
    write_json(work / "started.json", {**saved_job(payload), "derived_seeds": {"dataset": seeds[0], "split": seeds[1], "reference": seeds[2]}})
    data = simulate_dataset(payload["setting"], payload["n_stays"], seeds[0])
    fit, tune, assessment = split_indices(payload["n_stays"], seeds[1])
    save_synthetic(data, work / "synthetic_records.npz", fit, tune, assessment)
    beta = fit_population(data, fit, work)
    fitting_stays, tuning_stays, assessment_stays = [build_stays(data, indices, beta) for indices in (fit, tune, assessment)]
    reference_data = simulate_dataset(payload["setting"], payload["reference_stays"], seeds[2])
    reference_stays = build_stays(reference_data, np.arange(payload["reference_stays"]), beta)
    fitting_features = np.asarray([index_features(stay) for stay in fitting_stays])
    tuning_features = np.asarray([index_features(stay) for stay in tuning_stays])
    diagnostics = {"qp_fits": 0, "iterations_total": 0, "max_duality_gap": 0.0,
                   "max_relative_duality_gap": 0.0, "max_projected_kkt_violation": 0.0,
                   "max_stationarity_violation": 0.0}
    tuning_cache, assessment_cache, reference_cache = {}, {}, {}
    method_rows, paired_rows, selection_rows, all_tuning, all_calibration, stay_rows = [], [], [], [], [], []
    for horizon in payload["horizons"]:
        rules, loss_matrix, calibrations = tuning_rules(tuning_stays, horizon, tuning_cache, diagnostics)
        selected, stump = select_methods(rules, loss_matrix, fitting_features, tuning_features)
        write_json(work / f"frozen_h{int(horizon)}.json", {"rules": selected, "stump": stump})
        evaluated = assess_selected(assessment_stays, horizon, selected, stump, rules, assessment_cache, diagnostics)
        reference = assess_selected(reference_stays, horizon, selected, stump, rules, reference_cache, diagnostics)
        all_tuning.extend({"horizon": horizon, "name": rule["name"], "tuning_loss": rule["tuning_loss"],
                           "family": rule["spec"]["family"], "calibrated": rule["calibrated"]} for rule in rules)
        all_calibration.extend(calibrations)
        for method in METHODS:
            for scope, values in (("assessment", evaluated[method]), ("reference", reference[method])):
                loss, lower, upper = [float(values[name].mean()) for name in ("loss", "p_lt", "p_le")]
                method_rows.append({"horizon": horizon, "method": method, "scope": scope,
                                    "n_stays": len(values["loss"]), "loss": loss, "p_lt": lower, "p_le": upper,
                                    "violation": max(lower - TAU, TAU - upper, 0),
                                    "within_sample_se": float(np.std(values["loss"], ddof=1) / np.sqrt(len(values["loss"])) )})
            chosen = selected[method] if method != "stump_original" else {"name": "stump", "tuning_loss": stump["tuning_loss"]}
            selection_rows.append({"horizon": horizon, "method": method, "rule": chosen["name"],
                                   "tuning_loss": chosen["tuning_loss"],
                                   "stump_feature": stump["feature"] if method == "stump_original" else -2,
                                   "stump_left_rule": stump["left_rule_name"] if method == "stump_original" else "",
                                   "stump_right_rule": stump["right_rule_name"] if method == "stump_original" else ""})
            for local, original in enumerate(assessment):
                stay_rows.append({"horizon": horizon, "method": method, "synthetic_stay_index": int(original),
                                  "latent_type": TYPE_NAMES[data.latent_type[original]],
                                  **{name: float(value[local]) for name, value in evaluated[method].items()}})
        for first, second in CONTRASTS:
            for scope, values in (("assessment", evaluated), ("reference", reference)):
                difference = values[first]["loss"] - values[second]["loss"]
                paired_rows.append({"horizon": horizon, "first": first, "second": second, "scope": scope,
                                    "difference": float(difference.mean()),
                                    "within_sample_se": float(np.std(difference, ddof=1) / np.sqrt(difference.size))})
    for name, rows in (("methods", method_rows), ("paired", paired_rows), ("selections", selection_rows),
                       ("tuning", all_tuning), ("calibration", all_calibration)):
        pd.DataFrame(rows).to_csv(work / f"{name}.csv", index=False)
    pd.DataFrame(stay_rows).to_parquet(work / "assessment_stay_metrics.parquet", index=False)
    features = np.asarray([index_features(stay) for stay in build_stays(data, np.arange(payload["n_stays"]), beta)])
    latent_rows = [{"latent_type": TYPE_NAMES[k], "n_stays": int((data.latent_type == k).sum()),
                    "mean_marker": float(features[data.latent_type == k, 0].mean()),
                    "mean_level": float(data.latent_level[data.latent_type == k].mean()),
                    "mean_slope": float(data.latent_slope[data.latent_type == k].mean())} for k in range(3)]
    pd.DataFrame(latent_rows).to_csv(work / "synthetic_latent_diagnostics.csv", index=False)
    result = {**saved_job(payload), "derived_seeds": {"dataset": seeds[0], "split": seeds[1], "reference": seeds[2]},
              "n_fit": len(fit), "n_tune": len(tune), "n_assessment": len(assessment),
              "runtime_seconds": time.perf_counter() - started, "status": "complete", "diagnostics": diagnostics}
    write_json(work / "complete.json", result)
    return result


def aggregate(output: Path, jobs: list[dict]) -> None:
    frames = {name: [] for name in ("methods", "paired", "selections")}
    completions = []
    resolved_attempts, unresolved_failures = 0, 0
    for job in jobs:
        work = Path(job["work"])
        complete = json.loads((work / "complete.json").read_text())
        completions.append(complete)
        resolution_path = work / "resolution_summary.json"
        if resolution_path.exists():
            resolution = json.loads(resolution_path.read_text())
            resolved_attempts += int(resolution["resolved_numerical_attempts"])
            unresolved_failures += int(resolution["unresolved_failures"])
        else:
            unresolved_failures += len(list(work.glob("failure*.json")))
        for name in frames:
            frames[name].append(pd.read_csv(work / f"{name}.csv").assign(setting=job["setting"], replicate=job["replicate"]))
    for name, pieces in frames.items():
        frames[name] = pd.concat(pieces, ignore_index=True)
        frames[name].to_csv(output / f"{name}_replicates.csv", index=False)
    rows = []
    for keys, group in frames["methods"].groupby(["setting", "horizon", "method", "scope"], sort=False):
        row = dict(zip(("setting", "horizon", "method", "scope"), keys))
        row["replicates"] = len(group)
        for metric in ("loss", "p_lt", "p_le", "violation"):
            row[metric] = float(group[metric].mean())
            row[f"{metric}_mcse"] = float(group[metric].std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else 0.0
        row["mean_bracket_violation"] = max(row["p_lt"] - TAU, TAU - row["p_le"], 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "method_summary.csv", index=False)
    rows = []
    for keys, group in frames["paired"].groupby(["setting", "horizon", "first", "second", "scope"], sort=False):
        row = dict(zip(("setting", "horizon", "first", "second", "scope"), keys))
        row.update({"replicates": len(group), "mean_difference": float(group["difference"].mean()),
                    "mcse": float(group["difference"].std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else 0.0,
                    "mean_within_sample_se": float(group["within_sample_se"].mean())})
        row["mc_lower"], row["mc_upper"] = row["mean_difference"] - 1.96 * row["mcse"], row["mean_difference"] + 1.96 * row["mcse"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "paired_summary.csv", index=False)
    frequencies = frames["selections"].groupby(["setting", "horizon", "method", "rule"], sort=False).size().rename("count").reset_index()
    frequencies["frequency"] = frequencies["count"] / frequencies.groupby(["setting", "horizon", "method"])["count"].transform("sum")
    frequencies.to_csv(output / "selection_frequencies.csv", index=False)
    stump_rows = frames["selections"].loc[frames["selections"]["method"] == "stump_original"]
    stump_frequencies = stump_rows.groupby(["setting", "horizon", "stump_feature", "stump_left_rule", "stump_right_rule"],
                                           sort=False).size().rename("count").reset_index()
    stump_frequencies["frequency"] = stump_frequencies["count"] / stump_frequencies.groupby(["setting", "horizon"])["count"].transform("sum")
    stump_frequencies.to_csv(output / "stump_selection_frequencies.csv", index=False)
    write_json(output / "completion_manifest.json", {"expected_replicates": len(jobs), "completed_replicates": len(completions),
                                                     "resolved_numerical_attempts": resolved_attempts,
                                                     "unresolved_failures": unresolved_failures, "replicates": completions})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--n-stays", type=int, default=600)
    parser.add_argument("--reference-stays", type=int, default=2000)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--horizons", nargs="+", type=float, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.replicates < 1 or args.n_stays < 30 or args.reference_stays < 30:
        raise ValueError("Invalid simulation size or worker count")
    if len(set(args.settings)) != len(args.settings) or len(set(args.horizons)) != len(args.horizons):
        raise ValueError("Settings and horizons must be distinct")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = design_configuration()
    execution = {"replicates": args.replicates, "n_stays": args.n_stays, "reference_stays": args.reference_stays,
                 "settings": args.settings, "horizons": args.horizons, "seed": args.seed}
    config = {"prespecified_protocol": protocol, "execution": execution}
    config_path = output / "design.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("The output directory already contains a different simulation configuration")
    write_json(config_path, config)
    write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
                                           "pandas": pd.__version__, "scipy": scipy.__version__, "clarabel": clarabel.__version__, "workers": args.workers,
                                           "source": Path(__file__).name, "core": "adaptive_profile_core.py"})
    jobs = [{"setting": setting, "setting_index": SETTINGS.index(setting), "replicate": replicate,
             "seed": args.seed, "n_stays": args.n_stays, "reference_stays": args.reference_stays,
             "horizons": args.horizons, "work": str(output / "replicates" / setting / f"rep_{replicate:04d}")}
            for setting in args.settings for replicate in range(1, args.replicates + 1)]
    remaining = []
    for job in jobs:
        complete = Path(job["work"]) / "complete.json"
        if complete.exists():
            if not args.resume:
                raise RuntimeError("Completed output exists; explicit resume is required")
            recorded = json.loads(complete.read_text())
            if any(recorded[key] != value for key, value in saved_job(job).items()):
                raise RuntimeError("Completed replicate configuration differs from the requested job")
        else:
            remaining.append(job)
    print(f"Replicates scheduled={len(jobs)} remaining={len(remaining)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending_jobs = iter(remaining)
        futures = {}
        for _ in range(2 * args.workers):
            job = next(pending_jobs, None)
            if job is not None:
                futures[pool.submit(replicate_job, job)] = job
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                job = futures.pop(future)
                failure = future.exception()
                if failure is not None:
                    failure_paths = list(Path(job["work"]).glob("failure*.json"))
                    failure_name = "failure.json" if not failure_paths else f"failure_{len(failure_paths) + 1:04d}.json"
                    write_json(Path(job["work"]) / failure_name, {**saved_job(job), "status": "failed", "error_type": type(failure).__name__,
                                                                "error": str(failure)})
                result = future.result()
                print(f"Completed {result['setting']} replicate={result['replicate']} seconds={result['runtime_seconds']:.2f} qp={result['diagnostics']['qp_fits']}", flush=True)
                job = next(pending_jobs, None)
                if job is not None:
                    futures[pool.submit(replicate_job, job)] = job
    aggregate(output, jobs)
    print(f"Complete results={output}", flush=True)


if __name__ == "__main__":
    main()
