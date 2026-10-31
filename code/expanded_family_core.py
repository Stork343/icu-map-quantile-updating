from __future__ import annotations

import subprocess
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from expanded_profile_qp import QP_SETTINGS, solve_profile_qp
from generate_split_window_hard_result_tables import X_PREDICTORS, contiguous_stay_slices, strict_split_indices
from run_submission_validation_extensions import CALIBRATED_SUMMARIES, write_json
from split_window_analysis_core import check_loss, empirical_check_quantile, profiled_intercept


SCALAR_GRID = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LEVEL_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SLOPE_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PROFILE_KKT_TOLERANCE = 5e-7
MODEL_ORDER = (
    "population", "calibrated_population", "raw_q10", "scalar_profile", "calibrated_profile",
    "level_with_slope", "calibrated_level_with_slope",
    *CALIBRATED_SUMMARIES, "selected_family",
)


@dataclass
class StayLayout:
    global_indices: np.ndarray
    residual_index: list[np.ndarray]
    time_index: list[np.ndarray]
    later_values: np.ndarray
    later_basis: np.ndarray
    later_population: np.ndarray
    later_stay: np.ndarray
    later_starts: np.ndarray
    later_counts: np.ndarray
    summaries: pd.DataFrame

    @property
    def n_stays(self) -> int:
        return int(self.global_indices.size)


def build_layout(design: pd.DataFrame, beta: np.ndarray, lookup: Mapping[int, int], index_hours: float) -> StayLayout:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    population = design.loc[:, X_PREDICTORS].to_numpy(dtype=float) @ np.asarray(beta, dtype=float)
    global_indices, residuals, index_times, later, basis, later_pop, counts, summaries = [], [], [], [], [], [], [], []
    for start, end in contiguous_stay_slices(design):
        y, t, fitted = y_all[start:end], t_all[start:end], population[start:end]
        idx, late = strict_split_indices(t, index_hours)
        if idx.size < 4 or late.size < 1 or y.size < 8:
            raise RuntimeError("A layout stay does not satisfy the fixed 8/4/1 qualification")
        stay_id = int(design["stay_id"].iat[start])
        global_indices.append(int(lookup[stay_id]))
        residuals.append(y[idx] - fitted[idx])
        index_times.append(t[idx])
        later.append(y[late])
        basis.append((t[late] - index_hours) / index_hours)
        later_pop.append(fitted[late])
        counts.append(late.size)
        summaries.append({
            "index_last": float(y[idx][-1]), "index_min": float(np.min(y[idx])),
            "index_mean": float(np.mean(y[idx])), "index_q10": empirical_check_quantile(y[idx], 0.1),
            "index_below65_fraction": float(np.mean(y[idx] < 65.0)),
        })
    later_counts = np.asarray(counts, dtype=np.int64)
    return StayLayout(
        np.asarray(global_indices, dtype=np.int64), residuals, index_times,
        np.concatenate(later), np.concatenate(basis), np.concatenate(later_pop),
        np.repeat(np.arange(len(counts), dtype=np.int64), later_counts),
        np.r_[0, np.cumsum(later_counts)[:-1]], later_counts, pd.DataFrame(summaries),
    )


def candidate_specs() -> list[dict[str, object]]:
    specs = [{"key": "population", "basis": "none", "lambda_level": None, "lambda_slope": None}]
    specs.extend({"key": f"scalar_{i}", "basis": "level", "lambda_level": level, "lambda_slope": None}
                 for i, level in enumerate(SCALAR_GRID))
    specs.extend({"key": f"vector_{i}_{j}", "basis": "level_slope", "lambda_level": level, "lambda_slope": slope}
                 for i, level in enumerate(LEVEL_GRID) for j, slope in enumerate(SLOPE_GRID))
    return specs


def profile_coefficients(layout: StayLayout, spec: Mapping[str, object], tau: float, index_hours: float,
                         diagnostic_path: Path | None = None) -> tuple[np.ndarray, dict[str, object]]:
    coefficients = np.zeros((layout.n_stays, 2), dtype=float)
    gap_max = 0.0
    relative_gap_max = 0.0
    kkt_max = 0.0
    iterations_total = 0
    iterations_max = 0
    status_counts = {}
    polishing_counts = {}
    started = time.perf_counter()
    if spec["basis"] == "level":
        coefficients[:, 0] = [profiled_intercept(r, tau, float(spec["lambda_level"])) for r in layout.residual_index]
    elif spec["basis"] == "level_slope":
        penalties = np.asarray([spec["lambda_level"], spec["lambda_slope"]], dtype=float)
        for i, (residual, times) in enumerate(zip(layout.residual_index, layout.time_index)):
            design = np.column_stack([np.ones(residual.size), (times - index_hours) / index_hours])
            coefficients[i], diagnostics = solve_profile_qp(residual, design, tau, penalties, diagnostic_path)
            if not bool(diagnostics["success"]):
                raise RuntimeError(f"Uncertified profile solution for {spec['key']} at local stay {i}")
            gap_max = max(gap_max, float(diagnostics["duality_gap"]))
            relative_gap_max = max(relative_gap_max, float(diagnostics["relative_duality_gap"]))
            kkt_max = max(kkt_max, float(diagnostics["projected_kkt_violation"]))
            iterations_total += int(diagnostics["iterations"])
            iterations_max = max(iterations_max, int(diagnostics["iterations"]))
            status = str(diagnostics["status"])
            polishing = str(diagnostics["polishing_status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            polishing_counts[polishing] = polishing_counts.get(polishing, 0) + 1
    elif spec["basis"] != "none":
        raise ValueError(f"Unknown profile basis: {spec['basis']}")
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Nonfinite profile coefficients")
    return coefficients, {"max_duality_gap": gap_max, "max_relative_duality_gap": relative_gap_max,
                          "max_projected_kkt_violation": kkt_max, "kkt_tolerance": PROFILE_KKT_TOLERANCE,
                          "solver_method": "osqp_fenchel_dual_qp_polished" if spec["basis"] == "level_slope" else "exact_scalar",
                          "solver_settings": QP_SETTINGS if spec["basis"] == "level_slope" else {},
                          "solver_iterations_total": iterations_total, "solver_iterations_max": iterations_max,
                          "solver_status_counts": status_counts, "solver_polishing_status_counts": polishing_counts,
                          "runtime_seconds": time.perf_counter() - started, "n_stays": layout.n_stays}


def later_prediction(layout: StayLayout, coefficients: np.ndarray) -> np.ndarray:
    return layout.later_population + coefficients[layout.later_stay, 0] + coefficients[layout.later_stay, 1] * layout.later_basis


def stay_mean(layout: StayLayout, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != layout.later_values.shape:
        raise ValueError("Observation arrays must match the later-window layout")
    return np.add.reduceat(values, layout.later_starts) / layout.later_counts


def evaluate_prediction(layout: StayLayout, prediction: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Nonfinite later predictions")
    return {
        "loss": stay_mean(layout, check_loss(layout.later_values - prediction, tau)),
        "p_lt": stay_mean(layout, (layout.later_values < prediction).astype(float)),
        "p_le": stay_mean(layout, (layout.later_values <= prediction).astype(float)),
        "prediction_mean": stay_mean(layout, prediction),
    }


def _profile_worker(payload: tuple[StayLayout, dict[str, object], float, float, str]) -> dict[str, object]:
    layout, spec, tau, index_hours, path_text = payload
    path = Path(path_text)
    coefficients, diagnostics = profile_coefficients(layout, spec, tau, index_hours,
                                                       path.parent / "solver_diagnostics" / f"{spec['key']}.npz")
    np.save(path, coefficients, allow_pickle=False)
    losses = evaluate_prediction(layout, later_prediction(layout, coefficients), tau)["loss"]
    return {**spec, "raw_tuning_loss": float(np.mean(losses)), **diagnostics}


def build_profile_grid(layout: StayLayout, tau: float, index_hours: float, work_dir: Path, workers: int) -> list[dict[str, object]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs()
    payloads = [(layout, spec, tau, index_hours, str(work_dir / f"{spec['key']}.npy")) for spec in specs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_profile_worker, payloads):
            results.append(result)
            print(f"  {result['key']} tuning loss={result['raw_tuning_loss']:.6f} seconds={result['runtime_seconds']:.1f}", flush=True)
    return results


def fit_calibration_batch(layout: StayLayout, profile_dir: Path, work_dir: Path, tau: float, r_script: Path) -> dict[str, dict[str, float]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y": layout.later_values, "weight": 1.0 / layout.later_counts[layout.later_stay]})
    predictors = {}
    for spec in candidate_specs():
        if spec["key"] == "scalar_0":
            continue
        coefficients = np.load(profile_dir / f"{spec['key']}.npy", allow_pickle=False)
        predictors[str(spec["key"])] = later_prediction(layout, coefficients)
    for model, column in CALIBRATED_SUMMARIES.items():
        predictors[model] = layout.summaries[column].to_numpy(dtype=float)[layout.later_stay]
    frame = pd.concat([frame, pd.DataFrame(predictors)], axis=1)
    input_path, output_path = work_dir / "calibration_input.csv", work_dir / "calibration_coefficients.csv"
    frame.to_csv(input_path, index=False)
    r_environment = os.environ.copy()
    r_environment["TMPDIR"] = "."
    subprocess.run(["Rscript", str(r_script.resolve()), str(input_path.resolve()), str(tau), str(output_path.resolve())],
                   cwd=work_dir.resolve(), env=r_environment, check=True)
    fitted = pd.read_csv(output_path)
    if set(fitted["candidate"]) != set(predictors) or fitted["candidate"].duplicated().any():
        raise RuntimeError("Batch calibration candidate names are inconsistent")
    result = {}
    for row in fitted.itertuples(index=False):
        prediction = float(row.intercept) + float(row.slope) * predictors[row.candidate]
        losses = evaluate_prediction(layout, prediction, tau)["loss"]
        result[row.candidate] = {"intercept": float(row.intercept), "slope": float(row.slope),
                                 "tuning_loss": float(np.mean(losses))}
    result["scalar_0"] = dict(result["calibrated_q10"])
    write_json(work_dir / "calibrations.json", result)
    input_path.unlink()
    return result


def selected_rules(grid: Sequence[Mapping[str, object]], calibrations: Mapping[str, Mapping[str, float]], layout: StayLayout, tau: float) -> dict[str, dict[str, object]]:
    scalar = [dict(row) for row in grid if row["basis"] == "level"]
    vector = [dict(row) for row in grid if row["basis"] == "level_slope"]
    rules = {
        "population": {"key": "population", "calibrated": False, "tuning_loss": float(grid[0]["raw_tuning_loss"])},
        "calibrated_population": {"key": "population", "calibrated": True, "tuning_loss": float(calibrations["population"]["tuning_loss"])},
        "raw_q10": {"key": "raw_q10", "calibrated": False, "tuning_loss": float(np.mean(evaluate_prediction(layout, layout.summaries["index_q10"].to_numpy()[layout.later_stay], tau)["loss"]))},
    }
    for model, candidates, calibrated in (
        ("scalar_profile", scalar, False), ("calibrated_profile", scalar, True),
        ("level_with_slope", vector, False), ("calibrated_level_with_slope", vector, True),
    ):
        key = min(candidates, key=lambda item: float(calibrations[str(item["key"])]["tuning_loss"]) if calibrated else float(item["raw_tuning_loss"]))
        rules[model] = {**key, "calibrated": calibrated, "tuning_loss": float(calibrations[str(key["key"])]["tuning_loss"]) if calibrated else float(key["raw_tuning_loss"])}
    for model in CALIBRATED_SUMMARIES:
        rules[model] = {"key": model, "calibrated": True, "tuning_loss": float(calibrations[model]["tuning_loss"])}
    canonical_candidates = dict(rules)
    if rules["scalar_profile"]["key"] == "scalar_0":
        canonical_candidates.pop("scalar_profile")
    if rules["calibrated_profile"]["key"] == "scalar_0":
        canonical_candidates.pop("calibrated_profile")
    selected = min(canonical_candidates, key=lambda model: float(canonical_candidates[model]["tuning_loss"]))
    rules["selected_family"] = {**rules[selected], "selected_model": selected}
    return rules


def assess_rules(layout: StayLayout, rules: Mapping[str, Mapping[str, object]], calibrations: Mapping[str, Mapping[str, float]], tau: float, index_hours: float) -> tuple[pd.DataFrame, dict[str, object]]:
    result = pd.DataFrame({"global_stay_index": layout.global_indices, "late_obs": layout.later_counts,
                           "index_q10": layout.summaries["index_q10"].to_numpy()})
    coefficient_cache, diagnostics = {}, {}
    for model in MODEL_ORDER:
        rule = rules[model]
        key = str(rule["key"])
        if key in CALIBRATED_SUMMARIES:
            source = layout.summaries[CALIBRATED_SUMMARIES[key]].to_numpy(dtype=float)
            prediction = source[layout.later_stay]
        elif key in ("raw_q10", "scalar_0"):
            prediction = layout.summaries["index_q10"].to_numpy(dtype=float)[layout.later_stay]
        else:
            if key not in coefficient_cache:
                spec = next(item for item in candidate_specs() if item["key"] == key)
                coefficient_cache[key], diagnostics[key] = profile_coefficients(layout, spec, tau, index_hours)
            prediction = later_prediction(layout, coefficient_cache[key])
        if rule["calibrated"]:
            prediction = float(calibrations[key]["intercept"]) + float(calibrations[key]["slope"]) * prediction
        for metric, values in evaluate_prediction(layout, prediction, tau).items():
            result[f"{metric}_{model}"] = values
    return result, diagnostics


def aggregate_metrics(records: pd.DataFrame, scope: str, tau: float) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        p_lt, p_le = float(records[f"p_lt_{model}"].mean()), float(records[f"p_le_{model}"].mean())
        rows.append({"scope": scope, "model": model, "n_stays": records.shape[0],
                     "mean_stay_level_check_loss": float(records[f"loss_{model}"].mean()),
                     "p_y_lt_q_stay_equal": p_lt, "p_y_le_q_stay_equal": p_le,
                     "calibration_bracket_violation": max(p_lt - tau, tau - p_le, 0.0)})
    return pd.DataFrame(rows)
