#!/usr/bin/env python3
"""Submission-stage validation extensions for split-window ICU MAP analyses.

The independent analysis unit is always the ICU stay.  The script adds four
auditable extensions without changing the manuscript or any figure/simulation
generator:

1. Discrete-quantile calibration using the probability-mass bracket
   P(Y < q) <= tau <= P(Y <= q), with stay-equal primary summaries and
   observation-weighted sensitivity summaries.
2. Fixed observation-opportunity analyses using the first K later records.
3. Paired comparisons of simple index-window summaries, calibrated scalar
   rules, the population rule, and the primary profiled level update.
4. Five-fold nested cross-fitting.  Every outer fold is assessed once; within
   the remaining four folds, 75% of stays fit the population component and 25%
   select the penalty and scalar calibrations.  This reproduces a 60/20/20
   fit/tune/assess allocation in every outer fold.

Nested cross-fitting is internal validation, not external or never-seen data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import split_window_data as data_utils
from generate_split_window_hard_result_tables import X_PREDICTORS, contiguous_stay_slices, strict_split_indices
from run_split_window_mixed_effects_analysis import apply_training_age_standardization, split_cluster_indices
from split_window_analysis_core import (
    auc_score,
    check_loss,
    design_frame,
    empirical_check_quantile,
    fit_common_quantile,
    profiled_intercept,
)
from split_window_clinical_core import tune_lambda_generic
from split_window_data import _safe_read_frame, build_dataset_from_cache


CALIBRATED_SUMMARIES: Mapping[str, str] = {
    "calibrated_last": "index_last",
    "calibrated_min": "index_min",
    "calibrated_mean": "index_mean",
    "calibrated_q10": "index_q10",
    "calibrated_below65_burden": "index_below65_fraction",
}

RAW_MAP_MODELS: Mapping[str, str] = {
    "raw_last": "index_last",
    "raw_min": "index_min",
    "raw_mean": "index_mean",
    "raw_q10": "index_q10",
}

CORE_MODELS: Mapping[str, str] = {
    "population": "prediction_population",
    "primary_level_update": "prediction_primary_level_update",
}

OPERATIONAL_THRESHOLDS = (60.0, 65.0, 70.0)

CALIBRATION_WORK_KEYS: Mapping[str, str] = {
    "calibrated_below65_burden": "b65",
}

MODEL_DISPLAY_ORDER = (
    "population",
    "primary_level_update",
    "inner_selected_calibrated_rule",
    "calibrated_q10",
    "calibrated_last",
    "calibrated_min",
    "calibrated_mean",
    "calibrated_below65_burden",
    "raw_q10",
    "raw_last",
    "raw_min",
    "raw_mean",
)

MODEL_DISPLAY_LABELS: Mapping[str, str] = {
    "population": "Population rule",
    "primary_level_update": "Primary profiled offset",
    "inner_selected_calibrated_rule": "Inner-selected calibrated rule",
    "calibrated_q10": "Calibrated q10",
    "calibrated_last": "Calibrated last MAP",
    "calibrated_min": "Calibrated minimum MAP",
    "calibrated_mean": "Calibrated mean MAP",
    "calibrated_below65_burden": "Calibrated MAP <65 burden",
    "raw_q10": "Raw q10",
    "raw_last": "Raw last MAP",
    "raw_min": "Raw minimum MAP",
    "raw_mean": "Raw mean MAP",
}


def parse_float_grid(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def parse_int_grid(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid must contain positive integers")
    return values


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def se_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def interval_violation(p_strict_below: float, p_below_or_equal: float, tau: float) -> float:
    """Distance of tau from the discrete-quantile probability-mass bracket."""

    return float(max(float(p_strict_below) - tau, tau - float(p_below_or_equal), 0.0))


def balanced_fold_ids(n_stays: int, n_folds: int, seed: int) -> np.ndarray:
    if n_folds < 2 or n_folds > n_stays:
        raise ValueError("n_folds must be between 2 and n_stays")
    rng = np.random.default_rng(seed + 45007)
    permutation = rng.permutation(n_stays)
    fold_ids = np.empty(n_stays, dtype=int)
    fold_ids[permutation] = np.arange(n_stays, dtype=int) % n_folds
    return fold_ids


def inner_fit_tune_indices(outer_train: np.ndarray, fold: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    outer_train = np.asarray(outer_train, dtype=int)
    rng = np.random.default_rng(seed + 104729 * (fold + 1))
    shuffled = outer_train[rng.permutation(outer_train.size)]
    n_fit = int(round(0.75 * shuffled.size))
    n_fit = int(np.clip(n_fit, 1, shuffled.size - 1))
    return np.sort(shuffled[:n_fit]), np.sort(shuffled[n_fit:])


def global_index_lookup(dataset: Mapping[str, object]) -> Dict[int, int]:
    cluster_ids = np.asarray(dataset["cluster_ids"], dtype=np.int64)
    return {int(stay_id): int(index) for index, stay_id in enumerate(cluster_ids)}


def build_stay_records(
    design: pd.DataFrame,
    beta: np.ndarray,
    lambda_b: float,
    tau: float,
    index_hours: float,
    global_lookup: Mapping[int, int],
) -> pd.DataFrame:
    """Build one row per stay and retain later arrays only for in-process scoring."""

    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    fitted_all = design.loc[:, X_PREDICTORS].to_numpy(dtype=float) @ np.asarray(beta, dtype=float)
    rows: List[Dict[str, object]] = []

    for start, end in contiguous_stay_slices(design):
        y = y_all[start:end]
        t = t_all[start:end]
        fitted = fitted_all[start:end]
        index_idx, late_idx = strict_split_indices(t, index_hours)
        if index_idx.size == 0 or late_idx.size == 0:
            continue
        y_index = y[index_idx]
        y_late = y[late_idx]
        residual_index = y_index - fitted[index_idx]
        offset = profiled_intercept(residual_index, tau=tau, lambda_b=lambda_b)
        population_later_predictions = np.asarray(fitted[late_idx], dtype=float)
        primary_later_predictions = population_later_predictions + offset
        population_prediction = float(np.median(population_later_predictions))
        stay_id = int(design["stay_id"].iat[start])
        rows.append(
            {
                "local_stay_index": int(design["stay_index"].iat[start]),
                "stay_id": stay_id,
                "global_stay_index": int(global_lookup[stay_id]),
                "index_obs": int(index_idx.size),
                "late_obs": int(late_idx.size),
                "index_last": float(y_index[-1]),
                "index_min": float(np.min(y_index)),
                "index_mean": float(np.mean(y_index)),
                "index_q10": empirical_check_quantile(y_index, tau),
                "index_below65_fraction": float(np.mean(y_index < 65.0)),
                "prediction_population": population_prediction,
                "prediction_primary_level_update": float(np.median(primary_later_predictions)),
                "primary_offset": float(offset),
                "later_values": np.asarray(y_late, dtype=float),
                "later_times": np.asarray(t[late_idx], dtype=float),
                "later_population_predictions": population_later_predictions,
                "later_primary_level_update_predictions": primary_later_predictions,
            }
        )
    return pd.DataFrame(rows)


def synthetic_fit_design(design: pd.DataFrame) -> pd.DataFrame:
    """Prevent direct MIMIC identifiers from entering reusable fit work files."""

    out = design.copy()
    out["stay_id"] = out["stay_index"].to_numpy(dtype=np.int64)
    return out


def cleanup_fit_design(work_dir: Path) -> None:
    generated = work_dir / "ordinary_update_design.csv"
    if generated.exists():
        generated.unlink()


def fit_population_component(
    fit_design: pd.DataFrame,
    tau: float,
    work_dir: Path,
    r_script: Path,
    resume: bool,
) -> np.ndarray:
    metadata_path = work_dir / "population_coefficients.json"
    if resume and metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return np.asarray(payload["coefficients"], dtype=float)
    beta = fit_common_quantile(
        synthetic_fit_design(fit_design),
        X_PREDICTORS,
        tau=tau,
        work_dir=work_dir / "fit",
        r_script=r_script,
        force_refit=True,
    )
    cleanup_fit_design(work_dir / "fit")
    write_json(metadata_path, {"predictors": X_PREDICTORS, "tau": tau, "coefficients": beta.tolist()})
    return beta


def calibration_long_frame(records: pd.DataFrame, predictor_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y_parts: List[np.ndarray] = []
    predictor_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    stay_parts: List[np.ndarray] = []
    for row_index, row in records.reset_index(drop=True).iterrows():
        y_late = np.asarray(row["later_values"], dtype=float)
        predictor = float(row[predictor_column])
        y_parts.append(y_late)
        predictor_parts.append(np.full(y_late.size, predictor, dtype=float))
        weight_parts.append(np.full(y_late.size, 1.0 / y_late.size, dtype=float))
        stay_parts.append(np.full(y_late.size, row_index, dtype=np.int64))
    y = np.concatenate(y_parts)
    predictor = np.concatenate(predictor_parts)
    stay_index = np.concatenate(stay_parts)
    frame = pd.DataFrame(
        {
            "y": y,
            "stay_index": stay_index,
            "stay_id": stay_index,
            "time_hours": np.zeros(y.size, dtype=float),
            "index_flag": np.zeros(y.size, dtype=int),
            "late_flag": np.ones(y.size, dtype=int),
            "cal_intercept": np.ones(y.size, dtype=float),
            "cal_predictor": predictor,
        }
    )
    return frame, np.concatenate(weight_parts)


def fit_affine_calibration(
    records: pd.DataFrame,
    predictor_column: str,
    tau: float,
    work_dir: Path,
    r_script: Path,
    resume: bool,
) -> Dict[str, object]:
    metadata_path = work_dir / "calibration.json"
    if resume and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "tuning_mean_stay_level_check_loss" in cached:
            return cached
    frame, weights = calibration_long_frame(records, predictor_column)
    coefficients = fit_common_quantile(
        frame,
        ["cal_intercept", "cal_predictor"],
        tau=tau,
        work_dir=work_dir / "fit",
        r_script=r_script,
        force_refit=True,
        observation_weights=weights,
    )
    cleanup_fit_design(work_dir / "fit")
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    tuning_losses = [
        float(
            np.mean(
                check_loss(
                    np.asarray(row["later_values"], dtype=float)
                    - (intercept + slope * float(row[predictor_column])),
                    tau,
                )
            )
        )
        for _, row in records.iterrows()
    ]
    result = {
        "predictor_column": predictor_column,
        "intercept": intercept,
        "slope": slope,
        "tau": float(tau),
        "objective_weighting": "equal total weight per tuning stay",
        "tuning_mean_stay_level_check_loss": float(np.mean(tuning_losses)),
        "tuning_se_stay_level_check_loss": se_mean(tuning_losses),
    }
    write_json(metadata_path, result)
    return result


def fit_all_calibrations(
    tuning_records: pd.DataFrame,
    tau: float,
    work_dir: Path,
    r_script: Path,
    resume: bool,
) -> Dict[str, Dict[str, object]]:
    calibrations: Dict[str, Dict[str, object]] = {}
    for model, predictor_column in CALIBRATED_SUMMARIES.items():
        print(f"  fitting {model} from {predictor_column}", flush=True)
        model_work_key = CALIBRATION_WORK_KEYS.get(model, model)
        calibrations[model] = fit_affine_calibration(
            tuning_records,
            predictor_column,
            tau=tau,
            work_dir=work_dir / model_work_key,
            r_script=r_script,
            resume=resume,
        )
    return calibrations


def add_prediction_columns(
    records: pd.DataFrame,
    calibrations: Mapping[str, Mapping[str, object]],
    selected_calibration_model: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = records.copy()
    model_columns: Dict[str, str] = dict(CORE_MODELS)
    for model, source_column in RAW_MAP_MODELS.items():
        column = f"prediction_{model}"
        out[column] = out[source_column].to_numpy(dtype=float)
        model_columns[model] = column
    for model, calibration in calibrations.items():
        source_column = str(calibration["predictor_column"])
        column = f"prediction_{model}"
        out[column] = float(calibration["intercept"]) + float(calibration["slope"]) * out[source_column].to_numpy(
            dtype=float
        )
        model_columns[model] = column
    if selected_calibration_model is not None:
        if selected_calibration_model not in calibrations:
            raise ValueError(f"Unknown selected calibration model: {selected_calibration_model}")
        selected_column = f"prediction_{selected_calibration_model}"
        out["prediction_inner_selected_calibrated_rule"] = out[selected_column].to_numpy(dtype=float)
        out["inner_selected_calibration_model"] = selected_calibration_model
        model_columns["inner_selected_calibrated_rule"] = "prediction_inner_selected_calibrated_rule"
    return out, model_columns


def select_calibration_model(calibrations: Mapping[str, Mapping[str, object]]) -> str:
    """Choose from the frozen scalar menu using tuning stays only."""

    if not calibrations:
        raise ValueError("At least one calibration candidate is required")
    return min(
        calibrations,
        key=lambda model: (
            float(calibrations[model]["tuning_mean_stay_level_check_loss"]),
            list(CALIBRATED_SUMMARIES).index(model),
        ),
    )


def attach_model_metrics(
    records: pd.DataFrame,
    model_columns: Mapping[str, str],
    tau: float,
) -> pd.DataFrame:
    out = records.copy()
    for model, prediction_column in model_columns.items():
        losses: List[float] = []
        strict_fractions: List[float] = []
        inclusive_fractions: List[float] = []
        strict_counts: List[int] = []
        inclusive_counts: List[int] = []
        for _, row in out.iterrows():
            y = np.asarray(row["later_values"], dtype=float)
            if model == "population":
                prediction = np.asarray(row["later_population_predictions"], dtype=float)
            elif model == "primary_level_update":
                prediction = np.asarray(row["later_primary_level_update_predictions"], dtype=float)
            else:
                prediction = np.full(y.size, float(row[prediction_column]), dtype=float)
            if prediction.size != y.size:
                raise RuntimeError(f"Prediction length mismatch for {model}")
            losses.append(float(np.mean(check_loss(y - prediction, tau))))
            strict_counts.append(int(np.sum(y < prediction)))
            inclusive_counts.append(int(np.sum(y <= prediction)))
            strict_fractions.append(float(strict_counts[-1] / y.size))
            inclusive_fractions.append(float(inclusive_counts[-1] / y.size))
        out[f"loss_{model}"] = losses
        out[f"p_lt_{model}"] = strict_fractions
        out[f"p_le_{model}"] = inclusive_fractions
        out[f"count_lt_{model}"] = strict_counts
        out[f"count_le_{model}"] = inclusive_counts
    return out


def model_loss_summary(records: pd.DataFrame, model_columns: Mapping[str, str], scope: str) -> pd.DataFrame:
    rows = []
    for model in model_columns:
        losses = records[f"loss_{model}"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope,
                "model": model,
                "n_stays": int(losses.size),
                "mean_stay_level_check_loss": float(np.mean(losses)),
                "se_stay_level_check_loss": se_mean(losses),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(
    records: pd.DataFrame,
    model_columns: Mapping[str, str],
    scope: str,
    references: Sequence[str] = ("calibrated_q10", "primary_level_update"),
) -> pd.DataFrame:
    rows = []
    for reference in references:
        reference_losses = records[f"loss_{reference}"].to_numpy(dtype=float)
        for candidate in model_columns:
            candidate_losses = records[f"loss_{candidate}"].to_numpy(dtype=float)
            difference = candidate_losses - reference_losses
            mean_difference = float(np.mean(difference))
            difference_se = se_mean(difference)
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "reference": reference,
                    "n_stays": int(difference.size),
                    "candidate_mean_loss": float(np.mean(candidate_losses)),
                    "reference_mean_loss": float(np.mean(reference_losses)),
                    "paired_difference_candidate_minus_reference": mean_difference,
                    "paired_difference_se": difference_se,
                    "paired_difference_ci95_low": float(mean_difference - 1.96 * difference_se),
                    "paired_difference_ci95_high": float(mean_difference + 1.96 * difference_se),
                    "candidate_better_stay_fraction": float(np.mean(difference < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def stable_equal_count_groups(prediction: np.ndarray, stable_index: np.ndarray, groups: int) -> np.ndarray:
    order = np.lexsort((np.asarray(stable_index, dtype=np.int64), np.asarray(prediction, dtype=float)))
    labels = np.empty(order.size, dtype=int)
    labels[order] = np.minimum((np.arange(order.size) * groups) // order.size, groups - 1)
    return labels


def fixed_capacity_extreme_groups(
    prediction: np.ndarray,
    stable_index: np.ndarray,
    fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return disjoint low/high masks with deterministic tie resolution."""

    prediction = np.asarray(prediction, dtype=float)
    stable_index = np.asarray(stable_index, dtype=np.int64)
    if prediction.size != stable_index.size or prediction.size < 2:
        raise ValueError("prediction and stable_index must have the same length of at least two")
    group_size = max(1, int(round(fraction * prediction.size)))
    group_size = min(group_size, prediction.size // 2)
    order = np.lexsort((stable_index, prediction))
    low = np.zeros(prediction.size, dtype=bool)
    high = np.zeros(prediction.size, dtype=bool)
    low[order[:group_size]] = True
    high[order[-group_size:]] = True
    return low, high


def two_group_mean_difference(values: np.ndarray, first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    first_values = values[np.asarray(first, dtype=bool)]
    second_values = values[np.asarray(second, dtype=bool)]
    difference = float(np.mean(first_values) - np.mean(second_values))
    first_variance = float(np.var(first_values, ddof=1)) if first_values.size > 1 else 0.0
    second_variance = float(np.var(second_values, ddof=1)) if second_values.size > 1 else 0.0
    difference_se = float(np.sqrt(first_variance / first_values.size + second_variance / second_values.size))
    return {
        "difference": difference,
        "se": difference_se,
        "ci95_low": float(difference - 1.96 * difference_se),
        "ci95_high": float(difference + 1.96 * difference_se),
    }


def discrete_calibration(
    records: pd.DataFrame,
    model_columns: Mapping[str, str],
    tau: float,
    scope: str,
    groups: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    late_counts = records["late_obs"].to_numpy(dtype=int)
    stable_index = records["global_stay_index"].to_numpy(dtype=np.int64)

    for model, prediction_column in model_columns.items():
        predictions = records[prediction_column].to_numpy(dtype=float)
        labels = stable_equal_count_groups(predictions, stable_index, groups)
        model_rows: List[Dict[str, object]] = []
        masks: List[Tuple[str, np.ndarray]] = [("overall", np.ones(records.shape[0], dtype=bool))]
        masks.extend((f"decile_{group + 1}", labels == group) for group in range(groups))
        for group_label, mask in masks:
            if not np.any(mask):
                continue
            p_lt_stay = float(records.loc[mask, f"p_lt_{model}"].mean())
            p_le_stay = float(records.loc[mask, f"p_le_{model}"].mean())
            total_observations = int(np.sum(late_counts[mask]))
            p_lt_obs = float(records.loc[mask, f"count_lt_{model}"].sum() / total_observations)
            p_le_obs = float(records.loc[mask, f"count_le_{model}"].sum() / total_observations)
            row = {
                "scope": scope,
                "model": model,
                "group": group_label,
                "n_stays": int(np.sum(mask)),
                "n_later_observations": total_observations,
                "prediction_min": float(np.min(predictions[mask])),
                "prediction_median": float(np.median(predictions[mask])),
                "prediction_max": float(np.max(predictions[mask])),
                "p_y_lt_q_stay_equal": p_lt_stay,
                "p_y_le_q_stay_equal": p_le_stay,
                "interval_violation_stay_equal": interval_violation(p_lt_stay, p_le_stay, tau),
                "p_y_lt_q_observation_weighted": p_lt_obs,
                "p_y_le_q_observation_weighted": p_le_obs,
                "interval_violation_observation_weighted": interval_violation(p_lt_obs, p_le_obs, tau),
            }
            model_rows.append(row)
            detail_rows.append(row)

        overall = next(row for row in model_rows if row["group"] == "overall")
        deciles = [row for row in model_rows if row["group"].startswith("decile_")]
        decile_obs_weights = np.asarray([row["n_later_observations"] for row in deciles], dtype=float)
        summary_rows.append(
            {
                "scope": scope,
                "model": model,
                "n_stays": overall["n_stays"],
                "overall_p_y_lt_q_stay_equal": overall["p_y_lt_q_stay_equal"],
                "overall_p_y_le_q_stay_equal": overall["p_y_le_q_stay_equal"],
                "overall_interval_violation_stay_equal": overall["interval_violation_stay_equal"],
                "overall_p_y_lt_q_observation_weighted": overall["p_y_lt_q_observation_weighted"],
                "overall_p_y_le_q_observation_weighted": overall["p_y_le_q_observation_weighted"],
                "overall_interval_violation_observation_weighted": overall[
                    "interval_violation_observation_weighted"
                ],
                "mean_decile_interval_violation_stay_equal": float(
                    np.mean([row["interval_violation_stay_equal"] for row in deciles])
                ),
                "observation_weighted_mean_decile_interval_violation": float(
                    np.average(
                        [row["interval_violation_observation_weighted"] for row in deciles],
                        weights=decile_obs_weights,
                    )
                ),
                "calibration_rule": "P(Y<q) <= tau <= P(Y<=q)",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def fixed_opportunity_analysis(
    records: pd.DataFrame,
    model_columns: Mapping[str, str],
    tau: float,
    scope: str,
    k_grid: Sequence[int],
    thresholds: Sequence[float] = OPERATIONAL_THRESHOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: List[Dict[str, object]] = []
    loss_rows: List[Dict[str, object]] = []
    for k in k_grid:
        eligible = records.loc[records["late_obs"] >= int(k)].copy()
        if eligible.empty:
            continue
        y_first = [np.asarray(values, dtype=float)[: int(k)] for values in eligible["later_values"]]
        risk_score = -eligible["index_q10"].to_numpy(dtype=float)
        index_q10 = eligible["index_q10"].to_numpy(dtype=float)
        low_q10, high_q10 = fixed_capacity_extreme_groups(
            index_q10,
            eligible["global_stay_index"].to_numpy(dtype=np.int64),
        )
        low_cutpoint = float(np.max(index_q10[low_q10]))
        high_cutpoint = float(np.min(index_q10[high_q10]))

        for threshold in thresholds:
            any_event = np.asarray([np.any(values < threshold) for values in y_first], dtype=float)
            burden = np.asarray([np.mean(values < threshold) for values in y_first], dtype=float)
            risk_difference = two_group_mean_difference(any_event, low_q10, high_q10)
            burden_difference = two_group_mean_difference(burden, low_q10, high_q10)
            threshold_rows.append(
                {
                    "scope": scope,
                    "first_k_later_records": int(k),
                    "operational_threshold_mmhg": float(threshold),
                    "n_eligible_stays": int(eligible.shape[0]),
                    "fixed_later_observations_per_stay": int(k),
                    "any_event_rate": float(np.mean(any_event)),
                    "mean_within_stay_burden": float(np.mean(burden)),
                    "admission_q10_auc_for_any_event": auc_score(any_event, risk_score),
                    "low_q10_group_max": low_cutpoint,
                    "high_q10_group_min": high_cutpoint,
                    "low_q10_group_stays": int(np.sum(low_q10)),
                    "high_q10_group_stays": int(np.sum(high_q10)),
                    "low_q10_any_event_risk": float(np.mean(any_event[low_q10])),
                    "high_q10_any_event_risk": float(np.mean(any_event[high_q10])),
                    "low_minus_high_q10_risk_difference": risk_difference["difference"],
                    "low_minus_high_q10_risk_difference_se": risk_difference["se"],
                    "low_minus_high_q10_risk_difference_ci95_low": risk_difference["ci95_low"],
                    "low_minus_high_q10_risk_difference_ci95_high": risk_difference["ci95_high"],
                    "risk_difference_interval_interpretation": "approximate independent-stay normal 95% interval",
                    "low_minus_high_q10_burden_difference": burden_difference["difference"],
                    "low_minus_high_q10_burden_difference_se": burden_difference["se"],
                    "low_minus_high_q10_burden_difference_ci95_low": burden_difference["ci95_low"],
                    "low_minus_high_q10_burden_difference_ci95_high": burden_difference["ci95_high"],
                    "extreme_group_rule": "fixed 20% capacity; q10 ties resolved by stable deidentified stay index",
                    "threshold_role": "operational sensitivity only; check loss retains tau=0.10 MAP target",
                }
            )

        for model, prediction_column in model_columns.items():
            stay_losses_list: List[float] = []
            for (_, row), values in zip(eligible.iterrows(), y_first):
                if model == "population":
                    prediction = np.asarray(row["later_population_predictions"], dtype=float)[: int(k)]
                elif model == "primary_level_update":
                    prediction = np.asarray(row["later_primary_level_update_predictions"], dtype=float)[: int(k)]
                else:
                    prediction = np.full(int(k), float(row[prediction_column]), dtype=float)
                stay_losses_list.append(float(np.mean(check_loss(values - prediction, tau))))
            stay_losses = np.asarray(stay_losses_list, dtype=float)
            loss_rows.append(
                {
                    "scope": scope,
                    "first_k_later_records": int(k),
                    "model": model,
                    "n_eligible_stays": int(eligible.shape[0]),
                    "mean_stay_level_check_loss": float(np.mean(stay_losses)),
                    "se_stay_level_check_loss": se_mean(stay_losses),
                    "check_loss_target": f"tau={tau:.2f} MAP quantile; independent of operational threshold",
                }
            )
    return pd.DataFrame(threshold_rows), pd.DataFrame(loss_rows)


def common_cohort_fixed_opportunity_analysis(
    records: pd.DataFrame,
    tau: float,
    scope: str,
    k_grid: Sequence[int],
    thresholds: Sequence[float] = OPERATIONAL_THRESHOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Evaluate every K on one cohort eligible for the largest requested K."""

    k_values = sorted({int(k) for k in k_grid})
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_grid must contain positive integers")
    required_columns = {
        "late_obs",
        "later_values",
        "later_primary_level_update_predictions",
        "prediction_calibrated_q10",
        "index_q10",
        "global_stay_index",
    }
    missing = sorted(required_columns - set(records.columns))
    if missing:
        raise ValueError(f"Missing common-cohort columns: {missing}")

    common_min_later_records = max(k_values)
    eligible = records.loc[records["late_obs"] >= common_min_later_records].copy()
    eligible = eligible.sort_values("global_stay_index").reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No stays satisfy the common fixed-opportunity cohort rule")

    index_q10 = eligible["index_q10"].to_numpy(dtype=float)
    stable_index = eligible["global_stay_index"].to_numpy(dtype=np.int64)
    low_q10, high_q10 = fixed_capacity_extreme_groups(index_q10, stable_index)
    low_cutpoint = float(np.max(index_q10[low_q10]))
    high_cutpoint = float(np.min(index_q10[high_q10]))
    risk_score = -index_q10
    common_n = int(eligible.shape[0])

    threshold_rows: List[Dict[str, object]] = []
    loss_rows: List[Dict[str, object]] = []
    for k in k_values:
        y_first = [np.asarray(values, dtype=float)[:k] for values in eligible["later_values"]]
        if any(values.size != k for values in y_first):
            raise RuntimeError("Common cohort did not supply exactly K later records per stay")

        for threshold in thresholds:
            any_event = np.asarray([np.any(values < threshold) for values in y_first], dtype=float)
            burden = np.asarray([np.mean(values < threshold) for values in y_first], dtype=float)
            risk_difference = two_group_mean_difference(any_event, low_q10, high_q10)
            burden_difference = two_group_mean_difference(burden, low_q10, high_q10)
            threshold_rows.append(
                {
                    "scope": scope,
                    "cohort_rule": f"late_obs >= {common_min_later_records} for every K",
                    "common_min_later_records": common_min_later_records,
                    "first_k_later_records": k,
                    "operational_threshold_mmhg": float(threshold),
                    "n_common_cohort_stays": common_n,
                    "fixed_later_observations_per_stay": k,
                    "any_event_rate": float(np.mean(any_event)),
                    "mean_within_stay_burden": float(np.mean(burden)),
                    "admission_q10_auc_for_any_event": auc_score(any_event, risk_score),
                    "low_q10_group_max": low_cutpoint,
                    "high_q10_group_min": high_cutpoint,
                    "low_q10_group_stays": int(np.sum(low_q10)),
                    "high_q10_group_stays": int(np.sum(high_q10)),
                    "low_q10_any_event_risk": float(np.mean(any_event[low_q10])),
                    "high_q10_any_event_risk": float(np.mean(any_event[high_q10])),
                    "low_minus_high_q10_risk_difference": risk_difference["difference"],
                    "low_minus_high_q10_risk_difference_se": risk_difference["se"],
                    "low_minus_high_q10_risk_difference_ci95_low": risk_difference["ci95_low"],
                    "low_minus_high_q10_risk_difference_ci95_high": risk_difference["ci95_high"],
                    "risk_difference_interval_interpretation": "approximate independent-stay normal 95% interval",
                    "low_minus_high_q10_burden_difference": burden_difference["difference"],
                    "low_minus_high_q10_burden_difference_se": burden_difference["se"],
                    "low_minus_high_q10_burden_difference_ci95_low": burden_difference["ci95_low"],
                    "low_minus_high_q10_burden_difference_ci95_high": burden_difference["ci95_high"],
                    "extreme_group_rule": "fixed 20% capacity defined once in common cohort; q10 ties resolved by stable deidentified stay index",
                    "threshold_role": "operational sensitivity only; check loss retains tau=0.10 MAP target",
                }
            )

        calibrated_predictions = eligible["prediction_calibrated_q10"].to_numpy(dtype=float)
        calibrated_losses: List[float] = []
        primary_losses: List[float] = []
        for (_, row), values, calibrated_prediction in zip(
            eligible.iterrows(), y_first, calibrated_predictions
        ):
            primary_prediction = np.asarray(
                row["later_primary_level_update_predictions"], dtype=float
            )[:k]
            if primary_prediction.size != k:
                raise RuntimeError("Primary prediction did not supply exactly K later values")
            calibrated_losses.append(
                float(np.mean(check_loss(values - float(calibrated_prediction), tau)))
            )
            primary_losses.append(float(np.mean(check_loss(values - primary_prediction, tau))))
        calibrated_array = np.asarray(calibrated_losses, dtype=float)
        primary_array = np.asarray(primary_losses, dtype=float)
        paired_difference = primary_array - calibrated_array
        paired_se = se_mean(paired_difference)
        paired_mean = float(np.mean(paired_difference))
        loss_rows.append(
            {
                "scope": scope,
                "cohort_rule": f"late_obs >= {common_min_later_records} for every K",
                "common_min_later_records": common_min_later_records,
                "first_k_later_records": k,
                "n_common_cohort_stays": common_n,
                "calibrated_q10_mean_stay_level_check_loss": float(np.mean(calibrated_array)),
                "calibrated_q10_se_stay_level_check_loss": se_mean(calibrated_array),
                "primary_level_update_mean_stay_level_check_loss": float(np.mean(primary_array)),
                "primary_level_update_se_stay_level_check_loss": se_mean(primary_array),
                "paired_difference_primary_minus_calibrated_q10": paired_mean,
                "paired_difference_se": paired_se,
                "paired_difference_ci95_low": float(paired_mean - 1.96 * paired_se),
                "paired_difference_ci95_high": float(paired_mean + 1.96 * paired_se),
                "paired_interval_interpretation": "descriptive estimate plus or minus 1.96 stay-level SE, conditional on realized fold-trained rules; does not propagate refitting, tuning, or rule-selection variation",
                "primary_better_stay_fraction": float(np.mean(primary_array < calibrated_array)),
                "equal_loss_stay_fraction": float(np.mean(primary_array == calibrated_array)),
                "difference_direction": "positive favors calibrated q10",
                "check_loss_target": f"tau={tau:.2f} MAP quantile; independent of operational threshold",
            }
        )

    metadata = {
        "scope": scope,
        "cohort_rule": f"late_obs >= {common_min_later_records}",
        "common_min_later_records": common_min_later_records,
        "common_cohort_stays": common_n,
        "first_k_later_records": k_values,
        "operational_thresholds_mmhg": [float(value) for value in thresholds],
        "independent_unit": "ICU stay",
        "prediction_boundary": "nested out-of-fold predictions only",
        "grouping_rule": "one fixed 20% low/high admission-q10 grouping across all K and thresholds; deterministic deidentified-index tie break",
        "risk_difference_interval_interpretation": "approximate independent-stay normal 95% interval",
        "paired_check_loss_interval_interpretation": "descriptive estimate plus or minus 1.96 stay-level SE, conditional on realized fold-trained rules; does not propagate refitting, tuning, or rule-selection variation",
    }
    return pd.DataFrame(threshold_rows), pd.DataFrame(loss_rows), metadata


def rehydrate_common_cohort_records_from_frozen_oof(
    dataset: Mapping[str, object],
    frozen_predictions: pd.DataFrame,
    index_hours: float,
    min_later_records: int,
) -> pd.DataFrame:
    """Attach cached later outcomes to deidentified, frozen OOF scalar predictions."""

    required_prediction_columns = {
        "stay_index",
        "late_obs",
        "index_q10",
        "prediction_calibrated_q10",
        "prediction_primary_level_update",
        "outer_fold",
    }
    missing = sorted(required_prediction_columns - set(frozen_predictions.columns))
    if missing:
        raise ValueError(f"Frozen OOF prediction file is missing columns: {missing}")
    direct_identifiers = {"stay_id", "subject_id", "hadm_id"} & set(frozen_predictions.columns)
    if direct_identifiers:
        raise ValueError(f"Frozen OOF prediction file contains direct identifiers: {sorted(direct_identifiers)}")

    enriched = data_utils.ensure_cluster_lists(dict(dataset))
    y_list = [np.asarray(values, dtype=float) for values in enriched["y_list"]]
    t_list = [np.asarray(values, dtype=float) for values in enriched["t_list"]]
    n_stays = len(y_list)
    prediction = frozen_predictions.copy()
    prediction["stay_index"] = prediction["stay_index"].astype(np.int64)
    if prediction.shape[0] != n_stays or prediction["stay_index"].nunique() != n_stays:
        raise RuntimeError("Frozen OOF predictions must contain every analytic stay exactly once")
    if set(prediction["stay_index"].tolist()) != set(range(n_stays)):
        raise RuntimeError("Frozen OOF stay_index must match the analytic dataset's global index")
    prediction = prediction.loc[prediction["late_obs"] >= int(min_later_records)].sort_values(
        "stay_index"
    )

    rows: List[Dict[str, object]] = []
    for row in prediction.itertuples(index=False):
        global_index = int(row.stay_index)
        y = y_list[global_index]
        t = t_list[global_index]
        _, late_idx = strict_split_indices(t, index_hours)
        y_late = np.asarray(y[late_idx], dtype=float)
        if y_late.size != int(row.late_obs) or y_late.size < int(min_later_records):
            raise RuntimeError("Cached later-observation count disagrees with frozen OOF output")
        primary_scalar = float(row.prediction_primary_level_update)
        rows.append(
            {
                "global_stay_index": global_index,
                "outer_fold": int(row.outer_fold),
                "late_obs": int(y_late.size),
                "later_values": y_late,
                "later_primary_level_update_predictions": np.full(
                    y_late.size, primary_scalar, dtype=float
                ),
                "prediction_calibrated_q10": float(row.prediction_calibrated_q10),
                "index_q10": float(row.index_q10),
            }
        )
    records = pd.DataFrame(rows)
    if records["global_stay_index"].nunique() != records.shape[0]:
        raise RuntimeError("Rehydrated common cohort contains duplicate stays")
    return records


def public_prediction_frame(records: pd.DataFrame) -> pd.DataFrame:
    drop_columns = {
        "stay_id",
        "local_stay_index",
        "later_values",
        "later_times",
        "later_population_predictions",
        "later_primary_level_update_predictions",
    }
    out = records.drop(columns=[column for column in drop_columns if column in records.columns]).copy()
    out = out.rename(columns={"global_stay_index": "stay_index"})
    return out


def write_fixed_opportunity_tex(thresholds: pd.DataFrame, path: Path, preferred_scope: str) -> None:
    table = thresholds.loc[thresholds["scope"] == preferred_scope].copy()
    if table.empty:
        table = thresholds.copy()
    lines = [
        "\\begin{table*}[!htbp]",
        "\\caption{Fixed later-observation opportunity sensitivity across operational MAP thresholds.}",
        "\\label{tab:fixed_opportunity_threshold_sensitivity}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{rrrrrrr}",
        "\\hline",
        "$K$ & Threshold & Stays & Any event & Burden & q10 AUC & Low--high risk difference (95\\% CI)\\\\",
        "\\hline",
    ]
    for _, row in table.sort_values(["first_k_later_records", "operational_threshold_mmhg"]).iterrows():
        lines.append(
            f"{int(row['first_k_later_records'])} & {float(row['operational_threshold_mmhg']):.0f} & "
            f"{int(row['n_eligible_stays'])} & {100.0 * float(row['any_event_rate']):.1f}\\% & "
            f"{100.0 * float(row['mean_within_stay_burden']):.1f}\\% & "
            f"{float(row['admission_q10_auc_for_any_event']):.3f} & "
            f"{100.0 * float(row['low_minus_high_q10_risk_difference']):.1f} "
            f"({100.0 * float(row['low_minus_high_q10_risk_difference_ci95_low']):.1f} to "
            f"{100.0 * float(row['low_minus_high_q10_risk_difference_ci95_high']):.1f}) pp\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Rows aggregate nested five-fold out-of-fold predictions across the eligible cohort. Each row includes only stays with at least $K$ observations after the 12-hour landmark and uses exactly the first $K$ later records. Low and high groups each use a fixed 20\\% capacity; admission-window q10 ties are resolved deterministically by deidentified stay index. Risk-difference confidence intervals use the independent-stay normal approximation. Thresholds of 60, 65, and 70 mmHg are operational sensitivities; they do not change the primary ordinary check-loss target at $\\tau=0.10$.}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_common_cohort_fixed_opportunity_tex(
    thresholds: pd.DataFrame,
    losses: pd.DataFrame,
    threshold_path: Path,
    loss_path: Path,
) -> None:
    if thresholds.empty or losses.empty:
        raise ValueError("Common-cohort TeX outputs require nonempty result frames")
    threshold_lines = [
        "\\begin{table*}[!htbp]",
        "\\caption{Fixed observation-opportunity sensitivity in the common cohort with at least 12 later MAP records.}",
        "\\label{tab:common_cohort_fixed_opportunity_thresholds}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{rrrrrrr}",
        "\\hline",
        "$K$ & Threshold & Stays & Any event & Burden & q10 AUC & Low--high risk difference (95\\% CI)\\\\",
        "\\hline",
    ]
    for _, row in thresholds.sort_values(
        ["first_k_later_records", "operational_threshold_mmhg"]
    ).iterrows():
        threshold_lines.append(
            f"{int(row['first_k_later_records'])} & {float(row['operational_threshold_mmhg']):.0f} & "
            f"{int(row['n_common_cohort_stays'])} & {100.0 * float(row['any_event_rate']):.1f}\\% & "
            f"{100.0 * float(row['mean_within_stay_burden']):.1f}\\% & "
            f"{float(row['admission_q10_auc_for_any_event']):.3f} & "
            f"{100.0 * float(row['low_minus_high_q10_risk_difference']):.1f} "
            f"({100.0 * float(row['low_minus_high_q10_risk_difference_ci95_low']):.1f} to "
            f"{100.0 * float(row['low_minus_high_q10_risk_difference_ci95_high']):.1f}) pp\\\\"
        )
    threshold_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{All rows use the same nested out-of-fold cohort of stays with at least 12 later observations; within each stay, only the first $K=4$, 8, or 12 later records are used. The fixed 20\\% low/high admission-q10 groups are defined once in this common cohort, with ties resolved by deidentified stay index. Risk-difference confidence intervals use the independent-stay normal approximation. Thresholds are operational sensitivities and do not alter the $\\tau=0.10$ check-loss target.}",
            "\\end{table*}",
        ]
    )
    threshold_path.write_text("\n".join(threshold_lines) + "\n", encoding="utf-8")

    loss_lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Paired check loss in the common fixed-opportunity cohort.}",
        "\\label{tab:common_cohort_fixed_opportunity_check_loss}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{rrrrr}",
        "\\hline",
        "$K$ & Stays & Calibrated q10 & Profiled offset & Offset minus calibrated (descriptive interval)\\\\",
        "\\hline",
    ]
    for _, row in losses.sort_values("first_k_later_records").iterrows():
        loss_lines.append(
            f"{int(row['first_k_later_records'])} & {int(row['n_common_cohort_stays'])} & "
            f"{float(row['calibrated_q10_mean_stay_level_check_loss']):.4f} & "
            f"{float(row['primary_level_update_mean_stay_level_check_loss']):.4f} & "
            f"{float(row['paired_difference_primary_minus_calibrated_q10']):.4f} "
            f"({float(row['paired_difference_ci95_low']):.4f} to "
            f"{float(row['paired_difference_ci95_high']):.4f})\\\\"
        )
    loss_lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{All three rows compare the same stays with at least 12 later records using nested out-of-fold predictions. Loss is averaged within stay over exactly the first $K$ later records and then across stays. Differences are paired at the stay level; positive values favor calibrated q10. Each descriptive interval is the paired estimate plus or minus 1.96 times its stay-level standard error, conditional on the realized fold-trained rules; it does not propagate uncertainty from refitting, tuning, or rule selection.}",
            "\\end{table}",
        ]
    )
    loss_path.write_text("\n".join(loss_lines) + "\n", encoding="utf-8")


def write_common_cohort_fixed_opportunity_artifacts(
    thresholds: pd.DataFrame,
    losses: pd.DataFrame,
    metadata: Mapping[str, object],
    output_dir: Path,
) -> Tuple[Dict[str, object], Dict[str, str]]:
    common_threshold_csv = output_dir / "common_cohort_fixed_opportunity_threshold_sensitivity.csv"
    common_loss_csv = output_dir / "common_cohort_fixed_opportunity_check_loss_comparison.csv"
    common_threshold_tex = output_dir / "common_cohort_fixed_opportunity_threshold_sensitivity.tex"
    common_loss_tex = output_dir / "common_cohort_fixed_opportunity_check_loss_comparison.tex"
    common_json = output_dir / "common_cohort_fixed_opportunity_results.json"
    thresholds.to_csv(common_threshold_csv, index=False)
    losses.to_csv(common_loss_csv, index=False)
    write_common_cohort_fixed_opportunity_tex(
        thresholds,
        losses,
        common_threshold_tex,
        common_loss_tex,
    )
    results_payload: Dict[str, object] = {
        "status": "complete",
        "metadata": dict(metadata),
        "threshold_sensitivity": thresholds.to_dict("records"),
        "paired_check_loss": losses.to_dict("records"),
    }
    write_json(common_json, results_payload)
    artifacts = {
        "results_json": str(common_json),
        "threshold_csv": str(common_threshold_csv),
        "check_loss_csv": str(common_loss_csv),
        "threshold_tex": str(common_threshold_tex),
        "check_loss_tex": str(common_loss_tex),
    }
    return results_payload, artifacts


def write_paired_comparison_tex(comparisons: pd.DataFrame, path: Path, preferred_scope: str) -> None:
    table = comparisons.loc[
        (comparisons["scope"] == preferred_scope) & (comparisons["reference"] == "calibrated_q10")
    ].copy()
    if table.empty:
        table = comparisons.loc[comparisons["reference"] == "calibrated_q10"].copy()
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Paired stay-level check-loss comparisons against calibrated admission-window q10.}",
        "\\label{tab:validation_extension_paired_comparisons}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Candidate & Mean loss & Paired difference & Descriptive interval & Better stays\\\\",
        "\\hline",
    ]
    for _, row in table.sort_values("candidate_mean_loss").iterrows():
        label = str(row["candidate"]).replace("_", " ")
        lines.append(
            f"{label} & {float(row['candidate_mean_loss']):.4f} & "
            f"{float(row['paired_difference_candidate_minus_reference']):.4f} & "
            f"{float(row['paired_difference_ci95_low']):.4f} to {float(row['paired_difference_ci95_high']):.4f} & "
            f"{100.0 * float(row['candidate_better_stay_fraction']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Differences are candidate minus calibrated-q10 stay-level later check loss; negative values favor the candidate. Each descriptive interval is the estimate plus or minus 1.96 times its pooled stay-level standard error, conditional on the realized fold-trained rules; it does not propagate uncertainty from refitting, tuning, or rule selection. The independent unit is the ICU stay. The preferred scope is nested five-fold internal cross-fitting when available.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tie_aware_calibration_summary_tex(
    calibration_summary: pd.DataFrame,
    path: Path,
    preferred_scope: str,
) -> None:
    manuscript_models = ("population", "primary_level_update", "raw_q10", "calibrated_q10")
    table = calibration_summary.loc[
        (calibration_summary["scope"] == preferred_scope)
        & calibration_summary["model"].isin(manuscript_models)
    ].copy()
    order = {model: index for index, model in enumerate(manuscript_models)}
    table["display_order"] = table["model"].map(order)
    table = table.sort_values("display_order")
    if table.shape[0] != len(manuscript_models):
        raise RuntimeError("Manuscript calibration summary is missing a required model")
    lines = [
        "\\begin{table*}[!htbp]",
        "\\caption{Tie-aware calibration of later 0.10-quantile predictions under nested five-fold internal cross-fitting.}",
        "\\label{tab:tie_aware_calibration_summary}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Model & Stay $L$ & Stay $U$ & Stay $V$ & Obs. $L$ & Obs. $U$ & Obs. $V$ & Mean decile $V$\\\\",
        "\\hline",
    ]
    for _, row in table.iterrows():
        label = MODEL_DISPLAY_LABELS[str(row["model"])]
        lines.append(
            f"{label} & {float(row['overall_p_y_lt_q_stay_equal']):.4f} & "
            f"{float(row['overall_p_y_le_q_stay_equal']):.4f} & "
            f"{float(row['overall_interval_violation_stay_equal']):.4f} & "
            f"{float(row['overall_p_y_lt_q_observation_weighted']):.4f} & "
            f"{float(row['overall_p_y_le_q_observation_weighted']):.4f} & "
            f"{float(row['overall_interval_violation_observation_weighted']):.4f} & "
            f"{float(row['mean_decile_interval_violation_stay_equal']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{$L=P(Y<q)$ and $U=P(Y\\le q)$. The probability-mass calibration-bracket violation is $V=\\max\\{L-0.10,\\ 0.10-U,\\ 0\\}$, so outcome ties are not penalized when $L\\le0.10\\le U$. Stay-equal summaries are primary; observation-weighted summaries are sensitivity analyses. Mean decile $V$ is the unweighted mean stay-equal violation across ten deterministic equal-count prediction groups. Forecast ties at group boundaries are split by stable deidentified stay index; consequently, the mean-decile summary is descriptive and is not invariant to how those boundary ties are split. These quantities replace strict-below-only ACE/WCE summaries. The independent unit is the ICU stay.}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_tie_aware_decile_calibration(
    calibration_detail: pd.DataFrame,
    output_dir: Path,
    preferred_scope: str,
    tau: float,
) -> Dict[str, str]:
    """Plot strict/inclusive calibration intervals without treating strict-below as ACE."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source = calibration_detail.loc[
        (calibration_detail["scope"] == preferred_scope)
        & calibration_detail["group"].astype(str).str.startswith("decile_")
    ].copy()
    if source.empty:
        raise RuntimeError(f"No decile calibration rows for {preferred_scope}")
    source["decile"] = source["group"].str.replace("decile_", "", regex=False).astype(int)
    source["interval_definition"] = "L=P(Y<q), U=P(Y<=q); violation=max(L-tau,tau-U,0)"
    source_path = output_dir / "tie_aware_decile_calibration_source.csv"
    source.to_csv(source_path, index=False)

    for prefix in ("p_y_lt_q", "p_y_le_q"):
        for suffix in ("stay_equal", "observation_weighted"):
            values = source[f"{prefix}_{suffix}"].to_numpy(dtype=float)
            if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
                raise RuntimeError("Calibration probabilities must be finite and lie in [0, 1]")
    if np.any(
        source["p_y_lt_q_stay_equal"].to_numpy(dtype=float)
        > source["p_y_le_q_stay_equal"].to_numpy(dtype=float) + 1e-12
    ):
        raise RuntimeError("Tie-aware calibration endpoints are reversed")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    stay_color = "#2C6E9B"
    observation_color = "#D88935"
    neutral = "#404040"
    models_present = set(source["model"].astype(str))
    model_order = [model for model in MODEL_DISPLAY_ORDER if model in models_present]
    model_order.extend(sorted(models_present - set(model_order)))
    n_columns = 4
    n_rows = int(math.ceil(len(model_order) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(183.0 / 25.4, max(130.0, 48.0 * n_rows) / 25.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    upper_max = float(
        source[["p_y_le_q_stay_equal", "p_y_le_q_observation_weighted"]].to_numpy(dtype=float).max()
    )
    y_max = min(1.0, max(0.25, math.ceil((upper_max + 0.03) * 10.0) / 10.0))

    for panel_index, model in enumerate(model_order):
        ax = axes.flat[panel_index]
        local = source.loc[source["model"] == model].sort_values("decile")
        x = local["decile"].to_numpy(dtype=float)
        stay_l = local["p_y_lt_q_stay_equal"].to_numpy(dtype=float)
        stay_u = local["p_y_le_q_stay_equal"].to_numpy(dtype=float)
        obs_l = local["p_y_lt_q_observation_weighted"].to_numpy(dtype=float)
        obs_u = local["p_y_le_q_observation_weighted"].to_numpy(dtype=float)
        ax.axhline(tau, color=neutral, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.vlines(x - 0.10, stay_l, stay_u, color=stay_color, lw=2.0, zorder=3)
        ax.plot(x - 0.10, stay_l, linestyle="none", marker="_", ms=4.5, color=stay_color, zorder=4)
        ax.plot(x - 0.10, stay_u, linestyle="none", marker="_", ms=4.5, color=stay_color, zorder=4)
        ax.vlines(x + 0.10, obs_l, obs_u, color=observation_color, lw=1.1, ls="--", zorder=2)
        ax.plot(x + 0.10, obs_l, linestyle="none", marker="_", ms=3.8, color=observation_color, zorder=3)
        ax.plot(x + 0.10, obs_u, linestyle="none", marker="_", ms=3.8, color=observation_color, zorder=3)
        ax.set_title(MODEL_DISPLAY_LABELS.get(model, model.replace("_", " ")), pad=3.0)
        ax.set_xlim(0.45, 10.55)
        ax.set_ylim(0.0, y_max)
        ax.set_xticks([1, 3, 5, 7, 9, 10])
        ax.grid(axis="y", color="#E8E8E8", lw=0.45, zorder=0)
        if panel_index % n_columns == 0:
            ax.set_ylabel("Later probability")
        if panel_index // n_columns == n_rows - 1:
            ax.set_xlabel("Predicted-q decile")

    for panel_index in range(len(model_order), n_rows * n_columns):
        axes.flat[panel_index].axis("off")

    handles = [
        Line2D([0], [0], color=stay_color, lw=2.0, label="Stay-equal [L, U] (primary)"),
        Line2D([0], [0], color=observation_color, lw=1.2, ls="--", label="Observation-weighted [L, U] (sensitivity)"),
        Line2D([0], [0], color=neutral, lw=0.8, ls=(0, (3, 2)), label=f"Nominal tau = {tau:.2f}"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=6.0)
    fig.suptitle("Tie-aware decile calibration in nested five-fold internal cross-fitting", y=0.992, fontsize=8.0)
    fig.text(
        0.5,
        0.018,
        "Each vertical segment is [L, U], where L=P(Y<q) and U=P(Y<=q); tau inside the segment implies zero interval violation.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.995, 0.925), h_pad=1.0, w_pad=0.8)
    output_stem = output_dir / "tie_aware_decile_calibration"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    main_models = ("population", "primary_level_update", "raw_q10", "calibrated_q10")
    main_source = source.loc[source["model"].isin(main_models)].copy()
    main_upper = float(
        main_source[["p_y_le_q_stay_equal", "p_y_le_q_observation_weighted"]]
        .to_numpy(dtype=float)
        .max()
    )
    main_y_max = 0.38 if main_upper <= 0.38 else min(1.0, math.ceil((main_upper + 0.02) * 20.0) / 20.0)
    fig_main, axes_main = plt.subplots(
        1,
        4,
        figsize=(183.0 / 25.4, 76.0 / 25.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for panel_index, model in enumerate(main_models):
        ax = axes_main.flat[panel_index]
        local = main_source.loc[main_source["model"] == model].sort_values("decile")
        x = local["decile"].to_numpy(dtype=float)
        stay_l = local["p_y_lt_q_stay_equal"].to_numpy(dtype=float)
        stay_u = local["p_y_le_q_stay_equal"].to_numpy(dtype=float)
        obs_l = local["p_y_lt_q_observation_weighted"].to_numpy(dtype=float)
        obs_u = local["p_y_le_q_observation_weighted"].to_numpy(dtype=float)
        ax.axhline(tau, color=neutral, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.vlines(x - 0.10, stay_l, stay_u, color=stay_color, lw=2.0, zorder=3)
        ax.plot(x - 0.10, stay_l, linestyle="none", marker="_", ms=4.5, color=stay_color, zorder=4)
        ax.plot(x - 0.10, stay_u, linestyle="none", marker="_", ms=4.5, color=stay_color, zorder=4)
        ax.vlines(x + 0.10, obs_l, obs_u, color=observation_color, lw=1.1, ls="--", zorder=2)
        ax.plot(x + 0.10, obs_l, linestyle="none", marker="_", ms=3.8, color=observation_color, zorder=3)
        ax.plot(x + 0.10, obs_u, linestyle="none", marker="_", ms=3.8, color=observation_color, zorder=3)
        ax.set_title(MODEL_DISPLAY_LABELS[model], pad=3.0)
        ax.set_xlim(0.45, 10.55)
        ax.set_ylim(0.0, main_y_max)
        ax.set_xticks([1, 3, 5, 7, 9, 10])
        ax.set_xlabel("Predicted-q decile")
        ax.grid(axis="y", color="#E8E8E8", lw=0.45, zorder=0)
        if panel_index == 0:
            ax.set_ylabel("Later probability")
    fig_main.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.900), ncol=3, fontsize=6.0)
    fig_main.suptitle(
        "Tie-aware decile calibration in nested five-fold internal cross-fitting",
        y=0.990,
        fontsize=8.0,
    )
    fig_main.text(
        0.5,
        0.018,
        "Vertical segments show [L, U], with L=P(Y<q), U=P(Y<=q), and the dashed reference at tau=0.10.",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    fig_main.tight_layout(rect=(0.025, 0.095, 0.995, 0.790), h_pad=0.8, w_pad=0.8)
    main_output_stem = output_dir / "tie_aware_decile_calibration_main"
    fig_main.savefig(main_output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig_main.savefig(main_output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig_main.savefig(main_output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig_main)
    return {
        "source_csv": str(source_path),
        "full_pdf": str(output_stem.with_suffix(".pdf")),
        "full_svg": str(output_stem.with_suffix(".svg")),
        "full_png": str(output_stem.with_suffix(".png")),
        "main_pdf": str(main_output_stem.with_suffix(".pdf")),
        "main_svg": str(main_output_stem.with_suffix(".svg")),
        "main_png": str(main_output_stem.with_suffix(".png")),
    }


def run_fixed_split(
    dataset: Dict[str, object],
    stays: pd.DataFrame,
    primary_results: Mapping[str, object],
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
    work_dir: Path,
    r_script: Path,
    seed: int,
    resume: bool,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    enriched = data_utils.ensure_cluster_lists(dataset)
    train_idx, tuning_idx, assessment_idx = split_cluster_indices(
        len(enriched["y_list"]), seed=seed, train_fraction=0.60, tuning_fraction=0.20
    )
    standardized, age_info = apply_training_age_standardization(dataset, stays, train_idx)
    tuning_design, _ = design_frame(data_utils.subset_cluster_data(standardized, tuning_idx))
    assessment_design, _ = design_frame(data_utils.subset_cluster_data(standardized, assessment_idx))
    coefficient_key = f"baseline_tau_{tau:.2f}"
    beta = np.asarray(primary_results["coefficients"][coefficient_key], dtype=float)
    lambda_b = float(primary_results["settings"]["selected_lambda_baseline_update"])
    lookup = global_index_lookup(dataset)
    tuning_records = build_stay_records(tuning_design, beta, lambda_b, tau, index_hours, lookup)
    assessment_records = build_stay_records(assessment_design, beta, lambda_b, tau, index_hours, lookup)
    print("Fitting fixed-split affine clinical-summary calibrations", flush=True)
    calibrations = fit_all_calibrations(tuning_records, tau, work_dir / "calibrations", r_script, resume)
    selected_calibration_model = select_calibration_model(calibrations)
    assessment_records, model_columns = add_prediction_columns(
        assessment_records, calibrations, selected_calibration_model
    )
    assessment_records = attach_model_metrics(assessment_records, model_columns, tau)
    metadata = {
        "scope": "historical_assessment_split",
        "train_stays": int(train_idx.size),
        "tuning_stays": int(tuning_idx.size),
        "assessment_stays": int(assessment_idx.size),
        "age_standardization": age_info,
        "population_coefficients": beta.tolist(),
        "selected_lambda_b": lambda_b,
        "lambda_grid": [float(item) for item in lambda_grid],
        "calibrations": calibrations,
        "inner_selected_calibration_model": selected_calibration_model,
        "assessment_history_warning": "This nominal assessment split was inspected in earlier work.",
    }
    return assessment_records, model_columns, metadata


def run_nested_crossfit(
    dataset: Dict[str, object],
    stays: pd.DataFrame,
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
    n_folds: int,
    work_dir: Path,
    r_script: Path,
    seed: int,
    resume: bool,
) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict[str, object]]]:
    n_stays = len(data_utils.ensure_cluster_lists(dataset)["y_list"])
    fold_ids = balanced_fold_ids(n_stays, n_folds, seed)
    lookup = global_index_lookup(dataset)
    fold_records: List[pd.DataFrame] = []
    fold_metadata: List[Dict[str, object]] = []
    expected_model_columns: Dict[str, str] | None = None

    for fold in range(n_folds):
        fold_start = time.time()
        outer_assessment = np.flatnonzero(fold_ids == fold)
        outer_train = np.flatnonzero(fold_ids != fold)
        inner_fit, inner_tune = inner_fit_tune_indices(outer_train, fold, seed)
        if (
            np.intersect1d(inner_fit, inner_tune).size
            or np.intersect1d(outer_train, outer_assessment).size
            or np.union1d(inner_fit, inner_tune).size != outer_train.size
        ):
            raise RuntimeError("Nested cross-fitting partitions overlap")
        print(
            f"Outer fold {fold + 1}/{n_folds}: fit={inner_fit.size}, tune={inner_tune.size}, "
            f"assessment={outer_assessment.size}",
            flush=True,
        )
        fold_dir = work_dir / f"fold_{fold + 1}"
        standardized, age_info = apply_training_age_standardization(dataset, stays, inner_fit)
        fit_design, _ = design_frame(data_utils.subset_cluster_data(standardized, inner_fit))
        tuning_design, _ = design_frame(data_utils.subset_cluster_data(standardized, inner_tune))
        assessment_design, _ = design_frame(data_utils.subset_cluster_data(standardized, outer_assessment))
        beta = fit_population_component(fit_design, tau, fold_dir / "population", r_script, resume)
        tuning_profile = tune_lambda_generic(
            tuning_design,
            X_PREDICTORS,
            beta,
            tau=tau,
            index_hours=index_hours,
            lambda_grid=lambda_grid,
        )
        lambda_b = float(tuning_profile["best"]["lambda_b"])
        tuning_records = build_stay_records(tuning_design, beta, lambda_b, tau, index_hours, lookup)
        assessment_records = build_stay_records(assessment_design, beta, lambda_b, tau, index_hours, lookup)
        print(f"  selected lambda_b={lambda_b:g}; fitting frozen scalar menu", flush=True)
        calibrations = fit_all_calibrations(
            tuning_records, tau, fold_dir / "calibrations", r_script, resume
        )
        selected_calibration_model = select_calibration_model(calibrations)
        assessment_records, model_columns = add_prediction_columns(
            assessment_records, calibrations, selected_calibration_model
        )
        assessment_records = attach_model_metrics(assessment_records, model_columns, tau)
        assessment_records["outer_fold"] = fold + 1
        if expected_model_columns is None:
            expected_model_columns = model_columns
        elif expected_model_columns != model_columns:
            raise RuntimeError("Candidate menu changed across outer folds")
        public_prediction_frame(assessment_records).to_csv(
            fold_dir / "outer_assessment_predictions.csv", index=False
        )
        fold_payload = {
            "outer_fold": fold + 1,
            "inner_fit_stays": int(inner_fit.size),
            "inner_tuning_stays": int(inner_tune.size),
            "outer_assessment_stays": int(outer_assessment.size),
            "age_standardization": age_info,
            "population_coefficients": beta.tolist(),
            "selected_lambda_b": lambda_b,
            "lambda_tuning_grid": tuning_profile["grid"],
            "calibrations": calibrations,
            "inner_selected_calibration_model": selected_calibration_model,
            "runtime_seconds": float(time.time() - fold_start),
        }
        write_json(fold_dir / "fold_metadata.json", fold_payload)
        fold_metadata.append(fold_payload)
        fold_records.append(assessment_records)

    combined = pd.concat(fold_records, ignore_index=True).sort_values("global_stay_index").reset_index(drop=True)
    if combined["global_stay_index"].nunique() != n_stays or combined.shape[0] != n_stays:
        raise RuntimeError("Every eligible stay must appear in exactly one outer assessment fold")
    assert expected_model_columns is not None
    return combined, expected_model_columns, fold_metadata


def collect_scope_outputs(
    records: pd.DataFrame,
    model_columns: Mapping[str, str],
    tau: float,
    scope: str,
    k_grid: Sequence[int],
) -> Dict[str, pd.DataFrame]:
    calibration_summary, calibration_detail = discrete_calibration(records, model_columns, tau, scope)
    fixed_thresholds, fixed_losses = fixed_opportunity_analysis(records, model_columns, tau, scope, k_grid)
    return {
        "loss_summary": model_loss_summary(records, model_columns, scope),
        "paired_comparisons": paired_comparisons(records, model_columns, scope),
        "discrete_calibration_summary": calibration_summary,
        "discrete_calibration_detail": calibration_detail,
        "fixed_opportunity_thresholds": fixed_thresholds,
        "fixed_opportunity_losses": fixed_losses,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    paper_root = script_dir.parent
    recovery_root = paper_root / "recovery_20260822"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs-cache",
        type=Path,
        default=recovery_root / "cache" / "combined" / "mimic_map_observations.parquet",
    )
    parser.add_argument(
        "--stays-cache",
        type=Path,
        default=recovery_root / "cache" / "combined" / "mimic_map_stays.parquet",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=recovery_root / "split_window_mixed_effects_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=recovery_root / "validation_extensions")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--fit-stays", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--index-hours", type=float, default=12.0)
    parser.add_argument("--lambda-grid", type=parse_float_grid, default=parse_float_grid("0,0.03,0.10,0.30,1,3,10"))
    parser.add_argument("--k-grid", type=parse_int_grid, default=parse_int_grid("4,8,12"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--skip-crossfit", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--common-fixed-only", action="store_true")
    args = parser.parse_args()

    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only and args.common_fixed_only:
        raise ValueError("--plot-only and --common-fixed-only are mutually exclusive")
    if args.common_fixed_only:
        frozen_prediction_path = args.output_dir / "nested_crossfit_stay_level_predictions.csv"
        if not frozen_prediction_path.exists():
            raise FileNotFoundError(
                "--common-fixed-only requires nested_crossfit_stay_level_predictions.csv"
            )
        print("Loading analytic cache and frozen nested OOF predictions", flush=True)
        obs = _safe_read_frame(args.obs_cache)
        stays = _safe_read_frame(args.stays_cache)
        dataset, data_summary, _ = build_dataset_from_cache(
            obs, stays, fit_stays=args.fit_stays, seed=args.seed, analysis_hours=24.0
        )
        frozen_predictions = pd.read_csv(frozen_prediction_path)
        common_records = rehydrate_common_cohort_records_from_frozen_oof(
            dataset,
            frozen_predictions,
            index_hours=args.index_hours,
            min_later_records=max(args.k_grid),
        )
        common_thresholds, common_losses, common_metadata = common_cohort_fixed_opportunity_analysis(
            common_records,
            tau=args.tau,
            scope="nested_5fold_internal_crossfit_common_late12_cohort",
            k_grid=args.k_grid,
        )
        common_metadata.update(
            {
                "status": "complete",
                "analytic_stays_in_frozen_oof_file": int(frozen_predictions.shape[0]),
                "analytic_cache_stays": int(data_summary["fit_stays"]),
                "rehydration": "later outcomes from local analytic cache joined by deidentified global stay_index; frozen OOF calibrated-q10 and primary scalar predictions were not refitted",
            }
        )
        common_results_payload, common_artifacts = write_common_cohort_fixed_opportunity_artifacts(
            common_thresholds,
            common_losses,
            common_metadata,
            args.output_dir,
        )
        completed_summary_path = args.output_dir / "submission_validation_extensions_results.json"
        if completed_summary_path.exists():
            completed_summary = json.loads(completed_summary_path.read_text(encoding="utf-8"))
            completed_summary.setdefault("settings", {})[
                "common_fixed_opportunity_cohort"
            ] = "late_obs >= max(first_k_later_records), fixed before K-specific truncation"
            completed_summary.setdefault("results", {})[
                "common_cohort_fixed_opportunity"
            ] = common_results_payload
            completed_summary.setdefault("artifacts", {})[
                "common_cohort_fixed_opportunity"
            ] = common_artifacts
            completed_summary["common_fixed_opportunity_refresh_runtime_seconds"] = float(
                time.time() - start
            )
            write_json(completed_summary_path, completed_summary)
        print(
            f"Completed common-cohort fixed-opportunity extension in {time.time() - start:.1f} seconds",
            flush=True,
        )
        return
    if args.plot_only:
        detail_path = args.output_dir / "discrete_calibration_by_decile.csv"
        summary_csv_path = args.output_dir / "discrete_calibration_summary.csv"
        if not detail_path.exists() or not summary_csv_path.exists():
            raise FileNotFoundError("--plot-only requires completed discrete calibration CSV outputs")
        detail = pd.read_csv(detail_path)
        calibration_summary = pd.read_csv(summary_csv_path)
        preferred_scope = (
            "nested_5fold_internal_crossfit"
            if (detail["scope"] == "nested_5fold_internal_crossfit").any()
            else "historical_assessment_split"
        )
        figure_artifacts = plot_tie_aware_decile_calibration(
            detail,
            args.output_dir,
            preferred_scope,
            args.tau,
        )
        calibration_tex_path = args.output_dir / "tie_aware_calibration_summary.tex"
        write_tie_aware_calibration_summary_tex(
            calibration_summary,
            calibration_tex_path,
            preferred_scope,
        )
        threshold_csv_path = args.output_dir / "fixed_opportunity_threshold_sensitivity.csv"
        if threshold_csv_path.exists():
            write_fixed_opportunity_tex(
                pd.read_csv(threshold_csv_path),
                args.output_dir / "fixed_opportunity_threshold_sensitivity.tex",
                preferred_scope,
            )
        paired_csv_path = args.output_dir / "paired_comparator_results.csv"
        if paired_csv_path.exists():
            write_paired_comparison_tex(
                pd.read_csv(paired_csv_path),
                args.output_dir / "paired_comparator_results.tex",
                preferred_scope,
            )
        common_threshold_csv = args.output_dir / "common_cohort_fixed_opportunity_threshold_sensitivity.csv"
        common_loss_csv = args.output_dir / "common_cohort_fixed_opportunity_check_loss_comparison.csv"
        refreshed_common_payload: Dict[str, object] | None = None
        refreshed_common_artifacts: Dict[str, str] | None = None
        if common_threshold_csv.exists() and common_loss_csv.exists():
            common_threshold_frame = pd.read_csv(common_threshold_csv)
            common_loss_frame = pd.read_csv(common_loss_csv)
            common_threshold_frame[
                "risk_difference_interval_interpretation"
            ] = "approximate independent-stay normal 95% interval"
            common_loss_frame[
                "paired_interval_interpretation"
            ] = "descriptive estimate plus or minus 1.96 stay-level SE, conditional on realized fold-trained rules; does not propagate refitting, tuning, or rule-selection variation"
            common_json_path = args.output_dir / "common_cohort_fixed_opportunity_results.json"
            common_metadata = (
                json.loads(common_json_path.read_text(encoding="utf-8")).get("metadata", {})
                if common_json_path.exists()
                else {}
            )
            common_metadata[
                "risk_difference_interval_interpretation"
            ] = "approximate independent-stay normal 95% interval"
            common_metadata[
                "paired_check_loss_interval_interpretation"
            ] = "descriptive estimate plus or minus 1.96 stay-level SE, conditional on realized fold-trained rules; does not propagate refitting, tuning, or rule-selection variation"
            refreshed_common_payload, refreshed_common_artifacts = write_common_cohort_fixed_opportunity_artifacts(
                common_threshold_frame,
                common_loss_frame,
                common_metadata,
                args.output_dir,
            )
        artifact_payload = {
            **figure_artifacts,
            "manuscript_summary_tex": str(calibration_tex_path),
        }
        write_json(args.output_dir / "tie_aware_decile_calibration_artifacts.json", artifact_payload)
        completed_summary_path = args.output_dir / "submission_validation_extensions_results.json"
        if completed_summary_path.exists():
            completed_summary = json.loads(completed_summary_path.read_text(encoding="utf-8"))
            completed_summary.setdefault("artifacts", {})["tie_aware_decile_calibration"] = artifact_payload
            if refreshed_common_payload is not None and refreshed_common_artifacts is not None:
                completed_summary.setdefault("results", {})[
                    "common_cohort_fixed_opportunity"
                ] = refreshed_common_payload
                completed_summary.setdefault("artifacts", {})[
                    "common_cohort_fixed_opportunity"
                ] = refreshed_common_artifacts
            write_json(completed_summary_path, completed_summary)
        print(f"Regenerated tie-aware calibration artifacts in {time.time() - start:.1f} seconds", flush=True)
        return
    work_dir = args.work_dir or (args.output_dir / "work")
    work_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    r_script = script_dir / "fit_quantile_common.R"
    if not r_script.exists():
        raise FileNotFoundError(r_script)

    print("Loading combined recovery cache", flush=True)
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, _ = build_dataset_from_cache(
        obs, stays, fit_stays=args.fit_stays, seed=args.seed, analysis_hours=24.0
    )
    primary_results = json.loads(args.results_json.read_text(encoding="utf-8"))
    candidate_menu = {
        "core": list(CORE_MODELS),
        "raw_map_scale": list(RAW_MAP_MODELS),
        "stay_equal_affine_calibrated": list(CALIBRATED_SUMMARIES),
        "inner_selection_rule": "minimum stay-equal tuning check loss across calibrated scalar menu; frozen-order tie break",
        "frozen_before_outer_assessment": True,
    }
    analysis_boundaries = {
        "raw_last_min_mean_q10": "computed from each outer-assessment stay's index window only; no fitted parameters",
        "calibrated_last_min_mean_q10_below65_burden": "affine coefficients fitted on inner-tuning stays only; frozen for outer assessment",
        "population": "age standardization and population quantile coefficients fitted on inner-fitting stays only; frozen for outer assessment",
        "primary_level_update": "population component fitted on inner-fitting stays; penalty selected on inner-tuning stays; stay offset uses that outer-assessment stay's index window; later outcomes are evaluation only",
        "inner_selected_calibrated_rule": "candidate chosen by minimum stay-equal check loss on inner-tuning stays; chosen rule and coefficients frozen for outer assessment",
    }

    print("Running historical fixed-split extensions", flush=True)
    fixed_records, fixed_models, fixed_metadata = run_fixed_split(
        dataset,
        stays,
        primary_results,
        tau=args.tau,
        index_hours=args.index_hours,
        lambda_grid=args.lambda_grid,
        work_dir=work_dir / "fixed_split",
        r_script=r_script,
        seed=args.seed,
        resume=resume,
    )
    public_prediction_frame(fixed_records).to_csv(args.output_dir / "fixed_split_stay_level_predictions.csv", index=False)
    all_outputs = collect_scope_outputs(
        fixed_records, fixed_models, args.tau, "historical_assessment_split", args.k_grid
    )

    crossfit_metadata: List[Dict[str, object]] = []
    crossfit_status: Dict[str, object]
    common_thresholds = pd.DataFrame()
    common_losses = pd.DataFrame()
    common_metadata: Dict[str, object] = {"status": "not_available_without_nested_crossfit"}
    if args.skip_crossfit:
        crossfit_status = {"status": "skipped_by_command", "reason": "--skip-crossfit"}
    else:
        print("Running nested outer cross-fitting", flush=True)
        crossfit_records, crossfit_models, crossfit_metadata = run_nested_crossfit(
            dataset,
            stays,
            tau=args.tau,
            index_hours=args.index_hours,
            lambda_grid=args.lambda_grid,
            n_folds=args.outer_folds,
            work_dir=work_dir / "nested_crossfit",
            r_script=r_script,
            seed=args.seed,
            resume=resume,
        )
        public_prediction_frame(crossfit_records).to_csv(
            args.output_dir / "nested_crossfit_stay_level_predictions.csv", index=False
        )
        crossfit_outputs = collect_scope_outputs(
            crossfit_records, crossfit_models, args.tau, "nested_5fold_internal_crossfit", args.k_grid
        )
        common_thresholds, common_losses, common_metadata = common_cohort_fixed_opportunity_analysis(
            crossfit_records,
            tau=args.tau,
            scope="nested_5fold_internal_crossfit_common_late12_cohort",
            k_grid=args.k_grid,
        )
        common_metadata["status"] = "complete"
        all_outputs = {
            key: pd.concat([all_outputs[key], crossfit_outputs[key]], ignore_index=True)
            for key in all_outputs
        }
        crossfit_status = {
            "status": "complete",
            "outer_folds": int(args.outer_folds),
            "eligible_stays_scored_once": int(crossfit_records.shape[0]),
            "interpretation": "Nested internal cross-fitting; not external or never-seen validation.",
        }

    output_filenames = {
        "loss_summary": "model_loss_summary.csv",
        "paired_comparisons": "paired_comparator_results.csv",
        "discrete_calibration_summary": "discrete_calibration_summary.csv",
        "discrete_calibration_detail": "discrete_calibration_by_decile.csv",
        "fixed_opportunity_thresholds": "fixed_opportunity_threshold_sensitivity.csv",
        "fixed_opportunity_losses": "fixed_opportunity_check_loss.csv",
    }
    for key, filename in output_filenames.items():
        all_outputs[key].to_csv(args.output_dir / filename, index=False)

    common_artifacts: Dict[str, str] = {}
    if not common_thresholds.empty and not common_losses.empty:
        common_results_payload, common_artifacts = write_common_cohort_fixed_opportunity_artifacts(
            common_thresholds,
            common_losses,
            common_metadata,
            args.output_dir,
        )
    else:
        common_results_payload = {"status": "not_available_without_nested_crossfit"}

    preferred_scope = (
        "nested_5fold_internal_crossfit" if crossfit_status["status"] == "complete" else "historical_assessment_split"
    )
    write_fixed_opportunity_tex(
        all_outputs["fixed_opportunity_thresholds"],
        args.output_dir / "fixed_opportunity_threshold_sensitivity.tex",
        preferred_scope,
    )
    write_paired_comparison_tex(
        all_outputs["paired_comparisons"],
        args.output_dir / "paired_comparator_results.tex",
        preferred_scope,
    )
    calibration_figure_artifacts = plot_tie_aware_decile_calibration(
        all_outputs["discrete_calibration_detail"],
        args.output_dir,
        preferred_scope,
        args.tau,
    )
    calibration_tex_path = args.output_dir / "tie_aware_calibration_summary.tex"
    write_tie_aware_calibration_summary_tex(
        all_outputs["discrete_calibration_summary"],
        calibration_tex_path,
        preferred_scope,
    )
    calibration_figure_artifacts["manuscript_summary_tex"] = str(calibration_tex_path)
    write_json(
        args.output_dir / "tie_aware_decile_calibration_artifacts.json",
        calibration_figure_artifacts,
    )

    feasibility = {
        "status": crossfit_status["status"],
        "local_cache_available": True,
        "fold_specific_population_refitting_available": True,
        "inner_penalty_selection_available": True,
        "inner_scalar_calibration_available": True,
        "stay_is_independent_unit": True,
        "limitations": [
            "The source database and candidate ideas were previously inspected; nested cross-fitting prevents fold-level leakage but does not create an external cohort.",
            "Operational threshold analyses are descriptive sensitivities and do not redefine the tau=0.10 check-loss target.",
        ],
    }
    write_json(args.output_dir / "crossfit_feasibility.json", feasibility)

    summary_payload = {
        "status": "complete",
        "runtime_seconds": float(time.time() - start),
        "data_summary": data_summary,
        "settings": {
            "seed": args.seed,
            "tau": args.tau,
            "index_hours": args.index_hours,
            "lambda_grid": args.lambda_grid,
            "first_k_later_records": args.k_grid,
            "operational_thresholds_mmhg": list(OPERATIONAL_THRESHOLDS),
            "outer_folds": args.outer_folds,
            "independent_unit": "ICU stay",
            "candidate_menu": candidate_menu,
            "analysis_boundaries": analysis_boundaries,
            "common_fixed_opportunity_cohort": "late_obs >= max(first_k_later_records), fixed before K-specific truncation",
        },
        "fixed_split": fixed_metadata,
        "nested_crossfit": crossfit_status,
        "nested_crossfit_folds": crossfit_metadata,
        "results": {
            **{key: frame.to_dict("records") for key, frame in all_outputs.items()},
            "common_cohort_fixed_opportunity": common_results_payload,
        },
        "artifacts": {
            **{key: str(args.output_dir / filename) for key, filename in output_filenames.items()},
            "fixed_opportunity_threshold_tex": str(
                args.output_dir / "fixed_opportunity_threshold_sensitivity.tex"
            ),
            "paired_comparator_tex": str(args.output_dir / "paired_comparator_results.tex"),
            "crossfit_feasibility": str(args.output_dir / "crossfit_feasibility.json"),
            "tie_aware_decile_calibration": calibration_figure_artifacts,
            "common_cohort_fixed_opportunity": common_artifacts,
        },
    }
    write_json(args.output_dir / "submission_validation_extensions_results.json", summary_payload)
    print(
        f"Completed validation extensions in {time.time() - start:.1f} seconds. "
        f"Results: {args.output_dir / 'submission_validation_extensions_results.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
