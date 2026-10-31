from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from run_adaptive_clinical import HORIZONS, layout_stays, make_layout
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import balanced_fold_ids, inner_fit_tune_indices


BASELINE_MODELS = ("population", "calibrated_q10", "original_selected", "vector_direct",
                   "vector_affine", "stump_original")


@dataclass
class ClinicalPartition:
    name: str
    fit_indices: np.ndarray
    tuning_indices: np.ndarray
    assessment_indices: np.ndarray
    population_coefficients: np.ndarray
    age_standardization: dict
    tuning_layouts: dict
    assessment_layouts: dict
    baselines: dict


def load_clinical_cache(path: Path, cache_directory: Path | None = None) -> dict:
    with path.open("rb") as stream:
        cached = pickle.load(stream)
    if len(cached["dataset"]["cluster_ids"]) != 60310:
        raise ValueError("Continuous component learning requires the fixed 60310 stay cohort")
    if int(cached["data_summary"]["fit_stays"]) != 60310:
        raise ValueError("The retained clinical cache summary differs")
    if cache_directory is not None:
        for filename, signature in cached["source_signature"].items():
            current = (cache_directory / Path(filename).name).stat()
            if {"bytes": current.st_size, "mtime_ns": current.st_mtime_ns} != signature:
                raise ValueError("The original clinical input file changed")
    return cached


def partition_directory(root: Path, name: str, horizon: float) -> Path:
    if name == "temporal":
        return root / f"horizon_{int(horizon)}" / "temporal"
    if name not in {f"fold_{number}" for number in range(1, 6)}:
        raise ValueError("Unknown clinical partition")
    return root / f"horizon_{int(horizon)}" / "nested" / name


def verify_nested_indices(name: str, fit: np.ndarray, tune: np.ndarray, assessment: np.ndarray) -> None:
    if name == "temporal":
        return
    fold = int(name.removeprefix("fold_")) - 1
    assignments = balanced_fold_ids(60310, 5, 20260522)
    expected_fit, expected_tune = inner_fit_tune_indices(np.flatnonzero(assignments != fold), fold, 20260522)
    expected_assessment = np.flatnonzero(assignments == fold)
    for actual, expected in ((fit, expected_fit), (tune, expected_tune), (assessment, expected_assessment)):
        np.testing.assert_array_equal(actual, expected)


def load_partition(cached: dict, source_root: Path, name: str) -> ClinicalPartition:
    configs = {}
    for horizon in HORIZONS:
        directory = partition_directory(source_root, name, horizon)
        configs[horizon] = json.loads((directory / "private_partition_config.json").read_text())
        metadata = json.loads((directory / "partition_results.json").read_text())
        if metadata["prediction_protocol_version"] != 2 or configs[horizon]["tau"] != .1:
            raise ValueError("The retained clinical prediction protocol differs")
        if configs[horizon]["index_hours"] != 12 or configs[horizon]["horizon"] != horizon:
            raise ValueError("The retained prediction clock differs")
    reference = configs[12.0]
    arrays = {key: np.asarray(reference[key], dtype=np.int64)
              for key in ("fit_indices", "tune_indices", "assessment_indices")}
    fit, tune, assessment = (arrays[key] for key in ("fit_indices", "tune_indices", "assessment_indices"))
    if len(np.unique(np.concatenate([fit, tune, assessment]))) != len(fit) + len(tune) + len(assessment):
        raise ValueError("Clinical fitting, tuning, and assessment populations overlap")
    verify_nested_indices(name, fit, tune, assessment)
    standardized, age_info = apply_training_age_standardization(cached["dataset"], cached["stays"], fit)
    beta = np.asarray(reference["population_coefficients"], dtype=float)
    if beta.shape != (4,) or not np.isfinite(beta).all():
        raise ValueError("Invalid retained clinical population fit")
    for key, value in age_info.items():
        np.testing.assert_allclose(value, reference["age_standardization"][key], rtol=1e-12, atol=1e-12)
    tuning_layouts, assessment_layouts, baselines = {}, {}, {}
    for horizon, config in configs.items():
        for key, expected in arrays.items():
            np.testing.assert_array_equal(config[key], expected)
        np.testing.assert_allclose(config["population_coefficients"], beta, rtol=0, atol=0)
        if config["age_standardization"] != reference["age_standardization"]:
            raise ValueError("Age standardization differs across forecast horizons")
        tuning_layouts[horizon], _ = make_layout(standardized, tune, beta, horizon)
        assessment_layouts[horizon], _ = make_layout(standardized, assessment, beta, horizon)
        np.testing.assert_array_equal(tuning_layouts[horizon].global_indices, config["scored_tuning_indices"])
        layout = assessment_layouts[horizon]
        np.testing.assert_array_equal(layout.global_indices, config["scored_assessment_indices"])
        columns = ["global_stay_index", "late_obs"] + [f"{metric}_{model}"
                   for model in BASELINE_MODELS for metric in ("loss", "p_lt", "p_le")]
        frame = pd.read_csv(partition_directory(source_root, name, horizon) / "assessment_metrics.csv", usecols=columns)
        if frame["global_stay_index"].duplicated().any():
            raise ValueError("A clinical baseline contains repeated assessment stays")
        np.testing.assert_array_equal(frame["global_stay_index"].to_numpy(), layout.global_indices)
        np.testing.assert_array_equal(frame["late_obs"].to_numpy(), layout.later_counts)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("The retained clinical baseline scores are nonfinite")
        baselines[horizon] = frame
    return ClinicalPartition(name, fit, tune, assessment, beta, age_info,
                             tuning_layouts, assessment_layouts, baselines)


def combined_records(layouts: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    stays, horizons, weights, identities = [], [], [], []
    for horizon in HORIZONS:
        layout = layouts[horizon]
        stays.extend(layout_stays(layout))
        horizons.extend([horizon] * layout.n_stays)
        weights.extend([1.0 / (len(HORIZONS) * layout.n_stays)] * layout.n_stays)
        identities.extend(layout.global_indices)
    np.testing.assert_allclose(sum(weights), 1.0, rtol=0, atol=1e-12)
    return stays, np.asarray(horizons), np.asarray(weights), np.asarray(identities, dtype=np.int64)


def tuning_fold_assignments(identities: np.ndarray, seed: int, folds: int = 3) -> np.ndarray:
    unique = np.unique(identities)
    if folds < 2 or len(unique) < folds:
        raise ValueError("The learning validation requires independent nonempty stay folds")
    order = np.random.default_rng(seed).permutation(len(unique))
    labels = np.empty(len(unique), dtype=np.int64)
    labels[order] = np.arange(len(unique)) % folds
    mapped = labels[np.searchsorted(unique, identities)]
    for identifier in unique:
        if len(np.unique(mapped[identities == identifier])) != 1:
            raise ValueError("A repeated stay entered different horizon validation folds")
    return mapped


def equal_horizon_weights(horizons: np.ndarray) -> np.ndarray:
    if set(np.unique(horizons)) != set(HORIZONS):
        raise ValueError("Each learning sample must include every forecast horizon")
    weights = np.empty(len(horizons), dtype=float)
    for horizon in HORIZONS:
        mask = horizons == horizon
        weights[mask] = 1.0 / (len(HORIZONS) * int(mask.sum()))
    return weights
