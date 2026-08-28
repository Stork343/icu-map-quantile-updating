#!/usr/bin/env python3
"""Simulation studies for split-window lower-quantile updating.

Two explicitly versioned designs are available. ``legacy-v1`` reproduces the
original seven-scenario experiment and its file contract. ``ademp-v2`` is the
submission-strengthening design: it separates data-generating mechanisms,
estimands, methods and performance measures (ADEMP), preserves the cluster as
the analysis unit, and reports Monte Carlo uncertainty rather than treating a
fixed number of repetitions as intrinsically adequate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.stats import norm, t as student_t

from split_window_analysis_core import check_loss, empirical_check_quantile, profiled_intercept


PREDICTOR_COLS = ["intercept", "age_z", "male", "urgent"]

SCENARIOS: Dict[str, Dict[str, object]] = {
    "persistent": {
        "label": "Persistent level",
        "description": "A stable stay level lower tail intercept persists from index window to later observations.",
        "level_sd": 5.5,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "noise": "normal",
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
    "weak": {
        "label": "Weak level",
        "description": "A smaller persistent stay level lower tail intercept is present.",
        "level_sd": 2.5,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "noise": "normal",
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
    "sparse": {
        "label": "Sparse index",
        "description": "The persistent intercept is present but only two to five index window observations are available.",
        "level_sd": 5.5,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "noise": "normal",
        "index_count_range": (2, 5),
        "late_count_range": (6, 16),
    },
    "heavy_tail": {
        "label": "Heavy tail level",
        "description": "The persistent intercept is present under heavy tailed residual variation.",
        "level_sd": 5.5,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "noise": "student_t",
        "noise_df": 3.0,
        "noise_scale": 0.85,
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
    "no_level": {
        "label": "No level effect",
        "description": "No persistent stay level lower tail intercept is present.",
        "level_sd": 0.0,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "noise": "normal",
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
    "transient": {
        "label": "Transient shift",
        "description": "A stay level index window shift is present before 12 hours but does not persist later.",
        "level_sd": 0.0,
        "index_shift_sd": 6.0,
        "slope_sd": 0.0,
        "noise": "normal",
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
    "shape": {
        "label": "Random shape",
        "description": "A lower amplitude intercept is mixed with stay specific time shape heterogeneity.",
        "level_sd": 2.5,
        "index_shift_sd": 0.0,
        "slope_sd": 4.5,
        "noise": "normal",
        "index_count_range": (6, 16),
        "late_count_range": (6, 16),
    },
}


# ADEMP v2 deliberately changes one or two stressors at a time.  Scenario-level
# sample sizes are part of the design, so the small-sample and sparse-observation
# axes are not silently collapsed into one setting.
ADEMP_V2_SCENARIOS: Dict[str, Dict[str, object]] = {
    "ideal_large_dense": {
        "label": "Ideal: large, dense",
        "description": "Correct common component, independent errors, persistent level, and dense non-informative observation.",
        "n_stays": 600,
        "level_sd": 5.5,
        "index_count_range": (8, 16),
        "late_count_range": (8, 16),
    },
    "ideal_small_dense": {
        "label": "Ideal: small, dense",
        "description": "The ideal mechanism with 240 stays to isolate sample-size sensitivity.",
        "n_stays": 240,
        "level_sd": 5.5,
        "index_count_range": (8, 16),
        "late_count_range": (8, 16),
    },
    "ideal_large_sparse": {
        "label": "Ideal: large, sparse",
        "description": "The ideal mechanism with only two to five index observations to isolate observation sparsity.",
        "n_stays": 600,
        "level_sd": 5.5,
        "index_count_range": (2, 5),
        "late_count_range": (4, 8),
    },
    "serial_dependence": {
        "label": "Serial dependence",
        "description": "A persistent level is observed with a Gaussian-copula AR(1) residual process on a 15-minute latent grid.",
        "n_stays": 600,
        "level_sd": 5.5,
        "serial_rho_15min": 0.82,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "discrete_map_rounding": {
        "label": "Integer MAP rounding",
        "description": "Observed and oracle MAP quantiles are rounded to integer mmHg, creating ties that require interval calibration.",
        "n_stays": 600,
        "level_sd": 5.5,
        "serial_rho_15min": 0.55,
        "rounding_increment": 1.0,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "informative_monitoring": {
        "label": "Informative monitoring",
        "description": "Observation probability increases for lower latent levels and after a recently observed low MAP.",
        "n_stays": 600,
        "level_sd": 5.5,
        "serial_rho_15min": 0.72,
        "sampling": "informative_grid",
        "index_sampling_probability": 0.12,
        "late_sampling_probability": 0.12,
        "measurement_level_strength": 0.60,
        "measurement_recent_strength": 0.85,
        "index_count_range": (4, 22),
        "late_count_range": (4, 22),
    },
    "cluster_size_informative": {
        "label": "Cluster size--level association",
        "description": "Cluster size is associated with the latent stay level even though observation times are otherwise non-informative.",
        "n_stays": 600,
        "level_sd": 5.5,
        "count_level_strength": 1.05,
        "index_count_range": (3, 20),
        "late_count_range": (3, 20),
    },
    "common_time_misspecified": {
        "label": "Misspecified common time",
        "description": "The true common quantile contains nonlinear circadian and quadratic time structure omitted from the fitted common component.",
        "n_stays": 600,
        "level_sd": 5.5,
        "common_time_amplitude": 5.0,
        "serial_rho_15min": 0.55,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "level_plus_shape": {
        "label": "Persistent level plus shape",
        "description": "Persistent random intercepts coexist with stay-specific linear time shape, favoring a correctly targeted level-plus-slope rule.",
        "n_stays": 600,
        "level_sd": 3.0,
        "slope_sd": 5.0,
        "serial_rho_15min": 0.55,
        "index_count_range": (8, 16),
        "late_count_range": (8, 16),
    },
    "treatment_feedback": {
        "label": "Treatment feedback",
        "description": "An early low lower tail triggers a later MAP-raising intervention whose effect decays over time.",
        "n_stays": 600,
        "level_sd": 5.5,
        "serial_rho_15min": 0.60,
        "treatment_threshold": 68.0,
        "treatment_gain": 8.0,
        "treatment_decay_hours": 7.0,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "transient_nonpersistent": {
        "label": "Transient non-persistence",
        "description": "A large index-only shift disappears later, providing a negative control under serial residual dependence.",
        "n_stays": 600,
        "level_sd": 0.0,
        "index_shift_sd": 6.0,
        "serial_rho_15min": 0.60,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "null_serial": {
        "label": "Pure null with serial dependence",
        "description": "No persistent level, index shift, or random shape is present; serial residual dependence tests whether tuning suppresses noise-only offsets.",
        "n_stays": 600,
        "level_sd": 0.0,
        "index_shift_sd": 0.0,
        "slope_sd": 0.0,
        "serial_rho_15min": 0.82,
        "index_count_range": (6, 14),
        "late_count_range": (6, 14),
    },
    "weak_level": {
        "label": "Weak persistent level",
        "description": "A small but persistent stay-specific lower-tail level tests power and shrinkage under a weak signal.",
        "n_stays": 600,
        "level_sd": 2.5,
        "index_count_range": (8, 16),
        "late_count_range": (8, 16),
    },
    "heavy_tail_t3": {
        "label": "Heavy-tailed t3 residual",
        "description": "A persistent level is observed with independent Student-t3 marginal residuals centered at their exact 0.10 quantile.",
        "n_stays": 600,
        "level_sd": 5.5,
        "noise": "student_t",
        "noise_df": 3.0,
        "noise_scale": 0.85,
        "index_count_range": (8, 16),
        "late_count_range": (8, 16),
    },
}

ADEMP_V2_METHODS = (
    "oracle",
    "population",
    "raw_index_q10",
    "affine_calibrated_q10",
    "unpenalized_level",
    "tuned_level",
    "tuned_level_slope",
)


@dataclass
class SimulatedDataset:
    x: np.ndarray
    y_list: List[np.ndarray]
    t_list: List[np.ndarray]
    true_late_offset: np.ndarray


@dataclass
class AdempV2Dataset:
    """One complete simulated dataset with oracle quantities retained.

    ``true_quantile_list`` is the observation-time marginal tau-quantile given
    the fixed and latent trajectory components.  Under serial residual
    dependence it is not a one-step quantile additionally conditioned on the
    realized residual history.
    ``true_update_list`` contains only the stay-specific component (including
    treatment feedback when triggered), which defines the offset-recovery
    estimand independently of common-component misspecification.
    """

    x: np.ndarray
    y_list: List[np.ndarray]
    t_list: List[np.ndarray]
    true_quantile_list: List[np.ndarray]
    true_update_list: List[np.ndarray]
    error_list: List[np.ndarray]
    true_late_offset: np.ndarray
    latent_level: np.ndarray
    treatment_trigger: np.ndarray


def split_cluster_indices(
    n_stays: int,
    rng: np.random.Generator,
    train_fraction: float = 0.60,
    tuning_fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = rng.permutation(n_stays)
    n_train = int(round(train_fraction * n_stays))
    n_tuning = int(round(tuning_fraction * n_stays))
    train_idx = np.sort(perm[:n_train])
    tuning_idx = np.sort(perm[n_train : n_train + n_tuning])
    assessment_idx = np.sort(perm[n_train + n_tuning :])
    return train_idx, tuning_idx, assessment_idx


def simulate_dataset(
    n_stays: int,
    tau: float,
    scenario: Dict[str, object],
    rng: np.random.Generator,
) -> SimulatedDataset:
    beta = np.asarray([72.0, -2.0, 1.0, -1.8], dtype=float)
    age_z = rng.normal(size=n_stays)
    male = rng.binomial(1, 0.55, size=n_stays).astype(float)
    urgent = rng.binomial(1, 0.65, size=n_stays).astype(float)
    x = np.column_stack([np.ones(n_stays), age_z, male, urgent])

    level = rng.normal(0.0, float(scenario["level_sd"]), size=n_stays)
    index_shift = rng.normal(0.0, float(scenario["index_shift_sd"]), size=n_stays)
    slope = rng.normal(0.0, float(scenario["slope_sd"]), size=n_stays)
    noise_scale = 4.5 + 0.5 * np.abs(age_z) + 0.9 * urgent
    z_tau = norm.ppf(tau)
    noise_kind = str(scenario.get("noise", "normal"))
    noise_scale_multiplier = float(scenario.get("noise_scale", 1.0))
    noise_df = float(scenario.get("noise_df", 3.0))
    index_count_min, index_count_max = tuple(int(x) for x in scenario.get("index_count_range", (6, 16)))
    late_count_min, late_count_max = tuple(int(x) for x in scenario.get("late_count_range", (6, 16)))

    y_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    true_late_offset = np.zeros(n_stays, dtype=float)

    for i in range(n_stays):
        n_index = int(rng.integers(index_count_min, index_count_max + 1))
        n_late = int(rng.integers(late_count_min, late_count_max + 1))
        t_index = np.sort(rng.uniform(0.0, 12.0, size=n_index))
        t_late = np.sort(rng.uniform(12.0, 24.0, size=n_late))
        t = np.concatenate([t_index, t_late])
        time_c = (t - 12.0) / 12.0
        index_flag = t <= 12.0
        latent = level[i] + slope[i] * time_c + index_shift[i] * index_flag.astype(float)
        if noise_kind == "student_t":
            noise_draw = rng.standard_t(df=noise_df, size=t.size)
            noise_quantile = student_t.ppf(tau, df=noise_df)
        else:
            noise_draw = rng.normal(size=t.size)
            noise_quantile = z_tau
        eps = noise_scale_multiplier * noise_scale[i] * (noise_draw - noise_quantile)
        y = float(x[i] @ beta) + latent + eps
        y_list.append(y.astype(float))
        t_list.append(t.astype(float))
        true_late_offset[i] = float(np.mean(latent[~index_flag]))

    return SimulatedDataset(x=x, y_list=y_list, t_list=t_list, true_late_offset=true_late_offset)


def _logit_probability(linear_predictor: float) -> float:
    clipped = float(np.clip(linear_predictor, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _ar1_copula_errors(
    n: int,
    tau: float,
    rho: float,
    noise_kind: str,
    noise_df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate serial errors with a known continuous marginal tau-quantile of zero."""

    if n <= 0:
        return np.empty(0, dtype=float)
    if not 0.0 <= rho < 1.0:
        raise ValueError("serial_rho_15min must lie in [0, 1).")
    z = np.empty(n, dtype=float)
    z[0] = rng.normal()
    innovation_scale = np.sqrt(max(1.0 - rho * rho, 0.0))
    for j in range(1, n):
        z[j] = rho * z[j - 1] + innovation_scale * rng.normal()
    if noise_kind == "student_t":
        probabilities = np.clip(norm.cdf(z), 1e-12, 1.0 - 1e-12)
        draws = student_t.ppf(probabilities, df=noise_df)
        marginal_tau = student_t.ppf(tau, df=noise_df)
    elif noise_kind == "normal":
        draws = z
        marginal_tau = norm.ppf(tau)
    else:
        raise ValueError(f"Unsupported noise distribution: {noise_kind}")
    return np.asarray(draws - marginal_tau, dtype=float)


def _fixed_grid_indices(
    candidate_indices: np.ndarray,
    count_range: Tuple[int, int],
    latent_level: float,
    count_level_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lower, upper = (int(count_range[0]), int(count_range[1]))
    upper = min(upper, int(candidate_indices.size))
    lower = min(lower, upper)
    if count_level_strength == 0.0:
        count = int(rng.integers(lower, upper + 1))
    else:
        # Lower latent MAP levels generate larger clusters.  A binomial count
        # keeps the declared lower/upper support exact.
        probability = _logit_probability(-count_level_strength * latent_level / 5.5)
        count = lower + int(rng.binomial(upper - lower, probability))
    return np.sort(rng.choice(candidate_indices, size=count, replace=False)).astype(int)


def _informative_grid_indices(
    candidate_indices: np.ndarray,
    y_grid: np.ndarray,
    base_probability: float,
    count_range: Tuple[int, int],
    latent_level: float,
    level_strength: float,
    recent_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sequentially sample times using latent state and the last observed MAP."""

    if not 0.0 < base_probability < 1.0:
        raise ValueError("Informative-grid base probabilities must lie in (0, 1).")
    lower, upper = (int(count_range[0]), int(count_range[1]))
    lower = min(lower, int(candidate_indices.size))
    upper = min(max(upper, lower), int(candidate_indices.size))
    base_logit = float(np.log(base_probability / (1.0 - base_probability)))
    selected: List[int] = []
    selection_probabilities: List[float] = []
    previous_observed_map = 72.0 + latent_level
    for grid_index in candidate_indices:
        recent_low_score = float(np.clip((70.0 - previous_observed_map) / 8.0, -3.0, 3.0))
        linear_predictor = (
            base_logit
            - level_strength * latent_level / 5.5
            + recent_strength * recent_low_score
        )
        probability = _logit_probability(linear_predictor)
        selection_probabilities.append(probability)
        if rng.random() < probability:
            selected.append(int(grid_index))
            previous_observed_map = float(y_grid[int(grid_index)])

    if len(selected) < lower:
        remaining = np.setdiff1d(candidate_indices, np.asarray(selected, dtype=int), assume_unique=False)
        probability_lookup = dict(zip(candidate_indices.tolist(), selection_probabilities))
        weights = np.asarray([probability_lookup[int(idx)] for idx in remaining], dtype=float)
        weights = weights / np.sum(weights)
        additions = rng.choice(remaining, size=lower - len(selected), replace=False, p=weights)
        selected.extend(int(idx) for idx in additions)
    if len(selected) > upper:
        selected = rng.choice(np.asarray(selected, dtype=int), size=upper, replace=False).astype(int).tolist()
    return np.sort(np.asarray(selected, dtype=int))


def simulate_ademp_v2_dataset(
    n_stays: int,
    tau: float,
    scenario: Mapping[str, object],
    rng: np.random.Generator,
) -> AdempV2Dataset:
    """Simulate one ADEMP-v2 dataset on a latent 15-minute trajectory grid."""

    if n_stays < 30:
        raise ValueError("ADEMP-v2 requires at least 30 stays for a stable three-way split.")
    beta = np.asarray([72.0, -2.0, 1.0, -1.8], dtype=float)
    age_z = rng.normal(size=n_stays)
    male = rng.binomial(1, 0.55, size=n_stays).astype(float)
    urgent = rng.binomial(1, 0.65, size=n_stays).astype(float)
    x = np.column_stack([np.ones(n_stays), age_z, male, urgent])

    level = rng.normal(0.0, float(scenario.get("level_sd", 0.0)), size=n_stays)
    index_shift = rng.normal(0.0, float(scenario.get("index_shift_sd", 0.0)), size=n_stays)
    slope = rng.normal(0.0, float(scenario.get("slope_sd", 0.0)), size=n_stays)
    noise_scale = 4.5 + 0.5 * np.abs(age_z) + 0.9 * urgent
    noise_kind = str(scenario.get("noise", "normal"))
    noise_df = float(scenario.get("noise_df", 3.0))
    noise_scale_multiplier = float(scenario.get("noise_scale", 1.0))
    serial_rho = float(scenario.get("serial_rho_15min", 0.0))
    common_amplitude = float(scenario.get("common_time_amplitude", 0.0))
    sampling = str(scenario.get("sampling", "fixed_grid"))
    count_level_strength = float(scenario.get("count_level_strength", 0.0))
    level_measurement_strength = float(scenario.get("measurement_level_strength", 0.0))
    recent_measurement_strength = float(scenario.get("measurement_recent_strength", 0.0))
    index_range = tuple(int(v) for v in scenario.get("index_count_range", (6, 16)))
    late_range = tuple(int(v) for v in scenario.get("late_count_range", (6, 16)))
    index_probability = float(scenario.get("index_sampling_probability", 0.12))
    late_probability = float(scenario.get("late_sampling_probability", 0.12))
    treatment_threshold = float(scenario.get("treatment_threshold", -np.inf))
    treatment_gain = float(scenario.get("treatment_gain", 0.0))
    treatment_decay = float(scenario.get("treatment_decay_hours", 7.0))
    rounding_increment = float(scenario.get("rounding_increment", 0.0))

    grid_times = np.arange(0.25, 24.0 + 1e-9, 0.25, dtype=float)
    grid_index_candidates = np.flatnonzero(grid_times <= 12.0)
    grid_late_candidates = np.flatnonzero(grid_times > 12.0)
    time_c = (grid_times - 12.0) / 12.0
    # This shape is intentionally omitted from the fitted common component.
    common_time = common_amplitude * (
        0.65 * np.sin(2.0 * np.pi * grid_times / 24.0)
        + 0.35 * (np.square(time_c) - np.mean(np.square(time_c)))
    )

    y_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    true_quantile_list: List[np.ndarray] = []
    true_update_list: List[np.ndarray] = []
    error_list: List[np.ndarray] = []
    true_late_offset = np.zeros(n_stays, dtype=float)
    treatment_trigger = np.zeros(n_stays, dtype=bool)

    for i in range(n_stays):
        individual = (
            level[i]
            + slope[i] * time_c
            + index_shift[i] * (grid_times <= 12.0).astype(float)
        )
        standardized_error = _ar1_copula_errors(
            grid_times.size,
            tau=tau,
            rho=serial_rho,
            noise_kind=noise_kind,
            noise_df=noise_df,
            rng=rng,
        )
        error = noise_scale_multiplier * noise_scale[i] * standardized_error
        true_quantile = float(x[i] @ beta) + common_time + individual
        y_grid = true_quantile + error

        if treatment_gain > 0.0:
            early = grid_times <= 6.0
            early_q = empirical_check_quantile(y_grid[early], tau)
            if early_q < treatment_threshold:
                treatment_trigger[i] = True
                treatment_effect = treatment_gain * np.exp(
                    -np.maximum(grid_times - 12.0, 0.0) / treatment_decay
                ) * (grid_times > 12.0).astype(float)
                individual = individual + treatment_effect
                true_quantile = true_quantile + treatment_effect
                y_grid = y_grid + treatment_effect

        if rounding_increment > 0.0:
            y_grid = rounding_increment * np.round(y_grid / rounding_increment)
            true_quantile = rounding_increment * np.round(true_quantile / rounding_increment)
            error = y_grid - true_quantile

        if sampling == "informative_grid":
            selected_index = _informative_grid_indices(
                grid_index_candidates,
                y_grid,
                base_probability=index_probability,
                count_range=index_range,
                latent_level=float(level[i]),
                level_strength=level_measurement_strength,
                recent_strength=recent_measurement_strength,
                rng=rng,
            )
            selected_late = _informative_grid_indices(
                grid_late_candidates,
                y_grid,
                base_probability=late_probability,
                count_range=late_range,
                latent_level=float(level[i]),
                level_strength=level_measurement_strength,
                recent_strength=recent_measurement_strength,
                rng=rng,
            )
        elif sampling == "fixed_grid":
            selected_index = _fixed_grid_indices(
                grid_index_candidates,
                index_range,
                latent_level=float(level[i]),
                count_level_strength=count_level_strength,
                rng=rng,
            )
            selected_late = _fixed_grid_indices(
                grid_late_candidates,
                late_range,
                latent_level=float(level[i]),
                count_level_strength=count_level_strength,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown observation sampling mechanism: {sampling}")

        selected = np.r_[selected_index, selected_late]
        y_list.append(np.asarray(y_grid[selected], dtype=float))
        t_list.append(np.asarray(grid_times[selected], dtype=float))
        true_quantile_list.append(np.asarray(true_quantile[selected], dtype=float))
        true_update_list.append(np.asarray(individual[selected], dtype=float))
        error_list.append(np.asarray(error[selected], dtype=float))
        late_selected_mask = grid_times[selected] > 12.0
        true_late_offset[i] = float(np.mean(individual[selected][late_selected_mask]))

    return AdempV2Dataset(
        x=x,
        y_list=y_list,
        t_list=t_list,
        true_quantile_list=true_quantile_list,
        true_update_list=true_update_list,
        error_list=error_list,
        true_late_offset=true_late_offset,
        latent_level=level,
        treatment_trigger=treatment_trigger,
    )


def training_design(data: SimulatedDataset, train_idx: np.ndarray, group_id: int) -> pd.DataFrame:
    repeats = np.asarray([data.y_list[int(i)].size for i in train_idx], dtype=int)
    y = np.concatenate([data.y_list[int(i)] for i in train_idx])
    x_long = np.repeat(data.x[train_idx], repeats, axis=0)
    frame = pd.DataFrame(x_long, columns=PREDICTOR_COLS)
    frame.insert(0, "y", y)
    frame.insert(0, "group_id", int(group_id))
    return frame


def evaluate_split(
    data: SimulatedDataset,
    split_idx: np.ndarray,
    beta_hat: np.ndarray,
    tau: float,
    lambda_b: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    losses: List[float] = []
    b_hats: List[float] = []
    for stay_idx in split_idx:
        i = int(stay_idx)
        t_i = data.t_list[i]
        y_i = data.y_list[i]
        index_idx = t_i <= 12.0
        late_idx = ~index_idx
        residual = y_i - float(data.x[i] @ beta_hat)
        if lambda_b is None:
            b_hat = 0.0
        else:
            b_hat = profiled_intercept(residual[index_idx], tau=tau, lambda_b=float(lambda_b))
        losses.append(float(np.mean(check_loss(residual[late_idx] - b_hat, tau))))
        b_hats.append(float(b_hat))
    return np.asarray(losses, dtype=float), np.asarray(b_hats, dtype=float)


def tune_lambda(
    data: SimulatedDataset,
    tuning_idx: np.ndarray,
    beta_hat: np.ndarray,
    tau: float,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    grid_rows: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None
    for lam in lambda_grid:
        losses, _ = evaluate_split(data, tuning_idx, beta_hat, tau, lambda_b=float(lam))
        row = {
            "lambda_b": float(lam),
            "tuning_loss": float(np.mean(losses)),
            "tuning_loss_se": float(np.std(losses, ddof=1) / np.sqrt(losses.size)),
        }
        grid_rows.append(row)
        if best is None or row["tuning_loss"] < best["tuning_loss"]:
            best = row
    assert best is not None
    return {"best": best, "grid": grid_rows}


def calibration_interval_metrics(
    outcomes: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    tau: float,
) -> Dict[str, float]:
    """Stay-weighted discrete-quantile probability-mass bracket diagnostics.

    A valid discrete tau-quantile satisfies P(Y < q) <= tau <= P(Y <= q).
    Reporting both probabilities avoids declaring a tied empirical quantile
    miscalibrated merely because its strict-below fraction is not exactly tau.
    """

    if len(outcomes) != len(predictions) or len(outcomes) == 0:
        raise ValueError("outcomes and predictions must be non-empty and aligned by stay.")
    below: List[float] = []
    below_equal: List[float] = []
    for y, q in zip(outcomes, predictions):
        y_arr = np.asarray(y, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        if q_arr.ndim == 0:
            q_arr = np.full(y_arr.size, float(q_arr), dtype=float)
        if y_arr.shape != q_arr.shape:
            raise ValueError("Each prediction vector must match its outcome vector.")
        below.append(float(np.mean(y_arr < q_arr)))
        below_equal.append(float(np.mean(y_arr <= q_arr)))
    p_lt = float(np.mean(below))
    p_le = float(np.mean(below_equal))
    violation = float(max(p_lt - tau, tau - p_le, 0.0))
    return {
        "calibration_p_lt": p_lt,
        "calibration_p_le": p_le,
        "calibration_interval_violation": violation,
        "calibration_strict_absolute_error": float(abs(p_lt - tau)),
    }


def fit_affine_q10_calibration(
    data: AdempV2Dataset,
    tuning_idx: np.ndarray,
    tau: float,
) -> Dict[str, object]:
    """Fit a stay-weighted affine q10 comparator by exact linear programming."""

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    raw_identity_loss = 0.0
    for stay_idx in np.asarray(tuning_idx, dtype=int):
        t_i = data.t_list[int(stay_idx)]
        y_i = data.y_list[int(stay_idx)]
        index = t_i <= 12.0
        late = ~index
        raw_q = empirical_check_quantile(y_i[index], tau)
        y_late = np.asarray(y_i[late], dtype=float)
        x_parts.append(np.full(y_late.size, raw_q, dtype=float))
        y_parts.append(y_late)
        weights = np.full(y_late.size, 1.0 / (len(tuning_idx) * y_late.size), dtype=float)
        weight_parts.append(weights)
        raw_identity_loss += float(np.sum(weights * check_loss(y_late - raw_q, tau)))

    predictor = np.concatenate(x_parts)
    outcome = np.concatenate(y_parts)
    weights = np.concatenate(weight_parts)
    n_obs = int(outcome.size)
    # outcome - (intercept + slope * predictor) = positive - negative
    # gives intercept + slope*x + positive - negative = outcome.
    a_eq = sparse.hstack(
        [
            sparse.csr_matrix(np.ones((n_obs, 1), dtype=float)),
            sparse.csr_matrix(predictor.reshape(-1, 1)),
            sparse.eye(n_obs, format="csr"),
            -sparse.eye(n_obs, format="csr"),
        ],
        format="csr",
    )
    objective = np.r_[0.0, 0.0, tau * weights, (1.0 - tau) * weights]
    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n_obs)
    fit = linprog(
        objective,
        A_eq=a_eq,
        b_eq=outcome,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not fit.success or fit.x is None or not np.all(np.isfinite(fit.x[:2])):
        raise RuntimeError(f"Affine q10 calibration LP failed: {fit.message}")
    equality_residual = np.asarray(a_eq @ fit.x - outcome, dtype=float)
    fitted_loss = float(fit.fun)
    tolerance = 1e-8 * max(1.0, abs(raw_identity_loss))
    if fitted_loss > raw_identity_loss + tolerance:
        raise RuntimeError("Calibrated q10 LP did not improve on its feasible identity transformation.")
    return {
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "tuning_loss": fitted_loss,
        "raw_identity_tuning_loss": float(raw_identity_loss),
        "success": True,
        "solver": "scipy.optimize.linprog(method='highs')",
        "solver_status": int(fit.status),
        "solver_message": str(fit.message),
        "max_equality_residual": float(np.max(np.abs(equality_residual))),
        "n_tuning_stays": int(len(tuning_idx)),
        "n_tuning_later_observations": n_obs,
    }


def profiled_level_slope(
    residual_index: np.ndarray,
    time_index: np.ndarray,
    tau: float,
    lam: float,
    max_cycles: int = 100,
) -> Tuple[float, float, Dict[str, object]]:
    """Solve the strictly convex level-plus-slope update through its dual.

    The unused ``max_cycles`` argument is retained for API compatibility with
    the smoke-test version of ADEMP v2.  The implemented solver is now the
    smooth box-constrained Fenchel dual and returns a primal--dual certificate.
    """

    residual = np.asarray(residual_index, dtype=float)
    time_basis = (np.asarray(time_index, dtype=float) - 12.0) / 12.0
    if residual.size != time_basis.size or residual.size == 0:
        raise ValueError("Residuals and times must be non-empty and aligned.")
    if lam <= 0.0:
        raise ValueError("The level-plus-slope penalty must be strictly positive.")
    design = np.column_stack([np.ones(residual.size, dtype=float), time_basis])

    def negative_dual_and_gradient(dual: np.ndarray) -> Tuple[float, np.ndarray]:
        xta = design.T @ dual
        value = -float(dual @ residual) + 0.25 * float(xta @ xta) / lam
        gradient = -residual + 0.5 * (design @ xta) / lam
        return value, gradient

    lower = tau - 1.0
    upper = tau
    fit = minimize(
        negative_dual_and_gradient,
        np.zeros(residual.size, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * residual.size,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000, "maxls": 50},
    )
    if fit.x is None or not np.all(np.isfinite(fit.x)):
        raise RuntimeError("The level-plus-slope dual solver returned non-finite values.")
    dual = np.clip(np.asarray(fit.x, dtype=float), lower, upper)
    # L-BFGS-B occasionally stops on a flat face of this rank-two box QP.
    # Exact cyclic coordinate minimization of the smooth dual cheaply polishes
    # that solution and makes the primal--dual check reliable for sparse stays.
    xta = design.T @ dual
    coordinate_cycles = 0
    for coordinate_cycles in range(1, 5001):
        maximum_change = 0.0
        for row_index in range(residual.size):
            x_row = design[row_index]
            gradient = -residual[row_index] + 0.5 * float(x_row @ xta) / lam
            curvature = 0.5 * float(x_row @ x_row) / lam
            old_value = float(dual[row_index])
            new_value = float(np.clip(old_value - gradient / curvature, lower, upper))
            change = new_value - old_value
            if change != 0.0:
                dual[row_index] = new_value
                xta = xta + change * x_row
                maximum_change = max(maximum_change, abs(change))
        if maximum_change <= 1e-12:
            break
    coefficients = 0.5 * (design.T @ dual) / lam
    fitted_residual = residual - design @ coefficients
    primal = float(np.sum(check_loss(fitted_residual, tau)) + lam * np.sum(np.square(coefficients)))
    xta = design.T @ dual
    dual_value = float(dual @ residual - 0.25 * (xta @ xta) / lam)
    duality_gap = float(primal - dual_value)
    # The prediction loss is reported to 1e-3; a 2e-6 relative primal--dual
    # tolerance is several orders of magnitude tighter than any reported
    # Monte Carlo contrast while avoiding false solver failures on tied faces.
    tolerance = 2e-6 * max(1.0, abs(primal), abs(dual_value))
    converged = bool(duality_gap >= -tolerance and duality_gap <= tolerance)
    if not converged:
        raise RuntimeError(
            "The level-plus-slope dual solver failed its primal--dual certificate: "
            f"success={fit.success}, gap={duality_gap:.3e}, tolerance={tolerance:.3e}."
        )
    return float(coefficients[0]), float(coefficients[1]), {
        "solver": "fenchel_dual_lbfgsb_certified",
        "converged": True,
        "cycles": int(fit.nit),
        "dual_coordinate_polish_cycles": int(coordinate_cycles),
        "objective": primal,
        "dual_objective": dual_value,
        "duality_gap": duality_gap,
        "duality_gap_tolerance": tolerance,
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
    }


def evaluate_level_slope_split(
    data: AdempV2Dataset,
    split_idx: np.ndarray,
    beta_hat: np.ndarray,
    tau: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict[str, float]]:
    losses: List[float] = []
    intercepts: List[float] = []
    slopes: List[float] = []
    predictions: List[np.ndarray] = []
    cycles: List[float] = []
    convergence: List[float] = []
    for stay_idx in np.asarray(split_idx, dtype=int):
        t_i = data.t_list[int(stay_idx)]
        y_i = data.y_list[int(stay_idx)]
        index = t_i <= 12.0
        late = ~index
        population = float(data.x[int(stay_idx)] @ beta_hat)
        residual = y_i - population
        intercept, slope, diagnostics = profiled_level_slope(
            residual[index], t_i[index], tau=tau, lam=lam
        )
        late_prediction = population + intercept + slope * ((t_i[late] - 12.0) / 12.0)
        losses.append(float(np.mean(check_loss(y_i[late] - late_prediction, tau))))
        intercepts.append(intercept)
        slopes.append(slope)
        predictions.append(np.asarray(late_prediction, dtype=float))
        cycles.append(float(diagnostics["cycles"]))
        convergence.append(float(bool(diagnostics["converged"])))
    return (
        np.asarray(losses, dtype=float),
        np.asarray(intercepts, dtype=float),
        np.asarray(slopes, dtype=float),
        predictions,
        {
            "mean_coordinate_cycles": float(np.mean(cycles)),
            "converged_fraction": float(np.mean(convergence)),
        },
    )


def tune_level_slope_lambda(
    data: AdempV2Dataset,
    tuning_idx: np.ndarray,
    beta_hat: np.ndarray,
    tau: float,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None
    positive_grid = [float(value) for value in lambda_grid if float(value) > 0.0]
    if not positive_grid:
        raise ValueError("The level-plus-slope grid must include at least one positive penalty.")
    for lam in positive_grid:
        losses, _, _, _, diagnostics = evaluate_level_slope_split(
            data, tuning_idx, beta_hat, tau=tau, lam=lam
        )
        row = {
            "lambda_shape": lam,
            "tuning_loss": float(np.mean(losses)),
            "tuning_loss_se": float(np.std(losses, ddof=1) / np.sqrt(losses.size)),
            **diagnostics,
        }
        rows.append(row)
        if best is None or row["tuning_loss"] < best["tuning_loss"]:
            best = row
    assert best is not None
    return {"best": best, "grid": rows}


def grouped_quantile_fit(
    design_csv: Path,
    coef_csv: Path,
    tau: float,
    r_helper: Path,
) -> pd.DataFrame:
    subprocess.run(
        ["Rscript", str(r_helper), str(design_csv), f"{tau:.12g}", str(coef_csv), "group_id"],
        check=True,
    )
    return pd.read_csv(coef_csv)


def corr_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _assessment_arrays(
    data: AdempV2Dataset,
    split_idx: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    outcomes: List[np.ndarray] = []
    oracle: List[np.ndarray] = []
    true_updates: List[np.ndarray] = []
    late_times: List[np.ndarray] = []
    for stay_idx in np.asarray(split_idx, dtype=int):
        i = int(stay_idx)
        late = data.t_list[i] > 12.0
        outcomes.append(np.asarray(data.y_list[i][late], dtype=float))
        oracle.append(np.asarray(data.true_quantile_list[i][late], dtype=float))
        true_updates.append(np.asarray(data.true_update_list[i][late], dtype=float))
        late_times.append(np.asarray(data.t_list[i][late], dtype=float))
    return outcomes, oracle, true_updates, late_times


def score_method(
    method: str,
    outcomes: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    oracle_predictions: Sequence[np.ndarray],
    tau: float,
    estimated_offsets: np.ndarray | None = None,
    true_offsets: np.ndarray | None = None,
) -> Dict[str, object]:
    losses: List[float] = []
    oracle_losses: List[float] = []
    for y, prediction, oracle in zip(outcomes, predictions, oracle_predictions):
        y_arr = np.asarray(y, dtype=float)
        q_arr = np.asarray(prediction, dtype=float)
        oracle_arr = np.asarray(oracle, dtype=float)
        if q_arr.ndim == 0:
            q_arr = np.full(y_arr.size, float(q_arr), dtype=float)
        losses.append(float(np.mean(check_loss(y_arr - q_arr, tau))))
        oracle_losses.append(float(np.mean(check_loss(y_arr - oracle_arr, tau))))
    loss = float(np.mean(losses))
    oracle_loss = float(np.mean(oracle_losses))
    row: Dict[str, object] = {
        "method": method,
        "loss": loss,
        "oracle_loss": oracle_loss,
        "regret": float(loss - oracle_loss),
        "regret_percent_of_oracle": float(100.0 * (loss - oracle_loss) / oracle_loss),
        "stay_loss_sd": float(np.std(losses, ddof=1)),
        "n_assessment_stays": int(len(losses)),
        **calibration_interval_metrics(outcomes, predictions, tau),
    }
    if estimated_offsets is not None and true_offsets is not None:
        errors = np.asarray(estimated_offsets, dtype=float) - np.asarray(true_offsets, dtype=float)
        row.update(
            {
                "offset_bias": float(np.mean(errors)),
                "offset_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "offset_correlation": corr_or_nan(estimated_offsets, true_offsets),
            }
        )
    else:
        row.update(
            {
                "offset_bias": float("nan"),
                "offset_rmse": float("nan"),
                "offset_correlation": float("nan"),
            }
        )
    return row


def ademp_v2_scenario_diagnostics(data: AdempV2Dataset) -> Dict[str, float]:
    index_counts = np.asarray([np.sum(t <= 12.0) for t in data.t_list], dtype=float)
    late_counts = np.asarray([np.sum(t > 12.0) for t in data.t_list], dtype=float)
    lag_left: List[np.ndarray] = []
    lag_right: List[np.ndarray] = []
    for errors in data.error_list:
        if len(errors) >= 2:
            lag_left.append(np.asarray(errors[:-1], dtype=float))
            lag_right.append(np.asarray(errors[1:], dtype=float))
    if lag_left:
        serial_correlation = corr_or_nan(np.concatenate(lag_left), np.concatenate(lag_right))
    else:
        serial_correlation = float("nan")
    return {
        "mean_index_count": float(np.mean(index_counts)),
        "mean_late_count": float(np.mean(late_counts)),
        "sd_index_count": float(np.std(index_counts, ddof=1)),
        "sd_late_count": float(np.std(late_counts, ddof=1)),
        "total_count_latent_level_correlation": corr_or_nan(index_counts + late_counts, data.latent_level),
        "observed_error_lag1_correlation": serial_correlation,
        "treatment_trigger_fraction": float(np.mean(data.treatment_trigger)),
    }


def summarize_mc_replicates(method_replicates: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "loss",
        "oracle_loss",
        "regret",
        "regret_percent_of_oracle",
        "calibration_p_lt",
        "calibration_p_le",
        "calibration_interval_violation",
        "calibration_strict_absolute_error",
        "offset_bias",
        "offset_rmse",
        "offset_correlation",
    ]
    rows: List[Dict[str, object]] = []
    grouped = method_replicates.groupby(
        ["scenario_key", "scenario_label", "n_stays", "method"], sort=False, dropna=False
    )
    for keys, local in grouped:
        scenario_key, scenario_label, n_stays, method = keys
        row: Dict[str, object] = {
            "scenario_key": scenario_key,
            "scenario_label": scenario_label,
            "n_stays": int(n_stays),
            "method": method,
            "n_attempted_replicates": int(local.shape[0]),
            "n_effective_replicates": int(np.sum(np.isfinite(local["loss"].to_numpy(dtype=float)))),
        }
        row["failure_count"] = int(row["n_attempted_replicates"] - row["n_effective_replicates"])
        row["failure_rate"] = float(row["failure_count"] / row["n_attempted_replicates"])
        for metric in metrics:
            values = local[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                mean_value = float("nan")
                mcse = float("nan")
            else:
                mean_value = float(np.mean(finite))
                mcse = (
                    float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                    if finite.size > 1
                    else float("nan")
                )
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_mcse"] = mcse
            row[f"{metric}_mc_ci_lower"] = mean_value - 1.96 * mcse
            row[f"{metric}_mc_ci_upper"] = mean_value + 1.96 * mcse
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_penalty_selection(penalties: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (scenario_key, scenario_label, method), local in penalties.groupby(
        ["scenario_key", "scenario_label", "method"], sort=False
    ):
        total = int(local.shape[0])
        for penalty, selected in local.groupby("selected_penalty", sort=True):
            rows.append(
                {
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label,
                    "method": method,
                    "selected_penalty": float(penalty),
                    "selected_count": int(selected.shape[0]),
                    "n_replicates": total,
                    "selected_proportion": float(selected.shape[0] / total),
                }
            )
    return pd.DataFrame(rows)


ADEMP_V2_PAIRED_CONTRASTS = (
    ("tuned_level", "population"),
    ("affine_calibrated_q10", "population"),
    ("tuned_level", "affine_calibrated_q10"),
    ("tuned_level_slope", "tuned_level"),
    ("raw_index_q10", "affine_calibrated_q10"),
)


def summarize_paired_loss_differences(method_replicates: pd.DataFrame) -> pd.DataFrame:
    """Paired Monte Carlo method differences; negative values favor method A."""

    rows: List[Dict[str, object]] = []
    for (scenario_key, scenario_label), local in method_replicates.groupby(
        ["scenario_key", "scenario_label"], sort=False
    ):
        pivot = local.pivot(index="rep", columns="method", values="loss")
        for method_a, method_b in ADEMP_V2_PAIRED_CONTRASTS:
            if method_a not in pivot or method_b not in pivot:
                continue
            differences = (pivot[method_a] - pivot[method_b]).to_numpy(dtype=float)
            finite = differences[np.isfinite(differences)]
            mean_value = float(np.mean(finite)) if finite.size else float("nan")
            mcse = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else float("nan")
            )
            rows.append(
                {
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label,
                    "method_a": method_a,
                    "method_b": method_b,
                    "estimand": "mean paired stay-level assessment check-loss difference (A - B)",
                    "n_effective_replicates": int(finite.size),
                    "loss_difference_mean": mean_value,
                    "loss_difference_mcse": mcse,
                    "loss_difference_mc_ci_lower": mean_value - 1.96 * mcse,
                    "loss_difference_mc_ci_upper": mean_value + 1.96 * mcse,
                }
            )
    return pd.DataFrame(rows)


def write_ademp_v2_key_conclusions(
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# ADEMP v2: manuscript-ready key conclusions",
        "",
        "Negative paired loss differences favor the first named method. Monte Carlo intervals quantify simulation error only.",
        "",
    ]
    tuned_vs_population = paired[
        (paired["method_a"] == "tuned_level") & (paired["method_b"] == "population")
    ]
    tuned_vs_affine = paired[
        (paired["method_a"] == "tuned_level")
        & (paired["method_b"] == "affine_calibrated_q10")
    ]
    better_population = tuned_vs_population[
        tuned_vs_population["loss_difference_mc_ci_upper"] < 0.0
    ]["scenario_label"].tolist()
    worse_population = tuned_vs_population[
        tuned_vs_population["loss_difference_mc_ci_lower"] > 0.0
    ]["scenario_label"].tolist()
    better_affine = tuned_vs_affine[
        tuned_vs_affine["loss_difference_mc_ci_upper"] < 0.0
    ]["scenario_label"].tolist()
    worse_affine = tuned_vs_affine[
        tuned_vs_affine["loss_difference_mc_ci_lower"] > 0.0
    ]["scenario_label"].tolist()
    lines.extend(
        [
            "## Directly supported summary",
            "",
            f"- The profiled offset rule had lower loss than the population-only rule with a Monte Carlo 95% interval below zero in {len(better_population)} {'scenario' if len(better_population) == 1 else 'scenarios'}: "
            + (", ".join(better_population) if better_population else "none")
            + ".",
            f"- It had higher loss than the population-only rule with a Monte Carlo 95% interval above zero in {len(worse_population)} {'scenario' if len(worse_population) == 1 else 'scenarios'}: "
            + (", ".join(worse_population) if worse_population else "none")
            + ".",
            f"- The profiled offset rule had lower loss than the tuning-calibrated affine q10 comparator in {len(better_affine)} {'scenario' if len(better_affine) == 1 else 'scenarios'}: "
            + (", ".join(better_affine) if better_affine else "none")
            + ".",
            f"- The affine q10 comparator had lower loss than the profiled offset rule in {len(worse_affine)} {'scenario' if len(worse_affine) == 1 else 'scenarios'}: "
            + (", ".join(worse_affine) if worse_affine else "none")
            + ".",
            "",
            "## Scenario-level paired estimates",
            "",
            "| Scenario | Contrast (A - B) | Difference | MCSE | Monte Carlo 95% interval | Effective replicates |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired.to_dict("records"):
        lines.append(
            f"| {row['scenario_label']} | {row['method_a']} - {row['method_b']} | "
            f"{row['loss_difference_mean']:.4f} | {row['loss_difference_mcse']:.4f} | "
            f"[{row['loss_difference_mc_ci_lower']:.4f}, {row['loss_difference_mc_ci_upper']:.4f}] | "
            f"{row['n_effective_replicates']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These experiments support statements about predictive check loss under the declared data-generating mechanisms. "
            "They do not establish clinical utility, external transportability, or universal superiority over other longitudinal quantile methods.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ademp_v2_summary(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    out_pdf: Path,
    out_svg: Path,
    out_png: Path,
) -> None:
    """Four-panel, claim-aligned ADEMP-v2 summary with Monte Carlo intervals."""

    available_keys = set(summary["scenario_key"])
    scenario_keys = [key for key in ADEMP_V2_SCENARIOS if key in available_keys]
    scenario_labels = [str(ADEMP_V2_SCENARIOS[key]["label"]) for key in scenario_keys]
    scenario_keys = list(reversed(scenario_keys))
    scenario_labels = list(reversed(scenario_labels))
    y_position = np.arange(len(scenario_keys), dtype=float)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure_height = max(5.5, 1.30 + 0.42 * len(scenario_keys))
        fig, axes = plt.subplots(2, 2, figsize=(7.2047, figure_height), sharey=True)
        contrasts = [
            ("tuned_level", "population", "A. Profiled offset vs population", "negative favors profiled offset"),
            ("tuned_level", "affine_calibrated_q10", "B. Profiled offset vs calibrated q10", "negative favors profiled offset"),
            ("tuned_level_slope", "tuned_level", "C. Level + slope vs profiled offset", "negative favors level + slope"),
        ]
        for axis, (method_a, method_b, title, favor_text) in zip(axes.ravel()[:3], contrasts):
            local = paired[
                (paired["method_a"] == method_a) & (paired["method_b"] == method_b)
            ].set_index("scenario_key")
            means = np.asarray(
                [float(local.loc[key, "loss_difference_mean"]) for key in scenario_keys], dtype=float
            )
            errors = 1.96 * np.asarray(
                [float(local.loc[key, "loss_difference_mcse"]) for key in scenario_keys], dtype=float
            )
            colors = np.where(means <= 0.0, "#246a73", "#b85c32")
            axis.axvspan(axis.get_xlim()[0], 0.0, color="#eaf3f2", alpha=0.55, zorder=0)
            axis.errorbar(
                means,
                y_position,
                xerr=errors,
                fmt="none",
                ecolor="#4a4a4a",
                elinewidth=1.0,
                capsize=2.2,
                zorder=2,
            )
            axis.scatter(means, y_position, c=colors, s=34, edgecolor="white", linewidth=0.5, zorder=3)
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.9)
            axis.set_title(title, loc="left")
            axis.set_xlabel(f"Paired check-loss difference (A - B)\n{favor_text}")
            axis.grid(axis="x", color="#dddddd", linewidth=0.65)
            axis.set_yticks(y_position)
            axis.set_yticklabels(scenario_labels)

        axis = axes.ravel()[3]
        calibration_methods = [
            ("population", "Population", "#7f7f7f", -0.22),
            ("affine_calibrated_q10", "Calibrated q10", "#2a9d8f", 0.0),
            ("tuned_level", "Profiled offset", "#264653", 0.22),
        ]
        indexed_summary = summary.set_index(["scenario_key", "method"])
        for method, label, color, offset in calibration_methods:
            means = np.asarray(
                [
                    float(indexed_summary.loc[(key, method), "calibration_interval_violation_mean"])
                    for key in scenario_keys
                ],
                dtype=float,
            )
            errors = 1.96 * np.asarray(
                [
                    float(indexed_summary.loc[(key, method), "calibration_interval_violation_mcse"])
                    for key in scenario_keys
                ],
                dtype=float,
            )
            axis.errorbar(
                means,
                y_position + offset,
                xerr=errors,
                fmt="o",
                color=color,
                ecolor=color,
                markersize=4.0,
                elinewidth=0.9,
                capsize=1.8,
                label=label,
            )
        axis.axvline(0.0, color="#222222", linewidth=0.8)
        axis.set_xlim(left=-0.003)
        axis.set_title("D. Probability-mass calibration", loc="left")
        axis.set_xlabel("Probability-mass bracket violation")
        axis.grid(axis="x", color="#dddddd", linewidth=0.65)
        axis.set_yticks(y_position)
        axis.set_yticklabels(scenario_labels)
        axis.legend(loc="lower right", frameon=False)

        for axis in axes.ravel():
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        fig.suptitle(
            "ADEMP v2: performance against prespecified strong comparators",
            fontsize=10.8,
            y=0.995,
        )
        fig.text(
            0.5,
            0.014,
            "Points are Monte Carlo means; horizontal bars are Monte Carlo 95% intervals (mean ± 1.96 MCSE).",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        fig.subplots_adjust(left=0.315, right=0.985, top=0.950, bottom=0.125, wspace=0.22, hspace=0.28)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _tex_escape(text_value: object) -> str:
    text_value = str(text_value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
    }
    for source, target in replacements.items():
        text_value = text_value.replace(source, target)
    return text_value


def write_ademp_v2_summary_tex(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    path: Path,
    n_rep: int,
) -> None:
    indexed = summary.set_index(["scenario_key", "method"])
    paired_index = paired.set_index(["scenario_key", "method_a", "method_b"])

    def mean_mcse(key: str, method: str, metric: str = "loss") -> str:
        row = indexed.loc[(key, method)]
        return f"{float(row[f'{metric}_mean']):.3f} ({float(row[f'{metric}_mcse']):.3f})"

    def paired_mean_mcse(key: str, first: str, second: str) -> str:
        row = paired_index.loc[(key, first, second)]
        return f"{float(row['loss_difference_mean']):+.4f} ({float(row['loss_difference_mcse']):.4f})"

    lines = [
        "\\begin{table*}[!htbp]",
        "\\caption{Extended ADEMP simulation with strong comparators and Monte Carlo precision.}",
        "\\label{tab:ademp_v2_summary}",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Scenario & Population & Calibrated q10 & Profiled offset & Level + slope & $\\Delta_{\\mathrm{offset-pop}}$ & $\\Delta_{\\mathrm{offset-cal}}$\\\\",
        "\\hline",
    ]
    for key in ADEMP_V2_SCENARIOS:
        if key not in set(summary["scenario_key"]):
            continue
        lines.append(
            f"{_tex_escape(ADEMP_V2_SCENARIOS[key]['label'])} & "
            f"{mean_mcse(key, 'population')} & "
            f"{mean_mcse(key, 'affine_calibrated_q10')} & "
            f"{mean_mcse(key, 'tuned_level')} & "
            f"{mean_mcse(key, 'tuned_level_slope')} & "
            f"{paired_mean_mcse(key, 'tuned_level', 'population')} & "
            f"{paired_mean_mcse(key, 'tuned_level', 'affine_calibrated_q10')}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\par\\smallskip",
            (
                "\\footnotesize{Entries are mean stay-level assessment check loss (Monte Carlo standard error) across "
                f"{int(n_rep)} independent simulated datasets per scenario. Differences are paired within replicate; "
                "negative values in both $\\Delta$ columns favor the profiled offset rule. All methods had "
                f"{int(n_rep)}/{int(n_rep)} effective replicates (failure rate 0\\%). Calibrated q10 is an affine transformation fitted on tuning stays "
                "by stay-weighted quantile loss. The 240-stay and 600-stay settings, dense and sparse observation regimes, serial "
                "dependence, informative monitoring, cluster-size informativeness, common-time misspecification, random shape, "
                "treatment feedback, transient non-persistence, pure null, weak persistent signal, Student-$t_3$ heavy tails, "
                "and integer rounding are prespecified data-generating mechanisms.}"
            ),
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_ademp_v2_manuscript_assets(paths: Mapping[str, Path], template_dir: Path | None) -> None:
    if template_dir is None:
        return
    targets = [template_dir]
    submission = template_dir / "revised_quantile_adaptive_basis_mimic_submission_20260531"
    if submission.exists():
        targets.append(submission)
    for target in targets:
        figure_dir = target / "figures"
        table_dir = target / "tables"
        figure_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        for key in ("summary_figure_pdf", "summary_figure_svg", "summary_figure_png"):
            source = paths[key]
            (figure_dir / source.name).write_bytes(source.read_bytes())
        source_table = paths["summary_tex"]
        (table_dir / source_table.name).write_text(
            source_table.read_text(encoding="utf-8"), encoding="utf-8"
        )


def write_ademp_v2_markdown(
    summary: pd.DataFrame,
    scenario_diagnostics: pd.DataFrame,
    penalty_summary: pd.DataFrame,
    path: Path,
    parameters: Mapping[str, object],
) -> None:
    lines = [
        "# ADEMP v2 split-window simulation",
        "",
        "## Monte Carlo design",
        "",
        f"- Independent unit: stay; independent Monte Carlo unit: complete simulated dataset.",
        f"- Replicates per scenario: {parameters['n_rep']}.",
        f"- Target quantile: {parameters['tau']}.",
        "- Losses and calibration probabilities are first averaged within stay and then across stays.",
        "- A discrete quantile obeys the probability-mass bracket P(Y < q) <= tau <= P(Y <= q); probability-mass bracket violation is the distance outside that bracket.",
        "- Parenthesized values below are Monte Carlo standard errors across independent datasets.",
        "",
        "## Performance summary",
        "",
        "| Scenario | Method | Effective/attempted | Failure (%) | Loss (MCSE) | Regret (MCSE) | P(Y<q) | P(Y<=q) | Probability-mass bracket violation (MCSE) | Offset RMSE (MCSE) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        offset_text = "--"
        if np.isfinite(float(row["offset_rmse_mean"])):
            offset_text = f"{row['offset_rmse_mean']:.3f} ({row['offset_rmse_mcse']:.3f})"
        lines.append(
            f"| {row['scenario_label']} | {row['method']} | "
            f"{row['n_effective_replicates']}/{row['n_attempted_replicates']} | "
            f"{100.0 * row['failure_rate']:.1f} | "
            f"{row['loss_mean']:.3f} ({row['loss_mcse']:.3f}) | "
            f"{row['regret_mean']:.3f} ({row['regret_mcse']:.3f}) | "
            f"{row['calibration_p_lt_mean']:.3f} | {row['calibration_p_le_mean']:.3f} | "
            f"{row['calibration_interval_violation_mean']:.3f} "
            f"({row['calibration_interval_violation_mcse']:.3f}) | {offset_text} |"
        )
    lines.extend(["", "## Data-generating-mechanism checks", ""])
    diagnostic_summary = (
        scenario_diagnostics.groupby(["scenario_key", "scenario_label"], sort=False)
        .agg(
            mean_index_count=("mean_index_count", "mean"),
            mean_late_count=("mean_late_count", "mean"),
            count_level_r=("total_count_latent_level_correlation", "mean"),
            observed_error_lag1_r=("observed_error_lag1_correlation", "mean"),
            treatment_fraction=("treatment_trigger_fraction", "mean"),
        )
        .reset_index()
    )
    lines.extend(
        [
            "| Scenario | Mean index n | Mean later n | Count--level r | Observed error lag-1 r | Treated fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_summary.to_dict("records"):
        lines.append(
            f"| {row['scenario_label']} | {row['mean_index_count']:.2f} | "
            f"{row['mean_late_count']:.2f} | {row['count_level_r']:.3f} | "
            f"{row['observed_error_lag1_r']:.3f} | {row['treatment_fraction']:.3f} |"
        )
    lines.extend(["", "## Penalty selection", ""])
    if penalty_summary.empty:
        lines.append("No penalty-selection records were produced.")
    else:
        lines.extend(
            [
                "| Scenario | Method | Penalty | Selected n/N (%) |",
                "|---|---|---:|---:|",
            ]
        )
        for row in penalty_summary.to_dict("records"):
            lines.append(
                f"| {row['scenario_label']} | {row['method']} | {row['selected_penalty']:g} | "
                f"{row['selected_count']}/{row['n_replicates']} ({100.0 * row['selected_proportion']:.1f}) |"
            )
    lines.extend(
        [
            "",
            "## Precision boundary",
            "",
            "Monte Carlo confidence intervals quantify simulation error only. They do not validate the data-generating mechanisms, "
            "and no fixed replicate count is treated as automatically sufficient; the reported MCSE should be compared with the "
            "smallest method difference the manuscript intends to interpret.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_mean_se(mean_value: float, se_value: float, digits: int = 3) -> str:
    return f"{mean_value:.{digits}f} ({se_value:.{digits}f})"


def format_optional_mean_se(mean_value: float, se_value: float, digits: int = 3) -> str:
    if not np.isfinite(mean_value):
        return "--"
    return format_mean_se(mean_value, se_value, digits=digits)


def write_summary_tex(summary: pd.DataFrame, path: Path, n_rep: int, n_stays: int) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Simulation assessment of the trajectory updating model.}",
        "\\label{tab:simulation_study}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Scenario & $\\lambda_b$ & Pop. & Unpen. offset & Profiled offset & Reduction (\\%) & Offset $r$\\\\",
        "\\hline",
    ]
    for row in summary.to_dict("records"):
        lambda_text = f"{row['lambda_median']:.2g} [{row['lambda_q1']:.2g}, {row['lambda_q3']:.2g}]"
        lines.append(
            f"{row['scenario_label']} & {lambda_text} & "
            f"{format_mean_se(row['population_loss_mean'], row['population_loss_mcse'])} & "
            f"{format_mean_se(row['unpenalized_loss_mean'], row['unpenalized_loss_mcse'])} & "
            f"{format_mean_se(row['tuned_loss_mean'], row['tuned_loss_mcse'])} & "
            f"{format_mean_se(row['tuned_reduction_percent_mean'], row['tuned_reduction_percent_mcse'])} & "
            f"{format_optional_mean_se(row['tuned_b_true_late_offset_corr_mean'], row['tuned_b_true_late_offset_corr_mcse'])}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            (
                "\\footnotesize{Each scenario used "
                f"{n_rep} Monte Carlo replicates with {n_stays} stays per replicate. "
                "Losses are mean stay level later check losses at $\\tau=0.10$; values in parentheses are Monte Carlo standard errors. "
                "The selected penalty is reported as median [IQR] across replicates. Offset correlation compares the profiled offset "
                "with the true later latent offset when that offset has nonzero simulation variance.}"
            ),
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_simulation(replicates: pd.DataFrame, summary: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    scenario_order = [SCENARIOS[key]["label"] for key in SCENARIOS]
    y_labels = list(reversed(scenario_order))
    y_plot = np.arange(len(y_labels), dtype=float)
    ordered = summary.set_index("scenario_label").loc[y_labels]

    def value_label(value: float) -> str:
        return f"{value:+.1f}"

    with plt.rc_context(
        {
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8,
        }
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.4, 7.4),
            gridspec_kw={"width_ratios": [1.05, 1.05], "height_ratios": [1.0, 1.0]},
        )

        ax = axes[0, 0]
        tuned_mean = ordered["tuned_reduction_percent_mean"].to_numpy(dtype=float)
        tuned_ci = 1.96 * ordered["tuned_reduction_percent_mcse"].to_numpy(dtype=float)
        tuned_colors = np.where(tuned_mean >= 0.0, "#246a73", "#b85c32")
        ax.axvspan(-1.5, 0.0, color="#f8ebe6", zorder=0)
        ax.barh(y_plot, tuned_mean, color=tuned_colors, alpha=0.88, height=0.58, zorder=2)
        ax.errorbar(
            tuned_mean,
            y_plot,
            xerr=tuned_ci,
            fmt="none",
            color="#222222",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        for y, value in zip(y_plot, tuned_mean):
            if value >= 0.0:
                ax.text(value + 0.45, y, value_label(value), va="center", ha="left", color="#1f3f43")
            else:
                ax.text(0.35, y, value_label(value), va="center", ha="left", color="#7a3a23")
        ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_plot)
        ax.set_yticklabels(y_labels)
        ax.set_xlim(-1.5, 22.5)
        ax.set_xlabel("Loss reduction vs population (%)")
        ax.set_title("A. Profiled offset: zoomed scale")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)

        ax = axes[0, 1]
        unpen_mean = ordered["unpenalized_reduction_percent_mean"].to_numpy(dtype=float)
        unpen_ci = 1.96 * ordered["unpenalized_reduction_percent_mcse"].to_numpy(dtype=float)
        unpen_colors = np.where(unpen_mean >= 0.0, "#246a73", "#b85c32")
        ax.axvspan(-90.0, 0.0, color="#f8ebe6", zorder=0)
        ax.barh(y_plot, unpen_mean, color=unpen_colors, alpha=0.82, height=0.58, zorder=2)
        ax.errorbar(
            unpen_mean,
            y_plot,
            xerr=unpen_ci,
            fmt="none",
            color="#222222",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        for y, value in zip(y_plot, unpen_mean):
            if value >= 0.0:
                ax.text(value + 1.2, y, value_label(value), va="center", ha="left", color="#1f3f43")
            else:
                ax.text(value - 1.2, y, value_label(value), va="center", ha="right", color="#7a3a23")
        ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_plot)
        ax.set_yticklabels([])
        ax.set_xlim(-90.0, 25.0)
        ax.set_xlabel("Loss reduction vs population (%)")
        ax.set_title("B. Unpenalized offset: overfitting scale")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)

        ax = axes[1, 0]
        lambda_values = sorted(replicates["selected_lambda"].unique())
        lambda_matrix = []
        for label in y_labels:
            local = replicates.loc[replicates["scenario_label"] == label, "selected_lambda"].to_numpy(dtype=float)
            lambda_matrix.append([float(np.mean(np.isclose(local, lam))) for lam in lambda_values])
        lambda_matrix_arr = np.asarray(lambda_matrix, dtype=float)
        image = ax.imshow(lambda_matrix_arr, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="YlGnBu")
        for yi in range(lambda_matrix_arr.shape[0]):
            for xi in range(lambda_matrix_arr.shape[1]):
                prop = lambda_matrix_arr[yi, xi]
                if prop >= 0.08:
                    ax.text(
                        xi,
                        yi,
                        f"{100.0 * prop:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color="white" if prop >= 0.45 else "#1f2d32",
                    )
        ax.set_yticks(y_plot)
        ax.set_yticklabels(y_labels)
        ax.set_xticks(np.arange(len(lambda_values)))
        ax.set_xticklabels([f"{lam:g}" for lam in lambda_values])
        ax.set_xlabel("Selected penalty $\\lambda_b$")
        ax.set_title("C. Tuning behavior (% of replicates)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Proportion", rotation=90)

        ax = axes[1, 1]
        corr_mean = ordered["tuned_b_true_late_offset_corr_mean"].to_numpy(dtype=float)
        corr_ci = 1.96 * ordered["tuned_b_true_late_offset_corr_mcse"].to_numpy(dtype=float)
        finite = np.isfinite(corr_mean)
        ax.barh(y_plot[finite], corr_mean[finite], color="#246a73", alpha=0.88, height=0.58, zorder=2)
        ax.errorbar(
            corr_mean[finite],
            y_plot[finite],
            xerr=corr_ci[finite],
            fmt="none",
            color="#222222",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        for y, value, is_finite in zip(y_plot, corr_mean, finite):
            if is_finite:
                ax.text(value + 0.018, y, f"{value:.2f}", va="center", ha="left", color="#1f3f43")
            else:
                ax.text(0.04, y, "not defined", va="center", ha="left", color="#777777")
        ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlim(0.0, 0.92)
        ax.set_yticks(y_plot)
        ax.set_yticklabels([])
        ax.set_xlabel("Correlation with true later offset")
        ax.set_title("D. Offset recovery by profiling")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)

        for ax in axes.ravel():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.10, wspace=0.16, hspace=0.38)
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _json_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def _comma_separated_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def run_ademp_v2(
    args: argparse.Namespace,
    scenarios: Mapping[str, Mapping[str, object]],
) -> None:
    """Run the versioned submission-strengthening ADEMP experiment."""

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Dict[str, object]] = []
    train_frames: List[pd.DataFrame] = []
    full_scenario_order = list(ADEMP_V2_SCENARIOS)
    for scenario_key, scenario in scenarios.items():
        configured_n_stays = int(scenario["n_stays"])
        n_stays = int(args.n_stays) if args.n_stays is not None else configured_n_stays
        scenario_position = full_scenario_order.index(scenario_key)
        for local_rep in range(int(args.n_rep)):
            global_rep_zero_based = int(args.rep_offset) + local_rep
            group_id = scenario_position * 1_000_000 + global_rep_zero_based
            rng = np.random.default_rng(args.seed + 1009 * group_id)
            split_rng = np.random.default_rng(args.seed + 7919 * group_id + 17)
            data = simulate_ademp_v2_dataset(n_stays, args.tau, scenario, rng)
            train_idx, tuning_idx, assessment_idx = split_cluster_indices(n_stays, split_rng)
            train_frames.append(training_design(data, train_idx, group_id))
            jobs.append(
                {
                    "group_id": group_id,
                    "scenario_key": scenario_key,
                    "scenario_label": str(scenario["label"]),
                    "scenario_n_stays": configured_n_stays,
                    "n_stays": n_stays,
                    "rep": global_rep_zero_based + 1,
                    "data": data,
                    "train_idx": train_idx,
                    "tuning_idx": tuning_idx,
                    "assessment_idx": assessment_idx,
                }
            )

    design_csv = args.work_dir / "ademp_v2_training_design.csv"
    coefficient_csv = args.work_dir / "ademp_v2_training_coefficients.csv"
    pd.concat(train_frames, ignore_index=True).to_csv(design_csv, index=False)
    del train_frames
    coefficient_frame = grouped_quantile_fit(
        design_csv, coefficient_csv, args.tau, args.r_helper
    )
    coefficient_wide = coefficient_frame.pivot(index="group_id", columns="term", values="estimate")

    method_rows: List[Dict[str, object]] = []
    penalty_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    for job_number, job in enumerate(jobs, start=1):
        gid = int(job["group_id"])
        data = job["data"]
        assert isinstance(data, AdempV2Dataset)
        beta_hat = coefficient_wide.loc[gid, PREDICTOR_COLS].to_numpy(dtype=float)
        tuning_idx = np.asarray(job["tuning_idx"], dtype=int)
        assessment_idx = np.asarray(job["assessment_idx"], dtype=int)

        level_tuning = tune_lambda(
            data, tuning_idx, beta_hat, args.tau, args.lambda_grid
        )
        selected_level_penalty = float(level_tuning["best"]["lambda_b"])
        shape_tuning = tune_level_slope_lambda(
            data, tuning_idx, beta_hat, args.tau, args.shape_lambda_grid
        )
        selected_shape_penalty = float(shape_tuning["best"]["lambda_shape"])
        calibration = fit_affine_q10_calibration(data, tuning_idx, args.tau)

        outcomes, oracle_predictions, _, late_times = _assessment_arrays(data, assessment_idx)
        true_offsets = data.true_late_offset[assessment_idx]
        population_predictions: List[np.ndarray] = []
        raw_q10_predictions: List[np.ndarray] = []
        calibrated_q10_predictions: List[np.ndarray] = []
        unpenalized_predictions: List[np.ndarray] = []
        tuned_predictions: List[np.ndarray] = []
        _, unpenalized_offsets = evaluate_split(
            data, assessment_idx, beta_hat, args.tau, lambda_b=0.0
        )
        _, tuned_offsets = evaluate_split(
            data,
            assessment_idx,
            beta_hat,
            args.tau,
            lambda_b=selected_level_penalty,
        )
        for local_index, stay_idx in enumerate(assessment_idx):
            i = int(stay_idx)
            t_i = data.t_list[i]
            y_i = data.y_list[i]
            index = t_i <= 12.0
            late = ~index
            population = float(data.x[i] @ beta_hat)
            raw_q = empirical_check_quantile(y_i[index], args.tau)
            n_late = int(np.sum(late))
            population_predictions.append(np.full(n_late, population, dtype=float))
            raw_q10_predictions.append(np.full(n_late, raw_q, dtype=float))
            calibrated_q = float(calibration["intercept"]) + float(calibration["slope"]) * raw_q
            calibrated_q10_predictions.append(np.full(n_late, calibrated_q, dtype=float))
            unpenalized_predictions.append(
                np.full(n_late, population + unpenalized_offsets[local_index], dtype=float)
            )
            tuned_predictions.append(
                np.full(n_late, population + tuned_offsets[local_index], dtype=float)
            )

        (
            _,
            shape_intercepts,
            shape_slopes,
            shape_predictions,
            shape_solver_diagnostics,
        ) = evaluate_level_slope_split(
            data,
            assessment_idx,
            beta_hat,
            args.tau,
            lam=selected_shape_penalty,
        )
        shape_mean_offsets = np.asarray(
            [
                intercept
                + slope * float(np.mean((times - 12.0) / 12.0))
                for intercept, slope, times in zip(shape_intercepts, shape_slopes, late_times)
            ],
            dtype=float,
        )

        prediction_map: Dict[str, Sequence[np.ndarray]] = {
            "oracle": oracle_predictions,
            "population": population_predictions,
            "raw_index_q10": raw_q10_predictions,
            "affine_calibrated_q10": calibrated_q10_predictions,
            "unpenalized_level": unpenalized_predictions,
            "tuned_level": tuned_predictions,
            "tuned_level_slope": shape_predictions,
        }
        offset_map: Dict[str, np.ndarray] = {
            "unpenalized_level": unpenalized_offsets,
            "tuned_level": tuned_offsets,
            "tuned_level_slope": shape_mean_offsets,
        }
        for method in ADEMP_V2_METHODS:
            score = score_method(
                method,
                outcomes,
                prediction_map[method],
                oracle_predictions,
                args.tau,
                estimated_offsets=offset_map.get(method),
                true_offsets=true_offsets if method in offset_map else None,
            )
            method_rows.append(
                {
                    "design_version": "ademp-v2",
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "scenario_n_stays": int(job["scenario_n_stays"]),
                    "n_stays": int(job["n_stays"]),
                    "rep": int(job["rep"]),
                    **score,
                }
            )

        penalty_rows.extend(
            [
                {
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "rep": int(job["rep"]),
                    "method": "tuned_level",
                    "selected_penalty": selected_level_penalty,
                },
                {
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "rep": int(job["rep"]),
                    "method": "tuned_level_slope",
                    "selected_penalty": selected_shape_penalty,
                },
            ]
        )
        for row in level_tuning["grid"]:
            tuning_rows.append(
                {
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "rep": int(job["rep"]),
                    "method": "tuned_level",
                    "penalty": float(row["lambda_b"]),
                    "tuning_loss": float(row["tuning_loss"]),
                    "tuning_loss_se": float(row["tuning_loss_se"]),
                }
            )
        for row in shape_tuning["grid"]:
            tuning_rows.append(
                {
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "rep": int(job["rep"]),
                    "method": "tuned_level_slope",
                    "penalty": float(row["lambda_shape"]),
                    "tuning_loss": float(row["tuning_loss"]),
                    "tuning_loss_se": float(row["tuning_loss_se"]),
                    "mean_coordinate_cycles": float(row["mean_coordinate_cycles"]),
                    "converged_fraction": float(row["converged_fraction"]),
                }
            )
        calibration_rows.append(
            {
                "group_id": gid,
                "scenario_key": job["scenario_key"],
                "scenario_label": job["scenario_label"],
                "rep": int(job["rep"]),
                **calibration,
            }
        )
        diagnostic_rows.append(
            {
                "group_id": gid,
                "scenario_key": job["scenario_key"],
                "scenario_label": job["scenario_label"],
                "rep": int(job["rep"]),
                "n_stays": int(job["n_stays"]),
                **ademp_v2_scenario_diagnostics(data),
                "selected_shape_solver_converged_fraction": float(
                    shape_solver_diagnostics["converged_fraction"]
                ),
            }
        )
        if args.progress_every > 0 and (
            job_number % args.progress_every == 0 or job_number == len(jobs)
        ):
            elapsed = time.time() - start
            print(
                f"ADEMP v2 completed {job_number}/{len(jobs)} datasets "
                f"({elapsed:.1f} seconds elapsed)",
                flush=True,
            )

    method_replicates = pd.DataFrame(method_rows)
    penalties = pd.DataFrame(penalty_rows)
    tuning_grid = pd.DataFrame(tuning_rows)
    calibrations = pd.DataFrame(calibration_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = summarize_mc_replicates(method_replicates)
    penalty_summary = summarize_penalty_selection(penalties)
    paired_comparisons = summarize_paired_loss_differences(method_replicates)
    failure_summary = summary.loc[
        :,
        [
            "scenario_key",
            "scenario_label",
            "method",
            "n_attempted_replicates",
            "n_effective_replicates",
            "failure_count",
            "failure_rate",
        ],
    ].copy()
    design_rows = []
    for key, scenario in scenarios.items():
        design_rows.append({"scenario_key": key, **dict(scenario)})
    design_frame_out = pd.DataFrame(design_rows)

    paths = {
        "method_replicates_csv": args.out_dir / "ademp_v2_method_replicates.csv",
        "summary_csv": args.out_dir / "ademp_v2_summary.csv",
        "penalty_replicates_csv": args.out_dir / "ademp_v2_penalty_replicates.csv",
        "penalty_summary_csv": args.out_dir / "ademp_v2_penalty_summary.csv",
        "paired_comparisons_csv": args.out_dir / "ademp_v2_paired_loss_comparisons.csv",
        "failure_summary_csv": args.out_dir / "ademp_v2_failure_summary.csv",
        "tuning_grid_csv": args.out_dir / "ademp_v2_tuning_grid.csv",
        "calibration_fits_csv": args.out_dir / "ademp_v2_affine_calibration_fits.csv",
        "scenario_diagnostics_csv": args.out_dir / "ademp_v2_scenario_diagnostics.csv",
        "design_csv": args.out_dir / "ademp_v2_design.csv",
        "summary_markdown": args.out_dir / "ademp_v2_summary.md",
        "key_conclusions_markdown": args.out_dir / "ademp_v2_key_conclusions.md",
        "summary_figure_pdf": args.out_dir / "ademp_v2_summary_figure.pdf",
        "summary_figure_svg": args.out_dir / "ademp_v2_summary_figure.svg",
        "summary_figure_png": args.out_dir / "ademp_v2_summary_figure.png",
        "summary_tex": args.out_dir / "ademp_v2_summary.tex",
        "results_json": args.out_dir / "ademp_v2_results.json",
    }
    method_replicates.to_csv(paths["method_replicates_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    penalties.to_csv(paths["penalty_replicates_csv"], index=False)
    penalty_summary.to_csv(paths["penalty_summary_csv"], index=False)
    paired_comparisons.to_csv(paths["paired_comparisons_csv"], index=False)
    failure_summary.to_csv(paths["failure_summary_csv"], index=False)
    tuning_grid.to_csv(paths["tuning_grid_csv"], index=False)
    calibrations.to_csv(paths["calibration_fits_csv"], index=False)
    diagnostics.to_csv(paths["scenario_diagnostics_csv"], index=False)
    design_frame_out.to_csv(paths["design_csv"], index=False)
    parameters = {
        "design_version": "ademp-v2",
        "n_rep": int(args.n_rep),
        "rep_start": int(args.rep_offset) + 1,
        "rep_end": int(args.rep_offset) + int(args.n_rep),
        "scenario_n_stays": {
            key: int(args.n_stays) if args.n_stays is not None else int(value["n_stays"])
            for key, value in scenarios.items()
        },
        "tau": float(args.tau),
        "seed": int(args.seed),
        "lambda_grid": [float(value) for value in args.lambda_grid],
        "shape_lambda_grid": [float(value) for value in args.shape_lambda_grid],
        "train_fraction": 0.60,
        "tuning_fraction": 0.20,
        "assessment_fraction": 0.20,
        "elapsed_seconds": float(time.time() - start),
        "independent_mc_unit": "one complete simulated dataset",
        "analysis_unit": "stay",
        "calibration_definition": "P(Y < q) <= tau <= P(Y <= q), averaged within stay then across stays",
    }
    write_ademp_v2_markdown(
        summary,
        diagnostics,
        penalty_summary,
        paths["summary_markdown"],
        parameters,
    )
    write_ademp_v2_key_conclusions(
        paired_comparisons,
        diagnostics,
        paths["key_conclusions_markdown"],
    )
    plot_ademp_v2_summary(
        summary,
        paired_comparisons,
        paths["summary_figure_pdf"],
        paths["summary_figure_svg"],
        paths["summary_figure_png"],
    )
    write_ademp_v2_summary_tex(
        summary,
        paired_comparisons,
        paths["summary_tex"],
        n_rep=int(args.n_rep),
    )
    sync_ademp_v2_manuscript_assets(paths, args.template_dir)
    results = {
        "parameters": parameters,
        "scenarios": {key: dict(value) for key, value in scenarios.items()},
        "summary": _json_records(summary),
        "penalty_selection": _json_records(penalty_summary),
        "paired_loss_comparisons": _json_records(paired_comparisons),
        "failure_summary": _json_records(failure_summary),
        "scenario_diagnostics": _json_records(diagnostics),
        "outputs": {key: str(path) for key, path in paths.items() if key != "results_json"},
    }
    paths["results_json"].write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote ADEMP-v2 simulation results to {paths['results_json']}")


def combine_ademp_v2_chunks(
    chunks_root: Path,
    out_dir: Path,
    expected_replicates: int | None = None,
    template_dir: Path | None = None,
) -> None:
    """Combine deterministic replicate chunks and recompute every MC summary."""

    chunk_dirs = sorted(
        path for path in chunks_root.iterdir() if path.is_dir() and path.name.startswith("chunk_")
    )
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk_* directories found under {chunks_root}")

    def combine_csv(filename: str) -> pd.DataFrame:
        paths = [directory / filename for directory in chunk_dirs]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing chunk products for {filename}: {missing}")
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    method_replicates = combine_csv("ademp_v2_method_replicates.csv")
    penalties = combine_csv("ademp_v2_penalty_replicates.csv")
    tuning_grid = combine_csv("ademp_v2_tuning_grid.csv")
    calibrations = combine_csv("ademp_v2_affine_calibration_fits.csv")
    diagnostics = combine_csv("ademp_v2_scenario_diagnostics.csv")
    design = combine_csv("ademp_v2_design.csv").drop_duplicates("scenario_key", keep="first")

    duplicate_method = method_replicates.duplicated(["scenario_key", "rep", "method"])
    duplicate_penalty = penalties.duplicated(["scenario_key", "rep", "method"])
    duplicate_calibration = calibrations.duplicated(["scenario_key", "rep"])
    duplicate_diagnostics = diagnostics.duplicated(["scenario_key", "rep"])
    if duplicate_method.any() or duplicate_penalty.any() or duplicate_calibration.any() or duplicate_diagnostics.any():
        raise RuntimeError("Chunk inputs contain duplicate scenario/replicate keys.")

    replicate_counts = (
        method_replicates.groupby("scenario_key", sort=False)["rep"].nunique().astype(int).to_dict()
    )
    if expected_replicates is not None:
        incomplete = {
            key: count for key, count in replicate_counts.items() if count != int(expected_replicates)
        }
        if incomplete:
            raise RuntimeError(
                f"Expected {expected_replicates} unique replicates per scenario; observed {incomplete}."
            )
    if len(set(replicate_counts.values())) != 1:
        raise RuntimeError(f"Scenarios have unequal replicate counts: {replicate_counts}")
    n_rep = int(next(iter(replicate_counts.values())))

    summary = summarize_mc_replicates(method_replicates)
    penalty_summary = summarize_penalty_selection(penalties)
    paired_comparisons = summarize_paired_loss_differences(method_replicates)
    failure_summary = summary.loc[
        :,
        [
            "scenario_key",
            "scenario_label",
            "method",
            "n_attempted_replicates",
            "n_effective_replicates",
            "failure_count",
            "failure_rate",
        ],
    ].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "method_replicates_csv": out_dir / "ademp_v2_method_replicates.csv",
        "summary_csv": out_dir / "ademp_v2_summary.csv",
        "penalty_replicates_csv": out_dir / "ademp_v2_penalty_replicates.csv",
        "penalty_summary_csv": out_dir / "ademp_v2_penalty_summary.csv",
        "paired_comparisons_csv": out_dir / "ademp_v2_paired_loss_comparisons.csv",
        "failure_summary_csv": out_dir / "ademp_v2_failure_summary.csv",
        "tuning_grid_csv": out_dir / "ademp_v2_tuning_grid.csv",
        "calibration_fits_csv": out_dir / "ademp_v2_affine_calibration_fits.csv",
        "scenario_diagnostics_csv": out_dir / "ademp_v2_scenario_diagnostics.csv",
        "design_csv": out_dir / "ademp_v2_design.csv",
        "summary_markdown": out_dir / "ademp_v2_summary.md",
        "key_conclusions_markdown": out_dir / "ademp_v2_key_conclusions.md",
        "summary_figure_pdf": out_dir / "ademp_v2_summary_figure.pdf",
        "summary_figure_svg": out_dir / "ademp_v2_summary_figure.svg",
        "summary_figure_png": out_dir / "ademp_v2_summary_figure.png",
        "summary_tex": out_dir / "ademp_v2_summary.tex",
        "results_json": out_dir / "ademp_v2_results.json",
    }
    method_replicates.to_csv(paths["method_replicates_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    penalties.to_csv(paths["penalty_replicates_csv"], index=False)
    penalty_summary.to_csv(paths["penalty_summary_csv"], index=False)
    paired_comparisons.to_csv(paths["paired_comparisons_csv"], index=False)
    failure_summary.to_csv(paths["failure_summary_csv"], index=False)
    tuning_grid.to_csv(paths["tuning_grid_csv"], index=False)
    calibrations.to_csv(paths["calibration_fits_csv"], index=False)
    diagnostics.to_csv(paths["scenario_diagnostics_csv"], index=False)
    design.to_csv(paths["design_csv"], index=False)

    elapsed_seconds = 0.0
    for directory in chunk_dirs:
        chunk_json = directory / "ademp_v2_results.json"
        payload = json.loads(chunk_json.read_text(encoding="utf-8"))
        elapsed_seconds += float(payload["parameters"]["elapsed_seconds"])
    scenario_keys = method_replicates["scenario_key"].drop_duplicates().tolist()
    parameters = {
        "design_version": "ademp-v2",
        "n_rep": n_rep,
        "rep_start": int(method_replicates["rep"].min()),
        "rep_end": int(method_replicates["rep"].max()),
        "replicates_per_scenario": replicate_counts,
        "scenario_n_stays": {
            key: int(
                method_replicates.loc[method_replicates["scenario_key"] == key, "n_stays"].iloc[0]
            )
            for key in scenario_keys
        },
        "tau": 0.10,
        "seed": 20260529,
        "lambda_grid": [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0],
        "shape_lambda_grid": [0.10, 0.30, 1.0, 3.0, 10.0],
        "train_fraction": 0.60,
        "tuning_fraction": 0.20,
        "assessment_fraction": 0.20,
        "elapsed_seconds": elapsed_seconds,
        "chunks": [str(directory) for directory in chunk_dirs],
        "independent_mc_unit": "one complete simulated dataset",
        "analysis_unit": "stay",
        "calibration_definition": "P(Y < q) <= tau <= P(Y <= q), averaged within stay then across stays",
    }
    write_ademp_v2_markdown(summary, diagnostics, penalty_summary, paths["summary_markdown"], parameters)
    write_ademp_v2_key_conclusions(
        paired_comparisons, diagnostics, paths["key_conclusions_markdown"]
    )
    plot_ademp_v2_summary(
        summary,
        paired_comparisons,
        paths["summary_figure_pdf"],
        paths["summary_figure_svg"],
        paths["summary_figure_png"],
    )
    write_ademp_v2_summary_tex(
        summary,
        paired_comparisons,
        paths["summary_tex"],
        n_rep=n_rep,
    )
    sync_ademp_v2_manuscript_assets(paths, template_dir)
    results = {
        "parameters": parameters,
        "scenarios": {key: ADEMP_V2_SCENARIOS[key] for key in scenario_keys},
        "summary": _json_records(summary),
        "penalty_selection": _json_records(penalty_summary),
        "paired_loss_comparisons": _json_records(paired_comparisons),
        "failure_summary": _json_records(failure_summary),
        "scenario_diagnostics": _json_records(diagnostics),
        "outputs": {key: str(path) for key, path in paths.items() if key != "results_json"},
    }
    paths["results_json"].write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Combined {len(chunk_dirs)} chunks with {n_rep} replicates per scenario into "
        f"{paths['results_json']}"
    )


def parse_lambda_grid(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    paper_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        choices=("legacy-v1", "ademp-v2"),
        default="legacy-v1",
        help="Versioned simulation contract; legacy-v1 remains the default for backward compatibility.",
    )
    parser.add_argument("--n-rep", type=int, default=None)
    parser.add_argument(
        "--rep-offset",
        type=int,
        default=0,
        help="Zero-based replicate offset for deterministic chunked ADEMP-v2 runs.",
    )
    parser.add_argument(
        "--n-stays",
        type=int,
        default=None,
        help="Override every scenario size. Omit in ADEMP v2 to retain its 240/600 sample-size axis.",
    )
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--lambda-grid", type=parse_lambda_grid, default=parse_lambda_grid("0,0.03,0.10,0.30,1,3,10"))
    parser.add_argument(
        "--shape-lambda-grid",
        type=parse_lambda_grid,
        default=parse_lambda_grid("0.10,0.30,1,3,10"),
    )
    parser.add_argument(
        "--scenarios",
        type=_comma_separated_strings,
        default=None,
        help="Comma-separated ADEMP-v2 scenario keys; omit to run the complete design.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--combine-chunks-dir",
        type=Path,
        default=None,
        help="Combine completed chunk_* ADEMP-v2 directories without rerunning simulation.",
    )
    parser.add_argument("--expected-replicates", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--r-helper", type=Path, default=script_dir / "fit_grouped_quantile_common.R")
    args = parser.parse_args()

    if not 0.0 < args.tau < 1.0:
        parser.error("--tau must lie strictly between zero and one.")
    if args.n_rep is not None and args.n_rep <= 0:
        parser.error("--n-rep must be positive.")
    if args.rep_offset < 0:
        parser.error("--rep-offset must be nonnegative.")
    if args.n_stays is not None and args.n_stays < 30:
        parser.error("--n-stays must be at least 30.")

    if args.combine_chunks_dir is not None:
        combined_out = (
            paper_root / "recovery_20260822" / "simulation_extended_v2"
            if args.out_dir is None
            else args.out_dir
        )
        combine_ademp_v2_chunks(
            args.combine_chunks_dir,
            combined_out,
            expected_replicates=args.expected_replicates,
            template_dir=args.template_dir,
        )
        return

    if args.design == "ademp-v2":
        args.n_rep = 100 if args.n_rep is None else int(args.n_rep)
        args.out_dir = (
            paper_root / "recovery_20260822" / "simulation_extended_v2"
            if args.out_dir is None
            else args.out_dir
        )
        args.work_dir = (
            paper_root / "recovery_20260822" / "simulation_work" / "ademp_v2"
            if args.work_dir is None
            else args.work_dir
        )
        selected_keys = list(ADEMP_V2_SCENARIOS) if args.scenarios is None else list(args.scenarios)
        unknown = [key for key in selected_keys if key not in ADEMP_V2_SCENARIOS]
        if unknown:
            parser.error(f"Unknown ADEMP-v2 scenarios: {', '.join(unknown)}")
        selected_scenarios = {key: ADEMP_V2_SCENARIOS[key] for key in selected_keys}
        run_ademp_v2(args, selected_scenarios)
        return

    args.n_rep = 100 if args.n_rep is None else int(args.n_rep)
    args.n_stays = 600 if args.n_stays is None else int(args.n_stays)
    args.out_dir = paper_root / "paper" / "simulation" if args.out_dir is None else args.out_dir
    args.work_dir = (
        paper_root / "code" / "split_window_simulation_work"
        if args.work_dir is None
        else args.work_dir
    )

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, object]] = []
    train_frames: List[pd.DataFrame] = []
    group_id = 0
    for scenario_key, scenario in SCENARIOS.items():
        for rep in range(args.n_rep):
            rng = np.random.default_rng(args.seed + 1009 * group_id)
            split_rng = np.random.default_rng(args.seed + 7919 * group_id + 17)
            data = simulate_dataset(args.n_stays, args.tau, scenario, rng)
            train_idx, tuning_idx, assessment_idx = split_cluster_indices(args.n_stays, split_rng)
            train_frames.append(training_design(data, train_idx, group_id))
            jobs.append(
                {
                    "group_id": group_id,
                    "scenario_key": scenario_key,
                    "scenario_label": str(scenario["label"]),
                    "rep": rep + 1,
                    "data": data,
                    "train_idx": train_idx,
                    "tuning_idx": tuning_idx,
                    "assessment_idx": assessment_idx,
                }
            )
            group_id += 1

    design_csv = args.work_dir / "split_window_simulation_training_design.csv"
    coef_csv = args.work_dir / "split_window_simulation_coefficients.csv"
    pd.concat(train_frames, ignore_index=True).to_csv(design_csv, index=False)
    coef_df = grouped_quantile_fit(design_csv, coef_csv, args.tau, args.r_helper)
    coef_wide = coef_df.pivot(index="group_id", columns="term", values="estimate")

    replicate_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    for job in jobs:
        gid = int(job["group_id"])
        data = job["data"]
        assert isinstance(data, SimulatedDataset)
        beta_hat = coef_wide.loc[gid, PREDICTOR_COLS].to_numpy(dtype=float)
        tuning = tune_lambda(data, job["tuning_idx"], beta_hat, args.tau, args.lambda_grid)
        selected_lambda = float(tuning["best"]["lambda_b"])
        pop_losses, _ = evaluate_split(data, job["assessment_idx"], beta_hat, args.tau, lambda_b=None)
        unpen_losses, unpen_b = evaluate_split(data, job["assessment_idx"], beta_hat, args.tau, lambda_b=0.0)
        tuned_losses, tuned_b = evaluate_split(data, job["assessment_idx"], beta_hat, args.tau, lambda_b=selected_lambda)
        true_offset = data.true_late_offset[np.asarray(job["assessment_idx"], dtype=int)]

        pop_loss = float(np.mean(pop_losses))
        unpen_loss = float(np.mean(unpen_losses))
        tuned_loss = float(np.mean(tuned_losses))
        replicate_rows.append(
            {
                "group_id": gid,
                "scenario_key": job["scenario_key"],
                "scenario_label": job["scenario_label"],
                "rep": int(job["rep"]),
                "selected_lambda": selected_lambda,
                "population_loss": pop_loss,
                "unpenalized_loss": unpen_loss,
                "tuned_loss": tuned_loss,
                "unpenalized_reduction_percent": float(100.0 * (pop_loss - unpen_loss) / pop_loss),
                "tuned_reduction_percent": float(100.0 * (pop_loss - tuned_loss) / pop_loss),
                "tuned_beats_population_fraction": float(np.mean(tuned_losses < pop_losses)),
                "unpenalized_b_true_late_offset_corr": corr_or_nan(unpen_b, true_offset),
                "tuned_b_true_late_offset_corr": corr_or_nan(tuned_b, true_offset),
            }
        )
        for row in tuning["grid"]:
            tuning_rows.append(
                {
                    "group_id": gid,
                    "scenario_key": job["scenario_key"],
                    "scenario_label": job["scenario_label"],
                    "rep": int(job["rep"]),
                    **row,
                }
            )

    replicates = pd.DataFrame(replicate_rows)
    tuning_grid = pd.DataFrame(tuning_rows)

    summary_rows: List[Dict[str, object]] = []
    for scenario_key, scenario in SCENARIOS.items():
        local = replicates.loc[replicates["scenario_key"] == scenario_key].copy()
        row: Dict[str, object] = {
            "scenario_key": scenario_key,
            "scenario_label": scenario["label"],
            "description": scenario["description"],
            "n_replicates": int(local.shape[0]),
            "lambda_median": float(local["selected_lambda"].median()),
            "lambda_q1": float(local["selected_lambda"].quantile(0.25)),
            "lambda_q3": float(local["selected_lambda"].quantile(0.75)),
        }
        for col in [
            "population_loss",
            "unpenalized_loss",
            "tuned_loss",
            "unpenalized_reduction_percent",
            "tuned_reduction_percent",
            "tuned_beats_population_fraction",
            "tuned_b_true_late_offset_corr",
        ]:
            values = local[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_mcse"] = float("nan")
            else:
                row[f"{col}_mean"] = float(np.mean(finite))
                row[f"{col}_mcse"] = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    replicate_csv = args.out_dir / "simulation_study_replicates.csv"
    tuning_csv = args.out_dir / "simulation_study_tuning_grid.csv"
    summary_csv = args.out_dir / "simulation_study_summary.csv"
    summary_tex = args.out_dir / "simulation_study_summary.tex"
    figure_pdf = args.out_dir / "simulation_study_plot.pdf"
    figure_png = args.out_dir / "simulation_study_plot.png"
    json_path = args.out_dir / "split_window_simulation_results.json"

    replicates.to_csv(replicate_csv, index=False)
    tuning_grid.to_csv(tuning_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_summary_tex(summary, summary_tex, args.n_rep, args.n_stays)
    plot_simulation(replicates, summary, figure_pdf, figure_png)

    template_targets = []
    if args.template_dir is not None:
        template_targets.append(args.template_dir)
        submission_dir = args.template_dir / "revised_quantile_adaptive_basis_mimic_submission_20260531"
        if submission_dir.exists():
            template_targets.append(submission_dir)
        for template_target in template_targets:
            (template_target / "tables").mkdir(parents=True, exist_ok=True)
            (template_target / "figures").mkdir(parents=True, exist_ok=True)
            (template_target / "tables" / summary_tex.name).write_text(
                summary_tex.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (template_target / "figures" / figure_pdf.name).write_bytes(figure_pdf.read_bytes())
            (template_target / "figures" / figure_png.name).write_bytes(figure_png.read_bytes())

    results = {
        "parameters": {
            "n_rep": args.n_rep,
            "n_stays": args.n_stays,
            "tau": args.tau,
            "seed": args.seed,
            "lambda_grid": [float(x) for x in args.lambda_grid],
            "elapsed_seconds": time.time() - start,
        },
        "scenarios": SCENARIOS,
        "summary": summary.to_dict("records"),
        "outputs": {
            "replicates_csv": str(replicate_csv),
            "tuning_csv": str(tuning_csv),
            "summary_csv": str(summary_csv),
            "summary_tex": str(summary_tex),
            "figure_pdf": str(figure_pdf),
            "figure_png": str(figure_png),
        },
    }
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote simulation results to {json_path}")
    for template_target in template_targets:
        print(f"Wrote manuscript table to {template_target / 'tables' / summary_tex.name}")
        print(f"Wrote manuscript figure to {template_target / 'figures' / figure_pdf.name}")


if __name__ == "__main__":
    main()
