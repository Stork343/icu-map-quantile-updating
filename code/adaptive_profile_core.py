from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import clarabel
from scipy import sparse
from scipy.optimize import linprog


TAU = 0.1
KAPPA = 1.0 / 16.0
LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
GAMMA_GRID = (0.0, 0.3, 3.0)
THETA_GRID = (-2.0, 0.0, 2.0)
QP_SETTINGS = {
    "tol_gap_abs": 1e-14, "tol_gap_rel": 1e-14, "tol_feas": 1e-14,
    "max_iter": 200, "verbose": False, "max_threads": 1,
    "iterative_refinement_reltol": 1e-14, "iterative_refinement_abstol": 1e-14,
}
POLISH_SETTINGS = {"residual_tolerance": 1e-5, "dual_margin": 1e-8, "lstsq_rcond": 1e-13}


@dataclass(frozen=True)
class ProfileStay:
    index_residual: np.ndarray
    index_time: np.ndarray
    later_y: np.ndarray
    later_time: np.ndarray
    later_population: np.ndarray
    index_hours: float = 12.0

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value, dtype=float) for value in (
            self.index_residual, self.index_time, self.later_y,
            self.later_time, self.later_population,
        ))
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("Stay arrays must be one dimensional")
        if arrays[0].size < 2 or arrays[1].shape != arrays[0].shape:
            raise ValueError("At least two matched index observations are required")
        if arrays[2].shape != arrays[3].shape or arrays[2].shape != arrays[4].shape:
            raise ValueError("Later observations, times, and population forecasts must match")
        if self.index_hours <= 0 or not np.isfinite(self.index_hours):
            raise ValueError("The landmark must be positive and finite")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Stay arrays must contain finite values")
        if np.any(np.diff(arrays[1]) < 0) or np.any(arrays[1] < 0) or np.any(arrays[1] > self.index_hours):
            raise ValueError("Index times must be ordered and no later than the landmark")
        if np.any(arrays[3] <= self.index_hours):
            raise ValueError("Later observations must follow the landmark")
        for name, value in zip(("index_residual", "index_time", "later_y", "later_time", "later_population"), arrays):
            object.__setattr__(self, name, value)
        normalized_time = (arrays[1] - self.index_hours) / self.index_hours
        design = np.column_stack([np.ones(arrays[0].size), normalized_time])
        covariance = design.T @ design / design.shape[0]
        middle = arrays[0].size // 2
        change = (empirical_quantile(arrays[0][middle:]) - empirical_quantile(arrays[0][:middle])) / 10.0
        object.__setattr__(self, "_index_design", design)
        object.__setattr__(self, "_regularized_inverse", np.linalg.inv(covariance + KAPPA * np.eye(2)))
        object.__setattr__(self, "_index_features", np.asarray([np.clip(change, -1.0, 1.0),
                                                              np.log(arrays[0].size), np.var(normalized_time)]))


def check_loss(residual: np.ndarray, tau: float) -> np.ndarray:
    if not 0 < tau < 1:
        raise ValueError("Quantile level must lie strictly between zero and one")
    residual = np.asarray(residual, dtype=float)
    return residual * (tau - (residual < 0))


def empirical_quantile(values: np.ndarray, tau: float = TAU) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not 0 < tau < 1 or not np.all(np.isfinite(values)):
        raise ValueError("An empirical quantile requires a finite nonempty vector and valid level")
    rank = int(np.ceil(values.size * tau)) - 1
    return float(np.partition(values, rank)[rank])


def index_features(stay: ProfileStay) -> np.ndarray:
    return stay._index_features.copy()


def horizon_geometry(horizon: float, index_hours: float = 12.0) -> np.ndarray:
    if horizon <= 0 or index_hours <= 0 or not np.isfinite(horizon + index_hours):
        raise ValueError("Forecast horizon and landmark must be positive and finite")
    ratio = horizon / index_hours
    return np.asarray([[1.0, ratio / 2.0], [ratio / 2.0, ratio * ratio / 3.0]])


def adaptive_candidate_specs() -> list[dict[str, object]]:
    specs = []
    for level_index, level in enumerate(LAMBDA_GRID):
        for gamma_index, gamma in enumerate(GAMMA_GRID):
            theta_values = (0.0,) if gamma == 0 else THETA_GRID
            for theta_index, theta in enumerate(theta_values):
                specs.append({"key": f"adaptive_{level_index}_{gamma_index}_{theta_index}",
                              "lambda_base": level, "gamma": gamma, "theta": theta})
    if len(specs) != 42 or len({row["key"] for row in specs}) != 42:
        raise RuntimeError("The prespecified adaptive grid must contain 42 distinct candidates")
    return specs


def penalty_matrix(stay: ProfileStay, horizon: float, spec: Mapping[str, object]) -> np.ndarray:
    level, gamma, theta = (float(spec[name]) for name in ("lambda_base", "gamma", "theta"))
    if level <= 0 or gamma < 0 or not np.all(np.isfinite([level, gamma, theta])):
        raise ValueError("Adaptive penalties require a positive base and nonnegative geometry weight")
    geometry = horizon_geometry(horizon, stay.index_hours)
    weight = gamma * np.exp(theta * stay._index_features[0])
    result = level * geometry + weight * geometry @ stay._regularized_inverse @ geometry
    return (result + result.T) / 2.0


def solve_spd_profile(residual: np.ndarray, design: np.ndarray, tau: float,
                      penalty: np.ndarray, diagnostic_path: Path | None = None) -> tuple[np.ndarray, dict[str, object]]:
    residual, design, penalty = (np.asarray(value, dtype=float) for value in (residual, design, penalty))
    if residual.ndim != 1 or residual.size == 0 or design.shape != (residual.size, 2) or penalty.shape != (2, 2):
        raise ValueError("SPD profile inputs have inconsistent dimensions")
    if not 0 < tau < 1 or not all(np.all(np.isfinite(value)) for value in (residual, design, penalty)):
        raise ValueError("SPD profiles require finite values and a valid quantile")
    if not np.allclose(penalty, penalty.T, rtol=0.0, atol=1e-13):
        raise ValueError("The penalty matrix must be symmetric")
    np.linalg.cholesky(penalty)
    size = residual.size
    identity = sparse.eye(size, format="csc")
    quadratic = sparse.block_diag([sparse.csc_matrix(2.0 * penalty),
                                    sparse.csc_matrix((2 * size, 2 * size))], format="csc")
    constraints = sparse.vstack([
        sparse.hstack([sparse.csc_matrix(design), identity, -identity]),
        sparse.hstack([sparse.csc_matrix((2 * size, 2)), -sparse.eye(2 * size)]),
    ], format="csc")
    bounds = np.r_[residual, np.zeros(2 * size)]
    settings = clarabel.DefaultSettings()
    for name, value in QP_SETTINGS.items():
        setattr(settings, name, value)
    solver = clarabel.DefaultSolver(quadratic, np.r_[np.zeros(2), np.full(size, tau), np.full(size, 1.0 - tau)],
                                    constraints, bounds,
                                    [clarabel.ZeroConeT(size), clarabel.NonnegativeConeT(2 * size)], settings)
    solution = solver.solve()
    if solution.x is None:
        raise RuntimeError(f"SPD dual solver did not solve the problem: {solution.status}")
    coefficients = np.asarray(solution.x[:2], dtype=float)
    dual_variables = np.clip(-np.asarray(solution.z[:size], dtype=float), tau - 1.0, tau)
    preliminary_residual = residual - design @ coefficients
    active = ((np.abs(preliminary_residual) <= POLISH_SETTINGS["residual_tolerance"])
              & (dual_variables > tau - 1.0 + POLISH_SETTINGS["dual_margin"])
              & (dual_variables < tau - POLISH_SETTINGS["dual_margin"]))
    dual_variables[~active] = np.where(dual_variables[~active] > tau - 0.5, tau, tau - 1.0)
    polish_iterations = 0
    for polish_iterations in range(2 * size + 10):
        if np.any(active):
            active_design = design[active]
            system = np.block([[2.0 * penalty, -active_design.T],
                               [active_design, np.zeros((int(np.sum(active)), int(np.sum(active))))]])
            right = np.r_[design.T @ dual_variables - 2.0 * penalty @ coefficients,
                           residual[active] - active_design @ coefficients]
            correction = np.linalg.lstsq(system, right, rcond=POLISH_SETTINGS["lstsq_rcond"])[0]
            active_indices = np.flatnonzero(active)
            updated_dual = dual_variables[active] + correction[2:]
            bound_violation = np.maximum(tau - 1.0 - updated_dual, updated_dual - tau)
            if np.max(bound_violation) > 1e-13:
                position = int(np.argmax(bound_violation))
                released = int(active_indices[position])
                active[released] = False
                dual_variables[released] = tau if updated_dual[position] > tau else tau - 1.0
                continue
            coefficients = coefficients + correction[:2]
            dual_variables[active] = np.clip(updated_dual, tau - 1.0, tau)
        else:
            coefficients = 0.5 * np.linalg.solve(penalty, design.T @ dual_variables)
        current_residual = residual - design @ coefficients
        inactive_violation = np.abs(dual_variables - np.clip(dual_variables + current_residual, tau - 1.0, tau))
        inactive_violation[active] = 0.0
        if np.max(inactive_violation) <= 1e-9:
            break
        active[int(np.argmax(inactive_violation))] = True
    score = design.T @ dual_variables
    fitted_residual = residual - design @ coefficients
    quadratic_value = float(coefficients @ penalty @ coefficients)
    primal = float(np.sum(check_loss(fitted_residual, tau)) + quadratic_value)
    dual = float(dual_variables @ residual - 0.25 * score @ np.linalg.solve(penalty, score))
    scale = max(1.0, abs(primal), abs(dual))
    gap = primal - dual
    kkt = float(np.max(np.abs(dual_variables - np.clip(dual_variables + fitted_residual, tau - 1.0, tau))))
    stationarity = float(np.max(np.abs(2.0 * penalty @ coefficients - score)))
    certified = (np.all(np.isfinite(coefficients)) and gap >= -1e-10 * scale
                 and gap <= max(1e-7, 1e-8 * scale) and kkt <= 5e-7 and stationarity <= 1e-10)
    diagnostics = {"success": bool(certified), "method": "clarabel_spd_primal_qp_active_set_certified",
                   "status": str(solution.status), "iterations": int(solution.iterations),
                   "polishing_status": 1, "active_set_size": int(np.sum(active)),
                   "polishing_iterations": polish_iterations + 1, "primal_objective": primal,
                   "dual_objective": dual, "duality_gap": gap,
                   "relative_duality_gap": max(gap, 0.0) / scale,
                   "projected_kkt_violation": kkt, "stationarity_violation": stationarity}
    if not certified:
        if diagnostic_path is not None:
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(diagnostic_path, residual=residual, design=design, penalty=penalty,
                     tau=tau, coefficients=coefficients, dual_variables=dual_variables)
        raise RuntimeError(f"Uncertified SPD profile solution: {diagnostics}")
    return coefficients, diagnostics


def fit_profile(stay: ProfileStay, horizon: float, spec: Mapping[str, object],
                tau: float = TAU, diagnostic_path: Path | None = None) -> tuple[np.ndarray, dict[str, object]]:
    return solve_spd_profile(stay.index_residual, stay._index_design, tau,
                             penalty_matrix(stay, horizon, spec), diagnostic_path)


def evaluate_profile(stay: ProfileStay, coefficients: np.ndarray, horizon: float,
                     tau: float = TAU, intercept: float = 0.0, slope: float = 1.0) -> dict[str, float | int]:
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("Forecast horizon must be positive and finite")
    selected = stay.later_time <= stay.index_hours + horizon
    if not np.any(selected):
        raise ValueError("A stay has no later observation in the specified horizon")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Profile coefficients must be a finite length two vector")
    base = (stay.later_population[selected] + coefficients[0]
            + coefficients[1] * (stay.later_time[selected] - stay.index_hours) / stay.index_hours)
    prediction = intercept + slope * base
    outcomes = stay.later_y[selected]
    return {"loss": float(np.mean(check_loss(outcomes - prediction, tau))),
            "p_lt": float(np.mean(outcomes < prediction)),
            "p_le": float(np.mean(outcomes <= prediction)), "n_later": int(outcomes.size)}


def fit_affine_calibration(y: np.ndarray, prediction: np.ndarray, observation_weight: np.ndarray,
                           tau: float = TAU) -> dict[str, float]:
    y, prediction, weight = (np.asarray(value, dtype=float) for value in (y, prediction, observation_weight))
    if y.ndim != 1 or y.size == 0 or prediction.shape != y.shape or weight.shape != y.shape:
        raise ValueError("Calibration arrays must be matched nonempty vectors")
    if not all(np.all(np.isfinite(value)) for value in (y, prediction, weight)) or np.any(weight <= 0):
        raise ValueError("Calibration inputs must be finite with positive weights")
    if not 0 < tau < 1:
        raise ValueError("Calibration quantile must lie strictly between zero and one")
    identity = sparse.eye(y.size, format="csc")
    equality = sparse.hstack([np.ones((y.size, 1)), prediction[:, None], identity, -identity], format="csc")
    objective = np.r_[0.0, 0.0, tau * weight, (1.0 - tau) * weight]
    solution = linprog(objective, A_eq=equality, b_eq=y,
                       bounds=[(None, None), (None, None)] + [(0.0, None)] * (2 * y.size),
                       method="highs", options={"dual_feasibility_tolerance": 1e-9,
                                                 "primal_feasibility_tolerance": 1e-9})
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(f"Affine calibration LP did not solve: {solution.message}")
    intercept, slope = (float(value) for value in solution.x[:2])
    loss = float(np.sum(weight * check_loss(y - intercept - slope * prediction, tau)))
    if not np.isclose(loss, float(solution.fun), rtol=1e-8, atol=1e-7):
        raise RuntimeError("Calibration LP objective does not match the fitted check loss")
    return {"intercept": intercept, "slope": slope, "weighted_loss": loss,
            "max_equality_residual": float(np.max(np.abs(equality @ solution.x - y)))}


def fit_stump_selector(training_features: np.ndarray, tuning_features: np.ndarray,
                       tuning_loss_matrix: np.ndarray, rule_names: list[str]) -> dict[str, object]:
    training_features, tuning_features, losses = (np.asarray(value, dtype=float) for value in (
        training_features, tuning_features, tuning_loss_matrix,
    ))
    if (training_features.ndim != 2 or training_features.shape[1] != 3 or training_features.shape[0] == 0
            or tuning_features.ndim != 2 or tuning_features.shape[1] != 3
            or losses.shape != (tuning_features.shape[0], len(rule_names)) or losses.shape[0] == 0):
        raise ValueError("Stump features and losses have inconsistent dimensions")
    if len(rule_names) == 0 or len(set(rule_names)) != len(rule_names):
        raise ValueError("Stump rule names must be nonempty and distinct")
    if not all(np.all(np.isfinite(value)) for value in (training_features, tuning_features, losses)):
        raise ValueError("Stump selection requires finite features and losses")
    global_rule = int(np.argmin(np.mean(losses, axis=0)))
    result = {"feature": -1, "threshold": None, "left_rule": global_rule, "right_rule": global_rule,
              "left_rule_name": rule_names[global_rule], "right_rule_name": rule_names[global_rule],
              "tuning_loss": float(np.mean(losses[:, global_rule]))}
    cutpoints = np.quantile(training_features, [0.25, 0.5, 0.75], axis=0)
    for feature in range(3):
        for threshold in np.unique(cutpoints[:, feature]):
            left = tuning_features[:, feature] <= threshold
            if not np.any(left) or np.all(left):
                continue
            left_rule = int(np.argmin(np.mean(losses[left], axis=0)))
            right_rule = int(np.argmin(np.mean(losses[~left], axis=0)))
            value = float((np.sum(losses[left, left_rule]) + np.sum(losses[~left, right_rule])) / losses.shape[0])
            if value < float(result["tuning_loss"]):
                result = {"feature": feature, "threshold": float(threshold),
                          "left_rule": left_rule, "right_rule": right_rule,
                          "left_rule_name": rule_names[left_rule], "right_rule_name": rule_names[right_rule],
                          "tuning_loss": value}
    return {**result, "training_quartile_cutpoints": cutpoints.tolist(),
            "feature_names": ["rank_half_q10_change", "log_index_count", "normalized_time_variance"],
            "rule_names": list(rule_names), "tie_order": "constant_then_feature_then_ascending_cutpoint_then_rule_order"}


def apply_stump_selector(selector: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != 3 or not np.all(np.isfinite(features)):
        raise ValueError("Stump prediction requires three finite features per stay")
    feature = int(selector["feature"])
    if feature == -1:
        return np.full(features.shape[0], int(selector["left_rule"]), dtype=np.int64)
    if feature not in (0, 1, 2):
        raise ValueError("Unknown stump feature")
    return np.where(features[:, feature] <= float(selector["threshold"]),
                    int(selector["left_rule"]), int(selector["right_rule"])).astype(np.int64)
