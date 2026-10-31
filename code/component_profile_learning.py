from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize
from scipy.special import expit

from adaptive_profile_core import ProfileStay, empirical_quantile


PENALTY_LOW = np.asarray([0.03, 0.03])
PENALTY_HIGH = np.asarray([10.0, 100.0])
LOG_LOW = np.log(PENALTY_LOW)
LOG_RANGE = np.log(PENALTY_HIGH) - LOG_LOW
PARAMETER_BOUNDS = [(-12.0, 12.0)] * 3 + [(-4.0, 4.0)] * 3
PARAMETER_BOUNDS = PARAMETER_BOUNDS * 2


@dataclass(frozen=True)
class ProfileBatch:
    index_residual: np.ndarray
    index_time: np.ndarray
    index_indptr: np.ndarray
    later_residual: np.ndarray
    later_time: np.ndarray
    later_population: np.ndarray
    later_indptr: np.ndarray
    features: np.ndarray
    horizons: np.ndarray
    weights: np.ndarray

    @property
    def n_records(self) -> int:
        return len(self.horizons)


def make_profile_batch(stays: list[ProfileStay], horizons: np.ndarray | float,
                       weights: np.ndarray | None = None) -> ProfileBatch:
    count = len(stays)
    horizon = np.broadcast_to(np.asarray(horizons, dtype=float), (count,)).copy()
    if count == 0 or not np.isfinite(horizon).all() or np.any((horizon < 4) | (horizon > 12)):
        raise ValueError("Batch horizons must lie in [4,12] for nonempty records")
    weight = np.ones(count) / count if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != (count,) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("Batch weights must be finite, positive, and matched to records")
    if not np.isclose(weight.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Batch weights must sum to one")
    indices, times, later, later_times, populations, features = [], [], [], [], [], []
    for stay, duration in zip(stays, horizon):
        selected = stay.later_time <= stay.index_hours + duration
        if not np.any(selected):
            raise ValueError("Every batch record requires an eligible later observation")
        u = (stay.index_time - stay.index_hours) / stay.index_hours
        h = np.asarray([max((8.0 - duration) / 4.0, 0.0),
                        1.0 - abs(duration - 8.0) / 4.0,
                        max((duration - 8.0) / 4.0, 0.0)])
        middle = len(u) // 2
        z = np.clip((empirical_quantile(stay.index_residual[middle:])
                     - empirical_quantile(stay.index_residual[:middle])) / 10.0, -1.0, 1.0)
        features.append(np.r_[h, np.clip(np.log(len(u) / 12.0) / np.log(4.0), -1.0, 1.0),
                              np.clip(12.0 * np.var(u) - 1.0, -1.0, 1.0), z])
        indices.append(stay.index_residual)
        times.append(u)
        later.append(stay.later_y[selected] - stay.later_population[selected])
        later_times.append((stay.later_time[selected] - stay.index_hours) / stay.index_hours)
        populations.append(stay.later_population[selected])
    return ProfileBatch(np.concatenate(indices), np.concatenate(times),
        np.r_[0, np.cumsum([len(value) for value in indices])].astype(np.int64),
        np.concatenate(later), np.concatenate(later_times), np.concatenate(populations),
        np.r_[0, np.cumsum([len(value) for value in later])].astype(np.int64),
        np.asarray(features), horizon, weight.copy())


def subset_profile_batch(batch: ProfileBatch, selected: np.ndarray,
                         weights: np.ndarray | None = None) -> ProfileBatch:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or np.any(selected < 0) or np.any(selected >= batch.n_records):
        raise ValueError("Batch subset indices must be nonempty and valid")
    index_parts = [np.arange(batch.index_indptr[i], batch.index_indptr[i + 1]) for i in selected]
    later_parts = [np.arange(batch.later_indptr[i], batch.later_indptr[i + 1]) for i in selected]
    early, late = np.concatenate(index_parts), np.concatenate(later_parts)
    subset_weights = batch.weights[selected] / batch.weights[selected].sum() if weights is None else np.asarray(weights, dtype=float)
    if subset_weights.shape != (len(selected),) or np.any(subset_weights <= 0) or not np.isclose(subset_weights.sum(), 1.0, atol=1e-10):
        raise ValueError("Subset weights must be positive and sum to one")
    return ProfileBatch(batch.index_residual[early], batch.index_time[early],
        np.r_[0, np.cumsum([len(value) for value in index_parts])].astype(np.int64),
        batch.later_residual[late], batch.later_time[late], batch.later_population[late],
        np.r_[0, np.cumsum([len(value) for value in later_parts])].astype(np.int64),
        batch.features[selected], batch.horizons[selected], subset_weights)


def penalty_values(params: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params, features = np.asarray(params, dtype=float), np.asarray(features, dtype=float)
    if params.shape != (2, 6) or features.ndim != 2 or features.shape[1] != 6:
        raise ValueError("Component parameters and features require shapes (2,6) and (n,6)")
    if not np.isfinite(params).all() or not np.isfinite(features).all() or np.max(np.abs(features)) > 1 + 1e-12:
        raise ValueError("Component parameters and bounded features must be finite")
    probability = expit(features @ params.T)
    penalties = np.exp(LOG_LOW + LOG_RANGE * probability)
    return penalties, LOG_RANGE * probability * (1.0 - probability)


@njit(cache=True)
def _certificate(residual, time_u, dual, level, slope, lambda_level, lambda_slope, tau):
    primal, dual_linear, projected = 0.0, 0.0, 0.0
    score_level, score_slope = 0.0, 0.0
    for j in range(len(residual)):
        value = residual[j] - level - slope * time_u[j]
        primal += value * (tau - (value < 0.0))
        dual_linear += dual[j] * residual[j]
        score_level += dual[j]
        score_slope += dual[j] * time_u[j]
        projected = max(projected, abs(dual[j] - min(tau, max(tau - 1.0, dual[j] + value))))
    quadratic = lambda_level * level * level + lambda_slope * slope * slope
    primal += quadratic
    dual_value = dual_linear - .25 * (score_level * score_level / lambda_level
                                     + score_slope * score_slope / lambda_slope)
    stationarity = max(abs(2 * lambda_level * level - score_level),
                       abs(2 * lambda_slope * slope - score_slope))
    scale = max(1.0, abs(primal), abs(dual_value))
    gap = primal - dual_value
    success = (gap >= -1e-10 * scale and gap <= max(1e-7, 1e-8 * scale)
               and projected <= 5e-7 and stationarity <= 1e-10)
    return success, primal, gap, projected, stationarity


@njit(cache=True)
def _pair_step(g1, g2, h11, h12, h22, low1, high1, low2, high2):
    best1, best2, improvement = 0.0, 0.0, 0.0
    determinant = h11 * h22 - h12 * h12
    for candidate in range(5):
        if candidate == 0:
            if determinant <= 1e-14 * h11 * h22:
                continue
            d1 = (h22 * g1 - h12 * g2) / determinant
            d2 = (h11 * g2 - h12 * g1) / determinant
            if d1 < low1 or d1 > high1 or d2 < low2 or d2 > high2:
                continue
        elif candidate <= 2:
            d1 = low1 if candidate == 1 else high1
            d2 = min(high2, max(low2, (g2 - h12 * d1) / h22))
        else:
            d2 = low2 if candidate == 3 else high2
            d1 = min(high1, max(low1, (g1 - h12 * d2) / h11))
        gain = g1 * d1 + g2 * d2 - .5 * (h11 * d1 * d1 + 2.0 * h12 * d1 * d2 + h22 * d2 * d2)
        if gain > improvement:
            best1, best2, improvement = d1, d2, gain
    return best1, best2


@njit(cache=True)
def _polish_dual(residual, time_u, dual, lambda_level, lambda_slope, tau):
    size = len(residual)
    for iteration in range(2000):
        level = np.sum(dual) / (2.0 * lambda_level)
        slope = np.dot(dual, time_u) / (2.0 * lambda_slope)
        maximum, pivot = 0.0, 0
        for j in range(size):
            gradient = residual[j] - level - slope * time_u[j]
            violation = abs(dual[j] - min(tau, max(tau - 1.0, dual[j] + gradient)))
            if violation > maximum:
                maximum, pivot = violation, j
        if maximum <= 1e-9:
            return iteration + 1
        free = np.flatnonzero((dual > tau - 1.0 + 1e-9) & (dual < tau - 1e-9))
        if len(free) >= 2:
            design = np.empty((len(free), 2))
            design[:, 0] = 1.0
            design[:, 1] = time_u[free]
            gram = design.T @ design
            eigenvalues, vectors = np.linalg.eigh(gram)
            inverse = np.zeros((2, 2))
            for k in range(2):
                if eigenvalues[k] > max(1e-13, eigenvalues[-1] * 1e-13):
                    inverse += np.outer(vectors[:, k], vectors[:, k]) / eigenvalues[k]
            # 对偶Hessian的零空间方向保留profile值并提高线性目标。
            local = residual[free]
            direction = local - design @ (inverse @ (design.T @ local))
            if np.max(np.abs(direction)) > 1e-9:
                step = np.inf
                for k, j in enumerate(free):
                    if direction[k] > 1e-15:
                        step = min(step, (tau - dual[j]) / direction[k])
                    elif direction[k] < -1e-15:
                        step = min(step, (tau - 1.0 - dual[j]) / direction[k])
                if not np.isfinite(step):
                    raise RuntimeError("Unbounded dual null-space step")
                for k, j in enumerate(free):
                    dual[j] = min(tau, max(tau - 1.0, dual[j] + step * direction[k]))
                continue
        # 精确求解二维盒约束子问题，处理接近平行的活跃观测。
        for j in range(size):
            if j == pivot:
                continue
            g1 = residual[pivot] - level - slope * time_u[pivot]
            g2 = residual[j] - level - slope * time_u[j]
            h11 = .5 / lambda_level + .5 * time_u[pivot] ** 2 / lambda_slope
            h12 = .5 / lambda_level + .5 * time_u[pivot] * time_u[j] / lambda_slope
            h22 = .5 / lambda_level + .5 * time_u[j] ** 2 / lambda_slope
            d1, d2 = _pair_step(g1, g2, h11, h12, h22,
                tau - 1.0 - dual[pivot], tau - dual[pivot], tau - 1.0 - dual[j], tau - dual[j])
            dual[pivot] = min(tau, max(tau - 1.0, dual[pivot] + d1))
            dual[j] = min(tau, max(tau - 1.0, dual[j] + d2))
            level += .5 * (d1 + d2) / lambda_level
            slope += .5 * (d1 * time_u[pivot] + d2 * time_u[j]) / lambda_slope
    return 2000


@njit(cache=True)
def _solve_record(residual, time_u, dual, lambda_level, lambda_slope, tau, max_sweeps):
    level = np.sum(dual) / (2.0 * lambda_level)
    slope = np.dot(dual, time_u) / (2.0 * lambda_slope)
    sweeps = 0
    for sweeps in range(max_sweeps):
        for j in range(len(residual)):
            error = residual[j] - level - slope * time_u[j]
            curvature = .5 / lambda_level + .5 * time_u[j] * time_u[j] / lambda_slope
            updated = min(tau, max(tau - 1.0, dual[j] + error / curvature))
            change = updated - dual[j]
            dual[j] = updated
            level += .5 * change / lambda_level
            slope += .5 * change * time_u[j] / lambda_slope
        if sweeps % 5 == 4:
            level = np.sum(dual) / (2.0 * lambda_level)
            slope = np.dot(dual, time_u) / (2.0 * lambda_slope)
            maximum = 0.0
            for j in range(len(residual)):
                error = residual[j] - level - slope * time_u[j]
                maximum = max(maximum, abs(dual[j] - min(tau, max(tau - 1.0, dual[j] + error))))
            if maximum <= 1e-10:
                break
    level = np.sum(dual) / (2.0 * lambda_level)
    slope = np.dot(dual, time_u) / (2.0 * lambda_slope)
    success, primal, gap, projected, stationarity = _certificate(
        residual, time_u, dual, level, slope, lambda_level, lambda_slope, tau)
    if not success or projected > 1e-9:
        _polish_dual(residual, time_u, dual, lambda_level, lambda_slope, tau)
        level = np.sum(dual) / (2.0 * lambda_level)
        slope = np.dot(dual, time_u) / (2.0 * lambda_slope)
    success, primal, gap, projected, stationarity = _certificate(
        residual, time_u, dual, level, slope, lambda_level, lambda_slope, tau)
    # 活跃观测的张成空间决定局部超梯度。
    rank, active_time = 0, 0.0
    for j in range(len(residual)):
        if tau - 1.0 + 1e-8 < dual[j] < tau - 1e-8:
            if rank == 0:
                rank, active_time = 1, time_u[j]
            elif abs(time_u[j] - active_time) > 1e-10:
                rank = 2
                break
    jacobian = np.zeros((2, 2))
    if rank == 0:
        jacobian[0, 0] = -level
        jacobian[1, 1] = -slope
    elif rank == 1:
        vector0, vector1 = -active_time, 1.0
        denominator = lambda_level * vector0 * vector0 + lambda_slope
        jacobian[0, 0] = -vector0 * vector0 * level * lambda_level / denominator
        jacobian[1, 0] = -vector1 * vector0 * level * lambda_level / denominator
        jacobian[0, 1] = -vector0 * vector1 * slope * lambda_slope / denominator
        jacobian[1, 1] = -vector1 * vector1 * slope * lambda_slope / denominator
    return level, slope, jacobian, success, primal, gap, projected, stationarity, sweeps + 1


@njit(cache=True, parallel=True)
def _solve_batch(index_residual, index_time, index_indptr, penalties, tau, dual, max_sweeps):
    n = len(index_indptr) - 1
    coefficients = np.empty((n, 2))
    jacobians = np.empty((n, 2, 2))
    certificates = np.empty((n, 6))
    for i in prange(n):
        first, end = index_indptr[i], index_indptr[i + 1]
        result = _solve_record(index_residual[first:end], index_time[first:end], dual[first:end],
                               penalties[i, 0], penalties[i, 1], tau, max_sweeps)
        coefficients[i, 0], coefficients[i, 1] = result[0], result[1]
        jacobians[i] = result[2]
        certificates[i, 0] = 1.0 if result[3] else 0.0
        certificates[i, 1] = result[4]
        certificates[i, 2] = result[5]
        certificates[i, 3] = result[6]
        certificates[i, 4] = result[7]
        certificates[i, 5] = result[8]
    return coefficients, jacobians, certificates


@njit(cache=True)
def _score_batch(later_residual, later_time, later_indptr, coefficients, tau):
    n = len(later_indptr) - 1
    metrics, gradients = np.zeros((n, 3)), np.zeros((n, 2))
    prediction_correction = np.empty(len(later_residual))
    for i in range(n):
        first, end = later_indptr[i], later_indptr[i + 1]
        for j in range(first, end):
            prediction_correction[j] = coefficients[i, 0] + coefficients[i, 1] * later_time[j]
            residual = later_residual[j] - prediction_correction[j]
            score = tau - (residual < 0.0)
            if residual == 0.0:
                score = tau - .5
            metrics[i, 0] += residual * (tau - (residual < 0.0))
            metrics[i, 1] += residual < 0.0
            metrics[i, 2] += residual <= 0.0
            gradients[i, 0] -= score
            gradients[i, 1] -= score * later_time[j]
        metrics[i] /= end - first
        gradients[i] /= end - first
    return metrics, gradients, prediction_correction


def solve_component_batch(batch: ProfileBatch, penalties: np.ndarray, tau: float = .1,
                           dual: np.ndarray | None = None, max_sweeps: int = 20000,
                           diagnostic_directory: Path | None = None) -> dict:
    penalties = np.asarray(penalties, dtype=float)
    if penalties.shape != (batch.n_records, 2) or not np.isfinite(penalties).all() or np.any(penalties <= 0):
        raise ValueError("Each record requires two finite positive penalties")
    values = np.zeros_like(batch.index_residual) if dual is None else np.asarray(dual, dtype=float).copy()
    if values.shape != batch.index_residual.shape or not np.isfinite(values).all():
        raise ValueError("Warm dual values must match index observations")
    if not 0 < tau < 1 or np.any((values < tau - 1) | (values > tau)):
        raise ValueError("Dual values must satisfy the quantile box constraints")
    coefficients, jacobians, certificates = _solve_batch(batch.index_residual, batch.index_time,
        batch.index_indptr, penalties, tau, values, max_sweeps)
    if not np.isfinite(coefficients).all() or not np.all(certificates[:, 0] == 1):
        failed = np.flatnonzero(certificates[:, 0] != 1)
        if diagnostic_directory is not None:
            diagnostic_directory = Path(diagnostic_directory)
            diagnostic_directory.mkdir(parents=True, exist_ok=True)
            for i in failed:
                first, end = batch.index_indptr[i], batch.index_indptr[i + 1]
                np.savez(diagnostic_directory / f"record_{i}.npz", residual=batch.index_residual[first:end],
                         time_u=batch.index_time[first:end], dual=values[first:end], penalties=penalties[i],
                         coefficients=coefficients[i], certificate=certificates[i], tau=tau)
        raise RuntimeError(f"Uncertified component profiles at records {failed[:10].tolist()}, certificates={certificates[failed[:10]].tolist()}")
    metrics, score_gradients, correction = _score_batch(batch.later_residual, batch.later_time,
        batch.later_indptr, coefficients, tau)
    return {"coefficients": coefficients, "log_penalty_jacobians": jacobians,
            "penalties": penalties, "dual": values, "loss": metrics[:, 0], "p_lt": metrics[:, 1],
            "p_le": metrics[:, 2], "score_gradients": score_gradients,
            "flat_predictions": batch.later_population + correction,
            "diagnostics": {"profiles": batch.n_records, "all_certified": True,
                "max_duality_gap": float(certificates[:, 2].max()),
                "max_relative_duality_gap": float(np.max(np.maximum(certificates[:, 2], 0.0)
                    / np.maximum(1.0, np.abs(certificates[:, 1])))),
                "minimum_relative_duality_gap": float(np.min(certificates[:, 2]
                    / np.maximum(1.0, np.abs(certificates[:, 1])))),
                "max_projected_kkt_violation": float(certificates[:, 3].max()),
                "max_stationarity_violation": float(certificates[:, 4].max()),
                "max_sweeps": int(certificates[:, 5].max()),
                "mean_sweeps": float(certificates[:, 5].mean())}}


def evaluate_component_map(batch: ProfileBatch, params: np.ndarray, tau: float = .1,
                             dual: np.ndarray | None = None,
                             diagnostic_directory: Path | None = None) -> dict:
    penalties, link_jacobian = penalty_values(params, batch.features)
    result = solve_component_batch(batch, penalties, tau, dual, diagnostic_directory=diagnostic_directory)
    log_gradient = np.einsum("ni,nij->nj", result["score_gradients"], result["log_penalty_jacobians"])
    gradient = (batch.weights[:, None] * log_gradient * link_jacobian).T @ batch.features
    result["objective"] = float(batch.weights @ result["loss"])
    result["gradient"] = gradient
    return result


def evaluate_component_penalties(batch: ProfileBatch, penalties: np.ndarray,
                                 tau: float = .1, dual: np.ndarray | None = None,
                                 diagnostic_directory: Path | None = None) -> dict:
    values = np.asarray(penalties, dtype=float)
    if values.shape == (2,):
        values = np.tile(values, (batch.n_records, 1))
    result = solve_component_batch(batch, values, tau, dual, diagnostic_directory=diagnostic_directory)
    result["objective"] = float(batch.weights @ result["loss"])
    return result


def initial_parameters(initial_penalties: np.ndarray) -> np.ndarray:
    values = np.asarray(initial_penalties, dtype=float)
    if values.shape == (2,):
        values = np.tile(values, (3, 1))
    if values.shape != (3, 2) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Starting penalties must have shape (2,) or (3,2)")
    probability = np.clip((np.log(values) - LOG_LOW) / LOG_RANGE, 1e-4, 1 - 1e-4)
    params = np.zeros((2, 6))
    params[:, :3] = np.log(probability / (1.0 - probability)).T
    return params


def fit_component_map(batch: ProfileBatch, initial_penalties: np.ndarray = np.asarray([.3, .3]),
                      ridge_eta: float = 0.0, max_iter: int = 100,
                      deterministic_starts: int = 2, tau: float = .1,
                      diagnostic_directory: Path | None = None,
                      history_dependent: bool = True) -> dict:
    if ridge_eta < 0 or max_iter < 1 or deterministic_starts not in (1, 2, 3):
        raise ValueError("Invalid outer learning settings")
    if not isinstance(history_dependent, (bool, np.bool_)):
        raise ValueError("History dependence must be explicitly enabled or disabled")
    bounds = PARAMETER_BOUNDS if history_dependent else ([(-12.0, 12.0)] * 3 + [(0.0, 0.0)] * 3) * 2
    starts = [initial_parameters(initial_penalties)]
    for pair in ([.1, .3], [1., 3.])[:deterministic_starts - 1]:
        starts.append(initial_parameters(np.asarray(pair)))
    records, best, total_evaluations = [], None, 0
    all_certificates = {"profiles": 0, "all_certified": True, "max_duality_gap": 0.0,
                        "max_relative_duality_gap": 0.0, "minimum_relative_duality_gap": 0.0,
                        "max_projected_kkt_violation": 0.0, "max_stationarity_violation": 0.0,
                        "max_sweeps": 0}
    started = time.perf_counter()
    for start_index, initial in enumerate(starts):
        warm = np.zeros_like(batch.index_residual)
        best_returned = None

        def objective(flat):
            nonlocal warm, best_returned, total_evaluations
            params = flat.reshape(2, 6)
            if not history_dependent and np.any(params[:, 3:] != 0):
                raise RuntimeError("A constant component fit has a nonzero history coefficient")
            result = evaluate_component_map(batch, params, tau, warm, diagnostic_directory)
            warm = result["dual"]
            all_certificates["profiles"] += result["diagnostics"]["profiles"]
            for name in ("max_duality_gap", "max_relative_duality_gap", "max_projected_kkt_violation",
                         "max_stationarity_violation", "max_sweeps"):
                all_certificates[name] = max(all_certificates[name], result["diagnostics"][name])
            all_certificates["minimum_relative_duality_gap"] = min(
                all_certificates["minimum_relative_duality_gap"], result["diagnostics"]["minimum_relative_duality_gap"])
            value = result["objective"] + ridge_eta * float(np.sum(params[:, 3:] ** 2))
            gradient = result["gradient"].copy()
            gradient[:, 3:] += 2.0 * ridge_eta * params[:, 3:]
            if not history_dependent:
                gradient[:, 3:] = 0.0
            total_evaluations += 1
            if best_returned is None or value < best_returned["objective"]:
                best_returned = {"params": params.copy(), "objective": value,
                    "empirical_loss": result["objective"], "gradient": gradient.copy(),
                    "diagnostics": result["diagnostics"]}
            return value, gradient.ravel()

        fitted = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                          bounds=bounds,
                          options={"maxiter": max_iter, "maxls": 30, "ftol": 1e-11, "gtol": 1e-7})
        record = {"start": start_index, "optimizer_success": bool(fitted.success),
            "optimizer_status": int(fitted.status), "optimizer_message": str(fitted.message),
            "iterations": int(fitted.nit), "evaluations": int(fitted.nfev),
            "best_objective": best_returned["objective"],
            "best_empirical_loss": best_returned["empirical_loss"],
            "best_gradient_norm": float(np.max(np.abs(best_returned["gradient"])))}
        records.append(record)
        if best is None or best_returned["objective"] < best["objective"]:
            best = best_returned
            best["optimizer_success"] = bool(fitted.success)
    lower = np.asarray([bound[0] for bound in bounds]).reshape(2, 6)
    upper = np.asarray([bound[1] for bound in bounds]).reshape(2, 6)
    projected_gradient = best["params"] - np.clip(best["params"] - best["gradient"], lower, upper)
    return {**best, "ridge_eta": float(ridge_eta), "starts": records,
            "gradient_norm": float(np.max(np.abs(best["gradient"]))),
            "projected_gradient_norm": float(np.max(np.abs(projected_gradient))),
            "evaluations": total_evaluations, "seconds": time.perf_counter() - started,
            "history_dependent": bool(history_dependent),
            "active_parameters": 12 if history_dependent else 6,
            "global_optimum_certified": False, "profile_certificates_passed": True,
            "all_evaluation_diagnostics": all_certificates}
