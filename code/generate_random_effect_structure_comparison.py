import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

import split_window_data as data_utils
from run_split_window_mixed_effects_analysis import (
    apply_training_age_standardization,
    se_mean,
    split_cluster_indices,
)
from split_window_analysis_core import check_loss, design_frame, profiled_intercept
from split_window_data import _safe_read_frame, build_dataset_from_cache


X_PREDICTORS = ["x_intercept", "age_z", "male", "emergency_or_urgent"]


DUAL_GAP_ATOL = 1e-7
DUAL_GAP_RTOL = 1e-8
DUAL_KKT_TOL = 5e-7
GLOBAL_SOLVER_METHOD = "fenchel_dual_box_qp_certified"


def strict_split_indices(
    time_hours: np.ndarray,
    index_hours: float,
    min_index_obs: int = 4,
    min_late_obs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    time_hours = np.asarray(time_hours, dtype=float)
    index_idx = np.flatnonzero(time_hours <= float(index_hours))
    late_idx = np.flatnonzero(time_hours > float(index_hours))
    if index_idx.size < min_index_obs or late_idx.size < min_late_obs:
        return np.array([], dtype=int), np.array([], dtype=int)
    return index_idx, late_idx


def contiguous_stay_slices(design: pd.DataFrame) -> List[Tuple[int, int]]:
    stay_index = design["stay_index"].to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(stay_index)) + 1]
    ends = np.r_[starts[1:], stay_index.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def one_dimensional_penalized_update(
    residual: np.ndarray,
    covariate: np.ndarray,
    tau: float,
    lam: float,
) -> float:
    residual = np.asarray(residual, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    keep = np.abs(covariate) > 1e-12
    if not np.any(keep):
        return 0.0
    r = residual[keep]
    x = covariate[keep]
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if lam == 0.0:
        knots = np.unique((r / x)[np.isfinite(r / x)])
        if knots.size == 0:
            return 0.0
        objectives = np.asarray(
            [np.sum(check_loss(r - x * candidate, tau)) for candidate in knots],
            dtype=float,
        )
        minimum = float(np.min(objectives))
        tolerance = 1e-12 * max(1.0, abs(minimum))
        return float(knots[np.flatnonzero(objectives <= minimum + tolerance)[0]])

    def subgradient(u: float) -> float:
        score = tau - (r - x * u < 0.0).astype(float)
        return float(-np.sum(x * score) + 2.0 * lam * u)

    knots = r / x
    finite_knots = knots[np.isfinite(knots)]
    scale = float(np.max(np.abs(finite_knots))) if finite_knots.size else 1.0
    bound = max(100.0, 2.0 * scale + 100.0)
    lo = -bound
    hi = bound
    while subgradient(lo) > 0.0:
        lo *= 2.0
    while subgradient(hi) < 0.0:
        hi *= 2.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if subgradient(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def penalized_vector_update_dual(
    residual: np.ndarray,
    design: np.ndarray,
    tau: float,
    penalties: np.ndarray,
    *,
    gap_atol: float = DUAL_GAP_ATOL,
    gap_rtol: float = DUAL_GAP_RTOL,
    kkt_tolerance: float = DUAL_KKT_TOL,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Solve a strictly penalized quantile update through its Fenchel dual.

    The primal objective is

        sum_i rho_tau(r_i - x_i' b) + sum_j lambda_j b_j^2.

    With all penalties positive, its box-constrained smooth dual is

        max_a a' r - 1/4 (X' a)' Lambda^{-1} (X' a),
        tau - 1 <= a_i <= tau,

    and the primal solution is b = 1/2 Lambda^{-1} X' a.  The returned
    primal--dual gap is an a posteriori certificate of global optimality up
    to the declared numerical tolerances.
    """

    residual = np.asarray(residual, dtype=float)
    design = np.asarray(design, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    tau = float(tau)
    if residual.ndim != 1:
        raise ValueError("residual must be one-dimensional.")
    if design.ndim != 2 or design.shape[0] != residual.size:
        raise ValueError("design must have one row per residual.")
    if penalties.shape != (design.shape[1],):
        raise ValueError("penalties must contain one value per design column.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between zero and one.")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(design)):
        raise ValueError("residual and design must be finite.")
    if not np.all(np.isfinite(penalties)) or np.any(penalties <= 0.0):
        raise ValueError("Fenchel-dual vector updates require strictly positive penalties.")
    if residual.size == 0:
        raise ValueError("At least one residual is required.")

    inverse_penalties = 1.0 / penalties

    def negative_dual_and_gradient(a: np.ndarray) -> Tuple[float, np.ndarray]:
        xta = design.T @ a
        scaled_xta = inverse_penalties * xta
        value = -float(a @ residual) + 0.25 * float(xta @ scaled_xta)
        gradient = -residual + 0.5 * (design @ scaled_xta)
        return value, gradient

    lower = tau - 1.0
    upper = tau
    # Zero is feasible for every tau in (0, 1) and avoids imposing a sign
    # pattern before the globally convex box optimization begins.
    initial_dual = np.zeros(residual.size, dtype=float)
    result = minimize(
        negative_dual_and_gradient,
        initial_dual,
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * residual.size,
        options={
            "ftol": 1e-15,
            "gtol": 1e-11,
            "maxiter": 2000,
            "maxls": 100,
            "maxcor": 20,
        },
    )

    raw_dual = np.asarray(result.x, dtype=float)
    dual_variables = (
        np.clip(raw_dual, lower, upper)
        if np.all(np.isfinite(raw_dual))
        else initial_dual.copy()
    )
    xta = design.T @ dual_variables

    def primal_dual_metrics() -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
        coefficients_now = 0.5 * inverse_penalties * xta
        fitted_residual_now = residual - design @ coefficients_now
        primal_now = float(
            np.sum(check_loss(fitted_residual_now, tau))
            + np.sum(penalties * coefficients_now**2)
        )
        dual_now = float(
            dual_variables @ residual - 0.25 * np.sum(inverse_penalties * xta**2)
        )
        gap_now = float(primal_now - dual_now)
        scale_now = max(1.0, abs(primal_now), abs(dual_now))
        gap_tolerance_now = float(max(gap_atol, gap_rtol * scale_now))
        bound_tolerance = 1e-8
        at_lower = dual_variables <= lower + bound_tolerance
        at_upper = dual_variables >= upper - bound_tolerance
        interior = ~(at_lower | at_upper)
        projected_kkt = np.zeros_like(fitted_residual_now)
        projected_kkt[at_lower] = np.maximum(fitted_residual_now[at_lower], 0.0)
        projected_kkt[at_upper] = np.maximum(-fitted_residual_now[at_upper], 0.0)
        projected_kkt[interior] = np.abs(fitted_residual_now[interior])
        return (
            coefficients_now,
            fitted_residual_now,
            primal_now,
            dual_now,
            gap_now,
            gap_tolerance_now,
            float(np.max(projected_kkt)),
        )

    (
        coefficients,
        fitted_residual,
        primal_objective,
        dual_objective,
        duality_gap,
        gap_tolerance,
        projected_kkt_violation,
    ) = primal_dual_metrics()

    slsqp_attempted = False
    slsqp_success = False
    slsqp_status: int | None = None
    slsqp_message: str | None = None
    slsqp_iterations: int | None = None
    if (
        duality_gap > gap_tolerance
        or duality_gap < -1e-10 * max(1.0, abs(primal_objective), abs(dual_objective))
        or projected_kkt_violation > kkt_tolerance
    ):
        # The dual Hessian has rank at most the number of update terms (two
        # here), so L-BFGS-B can terminate in a nearly flat non-identifiable
        # dual direction. SLSQP's active-bound handling is a robust second
        # route to the same globally convex QP; acceptance still depends only
        # on the primal--dual/KKT certificate below.
        slsqp_attempted = True
        slsqp_result = minimize(
            negative_dual_and_gradient,
            dual_variables,
            method="SLSQP",
            jac=True,
            bounds=[(lower, upper)] * residual.size,
            options={"ftol": 1e-13, "maxiter": 2000, "disp": False},
        )
        slsqp_success = bool(slsqp_result.success)
        slsqp_status = int(slsqp_result.status)
        slsqp_message = str(slsqp_result.message)
        slsqp_iterations = int(slsqp_result.nit)
        slsqp_dual = np.asarray(slsqp_result.x, dtype=float)
        if np.all(np.isfinite(slsqp_dual)):
            slsqp_dual = np.clip(slsqp_dual, lower, upper)
            slsqp_xta = design.T @ slsqp_dual
            slsqp_dual_objective = float(
                slsqp_dual @ residual
                - 0.25 * np.sum(inverse_penalties * slsqp_xta**2)
            )
            if slsqp_dual_objective >= dual_objective - 1e-10 * max(1.0, abs(dual_objective)):
                dual_variables = slsqp_dual
                xta = slsqp_xta
                (
                    coefficients,
                    fitted_residual,
                    primal_objective,
                    dual_objective,
                    duality_gap,
                    gap_tolerance,
                    projected_kkt_violation,
                ) = primal_dual_metrics()

    primal_qp_attempted = False
    primal_qp_success = False
    primal_qp_status: int | None = None
    primal_qp_message: str | None = None
    primal_qp_iterations: int | None = None
    if (
        duality_gap > gap_tolerance
        or duality_gap < -1e-10 * max(1.0, abs(primal_objective), abs(dual_objective))
        or projected_kkt_violation > kkt_tolerance
    ):
        # Independent epigraph form of the primal convex QP. Its inequality
        # multipliers map back to a feasible Fenchel-dual vector, providing a
        # numerically distinct active-set fallback without weakening the same
        # final global-optimality certificate.
        primal_qp_attempted = True
        n_observations = residual.size
        n_coefficients = design.shape[1]
        constraint_matrix = np.zeros(
            (2 * n_observations, n_coefficients + n_observations), dtype=float
        )
        constraint_matrix[:n_observations, :n_coefficients] = tau * design
        constraint_matrix[n_observations:, :n_coefficients] = (tau - 1.0) * design
        constraint_matrix[:n_observations, n_coefficients:] = np.eye(n_observations)
        constraint_matrix[n_observations:, n_coefficients:] = np.eye(n_observations)
        constraint_lower = np.concatenate(
            [tau * residual, (tau - 1.0) * residual]
        )
        epigraph_constraint = LinearConstraint(
            constraint_matrix,
            constraint_lower,
            np.full(2 * n_observations, np.inf),
        )
        primal_initial = np.concatenate(
            [coefficients, check_loss(residual - design @ coefficients, tau)]
        )

        def primal_epigraph_objective(parameters: np.ndarray) -> float:
            coefficients_now = parameters[:n_coefficients]
            return float(
                np.sum(parameters[n_coefficients:])
                + np.sum(penalties * coefficients_now**2)
            )

        def primal_epigraph_gradient(parameters: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    2.0 * penalties * parameters[:n_coefficients],
                    np.ones(n_observations, dtype=float),
                ]
            )

        primal_qp_result = minimize(
            primal_epigraph_objective,
            primal_initial,
            method="SLSQP",
            jac=primal_epigraph_gradient,
            constraints=[epigraph_constraint],
            options={"ftol": 1e-13, "maxiter": 2000, "disp": False},
        )
        primal_qp_success = bool(primal_qp_result.success)
        primal_qp_status = int(primal_qp_result.status)
        primal_qp_message = str(primal_qp_result.message)
        primal_qp_iterations = int(primal_qp_result.nit)
        multipliers = np.asarray(
            getattr(primal_qp_result, "multipliers", np.array([])), dtype=float
        )
        if multipliers.size >= 2 * n_observations and np.all(
            np.isfinite(multipliers[: 2 * n_observations])
        ):
            positive_branch = multipliers[:n_observations]
            negative_branch = multipliers[n_observations : 2 * n_observations]
            primal_dual = tau * positive_branch + (tau - 1.0) * negative_branch
            primal_dual = np.clip(primal_dual, lower, upper)
            primal_xta = design.T @ primal_dual
            primal_dual_objective = float(
                primal_dual @ residual
                - 0.25 * np.sum(inverse_penalties * primal_xta**2)
            )
            if primal_dual_objective >= dual_objective - 1e-10 * max(
                1.0, abs(dual_objective)
            ):
                dual_variables = primal_dual
                xta = primal_xta
                (
                    coefficients,
                    fitted_residual,
                    primal_objective,
                    dual_objective,
                    duality_gap,
                    gap_tolerance,
                    projected_kkt_violation,
                ) = primal_dual_metrics()

    # L-BFGS-B occasionally stops on relative objective change in a flat
    # (rank-deficient) dual direction before the primal--dual certificate is
    # tight. Exact cyclic coordinate minimization of the same convex box QP
    # is used only as a certificate-polishing step. Each update is the global
    # minimizer along that coordinate, and the loop is accepted only after
    # both the primal--dual gap and projected KKT residual pass.
    coordinate_polish_cycles = 0
    hessian_diagonal = 0.5 * np.sum(design**2 * inverse_penalties[None, :], axis=1)
    max_coordinate_polish_cycles = 50000
    while (
        duality_gap > gap_tolerance
        or duality_gap < -1e-10 * max(1.0, abs(primal_objective), abs(dual_objective))
        or projected_kkt_violation > kkt_tolerance
    ) and coordinate_polish_cycles < max_coordinate_polish_cycles:
        for i in range(residual.size):
            curvature = float(hessian_diagonal[i])
            if curvature <= np.finfo(float).eps:
                updated = upper if residual[i] > 0.0 else lower if residual[i] < 0.0 else dual_variables[i]
            else:
                gradient_i = float(
                    -residual[i] + 0.5 * np.dot(design[i] * inverse_penalties, xta)
                )
                updated = float(
                    np.clip(dual_variables[i] - gradient_i / curvature, lower, upper)
                )
            change = updated - dual_variables[i]
            if change != 0.0:
                dual_variables[i] = updated
                xta += design[i] * change
        coordinate_polish_cycles += 1
        if coordinate_polish_cycles < 10 or coordinate_polish_cycles % 10 == 0:
            (
                coefficients,
                fitted_residual,
                primal_objective,
                dual_objective,
                duality_gap,
                gap_tolerance,
                projected_kkt_violation,
            ) = primal_dual_metrics()

    (
        coefficients,
        fitted_residual,
        primal_objective,
        dual_objective,
        duality_gap,
        gap_tolerance,
        projected_kkt_violation,
    ) = primal_dual_metrics()
    stationarity_violation = float(
        np.max(np.abs(2.0 * penalties * coefficients - xta))
    )
    objective_scale = max(1.0, abs(primal_objective), abs(dual_objective))
    numerical_slack = 1e-10 * objective_scale
    certified = bool(
        duality_gap >= -numerical_slack
        and duality_gap <= gap_tolerance
        and projected_kkt_violation <= kkt_tolerance
        and stationarity_violation <= 1e-10
    )
    diagnostics: Dict[str, object] = {
        "method": GLOBAL_SOLVER_METHOD,
        "success": certified,
        "lbfgsb_success": bool(result.success),
        "lbfgsb_status": int(result.status),
        "lbfgsb_message": str(result.message),
        "lbfgsb_iterations": int(result.nit),
        "lbfgsb_function_evaluations": int(result.nfev),
        "slsqp_attempted": slsqp_attempted,
        "slsqp_success": slsqp_success,
        "slsqp_status": slsqp_status,
        "slsqp_message": slsqp_message,
        "slsqp_iterations": slsqp_iterations,
        "primal_qp_attempted": primal_qp_attempted,
        "primal_qp_success": primal_qp_success,
        "primal_qp_status": primal_qp_status,
        "primal_qp_message": primal_qp_message,
        "primal_qp_iterations": primal_qp_iterations,
        "coordinate_polish_cycles": int(coordinate_polish_cycles),
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "duality_gap_tolerance": gap_tolerance,
        "relative_duality_gap": float(max(duality_gap, 0.0) / objective_scale),
        "projected_kkt_violation": projected_kkt_violation,
        "kkt_tolerance": float(kkt_tolerance),
        "stationarity_violation": stationarity_violation,
    }
    if not certified:
        raise RuntimeError(
            "Fenchel-dual vector update failed its global-optimality checks: "
            f"lbfgsb_success={result.success}, status={result.status}, gap={duality_gap:.3e} "
            f"(tol={gap_tolerance:.3e}), projected_kkt={projected_kkt_violation:.3e} "
            f"(tol={kkt_tolerance:.3e}), stationarity={stationarity_violation:.3e}; "
            f"coordinate_cycles={coordinate_polish_cycles}; message={result.message}"
        )
    return coefficients, diagnostics


def profile_vector_update(
    residual_index: np.ndarray,
    time_index: np.ndarray,
    tau: float,
    lambda_intercept: float,
    lambda_slope: float,
    structure: str,
    return_diagnostics: bool = False,
) -> Tuple[float, float] | Tuple[float, float, Dict[str, object]]:
    residual_index = np.asarray(residual_index, dtype=float)
    time_index = np.asarray(time_index, dtype=float)
    slope_basis = (time_index - 12.0) / 12.0

    if structure == "none":
        result = (0.0, 0.0)
        diagnostics = {"method": "none", "success": True}
        return (*result, diagnostics) if return_diagnostics else result
    if structure == "intercept":
        result = (profiled_intercept(residual_index, tau=tau, lambda_b=lambda_intercept), 0.0)
        diagnostics = {"method": "exact_scalar", "success": True}
        return (*result, diagnostics) if return_diagnostics else result
    if structure == "slope":
        slope = one_dimensional_penalized_update(residual_index, slope_basis, tau=tau, lam=lambda_slope)
        result = (0.0, slope)
        diagnostics = {"method": "exact_scalar", "success": True}
        return (*result, diagnostics) if return_diagnostics else result
    if structure != "intercept_slope":
        raise ValueError(f"Unknown structure: {structure}")

    coefficients, diagnostics = penalized_vector_update_dual(
        residual_index,
        np.column_stack([np.ones(residual_index.size, dtype=float), slope_basis]),
        tau=tau,
        penalties=np.array([lambda_intercept, lambda_slope], dtype=float),
    )
    result = (float(coefficients[0]), float(coefficients[1]))
    return (*result, diagnostics) if return_diagnostics else result


def structure_losses(
    design: pd.DataFrame,
    beta: np.ndarray,
    tau: float,
    index_hours: float,
    structure: str,
    lambda_intercept: float = 0.0,
    lambda_slope: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    x_all = design.loc[:, X_PREDICTORS].to_numpy(dtype=float)
    fitted_all = x_all @ beta
    losses: List[float] = []
    rows: List[Dict[str, object]] = []
    for start, end in contiguous_stay_slices(design):
        y = y_all[start:end]
        t = t_all[start:end]
        fitted = fitted_all[start:end]
        index_idx, late_idx = strict_split_indices(t, index_hours)
        if index_idx.size == 0 or late_idx.size == 0:
            continue
        residual = y - fitted
        if structure == "none":
            intercept, slope = 0.0, 0.0
            solver_diagnostics: Dict[str, object] = {"method": "none", "success": True}
        else:
            intercept, slope, solver_diagnostics = profile_vector_update(
                residual[index_idx],
                t[index_idx],
                tau=tau,
                lambda_intercept=lambda_intercept,
                lambda_slope=lambda_slope,
                structure=structure,
                return_diagnostics=True,
            )
        late_slope_basis = (t[late_idx] - 12.0) / 12.0
        update = intercept + slope * late_slope_basis
        loss = float(np.mean(check_loss(residual[late_idx] - update, tau)))
        losses.append(loss)
        rows.append(
            {
                "stay_index": int(design["stay_index"].iat[start]),
                "loss": loss,
                "random_intercept": float(intercept),
                "random_slope": float(slope),
                "index_obs": int(index_idx.size),
                "late_obs": int(late_idx.size),
                "solver_method": str(solver_diagnostics["method"]),
                "solver_success": bool(solver_diagnostics["success"]),
                "solver_duality_gap": solver_diagnostics.get("duality_gap"),
                "solver_relative_duality_gap": solver_diagnostics.get("relative_duality_gap"),
                "solver_projected_kkt_violation": solver_diagnostics.get(
                    "projected_kkt_violation"
                ),
                "solver_stationarity_violation": solver_diagnostics.get(
                    "stationarity_violation"
                ),
                "solver_lbfgsb_iterations": solver_diagnostics.get("lbfgsb_iterations"),
                "solver_slsqp_attempted": solver_diagnostics.get("slsqp_attempted"),
                "solver_slsqp_success": solver_diagnostics.get("slsqp_success"),
                "solver_slsqp_iterations": solver_diagnostics.get("slsqp_iterations"),
                "solver_primal_qp_attempted": solver_diagnostics.get(
                    "primal_qp_attempted"
                ),
                "solver_primal_qp_success": solver_diagnostics.get("primal_qp_success"),
                "solver_primal_qp_iterations": solver_diagnostics.get(
                    "primal_qp_iterations"
                ),
                "solver_coordinate_polish_cycles": solver_diagnostics.get(
                    "coordinate_polish_cycles"
                ),
            }
        )
    return np.asarray(losses, dtype=float), pd.DataFrame(rows)


def summarize_global_solver_diagnostics(offsets: pd.DataFrame) -> Dict[str, object]:
    joint = offsets.loc[
        offsets["solver_method"] == GLOBAL_SOLVER_METHOD
    ].copy()
    if joint.empty:
        return {
            "global_solver_stays": 0,
            "global_solver_all_certified": True,
            "max_duality_gap": None,
            "max_relative_duality_gap": None,
            "max_projected_kkt_violation": None,
            "max_stationarity_violation": None,
            "mean_lbfgsb_iterations": None,
            "max_lbfgsb_iterations": None,
            "fraction_requiring_slsqp_fallback": None,
            "max_slsqp_iterations": None,
            "fraction_requiring_primal_qp_fallback": None,
            "max_primal_qp_iterations": None,
            "fraction_requiring_coordinate_polish": None,
            "max_coordinate_polish_cycles": None,
        }

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(joint[column], errors="raise").to_numpy(dtype=float)

    duality_gaps = numeric("solver_duality_gap")
    relative_gaps = numeric("solver_relative_duality_gap")
    projected_kkt = numeric("solver_projected_kkt_violation")
    stationarity = numeric("solver_stationarity_violation")
    lbfgsb_iterations = numeric("solver_lbfgsb_iterations")
    slsqp_attempted = joint["solver_slsqp_attempted"].astype(bool).to_numpy()
    slsqp_iterations = pd.to_numeric(
        joint.loc[slsqp_attempted, "solver_slsqp_iterations"], errors="raise"
    ).to_numpy(dtype=float)
    primal_qp_attempted = joint["solver_primal_qp_attempted"].astype(bool).to_numpy()
    primal_qp_iterations = pd.to_numeric(
        joint.loc[primal_qp_attempted, "solver_primal_qp_iterations"], errors="raise"
    ).to_numpy(dtype=float)
    polish_cycles = numeric("solver_coordinate_polish_cycles")
    return {
        "global_solver_stays": int(joint.shape[0]),
        "global_solver_all_certified": bool(joint["solver_success"].all()),
        "max_duality_gap": float(np.max(duality_gaps)),
        "max_relative_duality_gap": float(np.max(relative_gaps)),
        "max_projected_kkt_violation": float(np.max(projected_kkt)),
        "max_stationarity_violation": float(np.max(stationarity)),
        "mean_lbfgsb_iterations": float(np.mean(lbfgsb_iterations)),
        "max_lbfgsb_iterations": int(np.max(lbfgsb_iterations)),
        "fraction_requiring_slsqp_fallback": float(np.mean(slsqp_attempted)),
        "max_slsqp_iterations": (
            int(np.max(slsqp_iterations)) if slsqp_iterations.size else 0
        ),
        "fraction_requiring_primal_qp_fallback": float(np.mean(primal_qp_attempted)),
        "max_primal_qp_iterations": (
            int(np.max(primal_qp_iterations)) if primal_qp_iterations.size else 0
        ),
        "fraction_requiring_coordinate_polish": float(np.mean(polish_cycles > 0.0)),
        "max_coordinate_polish_cycles": int(np.max(polish_cycles)),
    }


def tune_structure(
    design: pd.DataFrame,
    beta: np.ndarray,
    tau: float,
    index_hours: float,
    structure: str,
    lambda_grid: Sequence[float],
    fixed_lambda_intercept: float | None = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    grid_rows: List[Dict[str, object]] = []
    candidates: List[Tuple[float, float]] = []
    if structure == "intercept":
        candidates = [(float(lam), 0.0) for lam in lambda_grid]
    elif structure == "slope":
        candidates = [(0.0, float(lam)) for lam in lambda_grid]
    elif structure == "intercept_slope":
        if fixed_lambda_intercept is None:
            candidates = [(float(l0), float(l1)) for l0 in lambda_grid for l1 in lambda_grid]
        else:
            candidates = [(float(fixed_lambda_intercept), float(l1)) for l1 in lambda_grid]
    else:
        raise ValueError(f"Unknown structure: {structure}")

    best = None
    for lambda_intercept, lambda_slope in candidates:
        losses, offsets = structure_losses(
            design,
            beta,
            tau=tau,
            index_hours=index_hours,
            structure=structure,
            lambda_intercept=lambda_intercept,
            lambda_slope=lambda_slope,
        )
        row = {
            "structure": structure,
            "lambda_intercept": float(lambda_intercept),
            "lambda_slope": float(lambda_slope),
            "tuning_loss": float(np.mean(losses)),
            "tuning_loss_se": se_mean(losses),
            "n_stays": int(losses.size),
            "median_abs_intercept": float(np.median(np.abs(offsets["random_intercept"].to_numpy(dtype=float)))),
            "median_abs_slope": float(np.median(np.abs(offsets["random_slope"].to_numpy(dtype=float)))),
        }
        row.update(summarize_global_solver_diagnostics(offsets))
        grid_rows.append(row)
        if best is None or row["tuning_loss"] < best["tuning_loss"]:
            best = row
    assert best is not None
    return best, pd.DataFrame(grid_rows)


def paired_comparison(
    reference_name: str,
    candidate_name: str,
    reference_losses: np.ndarray,
    candidate_losses: np.ndarray,
) -> Dict[str, object]:
    reference_losses = np.asarray(reference_losses, dtype=float)
    candidate_losses = np.asarray(candidate_losses, dtype=float)
    if reference_losses.shape != candidate_losses.shape:
        raise ValueError("Paired loss arrays must have identical shapes.")
    difference = reference_losses - candidate_losses
    mean = float(np.mean(difference))
    se = se_mean(difference)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "paired_reduction": mean,
        "paired_reduction_se": se,
        "ci_low": float(mean - 1.96 * se),
        "ci_high": float(mean + 1.96 * se),
        "relative_reduction_percent": float(100.0 * mean / np.mean(reference_losses)),
        "candidate_better_stay_fraction": float(np.mean(candidate_losses < reference_losses)),
        "n_stays": int(difference.size),
    }


def validate_reconstruction_inputs(
    results: Dict[str, object],
    args: argparse.Namespace,
    data_summary: Dict[str, object],
    age_standardization: Dict[str, float],
    split_counts: Dict[str, int],
) -> None:
    expected_split = results["split"]
    exact_checks = {
        "seed": (int(expected_split["seed"]), int(args.seed)),
        "train_stays": (int(expected_split["train_stays"]), int(split_counts["train_stays"])),
        "tuning_stays": (int(expected_split["tuning_stays"]), int(split_counts["tuning_stays"])),
        "assessment_stays": (
            int(expected_split["assessment_stays"]),
            int(split_counts["assessment_stays"]),
        ),
        "fit_stays": (int(results["data_summary"]["fit_stays"]), int(data_summary["fit_stays"])),
        "fit_observations": (
            int(results["data_summary"]["fit_observations"]),
            int(data_summary["fit_observations"]),
        ),
    }
    for label, (expected, observed) in exact_checks.items():
        if expected != observed:
            raise ValueError(f"Input mismatch for {label}: expected {expected}, observed {observed}.")

    close_checks = {
        "train_fraction": (float(expected_split["train_fraction"]), float(args.train_fraction)),
        "tuning_fraction": (float(expected_split["tuning_fraction"]), float(args.tuning_fraction)),
        "age_mean_training_split": (
            float(results["age_standardization"]["age_mean_training_split"]),
            float(age_standardization["age_mean_training_split"]),
        ),
        "age_sd_training_split": (
            float(results["age_standardization"]["age_sd_training_split"]),
            float(age_standardization["age_sd_training_split"]),
        ),
    }
    for label, (expected, observed) in close_checks.items():
        if not np.isclose(expected, observed, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Input mismatch for {label}: expected {expected}, observed {observed}.")


def write_structure_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Exploratory stay-specific landmark update structure comparison.}",
        "\\label{tab:random_effect_structure_comparison}",
        "\\centering",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Model & Update design & $\\lambda_0$ & $\\lambda_1$ & Assessment loss & SE\\\\",
        "\\hline",
    ]
    for row in rows:
        lam0 = "--" if row["lambda_intercept"] is None else f"{float(row['lambda_intercept']):.2g}"
        lam1 = "--" if row["lambda_slope"] is None else f"{float(row['lambda_slope']):.2g}"
        lines.append(
            f"{row['model']} & {row['random_effect_design']} & {lam0} & {lam1} & "
            f"{float(row['assessment_loss']):.4f} & {float(row['assessment_loss_se']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{The population component is the baseline-covariate 0.10 quantile model fitted on training stays. Stay-specific updates are estimated only from index-window residuals, with penalties selected on tuning stays and losses reported on assessment stays. All assessment-set structure comparisons in this table are exploratory. The centered slope basis is $(t-12)/12$. The level-plus-slope row fixes $\\lambda_0$ at the tuning-selected level-update value and tunes $\\lambda_1$; its joint convex update is solved through the box-constrained Fenchel dual and accepted only when primal--dual and KKT certificates pass. It is an exploratory incremental slope sensitivity, not an exhaustive joint penalty-grid search or a replacement for the frozen primary level update.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate random-effect structures for split-window QME.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--index-hours", type=float, default=12.0)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    beta = np.asarray(results["coefficients"]["baseline_tau_0.10"], dtype=float)

    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, _ = build_dataset_from_cache(
        obs,
        stays,
        fit_stays=args.fit_stays,
        seed=args.seed,
        analysis_hours=24.0,
    )
    enriched = data_utils.ensure_cluster_lists(dataset)
    train_idx, tuning_idx, assessment_idx = split_cluster_indices(
        len(enriched["y_list"]),
        seed=args.seed,
        train_fraction=args.train_fraction,
        tuning_fraction=args.tuning_fraction,
    )
    dataset, age_standardization = apply_training_age_standardization(dataset, stays, train_idx)
    split_counts = {
        "train_stays": int(train_idx.size),
        "tuning_stays": int(tuning_idx.size),
        "assessment_stays": int(assessment_idx.size),
    }
    validate_reconstruction_inputs(results, args, data_summary, age_standardization, split_counts)
    tuning_data = data_utils.subset_cluster_data(dataset, tuning_idx)
    assessment_data = data_utils.subset_cluster_data(dataset, assessment_idx)
    tuning_design, _ = design_frame(tuning_data)
    assessment_design, _ = design_frame(assessment_data)

    positive_grid = [0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0, 100.0]
    selected_lambda_intercept = float(results["settings"]["selected_lambda_baseline_update"])
    tuning_tables: List[pd.DataFrame] = []
    selected: Dict[str, Dict[str, object]] = {}
    for structure in ["intercept", "slope"]:
        best, grid = tune_structure(
            tuning_design,
            beta,
            tau=args.tau,
            index_hours=args.index_hours,
            structure=structure,
            lambda_grid=positive_grid,
        )
        selected[structure] = best
        tuning_tables.append(grid)
    best, grid = tune_structure(
        tuning_design,
        beta,
        tau=args.tau,
        index_hours=args.index_hours,
        structure="intercept_slope",
        lambda_grid=positive_grid,
        fixed_lambda_intercept=selected_lambda_intercept,
    )
    selected["intercept_slope"] = best
    tuning_tables.append(grid)

    pop_losses, _ = structure_losses(
        assessment_design,
        beta,
        tau=args.tau,
        index_hours=args.index_hours,
        structure="none",
    )

    loss_by_structure: Dict[str, np.ndarray] = {"none": pop_losses}
    rows: List[Dict[str, object]] = [
        {
            "model": "Population-only QR",
            "random_effect_design": "none",
            "lambda_intercept": None,
            "lambda_slope": None,
            "assessment_loss": float(np.mean(pop_losses)),
            "assessment_loss_se": se_mean(pop_losses),
            "n_stays": int(pop_losses.size),
            "median_abs_intercept": 0.0,
            "median_abs_slope": 0.0,
            **summarize_global_solver_diagnostics(
                pd.DataFrame(
                    {
                        "solver_method": ["none"],
                        "solver_success": [True],
                    }
                )
            ),
        }
    ]
    display = [
        ("Penalized stay-specific level update", "1", "intercept"),
        ("Penalized stay-specific slope-only update", "$(t-12)/12$", "slope"),
        ("Penalized stay-specific level + slope update", "$\\{1,(t-12)/12\\}$", "intercept_slope"),
    ]
    for model, design_label, structure in display:
        best = selected[structure]
        losses, offsets = structure_losses(
            assessment_design,
            beta,
            tau=args.tau,
            index_hours=args.index_hours,
            structure=structure,
            lambda_intercept=float(best["lambda_intercept"]),
            lambda_slope=float(best["lambda_slope"]),
        )
        loss_by_structure[structure] = losses
        comparison_row = {
                "model": model,
                "random_effect_design": design_label,
                "lambda_intercept": float(best["lambda_intercept"]),
                "lambda_slope": float(best["lambda_slope"]),
                "assessment_loss": float(np.mean(losses)),
                "assessment_loss_se": se_mean(losses),
                "n_stays": int(losses.size),
                "median_abs_intercept": float(np.median(np.abs(offsets["random_intercept"].to_numpy(dtype=float)))),
                "median_abs_slope": float(np.median(np.abs(offsets["random_slope"].to_numpy(dtype=float)))),
            }
        comparison_row.update(summarize_global_solver_diagnostics(offsets))
        rows.append(comparison_row)

    paired_rows = [
        paired_comparison("Population-only QR", "Level update", pop_losses, loss_by_structure["intercept"]),
        paired_comparison("Population-only QR", "Slope-only update", pop_losses, loss_by_structure["slope"]),
        paired_comparison(
            "Population-only QR",
            "Level + slope update",
            pop_losses,
            loss_by_structure["intercept_slope"],
        ),
        paired_comparison(
            "Level update",
            "Level + slope update",
            loss_by_structure["intercept"],
            loss_by_structure["intercept_slope"],
        ),
    ]

    diagnostic_keys = [
        "global_solver_stays",
        "global_solver_all_certified",
        "max_duality_gap",
        "max_relative_duality_gap",
        "max_projected_kkt_violation",
        "max_stationarity_violation",
        "mean_lbfgsb_iterations",
        "max_lbfgsb_iterations",
        "fraction_requiring_slsqp_fallback",
        "max_slsqp_iterations",
        "fraction_requiring_primal_qp_fallback",
        "max_primal_qp_iterations",
        "fraction_requiring_coordinate_polish",
        "max_coordinate_polish_cycles",
    ]
    assessment_joint = next(
        row for row in rows if row["model"] == "Penalized stay-specific level + slope update"
    )
    global_solver_diagnostics = {
        "solver": "fenchel_dual_box_qp",
        "primal_objective": "sum_i rho_tau(r_i - x_i^T b) + b^T Lambda b",
        "dual_objective": "a^T r - 0.25 a^T X Lambda^{-1} X^T a",
        "dual_bounds": [float(args.tau - 1.0), float(args.tau)],
        "gap_absolute_tolerance": float(DUAL_GAP_ATOL),
        "gap_relative_tolerance": float(DUAL_GAP_RTOL),
        "projected_kkt_tolerance": float(DUAL_KKT_TOL),
        "selected_tuning_candidate": {
            key: selected["intercept_slope"][key] for key in diagnostic_keys
        },
        "assessment": {key: assessment_joint[key] for key in diagnostic_keys},
    }

    pd.DataFrame(rows).to_csv(args.artifact_dir / "random_effect_structure_comparison.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        args.artifact_dir / "random_effect_structure_paired_comparisons.csv",
        index=False,
    )
    pd.concat(tuning_tables, ignore_index=True).to_csv(args.artifact_dir / "random_effect_structure_tuning_grid.csv", index=False)
    write_structure_tex(rows, args.artifact_dir / "random_effect_structure_comparison.tex")
    payload = {
        "status": "complete",
        "tau": float(args.tau),
        "index_hours": float(args.index_hours),
        "metadata": {
            "data_summary": data_summary,
            "age_standardization": age_standardization,
            "split": {
                "train_stays": int(train_idx.size),
                "tuning_stays": int(tuning_idx.size),
                "assessment_stays": int(assessment_idx.size),
            },
        },
        "selected": selected,
        "comparison": rows,
        "paired_comparisons": paired_rows,
        "global_solver_diagnostics": global_solver_diagnostics,
        "artifacts": {
            "random_effect_structure_comparison_tex": str(args.artifact_dir / "random_effect_structure_comparison.tex"),
            "random_effect_structure_comparison_csv": str(args.artifact_dir / "random_effect_structure_comparison.csv"),
            "random_effect_structure_tuning_grid_csv": str(args.artifact_dir / "random_effect_structure_tuning_grid.csv"),
            "random_effect_structure_paired_comparisons_csv": str(
                args.artifact_dir / "random_effect_structure_paired_comparisons.csv"
            ),
        },
    }
    (args.artifact_dir / "random_effect_structure_results.json").write_text(
        json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
