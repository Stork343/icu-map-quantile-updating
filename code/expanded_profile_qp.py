from __future__ import annotations

from pathlib import Path

import numpy as np
import osqp
from scipy import sparse

from split_window_analysis_core import check_loss


QP_SETTINGS = {
    "eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 1000000,
    "polishing": True, "polish_refine_iter": 10, "delta": 1e-10,
    "adaptive_rho_interval": 50, "verbose": False,
}


def solve_profile_qp(residual: np.ndarray, design: np.ndarray, tau: float, penalties: np.ndarray,
                     diagnostic_path: Path | None = None) -> tuple[np.ndarray, dict[str, object]]:
    residual, design, penalties = np.asarray(residual, dtype=float), np.asarray(design, dtype=float), np.asarray(penalties, dtype=float)
    if residual.ndim != 1 or design.shape != (residual.size, 2) or penalties.shape != (2,):
        raise ValueError("Vector profile inputs have inconsistent dimensions")
    if not 0 < tau < 1 or residual.size == 0 or np.any(penalties <= 0):
        raise ValueError("Vector profiles require observations, a valid quantile, and positive penalties")
    if not all(np.all(np.isfinite(value)) for value in (residual, design, penalties)):
        raise ValueError("Vector profile inputs must be finite")
    quadratic = .5 * (design / penalties[None, :]) @ design.T
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(quadratic)), q=-residual,
                 A=sparse.eye(residual.size, format="csc"),
                 l=np.full(residual.size, tau - 1.), u=np.full(residual.size, tau),
                 **QP_SETTINGS)
    solution = solver.solve(raise_error=False)
    a = np.clip(np.asarray(solution.x, dtype=float), tau - 1., tau)
    xta = design.T @ a
    coefficients = .5 * xta / penalties
    fitted_residual = residual - design @ coefficients
    primal = float(np.sum(check_loss(fitted_residual, tau)) + np.sum(penalties * coefficients ** 2))
    dual = float(a @ residual - .25 * np.sum(xta ** 2 / penalties))
    scale = max(1., abs(primal), abs(dual))
    gap = primal - dual
    kkt = float(np.max(np.abs(a - np.clip(a + fitted_residual, tau - 1., tau))))
    stationarity = float(np.max(np.abs(2. * penalties * coefficients - xta)))
    certified = gap >= -1e-10 * scale and gap <= max(1e-7, 1e-8 * scale) and kkt <= 5e-7 and stationarity <= 1e-10
    diagnostics = {"success": bool(certified), "method": "osqp_fenchel_dual_qp_polished", "status": solution.info.status,
                   "iterations": int(solution.info.iter), "polishing_status": int(solution.info.status_polish),
                   "primal_objective": primal, "dual_objective": dual, "duality_gap": gap,
                   "relative_duality_gap": max(gap, 0.) / scale, "projected_kkt_violation": kkt,
                   "stationarity_violation": stationarity}
    if not certified:
        if diagnostic_path is not None:
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(diagnostic_path, residual=residual, design=design, penalties=penalties, tau=tau,
                     coefficients=coefficients, dual_variables=a)
        raise RuntimeError(f"Uncertified OSQP profile QP: {diagnostics}")
    return coefficients, diagnostics
