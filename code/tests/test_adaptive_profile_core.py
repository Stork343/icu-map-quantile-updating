from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_profile_core import (
    ProfileStay, adaptive_candidate_specs, apply_stump_selector, check_loss,
    evaluate_profile, fit_affine_calibration, fit_profile, fit_stump_selector,
    horizon_geometry, index_features, penalty_matrix, solve_spd_profile,
)


class AdaptiveProfileTests(unittest.TestCase):
    def test_spd_dual_against_independent_primal_quadratic_program(self) -> None:
        rng = np.random.default_rng(20261021)
        for size in (2, 4, 9, 17):
            for tau in (0.1, 0.5, 0.9):
                design = np.column_stack([np.ones(size), rng.uniform(-1.0, 0.0, size)])
                residual = rng.normal(0.0, 7.0, size)
                penalty = np.asarray([[0.6, 0.23], [0.23, 0.4]])
                coefficients, diagnostics = solve_spd_profile(residual, design, tau, penalty)
                matrix = np.column_stack([design, np.eye(size), -np.eye(size)])
                initial = np.r_[np.zeros(2), np.maximum(residual, 0.0), np.maximum(-residual, 0.0)]

                def objective(value: np.ndarray) -> float:
                    return float(value[:2] @ penalty @ value[:2] + tau * np.sum(value[2:2 + size])
                                 + (1.0 - tau) * np.sum(value[2 + size:]))

                def gradient(value: np.ndarray) -> np.ndarray:
                    return np.r_[2.0 * penalty @ value[:2], np.full(size, tau), np.full(size, 1.0 - tau)]

                independent = minimize(objective, initial, jac=gradient,
                                       bounds=Bounds(np.r_[[-np.inf, -np.inf], np.zeros(2 * size)], np.inf),
                                       constraints=LinearConstraint(matrix, residual, residual),
                                       method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000})
                self.assertTrue(independent.success, independent.message)
                np.testing.assert_allclose(coefficients, independent.x[:2], atol=2e-6, rtol=2e-6)
                self.assertAlmostEqual(float(diagnostics["primal_objective"]), independent.fun, places=7)

    def test_coordinate_invariance_for_full_spd_penalty(self) -> None:
        design = np.column_stack([np.ones(5), np.linspace(-1.0, 0.0, 5)])
        residual = np.asarray([-7.0, -3.0, 0.0, 4.0, 9.0])
        penalty = np.asarray([[0.3, 0.12], [0.12, 0.2]])
        transform = np.asarray([[2.0, 0.4], [-0.3, 0.8]])
        original, _ = solve_spd_profile(residual, design, 0.1, penalty)
        changed, _ = solve_spd_profile(residual, design @ transform, 0.1,
                                       transform.T @ penalty @ transform)
        np.testing.assert_allclose(original, transform @ changed, atol=5e-8, rtol=5e-8)

    def test_active_set_polishing_near_dual_bounds(self) -> None:
        rng = np.random.default_rng(20261022)
        for size in (4, 10, 21):
            for repetition in range(12):
                design = np.column_stack([np.ones(size), np.sort(rng.uniform(-1.0, 0.0, size))])
                penalty = (0.03 if repetition % 2 == 0 else 3.0) * horizon_geometry(4.0)
                dual = rng.choice(np.asarray([-0.9, 0.1]), size=size)
                dual[0] = 0.1 - 10.0 ** (-4 - repetition % 6)
                expected = 0.5 * np.linalg.solve(penalty, design.T @ dual)
                residual = design @ expected
                residual[dual == 0.1] += rng.uniform(0.01, 8.0, int(np.sum(dual == 0.1)))
                residual[dual == -0.9] -= rng.uniform(0.01, 8.0, int(np.sum(dual == -0.9)))
                fitted, diagnostics = solve_spd_profile(residual, design, 0.1, penalty)
                self.assertTrue(diagnostics["success"])
                np.testing.assert_allclose(fitted, expected, atol=2e-7, rtol=2e-7)

    def test_degenerate_index_geometry_is_regularized(self) -> None:
        for times in (np.full(4, 12.0), np.asarray([0.0, 0.0, 0.0, 0.0]), np.asarray([12.0, 12.0])):
            stay = ProfileStay(np.zeros(times.size), times, np.asarray([60.0, 80.0]),
                               np.asarray([13.0, 23.0]), np.asarray([70.0, 70.0]))
            for horizon in (4.0, 8.0, 12.0):
                for spec in adaptive_candidate_specs():
                    penalty = penalty_matrix(stay, horizon, spec)
                    self.assertGreater(float(np.min(np.linalg.eigvalsh(penalty))), 0.0)
                    coefficient, diagnostics = fit_profile(stay, horizon, spec)
                    self.assertTrue(diagnostics["success"])
                    np.testing.assert_allclose(coefficient, 0.0, atol=1e-10)

    def test_index_features_and_horizon_scoring(self) -> None:
        stay = ProfileStay(np.asarray([-10.0, -5.0, 2.0, 9.0, 12.0]), np.arange(5.0),
                           np.asarray([60.0, 80.0, 100.0]), np.asarray([13.0, 17.0, 23.0]), np.full(3, 70.0))
        self.assertEqual(index_features(stay)[0], 1.0)
        np.testing.assert_allclose(horizon_geometry(12.0), [[1.0, 0.5], [0.5, 1.0 / 3.0]])
        result = evaluate_profile(stay, np.zeros(2), 4.0)
        self.assertEqual(result["n_later"], 1)
        self.assertAlmostEqual(result["loss"], float(check_loss(np.asarray([-10.0]), 0.1)[0]))
        changed = ProfileStay(stay.index_residual, stay.index_time, stay.later_y + 1000.0,
                              stay.later_time, stay.later_population)
        np.testing.assert_array_equal(index_features(stay), index_features(changed))

    def test_stump_uses_fitting_cutpoints_and_tuning_losses(self) -> None:
        training = np.column_stack([np.arange(8.0), np.ones(8), np.ones(8)])
        tuning = np.column_stack([np.arange(4.0), np.ones(4), np.ones(4)])
        losses = np.asarray([[0.0, 9.0], [0.0, 9.0], [9.0, 0.0], [9.0, 0.0]])
        selector = fit_stump_selector(training, tuning, losses, ["left", "right"])
        self.assertEqual(selector["threshold"], 1.75)
        np.testing.assert_array_equal(apply_stump_selector(selector, tuning), [0, 0, 1, 1])
        self.assertEqual(selector["tuning_loss"], 0.0)

    def test_exact_affine_calibration_uses_stay_weights(self) -> None:
        prediction = np.asarray([0.0, 1.0, 2.0, 3.0])
        outcomes = 5.0 + 2.0 * prediction
        fit = fit_affine_calibration(outcomes, prediction, np.asarray([1.0, 0.5, 0.5, 1.0]))
        self.assertAlmostEqual(fit["intercept"], 5.0)
        self.assertAlmostEqual(fit["slope"], 2.0)
        self.assertLess(fit["max_equality_residual"], 1e-8)

    def test_indefinite_penalty_fails(self) -> None:
        with self.assertRaises(np.linalg.LinAlgError):
            solve_spd_profile(np.ones(3), np.ones((3, 2)), 0.1, np.asarray([[1.0, 2.0], [2.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
