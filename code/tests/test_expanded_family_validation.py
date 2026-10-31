import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expanded_family_core import (
    LEVEL_GRID, SCALAR_GRID, SLOPE_GRID, StayLayout, candidate_specs,
    evaluate_prediction, later_prediction, profile_coefficients, selected_rules,
)
from expanded_profile_qp import solve_profile_qp
from run_submission_validation_extensions import CALIBRATED_SUMMARIES
from split_window_analysis_core import empirical_check_quantile
from split_window_analysis_core import check_loss


def numerical_layout():
    index_values = [np.array([62., 65., 70., 64.]), np.array([72., 76., 74., 80.])]
    population = np.array([60., 70.])
    return StayLayout(
        np.array([3, 8]), [y - base for y, base in zip(index_values, population)],
        [np.array([0., 4., 8., 12.]), np.array([0., 3., 7., 12.])],
        np.array([61., 66., 73., 75.]), np.array([.2, .8, .3, .9]),
        np.repeat(population, 2), np.repeat([0, 1], 2), np.array([0, 2]), np.array([2, 2]),
        pd.DataFrame({"index_q10": [empirical_check_quantile(y, .1) for y in index_values]}),
    )


class ExpandedFamilyTests(unittest.TestCase):
    def test_qp_matches_independent_epigraph_solution(self):
        residual = np.array([7.0, -4.0, 2.5, -8.0, 1.0, 11.0, -0.5])
        design = np.column_stack([np.ones(residual.size), [-1., -.82, -.63, -.41, -.24, -.08, 0.]])
        penalties = np.array([.03, .1])
        estimate, diagnostics = solve_profile_qp(residual, design, .1, penalties)
        n = residual.size
        matrix = np.vstack([np.column_stack([.1 * design, np.eye(n)]),
                            np.column_stack([-.9 * design, np.eye(n)])])
        constraint = LinearConstraint(matrix, np.r_[.1 * residual, -.9 * residual], np.full(2 * n, np.inf))
        objective = lambda parameters: float(np.sum(parameters[2:]) + np.sum(penalties * parameters[:2] ** 2))
        gradient = lambda parameters: np.r_[2 * penalties * parameters[:2], np.ones(n)]
        benchmark = minimize(objective, np.r_[np.zeros(2), check_loss(residual, .1)], jac=gradient,
                             method="SLSQP", constraints=[constraint], options={"ftol": 1e-12, "maxiter": 2000})
        self.assertTrue(benchmark.success, benchmark.message)
        np.testing.assert_allclose(estimate, benchmark.x[:2], atol=2e-6, rtol=0)
        self.assertAlmostEqual(diagnostics["primal_objective"], objective(benchmark.x), delta=2e-7)
        self.assertLessEqual(diagnostics["projected_kkt_violation"], 5e-7)

    def test_qp_certificates_with_discrete_and_near_collinear_inputs(self):
        rng = np.random.default_rng(20261015)
        for iteration in range(1000):
            n = int(rng.integers(4, 40))
            residual = rng.integers(-10, 20, n).astype(float)
            basis = rng.uniform(-1., 0., n)
            if iteration % 5 == 0:
                basis = -.5 + 1e-3 * basis
            design = np.column_stack([np.ones(n), basis])
            penalties = np.array([LEVEL_GRID[iteration % len(LEVEL_GRID)], SLOPE_GRID[iteration % len(SLOPE_GRID)]])
            _, diagnostics = solve_profile_qp(residual, design, .1, penalties)
            self.assertTrue(diagnostics["success"])
            self.assertLessEqual(diagnostics["projected_kkt_violation"], 5e-7)

    def test_complete_positive_definite_matrix_grid(self):
        specs = candidate_specs()
        vectors = [spec for spec in specs if spec["basis"] == "level_slope"]
        self.assertEqual(len(vectors), 48)
        self.assertEqual({(spec["lambda_level"], spec["lambda_slope"]) for spec in vectors},
                         {(level, slope) for level in LEVEL_GRID for slope in SLOPE_GRID})
        self.assertIn((.03, .1), {(spec["lambda_level"], spec["lambda_slope"]) for spec in vectors})
        self.assertTrue(all(np.linalg.eigvalsh(np.diag([spec["lambda_level"], spec["lambda_slope"]])).min() > 0 for spec in vectors))

    def test_zero_penalty_scalar_is_empirical_quantile(self):
        layout = numerical_layout()
        spec = next(spec for spec in candidate_specs() if spec["key"] == "scalar_0")
        coefficients, _ = profile_coefficients(layout, spec, .1, 12.)
        np.testing.assert_allclose(later_prediction(layout, coefficients),
                                   layout.summaries["index_q10"].to_numpy()[layout.later_stay], atol=1e-12)

    def test_profile_solution_uses_index_values_only(self):
        layout = numerical_layout()
        spec = next(spec for spec in candidate_specs() if spec["key"] == "vector_0_1")
        coefficients, diagnostics = profile_coefficients(layout, spec, .1, 12.)
        altered = replace(layout, later_values=layout.later_values + 1000.)
        altered_coefficients, _ = profile_coefficients(altered, spec, .1, 12.)
        np.testing.assert_array_equal(coefficients, altered_coefficients)
        self.assertLess(diagnostics["max_relative_duality_gap"], 1e-5)
        self.assertTrue(np.any(np.abs(coefficients[:, 1]) > 0))

    def test_time_dependent_predictions_are_scored_at_each_record(self):
        layout = numerical_layout()
        coefficients = np.array([[0., 2.], [0., -2.]])
        prediction = later_prediction(layout, coefficients)
        result = evaluate_prediction(layout, prediction, .1)
        expected = [np.mean(np.where((y - p) >= 0, .1 * (y - p), -.9 * (y - p)))
                    for y, p in zip(np.split(layout.later_values, 2), np.split(prediction, 2))]
        np.testing.assert_allclose(result["loss"], expected)

    def test_family_selection_uses_tuning_loss_and_canonical_labels(self):
        layout = numerical_layout()
        grid = [{**spec, "raw_tuning_loss": 4.} for spec in candidate_specs()]
        calibrations = {spec["key"]: {"tuning_loss": 3., "intercept": 0., "slope": 1.} for spec in candidate_specs()}
        calibrations.update({model: {"tuning_loss": 3., "intercept": 0., "slope": 1.} for model in CALIBRATED_SUMMARIES})
        calibrations["scalar_0"]["tuning_loss"] = .01
        calibrations["calibrated_q10"]["tuning_loss"] = .01
        rules = selected_rules(grid, calibrations, layout, .1)
        self.assertEqual(rules["selected_family"]["selected_model"], "calibrated_q10")
        self.assertEqual(tuple(spec["lambda_level"] for spec in candidate_specs() if spec["basis"] == "level"), SCALAR_GRID)


if __name__ == "__main__":
    unittest.main()
