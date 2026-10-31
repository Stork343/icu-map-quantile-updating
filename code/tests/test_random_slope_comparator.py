import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_random_slope_comparator import random_slope_blp


class RandomSlopePredictionTests(unittest.TestCase):
    def test_matches_dense_blp_for_correlated_coefficients(self):
        rng = np.random.default_rng(20261021)
        for count in (4, 13, 40):
            design = np.column_stack([np.ones(count), np.sort(rng.uniform(-1, 0, count))])
            covariance = np.array([[16.0, -3.0], [-3.0, 4.0]])
            residual = rng.normal(2.0, 6.0, count)
            mean, variance = 7.3, 21.0
            expected = covariance @ design.T @ np.linalg.solve(
                design @ covariance @ design.T + variance * np.eye(count), residual - mean
            )
            actual = random_slope_blp(residual, design, covariance, mean, variance)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    def test_rejects_invalid_covariance(self):
        with self.assertRaises(ValueError):
            random_slope_blp(np.ones(4), np.ones((4, 2)), np.array([[1.0, 2.0], [2.0, 1.0]]), 0.0, 1.0)

    def test_rejects_nonfinite_asymmetric_laplace_moments(self):
        with self.assertRaises(ValueError):
            random_slope_blp(np.ones(4), np.ones((4, 2)), np.eye(2), np.nan, 1.0)

    def test_rejects_asymmetric_covariance(self):
        with self.assertRaises(ValueError):
            random_slope_blp(np.ones(4), np.ones((4, 2)), np.array([[2.0, 0.0], [1.0, 2.0]]), 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
