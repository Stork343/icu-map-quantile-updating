from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component_learning_selection import (
    DIAGONAL_GRID, choose_constant, constant_penalties, grouped_folds, horizon_weights,
)


class ComponentSelectionTests(unittest.TestCase):
    def test_horizon_weights_keep_different_eligibility_counts_equal(self):
        horizons = np.asarray([4., 4., 8., 8., 8., 12., 12., 12., 12.])
        weights = horizon_weights(horizons)
        for horizon in (4, 8, 12):
            self.assertAlmostEqual(weights[horizons == horizon].sum(), 1.0 / 3.0, places=14)
        self.assertAlmostEqual(weights.sum(), 1.0, places=14)

    def test_repeated_stay_has_one_inner_fold_across_horizons(self):
        identities = np.asarray([9, 2, 7, 4, 5, 1, 9, 2, 7, 4, 5, 1, 9, 7, 4, 5, 1])
        folds = grouped_folds(identities, 20261022)
        self.assertEqual(set(folds), {0, 1, 2})
        np.testing.assert_array_equal(folds, grouped_folds(identities, 20261022))
        for identity in np.unique(identities):
            self.assertEqual(len(set(folds[identities == identity])), 1)

    def test_constant_selection_uses_training_rows_only(self):
        horizons = np.tile(np.asarray([4., 8., 12.]), 4)
        penalties = DIAGONAL_GRID
        losses = np.sum((np.log(penalties)[None, :, :] - np.asarray([.2, -.7])) ** 2, axis=2)
        losses = np.repeat(losses, len(horizons), axis=0)
        training = np.arange(6)
        selected, pairs = choose_constant(losses, horizons, training)
        changed = losses.copy()
        changed[6:] = changed[6:, ::-1]
        reselected, repa = choose_constant(changed, horizons, training)
        np.testing.assert_array_equal(selected, reselected)
        np.testing.assert_array_equal(pairs, repa)

    def test_constant_predictions_retain_each_horizon_pair(self):
        horizons = np.asarray([12., 4., 8., 4., 12.])
        pairs = np.asarray([[.03, .1], [.3, 3.], [1., 100.]])
        actual = constant_penalties(horizons, pairs)
        np.testing.assert_array_equal(actual, pairs[[2, 0, 1, 0, 2]])

    def test_training_rejects_an_absent_horizon(self):
        with self.assertRaisesRegex(ValueError, "all three"):
            horizon_weights(np.asarray([4., 8., 8.]))


if __name__ == "__main__":
    unittest.main()
