import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_submission_validation_extensions import (
    attach_model_metrics,
    balanced_fold_ids,
    common_cohort_fixed_opportunity_analysis,
    discrete_calibration,
    fixed_capacity_extreme_groups,
    fixed_opportunity_analysis,
    inner_fit_tune_indices,
    interval_violation,
    public_prediction_frame,
    rehydrate_common_cohort_records_from_frozen_oof,
)


class DiscreteCalibrationTests(unittest.TestCase):
    def test_interval_violation_uses_probability_mass_bracket(self) -> None:
        self.assertEqual(interval_violation(0.05, 0.20, 0.10), 0.0)
        self.assertAlmostEqual(interval_violation(0.14, 0.20, 0.10), 0.04)
        self.assertAlmostEqual(interval_violation(0.01, 0.06, 0.10), 0.04)

    def test_strict_and_inclusive_probabilities_are_both_reported(self) -> None:
        records = pd.DataFrame(
            {
                "global_stay_index": [0, 1],
                "late_obs": [2, 4],
                "later_values": [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])],
                "prediction_rule": [1.0, 1.0],
            }
        )
        records = attach_model_metrics(records, {"rule": "prediction_rule"}, tau=0.10)
        summary, detail = discrete_calibration(records, {"rule": "prediction_rule"}, 0.10, "test", groups=2)
        overall = detail.loc[detail["group"] == "overall"].iloc[0]
        self.assertEqual(float(overall["p_y_lt_q_stay_equal"]), 0.0)
        self.assertAlmostEqual(float(overall["p_y_le_q_stay_equal"]), 0.375)
        self.assertEqual(float(overall["interval_violation_stay_equal"]), 0.0)
        self.assertAlmostEqual(float(overall["p_y_le_q_observation_weighted"]), 1.0 / 3.0)
        self.assertEqual(float(summary.iloc[0]["overall_interval_violation_stay_equal"]), 0.0)


class PartitionTests(unittest.TestCase):
    def test_outer_folds_are_balanced_and_exhaustive(self) -> None:
        folds = balanced_fold_ids(103, 5, seed=7)
        counts = np.bincount(folds, minlength=5)
        self.assertEqual(folds.size, 103)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_inner_fit_and_tune_are_disjoint_and_exhaust_outer_training(self) -> None:
        outer_train = np.arange(80, dtype=int)
        fit, tune = inner_fit_tune_indices(outer_train, fold=2, seed=11)
        self.assertEqual(np.intersect1d(fit, tune).size, 0)
        np.testing.assert_array_equal(np.sort(np.r_[fit, tune]), outer_train)
        self.assertEqual(fit.size, 60)
        self.assertEqual(tune.size, 20)

    def test_fixed_capacity_extremes_are_stable_under_q10_ties(self) -> None:
        prediction = np.full(10, 60.0)
        stable_index = np.arange(10, dtype=int)
        low, high = fixed_capacity_extreme_groups(prediction, stable_index)
        np.testing.assert_array_equal(np.flatnonzero(low), np.array([0, 1]))
        np.testing.assert_array_equal(np.flatnonzero(high), np.array([8, 9]))
        self.assertFalse(np.any(low & high))


class FixedOpportunityTests(unittest.TestCase):
    @staticmethod
    def records() -> pd.DataFrame:
        later = [
            np.array([55.0] * 12),
            np.array([62.0] * 12),
            np.array([68.0] * 12),
            np.array([75.0] * 12),
        ]
        return pd.DataFrame(
            {
                "stay_id": [100, 101, 102, 103],
                "local_stay_index": [0, 1, 2, 3],
                "global_stay_index": [0, 1, 2, 3],
                "late_obs": [12, 12, 12, 12],
                "later_values": later,
                "later_times": [np.arange(12.0)] * 4,
                "later_population_predictions": [values.copy() for values in later],
                "later_primary_level_update_predictions": [values.copy() for values in later],
                "index_q10": [50.0, 60.0, 80.0, 90.0],
                "prediction_population": [55.0, 62.0, 68.0, 75.0],
                "prediction_primary_level_update": [55.0, 62.0, 68.0, 75.0],
                "prediction_scalar": [60.0, 60.0, 60.0, 60.0],
            }
        )

    def test_trajectory_rules_use_observation_specific_predictions(self) -> None:
        records = self.records()
        scored = attach_model_metrics(
            records,
            {
                "population": "prediction_population",
                "primary_level_update": "prediction_primary_level_update",
                "scalar": "prediction_scalar",
            },
            tau=0.10,
        )
        self.assertTrue(np.allclose(scored["loss_population"], 0.0))
        self.assertTrue(np.allclose(scored["loss_primary_level_update"], 0.0))
        self.assertGreater(float(scored["loss_scalar"].mean()), 0.0)

    def test_fixed_k_has_three_thresholds_and_threshold_independent_losses(self) -> None:
        records = self.records()
        thresholds, losses = fixed_opportunity_analysis(
            records,
            {
                "population": "prediction_population",
                "primary_level_update": "prediction_primary_level_update",
                "scalar": "prediction_scalar",
            },
            tau=0.10,
            scope="test",
            k_grid=[4, 8, 12],
        )
        self.assertEqual(thresholds.shape[0], 9)
        self.assertEqual(losses.shape[0], 9)
        self.assertEqual(set(thresholds["operational_threshold_mmhg"]), {60.0, 65.0, 70.0})
        self.assertTrue((thresholds.groupby("first_k_later_records")["n_eligible_stays"].nunique() == 1).all())
        self.assertTrue(losses["check_loss_target"].str.contains("independent").all())
        self.assertTrue(np.allclose(losses.loc[losses["model"] == "population", "mean_stay_level_check_loss"], 0.0))

    def test_common_late12_cohort_is_identical_for_every_k_and_losses_are_paired(self) -> None:
        records = self.records()
        records["prediction_calibrated_q10"] = [55.0, 62.0, 68.0, 75.0]
        records.at[0, "later_primary_level_update_predictions"] = np.full(12, 60.0)
        records.at[3, "later_primary_level_update_predictions"] = np.full(12, 70.0)
        excluded = records.iloc[[0]].copy()
        excluded["stay_id"] = 104
        excluded["local_stay_index"] = 4
        excluded["global_stay_index"] = 4
        excluded["late_obs"] = 8
        excluded["later_values"] = [np.full(8, 55.0)]
        excluded["later_primary_level_update_predictions"] = [np.full(8, 60.0)]
        records = pd.concat([records, excluded], ignore_index=True)

        thresholds, losses, metadata = common_cohort_fixed_opportunity_analysis(
            records,
            tau=0.10,
            scope="nested_test",
            k_grid=[4, 8, 12],
        )
        self.assertEqual(metadata["common_cohort_stays"], 4)
        self.assertEqual(metadata["common_min_later_records"], 12)
        self.assertEqual(thresholds.shape[0], 9)
        self.assertEqual(losses.shape[0], 3)
        self.assertEqual(set(thresholds["n_common_cohort_stays"]), {4})
        self.assertEqual(set(losses["n_common_cohort_stays"]), {4})
        self.assertEqual(set(thresholds["low_q10_group_stays"]), {1})
        self.assertEqual(set(thresholds["high_q10_group_stays"]), {1})
        self.assertTrue(
            np.allclose(losses["calibrated_q10_mean_stay_level_check_loss"], 0.0)
        )
        self.assertTrue(
            np.allclose(losses["primary_level_update_mean_stay_level_check_loss"], 1.25)
        )
        self.assertTrue(
            np.allclose(losses["paired_difference_primary_minus_calibrated_q10"], 1.25)
        )

    def test_public_export_drops_identifiers_and_array_columns(self) -> None:
        public = public_prediction_frame(self.records())
        for column in (
            "stay_id",
            "local_stay_index",
            "later_values",
            "later_times",
            "later_population_predictions",
            "later_primary_level_update_predictions",
        ):
            self.assertNotIn(column, public.columns)
        self.assertIn("stay_index", public.columns)

    def test_common_only_rehydration_joins_by_deidentified_global_index(self) -> None:
        time = np.r_[np.arange(4.0), np.arange(13.0, 25.0)]
        dataset = {
            "cluster_ids": np.array([100, 200], dtype=np.int64),
            "y_list": [np.arange(16.0), np.arange(100.0, 116.0)],
            "t_list": [time.copy(), time.copy()],
        }
        frozen = pd.DataFrame(
            {
                "stay_index": [1, 0],
                "late_obs": [12, 12],
                "index_q10": [70.0, 60.0],
                "prediction_calibrated_q10": [68.0, 58.0],
                "prediction_primary_level_update": [69.0, 59.0],
                "outer_fold": [2, 1],
            }
        )
        records = rehydrate_common_cohort_records_from_frozen_oof(
            dataset,
            frozen,
            index_hours=12.0,
            min_later_records=12,
        )
        self.assertListEqual(records["global_stay_index"].tolist(), [0, 1])
        np.testing.assert_array_equal(records.iloc[0]["later_values"], np.arange(4.0, 16.0))
        np.testing.assert_array_equal(
            records.iloc[1]["later_primary_level_update_predictions"],
            np.full(12, 69.0),
        )
        self.assertNotIn("stay_id", records.columns)


if __name__ == "__main__":
    unittest.main()
