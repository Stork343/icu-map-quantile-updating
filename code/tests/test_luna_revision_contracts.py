import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_landmark_table7 import horizon_eligible_indices, preserve_existing_assignments
from run_split_window_mixed_effects_analysis import split_cluster_indices
from run_temporal_validation import aligned_temporal_assignment, temporal_assignment_table, temporal_indices
from split_window_clinical_core import validation_losses_generic
from split_window_analysis_core import split_indices_for_hours
from split_window_data import build_dataset_from_cache, count_split_indices, derive_age_fields, clock_split_indices


class AgeContractTests(unittest.TestCase):
    def test_icu_proxy_follows_shifted_year_difference(self) -> None:
        self.assertEqual(
            derive_age_fields(60, 2153, "2153-06-01T00:00:00", "icu_proxy")["age"],
            60.0,
        )
        self.assertEqual(
            derive_age_fields(60, 2153, "2154-06-01T00:00:00", "icu_proxy")["age"],
            61.0,
        )
        self.assertEqual(
            derive_age_fields(60, 2153, "2152-06-01T00:00:00", "icu_proxy")["age"],
            59.0,
        )

    def test_high_age_code_is_retained_without_claiming_true_age(self) -> None:
        result = derive_age_fields(91, 2153, datetime(2154, 1, 1), "icu_proxy")
        self.assertTrue(result["high_age_coded"])
        self.assertEqual(result["age_at_icu_proxy"], 92.0)
        self.assertEqual(result["age_definition"], "icu_proxy")

    def test_invalid_age_inputs_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            derive_age_fields(None, 2153, "2153-01-01", "anchor_age")
        with self.assertRaises(ValueError):
            derive_age_fields(60, None, "2153-01-01", "icu_proxy")
        with self.assertRaises(ValueError):
            derive_age_fields(60, "not-a-year", "2153-01-01", "anchor_age")

    def test_old_cache_without_age_provenance_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            build_dataset_from_cache(
                pd.DataFrame(),
                pd.DataFrame({"stay_id": [1]}),
                fit_stays=0,
                seed=1,
                analysis_hours=24.0,
            )


class TemporalAlignmentContractTests(unittest.TestCase):
    FIT = ("2008 - 2010", "2011 - 2013")
    TUNING = ("2014 - 2016",)
    ASSESSMENT = ("2017 - 2019", "2020 - 2022")

    def assign(self, group: str, anchor_year: int, intake_year: int):
        return aligned_temporal_assignment(
            group,
            anchor_year,
            f"{intake_year}-06-01",
            self.FIT,
            self.TUNING,
            self.ASSESSMENT,
        )

    def test_complete_containment_and_boundary_states(self) -> None:
        self.assertEqual(self.assign("2008 - 2010", 2153, 2153)["phase"], "fit")
        self.assertEqual(self.assign("2008 - 2010", 2153, 2154)["phase"], "fit")
        self.assertEqual(self.assign("2011 - 2013", 2153, 2154)["status"], "boundary_uncertain")
        self.assertEqual(self.assign("2014 - 2016", 2153, 2153)["phase"], "tuning")
        self.assertEqual(self.assign("2017 - 2019", 2153, 2153)["phase"], "assessment")

    def test_negative_year_delta_and_invalid_metadata(self) -> None:
        result = self.assign("2011 - 2013", 2153, 2152)
        self.assertEqual(result["year_delta"], -1)
        self.assertEqual(result["possible_year_lower"], 2010)
        invalid = self.assign("2011 - 2013", 2153, 2145)
        self.assertEqual(invalid["status"], "out_of_range")
        missing = aligned_temporal_assignment(
            "2011 - 2013",
            None,
            "2153-01-01",
            self.FIT,
            self.TUNING,
            self.ASSESSMENT,
        )
        self.assertEqual(missing["status"], "missing_or_invalid")

    def test_assignment_table_keeps_one_status_per_stay(self) -> None:
        stays = pd.DataFrame(
            {
                "stay_id": [10, 11, 12],
                "subject_id": [1, 2, 3],
                "intime": ["2153-01-01", "2154-01-01", "2130-01-01"],
            }
        )
        patients = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "anchor_year": [2153, 2153, 2153],
                "anchor_year_group": ["2008 - 2010", "2011 - 2013", "2017 - 2019"],
            }
        )
        result = temporal_assignment_table(
            stays,
            patients,
            self.FIT,
            self.TUNING,
            self.ASSESSMENT,
        )
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result["status"].value_counts().sum(), 3)
        self.assertIn("boundary_uncertain", set(result["status"]))
        self.assertIn("out_of_range", set(result["status"]))

    def test_aligned_indices_exclude_boundary_uncertain_stays(self) -> None:
        stays = pd.DataFrame(
            {
                "stay_id": [10, 11, 12, 13],
                "subject_id": [1, 2, 3, 4],
                "intime": ["2153-01-01", "2153-01-01", "2153-01-01", "2154-01-01"],
            }
        )
        patients = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 4],
                "anchor_year": [2153, 2153, 2153, 2153],
                "anchor_year_group": ["2008 - 2010", "2014 - 2016", "2017 - 2019", "2011 - 2013"],
            }
        )
        fit, tuning, assessment, period_values = temporal_indices(
            {"cluster_ids": np.array([10, 11, 12, 13])},
            stays,
            patients,
            self.FIT,
            self.TUNING,
            self.ASSESSMENT,
            assignment_rule="aligned_full_containment",
        )
        self.assertEqual(fit.tolist(), [0])
        self.assertEqual(tuning.tolist(), [1])
        self.assertEqual(assessment.tolist(), [2])
        self.assertEqual(period_values.tolist()[-1], "2011 - 2013")


class LandmarkSplitContractTests(unittest.TestCase):
    def test_landmark_runner_preserves_historical_assignment(self) -> None:
        base = {
            "y_list": [np.arange(8.0) for _ in range(6)],
            "t_list": [
                np.array([0.0, 1.0, 2.0, 3.0, 7.0, 13.0, 18.0, 22.0]) if i < 4
                else np.array([0.0, 1.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0])
                for i in range(6)
            ],
            "cluster_ids": np.arange(6),
        }
        old = horizon_eligible_indices(base, 12.0)
        split, info = preserve_existing_assignments(base, 20260522, 0.60, 0.20)
        historical = split_cluster_indices(old.size, 20260522, 0.60, 0.20)
        for group, positions in enumerate(historical):
            np.testing.assert_array_equal(np.sort(np.intersect1d(split[group], old)), np.sort(old[positions]))
        self.assertEqual(info["historical_12h_stays"], 4)
        self.assertEqual(info["new_base_stays"], 2)
        self.assertEqual(np.unique(np.concatenate(split)).size, 6)

    def test_clock_split_assigns_landmark_to_index(self) -> None:
        index, late = clock_split_indices(
            np.array([0.0, 3.0, 6.0, 12.0, 12.01, 13.0, 18.0, 24.0]),
            12.0,
            min_index_obs=4,
            min_late_obs=1,
        )
        np.testing.assert_array_equal(index, np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(late, np.array([4, 5, 6, 7]))

    def test_clock_split_rejects_failed_qualification_without_fallback(self) -> None:
        index, late = clock_split_indices(np.array([0.0, 1.0, 2.0, 7.0, 9.0]), 6.0, 4, 1)
        self.assertEqual(index.size, 0)
        self.assertEqual(late.size, 0)
        index, late = split_indices_for_hours(np.array([0.0, 1.0, 2.0, 7.0, 9.0]), 6.0)
        self.assertEqual(index.size, 0)
        self.assertEqual(late.size, 0)

    def test_clock_split_rejects_empty_later_window_and_single_record(self) -> None:
        index, late = clock_split_indices(np.array([0.0, 3.0, 6.0, 18.0]), 18.0, 4, 1)
        self.assertEqual(index.size, 0)
        self.assertEqual(late.size, 0)
        index, late = clock_split_indices(np.array([12.0]), 12.0, 0, 1)
        self.assertEqual(index.size, 0)
        self.assertEqual(late.size, 0)

    def test_count_split_is_explicit_and_disjoint(self) -> None:
        index, late = count_split_indices(np.arange(6.0), index_fraction=0.5)
        np.testing.assert_array_equal(index, np.array([0, 1, 2]))
        np.testing.assert_array_equal(late, np.array([3, 4, 5]))
        self.assertEqual(np.intersect1d(index, late).size, 0)

    def test_simulation_can_request_clock_split_without_clinical_minimum(self) -> None:
        index, late = split_indices_for_hours(np.array([0.0, 5.0, 13.0, 18.0]), 12.0, min_index_obs=0)
        np.testing.assert_array_equal(index, np.array([0, 1]))
        np.testing.assert_array_equal(late, np.array([2, 3]))

    def test_clinical_loss_entrypoint_drops_failed_clock_qualification(self) -> None:
        design = pd.DataFrame(
            {
                "y": [60.0, 61.0, 62.0, 70.0, 71.0],
                "time_hours": [0.0, 1.0, 2.0, 7.0, 9.0],
                "stay_index": [0, 0, 0, 0, 0],
                "x_intercept": [1.0] * 5,
            }
        )
        losses, offsets = validation_losses_generic(
            design,
            ["x_intercept"],
            np.array([0.0]),
            tau=0.10,
            index_hours=6.0,
            update="none",
        )
        self.assertEqual(losses.size, 0)
        self.assertEqual(offsets.size, 0)


if __name__ == "__main__":
    unittest.main()
