from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_temporal_validation import paired_interval, q10_profile_strata, temporal_indices


class TemporalValidationTests(unittest.TestCase):
    def test_paired_interval_uses_stay_level_differences(self) -> None:
        candidate = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([2.0, 2.0, 4.0, 4.0])
        result = paired_interval(candidate, reference)
        difference = candidate - reference
        expected_se = np.std(difference, ddof=1) / np.sqrt(difference.size)
        self.assertEqual(result["n_stays"], 4)
        self.assertAlmostEqual(result["mean_difference"], float(np.mean(difference)))
        self.assertAlmostEqual(result["se"], float(expected_se))

    def test_temporal_indices_are_disjoint_and_complete(self) -> None:
        dataset = {"cluster_ids": np.array([101, 102, 103, 104])}
        stays = pd.DataFrame(
            {
                "stay_id": [101, 102, 103, 104],
                "subject_id": [1, 2, 3, 4],
            }
        )
        patients = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 4],
                "anchor_year_group": [
                    "2008 - 2010",
                    "2011 - 2013",
                    "2014 - 2016",
                    "2017 - 2019",
                ],
            }
        )
        fit, tuning, assessment, periods = temporal_indices(
            dataset,
            stays,
            patients,
            ("2008 - 2010", "2011 - 2013"),
            ("2014 - 2016",),
            ("2017 - 2019",),
        )
        self.assertEqual(fit.tolist(), [0, 1])
        self.assertEqual(tuning.tolist(), [2])
        self.assertEqual(assessment.tolist(), [3])
        self.assertEqual(
            periods.tolist(),
            ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"],
        )

    def test_q10_profile_strata_use_equal_count_groups(self) -> None:
        records = pd.DataFrame(
            {
                "index_q10": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
                "global_stay_index": [0, 1, 2, 3, 4, 5],
                "loss_calibrated_q10": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                "loss_primary_level_update": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        result = q10_profile_strata(records)
        self.assertEqual(result["n_stays"].tolist(), [2, 2, 2])
        self.assertEqual(result["mean_q10_minus_profile_loss"].tolist(), [-1.0, 0.0, 1.0])
        self.assertTrue(
            np.all(result["simultaneous_ci95_high"] - result["simultaneous_ci95_low"]
                   >= result["ci95_high"] - result["ci95_low"])
        )
