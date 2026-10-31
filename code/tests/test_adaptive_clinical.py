from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_profile_core import adaptive_candidate_specs
from run_adaptive_clinical import make_layout, select_groups


class AdaptiveClinicalTests(unittest.TestCase):
    def test_horizon_eligibility_retains_index_and_partition_order(self) -> None:
        dataset = {
            "X": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            "y_list": [np.arange(8.0) + 60.0, np.arange(8.0) + 70.0],
            "t_list": [np.asarray([1.0, 2.0, 4.0, 10.0, 15.0, 17.0, 20.0, 24.0]),
                       np.asarray([1.0, 2.0, 4.0, 10.0, 17.0, 18.0, 20.0, 24.0])],
        }
        layout, positions = make_layout(dataset, np.asarray([1, 0]), np.asarray([65.0, 0.0, 0.0, 0.0]), 4.0)
        np.testing.assert_array_equal(positions, [1])
        np.testing.assert_array_equal(layout.global_indices, [0])
        np.testing.assert_array_equal(layout.residual_index[0], [-5.0, -4.0, -3.0, -2.0])
        self.assertEqual(layout.later_counts.tolist(), [1])
        self.assertEqual(layout.later_values.tolist(), [64.0])
        all_stays, full_positions = make_layout(dataset, np.asarray([1, 0]), np.asarray([65.0, 0.0, 0.0, 0.0]), 12.0)
        np.testing.assert_array_equal(all_stays.global_indices, [1, 0])
        np.testing.assert_array_equal(full_positions, [0, 1])

    def test_group_selection_preserves_nested_candidate_sets(self) -> None:
        rules = []
        base_keys = ["population", "scalar_0", "scalar_1", "vector_0_0"]
        adaptive = adaptive_candidate_specs()
        for key in base_keys + [str(row["key"]) for row in adaptive]:
            for calibrated in (False, True):
                rules.append({"id": key + str(calibrated), "key": key,
                              "calibrated": calibrated, "tuning_loss": 10.0})
        best = next(rule for rule in rules if rule["key"] == adaptive[-1]["key"] and rule["calibrated"])
        best["tuning_loss"] = 0.0
        original = next(rule for rule in rules if rule["key"] == "scalar_1" and not rule["calibrated"])
        original["tuning_loss"] = 2.0
        selected = select_groups(rules)
        self.assertEqual(selected["original_selected"]["id"], original["id"])
        self.assertEqual(selected["expanded_selected"]["id"], best["id"])
        self.assertEqual(selected["adaptive_affine"]["id"], best["id"])
        specs = {str(row["key"]): row for row in adaptive}
        self.assertEqual(specs[selected["geometry_direct"]["key"]]["gamma"], 0.0)
        self.assertEqual(specs[selected["history_geometry_direct"]["key"]]["theta"], 0.0)


if __name__ == "__main__":
    unittest.main()
