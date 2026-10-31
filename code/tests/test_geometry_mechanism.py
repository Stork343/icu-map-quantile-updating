from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_adaptive_mixed_simulation import simulate_dataset
from verify_geometry_mechanism import compare_frames, reconstruct_intervals, reconstruct_records


def actual_constant_forecast_records() -> tuple[object, pd.DataFrame]:
    data = simulate_dataset("informative_recovery", 90, 7821)
    records = []
    forecasts = {"full_geometry": 74., "horizon_geometry": 73., "diagonal_profile": 70.}
    for scope in ("assessment", "reference"):
        for mode in ("joint", "direct"):
            for horizon in (4., 8., 12.):
                selected = data.later_time <= 12 + horizon
                for method, forecast in forecasts.items():
                    residual = data.later_y - forecast
                    losses = residual * (.1 - (residual < 0))
                    for index in range(len(data.x)):
                        records.append({"scope": scope, "selection_mode": mode, "horizon_hours": horizon,
                            "method": method, "synthetic_stay_index": index, "coverage_hours": data.coverage[index],
                            "index_count": len(data.index_time[index]),
                            "actual_span_hours": np.ptp(data.index_time[index]),
                            "normalized_time_variance": np.var((data.index_time[index] - 12) / 12),
                            "loss": losses[index, selected[index]].mean(),
                            "p_lt": (data.later_y[index, selected[index]] < forecast).mean(),
                            "p_le": (data.later_y[index, selected[index]] <= forecast).mean()})
    return data, pd.DataFrame(records)


class GeometryMechanismVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.records = actual_constant_forecast_records()

    def test_generated_record_pairing_matches_conditional_loss(self):
        methods, paired = reconstruct_records(self.records)
        self.assertEqual(len(methods), 108)
        self.assertEqual(len(paired), 108)
        condition = ((paired["scope"] == "assessment") & (paired["selection_mode"] == "direct")
            & (paired["horizon_hours"] == 8) & (paired["coverage_hours"] == 4)
            & (paired["first"] == "full_geometry") & (paired["second"] == "diagonal_profile"))
        observed = paired.loc[condition].iloc[0]
        differences = []
        for index in np.flatnonzero(self.data.coverage == 4):
            values = self.data.later_y[index, self.data.later_time[index] <= 20]
            residual_first, residual_second = values - 74, values - 70
            loss_first = residual_first * (.1 - (residual_first < 0))
            loss_second = residual_second * (.1 - (residual_second < 0))
            differences.append(np.mean(loss_first - loss_second))
        self.assertAlmostEqual(observed["mean_difference"], np.mean(differences), places=13)
        self.assertAlmostEqual(observed["within_sample_se"], np.std(differences, ddof=1) / np.sqrt(len(differences)), places=13)
        self.assertEqual(observed["n_stays"], len(differences))

    def test_rejects_missing_rule_within_a_coverage_group(self):
        incomplete = self.records.drop(self.records.index[0])
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            reconstruct_records(incomplete)

    def test_rejects_changed_early_information_between_rules(self):
        changed = self.records.copy()
        changed.loc[0, "index_count"] += 1
        with self.assertRaisesRegex(ValueError, "changes its early information"):
            reconstruct_records(changed)

    def test_rejects_repeated_stay_and_reversed_tail_probabilities(self):
        repeated = pd.concat([self.records, self.records.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Repeated stay"):
            reconstruct_records(repeated)
        reversed_bracket = self.records.copy()
        reversed_bracket.loc[0, ["p_lt", "p_le"]] = [.5, .25]
        with self.assertRaisesRegex(ValueError, "probability mass bracket"):
            reconstruct_records(reversed_bracket)

    def test_interval_unit_is_replicate_with_unequal_stay_counts(self):
        rows = pd.DataFrame({"setting": ["information_limit"] * 2, "scope": ["reference"] * 2,
            "selection_mode": ["direct"] * 2, "horizon_hours": [12.] * 2, "coverage_hours": [4.] * 2,
            "first": ["full_geometry"] * 2, "second": ["diagonal_profile"] * 2,
            "replicate": [1, 2], "mean_difference": [.1, .3], "n_stays": [10, 100]})
        result = reconstruct_intervals(rows, 2).iloc[0]
        self.assertAlmostEqual(result["mean_difference"], .2, places=14)
        self.assertAlmostEqual(result["mcse"], .1, places=14)
        self.assertAlmostEqual(result["pointwise_high"], .2 + t.ppf(.975, 1) * .1, places=12)
        self.assertAlmostEqual(result["simultaneous_low"], .2 - t.ppf(1 - .05 / (2 * 432), 1) * .1, places=10)
        self.assertEqual(result["total_stay_evaluations"], 110)
        self.assertGreater(result["simultaneous_high"], result["pointwise_high"])

    def test_interval_rejects_duplicate_replicate_membership(self):
        rows = pd.DataFrame({"setting": ["information_limit"] * 2, "scope": ["reference"] * 2,
            "selection_mode": ["direct"] * 2, "horizon_hours": [12.] * 2, "coverage_hours": [4.] * 2,
            "first": ["full_geometry"] * 2, "second": ["diagonal_profile"] * 2,
            "replicate": [1, 1], "mean_difference": [.1, .3], "n_stays": [10, 100]})
        with self.assertRaisesRegex(ValueError, "omitted an independent replicate"):
            reconstruct_intervals(rows, 2)

    def test_table_comparison_detects_changed_scientific_value(self):
        first = pd.DataFrame({"key": ["a", "b"], "loss": [.2, .3]})
        changed = pd.DataFrame({"key": ["a", "b"], "loss": [.2, .31]})
        with self.assertRaises(AssertionError):
            compare_frames(changed, first, ["key"], ["loss"])


if __name__ == "__main__":
    unittest.main()
