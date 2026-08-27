import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_revised_mimic_map_cache import _aggregate_map_buckets, _time_bucket_from_datetimes
from generate_enhanced_empirical_figures import attach_outcomes, clinical_yield_tables
from generate_random_effect_structure_comparison import (
    one_dimensional_penalized_update,
    penalized_vector_update_dual,
)
from run_split_window_mixed_effects_analysis import split_cluster_indices
from split_window_analysis_core import (
    check_loss,
    empirical_check_quantile,
    equal_stay_observation_weights,
    profiled_intercept,
)
from split_window_data import build_dataset_from_cache


class EmpiricalQuantileTests(unittest.TestCase):
    def test_noninteger_rank_is_check_loss_minimizer(self) -> None:
        values = np.array([0.0, 10.0, 20.0])
        estimate = empirical_check_quantile(values, 0.10)
        self.assertEqual(estimate, 0.0)
        self.assertLess(
            float(np.sum(check_loss(values - estimate, 0.10))),
            float(np.sum(check_loss(values - 2.0, 0.10))),
        )

    def test_integer_rank_uses_lower_endpoint(self) -> None:
        self.assertEqual(empirical_check_quantile(np.array([1.0, 2.0]), 0.50), 1.0)

    def test_ties_and_permutations_are_deterministic(self) -> None:
        values = np.array([1.0, 1.0, 2.0, 2.0])
        self.assertEqual(empirical_check_quantile(values, 0.50), 1.0)
        self.assertEqual(empirical_check_quantile(values[::-1], 0.50), 1.0)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            empirical_check_quantile(np.array([]), 0.10)
        with self.assertRaises(ValueError):
            empirical_check_quantile(np.array([1.0]), 0.0)
        with self.assertRaises(ValueError):
            empirical_check_quantile(np.array([np.nan]), 0.10)


class ProfiledInterceptTests(unittest.TestCase):
    def test_unpenalized_matches_empirical_check_quantile(self) -> None:
        values = np.arange(13.0)
        self.assertEqual(profiled_intercept(values, 0.10, 0.0), 1.0)

    def test_positive_penalty_can_select_residual_knot(self) -> None:
        values = np.arange(3.0, 13.0)
        estimate = profiled_intercept(values, 0.10, 0.10)
        self.assertEqual(estimate, 3.0)

    def test_positive_penalty_can_select_interval_stationary_point(self) -> None:
        values = np.array([-10.0, 10.0])
        estimate = profiled_intercept(values, 0.50, 1.0)
        self.assertEqual(estimate, 0.0)

    def test_negative_penalty_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profiled_intercept(np.array([1.0, 2.0]), 0.10, -0.1)

    def test_random_outputs_satisfy_subgradient_condition(self) -> None:
        rng = np.random.default_rng(20260822)
        for n in (2, 3, 5, 10, 17):
            for tau in (0.05, 0.10, 0.25, 0.50, 0.90):
                for lam in (0.03, 0.10, 1.0, 10.0):
                    for _ in range(20):
                        values = rng.integers(-5, 6, size=n).astype(float)
                        estimate = profiled_intercept(values, tau, lam)
                        n_less = int(np.sum(values < estimate))
                        n_less_equal = int(np.sum(values <= estimate))
                        left = n_less - n * tau + 2.0 * lam * estimate
                        right = n_less_equal - n * tau + 2.0 * lam * estimate
                        self.assertLessEqual(left, 1e-9)
                        self.assertGreaterEqual(right, -1e-9)


class ProfiledSlopeTests(unittest.TestCase):
    def test_unpenalized_slope_minimizes_weighted_check_loss(self) -> None:
        residual = np.array([0.0, 1000.0])
        covariate = np.array([1.0, 100.0])
        estimate = one_dimensional_penalized_update(residual, covariate, tau=0.10, lam=0.0)
        self.assertEqual(estimate, 10.0)
        self.assertLess(
            float(np.sum(check_loss(residual - covariate * estimate, 0.10))),
            float(np.sum(check_loss(residual - covariate * 5.0, 0.10))),
        )

    def test_negative_slope_penalty_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            one_dimensional_penalized_update(
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                tau=0.10,
                lam=-0.1,
            )


class ProfiledVectorTests(unittest.TestCase):
    def test_global_dual_escapes_legacy_coordinatewise_nonsmooth_trap(self) -> None:
        residual = np.array(
            [-5.983597961903152, -1.0003530739774353, -1.0260276520740645, -2.0019006525268637]
        )
        slope_basis = np.array(
            [-0.6956918068864825, -0.20394731669671584, -0.5989092218930165, -0.017405019252989762]
        )
        design = np.column_stack([np.ones(residual.size), slope_basis])
        penalties = np.array([0.03, 0.10])
        tau = 0.10

        global_estimate, diagnostics = penalized_vector_update_dual(
            residual,
            design,
            tau=tau,
            penalties=penalties,
        )

        # This reproduces the retired alternating exact-coordinate routine.
        legacy_intercept = profiled_intercept(residual, tau=tau, lambda_b=penalties[0])
        legacy_slope = 0.0
        previous = np.array([legacy_intercept, legacy_slope])
        for _ in range(100):
            legacy_intercept = profiled_intercept(
                residual - slope_basis * legacy_slope,
                tau=tau,
                lambda_b=penalties[0],
            )
            legacy_slope = one_dimensional_penalized_update(
                residual - legacy_intercept,
                slope_basis,
                tau=tau,
                lam=penalties[1],
            )
            current = np.array([legacy_intercept, legacy_slope])
            if np.max(np.abs(current - previous)) < 1e-10:
                break
            previous = current

        legacy_estimate = np.array([legacy_intercept, legacy_slope])

        def penalized_objective(coefficients: np.ndarray) -> float:
            return float(
                np.sum(check_loss(residual - design @ coefficients, tau))
                + np.sum(penalties * coefficients**2)
            )

        self.assertGreater(
            penalized_objective(legacy_estimate) - penalized_objective(global_estimate),
            0.30,
        )
        self.assertTrue(diagnostics["success"])

    def test_dual_solution_matches_independent_convex_epigraph_qp(self) -> None:
        residual = np.array([7.0, -4.0, 2.5, -8.0, 1.0, 11.0, -0.5])
        design = np.column_stack(
            [
                np.ones(residual.size),
                np.array([-1.0, -0.82, -0.63, -0.41, -0.24, -0.08, 0.0]),
            ]
        )
        penalties = np.array([0.03, 0.10])
        tau = 0.10
        estimate, diagnostics = penalized_vector_update_dual(
            residual,
            design,
            tau=tau,
            penalties=penalties,
        )

        n = residual.size
        constraint_matrix = np.zeros((2 * n, 2 + n), dtype=float)
        constraint_matrix[:n, :2] = tau * design
        constraint_matrix[n:, :2] = (tau - 1.0) * design
        constraint_matrix[:n, 2:] = np.eye(n)
        constraint_matrix[n:, 2:] = np.eye(n)
        lower = np.concatenate([tau * residual, (tau - 1.0) * residual])
        constraint = LinearConstraint(
            constraint_matrix,
            lower,
            np.full(2 * n, np.inf),
        )

        initial = np.concatenate([np.zeros(2), check_loss(residual, tau)])

        def objective(parameters: np.ndarray) -> float:
            coefficients = parameters[:2]
            epigraph = parameters[2:]
            return float(np.sum(epigraph) + np.sum(penalties * coefficients**2))

        def gradient(parameters: np.ndarray) -> np.ndarray:
            return np.concatenate([2.0 * penalties * parameters[:2], np.ones(n)])

        benchmark = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=gradient,
            constraints=[constraint],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        self.assertTrue(benchmark.success, msg=benchmark.message)
        np.testing.assert_allclose(estimate, benchmark.x[:2], atol=2e-6, rtol=0.0)
        self.assertAlmostEqual(
            diagnostics["primal_objective"],
            objective(benchmark.x),
            delta=2e-7,
        )
        self.assertTrue(diagnostics["success"])
        self.assertLessEqual(
            diagnostics["duality_gap"], diagnostics["duality_gap_tolerance"]
        )
        self.assertLessEqual(
            diagnostics["projected_kkt_violation"], diagnostics["kkt_tolerance"]
        )

    def test_dual_vector_update_rejects_nonpositive_penalties(self) -> None:
        residual = np.array([1.0, -1.0, 2.0])
        design = np.column_stack([np.ones(3), np.array([-1.0, -0.5, 0.0])])
        for penalties in (np.array([0.0, 0.1]), np.array([0.1, -0.1])):
            with self.assertRaises(ValueError):
                penalized_vector_update_dual(
                    residual,
                    design,
                    tau=0.10,
                    penalties=penalties,
                )


class MapBucketTests(unittest.TestCase):
    def test_exact_five_minute_boundaries_use_integer_buckets(self) -> None:
        intime = pd.Series(pd.to_datetime(["2026-01-01 00:00:00"] * 4))
        charttime = pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01 04:05:00",
                    "2026-01-01 08:10:00",
                    "2026-01-01 08:25:00",
                    "2026-01-01 16:05:00",
                ]
            )
        )
        buckets = _time_bucket_from_datetimes(charttime, intime, 5)
        np.testing.assert_array_equal(buckets, np.array([49, 98, 101, 193], dtype=np.int64))

    def test_bucket_source_priority_and_mean(self) -> None:
        frame = pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 1],
                "charttime": pd.to_datetime(
                    [
                        "2026-01-01 00:00:00",
                        "2026-01-01 00:01:00",
                        "2026-01-01 00:02:00",
                        "2026-01-01 00:05:00",
                        "2026-01-01 00:06:00",
                    ]
                ),
                "time_hours": [0.0, 1.0 / 60.0, 2.0 / 60.0, 5.0 / 60.0, 6.0 / 60.0],
                "time_bucket": [0, 0, 0, 1, 1],
                "source_priority": [0, 0, 1, 1, 1],
                "itemid": [220052, 220052, 220181, 220181, 220181],
                "map_value": [60.0, 64.0, 100.0, 70.0, 74.0],
            }
        )
        result = _aggregate_map_buckets(frame)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.loc[0, "map_source"], "invasive")
        self.assertEqual(int(result.loc[0, "itemid"]), 220052)
        self.assertEqual(float(result.loc[0, "map_value"]), 62.0)
        self.assertEqual(int(result.loc[0, "bucket_measurement_count"]), 2)
        self.assertEqual(result.loc[1, "map_source"], "noninvasive")
        self.assertEqual(float(result.loc[1, "map_value"]), 72.0)


class SplitArithmeticTests(unittest.TestCase):
    def test_all_60310_stays_split_exactly(self) -> None:
        train, tuning, assessment = split_cluster_indices(60310, 20260522, 0.60, 0.20)
        self.assertEqual(train.size, 36186)
        self.assertEqual(tuning.size, 12062)
        self.assertEqual(assessment.size, 12062)
        self.assertEqual(np.unique(np.concatenate([train, tuning, assessment])).size, 60310)

    def test_zero_fit_stays_includes_every_eligible_stay(self) -> None:
        stay_ids = np.arange(1000, 1020, dtype=np.int64)
        stays = pd.DataFrame(
            {
                "stay_id": stay_ids,
                "age": np.linspace(30.0, 70.0, stay_ids.size),
                "male": np.tile([0.0, 1.0], stay_ids.size // 2),
                "emergency_or_urgent": np.ones(stay_ids.size),
            }
        )
        rows = []
        for stay_id in stay_ids:
            for time_hours, map_value in zip([1.0, 4.0, 8.0, 12.0, 18.0], [70.0, 69.0, 68.0, 67.0, 66.0]):
                rows.append({"stay_id": stay_id, "time_hours": time_hours, "map_value": map_value})
        obs = pd.DataFrame(rows)
        _, summary, _ = build_dataset_from_cache(obs, stays, fit_stays=0, seed=20260522, analysis_hours=24.0)
        self.assertEqual(summary["fit_stays"], 20)
        self.assertEqual(summary["fit_stays_requested"], 0)

    def test_equal_stay_weights_sum_to_one_within_stay(self) -> None:
        design = pd.DataFrame({"stay_index": [0, 0, 1, 1, 1, 2]})
        weights = equal_stay_observation_weights(design)
        totals = pd.DataFrame({"stay_index": design["stay_index"], "weight": weights}).groupby("stay_index")[
            "weight"
        ].sum()
        np.testing.assert_allclose(totals.to_numpy(dtype=float), np.ones(3))


class DownstreamInterfaceTests(unittest.TestCase):
    def test_outcomes_attach_by_stay_index_for_deidentified_export(self) -> None:
        stay_frame = pd.DataFrame({"stay_index": [0, 1], "stay_id": [100, 200], "value": [1.0, 2.0]})
        exported_features = pd.DataFrame(
            {
                "stay_index": [0, 1],
                "hospital_mortality": [0.0, 1.0],
                "icu_los_days": [2.0, 4.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.csv"
            exported_features.to_csv(feature_path, index=False)
            merged = attach_outcomes(stay_frame, feature_path)
        self.assertListEqual(merged["stay_id"].tolist(), [100, 200])
        self.assertListEqual(merged["hospital_mortality"].tolist(), [0.0, 1.0])

    def test_fixed_capacity_q10_ties_use_stable_stay_index_order(self) -> None:
        frame = pd.DataFrame(
            {
                "stay_index": np.arange(10, dtype=int),
                "admission_window_q10": np.full(10, 60.0),
                "later_map_below65_fraction": np.linspace(0.10, 0.19, 10),
                "late_obs": np.full(10, 10),
                "any_later_map_below65": np.array([1.0, 1.0] + [0.0] * 8),
                "hospital_mortality": np.array([1.0, 1.0] + [0.0] * 8),
            }
        )
        _, ordered = clinical_yield_tables(frame)
        _, shuffled = clinical_yield_tables(frame.sample(frac=1.0, random_state=20260824))
        pd.testing.assert_frame_equal(ordered, shuffled)
        row_20 = ordered.loc[ordered["flagged_percent"] == 20].iloc[0]
        self.assertEqual(int(row_20["flagged_stays"]), 2)
        self.assertEqual(float(row_20["any_later_map_below65"]), 1.0)


if __name__ == "__main__":
    unittest.main()
