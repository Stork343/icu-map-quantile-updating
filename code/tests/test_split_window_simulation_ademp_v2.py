import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_split_window_simulation import (  # noqa: E402
    ADEMP_V2_SCENARIOS,
    _ar1_copula_errors,
    _informative_grid_indices,
    ademp_v2_scenario_diagnostics,
    calibration_interval_metrics,
    fit_affine_q10_calibration,
    profiled_level_slope,
    simulate_ademp_v2_dataset,
    split_cluster_indices,
)
from split_window_analysis_core import check_loss  # noqa: E402


class AdempV2SimulationTests(unittest.TestCase):
    def test_design_covers_declared_stressors_and_two_sample_sizes(self) -> None:
        self.assertGreaterEqual(len(ADEMP_V2_SCENARIOS), 14)
        sample_sizes = {int(config["n_stays"]) for config in ADEMP_V2_SCENARIOS.values()}
        self.assertGreaterEqual(len(sample_sizes), 2)
        self.assertTrue(any(float(config.get("serial_rho_15min", 0.0)) > 0 for config in ADEMP_V2_SCENARIOS.values()))
        self.assertTrue(any(config.get("sampling") == "informative_grid" for config in ADEMP_V2_SCENARIOS.values()))
        self.assertTrue(any(float(config.get("count_level_strength", 0.0)) > 0 for config in ADEMP_V2_SCENARIOS.values()))
        self.assertTrue(any(float(config.get("common_time_amplitude", 0.0)) > 0 for config in ADEMP_V2_SCENARIOS.values()))
        self.assertTrue(any(float(config.get("slope_sd", 0.0)) > 0 for config in ADEMP_V2_SCENARIOS.values()))
        self.assertTrue(any(float(config.get("treatment_gain", 0.0)) > 0 for config in ADEMP_V2_SCENARIOS.values()))
        self.assertEqual(float(ADEMP_V2_SCENARIOS["null_serial"]["level_sd"]), 0.0)
        self.assertGreater(float(ADEMP_V2_SCENARIOS["null_serial"]["serial_rho_15min"]), 0.0)
        self.assertEqual(float(ADEMP_V2_SCENARIOS["weak_level"]["level_sd"]), 2.5)
        self.assertEqual(ADEMP_V2_SCENARIOS["heavy_tail_t3"]["noise"], "student_t")
        self.assertEqual(float(ADEMP_V2_SCENARIOS["heavy_tail_t3"]["noise_df"]), 3.0)

    def test_ar1_copula_has_requested_marginal_quantile_and_dependence(self) -> None:
        errors = _ar1_copula_errors(
            30000,
            tau=0.10,
            rho=0.80,
            noise_kind="normal",
            noise_df=3.0,
            rng=np.random.default_rng(123),
        )
        self.assertAlmostEqual(float(np.mean(errors < 0.0)), 0.10, delta=0.015)
        self.assertGreater(float(np.corrcoef(errors[:-1], errors[1:])[0, 1]), 0.75)

        t_errors = _ar1_copula_errors(
            50000,
            tau=0.10,
            rho=0.0,
            noise_kind="student_t",
            noise_df=3.0,
            rng=np.random.default_rng(124),
        )
        self.assertAlmostEqual(float(np.mean(t_errors < 0.0)), 0.10, delta=0.01)

    def test_discrete_quantile_interval_does_not_penalize_ties(self) -> None:
        metrics = calibration_interval_metrics(
            outcomes=[np.array([0.0, 0.0, 1.0, 1.0])],
            predictions=[np.zeros(4)],
            tau=0.25,
        )
        self.assertEqual(metrics["calibration_p_lt"], 0.0)
        self.assertEqual(metrics["calibration_p_le"], 0.5)
        self.assertEqual(metrics["calibration_interval_violation"], 0.0)

    def test_recent_low_map_increases_sampling_opportunity(self) -> None:
        candidate = np.arange(48, dtype=int)
        low_counts = []
        high_counts = []
        for seed in range(200):
            low_counts.append(
                len(
                    _informative_grid_indices(
                        candidate,
                        np.full(48, 58.0),
                        base_probability=0.08,
                        count_range=(1, 30),
                        latent_level=0.0,
                        level_strength=0.0,
                        recent_strength=1.0,
                        rng=np.random.default_rng(seed),
                    )
                )
            )
            high_counts.append(
                len(
                    _informative_grid_indices(
                        candidate,
                        np.full(48, 82.0),
                        base_probability=0.08,
                        count_range=(1, 30),
                        latent_level=0.0,
                        level_strength=0.0,
                        recent_strength=1.0,
                        rng=np.random.default_rng(seed),
                    )
                )
            )
        self.assertGreater(float(np.mean(low_counts)), float(np.mean(high_counts)) + 2.0)

    def test_informative_cluster_size_is_realized(self) -> None:
        data = simulate_ademp_v2_dataset(
            300,
            tau=0.10,
            scenario=ADEMP_V2_SCENARIOS["cluster_size_informative"],
            rng=np.random.default_rng(20260824),
        )
        diagnostics = ademp_v2_scenario_diagnostics(data)
        self.assertLess(diagnostics["total_count_latent_level_correlation"], -0.75)

    def test_affine_calibration_is_lp_certified_and_no_worse_on_tuning(self) -> None:
        data = simulate_ademp_v2_dataset(
            90,
            tau=0.10,
            scenario=ADEMP_V2_SCENARIOS["ideal_large_dense"],
            rng=np.random.default_rng(44),
        )
        _, tuning, _ = split_cluster_indices(90, np.random.default_rng(45))
        fit = fit_affine_q10_calibration(data, tuning, tau=0.10)
        self.assertTrue(fit["success"])
        self.assertLessEqual(fit["tuning_loss"], fit["raw_identity_tuning_loss"] + 1e-9)
        self.assertLess(fit["max_equality_residual"], 1e-8)

    def test_level_slope_certified_solution_is_locally_optimal(self) -> None:
        residual = np.array([-7.0, -4.0, -2.0, 1.0, 3.0, 8.0])
        times = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        tau = 0.10
        lam = 0.7
        intercept, slope, diagnostics = profiled_level_slope(residual, times, tau, lam)
        basis = (times - 12.0) / 12.0

        def objective(b0: float, b1: float) -> float:
            return float(np.sum(check_loss(residual - b0 - b1 * basis, tau)) + lam * (b0 * b0 + b1 * b1))

        optimum = objective(intercept, slope)
        for delta0 in (-1e-4, 0.0, 1e-4):
            for delta1 in (-1e-4, 0.0, 1e-4):
                self.assertLessEqual(optimum, objective(intercept + delta0, slope + delta1) + 1e-9)
        self.assertTrue(diagnostics["converged"])


if __name__ == "__main__":
    unittest.main()
