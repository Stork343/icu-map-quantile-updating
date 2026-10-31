from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_adaptive_mixed_simulation import (
    HORIZONS,
    SETTINGS,
    TAU,
    TYPE_NAMES,
    adaptive_candidate_specs,
    base_predictions,
    build_stays,
    candidate_specs,
    design_configuration,
    evaluate_matrix,
    fit_population,
    index_features,
    select_methods,
    simulate_dataset,
    split_indices,
    tuning_rules,
)


def empty_diagnostics() -> dict:
    return {"qp_fits": 0, "iterations_total": 0, "max_duality_gap": 0.0,
            "max_relative_duality_gap": 0.0, "max_projected_kkt_violation": 0.0,
            "max_stationarity_violation": 0.0}


class MixedSimulationTests(unittest.TestCase):
    def test_prespecified_record_geometry_and_seed(self):
        for setting in SETTINGS:
            data = simulate_dataset(setting, 90, 2045)
            replay = simulate_dataset(setting, 90, 2045)
            np.testing.assert_array_equal(data.later_y, replay.later_y)
            np.testing.assert_array_equal(data.latent_type, replay.latent_type)
            self.assertEqual(set(data.latent_type), set(range(len(TYPE_NAMES))))
            self.assertEqual(set(map(len, data.index_y)), {6, 10, 16})
            self.assertEqual(set(data.coverage), {4, 8, 12})
            for early_y, early_t in zip(data.index_y, data.index_time):
                self.assertTrue(np.all((early_y >= 20) & (early_y <= 200)))
                self.assertEqual(early_t[-1], 12)
                self.assertTrue(np.all(np.diff(early_t) > 0))
            self.assertTrue(np.all((data.later_y >= 20) & (data.later_y <= 200)))
            for horizon in HORIZONS:
                np.testing.assert_array_equal((data.later_time <= 12 + horizon).sum(axis=1),
                                              np.full(90, int(horizon)))
            if setting == "serial_integer_stress":
                np.testing.assert_array_equal(data.later_y, np.rint(data.later_y))

    def test_information_limit_coefficients_share_the_early_law(self):
        data = simulate_dataset("information_limit", 6000, 7821)
        for latent_type in range(3):
            mask = data.latent_type == latent_type
            self.assertLess(abs(data.latent_level[mask].mean()), 0.4)
            self.assertLess(abs(data.latent_slope[mask].mean()), 0.35)
            self.assertLess(abs(data.latent_level[mask].std() - 5.5), 0.4)
            self.assertLess(abs(data.latent_slope[mask].std() - 4), 0.35)

    def test_disjoint_stay_allocation(self):
        fit, tune, assessment = split_indices(600, 7123)
        self.assertEqual((len(fit), len(tune), len(assessment)), (360, 120, 120))
        self.assertEqual(len(set(fit) | set(tune) | set(assessment)), 600)
        self.assertEqual(len(set(fit) & set(tune)), 0)
        self.assertEqual(len(set(fit) & set(assessment)), 0)
        self.assertEqual(len(set(tune) & set(assessment)), 0)

    def test_candidate_grid_and_unbounded_forecast_scoring(self):
        self.assertEqual(len(candidate_specs()), 56)
        self.assertEqual(len(adaptive_candidate_specs()), 42)
        self.assertFalse(design_configuration()["forecast_clipping"])
        data = simulate_dataset("informative_recovery", 30, 184)
        stays = build_stays(data, np.arange(30), np.asarray([70, 0, 0, 0]))
        zero_penalty = next(spec for spec in candidate_specs() if spec["key"] == "scalar_0")
        direct = base_predictions(stays, zero_penalty, 12, {}, empty_diagnostics())
        exact = np.asarray([np.full(12, np.quantile(values, TAU, method="inverted_cdf")) for values in data.index_y])
        np.testing.assert_array_equal(direct, exact)
        prediction = np.full((30, 12), 250.0)
        result = evaluate_matrix(stays, prediction, 12)
        expected = np.mean((TAU - 1) * (data.later_y - prediction), axis=1)
        np.testing.assert_allclose(result["loss"], expected)
        np.testing.assert_array_equal(result["p_lt"], np.ones(30))

    def test_actual_r_fitting_and_future_separation(self):
        work = Path.cwd() / ".review-work" / "adaptive" / "simulation" / "public_test_artifacts" / "actual_r_fit"
        work.mkdir(parents=True, exist_ok=True)
        data = simulate_dataset("informative_recovery", 120, 2046)
        fit, tune, assessment = split_indices(120, 8309)
        beta = fit_population(data, fit, work)
        self.assertEqual(beta.shape, (4,))
        self.assertTrue(np.all(np.isfinite(beta)))
        stays = build_stays(data, assessment[:2], beta)
        spec = {**adaptive_candidate_specs()[10], "basis": "adaptive"}
        diagnostics = empty_diagnostics()
        prediction = base_predictions(stays, spec, 8, {}, diagnostics)
        initial = np.asarray([index_features(stay) for stay in stays])
        data.later_y[assessment[:2]] = 199
        changed = build_stays(data, assessment[:2], beta)
        replay = base_predictions(changed, spec, 8, {}, diagnostics)
        np.testing.assert_array_equal(initial, np.asarray([index_features(stay) for stay in changed]))
        np.testing.assert_allclose(prediction, replay, rtol=0, atol=1e-8)
        self.assertEqual(diagnostics["qp_fits"], 4)
        self.assertLessEqual(diagnostics["max_projected_kkt_violation"], 5e-7)

    def test_full_tuning_and_same_feature_stump_use_real_losses(self):
        data = simulate_dataset("serial_integer_stress", 120, 2047)
        fit, tune, assessment = split_indices(120, 92)
        stays = build_stays(data, tune, np.asarray([70, -2, 1, -1.8]))
        fitting = build_stays(data, fit, np.asarray([70, -2, 1, -1.8]))
        rules, losses, calibrations = tuning_rules(stays, 4, {}, empty_diagnostics())
        self.assertEqual(losses.shape, (24, 200))
        self.assertEqual(len(calibrations), 102)
        self.assertTrue(np.all(np.isfinite(losses)))
        selected, stump = select_methods(rules, losses, np.asarray([index_features(stay) for stay in fitting]),
                                         np.asarray([index_features(stay) for stay in stays]))
        self.assertEqual(len(stump["rule_names"]), 116)
        self.assertTrue(all(not name.startswith("adaptive") for name in stump["rule_names"]))
        self.assertAlmostEqual(selected["expanded_selected"]["tuning_loss"], float(losses.mean(axis=0).min()))
        self.assertLessEqual(selected["expanded_selected"]["tuning_loss"], selected["original_selected"]["tuning_loss"])
        self.assertLessEqual(stump["tuning_loss"], selected["original_selected"]["tuning_loss"] + 1e-10)


if __name__ == "__main__":
    unittest.main()
