import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_paired_bootstrap_validation import reference_coverage, replay_assessment
from run_risk_inference_simulation import METHODS, run_replicate as run_risk_replicate
from run_split_window_simulation import ADEMP_V2_SCENARIOS
from run_temporal_validation import paired_bootstrap_interval


class PairedBootstrapValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = Path.cwd() / ".review-work" / "cache" / "paired_bootstrap_tests"
        parent.mkdir(parents=True, exist_ok=True)
        cls.workspace = tempfile.TemporaryDirectory(prefix="generated-", dir=parent)
        cls.addClassCleanup(cls.workspace.cleanup)
        original_tmpdir = os.environ.get("TMPDIR")

        def restore_tmpdir():
            if original_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = original_tmpdir

        cls.addClassCleanup(restore_tmpdir)
        cls.inference_dir = Path(cls.workspace.name).resolve()
        configuration = {
            "out_dir": str(cls.inference_dir), "n_rep": 2, "reference_n": 600,
            "sample_sizes": [240], "seed": 20261015, "tau": 0.10,
            "common_solver": "quantreg::rq.fit(method='br'); primal/dual certificate checked",
        }
        for scenario in ADEMP_V2_SCENARIOS:
            run_risk_replicate(configuration, scenario, 0)
        run_risk_replicate(configuration, "discrete_map_rounding", 1)

    def test_replays_all_fourteen_generated_mechanisms(self):
        for scenario in ADEMP_V2_SCENARIOS:
            with self.subTest(scenario=scenario):
                losses, normal = replay_assessment(self.inference_dir, scenario, 1)
                self.assertEqual(losses.shape[1], len(METHODS))
                self.assertEqual(losses.shape[0], int(0.20 * ADEMP_V2_SCENARIOS[scenario]["n_stays"]))
                self.assertEqual(len(normal), 3)
                self.assertTrue(np.isfinite(losses).all())

    def test_existing_bootstrap_matches_direct_stay_resampling(self):
        losses, _ = replay_assessment(self.inference_dir, "discrete_map_rounding", 2)
        candidate, reference = losses[:, 1], losses[:, 0]
        result = paired_bootstrap_interval(candidate, reference, 2000, 20261016)
        difference = candidate - reference
        draw = np.random.default_rng(20261016).integers(0, difference.size, size=(2000, difference.size))
        means = difference[draw].mean(axis=1)
        np.testing.assert_allclose([result["ci95_low"], result["ci95_high"]], np.quantile(means, [0.025, 0.975]), atol=1e-12)

    def test_reference_sensitivity_preserves_the_nested_inclusion_events(self):
        actual = reference_coverage(-0.2, 0.3, 0.1, 0.01)
        self.assertTrue(actual["coverage_at_reference_estimate"])
        self.assertTrue(actual["coverage_reference_interval_contained"])
        self.assertTrue(actual["coverage_reference_interval_intersects"])
