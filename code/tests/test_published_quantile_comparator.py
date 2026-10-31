import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_published_quantile_comparator import ensure_matching_configuration, random_intercept_blp


class PublishedComparatorTests(unittest.TestCase):
    def test_resume_rejects_changed_configuration(self):
        root = Path(__file__).resolve().parents[3]
        directory = root / ".review-work/strengthening/published_comparator/test_configuration"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "configuration.json"
        path.write_text('{"seed":20260522,"tau":0.1,"index_hours":12.0,"nodes":321}', encoding="utf-8")
        self.assertTrue(ensure_matching_configuration(path, {"seed": 20260522, "tau": 0.1, "index_hours": 12.0, "nodes": 321}))
        for name, changed in (("seed", 42), ("tau", 0.2), ("index_hours", 6.0), ("nodes", 11)):
            configuration = {"seed": 20260522, "tau": 0.1, "index_hours": 12.0, "nodes": 321}
            configuration[name] = changed
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                ensure_matching_configuration(path, configuration)

    def test_blp_matches_dense_covariance_solution(self):
        residuals = np.asarray([3.0, -2.0, 7.0, 1.0, -4.0])
        variance, mean, error_variance = 2.3, 0.7, 6.1
        covariance = variance * np.ones((5, 5)) + error_variance * np.eye(5)
        expected = variance * np.ones(5) @ np.linalg.solve(covariance, residuals - mean)
        self.assertAlmostEqual(random_intercept_blp(residuals, variance, mean, error_variance), expected, places=12)

    def test_official_lqmm_fit_and_ranef(self):
        root = Path(__file__).resolve().parents[3]
        r_tmp = root / ".review-work/strengthening/published_comparator/test_r_tmp"
        r_tmp.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().parents[1] / "fit_published_quantile_comparator.R"
        result = subprocess.run(["Rscript", str(script), "--self-test"],
                                env=dict(os.environ, TMPDIR=os.path.relpath(r_tmp, Path.cwd())), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BLP max absolute difference", result.stdout)


if __name__ == "__main__":
    unittest.main()
