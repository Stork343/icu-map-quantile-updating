import os
import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_risk_inference_simulation import (
    exact_profile_offsets,
    make_jobs,
    pack_records,
    paired_interval,
    rule_loss_matrix,
    select_penalty,
)
from run_split_window_simulation import (
    ADEMP_V2_SCENARIOS,
    evaluate_split,
    fit_affine_q10_calibration,
    grouped_quantile_fit,
    simulate_ademp_v2_dataset,
    training_design,
)
from split_window_analysis_core import profiled_intercept


def test_exact_profile_offsets_match_scalar_solver():
    rng = np.random.default_rng(20261015)
    counts = rng.integers(2, 23, size=100)
    values = np.full((len(counts), counts.max()), np.inf)
    for i, count in enumerate(counts):
        values[i, :count] = np.sort(np.round(rng.normal(0.0, 5.0, size=count), 0))
    for penalty in (0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0):
        expected = np.asarray([profiled_intercept(row[:count], 0.1, penalty) for row, count in zip(values, counts)])
        np.testing.assert_allclose(exact_profile_offsets(values, counts, 0.1, penalty), expected, atol=1e-12)


def test_real_grouped_fit_and_losses_match_existing_procedures():
    tmp_path = Path(__file__).resolve().parents[3] / ".review-work" / "strengthening" / "inference_simulation" / "test_runtime"
    tmp_path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.path.relpath(tmp_path, Path.cwd())
    data = simulate_ademp_v2_dataset(120, 0.1, ADEMP_V2_SCENARIOS["discrete_map_rounding"], np.random.default_rng(20261016))
    fit, tuning, evaluation = np.arange(72), np.arange(72, 96), np.arange(96, 120)
    design_path, coefficients_path = tmp_path / "design.csv", tmp_path / "coefficients.csv"
    training_design(data, fit, 0).to_csv(design_path, index=False)
    helper = Path(__file__).resolve().parents[1] / "fit_grouped_quantile_inference.R"
    coefficients = grouped_quantile_fit(design_path, coefficients_path, 0.1, helper)
    beta = coefficients.set_index("term").loc[["intercept", "age_z", "male", "urgent"], "estimate"].to_numpy()
    records = pack_records(data)
    penalty, _ = select_penalty(records.subset(tuning), beta, 0.1)
    calibration = fit_affine_q10_calibration(data, tuning, 0.1)
    actual = rule_loss_matrix(records.subset(evaluation), beta, penalty, calibration["intercept"], calibration["slope"], 0.1)
    expected_population, _ = evaluate_split(data, evaluation, beta, 0.1, None)
    expected_profile, _ = evaluate_split(data, evaluation, beta, 0.1, penalty)
    np.testing.assert_allclose(actual[:, 0], expected_population, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1], expected_profile, atol=1e-12)


def test_nested_fivefold_uses_disjoint_fitting_tuning_and_evaluation():
    jobs = make_jobs(600, 240, (240, 600), 20261017)
    for job in jobs:
        assert not np.intersect1d(job["fit"], job["tuning"]).size
        assert not np.intersect1d(job["fit"], job["evaluation"]).size
        assert not np.intersect1d(job["tuning"], job["evaluation"]).size
    for n_stays in (240, 600):
        evaluated = np.concatenate([job["evaluation"] for job in jobs if job["kind"] == "cross_fitted" and job["n_stays"] == n_stays])
        np.testing.assert_array_equal(np.sort(evaluated), np.arange(n_stays))


def test_paired_interval_uses_stay_variance_and_separate_reference_variance():
    data = simulate_ademp_v2_dataset(200, 0.1, ADEMP_V2_SCENARIOS["serial_dependence"], np.random.default_rng(20261018))
    beta = np.asarray([72.0, -2.0, 1.0, -1.8])
    population, _ = evaluate_split(data, np.arange(200), beta, 0.1, None)
    profile, _ = evaluate_split(data, np.arange(200), beta, 0.1, 0.3)
    differences = profile - population
    result = paired_interval(differences[:100], differences[100:])
    np.testing.assert_allclose(result["standard_error"], differences[:100].std(ddof=1) / 10.0)
    np.testing.assert_allclose(result["reference_mcse"], differences[100:].std(ddof=1) / 10.0)
    np.testing.assert_allclose(result["interval_length"], 3.92 * result["standard_error"])


class RiskInferenceSimulationTests(unittest.TestCase):
    test_exact_profile_offsets_match_scalar_solver = staticmethod(test_exact_profile_offsets_match_scalar_solver)
    test_real_grouped_fit_and_losses_match_existing_procedures = staticmethod(test_real_grouped_fit_and_losses_match_existing_procedures)
    test_nested_fivefold_uses_disjoint_fitting_tuning_and_evaluation = staticmethod(test_nested_fivefold_uses_disjoint_fitting_tuning_and_evaluation)
    test_paired_interval_uses_stay_variance_and_separate_reference_variance = staticmethod(test_paired_interval_uses_stay_variance_and_separate_reference_variance)
