from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))

from adaptive_profile_core import QP_SETTINGS, ProfileStay, check_loss, solve_spd_profile
from component_profile_learning import make_profile_batch, penalty_values


TAU = 0.1
STEPS = np.asarray([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5])


@dataclass
class Case:
    name: str
    residual: np.ndarray
    design: np.ndarray
    penalty: np.ndarray
    coefficient: np.ndarray
    dual: np.ndarray
    errors: np.ndarray


def case_from_kkt(name, x, dual, errors, penalty):
    design = np.column_stack([np.ones(len(x)), x])
    dual = np.asarray(dual, dtype=float)
    errors = np.asarray(errors, dtype=float)
    coefficient = 0.5 * np.linalg.solve(penalty, design.T @ dual)
    np.testing.assert_allclose(dual[errors > 0], TAU, atol=0, rtol=0)
    np.testing.assert_allclose(dual[errors < 0], TAU - 1.0, atol=0, rtol=0)
    assert np.all((dual >= TAU - 1.0) & (dual <= TAU))
    return Case(name, design @ coefficient + errors, design, penalty,
                coefficient, dual, errors)


def cone_matrices(case):
    equality, inequality = [], []
    for row, dual, error in zip(case.design, case.dual, case.errors):
        if error != 0:
            continue
        if dual == TAU:
            inequality.append(row)
        elif dual == TAU - 1.0:
            inequality.append(-row)
        else:
            equality.append(row)
    return np.asarray(equality).reshape(-1, 2), np.asarray(inequality).reshape(-1, 2)


def directional_profile(case, perturbation):
    equality, inequality = cone_matrices(case)
    matrix = np.vstack([equality, inequality])
    cones = []
    if len(equality):
        cones.append(clarabel.ZeroConeT(len(equality)))
    if len(inequality):
        cones.append(clarabel.NonnegativeConeT(len(inequality)))
    settings = clarabel.DefaultSettings()
    for name, value in QP_SETTINGS.items():
        setattr(settings, name, value)
    solver = clarabel.DefaultSolver(
        sparse.csc_matrix(2 * case.penalty), 2 * perturbation @ case.coefficient,
        sparse.csc_matrix(matrix), np.zeros(len(matrix)), cones, settings,
    )
    result = solver.solve()
    assert str(result.status) in {"Solved", "AlmostSolved"}, str(result.status)
    assert result.x is not None
    direction = np.asarray(result.x)
    assert not len(equality) or np.max(np.abs(equality @ direction)) < 1e-8
    assert not len(inequality) or np.max(inequality @ direction) < 1e-8
    unconstrained = -np.linalg.solve(case.penalty, perturbation @ case.coefficient)
    if not len(matrix):
        np.testing.assert_allclose(direction, unconstrained, atol=1e-10, rtol=1e-10)
    if len(equality) and np.linalg.matrix_rank(equality) == 2:
        np.testing.assert_allclose(direction, 0, atol=1e-10, rtol=0)
    if len(equality) == 1 and not len(inequality):
        normal = equality[0]
        inverse_normal = np.linalg.solve(case.penalty, normal)
        expected = unconstrained - inverse_normal * (normal @ unconstrained) / (normal @ inverse_normal)
        np.testing.assert_allclose(direction, expected, atol=1e-10, rtol=1e-10)
    if len(inequality) == 1 and not len(equality):
        normal = inequality[0]
        expected = unconstrained.copy()
        if normal @ expected > 0:
            inverse_normal = np.linalg.solve(case.penalty, normal)
            expected -= inverse_normal * (normal @ expected) / (normal @ inverse_normal)
        np.testing.assert_allclose(direction, expected, atol=3e-7, rtol=1e-8)
    return direction


def certified_profile(case, penalty):
    coefficient, diagnostic = solve_spd_profile(case.residual, case.design, TAU, penalty)
    assert diagnostic["success"]
    return coefficient, diagnostic


def fixed_cases(penalty):
    x = np.asarray([-1.0, -0.75, -0.25, 0.0])
    definitions = [
        ("nonzero_residuals", [.1, -.9, .1, -.9], [.8, -.7, 1.1, -.4]),
        ("interior_tie", [-.4, .1, -.9, .1], [0, .8, -.7, .9]),
        ("upper_tie", [.1, -.9, .1, -.9], [0, -.8, .7, -.9]),
        ("lower_tie", [-.9, .1, -.9, .1], [0, .8, -.7, .9]),
        ("two_interior_ties", [-.4, .1, -.9, -.3], [0, .8, -.7, 0]),
        ("mixed_boundary_ties", [.1, -.9, .1, -.9], [0, -.8, .7, 0]),
        ("all_upper_ties", [.1, .1, .1, .1], [0, 0, 0, 0]),
        ("all_interior_ties", [-.4, -.2, -.6, -.3], [0, 0, 0, 0]),
    ]
    return [case_from_kkt(name, x, dual, errors, penalty)
            for name, dual, errors in definitions]


def random_cases(generator, count):
    result = []
    for index in range(count):
        size = int(generator.integers(4, 13))
        x = np.sort(generator.uniform(-1, 0, size))
        positive = generator.uniform(size=size) > .5
        dual = np.where(positive, TAU, TAU - 1)
        errors = np.where(positive, 1, -1) * generator.uniform(.4, 1.5, size)
        matrix = generator.normal(size=(2, 2))
        penalty = matrix.T @ matrix + 0.75 * np.eye(2)
        result.append(case_from_kkt(f"generic_{index:03d}", x, dual, errors, penalty))
    return result


def verify_profile_cases(generator):
    cases = fixed_cases(np.asarray([[1.3, .25], [.25, .9]])) + random_cases(generator, 64)
    rows, maxima = [], {"profile_base_error": 0.0, "smallest_step_derivative_error": 0.0,
                        "objective_expansion_error": 0.0, "certificate_gap": 0.0}
    profile_solves = 0
    for case in cases:
        fitted, diagnostic = certified_profile(case, case.penalty)
        profile_solves += 1
        error = float(np.max(np.abs(fitted - case.coefficient)))
        np.testing.assert_allclose(fitted, case.coefficient, atol=2e-9, rtol=2e-10)
        maxima["profile_base_error"] = max(maxima["profile_base_error"], error)
        matrix = generator.normal(size=(2, 2))
        for sign in [-1, 1]:
            perturbation = sign * (matrix + matrix.T) / 4
            direction = directional_profile(case, perturbation)
            derivative_errors = []
            for step in STEPS:
                updated, diagnostic = certified_profile(case, case.penalty + step * perturbation)
                profile_solves += 1
                difference = (updated - case.coefficient) / step
                derivative_errors.append(float(np.linalg.norm(difference - direction)))
                maxima["certificate_gap"] = max(maxima["certificate_gap"], abs(diagnostic["duality_gap"]))
            assert derivative_errors[-1] < 1e-4, (case.name, derivative_errors)
            maxima["smallest_step_derivative_error"] = max(
                maxima["smallest_step_derivative_error"], derivative_errors[-1])
            for test_direction in generator.normal(size=(8, 2)):
                step = 1e-4
                proposed = case.coefficient + step * test_direction
                index_derivative = float(np.sum(
                    np.where(case.errors != 0,
                             ((case.errors < 0).astype(float) - TAU) * (case.design @ test_direction),
                             check_loss(-case.design @ test_direction, TAU))))
                first = index_derivative + 2 * case.coefficient @ case.penalty @ test_direction
                second = test_direction @ case.penalty @ test_direction + 2 * case.coefficient @ perturbation @ test_direction
                objective = lambda value: float(np.sum(check_loss(case.residual - case.design @ value, TAU))
                    + value @ (case.penalty + step * perturbation) @ value)
                measured = objective(proposed) - objective(case.coefficient)
                expected = step * first + step**2 * second + step**3 * test_direction @ perturbation @ test_direction
                error = abs(measured - expected)
                assert error < 2e-12, (case.name, error)
                maxima["objective_expansion_error"] = max(maxima["objective_expansion_error"], error)
            rows.append({"case": case.name, "direction_sign": sign,
                         "n_index": len(case.residual), "n_ties": int(np.sum(case.errors == 0)),
                         "direction_norm": float(np.linalg.norm(direction)),
                         **{f"derivative_error_{step:g}": value for step, value in zip(STEPS, derivative_errors)}})
    return rows, maxima, profile_solves


def verify_future_ties():
    rows = []
    for residual in [-.8, 0.0, .7]:
        for direction in [-1.4, -.3, 0.0, .4, 1.7]:
            expected = ((residual < 0) - TAU) * direction + (max(direction, 0) if residual == 0 else 0)
            step = 1e-5
            actual = float((check_loss(np.asarray(residual - step * direction), TAU)
                            - check_loss(np.asarray(residual), TAU)) / step)
            assert abs(actual - expected) < 2e-11
            rows.append({"residual": residual, "forecast_direction": direction,
                         "expected": expected, "difference_quotient": actual})
    return rows


def verify_history_path():
    params = np.zeros((2, 6))
    params[0, :3] = -1.0
    params[1, :3] = -0.6
    base_feature = np.asarray([[1.0, 0, 0, 0, 0, 0]])
    base_penalty, log_derivative = penalty_values(params, base_feature)
    penalty = np.diag(base_penalty[0])
    cases = fixed_cases(penalty)
    for size, span in [(6, .25), (8, .5), (12, .75), (16, 1.0)]:
        x = np.linspace(-span, 0, size)
        positive = np.arange(size) % 3 != 0
        dual = np.where(positive, TAU, TAU - 1)
        errors = np.where(positive, .8, -.6)
        cases.append(case_from_kkt(f"history_count_{size}", x, dual, errors, penalty))
    later_times = np.asarray([14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    later_design = np.column_stack([np.ones(6), (later_times - 12) / 12])
    stays = []
    for index, case in enumerate(cases):
        later_errors = np.asarray([-.6, 0, .8, 0, -.4, .5])
        if index % 2:
            later_errors = -later_errors
        stays.append(ProfileStay(case.residual, 12 + 12 * case.design[:, 1],
                                 later_design @ case.coefficient + later_errors,
                                 later_times, np.zeros(6)))
    batch = make_profile_batch(stays, 12.0)
    histories = batch.features[:, 3:]
    means = histories.mean(axis=0)
    b = np.asarray([[.5, -.7, .3], [-.4, .2, .6]])
    path_direction = np.zeros((2, 6))
    path_direction[:, :3] = -(b @ means)[:, None]
    path_direction[:, 3:] = b
    derivatives = []
    profile_solves = 0
    base_losses = []
    for horizon in [4.0, 8.0, 12.0]:
        current_batch = make_profile_batch(stays, horizon)
        values, derivatives_log = penalty_values(params, current_batch.features)
        change = (histories - means) @ b.T
        np.testing.assert_allclose(change.mean(axis=0), 0, atol=1e-14, rtol=0)
        horizon_scores, horizon_losses = [], []
        for index, (case, stay) in enumerate(zip(cases, stays)):
            np.testing.assert_allclose(values[index], np.diag(penalty), atol=1e-14, rtol=0)
            perturbation = np.diag(values[index] * derivatives_log[index] * change[index])
            direction = directional_profile(case, perturbation)
            selected = later_times <= 12 + horizon
            forecast_direction = later_design[selected] @ direction
            residual = stay.later_y[selected] - later_design[selected] @ case.coefficient
            ties = np.isclose(residual, 0.0, atol=1e-14, rtol=0)
            residual[ties] = 0.0
            scores = ((residual < 0).astype(float) - TAU) * forecast_direction + ties * np.maximum(forecast_direction, 0)
            horizon_scores.append(float(scores.mean()))
            horizon_losses.append(float(check_loss(residual, TAU).mean()))
        derivatives.append(float(np.mean(horizon_scores)))
        base_losses.append(float(np.mean(horizon_losses)))
    gamma = float(np.mean(derivatives))
    base_risk = float(np.mean(base_losses))
    rows = []
    for step in STEPS:
        next_params = params + step * path_direction
        assert np.max(np.abs(next_params[:, :3])) < 12
        assert np.max(np.abs(next_params[:, 3:])) < 4
        horizon_losses = []
        for horizon in [4.0, 8.0, 12.0]:
            current_batch = make_profile_batch(stays, horizon)
            values, _ = penalty_values(next_params, current_batch.features)
            record_losses = []
            for case, stay, current in zip(cases, stays, values):
                coefficient, _ = certified_profile(case, np.diag(current))
                profile_solves += 1
                selected = later_times <= 12 + horizon
                record_losses.append(float(check_loss(stay.later_y[selected] - later_design[selected] @ coefficient, TAU).mean()))
            horizon_losses.append(float(np.mean(record_losses)))
        difference = (float(np.mean(horizon_losses)) - base_risk) / step
        rows.append({"step": float(step), "gamma": gamma, "difference_quotient": difference,
                     "absolute_error": abs(difference - gamma)})
    assert rows[-1]["absolute_error"] < 1e-5, rows
    descent_stays, descent_directions = [], []
    repeated_times = np.repeat(later_times, 11)
    repeated_design = np.repeat(later_design, 11, axis=0)
    for index, (case, stay) in enumerate(zip(cases, stays)):
        change = b @ (histories[index] - means)
        perturbation = np.diag(base_penalty[0] * log_derivative[0] * change)
        direction = directional_profile(case, perturbation)
        forecast_direction = repeated_design @ direction
        later_errors = np.where(forecast_direction >= 0, 1.0, -1.0)
        later_errors[::11] = 0
        descent_stays.append(ProfileStay(case.residual, stay.index_time,
            repeated_design @ case.coefficient + later_errors, repeated_times, np.zeros(len(repeated_times))))
        descent_directions.append((forecast_direction, later_errors))
    descent_scores, descent_base = [], []
    for horizon in [4, 8, 12]:
        selected = repeated_times <= 12 + horizon
        for forecast_direction, errors in descent_directions:
            a, e = forecast_direction[selected], errors[selected]
            score = ((e < 0).astype(float) - TAU) * a + (e == 0) * np.maximum(a, 0)
            descent_scores.append(float(score.mean()))
            descent_base.append(float(check_loss(e, TAU).mean()))
    descent_gamma, descent_risk = float(np.mean(descent_scores)), float(np.mean(descent_base))
    assert descent_gamma < 0
    kappa = -descent_gamma / 2
    descent_rows = []
    for step in STEPS:
        next_params = params + step * path_direction
        losses = []
        for horizon in [4, 8, 12]:
            current_batch = make_profile_batch(descent_stays, horizon)
            values, _ = penalty_values(next_params, current_batch.features)
            selected = repeated_times <= 12 + horizon
            for case, stay, current in zip(cases, descent_stays, values):
                coefficient, _ = certified_profile(case, np.diag(current))
                profile_solves += 1
                losses.append(float(check_loss(stay.later_y[selected] - repeated_design[selected] @ coefficient, TAU).mean()))
        risk = float(np.mean(losses))
        difference = (risk - descent_risk) / step
        assert risk <= descent_risk - kappa * step + 1e-14
        descent_rows.append({"step": float(step), "gamma": descent_gamma,
            "difference_quotient": difference, "absolute_error": abs(difference - descent_gamma),
            "descent_margin": descent_risk - kappa * step - risk})
    assert descent_rows[-1]["absolute_error"] < 1e-5, descent_rows
    return {"n_stays": len(stays), "horizons": [4, 8, 12], "n_later_ties_per_stay": 2,
            "history_counts": [len(case.residual) for case in cases],
            "history_feature_ranges": np.ptp(histories, axis=0).tolist(),
            "base_risk": base_risk, "gamma": gamma, "rows": rows,
            "negative_direction": {"gamma": descent_gamma, "kappa": kappa,
                "n_later_ties_per_stay": 6, "rows": descent_rows}}, profile_solves


def main():
    parser = argparse.ArgumentParser(description="Verify directional quantile formulas on synthetic mathematical examples")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "analysis_continuous_constant")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(20261031)
    profile_rows, maxima, count = verify_profile_cases(generator)
    tie_rows = verify_future_ties()
    history, history_count = verify_history_path()
    with (args.output_dir / "nonsmooth_profile_direction_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    report = {"status": "passed", "study_role": "Theorem verification on synthetic mathematical examples, separate from the 800 replicate simulation study",
              "seed": 20261031, "fixed_index_cases": 8,
              "random_generic_cases": 64, "directional_profile_comparisons": len(profile_rows),
              "objective_expansion_checks": len(profile_rows) * 8,
              "production_profile_solves": count + history_count,
              "future_tie_derivative_checks": tie_rows, "maxima": maxima,
              "history_path": history,
              "scope": "Finite numerical checks support the stated formulas for these mathematical examples; theorem validity is assessed by proof review."}
    (args.output_dir / "nonsmooth_gain_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
