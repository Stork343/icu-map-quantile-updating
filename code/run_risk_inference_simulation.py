from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from run_split_window_simulation import (
    ADEMP_V2_SCENARIOS,
    PREDICTOR_COLS,
    AdempV2Dataset,
    fit_affine_q10_calibration,
    grouped_quantile_fit,
    simulate_ademp_v2_dataset,
    split_cluster_indices,
    training_design,
)


METHODS = ("population", "tuned_level", "affine_calibrated_q10")
CONTRASTS = (
    ("tuned_level", "population"),
    ("affine_calibrated_q10", "population"),
    ("tuned_level", "affine_calibrated_q10"),
)
LAMBDA_GRID = (0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)


@dataclass
class PackedRecords:
    x: np.ndarray
    index_ordered: np.ndarray
    index_counts: np.ndarray
    later_values: np.ndarray
    later_counts: np.ndarray

    def subset(self, indices: np.ndarray) -> PackedRecords:
        return PackedRecords(
            self.x[indices],
            self.index_ordered[indices],
            self.index_counts[indices],
            self.later_values[indices],
            self.later_counts[indices],
        )


def pack_records(data: AdempV2Dataset) -> PackedRecords:
    index_values = [y[t <= 12.0] for y, t in zip(data.y_list, data.t_list)]
    later_values = [y[t > 12.0] for y, t in zip(data.y_list, data.t_list)]
    index_counts = np.asarray([len(y) for y in index_values], dtype=int)
    later_counts = np.asarray([len(y) for y in later_values], dtype=int)
    if np.any(index_counts == 0) or np.any(later_counts == 0):
        raise ValueError("Every stay requires index and later records")
    index_matrix = np.full((len(index_values), int(index_counts.max())), np.inf)
    later_matrix = np.full((len(later_values), int(later_counts.max())), np.nan)
    for i, (early, later) in enumerate(zip(index_values, later_values)):
        index_matrix[i, : early.size] = np.sort(early)
        later_matrix[i, : later.size] = later
    return PackedRecords(data.x, index_matrix, index_counts, later_matrix, later_counts)


def exact_profile_offsets(
    ordered_residuals: np.ndarray,
    counts: np.ndarray,
    tau: float,
    penalty: float,
) -> np.ndarray:
    if not 0.0 < tau < 1.0 or penalty < 0.0:
        raise ValueError("Invalid quantile or penalty")
    required = np.arange(ordered_residuals.shape[1])[None, :] < counts[:, None]
    if not np.all(np.isfinite(ordered_residuals[required])):
        raise ValueError("Invalid residual values")
    if penalty == 0.0:
        ranks = np.ceil(tau * counts).astype(int) - 1
        return ordered_residuals[np.arange(len(counts)), ranks]
    width = ordered_residuals.shape[1]
    answer = np.full(len(counts), np.nan)
    for k in range(width + 1):
        candidate = (tau * counts - k) / (2.0 * penalty)
        lower = ordered_residuals[:, k - 1] if k > 0 else -np.inf
        upper = ordered_residuals[:, k] if k < width else np.inf
        valid = (k <= counts) & (lower < candidate) & (candidate < upper)
        answer[valid] = candidate[valid]
    # 重复取值共享相同最优解，逐个位置检查subgradient覆盖即可。
    for j in range(width):
        knot = ordered_residuals[:, j]
        left = j - tau * counts + 2.0 * penalty * knot
        right = left + 1.0
        tolerance = 1e-12 * np.maximum(1.0, counts)
        valid = (j < counts) & (left <= tolerance) & (right >= -tolerance)
        answer[valid] = knot[valid]
    if not np.all(np.isfinite(answer)):
        raise RuntimeError("Exact profile minimizer was not located for every stay")
    return answer


def stay_check_losses(records: PackedRecords, predictions: np.ndarray, tau: float) -> np.ndarray:
    residuals = records.later_values - predictions[:, None]
    losses = np.where(residuals >= 0.0, tau * residuals, (tau - 1.0) * residuals)
    return np.nansum(losses, axis=1) / records.later_counts


def rule_loss_matrix(
    records: PackedRecords,
    beta: np.ndarray,
    penalty: float,
    affine_intercept: float,
    affine_slope: float,
    tau: float,
) -> np.ndarray:
    population = records.x @ beta
    offsets = exact_profile_offsets(
        records.index_ordered - population[:, None], records.index_counts, tau, penalty
    )
    ranks = np.ceil(tau * records.index_counts).astype(int) - 1
    raw_q10 = records.index_ordered[np.arange(len(ranks)), ranks]
    predictions = (population, population + offsets, affine_intercept + affine_slope * raw_q10)
    return np.column_stack([stay_check_losses(records, values, tau) for values in predictions])


def select_penalty(records: PackedRecords, beta: np.ndarray, tau: float) -> tuple[float, float]:
    population = records.x @ beta
    ordered_residuals = records.index_ordered - population[:, None]
    risks = []
    for penalty in LAMBDA_GRID:
        offsets = exact_profile_offsets(ordered_residuals, records.index_counts, tau, penalty)
        risks.append(float(np.mean(stay_check_losses(records, population + offsets, tau))))
    chosen = int(np.argmin(risks))
    return LAMBDA_GRID[chosen], risks[chosen]


def paired_interval(assessment: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if assessment.size < 2 or reference.size < 2:
        raise ValueError("Risk inference requires at least two independent stays")
    estimate = float(np.mean(assessment))
    standard_error = float(np.std(assessment, ddof=1) / np.sqrt(assessment.size))
    lower, upper = estimate - 1.96 * standard_error, estimate + 1.96 * standard_error
    target = float(np.mean(reference))
    reference_mcse = float(np.std(reference, ddof=1) / np.sqrt(reference.size))
    reference_lower = target - 1.96 * reference_mcse
    reference_upper = target + 1.96 * reference_mcse
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_length": upper - lower,
        "reference_risk_difference": target,
        "reference_mcse": reference_mcse,
        "reference_mcse_over_evaluation_se": reference_mcse / standard_error if standard_error > 0.0 else 0.0,
        "coverage_at_reference_estimate": bool(lower <= target <= upper),
        "coverage_reference_interval_contained": bool(lower <= reference_lower and reference_upper <= upper),
        "coverage_reference_interval_intersects": bool(lower <= reference_upper and reference_lower <= upper),
        "estimation_error": estimate - target,
    }


def make_jobs(n_available: int, original_n: int, sample_sizes: tuple[int, ...], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    fit, tuning, evaluation = split_cluster_indices(original_n, rng)
    jobs = [{"group_id": 0, "kind": "independent", "n_stays": original_n,
             "fold": -1, "fit": fit, "tuning": tuning, "evaluation": evaluation}]
    for n_stays in sample_sizes:
        if n_stays > n_available or n_stays < 30:
            raise ValueError("Invalid cross-fitting sample size")
        permutation = rng.permutation(n_stays)
        folds = np.array_split(permutation, 5)
        for fold, evaluation in enumerate(folds):
            eligible = np.setdiff1d(np.arange(n_stays), evaluation)
            fit_tuning = rng.permutation(eligible)
            n_fit = int(round(0.75 * eligible.size))
            jobs.append({"group_id": len(jobs), "kind": "cross_fitted", "n_stays": n_stays,
                         "fold": fold, "fit": np.sort(fit_tuning[:n_fit]),
                         "tuning": np.sort(fit_tuning[n_fit:]), "evaluation": np.sort(evaluation)})
    return jobs


def run_replicate(config: dict, scenario_key: str, rep_zero: int) -> None:
    scenario_position = list(ADEMP_V2_SCENARIOS).index(scenario_key)
    group_id = scenario_position * 1_000_000 + rep_zero
    seed = int(config["seed"])
    data_seed = seed + 1009 * group_id
    split_seed = seed + 7919 * group_id + 17
    reference_seed = seed + 1009 * group_id + 80_000_000_000
    scenario = ADEMP_V2_SCENARIOS[scenario_key]
    sample_sizes = tuple(config["sample_sizes"])
    output = Path(config["out_dir"]) / "replicates" / scenario_key / f"rep_{rep_zero + 1:04d}"
    marker = output / "complete.json"
    if marker.exists():
        completed = json.loads(marker.read_text())
        if completed["configuration"] != config:
            raise ValueError(f"Checkpoint configuration mismatch: {marker}")
        return
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    r_temporary = work / "r_temporary"
    r_temporary.mkdir(parents=True, exist_ok=True)
    n_available = max(int(scenario["n_stays"]), max(sample_sizes))
    data = simulate_ademp_v2_dataset(n_available, config["tau"], scenario, np.random.default_rng(data_seed))
    records = pack_records(data)
    jobs = make_jobs(n_available, int(scenario["n_stays"]), sample_sizes, split_seed)
    design_path, coefficients_path = work / "training_design.csv", work / "coefficients.csv"
    pd.concat([training_design(data, job["fit"], job["group_id"]) for job in jobs], ignore_index=True).to_csv(design_path, index=False)
    os.environ["TMPDIR"] = os.path.relpath(r_temporary.resolve(), Path.cwd())
    helper = Path(__file__).with_name("fit_grouped_quantile_inference.R")
    coefficients = grouped_quantile_fit(design_path, coefficients_path, config["tau"], helper)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    wide = coefficients.pivot(index="group_id", columns="term", values="estimate")
    frozen_rows = []
    rules = {}
    for job in jobs:
        beta = wide.loc[job["group_id"], PREDICTOR_COLS].to_numpy(dtype=float)
        penalty, tuning_risk = select_penalty(records.subset(job["tuning"]), beta, config["tau"])
        calibration = fit_affine_q10_calibration(data, job["tuning"], config["tau"])
        rules[job["group_id"]] = (beta, penalty, float(calibration["intercept"]), float(calibration["slope"]))
        frozen_rows.append({"group_id": job["group_id"], "kind": job["kind"], "n_stays": job["n_stays"],
                            "fold": job["fold"], "n_fitting": len(job["fit"]), "n_tuning": len(job["tuning"]),
                            "n_evaluation": len(job["evaluation"]), "selected_penalty": penalty,
                            "profile_tuning_loss": tuning_risk, **calibration})
    del data
    reference_data = simulate_ademp_v2_dataset(config["reference_n"], config["tau"], scenario, np.random.default_rng(reference_seed))
    reference = pack_records(reference_data)
    del reference_data
    independent_rows, cf_rows = [], []
    cf_evaluation = {n: np.full((n, len(METHODS)), np.nan) for n in sample_sizes}
    cf_reference = {n: np.zeros((config["reference_n"], len(METHODS))) for n in sample_sizes}
    cf_fold_rows = []
    for job in jobs:
        beta, penalty, affine_intercept, affine_slope = rules[job["group_id"]]
        evaluation_losses = rule_loss_matrix(records.subset(job["evaluation"]), beta, penalty, affine_intercept, affine_slope, config["tau"])
        reference_losses = rule_loss_matrix(reference, beta, penalty, affine_intercept, affine_slope, config["tau"])
        if job["kind"] == "independent":
            for first, second in CONTRASTS:
                j, k = METHODS.index(first), METHODS.index(second)
                independent_rows.append({"scenario_key": scenario_key, "rep": rep_zero + 1,
                                         "n_stays": job["n_stays"], "n_evaluation": len(job["evaluation"]),
                                         "reference_n": config["reference_n"], "contrast": f"{first}_minus_{second}",
                                         **paired_interval(evaluation_losses[:, j] - evaluation_losses[:, k], reference_losses[:, j] - reference_losses[:, k])})
        else:
            n = job["n_stays"]
            cf_evaluation[n][job["evaluation"]] = evaluation_losses
            cf_reference[n] += len(job["evaluation"]) / n * reference_losses
            for j, method in enumerate(METHODS):
                cf_fold_rows.append({"scenario_key": scenario_key, "rep": rep_zero + 1, "n_stays": n,
                                     "fold": job["fold"], "method": method,
                                     "reference_risk": float(reference_losses[:, j].mean())})
    for n in sample_sizes:
        if not np.all(np.isfinite(cf_evaluation[n])):
            raise RuntimeError("Cross-fitting predictions do not cover every stay")
        for j, method in enumerate(METHODS):
            estimate = float(cf_evaluation[n][:, j].mean())
            target = float(cf_reference[n][:, j].mean())
            reference_mcse = float(cf_reference[n][:, j].std(ddof=1) / np.sqrt(config["reference_n"]))
            cf_rows.append({"scenario_key": scenario_key, "rep": rep_zero + 1, "n_stays": n,
                            "method": method, "reference_n": config["reference_n"], "oof_risk": estimate,
                            "fold_rule_reference_risk": target, "estimation_error": estimate - target,
                            "reference_mcse": reference_mcse})
    pd.DataFrame(independent_rows).to_csv(output / "independent_intervals.csv", index=False)
    pd.DataFrame(cf_rows).to_csv(output / "cross_fitted_risk.csv", index=False)
    pd.DataFrame(cf_fold_rows).to_csv(output / "cross_fitted_fold_risk.csv", index=False)
    pd.DataFrame(frozen_rows).to_csv(output / "frozen_rules.csv", index=False)
    design_path.unlink()
    coefficients_path.unlink()
    marker.write_text(json.dumps({"configuration": config, "data_seed": data_seed, "split_seed": split_seed,
                                  "reference_seed": reference_seed}, indent=2) + "\n")


def summarize(config: dict, scenarios: tuple[str, ...]) -> None:
    root = Path(config["out_dir"])
    for scenario in scenarios:
        for rep in range(config["n_rep"]):
            marker = root / "replicates" / scenario / f"rep_{rep + 1:04d}" / "complete.json"
            if json.loads(marker.read_text())["configuration"] != config:
                raise RuntimeError(f"Checkpoint configuration mismatch: {marker}")
    frames = {}
    for filename in ("independent_intervals", "cross_fitted_risk", "cross_fitted_fold_risk", "frozen_rules"):
        files = [root / "replicates" / scenario / f"rep_{rep + 1:04d}" / f"{filename}.csv"
                 for scenario in scenarios for rep in range(config["n_rep"])]
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        frame.to_csv(root / f"{filename}.csv", index=False)
        frames[filename] = frame
    coverage_rows = []
    for (scenario, contrast), local in frames["independent_intervals"].groupby(["scenario_key", "contrast"], sort=False):
        coverage = float(local["coverage_at_reference_estimate"].mean())
        coverage_rows.append({"scenario_key": scenario, "contrast": contrast, "n_rep": len(local),
                              "coverage": coverage, "coverage_mcse": np.sqrt(coverage * (1.0 - coverage) / len(local)),
                              "coverage_reference_interval_contained": float(local["coverage_reference_interval_contained"].mean()),
                              "coverage_reference_interval_intersects": float(local["coverage_reference_interval_intersects"].mean()),
                              "mean_interval_length": float(local["interval_length"].mean()),
                              "interval_length_mean_mcse": float(local["interval_length"].std(ddof=1) / np.sqrt(len(local))),
                              "mean_reference_mcse": float(local["reference_mcse"].mean()),
                              "mean_reference_mcse_over_evaluation_se": float(local["reference_mcse_over_evaluation_se"].mean()),
                              "mean_estimation_error": float(local["estimation_error"].mean())})
    pd.DataFrame(coverage_rows).to_csv(root / "independent_coverage_summary.csv", index=False)
    cf_rows = []
    for (scenario, method, n), local in frames["cross_fitted_risk"].groupby(["scenario_key", "method", "n_stays"], sort=False):
        error = local["estimation_error"].to_numpy()
        squared_error = np.square(error)
        cf_rows.append({"scenario_key": scenario, "method": method, "n_stays": n, "n_rep": len(local),
                        "mean_oof_risk": float(local["oof_risk"].mean()),
                        "mean_fold_rule_reference_risk": float(local["fold_rule_reference_risk"].mean()),
                        "mean_error": float(error.mean()), "error_mean_mcse": float(error.std(ddof=1) / np.sqrt(len(error))),
                        "mean_absolute_error": float(np.abs(error).mean()),
                        "absolute_error_mean_mcse": float(np.abs(error).std(ddof=1) / np.sqrt(len(error))),
                        "rmse": float(np.sqrt(squared_error.mean())),
                        "mean_reference_mcse": float(local["reference_mcse"].mean()),
                        "reference_noise_corrected_rmse": float(np.sqrt(max(0.0, squared_error.mean() - np.square(local["reference_mcse"]).mean())))})
    pd.DataFrame(cf_rows).to_csv(root / "cross_fitted_consistency_summary.csv", index=False)
    sample_size_rows = []
    minimum_n, maximum_n = min(config["sample_sizes"]), max(config["sample_sizes"])
    for (scenario, method), local in frames["cross_fitted_risk"].groupby(["scenario_key", "method"], sort=False):
        wide_errors = local.pivot(index="rep", columns="n_stays", values="estimation_error")
        small_error, large_error = wide_errors[minimum_n].to_numpy(), wide_errors[maximum_n].to_numpy()
        absolute_difference = np.abs(large_error) - np.abs(small_error)
        mean_difference = float(absolute_difference.mean())
        mcse = float(absolute_difference.std(ddof=1) / np.sqrt(len(absolute_difference)))
        sample_size_rows.append({"scenario_key": scenario, "method": method, "minimum_n": minimum_n,
                                 "maximum_n": maximum_n, "n_rep": len(absolute_difference),
                                 "mean_absolute_error_change": mean_difference, "change_mcse": mcse,
                                 "change_mc_lower": mean_difference - 1.96 * mcse,
                                 "change_mc_upper": mean_difference + 1.96 * mcse,
                                 "mae_ratio_large_over_small": float(np.abs(large_error).mean() / np.abs(small_error).mean()),
                                 "rmse_ratio_large_over_small": float(np.sqrt(np.square(large_error).mean() / np.square(small_error).mean()))})
    pd.DataFrame(sample_size_rows).to_csv(root / "cross_fitted_sample_size_comparison.csv", index=False)
    fitting_diagnostics = []
    for scenario in scenarios:
        for rep in range(config["n_rep"]):
            coefficients_path = root / "replicates" / scenario / f"rep_{rep + 1:04d}" / "coefficients.csv"
            local = pd.read_csv(coefficients_path).drop(columns=["term", "estimate"]).drop_duplicates()
            local.insert(0, "rep", rep + 1)
            local.insert(0, "scenario_key", scenario)
            fitting_diagnostics.append(local)
    diagnostics = pd.concat(fitting_diagnostics, ignore_index=True)
    diagnostics.to_csv(root / "common_fit_diagnostics.csv", index=False)
    n_datasets = len(scenarios) * config["n_rep"]
    expected_fits = n_datasets * (1 + 5 * len(config["sample_sizes"]))
    if len(diagnostics) != expected_fits or set(diagnostics["solver"]) != {"br"}:
        raise RuntimeError("Common fitting manifest does not match the registered configuration")
    if len(frames["independent_intervals"]) != n_datasets * len(CONTRASTS):
        raise RuntimeError("The independent interval count does not match the registered configuration")
    if len(frames["cross_fitted_risk"]) != n_datasets * len(config["sample_sizes"]) * len(METHODS):
        raise RuntimeError("The cross-fitted risk count does not match the registered configuration")
    manifest = {"n_mechanisms": len(scenarios), "n_replicates_per_mechanism": config["n_rep"],
                "n_complete_datasets": n_datasets, "n_common_fits": len(diagnostics),
                "n_independent_intervals": len(frames["independent_intervals"]),
                "n_cross_fitted_risk_estimates": len(frames["cross_fitted_risk"]),
                "common_solver": config["common_solver"],
                "max_dual_stationarity": float(diagnostics["dual_stationarity"].max()),
                "max_relative_objective_gap": float(diagnostics["relative_objective_gap"].max())}
    (root / "completion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--reference-n", type=int, default=10000)
    parser.add_argument("--sample-sizes", default="240,600,1800")
    parser.add_argument("--scenarios", default=",".join(ADEMP_V2_SCENARIOS))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = tuple(int(value) for value in args.sample_sizes.split(","))
    scenarios = tuple(args.scenarios.split(","))
    if len(sample_sizes) != len(set(sample_sizes)) or min(sample_sizes) < 30:
        raise ValueError("Sample sizes must be distinct integers of at least 30")
    if any(scenario not in ADEMP_V2_SCENARIOS for scenario in scenarios):
        raise ValueError("Unknown ADEMP scenario")
    if args.n_rep < 2 or args.reference_n < 30 or args.workers < 1:
        raise ValueError("Invalid replicate count, reference size, or worker count")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {"out_dir": str(args.out_dir.resolve()), "n_rep": args.n_rep, "reference_n": args.reference_n,
              "sample_sizes": list(sample_sizes), "seed": args.seed, "tau": args.tau,
              "common_solver": "quantreg::rq.fit(method='br'); primal/dual certificate checked"}
    design = {"configuration": config, "scenarios": {key: ADEMP_V2_SCENARIOS[key] for key in scenarios},
              "methods": METHODS, "contrasts": CONTRASTS, "lambda_grid": LAMBDA_GRID,
              "evaluation_unit": "Independent ICU stay, averaging check loss over later records within stay",
              "independent_target": "Conditional paired risk difference of rules frozen after fitting and tuning",
              "reference": "An independent generated cohort in every replicate, reused within that replicate across rules and sample sizes",
              "cross_fitted_target": "Fold-size-weighted conditional risk of the five frozen fold-specific rules",
              "cross_fitted_measure": "OOF mean loss minus the corresponding independent reference estimate; no CLT interval is claimed",
              "coverage_reference_uncertainty": "Coverage at the reference estimate, plus proportions containing or intersecting its 95% Monte Carlo interval as sensitivity measures, not deterministic bounds on true coverage",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                              "R": subprocess.check_output(["Rscript", "--version"], text=True).strip()}}
    (args.out_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    if not args.summarize_only:
        started = time.monotonic()
        completed = {scenario: 0 for scenario in scenarios}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_replicate, config, scenario, rep): scenario
                       for rep in range(config["n_rep"]) for scenario in scenarios}
            for future in as_completed(futures):
                future.result()
                scenario = futures[future]
                completed[scenario] += 1
                if completed[scenario] % 10 == 0 or completed[scenario] == config["n_rep"]:
                    print(f"{scenario}: {completed[scenario]}/{config['n_rep']} replicates, {time.monotonic() - started:.1f} seconds", flush=True)
    summarize(config, scenarios)


if __name__ == "__main__":
    main()
