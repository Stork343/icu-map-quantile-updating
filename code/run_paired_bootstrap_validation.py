from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_risk_inference_simulation import CONTRASTS, METHODS, pack_records, rule_loss_matrix
from run_split_window_simulation import ADEMP_V2_SCENARIOS, PREDICTOR_COLS, simulate_ademp_v2_dataset, split_cluster_indices
from run_temporal_validation import paired_bootstrap_interval


def replay_assessment(inference_dir: Path, scenario: str, rep: int) -> tuple[np.ndarray, pd.DataFrame]:
    location = inference_dir / "replicates" / scenario / f"rep_{rep:04d}"
    marker = json.loads((location / "complete.json").read_text())
    configuration = marker["configuration"]
    original_n = int(ADEMP_V2_SCENARIOS[scenario]["n_stays"])
    generated_n = max(original_n, max(configuration["sample_sizes"]))
    data = simulate_ademp_v2_dataset(
        generated_n, configuration["tau"], ADEMP_V2_SCENARIOS[scenario], np.random.default_rng(marker["data_seed"])
    )
    _, _, assessment = split_cluster_indices(original_n, np.random.default_rng(marker["split_seed"]))
    coefficients = pd.read_csv(location / "coefficients.csv")
    independent_coefficients = coefficients.loc[coefficients["group_id"] == 0].set_index("term")
    beta = independent_coefficients.loc[PREDICTOR_COLS, "estimate"].to_numpy(dtype=float)
    rules = pd.read_csv(location / "frozen_rules.csv")
    independent_rule = rules.loc[rules["group_id"] == 0].iloc[0]
    losses = rule_loss_matrix(
        pack_records(data).subset(assessment), beta, float(independent_rule["selected_penalty"]),
        float(independent_rule["intercept"]), float(independent_rule["slope"]), configuration["tau"]
    )
    normal = pd.read_csv(location / "independent_intervals.csv").set_index("contrast")
    for first, second in CONTRASTS:
        contrast = f"{first}_minus_{second}"
        differences = losses[:, METHODS.index(first)] - losses[:, METHODS.index(second)]
        mean = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / np.sqrt(differences.size))
        expected = normal.loc[contrast]
        actual_values = np.asarray([mean, standard_error, mean - 1.96 * standard_error, mean + 1.96 * standard_error])
        expected_values = expected[["estimate", "standard_error", "interval_lower", "interval_upper"]].to_numpy(dtype=float)
        if not np.allclose(actual_values, expected_values, rtol=1e-11, atol=1e-12):
            raise RuntimeError(f"Frozen-rule assessment replay mismatch: {scenario}, replicate {rep}, {contrast}")
    return losses, normal


def reference_coverage(lower: float, upper: float, mean: float, mcse: float) -> dict[str, float | bool]:
    return {
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_length": upper - lower,
        "coverage_at_reference_estimate": lower <= mean <= upper,
        "coverage_reference_interval_contained": lower <= mean - 1.96 * mcse and mean + 1.96 * mcse <= upper,
        "coverage_reference_interval_intersects": lower <= mean + 1.96 * mcse and mean - 1.96 * mcse <= upper,
    }


def run_replicate(configuration: dict, scenario: str, rep: int) -> None:
    output = Path(configuration["out_dir"]) / "replicates" / scenario / f"rep_{rep:04d}"
    marker = output / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text())["configuration"] != configuration:
            raise RuntimeError(f"Bootstrap checkpoint configuration mismatch: {marker}")
        return
    output.mkdir(parents=True, exist_ok=True)
    losses, normal = replay_assessment(Path(configuration["inference_dir"]), scenario, rep)
    scenario_position = list(ADEMP_V2_SCENARIOS).index(scenario)
    group_id = scenario_position * 1_000_000 + rep - 1
    rows = []
    for position, (first, second) in enumerate(CONTRASTS):
        contrast = f"{first}_minus_{second}"
        baseline = normal.loc[contrast]
        seed = configuration["seed"] + 104729 * group_id + 1009 * position
        bootstrap = paired_bootstrap_interval(
            losses[:, METHODS.index(first)], losses[:, METHODS.index(second)], configuration["bootstrap_replicates"], seed
        )
        reference_mean, reference_mcse = float(baseline["reference_risk_difference"]), float(baseline["reference_mcse"])
        common = {"scenario_key": scenario, "rep": rep, "contrast": contrast,
                  "n_evaluation": int(baseline["n_evaluation"]), "reference_n": int(baseline["reference_n"]),
                  "estimate": float(baseline["estimate"]), "standard_error": float(baseline["standard_error"]),
                  "reference_risk_difference": reference_mean, "reference_mcse": reference_mcse,
                  "assessment_replay_verified": True}
        rows.append({**common, "interval_method": "normal", "bootstrap_replicates": 0, "bootstrap_seed": -1,
                     **reference_coverage(float(baseline["interval_lower"]), float(baseline["interval_upper"]), reference_mean, reference_mcse)})
        rows.append({**common, "interval_method": "percentile_bootstrap",
                     "bootstrap_replicates": configuration["bootstrap_replicates"], "bootstrap_seed": seed,
                     **reference_coverage(bootstrap["ci95_low"], bootstrap["ci95_high"], reference_mean, reference_mcse)})
    pd.DataFrame(rows).to_csv(output / "paired_interval_comparison.csv", index=False)
    marker.write_text(json.dumps({"configuration": configuration, "scenario_key": scenario, "rep": rep,
                                  "assessment_replay_verified": True}, indent=2) + "\n")


def summarize(configuration: dict, scenarios: tuple[str, ...]) -> None:
    root = Path(configuration["out_dir"])
    frames = []
    for scenario in scenarios:
        for rep in range(1, configuration["n_rep"] + 1):
            location = root / "replicates" / scenario / f"rep_{rep:04d}"
            if json.loads((location / "complete.json").read_text())["configuration"] != configuration:
                raise RuntimeError(f"Bootstrap checkpoint configuration mismatch: {location}")
            frames.append(pd.read_csv(location / "paired_interval_comparison.csv"))
    intervals = pd.concat(frames, ignore_index=True)
    intervals.to_csv(root / "paired_interval_comparison.csv", index=False)
    rows = []
    for (scenario, contrast, method), local in intervals.groupby(["scenario_key", "contrast", "interval_method"], sort=False):
        coverage = float(local["coverage_at_reference_estimate"].mean())
        rows.append({"scenario_key": scenario, "contrast": contrast, "interval_method": method, "n_rep": len(local),
                     "n_evaluation": int(local["n_evaluation"].iloc[0]), "coverage": coverage,
                     "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / len(local))),
                     "mean_interval_length": float(local["interval_length"].mean()),
                     "interval_length_mean_mcse": float(local["interval_length"].std(ddof=1) / np.sqrt(len(local))),
                     "coverage_reference_interval_contained": float(local["coverage_reference_interval_contained"].mean()),
                     "coverage_reference_interval_intersects": float(local["coverage_reference_interval_intersects"].mean()),
                     "mean_reference_mcse": float(local["reference_mcse"].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "paired_interval_coverage_summary.csv", index=False)
    paired_changes = []
    for (scenario, contrast), local in intervals.groupby(["scenario_key", "contrast"], sort=False):
        covered = local.pivot(index="rep", columns="interval_method", values="coverage_at_reference_estimate").astype(float)
        lengths = local.pivot(index="rep", columns="interval_method", values="interval_length")
        difference = covered["percentile_bootstrap"] - covered["normal"]
        paired_changes.append({"scenario_key": scenario, "contrast": contrast,
                               "bootstrap_minus_normal_coverage": float(difference.mean()),
                               "coverage_difference_mcse": float(difference.std(ddof=1) / np.sqrt(len(difference))),
                               "bootstrap_over_normal_mean_length": float(lengths["percentile_bootstrap"].mean() / lengths["normal"].mean())})
    pd.DataFrame(paired_changes).to_csv(root / "bootstrap_normal_paired_comparison.csv", index=False)
    expected = len(scenarios) * configuration["n_rep"] * len(CONTRASTS) * 2
    if len(intervals) != expected or not intervals["assessment_replay_verified"].all():
        raise RuntimeError("Bootstrap interval manifest does not match the complete replayed study")
    manifest = {"n_mechanisms": len(scenarios), "n_replicates_per_mechanism": configuration["n_rep"],
                "n_replayed_independent_assessments": len(scenarios) * configuration["n_rep"],
                "n_normal_intervals_replayed": expected // 2, "n_percentile_bootstrap_intervals": expected // 2,
                "bootstrap_replicates_per_interval": configuration["bootstrap_replicates"],
                "all_assessment_means_and_standard_errors_verified": True}
    (root / "completion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--scenarios", default=",".join(ADEMP_V2_SCENARIOS))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    scenarios = tuple(args.scenarios.split(","))
    if args.n_rep < 2 or args.bootstrap_replicates < 200 or args.workers < 1:
        raise ValueError("Invalid replicate count, bootstrap count or worker count")
    if len(scenarios) != len(set(scenarios)) or any(key not in ADEMP_V2_SCENARIOS for key in scenarios):
        raise ValueError("Invalid scenario list")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configuration = {"inference_dir": str(args.inference_dir.resolve()), "out_dir": str(args.out_dir.resolve()),
                     "n_rep": args.n_rep, "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed}
    design = {"configuration": configuration, "scenarios": scenarios,
              "bootstrap_function": "run_temporal_validation.paired_bootstrap_interval",
              "quantile_rule": "Empirical bootstrap-mean 0.025 and 0.975 quantiles using NumPy's default linear interpolation",
              "sampling_unit": "Paired stay-level loss difference; within-stay records remain together",
              "reference": "Previously saved frozen-rule conditional risk estimate and Monte Carlo standard error",
              "reference_sensitivity": "Containing or intersecting a reference 95% Monte Carlo interval are sensitivity measures, not deterministic coverage bounds",
              "bootstrap_seed_formula": "seed + 104729 * (1000000*scenario_position + replicate_position) + 1009*contrast_position",
              "source_inference_design": json.loads((args.inference_dir / "design.json").read_text())}
    (args.out_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    if not args.summarize_only:
        started, completed = time.monotonic(), 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_replicate, configuration, scenario, rep)
                       for rep in range(1, args.n_rep + 1) for scenario in scenarios]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(f"Completed {completed}/{len(futures)} paired bootstrap replays in {time.monotonic() - started:.1f} seconds", flush=True)
    summarize(configuration, scenarios)


if __name__ == "__main__":
    main()
