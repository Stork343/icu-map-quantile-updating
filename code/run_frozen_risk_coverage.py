from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest


from run_paired_bootstrap_validation import replay_assessment
from run_risk_inference_simulation import CONTRASTS, METHODS, pack_records, rule_loss_matrix
from run_split_window_simulation import ADEMP_V2_SCENARIOS, PREDICTOR_COLS, simulate_ademp_v2_dataset


SCENARIOS = ("common_time_misspecified", "ideal_large_dense", "cluster_size_informative")
CONFIGURATION = {"scenarios": SCENARIOS, "replicates_per_scenario": 200,
                 "assessment_sizes": [120, 600, 1800], "reference_size": 100000,
                 "reference_block_size": 10000, "bootstrap_replicates": 2000,
                 "bootstrap_batch_size": 200, "master_seed": 20261025,
                 "contrasts": [f"{first}_minus_{second}" for first, second in CONTRASTS],
                 "interval_methods": ["normal", "percentile_bootstrap", "studentized_bootstrap"],
                 "selection_reason": "Common-time misspecification had undercoverage after Holm adjustment across the 84 existing coverage cells; informative cluster size had the lowest percentile-bootstrap coverage; large dense is the positive-control reference mechanism.",
                 "original_assessment": "Replay all original 120 assessment stays and frozen fitted rules exactly.",
                 "new_assessments": "A separately generated cohort of 1800 stays, with its first 600 used for the 600-stay assessment.",
                 "reference": "Ten independent 10000-stay blocks, independent of all fitting, tuning and assessment records.",
                 "stopping_rule": "Complete all 600 prespecified frozen-rule replicates without outcome-dependent stopping."}


def differences(losses):
    return np.column_stack([losses[:, METHODS.index(first)] - losses[:, METHODS.index(second)]
                            for first, second in CONTRASTS])


def generate_losses(count, scenario, seed, beta, rule):
    generated = simulate_ademp_v2_dataset(count, 0.1, ADEMP_V2_SCENARIOS[scenario], np.random.default_rng(seed))
    losses = rule_loss_matrix(pack_records(generated), beta, float(rule.selected_penalty),
                             float(rule.intercept), float(rule.slope), 0.1)
    assert losses.shape == (count, len(METHODS)) and np.isfinite(losses).all()
    return losses


def intervals(values, seed, configuration):
    count = len(values)
    estimate = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(count)
    assert np.all(standard_error > 0) and np.isfinite(standard_error).all()
    rng = np.random.default_rng(seed)
    means = []
    studentized = []
    for start in range(0, configuration["bootstrap_replicates"], configuration["bootstrap_batch_size"]):
        batch = min(configuration["bootstrap_batch_size"], configuration["bootstrap_replicates"] - start)
        indices = rng.integers(0, count, size=(batch, count))
        sampled = values[indices]
        boot_mean = sampled.mean(axis=1)
        boot_se = sampled.std(axis=1, ddof=1) / np.sqrt(count)
        if not np.isfinite(boot_se).all() or np.any(boot_se <= 0):
            raise RuntimeError("A studentized bootstrap draw has a nonpositive standard error")
        means.append(boot_mean)
        studentized.append((boot_mean - estimate) / boot_se)
    boot_quantiles = np.quantile(np.concatenate(means), [0.025, 0.975], axis=0)
    student_quantiles = np.quantile(np.concatenate(studentized), [0.025, 0.975], axis=0)
    bounds = {"normal": (estimate - 1.96 * standard_error, estimate + 1.96 * standard_error),
              "percentile_bootstrap": (boot_quantiles[0], boot_quantiles[1]),
              "studentized_bootstrap": (estimate - student_quantiles[1] * standard_error,
                                        estimate - student_quantiles[0] * standard_error)}
    return estimate, standard_error, bounds


def run_replicate(configuration, source, output, scenario, replicate):
    folder = Path(output) / scenario / f"rep_{replicate:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.json"
    if complete.exists():
        stored = json.loads(complete.read_text())
        if stored["configuration"] != json.loads(json.dumps(configuration)):
            raise RuntimeError("Checkpoint differs from the fixed coverage design")
        return stored["runtime_seconds"]
    started = time.monotonic()
    location = Path(source) / "replicates" / scenario / f"rep_{replicate:04d}"
    coefficients = pd.read_csv(location / "coefficients.csv")
    beta = coefficients.loc[coefficients.group_id == 0].set_index("term").loc[PREDICTOR_COLS, "estimate"].to_numpy()
    frozen = pd.read_csv(location / "frozen_rules.csv")
    rule = frozen.loc[frozen.group_id == 0].iloc[0]
    old_losses, old_intervals = replay_assessment(Path(source), scenario, replicate)
    original_count = configuration["assessment_sizes"][0]
    assert len(old_losses) == original_count
    block_count = configuration["reference_size"] // configuration["reference_block_size"]
    sequence = np.random.SeedSequence([configuration["master_seed"], configuration["scenarios"].index(scenario), replicate])
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(1 + block_count + len(configuration["assessment_sizes"]))]
    future_losses = generate_losses(max(configuration["assessment_sizes"]), scenario, seeds[0], beta, rule)
    np.savez_compressed(folder / "assessment_losses.npz", original_assessment=old_losses, independent_assessment=future_losses)
    reference_blocks = []
    total = np.zeros(len(CONTRASTS))
    total_squared = np.zeros(len(CONTRASTS))
    for block, seed in enumerate(seeds[1:block_count + 1]):
        reference = differences(generate_losses(configuration["reference_block_size"], scenario, seed, beta, rule))
        total += reference.sum(axis=0)
        total_squared += np.square(reference).sum(axis=0)
        for index, contrast in enumerate(configuration["contrasts"]):
            reference_blocks.append({"block": block + 1, "seed": seed, "n_stays": configuration["reference_block_size"],
                                     "contrast": contrast, "sum": float(reference[:, index].sum()),
                                     "sum_squared": float(np.square(reference[:, index]).sum())})
    reference_mean = total / configuration["reference_size"]
    reference_variance = (total_squared - configuration["reference_size"] * np.square(reference_mean)) / (configuration["reference_size"] - 1)
    assert np.all(reference_variance > 0)
    reference_mcse = np.sqrt(reference_variance / configuration["reference_size"])
    pd.DataFrame(reference_blocks).to_csv(folder / "reference_blocks.csv", index=False)
    rows = []
    for position, count in enumerate(configuration["assessment_sizes"]):
        values = differences(old_losses if count == original_count else future_losses[:count])
        estimate, standard_error, bounds = intervals(values, seeds[block_count + 1 + position], configuration)
        for method, (lower, upper) in bounds.items():
            for index, contrast in enumerate(configuration["contrasts"]):
                target, mcse = reference_mean[index], reference_mcse[index]
                original = old_intervals.loc[contrast]
                rows.append({"scenario_key": scenario, "replicate": replicate, "n_evaluation": count,
                             "contrast": contrast, "interval_method": method, "estimate": float(estimate[index]),
                             "standard_error": float(standard_error[index]), "interval_lower": float(lower[index]),
                             "interval_upper": float(upper[index]), "reference_risk_difference": float(target),
                             "reference_mcse": float(mcse), "reference_mcse_over_evaluation_se": float(mcse / standard_error[index]),
                             "coverage_at_reference_estimate": bool(lower[index] <= target <= upper[index]),
                             "coverage_reference_interval_contained": bool(lower[index] <= target - 1.96 * mcse and target + 1.96 * mcse <= upper[index]),
                             "coverage_reference_interval_intersects": bool(lower[index] <= target + 1.96 * mcse and target - 1.96 * mcse <= upper[index]),
                             "miss_below_reference": bool(upper[index] < target), "miss_above_reference": bool(lower[index] > target),
                             "original_reference_risk_difference": float(original.reference_risk_difference),
                             "original_reference_mcse": float(original.reference_mcse)})
    pd.DataFrame(rows).to_csv(folder / "intervals.csv", index=False)
    elapsed = time.monotonic() - started
    complete.write_text(json.dumps({"configuration": configuration, "scenario": scenario, "replicate": replicate,
                                    "source": str(location), "beta": beta.tolist(), "frozen_rule": rule.to_dict(),
                                    "derived_seeds": seeds, "runtime_seconds": elapsed,
                                    "original_assessment_replay_passed": True}, indent=2) + "\n")
    return elapsed


def summarize(output, configuration):
    frames = []
    count = configuration["replicates_per_scenario"]
    for scenario in configuration["scenarios"]:
        for replicate in range(1, count + 1):
            folder = output / scenario / f"rep_{replicate:04d}"
            complete = json.loads((folder / "complete.json").read_text())
            assert complete["original_assessment_replay_passed"]
            frames.append(pd.read_csv(folder / "intervals.csv"))
    intervals_frame = pd.concat(frames, ignore_index=True)
    expected_models = len(configuration["scenarios"]) * count
    expected_cells = len(configuration["scenarios"]) * len(configuration["assessment_sizes"]) * len(CONTRASTS) * len(configuration["interval_methods"])
    assert len(intervals_frame) == expected_cells * count
    rows = []
    for keys, group in intervals_frame.groupby(["scenario_key", "n_evaluation", "contrast", "interval_method"]):
        assert set(group.replicate) == set(range(1, count + 1))
        successes = int(group.coverage_at_reference_estimate.sum())
        lower = 0 if successes == 0 else beta.ppf(.025, successes, count - successes + 1)
        upper = 1 if successes == count else beta.ppf(.975, successes + 1, count - successes)
        rows.append(dict(zip(("scenario_key", "n_evaluation", "contrast", "interval_method"), keys)) |
                    {"n_replicates": count, "covered": successes, "coverage": successes / count,
                     "coverage_mcse": float(np.sqrt(successes / count * (1 - successes / count) / count)),
                     "exact_ci95_lower": float(lower), "exact_ci95_upper": float(upper),
                     "binomial_pvalue_at_nominal95": float(binomtest(successes, count, .95).pvalue),
                     "mean_interval_length": float((group.interval_upper - group.interval_lower).mean()),
                     "mean_reference_mcse": float(group.reference_mcse.mean()),
                     "mean_reference_mcse_over_evaluation_se": float(group.reference_mcse_over_evaluation_se.mean()),
                     "maximum_reference_mcse_over_evaluation_se": float(group.reference_mcse_over_evaluation_se.max()),
                     "reference_interval_contained_rate": float(group.coverage_reference_interval_contained.mean()),
                     "reference_interval_intersects_rate": float(group.coverage_reference_interval_intersects.mean()),
                     "miss_below_reference": int(group.miss_below_reference.sum()),
                     "miss_above_reference": int(group.miss_above_reference.sum())})
    summary = pd.DataFrame(rows)
    ordered = np.argsort(summary.binomial_pvalue_at_nominal95.to_numpy())
    adjusted = np.maximum.accumulate((len(summary) - np.arange(len(summary))) * summary.binomial_pvalue_at_nominal95.to_numpy()[ordered])
    summary["holm_adjusted_binomial_pvalue"] = np.nan
    summary.loc[ordered, "holm_adjusted_binomial_pvalue"] = np.minimum(1, adjusted)
    intervals_frame.to_csv(output / "all_intervals.csv", index=False)
    summary.to_csv(output / "coverage_summary.csv", index=False)
    (output / "completion_manifest.json").write_text(json.dumps({"status": "complete", "frozen_models": expected_models,
        "intervals": len(intervals_frame), "coverage_cells": len(summary), "configuration": configuration}, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--reference-n", type=int, default=100000)
    parser.add_argument("--reference-block-size", type=int, default=10000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--assessment-sizes", default="120,600,1800")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    scenarios = args.scenarios.split(",")
    sizes = [int(value) for value in args.assessment_sizes.split(",")]
    if args.n_rep < 2 or args.reference_n < 2 or args.reference_block_size < 2 or args.bootstrap_replicates < 200 or args.workers < 1:
        raise ValueError("Invalid replicate, reference, bootstrap or worker count")
    if args.reference_n % args.reference_block_size != 0 or args.reference_n < max(sizes):
        raise ValueError("Reference size must be a multiple of its block size and at least the assessment size")
    if sizes != sorted(set(sizes)) or len(sizes) < 2:
        raise ValueError("Assessment sizes must be distinct increasing integers")
    if len(scenarios) != len(set(scenarios)) or any(scenario not in ADEMP_V2_SCENARIOS for scenario in scenarios):
        raise ValueError("Invalid scenario list")
    configuration = json.loads(json.dumps(CONFIGURATION))
    configuration.update({"scenarios": scenarios, "replicates_per_scenario": args.n_rep,
                          "reference_size": args.reference_n, "reference_block_size": args.reference_block_size,
                          "bootstrap_replicates": args.bootstrap_replicates, "master_seed": args.seed,
                          "assessment_sizes": sizes,
                          "original_assessment": f"Replay all original {sizes[0]} assessment stays and frozen fitted rules exactly.",
                          "new_assessments": f"A separately generated cohort of {max(sizes)} stays, with prefixes used for the smaller new assessments.",
                          "reference": f"{args.reference_n // args.reference_block_size} independent {args.reference_block_size}-stay blocks, independent of all fitting, tuning and assessment records.",
                          "stopping_rule": f"Complete all {len(scenarios) * args.n_rep} prespecified frozen-rule replicates without outcome-dependent stopping."})
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != configuration:
        raise RuntimeError("Output directory already has a different design")
    design_path.write_text(json.dumps(configuration, indent=2) + "\n")
    if args.summarize_only:
        summarize(output, configuration)
        return
    if args.benchmark:
        elapsed = run_replicate(configuration, args.inference_dir.resolve(), output, scenarios[0], 1)
        print(json.dumps({"complete_replicate_seconds": elapsed, "coverage_results_not_used_for_design": True}), flush=True)
        return
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_replicate, configuration, args.inference_dir.resolve(), output, scenario, replicate)
                   for replicate in range(1, args.n_rep + 1) for scenario in scenarios]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(futures):
                print(f"Completed {count}/{len(futures)} frozen models in {time.monotonic() - start:.1f} seconds", flush=True)
    summarize(output, configuration)


if __name__ == "__main__":
    main()
