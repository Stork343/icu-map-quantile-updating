from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")


def summarize_pairs(directory, name, extra_keys, comparison_count):
    pairs = pd.read_csv(directory / "paired_replicates.csv")
    keys = ["setting", "scope", *extra_keys, "horizon_hours", "first", "second"]
    assert not pairs.duplicated([*keys, "replicate"]).any()
    assert np.isfinite(pairs[["mean_difference", "within_sample_se"]]).all().all()
    rows, blocks = [], []
    for key, group in pairs.groupby(keys, sort=True):
        group = group.sort_values("replicate")
        assert group.replicate.tolist() == list(range(1, 201))
        values = group.mean_difference.to_numpy()
        n = len(values)
        total = float(values.sum())
        mean = total / n
        mcse = float(values.std(ddof=1) / np.sqrt(n))
        gain = int(np.count_nonzero(values < 0))
        ties = int(np.count_nonzero(values == 0))
        interval = binomtest(gain, n).proportion_ci(confidence_level=0.95, method="exact")
        loo = (total - values) / (n - 1)
        identity = dict(zip(keys, key))
        rows.append({**identity, "replicates": n, "mean_difference": mean, "mcse": mcse,
                     "simultaneous_low": mean - t.ppf(1 - .05 / (2 * comparison_count), n - 1) * mcse,
                     "simultaneous_high": mean + t.ppf(1 - .05 / (2 * comparison_count), n - 1) * mcse,
                     "median_difference": float(np.median(values)), "gain_replicates": gain,
                     "zero_replicates": ties, "gain_proportion": gain / n,
                     "gain_proportion_exact95_low": interval.low,
                     "gain_proportion_exact95_high": interval.high,
                     "leave_one_out_minimum_mean": float(loo.min()),
                     "leave_one_out_maximum_mean": float(loo.max()),
                     "leave_one_out_same_sign": bool(np.all(np.sign(loo) == np.sign(mean))),
                     "maximum_absolute_leave_one_out_change": float(np.max(np.abs(loo - mean))),
                     "mean_after_removing_best_10": float(np.sort(values)[10:].mean()),
                     "mean_after_removing_worst_10": float(np.sort(values)[:-10].mean())})
        for label, first, last in (("first100", 1, 100), ("last100", 101, 200),
                                   ("block1", 1, 50), ("block2", 51, 100),
                                   ("block3", 101, 150), ("block4", 151, 200)):
            selected = values[first - 1:last]
            blocks.append({**identity, "block": label, "first_replicate": first,
                           "last_replicate": last, "replicates": len(selected),
                           "mean_difference": float(selected.mean()),
                           "gain_proportion": float(np.mean(selected < 0))})
    summary, block_frame = pd.DataFrame(rows), pd.DataFrame(blocks)
    assert len(summary) == comparison_count
    expected = pd.read_csv(directory / "paired_summary.csv").set_index(keys).sort_index()
    actual = summary.set_index(keys).sort_index()
    assert expected.index.equals(actual.index)
    np.testing.assert_allclose(actual[["mean_difference", "mcse", "simultaneous_low", "simultaneous_high"]],
                               expected[["mean_difference", "mcse", "simultaneous_low", "simultaneous_high"]],
                               rtol=0, atol=2e-14)
    summary.to_csv(OUTPUT / f"{name}_stability.csv", index=False)
    block_frame.to_csv(OUTPUT / f"{name}_seed_blocks.csv", index=False)
    matching_keys = [key for key in keys if key != "scope"]
    paired_scopes = summary.pivot(index=matching_keys, columns="scope", values="mean_difference")
    paired_scopes["same_direction"] = np.sign(paired_scopes.assessment) == np.sign(paired_scopes.reference)
    paired_scopes.reset_index().to_csv(OUTPUT / f"{name}_assessment_reference.csv", index=False)
    return summary, block_frame


def audit_frozen(directory):
    rows, optimization = [], []
    seeds_seen = set()
    for setting_index, setting in enumerate(SETTINGS):
        for number in range(1, 201):
            replicate = directory / "replicates" / setting / f"rep_{number:04d}"
            frozen = json.loads((replicate / "frozen.json").read_text())
            complete = json.loads((replicate / "complete.json").read_text())
            sequence = np.random.SeedSequence([20261022, setting_index, number])
            seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(4)]
            assert frozen["derived_seeds"] == seeds == complete["derived_seeds"]
            assert not seeds_seen.intersection(seeds)
            seeds_seen.update(seeds)
            assert complete["status"] == "complete" and complete["diagnostics"]["all_certified"]
            order = np.random.default_rng(seeds[1]).permutation(600)
            fitting, tuning, assessment = order[:360], order[360:480], order[480:]
            assert not (set(fitting) & set(tuning) or set(fitting) & set(assessment)
                        or set(tuning) & set(assessment))
            learning = frozen["learning"]
            scores = pd.DataFrame(learning["validation_scores"])
            assert len(scores) == 5 and len(learning["validation_rows"]) == 15
            continuous = scores[scores.ridge_eta.notna()].sort_values("validation_loss").iloc[0]
            constant = float(scores[scores.candidate == "constant"].validation_loss.iloc[0])
            expected = "constant" if constant <= continuous.validation_loss + 1e-12 else "continuous"
            assert expected == learning["selected_kind"] == complete["selected_kind"]
            assert learning["best_continuous_eta"] == continuous.ridge_eta
            rows.append({"setting": setting, "replicate": number, "selected_kind": expected,
                         "ridge_eta": learning["best_continuous_eta"],
                         "constant_cv_loss": constant, "continuous_cv_loss": continuous.validation_loss,
                         "constant_replay_maximum_loss_difference": learning["constant_replay_maximum_loss_difference"]})
            fits = [("full_tuning", learning["best_continuous"])]
            fits += [(f"fold_{row['fold']}_{row['candidate']}", row["fit"])
                     for row in learning["validation_rows"] if "fit" in row]
            for name, fit in fits:
                assert fit["profile_certificates_passed"]
                assert len(fit["starts"]) == 2
                optimization.append({"setting": setting, "replicate": number, "fit": name,
                    "returned_optimizer_success": fit["optimizer_success"],
                    "projected_gradient_norm": fit["projected_gradient_norm"],
                    "successful_starts": sum(start["optimizer_success"] for start in fit["starts"]),
                    "iteration_limit_starts": sum(start["optimizer_status"] == 1 for start in fit["starts"]),
                    "other_unsuccessful_starts": sum(not start["optimizer_success"] and start["optimizer_status"] != 1
                                                      for start in fit["starts"]),
                    "maximum_iterations": max(start["iterations"] for start in fit["starts"])})
    selections, fits = pd.DataFrame(rows), pd.DataFrame(optimization)
    actual = pd.read_csv(directory / "selection_replicates.csv").set_index(["setting", "replicate"]).sort_index()
    observed = selections.set_index(["setting", "replicate"]).sort_index()
    assert actual.selected_kind.equals(observed.selected_kind)
    np.testing.assert_array_equal(actual.ridge_eta, observed.ridge_eta)
    selections.to_csv(OUTPUT / "frozen_selection_replay.csv", index=False)
    fits.to_csv(OUTPUT / "optimization_status.csv", index=False)
    return {"completed_replicates": len(rows), "unique_child_seeds": len(seeds_seen),
            "frozen_branch_decisions_replayed": len(rows), "retained_fits": len(fits),
            "full_fit_optimizer_success_count": int(fits[(fits.fit == "full_tuning")].returned_optimizer_success.sum()),
            "all_fit_optimizer_success_count": int(fits.returned_optimizer_success.sum()),
            "all_fit_iteration_limit_starts": int(fits.iteration_limit_starts.sum()),
            "all_fit_other_unsuccessful_starts": int(fits.other_unsuccessful_starts.sum()),
            "maximum_projected_gradient_norm": float(fits.projected_gradient_norm.max())}


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-source", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    learning_directory = args.learning_source
    mechanism_directory = args.geometry_source
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    learning, blocks = summarize_pairs(learning_directory, "learning", [], 192)
    mechanism, mechanism_blocks = summarize_pairs(mechanism_directory, "geometry", ["selection_mode", "coverage_hours"], 432)
    frozen = audit_frozen(learning_directory)
    direct = learning[(learning.scope == "reference") & (learning["first"] == "selected_direct")
                      & (learning["second"] == "constant_direct")].copy()
    informative = direct[direct.setting != "information_limit"]
    selected_blocks = blocks[(blocks.scope == "reference") & (blocks["first"] == "selected_direct")
                             & (blocks["second"] == "constant_direct")
                             & (blocks.setting != "information_limit")]
    assessment = learning[(learning.scope == "assessment") & (learning["first"] == "selected_direct")
                          & (learning["second"] == "constant_direct")
                          & (learning.setting != "information_limit")]
    report = {"status": "passed", "diagnostic_interpretation": "Saved independent replicates are re-summarized; seed-order blocks are not new independent experiments.",
        "learning_comparisons_checked": len(learning), "geometry_comparisons_checked": len(mechanism),
        "frozen_integrity": frozen,
        "nine_direct_reference_comparisons": {
            "count": len(informative), "all_simultaneous_upper_limits_negative": bool((informative.simultaneous_high < 0).all()),
            "all_leave_one_out_means_negative": bool((informative.leave_one_out_maximum_mean < 0).all()),
            "all_means_negative_after_removing_best_ten": bool((informative.mean_after_removing_best_10 < 0).all()),
            "gain_proportion_range": [float(informative.gain_proportion.min()), float(informative.gain_proportion.max())],
            "half_block_means_negative": int((selected_blocks[selected_blocks.block.isin(["first100", "last100"])].mean_difference < 0).sum()),
            "half_block_comparisons": len(selected_blocks[selected_blocks.block.isin(["first100", "last100"])]),
            "quarter_block_means_negative": int((selected_blocks[selected_blocks.block.str.startswith("block")].mean_difference < 0).sum()),
            "quarter_block_comparisons": len(selected_blocks[selected_blocks.block.str.startswith("block")]),
            "assessment_means_negative": int((assessment.mean_difference < 0).sum()),
            "assessment_simultaneous_upper_limits_negative": int((assessment.simultaneous_high < 0).sum())},
        "complete_direct_reference_diagnostics": direct.to_dict("records")}
    (OUTPUT / "stability_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "complete_direct_reference_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
