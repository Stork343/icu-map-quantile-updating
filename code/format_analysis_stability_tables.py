import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def replicate_table(source, target):
    frame = pd.read_csv(source / "learning/learning_stability.csv")
    blocks = pd.read_csv(source / "learning/learning_seed_blocks.csv")
    selected = frame[(frame.scope == "reference") & (frame["first"] == "selected_direct")
                     & (frame["second"] == "constant_direct")].copy()
    selected_blocks = blocks[(blocks.scope == "reference") & (blocks["first"] == "selected_direct")
                            & (blocks["second"] == "constant_direct")
                            & blocks.block.str.startswith("block")]
    grouped = selected_blocks.groupby(["setting", "horizon_hours"]).mean_difference.max()
    selected["maximum_block_mean"] = [grouped.loc[(row.setting, row.horizon_hours)]
                                      for row in selected.itertuples()]
    columns = {
        "setting": "setting", "horizon_hours": "horizon",
        "gain_proportion": "gain_percent", "gain_proportion_exact95_low": "gain_low",
        "gain_proportion_exact95_high": "gain_high",
        "leave_one_out_maximum_mean": "loo_max", "maximum_block_mean": "block_max",
        "mean_after_removing_best_10": "remove10_mean",
    }
    output = selected[list(columns)].rename(columns=columns)
    output[["gain_percent", "gain_low", "gain_high"]] *= 100
    assert len(output) == 12 and not output.duplicated(["setting", "horizon"]).any()
    output.to_csv(target / "analysis_replicate_stability.csv", index=False)


def clinical_table(source, target):
    nested = pd.read_csv(source / "clinical/clinical_nested_stability_summary.csv")
    temporal = pd.read_csv(source / "clinical/clinical_temporal_joint_intervals.csv")
    nested = nested[(nested["first"] == "continuous_direct")
                    & (nested["second"] == "constant_diagonal_direct")]
    temporal = temporal[(temporal["first"] == "continuous_direct")
                        & (temporal["second"] == "constant_diagonal_direct")]
    output = nested[["horizon_hours", "n_stays", "minimum_fold_mean", "maximum_fold_mean",
                     "maximum_omitting_fold_mean"]].merge(
        temporal[["horizon_hours", "mean_difference", "joint_ci95_low", "joint_ci95_high"]],
        on="horizon_hours", validate="one_to_one").rename(columns={
            "horizon_hours": "horizon", "n_stays": "nested_stays",
            "minimum_fold_mean": "fold_min", "maximum_fold_mean": "fold_max",
            "maximum_omitting_fold_mean": "omit_max", "mean_difference": "temporal_difference",
            "joint_ci95_low": "temporal_low", "joint_ci95_high": "temporal_high"})
    assert len(output) == 3
    output.to_csv(target / "analysis_clinical_stability.csv", index=False)


def coverage_table(source, target):
    frame = pd.read_csv(source / "simulation/coverage_extension/coverage_summary.csv")
    keys = ["scenario_key", "n_evaluation", "contrast"]
    assert len(frame) == 81 and not frame.duplicated([*keys, "interval_method"]).any()
    assert (frame.n_replicates == 200).all()
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        values = dict(zip(keys, key))
        for method, name in (("normal", "normal"), ("percentile_bootstrap", "percentile"),
                             ("studentized_bootstrap", "studentized")):
            row = group[group.interval_method == method]
            assert len(row) == 1
            values[name] = 100 * row.coverage.iloc[0]
            values[f"{name}_mcse"] = 100 * row.coverage_mcse.iloc[0]
        values["reference_ratio_max"] = group.maximum_reference_mcse_over_evaluation_se.max()
        rows.append(values)
    output = pd.DataFrame(rows)
    assert len(output) == 27 and np.isfinite(output.select_dtypes(include="number")).all().all()
    output.to_csv(target / "analysis_interval_coverage.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replicate_table(args.source, args.output)
    clinical_table(args.source, args.output)
    coverage_table(args.source, args.output)
    print("Stability tables written: 12 replicate rows, 3 clinical rows, 27 coverage rows")


if __name__ == "__main__":
    main()
