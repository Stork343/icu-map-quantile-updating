import argparse
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import subprocess

import numpy as np
import pandas as pd


PARTITIONS = tuple(f"fold_{number}" for number in range(1, 6)) + ("temporal",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, zero_checks, completed_replays = {}, 0, []
    for name in PARTITIONS:
        source = args.private / name
        fit = json.loads((source / "fitted_control.json").read_text())
        completed = json.loads((source / "complete.json").read_text())
        if completed["status"] != "complete" or not completed["history_coefficients_fixed_at_zero"]:
            raise RuntimeError("Incomplete or nonconstant clinical control")
        if completed["maximum_original_metric_replay_error"] > 1e-8:
            raise RuntimeError("Original clinical score replay failed")
        completed_replays.append(completed["maximum_original_metric_replay_error"])
        for current in [row["fit"] for row in fit["validation_rows"]] + [fit["continuous_fit"]]:
            params = np.asarray(current["params"])
            if params.shape != (2, 6) or np.any(params[:, 3:] != 0) or np.any(np.abs(params[:, :3]) > 12):
                raise RuntimeError("A control coefficient differs from its fixed search space")
            if not current["all_evaluation_diagnostics"]["all_certified"]:
                raise RuntimeError("A clinical control profile lacks its numerical certificate")
            if len(current["starts"]) != 2 or any(row["iterations"] > 100 for row in current["starts"]):
                raise RuntimeError("A clinical control has a different optimization budget")
            zero_checks += 1
        for horizon in (4, 8, 12):
            frame = pd.read_csv(source / f"assessment_h{horizon}.csv")
            if frame.global_stay_index.duplicated().any():
                raise RuntimeError("Duplicate clinical assessment records")
            if fit["selected_kind"] == "grid":
                for suffix in ("direct", "affine"):
                    for metric in ("loss", "p_lt", "p_le"):
                        reference = frame[f"{metric}_constant_diagonal_{suffix}"].to_numpy()
                        np.testing.assert_allclose(frame[f"{metric}_selected_constant_{suffix}"], reference, rtol=0, atol=1e-8)
                        frame[f"{metric}_selected_constant_{suffix}"] = reference
            frames[name, horizon] = frame
    table = pd.read_csv(args.public / "clinical_temporal_paired.csv")
    if len(table) != 24 or table.duplicated(["horizon_hours", "first", "second"]).any():
        raise RuntimeError("The 24 temporal comparisons are incomplete")
    critical = NormalDist().inv_cdf(1 - .05 / 48)
    for row in table.itertuples(index=False):
        frame = frames["temporal", row.horizon_hours]
        values = frame[f"loss_{row.first}"].to_numpy() - frame[f"loss_{row.second}"].to_numpy()
        count = len(values)
        mean = float(sum(values) / count)
        se = math.sqrt(float(np.square(values - mean).sum()) / (count * (count - 1)))
        pvalue = 1.0 if se == 0 else math.erfc(abs(mean / se) / math.sqrt(2))
        actual = [mean, se, mean - 1.96 * se, mean + 1.96 * se,
                  mean - critical * se, mean + critical * se, pvalue]
        expected = [row.mean_difference, row.se, row.ci95_low, row.ci95_high,
                    row.simultaneous_ci95_low, row.simultaneous_ci95_high, row.raw_p_value]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    temporary = args.output / "r_tmp"
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["TMPDIR"] = os.path.relpath(temporary.resolve(), Path.cwd())
    adjusted = args.output / "holm_independent.csv"
    subprocess.run(["Rscript", str(Path(__file__).with_name("adjust_clinical_families.R")),
                    str(args.public / "clinical_temporal_paired.csv"), str(adjusted)],
                   env=environment, check=True)
    independent = pd.read_csv(adjusted)
    np.testing.assert_allclose(independent.holm_p_value, table.holm_p_value, rtol=0, atol=2e-12)
    nested = pd.read_csv(args.public / "clinical_nested_pairs.csv")
    omission = pd.read_csv(args.public / "clinical_nested_omission.csv")
    for row in nested.itertuples(index=False):
        selected = ([frames[name, row.horizon_hours] for name in PARTITIONS[:-1]]
                    if row.partition == "nested_pooled" else [frames[row.partition, row.horizon_hours]])
        source = pd.concat(selected, ignore_index=True)
        if source.global_stay_index.duplicated().any():
            raise RuntimeError("The nested assessment folds overlap")
        difference = source[f"loss_{row.first}"] - source[f"loss_{row.second}"]
        np.testing.assert_allclose(difference.mean(), row.mean_difference, rtol=0, atol=2e-12)
        if len(source) != row.n_stays:
            raise RuntimeError("A nested comparison has a different denominator")
    for row in omission.itertuples(index=False):
        source = pd.concat([frames[name, row.horizon_hours] for name in PARTITIONS[:-1]
                            if name != row.omitted_fold], ignore_index=True)
        difference = source[f"loss_{row.first}"] - source[f"loss_{row.second}"]
        np.testing.assert_allclose(difference.mean(), row.mean_difference, rtol=0, atol=2e-12)
    penalty = pd.read_csv(args.public / "clinical_penalty_summary.csv")
    for row in penalty.itertuples(index=False):
        values = np.asarray([row.q05, row.q25, row.median, row.q75, row.q95])
        if np.any(np.diff(values) < -1e-12) or values[0] < .03 or values[-1] > (10 if row.component == "level" else 100):
            raise RuntimeError("A penalty summary differs from its declared ranges")
        if row.count_below13_stays + row.count_atleast13_stays != row.n_stays:
            raise RuntimeError("The count groups do not cover the assessment records")
        if row.model in ("continuous_constant", "selected_constant", "grid_constant"):
            np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-12)
    affine = pd.read_csv(args.public / "clinical_affine_coefficients.csv")
    if len(affine) != 36 or affine.duplicated(["partition", "horizon_hours", "model"]).any():
        raise RuntimeError("Clinical control affine coefficients are incomplete")
    for row in affine.itertuples(index=False):
        source = json.loads((args.private / row.partition / "calibration_coefficients.json").read_text())
        expected = source[str(int(row.horizon_hours))][row.model]
        np.testing.assert_allclose([row.intercept, row.slope, row.stationarity, row.relative_gap],
                                  [expected[key] for key in ("intercept", "slope", "stationarity", "relative_gap")],
                                  rtol=0, atol=2e-12)
        if row.stationarity > 1e-6 or row.relative_gap > 1e-7:
            raise RuntimeError("An affine calibration failed its primal dual certificate")
    result = {"status": "passed", "partitions_checked": 6, "zero_history_fits_checked": zero_checks,
              "temporal_contrasts_checked": len(table), "nested_comparisons_checked": len(nested),
              "omission_summaries_checked": len(omission), "penalty_summary_rows_checked": len(penalty),
              "affine_coefficient_rows_checked": len(affine),
              "original_frozen_metric_maximum_error": max(completed_replays),
              "holm_verified_by": "R stats::p.adjust(method='holm')", "public_identifier_columns": 0}
    (args.public / "clinical_independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
