from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import t


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")
SCOPES = ("reference", "assessment")
HOURS = (4, 8, 12)
CONTRASTS = (
    ("continuous_direct", "constant_direct"),
    ("selected_direct", "constant_direct"),
    ("continuous_affine", "constant_affine"),
    ("selected_affine", "constant_affine"),
    ("selected_affine", "stump_original"),
    ("selected_affine", "original_selected"),
    ("continuous_affine", "original_selected"),
    ("selected_affine", "calibrated_q10"),
)
KEYS = ("setting", "scope", "horizon_hours", "first", "second")
FIELDS = KEYS + ("replicates", "mean_difference", "mcse", "pointwise_low", "pointwise_high",
                 "simultaneous_low", "simultaneous_high")
MAIN_FIELDS = ("setting", "horizon_hours", "constant_loss", "selected_loss", "mean_difference",
               "simultaneous_low", "simultaneous_high")


def generate(source, destination):
    frame = pd.read_csv(source / "paired_summary.csv")
    summary = pd.read_csv(source / "method_summary.csv")
    if tuple(frame.columns) != FIELDS or len(frame) != 192:
        raise ValueError("The component simulation table requires all 192 planned comparisons")
    if frame.isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise ValueError("Missing or repeated component simulation comparisons")
    expected = {(setting, scope, horizon, first, second)
                for setting, scope, horizon, (first, second) in itertools.product(SETTINGS, SCOPES, HOURS, CONTRASTS)}
    if set(frame.loc[:, KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete component simulation comparison grid")
    if not np.isfinite(frame.loc[:, FIELDS[5:]].to_numpy(dtype=float)).all() or not (frame.replicates == 200).all():
        raise ValueError("Nonfinite component simulation values or incomplete replicates")
    for prefix, critical in (("pointwise", t.ppf(.975, 199)),
                             ("simultaneous", t.ppf(1 - .05 / (2 * 192), 199))):
        for suffix, direction in (("low", -1), ("high", 1)):
            np.testing.assert_allclose(frame[f"{prefix}_{suffix}"],
                frame.mean_difference + direction * critical * frame.mcse, atol=1e-12, rtol=1e-12)
    metrics = summary.set_index(["setting", "scope", "horizon_hours", "method"])
    main_rows = []
    for row in frame.itertuples(index=False):
        first = metrics.loc[(row.setting, row.scope, row.horizon_hours, row.first), "mean_loss"]
        second = metrics.loc[(row.setting, row.scope, row.horizon_hours, row.second), "mean_loss"]
        np.testing.assert_allclose(row.mean_difference, first - second, atol=1e-12, rtol=0)
        if row.scope == "reference" and (row.first, row.second) == ("selected_direct", "constant_direct"):
            main_rows.append(dict(setting=row.setting, horizon_hours=row.horizon_hours, constant_loss=second,
                selected_loss=first, mean_difference=row.mean_difference,
                simultaneous_low=row.simultaneous_low, simultaneous_high=row.simultaneous_high))
    destination.mkdir(parents=True, exist_ok=True)
    for scope in SCOPES:
        order = [(setting, scope, horizon, first, second)
                 for setting, horizon, (first, second) in itertools.product(SETTINGS, HOURS, CONTRASTS)]
        output = frame.set_index(list(KEYS)).loc[order].reset_index().loc[:, FIELDS]
        if len(output) != 96:
            raise ValueError("Each component simulation scope requires 96 contrasts")
        output.to_csv(destination / f"component_simulation_{scope}.csv", index=False, float_format="%.17g", lineterminator="\n")
    main = pd.DataFrame(main_rows).set_index(["setting", "horizon_hours"])
    main = main.loc[list(itertools.product(SETTINGS, HOURS))].reset_index().loc[:, MAIN_FIELDS]
    main.to_csv(destination / "component_simulation_direct.csv", index=False, float_format="%.17g", lineterminator="\n")
    if len(main) != 12 or int((main.simultaneous_high < 0).sum()) != 9:
        raise ValueError("The direct learning table has an unexpected result inventory")
    return {"status": "passed", "source_comparisons": len(frame), "complete_scope_tables": 2,
            "scope_table_rows": 96, "direct_table_rows": len(main), "independent_replicates_per_setting": 200,
            "interval_family": 192, "supported_direct_reductions": 9}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / ".review-work/component_learning/public/simulation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "manuscript/tables")
    parser.add_argument("--manifest", type=Path, default=ROOT / ".review-work/component_learning/simulation_table_validation.json")
    args = parser.parse_args()
    report = generate(args.source, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
