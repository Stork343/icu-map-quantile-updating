from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import t


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
FIELDS = (
    "setting", "selection_mode", "scope", "horizon_hours", "coverage_hours",
    "first", "second", "replicates", "mean_difference", "mcse", "pointwise_low",
    "pointwise_high", "simultaneous_low", "simultaneous_high",
    "mean_stays_per_replicate", "minimum_stays_per_replicate", "total_stay_evaluations",
)
SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")
MODES = ("joint", "direct")
SCOPES = ("reference", "assessment")
HOURS = (4, 8, 12)
CONTRASTS = (
    ("full_geometry", "diagonal_profile"),
    ("horizon_geometry", "diagonal_profile"),
    ("full_geometry", "horizon_geometry"),
)
KEYS = ("setting", "selection_mode", "scope", "horizon_hours", "coverage_hours", "first", "second")


def verify_source(frame):
    if set(frame.columns) != set(FIELDS) or len(frame) != 432:
        raise ValueError("The mechanism table requires all 432 contrasts and exactly seventeen fields")
    if frame.isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise ValueError("Missing or duplicate mechanism table cells")
    expected = {
        (setting, mode, scope, horizon, coverage, first, second)
        for setting, mode, scope, horizon, coverage, (first, second) in itertools.product(
            SETTINGS, MODES, SCOPES, HOURS, HOURS, CONTRASTS)
    }
    observed = set(frame.loc[:, list(KEYS)].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("Incomplete mechanism setting, horizon, coverage or contrast grid")
    values = frame.loc[:, list(FIELDS[7:])].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (frame["replicates"] == 200).all():
        raise ValueError("Nonfinite mechanism values or incomplete independent replicates")
    if (frame["mcse"] < 0).any() or (frame["minimum_stays_per_replicate"] < 1).any():
        raise ValueError("Invalid Monte Carlo uncertainty or empty coverage groups")
    for field in ("minimum_stays_per_replicate", "total_stay_evaluations"):
        if not np.equal(frame[field], np.floor(frame[field])).all():
            raise ValueError("Coverage group counts must be integers")
    if not np.allclose(frame["mean_stays_per_replicate"] * 200,
                       frame["total_stay_evaluations"], atol=1e-8, rtol=0):
        raise ValueError("Coverage count denominators disagree")
    if (frame["minimum_stays_per_replicate"] > frame["mean_stays_per_replicate"]).any():
        raise ValueError("Coverage minimum exceeds its replicate mean")
    for prefix, critical in (("pointwise", t.ppf(.975, 199)),
                             ("simultaneous", t.ppf(1 - .05 / (2 * 432), 199))):
        for suffix, direction in (("low", -1), ("high", 1)):
            expected_endpoint = frame["mean_difference"] + direction * critical * frame["mcse"]
            if not np.allclose(frame[f"{prefix}_{suffix}"], expected_endpoint, atol=1e-12, rtol=1e-12):
                raise ValueError("Mechanism interval does not use its declared Student t critical value")
    return frame.loc[:, list(FIELDS)].copy()


def write_tables(source, output_directory):
    frame = verify_source(pd.read_csv(source))
    ordered_keys = [
        (setting, mode, "reference", horizon, coverage, first, second)
        for mode, setting, horizon, coverage, (first, second) in itertools.product(
            MODES, SETTINGS, HOURS, HOURS, CONTRASTS)
    ]
    indexed = frame.set_index(list(KEYS))
    output_directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "complete", "source_contrasts": len(frame), "interval_family": 432,
              "independent_replicates": 200, "tables": {}}
    for mode in MODES:
        mode_keys = [key for key in ordered_keys if key[1] == mode]
        selected = indexed.loc[mode_keys].reset_index().loc[:, list(FIELDS)]
        if len(selected) != 108:
            raise ValueError("A reference mechanism table must contain 108 contrasts")
        name = f"geometry_mechanism_{mode}.csv"
        destination = output_directory / name
        selected.to_csv(destination, index=False, float_format="%.17g", lineterminator="\n")
        reread = pd.read_csv(destination)
        if list(reread.columns) != list(FIELDS) or len(reread) != 108:
            raise RuntimeError("Written mechanism CSV differs from its declared schema")
        report["tables"][name] = {"rows": len(selected), "columns": list(FIELDS),
                                  "scope": "reference", "selection_mode": mode}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=WORKSPACE / ".review-work/mechanism/public/paired_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "manuscript/tables")
    parser.add_argument("--manifest", type=Path,
                        default=WORKSPACE / ".review-work/mechanism/table_validation.json")
    args = parser.parse_args()
    report = write_tables(args.source, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
