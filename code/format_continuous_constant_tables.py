import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")
HORIZONS = (4, 8, 12)


def unique_row(frame, **filters):
    selected = np.ones(len(frame), dtype=bool)
    for name, value in filters.items():
        selected &= frame[name].to_numpy() == value
    result = frame.loc[selected]
    if len(result) != 1:
        raise ValueError(f"The numerical table requires exactly one row: {filters}")
    return result.iloc[0]


def simulation_table(source, output, scale):
    frame = pd.read_csv(source / "simulation/paired_summary.csv")
    if len(frame) != 192 or not (frame.replicates == 200).all():
        raise ValueError("The full continuous constant comparison family is required")
    contrasts = (("continuous", f"continuous_{scale}", f"continuous_constant_{scale}"),
                 ("selected", f"selected_{scale}", f"selected_constant_{scale}"),
                 ("resolution", f"selected_constant_{scale}", f"constant_{scale}"))
    rows = []
    for setting in SETTINGS:
        for horizon in HORIZONS:
            values = {"setting": setting, "horizon_hours": horizon}
            for prefix, first, second in contrasts:
                row = unique_row(frame, setting=setting, scope="reference", horizon_hours=horizon,
                                 first=first, second=second)
                values[f"{prefix}_difference"] = row.mean_difference
                values[f"{prefix}_low"] = row.simultaneous_low
                values[f"{prefix}_high"] = row.simultaneous_high
            rows.append(values)
    result = pd.DataFrame(rows)
    if result.shape != (12, 11) or not np.isfinite(result.select_dtypes(include="number")).all().all():
        raise ValueError("The simulation table has incomplete numerical entries")
    result.to_csv(output / f"continuous_constant_simulation_{scale}.csv", index=False, float_format="%.17g")


def clinical_table(source, output):
    nested = pd.read_csv(source / "clinical/clinical_nested_pairs.csv")
    temporal = pd.read_csv(source / "clinical/clinical_temporal_paired.csv")
    if len(temporal) != 24 or not (temporal.comparison_family_size == 24).all():
        raise ValueError("The complete clinical temporal comparison family is required")
    rows = []
    for horizon in HORIZONS:
        values = {"horizon_hours": horizon}
        for prefix, first, second in (("continuous", "continuous_direct", "continuous_constant_direct"),
                                      ("selected", "selected_component_direct", "selected_constant_direct")):
            nested_row = unique_row(nested, partition="nested_pooled", horizon_hours=horizon, first=first, second=second)
            temporal_row = unique_row(temporal, horizon_hours=horizon, first=first, second=second)
            values[f"nested_{prefix}"] = nested_row.mean_difference
            values[f"temporal_{prefix}"] = temporal_row.mean_difference
            values[f"temporal_{prefix}_low"] = temporal_row.simultaneous_ci95_low
            values[f"temporal_{prefix}_high"] = temporal_row.simultaneous_ci95_high
            if prefix == "continuous":
                values["n_nested"] = int(nested_row.n_stays)
                values["n_temporal"] = int(temporal_row.n_stays)
            elif values["n_nested"] != nested_row.n_stays or values["n_temporal"] != temporal_row.n_stays:
                raise ValueError("Clinical comparison populations differ")
        rows.append(values)
    columns = ["horizon_hours", "nested_continuous", "nested_selected", "temporal_continuous",
               "temporal_continuous_low", "temporal_continuous_high", "temporal_selected",
               "temporal_selected_low", "temporal_selected_high", "n_nested", "n_temporal"]
    result = pd.DataFrame(rows)[columns]
    if result.shape != (3, 11) or not np.isfinite(result).all().all():
        raise ValueError("The clinical table has incomplete numerical entries")
    result.to_csv(output / "continuous_constant_clinical.csv", index=False, float_format="%.17g")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in ("direct", "affine"):
        simulation_table(args.source, args.output, scale)
    clinical_table(args.source, args.output)
    print("Written two 12 row simulation tables and one 3 row clinical table")


if __name__ == "__main__":
    main()
