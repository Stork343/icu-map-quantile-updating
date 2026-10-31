from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SCOPES = ("nested_pooled", "temporal_2017_2022")
HORIZONS = (4, 8, 12)
MODELS = ("continuous_direct", "continuous_affine", "constant_diagonal_direct", "constant_diagonal_affine",
          "stump_original", "original_selected", "calibrated_q10")
FIELDS = ("scope", "model", "n_d4", "n_d8", "n_d12", "loss_d4", "violation_d4", "loss_d8", "violation_d8",
          "loss_d12", "violation_d12", "difference_d4", "lower_d4", "upper_d4", "difference_d8", "lower_d8",
          "upper_d8", "difference_d12", "lower_d12", "upper_d12")


def generate(source: Path, destination: Path) -> dict:
    summary = pd.read_csv(source / "clinical_model_summary.csv")
    pairs = pd.read_csv(source / "clinical_temporal_paired.csv")
    verification = json.loads((source / "clinical_verification.json").read_text())
    independent = json.loads((source / "clinical_independent_verification.json").read_text())
    if verification["horizon_comparisons"] != 18 or independent["status"] != "passed":
        raise ValueError("The table requires the complete independently verified clinical study")
    if summary.duplicated(["scope", "horizon_hours", "model"]).any():
        raise ValueError("Repeated clinical summary model")
    if pairs.duplicated(["scope", "horizon_hours", "first", "second"]).any():
        raise ValueError("Repeated clinical paired comparison")
    indexed = summary.set_index(["scope", "horizon_hours", "model"])
    paired = pairs.set_index(["scope", "horizon_hours", "first", "second"])
    rows = []
    counts = {}
    for scope in SCOPES:
        for model in MODELS:
            row = {name: None for name in FIELDS}
            row.update(scope=scope, model=model)
            for horizon in HORIZONS:
                current = indexed.loc[(scope, horizon, model)]
                if not np.isfinite(current[["n_stays", "mean_loss", "p_lt", "p_le", "bracket_violation"]].to_numpy(dtype=float)).all():
                    raise ValueError("Nonfinite clinical table score")
                if not (0 <= current["p_lt"] <= current["p_le"] <= 1) or current["mean_loss"] < 0:
                    raise ValueError("Invalid clinical table metric")
                value = max(current["p_lt"] - .1, .1 - current["p_le"], 0)
                np.testing.assert_allclose(current["bracket_violation"], value, rtol=0, atol=1e-12)
                count = int(current["n_stays"])
                if (scope, horizon) in counts and counts[(scope, horizon)] != count:
                    raise ValueError("Clinical rules use different assessment cohorts")
                counts[(scope, horizon)] = count
                row.update({f"n_d{horizon}": count, f"loss_d{horizon}": current["mean_loss"],
                            f"violation_d{horizon}": current["bracket_violation"]})
                if scope == "temporal_2017_2022" and model in ("continuous_direct", "continuous_affine"):
                    reference = "constant_diagonal_direct" if model.endswith("direct") else "constant_diagonal_affine"
                    interval = paired.loc[(scope, horizon, model, reference)]
                    if interval["interval_type"] != "pointwise paired normal 95 percent" or int(interval["n_stays"]) != count:
                        raise ValueError("The temporal comparison has another unit or interval definition")
                    difference = current["mean_loss"] - indexed.loc[(scope, horizon, reference), "mean_loss"]
                    np.testing.assert_allclose(interval["mean_difference"], difference, rtol=0, atol=1e-12)
                    np.testing.assert_allclose([interval["ci95_low"], interval["ci95_high"]],
                        interval["mean_difference"] + np.asarray([-1.96, 1.96]) * interval["se"], rtol=0, atol=1e-12)
                    row.update({f"difference_d{horizon}": interval["mean_difference"],
                                f"lower_d{horizon}": interval["ci95_low"], f"upper_d{horizon}": interval["ci95_high"]})
            rows.append(row)
    result = pd.DataFrame(rows).loc[:, FIELDS]
    score_columns = [name for name in FIELDS if name.startswith(("n_", "loss_", "violation_"))]
    if len(result) != 14 or not np.isfinite(result[score_columns].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete clinical table")
    fits = json.loads((source / "clinical_learning_fits.json").read_text())
    selected_continuous = sum(value["selected_kind"] == "continuous" for value in fits.values())
    if selected_continuous != 6:
        raise ValueError("The clinical table note requires six continuous cross validation selections")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, float_format="%.17g", lineterminator="\n")
    return {"status": "passed", "source_directory": "component_learning/public/clinical", "table_rows": 14,
        "displayed_model_cells": 42, "paired_comparisons": 6, "horizon_hours": list(HORIZONS),
        "selection_uses_assessment": False, "continuous_selections": selected_continuous,
        "fields": list(FIELDS), "nullable_fields": list(FIELDS[11:]),
        "nullable_meaning": "Intervals are displayed for the two continuous temporal rules; other interval fields are unused"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / ".review-work/component_learning/public/clinical")
    parser.add_argument("--output", type=Path, default=PROJECT / "manuscript/tables/component_clinical_comparison.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / ".review-work/component_learning/clinical_table_validation.json")
    args = parser.parse_args()
    report = generate(args.source, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
