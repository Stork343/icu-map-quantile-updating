from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expanded_family_core import MODEL_ORDER
from run_temporal_validation import paired_bootstrap_interval, paired_interval


PUBLISHED_MODELS = ("published_lqmm", "calibrated_published_lqmm")
MODELS = (*MODEL_ORDER, *PUBLISHED_MODELS)


def join_assessments(family_path: Path, comparator_path: Path, expected_stays: int, nested: bool) -> pd.DataFrame:
    family = pd.read_csv(family_path)
    comparator = pd.read_csv(comparator_path)
    keys = ["global_stay_index"]
    if nested:
        keys.append("outer_fold")
    for frame in (family, comparator):
        if frame.shape[0] != expected_stays or frame["global_stay_index"].nunique() != expected_stays:
            raise ValueError("Assessment input must contain each eligible stay exactly once")
    columns = [*keys, "late_obs", *[f"{metric}_{model}" for model in PUBLISHED_MODELS
                                  for metric in ("loss", "p_lt", "p_le")]]
    merged = family.merge(comparator[columns], on=keys, validate="one_to_one", how="outer",
                          indicator=True, suffixes=("", "_comparator"))
    if merged.shape[0] != expected_stays or not merged["_merge"].eq("both").all():
        raise ValueError("Family and comparator use different assessment stays or folds")
    if not np.array_equal(merged["late_obs"].to_numpy(), merged["late_obs_comparator"].to_numpy()):
        raise ValueError("Family and comparator score different numbers of later observations")
    metrics = [f"{metric}_{model}" for model in MODELS for metric in ("loss", "p_lt", "p_le")]
    if not np.isfinite(merged[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Assessment metrics must be finite")
    if np.any(merged[[f"loss_{model}" for model in MODELS]].to_numpy(dtype=float) < 0):
        raise ValueError("Check loss must be nonnegative")
    return merged.sort_values("global_stay_index").reset_index(drop=True)


def model_summary(records: pd.DataFrame, scope: str, tau: float) -> list[dict]:
    population = float(records["loss_population"].mean())
    calibrated_lqmm = float(records["loss_calibrated_published_lqmm"].mean())
    rows = []
    for model in MODELS:
        loss = float(records[f"loss_{model}"].mean())
        lower = float(records[f"p_lt_{model}"].mean())
        upper = float(records[f"p_le_{model}"].mean())
        if lower < 0 or upper > 1 or lower > upper:
            raise ValueError("Invalid probability mass bracket")
        rows.append({"scope": scope, "model": model, "n_stays": len(records),
                     "mean_stay_level_check_loss": loss, "p_lt": lower, "p_le": upper,
                     "calibration_bracket_violation": max(lower - tau, tau - upper, 0.0),
                     "relative_reduction_vs_population_percent": 100.0 * (1 - loss / population),
                     "relative_reduction_vs_calibrated_lqmm_percent": 100.0 * (1 - loss / calibrated_lqmm)})
    return rows


def compare_temporal(records: pd.DataFrame, bootstrap_replicates: int, seed: int) -> list[dict]:
    rows = []
    for reference in ("population", "calibrated_q10", "calibrated_published_lqmm"):
        for candidate in MODELS:
            if candidate == reference:
                continue
            first = records[f"loss_{candidate}"].to_numpy(dtype=float)
            second = records[f"loss_{reference}"].to_numpy(dtype=float)
            row = {"candidate": candidate, "reference": reference, **paired_interval(first, second)}
            if candidate == "selected_family":
                interval = paired_bootstrap_interval(first, second, bootstrap_replicates, seed)
                row.update({"bootstrap_ci95_low": interval["ci95_low"],
                            "bootstrap_ci95_high": interval["ci95_high"],
                            "bootstrap_replicates": bootstrap_replicates})
            rows.append(row)
    return rows


def validate_partition_metadata(family: dict, comparator: dict) -> None:
    for key in ("fit_stays", "tuning_stays", "assessment_stays"):
        if int(family[key]) != int(comparator[key]):
            raise ValueError(f"Different {key} in compared model metadata")
    if family["age_standardization"].keys() != comparator["age_standardization"].keys():
        raise ValueError("Different age standardization fields")
    for key, value in family["age_standardization"].items():
        if not np.isclose(float(value), float(comparator["age_standardization"][key]), rtol=1e-12, atol=1e-12):
            raise ValueError("Compared models use different fitting age transformations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--comparator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparator_metadata = json.loads((args.comparator_dir / "published_comparator_results.json").read_text())
    family_metadata = json.loads((args.family_dir / "aggregate/expanded_family_results_all.json").read_text())
    for family, comparator in zip(family_metadata["nested_partitions"], comparator_metadata["nested_partitions"], strict=True):
        validate_partition_metadata(family, comparator)
    validate_partition_metadata(family_metadata["temporal_partition"], comparator_metadata["temporal_partition"])
    nested = join_assessments(args.family_dir / "nested_outer_assessment_metrics.csv",
                              args.comparator_dir / "nested_outer_assessment_metrics.csv", 60310, True)
    temporal = join_assessments(args.family_dir / "temporal_assessment_metrics.csv",
                                args.comparator_dir / "temporal_assessment_metrics.csv", 22005, False)
    rows = model_summary(nested, "nested_pooled", 0.1)
    for fold in range(1, 6):
        subset = nested.loc[nested["outer_fold"] == fold]
        if len(subset) != 12062:
            raise ValueError("Nested folds must preserve the fixed 12062-stay assessment allocation")
        rows.extend(model_summary(subset, f"nested_fold_{fold}", 0.1))
    rows.extend(model_summary(temporal, "temporal_2017_2022", 0.1))
    pd.DataFrame(rows).to_csv(args.output_dir / "complete_model_comparison.csv", index=False)
    pd.DataFrame(compare_temporal(temporal, args.bootstrap_replicates, args.bootstrap_seed)).to_csv(
        args.output_dir / "complete_temporal_paired_comparison.csv", index=False)
    metadata = {"tau": 0.1, "landmark_hours": 12, "nested_stays": len(nested), "temporal_stays": len(temporal),
                "same_stay_and_later_record_checks": "PASS", "same_partition_and_age_transformation_checks": "PASS",
                "models": MODELS, "published_comparator_quadrature_nodes": comparator_metadata["quadrature_nodes"],
                "nested_uncertainty": "Point estimates and individual fold results",
                "temporal_uncertainty": "Paired independent-stay normal intervals; selected-family comparisons also use paired stay bootstrap",
                "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed}
    (args.output_dir / "comparison_validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS: fixed-cohort comparisons paired by stay and later observation count; aggregate tables written")


if __name__ == "__main__":
    main()
