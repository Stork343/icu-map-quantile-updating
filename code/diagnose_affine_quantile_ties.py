from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_profile_core import check_loss, empirical_quantile
from run_submission_validation_extensions import write_json


def diagnose(dataset: dict[str, object], age_values: np.ndarray, partition: Path,
             horizon: float, tolerance: float, model: str) -> dict[str, object]:
    metadata = json.loads((partition / "partition_results.json").read_text(encoding="utf-8"))
    if metadata["prediction_protocol_version"] != 2:
        raise RuntimeError("Tie diagnostics require the exact q10 boundary protocol")
    assessment = pd.read_csv(partition / "assessment_metrics.csv")
    if model == "calibrated_q10":
        calibration = metadata["calibrations"]["scalar_0"]
        coefficients = None
    elif model == "vector_direct":
        calibration = {"intercept": 0.0, "slope": 1.0}
        key = metadata["selected_rules"][model]["key"]
        with np.load(partition / "assessment_coefficients" / f"{key}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["global_indices"], assessment["global_stay_index"].to_numpy()):
                raise RuntimeError("Frozen vector coefficient order differs from assessment")
            coefficients = saved["coefficients"]
    else:
        raise ValueError("Unknown tie diagnostic model")
    counts = {"exact": 0, "near": 0, "near_below_prediction": 0, "near_above_prediction": 0, "observations": 0}
    stay_rates, losses = [], []
    for position, index in enumerate(assessment["global_stay_index"].to_numpy(dtype=int)):
        y = np.asarray(dataset["y_list"][index], dtype=float)
        times = np.asarray(dataset["t_list"][index], dtype=float)
        early = times <= 12.0
        later = (times > 12.0) & (times <= 12.0 + horizon)
        if not np.any(later):
            raise RuntimeError("The frozen assessment contains a stay without a horizon observation")
        if model == "calibrated_q10":
            q10 = empirical_quantile(y[early])
            prediction = np.full(int(np.sum(later)), float(calibration["intercept"]) + float(calibration["slope"]) * q10)
        else:
            age = metadata["age_standardization"]
            x = np.asarray(dataset["X"][index], dtype=float).copy()
            x[1] = (age_values[index] - age["age_mean_training_split"]) / age["age_sd_training_split"]
            common = float(x @ np.asarray(metadata["population_coefficients"]))
            prediction = (common + coefficients[position, 0]
                          + coefficients[position, 1] * (times[later] - 12.0) / 12.0)
        difference = y[later] - prediction
        exact = difference == 0.0
        near = np.abs(difference) <= tolerance
        negative = near & (difference < 0.0)
        positive = near & (difference > 0.0)
        counts["exact"] += int(np.sum(exact))
        counts["near"] += int(np.sum(near))
        counts["near_below_prediction"] += int(np.sum(negative))
        counts["near_above_prediction"] += int(np.sum(positive))
        counts["observations"] += difference.size
        stay_rates.append([np.mean(exact), np.mean(near), np.mean(negative), np.mean(positive),
                           np.mean(difference < 0.0), np.mean(difference <= 0.0),
                           np.mean(y[later] < prediction - tolerance),
                           np.mean(y[later] <= prediction + tolerance)])
        losses.append(float(np.mean(check_loss(difference, 0.1))))
    stored = assessment[f"loss_{model}"].to_numpy(dtype=float)
    if not np.allclose(losses, stored, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Frozen affine q10 losses do not reproduce the stored assessment")
    rates = np.mean(np.asarray(stay_rates), axis=0)
    return {"partition": partition.name, "model": model, "horizon_hours": horizon, "n_stays": assessment.shape[0],
            "absolute_tolerance": tolerance, "affine_intercept": calibration["intercept"],
            "affine_slope": calibration["slope"], **counts,
            "near_nonexact": counts["near"] - counts["exact"],
            "exact_mass_stay_equal": float(rates[0]), "near_mass_stay_equal": float(rates[1]),
            "near_nonexact_mass_stay_equal": float(rates[1] - rates[0]),
            "near_below_prediction_mass_stay_equal": float(rates[2]),
            "near_above_prediction_mass_stay_equal": float(rates[3]),
            "original_strict_rate": float(rates[4]), "original_weak_rate": float(rates[5]),
            "tolerance_strict_rate": float(rates[6]), "tolerance_weak_rate": float(rates[7]),
            "original_bracket_violation": max(float(rates[4]) - 0.1, 0.1 - float(rates[5]), 0.0),
            "tolerance_bracket_violation": max(float(rates[6]) - 0.1, 0.1 - float(rates[7]), 0.0),
            "original_mean_check_loss": float(np.mean(losses)),
            "max_loss_reproduction_difference": float(np.max(np.abs(np.asarray(losses) - stored)))}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Measure floating point near ties of frozen affine q10 forecasts")
    parser.add_argument("--dataset-cache", type=Path, default=root / ".review-work" / "strengthening" / "expanded_family" / "local_dataset.pkl")
    parser.add_argument("--clinical-dir", type=Path, default=root / ".review-work" / "adaptive" / "clinical")
    parser.add_argument("--output-dir", type=Path, default=root / ".review-work" / "adaptive" / "clinical" / "diagnostics")
    parser.add_argument("--horizon", type=float, default=4.0)
    parser.add_argument("--partitions", nargs="+", default=("fold_1", "temporal"))
    parser.add_argument("--models", nargs="+", choices=("calibrated_q10", "vector_direct"), default=("calibrated_q10", "vector_direct"))
    parser.add_argument("--all-partitions", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if args.tolerance <= 0 or args.horizon not in (4.0, 8.0, 12.0):
        raise ValueError("Invalid diagnostic tolerance or horizon")
    with args.dataset_cache.open("rb") as handle:
        cached = pickle.load(handle)
    dataset = cached["dataset"]
    age_values = cached["stays"].set_index("stay_id")["age"].reindex(dataset["cluster_ids"]).to_numpy(dtype=float)
    rows = []
    horizons = (4.0, 8.0, 12.0) if args.all_partitions else (args.horizon,)
    partitions = tuple(f"fold_{number}" for number in range(1, 6)) + ("temporal",) if args.all_partitions else args.partitions
    for horizon in horizons:
        for name in partitions:
            partition = args.clinical_dir / f"horizon_{int(horizon)}" / ("temporal" if name == "temporal" else f"nested/{name}")
            for model in args.models:
                rows.append(diagnose(dataset, age_values, partition, horizon, args.tolerance, model))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "affine_q10_tie_diagnostics.csv", index=False)
    write_json(args.output_dir / "affine_q10_tie_diagnostics.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
