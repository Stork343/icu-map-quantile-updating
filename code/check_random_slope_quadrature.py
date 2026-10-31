from pathlib import Path
import argparse
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from run_random_slope_comparator import observation_predictions, horizon_records, fit_affine
from run_published_quantile_comparator import r_environment
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import global_index_lookup
from split_window_analysis_core import check_loss, design_frame
from split_window_data import subset_cluster_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="temporal")
    parser.add_argument("--nodes", type=int, nargs=2, default=(15, 31))
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / ".review-work/adaptive/random_slope_continuation")
    parser.add_argument("--second-model-dir", type=Path)
    parser.add_argument("--dataset-cache", type=Path,
                        default=ROOT / ".review-work/strengthening/expanded_family/local_dataset.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".review-work/adaptive/precision")
    args = parser.parse_args()
    source = args.model_dir / args.partition
    output = args.output_dir / args.partition
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "r_tmp"
    runtime.mkdir(exist_ok=True)
    config = json.loads((source / "configuration.json").read_text())
    with args.dataset_cache.open("rb") as handle:
        cached = pickle.load(handle)
    fitting = np.asarray(config["fitting"], dtype=int)
    tuning = np.asarray(config["tuning"], dtype=int)
    data, _ = apply_training_age_standardization(cached["dataset"], cached["stays"], fitting)
    design, _ = design_frame(subset_cluster_data(data, tuning))
    lookup = global_index_lookup(data)
    predictions = {}
    for nodes in args.nodes:
        model_source = (args.second_model_dir / args.partition
                        if nodes == args.nodes[1] and args.second_model_dir is not None else source)
        target = output / f"parameters_{nodes}.csv"
        subprocess.run(["Rscript", str(CODE_DIR / "export_random_slope_parameters.R"),
                        str(model_source / f"model.rds.stage_{nodes}"), str(target)],
                       env=r_environment(runtime), check=True)
        parameters = pd.read_csv(target).set_index("term")["estimate"].to_dict()
        if int(parameters["nodes_per_dimension"]) != nodes:
            raise RuntimeError("Saved quadrature stage differs from its declared node count")
        predictions[nodes] = observation_predictions(design, parameters, lookup, 12.0)
    first, second = args.nodes
    identity = ["global_stay_index", "time_hours", "y"]
    if not predictions[first][identity].equals(predictions[second][identity]):
        raise RuntimeError("Quadrature comparison uses different tuning observations")
    rows = []
    for horizon in (4, 8, 12):
        records = {nodes: horizon_records(predictions[nodes], horizon, 12.0) for nodes in args.nodes}
        calibrated = {}
        for nodes in args.nodes:
            values = records[nodes]
            calibration = fit_affine(values, output / f"nodes_{nodes}_h{horizon}", 0.1)
            calibrated[nodes] = calibration["intercept"] + calibration["slope"] * values["prediction"].to_numpy()
        for method in ("direct", "affine"):
            left = records[first]["prediction"].to_numpy() if method == "direct" else calibrated[first]
            right = records[second]["prediction"].to_numpy() if method == "direct" else calibrated[second]
            y = records[first]["y"].to_numpy()
            frame = pd.DataFrame({"stay": records[first]["global_stay_index"].to_numpy(),
                "absolute_change": np.abs(right - left), "change": right - left,
                "paired_loss_change": check_loss(y - right, 0.1) - check_loss(y - left, 0.1)})
            means = frame.groupby("stay", sort=True).mean()
            rows.append({"partition": args.partition, "horizon_hours": horizon, "method": method,
                         "nodes_first": first, "nodes_second": second, "n_tuning_stays": len(means),
                         **{f"mean_{name}": float(means[name].mean()) for name in means},
                         "maximum_observation_prediction_change": float(np.max(np.abs(right - left))),
                         "prediction_information": "tuning index histories and tuning calibration outcomes"})
    pd.DataFrame(rows).to_csv(output / f"quadrature_{first}_to_{second}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
