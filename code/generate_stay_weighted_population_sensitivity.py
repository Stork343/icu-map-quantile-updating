import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import split_window_data as data_utils
from run_split_window_mixed_effects_analysis import (
    apply_training_age_standardization,
    evaluate_population_and_update,
    se_mean,
    split_cluster_indices,
    tune_lambda_generic,
)
from split_window_analysis_core import design_frame, equal_stay_observation_weights, fit_common_quantile
from split_window_data import _safe_read_frame, build_dataset_from_cache


def paired_summary(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, object]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Paired loss arrays must have the same shape")
    reduction = reference - candidate
    mean = float(np.mean(reduction))
    se = se_mean(reduction)
    return {
        "paired_loss_reduction": mean,
        "paired_loss_reduction_se": se,
        "paired_loss_reduction_ci95": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
        "loss_reduction_percent": float(100.0 * mean / np.mean(reference)),
        "n_stays": int(reduction.size),
    }


def serializable_evaluation(result: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"population_losses", "update_losses"}
    }


def write_tex(rows: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Stay-weighted population quantile fitting sensitivity analysis.}",
        "\\label{tab:stay_weighted_population_sensitivity}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Model & $\\lambda_b$ & Assessment loss (SE) & Paired reduction & Descriptive interval\\\\",
        "\\hline",
    ]
    for row in rows.to_dict("records"):
        lambda_text = "--" if not np.isfinite(row["selected_lambda"]) else f"{row['selected_lambda']:.2g}"
        ci_text = f"{row['ci_low']:.4f}--{row['ci_high']:.4f}"
        lines.append(
            f"{row['model']} & {lambda_text} & {row['assessment_loss']:.4f} ({row['assessment_loss_se']:.4f}) & "
            f"{row['paired_reduction_vs_stay_weighted_population']:.4f} & {ci_text}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            (
                "\\footnotesize{The common quantile component was fitted with observation weights equal to the "
                "inverse number of training observations in each stay, so every training stay had total weight one. "
                "Assessment loss was averaged within stay and then across stays. Paired reductions and descriptive "
                "intervals use the stay-weighted baseline population model as the reference; each interval is the "
                "estimate plus or minus 1.96 times its stay-level standard error conditional on the realized fitted rules.}"
            ),
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit task-aligned stay-weighted population quantile sensitivity models.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--index-hours", type=float, default=12.0)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, _ = build_dataset_from_cache(
        obs,
        stays,
        fit_stays=args.fit_stays,
        seed=args.seed,
        analysis_hours=24.0,
    )
    enriched = data_utils.ensure_cluster_lists(dataset)
    train_idx, tuning_idx, assessment_idx = split_cluster_indices(
        len(enriched["y_list"]),
        seed=args.seed,
        train_fraction=args.train_fraction,
        tuning_fraction=args.tuning_fraction,
    )
    dataset, age_standardization = apply_training_age_standardization(dataset, stays, train_idx)
    train_data = data_utils.subset_cluster_data(dataset, train_idx)
    tuning_data = data_utils.subset_cluster_data(dataset, tuning_idx)
    assessment_data = data_utils.subset_cluster_data(dataset, assessment_idx)
    train_design, full_predictors = design_frame(train_data)
    tuning_design, _ = design_frame(tuning_data)
    assessment_design, _ = design_frame(assessment_data)
    baseline_predictors = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
    fit_weights = equal_stay_observation_weights(train_design)
    r_script = Path(__file__).with_name("fit_quantile_common.R")

    baseline_gamma = fit_common_quantile(
        train_design,
        baseline_predictors,
        tau=args.tau,
        work_dir=args.work_dir / "baseline",
        r_script=r_script,
        force_refit=True,
        observation_weights=fit_weights,
    )
    spline_gamma = fit_common_quantile(
        train_design,
        full_predictors,
        tau=args.tau,
        work_dir=args.work_dir / "spline",
        r_script=r_script,
        force_refit=True,
        observation_weights=fit_weights,
    )

    lambda_grid = [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
    baseline_tuning = tune_lambda_generic(
        tuning_design,
        baseline_predictors,
        baseline_gamma,
        args.tau,
        args.index_hours,
        lambda_grid,
    )
    spline_tuning = tune_lambda_generic(
        tuning_design,
        full_predictors,
        spline_gamma,
        args.tau,
        args.index_hours,
        lambda_grid,
    )
    baseline_lambda = float(baseline_tuning["best"]["lambda_b"])
    spline_lambda = float(spline_tuning["best"]["lambda_b"])
    baseline_eval = evaluate_population_and_update(
        assessment_design,
        baseline_predictors,
        baseline_gamma,
        args.tau,
        args.index_hours,
        baseline_lambda,
    )
    spline_eval = evaluate_population_and_update(
        assessment_design,
        full_predictors,
        spline_gamma,
        args.tau,
        args.index_hours,
        spline_lambda,
    )

    reference = np.asarray(baseline_eval["population_losses"], dtype=float)
    comparisons = {
        "baseline_population": paired_summary(reference, reference),
        "baseline_update": paired_summary(reference, np.asarray(baseline_eval["update_losses"], dtype=float)),
        "spline_population": paired_summary(reference, np.asarray(spline_eval["population_losses"], dtype=float)),
        "spline_update": paired_summary(reference, np.asarray(spline_eval["update_losses"], dtype=float)),
    }
    model_specs = [
        ("Stay-weighted population", "baseline_population", float("nan"), reference),
        ("Stay-weighted population + level update", "baseline_update", baseline_lambda, baseline_eval["update_losses"]),
        ("Stay-weighted spline population", "spline_population", float("nan"), spline_eval["population_losses"]),
        ("Stay-weighted spline + level update", "spline_update", spline_lambda, spline_eval["update_losses"]),
    ]
    rows: List[Dict[str, object]] = []
    for model, key, selected_lambda, losses in model_specs:
        loss_array = np.asarray(losses, dtype=float)
        comparison = comparisons[key]
        rows.append(
            {
                "model": model,
                "selected_lambda": selected_lambda,
                "assessment_loss": float(np.mean(loss_array)),
                "assessment_loss_se": se_mean(loss_array),
                "paired_reduction_vs_stay_weighted_population": comparison["paired_loss_reduction"],
                "paired_reduction_se": comparison["paired_loss_reduction_se"],
                "ci_low": comparison["paired_loss_reduction_ci95"][0],
                "ci_high": comparison["paired_loss_reduction_ci95"][1],
                "loss_reduction_percent": comparison["loss_reduction_percent"],
                "n_stays": comparison["n_stays"],
            }
        )
    table = pd.DataFrame(rows)
    csv_path = args.artifact_dir / "stay_weighted_population_sensitivity.csv"
    tex_path = args.artifact_dir / "stay_weighted_population_sensitivity.tex"
    json_path = args.artifact_dir / "stay_weighted_population_sensitivity.json"
    table.to_csv(csv_path, index=False)
    write_tex(table, tex_path)
    results = {
        "settings": {
            "tau": args.tau,
            "index_hours": args.index_hours,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "tuning_fraction": args.tuning_fraction,
            "fit_weight_rule": "each observation receives weight 1 / training observations in its stay",
            "lambda_grid": lambda_grid,
        },
        "data_summary": data_summary,
        "split_counts": {
            "train": int(train_idx.size),
            "tuning": int(tuning_idx.size),
            "assessment": int(assessment_idx.size),
        },
        "age_standardization": age_standardization,
        "coefficients": {
            "baseline": baseline_gamma.tolist(),
            "spline": spline_gamma.tolist(),
        },
        "tuning": {
            "baseline": baseline_tuning,
            "spline": spline_tuning,
        },
        "assessment": {
            "baseline": serializable_evaluation(baseline_eval),
            "spline": serializable_evaluation(spline_eval),
            "comparisons": comparisons,
        },
        "outputs": {"csv": str(csv_path), "tex": str(tex_path)},
    }
    json_path.write_text(json.dumps(data_utils.to_serializable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Wrote stay-weighted sensitivity results to {json_path}")


if __name__ == "__main__":
    main()
