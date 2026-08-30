#!/usr/bin/env python3
"""Temporal validation of the ICU MAP quantile profiling family.

The analysis fits the population component in 2008--2013, selects the penalty
and affine calibration in 2014--2016, and evaluates frozen prediction rules in
2017--2022.  All periods use the MIMIC-IV ``anchor_year_group`` field.  Public
outputs contain aggregate results only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import split_window_data as data_utils
from generate_split_window_hard_result_tables import X_PREDICTORS
from run_split_window_mixed_effects_analysis import apply_training_age_standardization
from run_submission_validation_extensions import (
    add_prediction_columns,
    attach_model_metrics,
    build_stay_records,
    fit_all_calibrations,
    fit_population_component,
    global_index_lookup,
    json_safe,
    parse_float_grid,
    select_calibration_model,
    write_json,
)
from split_window_analysis_core import design_frame
from split_window_clinical_core import tune_lambda_generic
from split_window_data import _safe_read_frame, build_dataset_from_cache


DEFAULT_FIT_PERIODS = ("2008 - 2010", "2011 - 2013")
DEFAULT_TUNING_PERIODS = ("2014 - 2016",)
DEFAULT_ASSESSMENT_PERIODS = ("2017 - 2019", "2020 - 2022")
DISPLAY_MODELS = ("population", "primary_level_update", "calibrated_q10")


def parse_periods(value: str) -> Tuple[str, ...]:
    periods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not periods:
        raise argparse.ArgumentTypeError("period list must contain at least one value")
    return periods


def temporal_indices(
    dataset: Mapping[str, object],
    stays: pd.DataFrame,
    patients: pd.DataFrame,
    fit_periods: Sequence[str],
    tuning_periods: Sequence[str],
    assessment_periods: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map analytic stays to disjoint MIMIC-IV calendar groups."""

    cluster_ids = np.asarray(dataset["cluster_ids"], dtype=np.int64)
    stay_period = (
        stays.loc[:, ["stay_id", "subject_id"]]
        .merge(patients.loc[:, ["subject_id", "anchor_year_group"]], on="subject_id", how="left")
        .set_index("stay_id")["anchor_year_group"]
        .reindex(cluster_ids)
    )
    if stay_period.isna().any():
        raise RuntimeError("Every analytic stay must have an anchor_year_group")

    period_values = stay_period.astype(str).to_numpy()
    fit_idx = np.flatnonzero(np.isin(period_values, tuple(fit_periods)))
    tuning_idx = np.flatnonzero(np.isin(period_values, tuple(tuning_periods)))
    assessment_idx = np.flatnonzero(np.isin(period_values, tuple(assessment_periods)))
    all_idx = np.concatenate([fit_idx, tuning_idx, assessment_idx])
    if np.unique(all_idx).size != all_idx.size:
        raise RuntimeError("Temporal fit, tuning, and assessment periods overlap")
    if np.unique(all_idx).size != cluster_ids.size:
        missing = sorted(set(period_values) - set(fit_periods) - set(tuning_periods) - set(assessment_periods))
        raise RuntimeError(f"Temporal period allocation is incomplete: {missing}")
    if min(fit_idx.size, tuning_idx.size, assessment_idx.size) == 0:
        raise RuntimeError("Temporal allocation produced an empty sample")
    return fit_idx, tuning_idx, assessment_idx, period_values


def paired_interval(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Stay-level normal interval for a paired mean loss difference."""

    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference losses must be paired vectors")
    difference = candidate - reference
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
    return {
        "n_stays": int(difference.size),
        "mean_difference": mean,
        "se": se,
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "candidate_better_fraction": float(np.mean(difference < 0.0)),
    }


def paired_bootstrap_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    """Percentile interval from stay-level paired bootstrap means."""

    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference losses must be paired vectors")
    if replicates < 200:
        raise ValueError("bootstrap requires at least 200 replicates")
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        draw = rng.integers(0, difference.size, size=(stop - start, difference.size))
        means[start:stop] = np.mean(difference[draw], axis=1)
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize_models(
    records: pd.DataFrame,
    period_label: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    population = records["loss_population"].to_numpy(dtype=float)
    model_rows = []
    comparison_rows = []
    for position, model in enumerate(DISPLAY_MODELS):
        losses = records[f"loss_{model}"].to_numpy(dtype=float)
        row = {
            "period": period_label,
            "model": model,
            "n_stays": int(losses.size),
            "mean_stay_level_check_loss": float(np.mean(losses)),
            "se_stay_level_check_loss": float(np.std(losses, ddof=1) / np.sqrt(losses.size)),
            "relative_reduction_vs_population_percent": float(
                100.0 * (np.mean(population) - np.mean(losses)) / np.mean(population)
            ),
        }
        model_rows.append(row)
        if model != "population":
            normal = paired_interval(losses, population)
            bootstrap = paired_bootstrap_interval(
                losses,
                population,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1009 * position,
            )
            comparison_rows.append(
                {
                    "period": period_label,
                    "candidate": model,
                    "reference": "population",
                    **normal,
                    "bootstrap_ci95_low": bootstrap["ci95_low"],
                    "bootstrap_ci95_high": bootstrap["ci95_high"],
                    "bootstrap_replicates": bootstrap["replicates"],
                    "bootstrap_seed": bootstrap["seed"],
                }
            )

    q10 = records["loss_calibrated_q10"].to_numpy(dtype=float)
    profile = records["loss_primary_level_update"].to_numpy(dtype=float)
    normal = paired_interval(q10, profile)
    bootstrap = paired_bootstrap_interval(
        q10,
        profile,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 7919,
    )
    comparison_rows.append(
        {
            "period": period_label,
            "candidate": "calibrated_q10",
            "reference": "primary_level_update",
            **normal,
            "bootstrap_ci95_low": bootstrap["ci95_low"],
            "bootstrap_ci95_high": bootstrap["ci95_high"],
            "bootstrap_replicates": bootstrap["replicates"],
            "bootstrap_seed": bootstrap["seed"],
        }
    )
    return pd.DataFrame(model_rows), pd.DataFrame(comparison_rows)


def q10_profile_strata(records: pd.DataFrame, groups: int = 3) -> pd.DataFrame:
    """Locate the paired q10 advantage across the index lower tail distribution."""

    if groups != 3:
        raise ValueError("The manuscript summary uses three equal count groups")
    q10 = records["index_q10"].to_numpy(dtype=float)
    stable_index = records["global_stay_index"].to_numpy(dtype=np.int64)
    order = np.lexsort((stable_index, q10))
    labels = np.empty(order.size, dtype=int)
    labels[order] = np.minimum(np.arange(order.size) * groups // order.size, groups - 1)
    difference = (
        records["loss_calibrated_q10"].to_numpy(dtype=float)
        - records["loss_primary_level_update"].to_numpy(dtype=float)
    )
    names = ("Lower q10 third", "Middle q10 third", "Upper q10 third")
    simultaneous_critical = float(NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * groups)))
    rows = []
    for group, name in enumerate(names):
        mask = labels == group
        values = difference[mask]
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(values.size))
        rows.append(
            {
                "stratum": name,
                "n_stays": int(values.size),
                "median_index_q10": float(np.median(q10[mask])),
                "mean_q10_minus_profile_loss": mean,
                "ci95_low": float(mean - 1.96 * se),
                "ci95_high": float(mean + 1.96 * se),
                "simultaneous_ci95_low": float(mean - simultaneous_critical * se),
                "simultaneous_ci95_high": float(mean + simultaneous_critical * se),
                "simultaneous_critical": simultaneous_critical,
                "q10_better_stay_fraction": float(np.mean(values < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def write_supplement_tex(
    models: pd.DataFrame,
    comparisons: pd.DataFrame,
    strata: pd.DataFrame,
    calibration: Mapping[str, Mapping[str, object]],
    selected_model: str,
    selected_lambda: float,
    path: Path,
) -> None:
    labels = {
        "population": "Population component",
        "primary_level_update": "Profiled offset",
        "calibrated_q10": "Calibrated q10",
    }
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Temporal assessment of the prediction family. Panel A gives loss by MIMIC-IV anchor year group, with differences defined as candidate minus population loss. Panel B gives the penalty and affine calibration selected from the tuning period. Panel C compares calibrated q10 with the profiled offset across the index q10 distribution.}",
        "\\label{tab:supp_temporal_validation}",
        "\\centering",
        "\\footnotesize",
        "\\textit{Panel A. Assessment loss by period.}\\par\\smallskip",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Assessment period & Rule & Stays & Mean loss & Difference (95\\% CI)\\\\",
        "\\midrule",
    ]
    period_labels = {"2017 - 2019": "2017 to 2019", "2020 - 2022": "2020 to 2022"}
    for period in ("2017 - 2019", "2020 - 2022"):
        period_models = models.loc[models["period"] == period].set_index("model")
        period_comparisons = comparisons.loc[
            (comparisons["period"] == period) & (comparisons["reference"] == "population")
        ].set_index("candidate")
        for model in DISPLAY_MODELS:
            row = period_models.loc[model]
            if model == "population":
                difference = "Reference"
            else:
                comp = period_comparisons.loc[model]
                difference = f"{comp['mean_difference']:.4f} ({comp['ci95_low']:.4f}, {comp['ci95_high']:.4f})"
            lines.append(
                f"{period_labels[period]} & {labels[model]} & {int(row['n_stays']):,} & "
                f"{row['mean_stay_level_check_loss']:.4f} & {difference}\\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\medskip",
            "\\textit{Panel B. Tuning results from 2014 to 2016.}\\par\\smallskip",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Calibrated summary & Intercept & Slope & Tuning loss\\\\",
            "\\midrule",
        ]
    )
    for model, result in calibration.items():
        label = model.replace("calibrated_", "").replace("_", " ")
        marker = " $^{*}$" if model == selected_model else ""
        lines.append(
            f"{label}{marker} & {float(result['intercept']):.4f} & {float(result['slope']):.4f} & "
            f"{float(result['tuning_mean_stay_level_check_loss']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\begin{{flushleft}}\\footnotesize The profiled offset selected $\\lambda_b={selected_lambda:g}$. "
            "$^{*}$Selected calibrated summary.\\end{flushleft}",
            "\\par\\medskip",
            "\\textit{Panel C. Calibrated q10 minus profile loss by index q10.}\\par\\smallskip",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Index stratum & Stays & Median q10 & Mean difference (simultaneous 95\\% CI) & q10 better (\\%)\\\\",
            "\\midrule",
        ]
    )
    for _, row in strata.iterrows():
        lines.append(
            f"{row['stratum']} & {int(row['n_stays']):,} & {row['median_index_q10']:.1f} & "
            f"{row['mean_q10_minus_profile_loss']:.4f} ({row['simultaneous_ci95_low']:.4f}, {row['simultaneous_ci95_high']:.4f}) & "
            f"{100.0 * row['q10_better_stay_fraction']:.1f}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--patients-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "validation_aggregate")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--supplement-table",
        type=Path,
        default=repo_root / "supplement_aggregate" / "supp_temporal_validation.tex",
    )
    parser.add_argument("--fit-periods", type=parse_periods, default=DEFAULT_FIT_PERIODS)
    parser.add_argument("--tuning-periods", type=parse_periods, default=DEFAULT_TUNING_PERIODS)
    parser.add_argument("--assessment-periods", type=parse_periods, default=DEFAULT_ASSESSMENT_PERIODS)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--index-hours", type=float, default=12.0)
    parser.add_argument(
        "--lambda-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.03,0.10,0.30,1,3,10"),
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    patients = pd.read_csv(
        args.patients_file,
        compression="infer",
        usecols=["subject_id", "anchor_year_group"],
    )
    dataset, data_summary, _ = build_dataset_from_cache(
        obs,
        stays,
        fit_stays=0,
        seed=args.seed,
        analysis_hours=24.0,
    )
    fit_idx, tuning_idx, assessment_idx, period_values = temporal_indices(
        dataset,
        stays,
        patients,
        args.fit_periods,
        args.tuning_periods,
        args.assessment_periods,
    )
    if min(fit_idx.size, tuning_idx.size, assessment_idx.size) < 20:
        raise RuntimeError("Temporal allocation produced too few stays for estimation")

    standardized, age_info = apply_training_age_standardization(dataset, stays, fit_idx)
    fit_design, _ = design_frame(data_utils.subset_cluster_data(standardized, fit_idx))
    tuning_design, _ = design_frame(data_utils.subset_cluster_data(standardized, tuning_idx))
    assessment_design, _ = design_frame(data_utils.subset_cluster_data(standardized, assessment_idx))
    r_script = script_dir / "fit_quantile_common.R"
    resume = not args.no_resume
    beta = fit_population_component(
        fit_design,
        args.tau,
        args.work_dir / "population",
        r_script,
        resume,
    )
    tuning_profile = tune_lambda_generic(
        tuning_design,
        X_PREDICTORS,
        beta,
        tau=args.tau,
        index_hours=args.index_hours,
        lambda_grid=args.lambda_grid,
    )
    selected_lambda = float(tuning_profile["best"]["lambda_b"])
    lookup = global_index_lookup(dataset)
    tuning_records = build_stay_records(
        tuning_design,
        beta,
        selected_lambda,
        args.tau,
        args.index_hours,
        lookup,
    )
    assessment_records = build_stay_records(
        assessment_design,
        beta,
        selected_lambda,
        args.tau,
        args.index_hours,
        lookup,
    )
    calibrations = fit_all_calibrations(
        tuning_records,
        args.tau,
        args.work_dir / "calibrations",
        r_script,
        resume,
    )
    selected_calibration = select_calibration_model(calibrations)
    assessment_records, model_columns = add_prediction_columns(
        assessment_records,
        calibrations,
        selected_calibration,
    )
    assessment_records = attach_model_metrics(assessment_records, model_columns, args.tau)

    overall_models, overall_comparisons = summarize_models(
        assessment_records,
        "2017 to 2022",
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    model_frames = [overall_models]
    comparison_frames = [overall_comparisons]
    assessment_period_values = period_values[assessment_idx]
    for offset, period in enumerate(args.assessment_periods, start=1):
        mask = assessment_period_values == period
        period_models, period_comparisons = summarize_models(
            assessment_records.loc[mask].reset_index(drop=True),
            period,
            args.bootstrap_replicates,
            args.bootstrap_seed + 100000 * offset,
        )
        model_frames.append(period_models)
        comparison_frames.append(period_comparisons)
    models = pd.concat(model_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    strata = q10_profile_strata(assessment_records)

    models.to_csv(args.output_dir / "temporal_validation_model_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "temporal_validation_paired_comparisons.csv", index=False)
    strata.to_csv(args.output_dir / "temporal_validation_q10_profile_strata.csv", index=False)
    write_supplement_tex(
        models,
        comparisons,
        strata,
        calibrations,
        selected_calibration,
        selected_lambda,
        args.supplement_table,
    )
    payload = {
        "design": {
            "fit_periods": list(args.fit_periods),
            "tuning_periods": list(args.tuning_periods),
            "assessment_periods": list(args.assessment_periods),
            "fit_stays": int(fit_idx.size),
            "tuning_stays": int(tuning_idx.size),
            "assessment_stays": int(assessment_idx.size),
            "age_standardization": age_info,
            "tau": float(args.tau),
            "index_hours": float(args.index_hours),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "population_coefficients": beta.tolist(),
        "penalty_tuning": tuning_profile,
        "selected_lambda_b": selected_lambda,
        "calibrations": calibrations,
        "selected_calibration_model": selected_calibration,
        "model_summary": models.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "q10_profile_strata": strata.to_dict(orient="records"),
        "analytic_data_summary": data_summary,
    }
    write_json(args.output_dir / "temporal_validation_results.json", json_safe(payload))


if __name__ == "__main__":
    main()
