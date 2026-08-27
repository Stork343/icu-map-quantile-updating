import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import split_window_data as data_utils
from split_window_analysis_core import (
    auc_score,
    check_loss,
    design_frame,
    admission_window_q10_strata,
    empirical_check_quantile,
    fit_common_quantile,
    profiled_intercept,
    split_indices_for_hours,
)
from split_window_clinical_core import (
    logistic_or_per_10,
    outcome_frame,
    risk_strata_table,
    stay_level_features,
    validation_losses_generic,
)
from split_window_data import _safe_read_frame, build_dataset_from_cache


def split_cluster_indices(
    n_stays: int,
    seed: int,
    train_fraction: float,
    tuning_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < tuning_fraction < 1.0):
        raise ValueError("tuning_fraction must be between 0 and 1")
    if train_fraction + tuning_fraction >= 1.0:
        raise ValueError("train_fraction + tuning_fraction must be less than 1")
    rng = np.random.default_rng(seed + 2026)
    perm = rng.permutation(n_stays)
    n_train = int(round(train_fraction * n_stays))
    n_tuning = int(round(tuning_fraction * n_stays))
    train_idx = np.sort(perm[:n_train])
    tuning_idx = np.sort(perm[n_train : n_train + n_tuning])
    assessment_idx = np.sort(perm[n_train + n_tuning :])
    return train_idx, tuning_idx, assessment_idx


def se_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def loss_row(model: str, index_information: str, losses: np.ndarray) -> Dict[str, object]:
    losses = np.asarray(losses, dtype=float)
    return {
        "model": model,
        "index_information": index_information,
        "assessment_loss": float(np.mean(losses)),
        "assessment_loss_se": se_mean(losses),
        "n_stays": int(losses.size),
    }


DIRECT_IDENTIFIER_COLUMNS = ("subject_id", "hadm_id", "stay_id")


def public_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in DIRECT_IDENTIFIER_COLUMNS if col in df.columns])


def apply_training_age_standardization(
    dataset: Dict[str, object],
    stays: pd.DataFrame,
    train_idx: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    cluster_ids = np.asarray(dataset["cluster_ids"], dtype=np.int64)
    age_by_stay = stays.set_index("stay_id")["age"]
    ages = age_by_stay.loc[cluster_ids].to_numpy(dtype=float)
    train_ages = ages[np.asarray(train_idx, dtype=int)]
    age_mean = float(np.mean(train_ages))
    age_sd = float(np.std(train_ages, ddof=1))
    if not np.isfinite(age_sd) or age_sd <= 0.0:
        age_sd = 1.0

    out = dict(dataset)
    x = np.asarray(out["X"], dtype=float).copy()
    x[:, 1] = (ages - age_mean) / age_sd
    out["X"] = x
    repeats = np.asarray([len(y_i) for y_i in out["y_list"]], dtype=int)
    out["X_long"] = np.repeat(x, repeats, axis=0)
    return out, {
        "age_mean_training_split": age_mean,
        "age_sd_training_split": age_sd,
    }


def cohort_summary_from_inputs(
    data_root: Path,
    obs: pd.DataFrame,
    stays: pd.DataFrame,
    data_summary: Dict[str, object],
) -> Dict[str, object]:
    paths = data_utils.MimicPaths.from_root(data_root)
    patients = data_utils._load_patients(paths.patients_path)
    admissions = data_utils._load_admissions(paths.admissions_path)
    candidate = data_utils._load_icu_stays(
        paths.icustays_path,
        patients,
        admissions,
        analysis_hours=24.0,
        keep_one_stay_per_subject=True,
    )
    total_obs = int(obs.shape[0])
    full_low = int(np.sum(obs["map_value"].to_numpy(dtype=float) < 65.0))
    split_counts = obs.groupby("stay_id").agg(
        index_count=("time_hours", lambda x: int(np.sum(np.asarray(x) <= 12.0))),
        late_count=("time_hours", lambda x: int(np.sum(np.asarray(x) > 12.0))),
    )
    split_eligible = int(((split_counts["index_count"] >= 4) & (split_counts["late_count"] >= 1)).sum())
    eligible_obs = int(data_summary["fit_observations"])
    eligible_low = int(round(float(data_summary["map_below_65_observation_fraction"]) * eligible_obs))
    return {
        "eligible_adult_first_icu_stays": int(len(candidate)),
        "excluded_before_analytic_cohort": int(len(candidate) - stays.shape[0]),
        "analytic_icu_stays": int(stays.shape[0]),
        "unique_subjects": int(stays["subject_id"].nunique()) if "subject_id" in stays.columns else int(stays.shape[0]),
        "total_map_observations": total_obs,
        "excluded_by_split_window_rule": int(stays.shape[0] - split_eligible),
        "split_window_eligible_stays": split_eligible,
        "not_included_in_analysis_subset": int(split_eligible - int(data_summary["fit_stays"])),
        "full_map_below65_count": full_low,
        "full_map_below65_percent": 100.0 * full_low / total_obs,
        "eligible_map_below65_count": eligible_low,
        "eligible_map_below65_percent": 100.0 * eligible_low / eligible_obs,
    }


def tune_lambda_generic(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    best = None
    for lam in lambda_grid:
        losses, _ = validation_losses_generic(
            design,
            predictor_cols,
            gamma,
            tau=tau,
            index_hours=index_hours,
            update="profiled",
            lambda_b=float(lam),
        )
        row = {
            "tau": float(tau),
            "index_hours": float(index_hours),
            "lambda_b": float(lam),
            "tuning_loss": float(np.mean(losses)),
            "tuning_loss_se": se_mean(losses),
            "n_stays": int(losses.size),
        }
        rows.append(row)
        if best is None or row["tuning_loss"] < best["tuning_loss"]:
            best = row
    assert best is not None
    return {"best": best, "grid": rows}


def evaluate_population_and_update(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_b: float,
) -> Dict[str, object]:
    pop_losses, _ = validation_losses_generic(design, predictor_cols, gamma, tau, index_hours, "none")
    update_losses, _ = validation_losses_generic(
        design,
        predictor_cols,
        gamma,
        tau,
        index_hours,
        "profiled",
        lambda_b=lambda_b,
    )
    pop = float(np.mean(pop_losses))
    update = float(np.mean(update_losses))
    paired_diff = np.asarray(pop_losses, dtype=float) - np.asarray(update_losses, dtype=float)
    paired_diff_mean = float(np.mean(paired_diff))
    paired_diff_se = se_mean(paired_diff)
    return {
        "population_losses": pop_losses,
        "update_losses": update_losses,
        "population_loss": pop,
        "population_loss_se": se_mean(pop_losses),
        "updated_loss": update,
        "updated_loss_se": se_mean(update_losses),
        "paired_loss_reduction": paired_diff_mean,
        "paired_loss_reduction_se": paired_diff_se,
        "paired_loss_reduction_ci95": [
            float(paired_diff_mean - 1.96 * paired_diff_se),
            float(paired_diff_mean + 1.96 * paired_diff_se),
        ],
        "loss_reduction_percent": float(100.0 * (pop - update) / pop),
        "n_stays": int(update_losses.size),
    }


def write_strata_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Later hypotension burden by quintile of admission window MAP 0.10 quantile in the assessment split.}",
        "\\label{tab:admission_q10_strata}",
        "\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Admission window MAP 0.10 quantile & Stays & Adm. q10 & Later q10 & Later MAP $<$65 & Any later MAP $<$65\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['stratum']} & {int(row['stays'])} & {float(row['admission_window_q10_median']):.1f} & "
            f"{float(row['later_q10_median']):.1f} & {float(row['later_map_below65_fraction']):.3f} & "
            f"{float(row['any_later_map_below65']):.3f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Admission window q10 and later q10 are median stay level MAP 0.10 quantiles in mmHg within each stratum. Mutually exclusive strata use empirical 20th, 40th, 60th, and 80th percentile cut points; observations equal to a cut point enter the next higher-q10 stratum, so discrete q10 ties can make stratum sizes unequal. Later MAP $<$65 is the mean within-stay fraction of later recorded MAP values below 65 mmHg.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_comparison_tex(
    rows: Sequence[Dict[str, object]],
    path: Path,
    paired_stats: Optional[Dict[str, object]] = None,
) -> None:
    paired_sentence = ""
    if paired_stats is not None:
        ci_low, ci_high = paired_stats["paired_loss_reduction_ci95"]
        paired_sentence = (
            " The paired mean loss reduction for the primary baseline-adjusted level update "
            "versus the baseline covariate quantile model was "
            f"{float(paired_stats['paired_loss_reduction']):.4f} "
            f"(SE {float(paired_stats['paired_loss_reduction_se']):.4f}; "
            f"descriptive interval {float(ci_low):.4f}--{float(ci_high):.4f})."
        )
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Split-window assessment of the primary stay-specific level update and comparator rules.}",
        "\\label{tab:model_comparison}",
        "\\centering",
        "\\begin{tabular}{llrr}",
        "\\hline",
        "Model & Index window information used & Assessment loss & SE\\\\",
        "\\hline",
    ]
    display_models = {
        "Spline + penalized random-intercept update": "Spline + penalized stay-specific level update",
        "Selected baseline-adjusted random-intercept update": "Primary baseline-adjusted level update",
    }
    display_information = {"profiled intercept": "profiled level offset"}
    for row in rows:
        lines.append(
            f"{display_models.get(str(row['model']), row['model'])} & "
            f"{display_information.get(str(row['index_information']), row['index_information'])} & "
            f"{float(row['assessment_loss']):.4f} & {float(row['assessment_loss_se']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Assessment loss is the mean within-stay ordinary check loss on later observations at $\\tau=0.10$, averaged across assessment stays. Population components are fitted on training stays and quadratic penalties are selected on tuning stays. Because the assessment outcomes had been examined during earlier work, these values are internal evaluations rather than an untouched external validation."
            + paired_sentence
            + "}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Sensitivity analyses for the primary penalized stay-specific level update.}",
        "\\label{tab:sensitivity_analysis}",
        "\\centering",
        "\\begin{tabular}{rrrrrr}",
        "\\hline",
        "$\\tau$ & Nominal window (h) & $\\lambda_b$ & Pop. loss & Updated loss & Reduction\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{float(row['tau']):.2f} & {float(row['index_hours']):.0f} & {float(row['lambda_b']):.2g} & "
            f"{float(row['population_assessment_loss']):.4f} & {float(row['updated_assessment_loss']):.4f} & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Population loss is the corresponding baseline-covariate quantile assessment loss. Updated loss adds the penalized stay-specific level offset. The penalty is selected on tuning stays for each quantile/window setting, and losses are reported descriptively on assessment stays. Window values are nominal clock-time cut points; if the clock-time split did not leave the required index and later observations, the prespecified observation-count fallback was used.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outcome_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Later hypotension and clinical outcomes by admission window lower tail vulnerability quintile in the assessment split.}",
        "\\label{tab:clinical_outcome_strata}",
        "\\centering",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Vulnerability stratum & Stays & Adm. q10 & Later MAP $<$65 & Hosp. mortality & ICU mortality & ICU LOS & ICU LOS $\\ge$3d\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['stratum']} & {int(row['stays'])} & {float(row['admission_window_q10_median']):.1f} & "
            f"{100.0 * float(row['later_map_below65_fraction']):.1f}\\% & "
            f"{100.0 * float(row['hospital_mortality']):.1f}\\% & "
            f"{100.0 * float(row['icu_mortality']):.1f}\\% & "
            f"{float(row['icu_los_median']):.1f} & {100.0 * float(row['icu_los_ge3_days']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Strata are quintiles of $-$admission window MAP 0.10 quantile, so higher strata correspond to lower admission window lower tail MAP. Boundary ties are assigned to the next higher-vulnerability stratum. Later MAP $<$65 is the mean within stay fraction of later recorded MAP values below 65 mmHg. ICU LOS is the median ICU length of stay in days.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cohort_tex(
    path: Path,
    data_summary: Dict[str, object],
    split_counts: Dict[str, int],
    cohort_summary: Dict[str, object],
) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Cohort construction, MAP preprocessing, and stay level analysis split for the MIMIC-IV application.}",
        "\\label{tab:app_cohort}",
        "\\centering",
        "\\begin{tabular}{lc}",
        "\\hline",
        "Quantity & Value\\\\",
        "\\hline",
        "MIMIC-IV version & mimic-iv-3.1\\\\",
        f"Eligible adult first ICU stays & {int(cohort_summary['eligible_adult_first_icu_stays'])}\\\\",
        f"Excluded before analytic cohort & {int(cohort_summary['excluded_before_analytic_cohort'])}\\\\",
        f"Analytic ICU stays & {int(cohort_summary['analytic_icu_stays'])}\\\\",
        f"Unique subjects & {int(cohort_summary['unique_subjects'])}\\\\",
        f"Total MAP observations & {int(cohort_summary['total_map_observations'])}\\\\",
        f"Excluded by 12 h split-window rule & {int(cohort_summary['excluded_by_split_window_rule'])}\\\\",
        f"12 h split-window eligible stays & {int(cohort_summary['split_window_eligible_stays'])}\\\\",
        f"Not included in analysis subset & {int(cohort_summary['not_included_in_analysis_subset'])}\\\\",
        f"Split-window analysis stays & {int(data_summary['fit_stays'])}\\\\",
        f"Split-window analysis observations & {int(data_summary['fit_observations'])}\\\\",
        f"Training stays & {split_counts['train']}\\\\",
        f"Tuning stays & {split_counts['tuning']}\\\\",
        f"Assessment stays & {split_counts['assessment']}\\\\",
        "Median MAP observations per analysis stay (IQR) & "
        f"{float(data_summary['obs_per_stay_median']):.1f} "
        f"({float(data_summary['obs_per_stay_iqr'][0]):.1f}--{float(data_summary['obs_per_stay_iqr'][1]):.1f})\\\\",
        "Minimum cache observation count criterion & 8\\\\",
        "Primary 12 h split rule & $\\ge$4 index obs.; $\\ge$1 later obs.\\\\",
        "MAP item identifiers & ABP mean 220052; NIBP mean 220181\\\\",
        "Physiologic MAP filter & 20.0--200.0 mmHg\\\\",
        "Duplicate handling rule & 5 minute bucket mean\\\\",
        "Invasive/noninvasive priority & invasive MAP preferred within bucket\\\\",
        "Full analytic MAP below 65 mmHg & "
        f"{int(cohort_summary['full_map_below65_count'])} ({float(cohort_summary['full_map_below65_percent']):.1f}\\%)\\\\",
        "Split-window analysis MAP below 65 mmHg & "
        f"{int(cohort_summary['eligible_map_below65_count'])} ({float(cohort_summary['eligible_map_below65_percent']):.1f}\\%)\\\\",
        "\\hline",
        "\\end{tabular}",
        "\\par\\smallskip",
        "\\footnotesize{All split-window eligible stays are included when the requested analysis size is at least the number eligible in the rebuilt cache.}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_admission_window_q10_strata(strata_csv: Path, output_stem: Path) -> None:
    df = pd.read_csv(strata_csv)
    labels = ["Lowest", "Second", "Third", "Fourth", "Highest"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, 100.0 * df["any_later_map_below65"], marker="o", lw=2.2, label="Any later MAP <65")
    ax.plot(x, 100.0 * df["later_map_below65_fraction"], marker="s", lw=2.2, label="Later observations <65")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Later hypotension burden (%)")
    ax.set_xlabel("Quintile of admission window MAP 0.10 quantile")
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_all(fig, output_stem)
    plt.close(fig)


def plot_validation_sensitivity(rows: Sequence[Dict[str, object]], output_stem: Path) -> None:
    labels = []
    pop = []
    update = []
    reductions = []
    for row in rows:
        if abs(float(row["tau"]) - 0.10) < 1e-12:
            label = f"{int(float(row['index_hours']))}h"
        else:
            label = rf"$\tau={float(row['tau']):.2f}$"
        labels.append(label)
        pop.append(float(row["population_assessment_loss"]))
        update.append(float(row["updated_assessment_loss"]))
        reductions.append(float(row["loss_reduction_percent"]))
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, pop, width, label="Population", color="0.72")
    ax.bar(x + width / 2, update, width, label="Trajectory update", color="#1f77b4")
    for xi, yi, red in zip(x, update, reductions):
        ax.text(xi + width / 2, yi + 0.03, f"{red:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Assessment check loss")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, output_stem)
    plt.close(fig)


def plot_clinical_vulnerability(rows: Sequence[Dict[str, object]], output_stem: Path) -> None:
    labels = ["Lowest", "Second", "Third", "Fourth", "Highest"]
    later_low = [100.0 * float(r["later_map_below65_fraction"]) for r in rows]
    mort = [100.0 * float(r["hospital_mortality"]) for r in rows]
    los3 = [100.0 * float(r["icu_los_ge3_days"]) for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
    panels = [
        (later_low, "Later MAP <65 (%)", "#1f77b4"),
        (mort, "Hospital mortality (%)", "#d62728"),
        (los3, "ICU LOS >=3 days (%)", "#2ca02c"),
    ]
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.bar(x, values, color=color, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save_all(fig, output_stem)
    plt.close(fig)


def stay_level_diagnostics(dataset: Dict[str, object], index_hours: float = 12.0) -> pd.DataFrame:
    enriched = data_utils.ensure_cluster_lists(dataset)
    rows: List[Dict[str, float]] = []
    for y_i, t_i in zip(enriched["y_list"], enriched["t_list"]):
        y_i = np.asarray(y_i, dtype=float)
        t_i = np.asarray(t_i, dtype=float)
        index_idx, late_idx = split_indices_for_hours(t_i, index_hours)
        y_index = y_i[index_idx]
        y_late = y_i[late_idx]
        t_index = t_i[index_idx]
        t_late = t_i[late_idx]
        if y_index.size < 2 or y_late.size < 2:
            continue
        rows.append(
            {
                "admission_window_q10": empirical_check_quantile(y_index, 0.10),
                "later_q10": empirical_check_quantile(y_late, 0.10),
                "admission_window_low_fraction": float(np.mean(y_index < 65.0)),
                "later_low_fraction": float(np.mean(y_late < 65.0)),
                "index_slope": float(np.polyfit(t_index, y_index, deg=1)[0]),
                "later_slope": float(np.polyfit(t_late, y_late, deg=1)[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_persistence_diagnostics(df: pd.DataFrame, output_stem: Path) -> Dict[str, float]:
    rng = np.random.default_rng(20260526)
    sample_n = min(12000, len(df))
    sample = df.iloc[rng.choice(len(df), size=sample_n, replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    panels = [
        ("admission_window_q10", "later_q10", "Admission window q10", "Later q10", (40, 105), (40, 105)),
        (
            "admission_window_low_fraction",
            "later_low_fraction",
            "Admission window MAP <65 fraction",
            "Later MAP <65 fraction",
            (-0.03, 1.03),
            (-0.03, 1.03),
        ),
        ("index_slope", "later_slope", "Index window slope", "Later slope", (-10, 10), (-10, 10)),
    ]
    correlations: Dict[str, float] = {}
    for ax, (xcol, ycol, xlabel, ylabel, xlim, ylim) in zip(axes, panels):
        ax.hexbin(sample[xcol], sample[ycol], gridsize=35, cmap="Blues", mincnt=1)
        r = float(np.corrcoef(df[xcol], df[ycol])[0, 1])
        correlations[f"{xcol}_vs_{ycol}"] = r
        ax.text(0.04, 0.92, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(color="0.9", linewidth=0.6)
    fig.tight_layout()
    save_all(fig, output_stem)
    plt.close(fig)
    return correlations


def plot_lambda_profile(grid_rows: Sequence[Dict[str, object]], output_stem: Path) -> None:
    grid = pd.DataFrame(grid_rows)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(grid["lambda_b"], grid["tuning_loss"], marker="o", lw=2.0, label="Trajectory update")
    best = grid.loc[grid["tuning_loss"].idxmin()]
    ax.scatter([best["lambda_b"]], [best["tuning_loss"]], color="tab:red", zorder=5)
    ax.annotate(
        rf"selected $\lambda_b={float(best['lambda_b']):.2g}$",
        xy=(best["lambda_b"], best["tuning_loss"]),
        xytext=(0.22, best["tuning_loss"] + 0.06),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlabel(r"Ridge penalty $\lambda_b$")
    ax.set_ylabel("Tuning check loss")
    ax.grid(color="0.9", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, output_stem)
    plt.close(fig)


def latex_escape_percent(value: float) -> str:
    return f"{value:.1f}\\%"


def compile_manuscript(tex_dir: Path, tex_name: str) -> None:
    subprocess.run(
        ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
        cwd=tex_dir,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final split based ordinary check loss mixed effects analysis.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("statistics-in-medicine-paper/paper/empirical"))
    parser.add_argument("--output", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_results.json"))
    parser.add_argument("--clinical-output", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_clinical_comparator_results.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_work"))
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, basis_spec = build_dataset_from_cache(
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
    cohort_summary = cohort_summary_from_inputs(args.data_root, obs, stays, data_summary)
    train_data = data_utils.subset_cluster_data(dataset, train_idx)
    tuning_data = data_utils.subset_cluster_data(dataset, tuning_idx)
    assessment_data = data_utils.subset_cluster_data(dataset, assessment_idx)

    train_design, full_predictors = design_frame(train_data)
    tuning_design, _ = design_frame(tuning_data)
    assessment_design, _ = design_frame(assessment_data)
    x_predictors = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
    r_script = Path(__file__).with_name("fit_quantile_common.R")
    lambda_grid = [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]

    baseline_fits: Dict[float, np.ndarray] = {}
    full_fits: Dict[float, np.ndarray] = {}
    for tau_fit in [0.05, 0.10, 0.20]:
        baseline_fits[tau_fit] = fit_common_quantile(
            train_design,
            x_predictors,
            tau=tau_fit,
            work_dir=args.work_dir / f"baseline_tau_{tau_fit:.2f}",
            r_script=r_script,
            force_refit=args.force_refit,
        )
        full_fits[tau_fit] = fit_common_quantile(
            train_design,
            full_predictors,
            tau=tau_fit,
            work_dir=args.work_dir / f"spline_tau_{tau_fit:.2f}",
            r_script=r_script,
            force_refit=args.force_refit,
        )

    tau = 0.10
    index_hours = 12.0
    baseline_gamma = baseline_fits[tau]
    full_gamma = full_fits[tau]
    baseline_tuned = tune_lambda_generic(tuning_design, x_predictors, baseline_gamma, tau, index_hours, lambda_grid)
    spline_tuned = tune_lambda_generic(tuning_design, full_predictors, full_gamma, tau, index_hours, lambda_grid)
    selected_lambda_baseline = float(baseline_tuned["best"]["lambda_b"])
    selected_lambda_spline = float(spline_tuned["best"]["lambda_b"])

    base_eval = evaluate_population_and_update(
        assessment_design,
        x_predictors,
        baseline_gamma,
        tau,
        index_hours,
        selected_lambda_baseline,
    )
    spline_eval = evaluate_population_and_update(
        assessment_design,
        full_predictors,
        full_gamma,
        tau,
        index_hours,
        selected_lambda_spline,
    )
    admission_window_q10_losses, _ = validation_losses_generic(
        assessment_design, x_predictors, baseline_gamma, tau, index_hours, "admission_window_q10"
    )
    full_unpen_losses, _ = validation_losses_generic(
        assessment_design,
        full_predictors,
        full_gamma,
        tau,
        index_hours,
        "profiled",
        lambda_b=0.0,
    )
    comparator_rows = [
        loss_row("Baseline covariates quantile", "none", base_eval["population_losses"]),
        loss_row("Admission window q10 carry forward", "raw admission window q10", admission_window_q10_losses),
        loss_row("Population spline quantile", "none", spline_eval["population_losses"]),
        loss_row("Spline + unpenalized update", "index residual q10", full_unpen_losses),
        loss_row("Spline + penalized random-intercept update", "profiled intercept", spline_eval["update_losses"]),
        loss_row("Selected baseline-adjusted random-intercept update", "profiled intercept", base_eval["update_losses"]),
    ]

    sensitivity_rows: List[Dict[str, object]] = []
    sensitivity_lambda_grids: Dict[str, object] = {}
    for tau_fit in [0.05, 0.10, 0.20]:
        windows = [6.0, 12.0, 18.0] if abs(tau_fit - 0.10) < 1e-12 else [12.0]
        for window in windows:
            gamma = baseline_fits[tau_fit]
            tuned = tune_lambda_generic(
                tuning_design,
                x_predictors,
                gamma,
                tau=tau_fit,
                index_hours=window,
                lambda_grid=lambda_grid,
            )
            lam = float(tuned["best"]["lambda_b"])
            assessed = evaluate_population_and_update(
                assessment_design,
                x_predictors,
                gamma,
                tau=tau_fit,
                index_hours=window,
                lambda_b=lam,
            )
            row = {
                "tau": float(tau_fit),
                "index_hours": float(window),
                "lambda_b": lam,
                "population_assessment_loss": float(assessed["population_loss"]),
                "population_assessment_loss_se": float(assessed["population_loss_se"]),
                "updated_assessment_loss": float(assessed["updated_loss"]),
                "updated_assessment_loss_se": float(assessed["updated_loss_se"]),
                "n_stays": int(assessed["n_stays"]),
                "loss_reduction_percent": float(assessed["loss_reduction_percent"]),
            }
            sensitivity_rows.append(row)
            sensitivity_lambda_grids[f"tau_{tau_fit:.2f}_index_{window:.0f}"] = tuned["grid"]

    strata_rows, aucs = admission_window_q10_strata(assessment_data, index_hours=index_hours)
    features = stay_level_features(
        assessment_data,
        assessment_design,
        x_predictors,
        baseline_gamma,
        tau=tau,
        index_hours=index_hours,
        lambda_b=selected_lambda_baseline,
    )
    outcomes = outcome_frame(args.data_root, features["stay_id"].to_numpy(dtype=np.int64))
    features = features.merge(outcomes, on="stay_id", how="left")
    features = features.dropna(subset=["hospital_mortality", "icu_los_days"])
    clinical_rows = risk_strata_table(features)
    cov = (
        assessment_design.groupby("stay_index", sort=True)[["age_z", "male", "emergency_or_urgent"]]
        .first()
        .reset_index(drop=True)
    )
    features = features.sort_values("stay_index").reset_index(drop=True)
    cov = cov.iloc[features["stay_index"].to_numpy(dtype=int)].to_numpy(dtype=float)
    y_mort = features["hospital_mortality"].to_numpy(dtype=float)
    outcome_metrics = {
        "hospital_mortality_rate": float(np.mean(y_mort)),
        "icu_mortality_rate": float(np.mean(features["icu_mortality"].to_numpy(dtype=float))),
        "icu_los_median": float(np.median(features["icu_los_days"].to_numpy(dtype=float))),
        "hospital_mortality_auc_admission_window_q10_low_is_risk": auc_score(
            y_mort, -features["admission_window_q10"].to_numpy(dtype=float)
        ),
        "hospital_mortality_auc_updated_vulnerability": auc_score(
            y_mort, features["updated_vulnerability_score"].to_numpy(dtype=float)
        ),
        "adjusted_or_hospital_mortality_per_10mmhg_lower_admission_window_q10": logistic_or_per_10(
            y_mort, -features["admission_window_q10"].to_numpy(dtype=float), cov
        ),
        "adjusted_or_hospital_mortality_per_10mmhg_higher_updated_vulnerability": logistic_or_per_10(
            y_mort, features["updated_vulnerability_score"].to_numpy(dtype=float), cov
        ),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparator_rows).to_csv(args.artifact_dir / "model_comparison.csv", index=False)
    pd.DataFrame(comparator_rows).to_csv(args.artifact_dir / "comparator_models.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.artifact_dir / "sensitivity_analysis.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(args.artifact_dir / "admission_window_q10_strata.csv", index=False)
    pd.DataFrame(clinical_rows).to_csv(args.artifact_dir / "clinical_outcome_strata.csv", index=False)
    public_export_frame(features).to_csv(args.artifact_dir / "stay_level_clinical_features.csv", index=False)

    write_model_comparison_tex(
        comparator_rows,
        args.artifact_dir / "model_comparison.tex",
        paired_stats=base_eval,
    )
    write_sensitivity_tex(sensitivity_rows, args.artifact_dir / "sensitivity_analysis.tex")
    write_strata_tex(strata_rows, args.artifact_dir / "admission_window_q10_strata.tex")
    write_outcome_tex(clinical_rows, args.artifact_dir / "clinical_outcome_strata.tex")
    write_cohort_tex(
        args.artifact_dir / "mimic_cohort_table.tex",
        data_summary,
        {"train": int(train_idx.size), "tuning": int(tuning_idx.size), "assessment": int(assessment_idx.size)},
        cohort_summary,
    )

    plot_admission_window_q10_strata(
        args.artifact_dir / "admission_window_q10_strata.csv",
        args.artifact_dir / "admission_window_q10_strata_plot",
    )
    plot_validation_sensitivity(sensitivity_rows, args.artifact_dir / "assessment_sensitivity_plot")
    plot_clinical_vulnerability(clinical_rows, args.artifact_dir / "clinical_vulnerability_plot")
    diagnostics = stay_level_diagnostics(assessment_data, index_hours=index_hours)
    diagnostics.to_csv(args.artifact_dir / "stay_level_diagnostics.csv", index=False)
    diagnostic_correlations = plot_persistence_diagnostics(
        diagnostics,
        args.artifact_dir / "persistence_diagnostics_plot",
    )
    plot_lambda_profile(baseline_tuned["grid"], args.artifact_dir / "lambda_profile_plot")

    payload = {
        "analysis_type": "split_window_mixed_effects_analysis",
        "status": "complete",
        "runtime_seconds": time.time() - t0,
        "data_summary": data_summary,
        "split": {
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "tuning_fraction": args.tuning_fraction,
            "train_stays": int(train_idx.size),
            "tuning_stays": int(tuning_idx.size),
            "assessment_stays": int(assessment_idx.size),
        },
        "age_standardization": age_standardization,
        "cohort_summary": cohort_summary,
        "basis": {
            "Tmax": basis_spec.Tmax,
            "knots": basis_spec.knots,
            "basis_dimension": basis_spec.L,
            "include_intercept": basis_spec.include_intercept,
            "center_basis": basis_spec.center_basis,
            "scale_basis": basis_spec.scale_basis,
        },
        "settings": {
            "lambda_grid": lambda_grid,
            "tau_values": [0.05, 0.10, 0.20],
            "index_windows_tau_010": [6.0, 12.0, 18.0],
            "common_fit": "ordinary quantile regression via quantreg::rq.fit(method='fn') on the training split",
            "stay_offset": "ordinary check loss profiled intercept from index observations",
            "selected_lambda_baseline_update": selected_lambda_baseline,
            "selected_lambda_spline_update": selected_lambda_spline,
        },
        "model_comparison_rows": comparator_rows,
        "selected_vs_baseline_paired_loss_reduction": {
            "mean": float(base_eval["paired_loss_reduction"]),
            "se": float(base_eval["paired_loss_reduction_se"]),
            "ci95": base_eval["paired_loss_reduction_ci95"],
        },
        "sensitivity_rows": sensitivity_rows,
        "lambda_grids": {
            "baseline_tau_0.10_index_12": baseline_tuned["grid"],
            "spline_tau_0.10_index_12": spline_tuned["grid"],
            **sensitivity_lambda_grids,
        },
        "admission_window_q10_strata": strata_rows,
        "auc": aucs,
        "diagnostic_correlations": diagnostic_correlations,
        "clinical_outcome_strata": clinical_rows,
        "outcome_metrics": outcome_metrics,
        "coefficients": {
            f"baseline_tau_{tau_key:.2f}": val.tolist() for tau_key, val in baseline_fits.items()
        }
        | {f"spline_tau_{tau_key:.2f}": val.tolist() for tau_key, val in full_fits.items()},
        "artifacts": {
            "model_comparison_tex": str(args.artifact_dir / "model_comparison.tex"),
            "sensitivity_tex": str(args.artifact_dir / "sensitivity_analysis.tex"),
            "admission_q10_strata_tex": str(args.artifact_dir / "admission_window_q10_strata.tex"),
            "clinical_outcome_strata_tex": str(args.artifact_dir / "clinical_outcome_strata.tex"),
            "model_comparison_csv": str(args.artifact_dir / "model_comparison.csv"),
            "sensitivity_csv": str(args.artifact_dir / "sensitivity_analysis.csv"),
            "admission_q10_strata_csv": str(args.artifact_dir / "admission_window_q10_strata.csv"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    clinical_payload = {
        "analysis_type": "split_window_clinical_comparator_analysis",
        "status": "complete",
        "settings": payload["settings"] | payload["split"],
        "comparator_rows": comparator_rows,
        "sensitivity_rows": sensitivity_rows,
        "clinical_outcome_strata": clinical_rows,
        "outcome_metrics": outcome_metrics,
        "artifacts": payload["artifacts"],
    }
    args.clinical_output.parent.mkdir(parents=True, exist_ok=True)
    args.clinical_output.write_text(
        json.dumps(data_utils.to_serializable(clinical_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
