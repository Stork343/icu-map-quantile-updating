import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import split_window_data as data_utils
from split_window_analysis_core import auc_score, check_loss, design_frame, empirical_check_quantile, profiled_intercept
from split_window_clinical_core import logistic_or_per_10
from run_split_window_mixed_effects_analysis import apply_training_age_standardization, split_cluster_indices
from split_window_data import _safe_read_frame, build_dataset_from_cache


X_PREDICTORS = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
DIRECT_IDENTIFIER_COLUMNS = ("subject_id", "hadm_id", "stay_id")


def public_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in DIRECT_IDENTIFIER_COLUMNS if col in df.columns])


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def strict_split_indices(
    time_hours: np.ndarray,
    index_hours: float,
    min_index_obs: int = 4,
    min_late_obs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    time_hours = np.asarray(time_hours, dtype=float)
    index_idx = np.flatnonzero(time_hours <= float(index_hours))
    late_idx = np.flatnonzero(time_hours > float(index_hours))
    if index_idx.size < min_index_obs or late_idx.size < min_late_obs:
        return np.array([], dtype=int), np.array([], dtype=int)
    return index_idx, late_idx


def contiguous_stay_slices(design: pd.DataFrame) -> List[Tuple[int, int]]:
    stay_index = design["stay_index"].to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(stay_index)) + 1]
    ends = np.r_[starts[1:], stay_index.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def load_assessment_design(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object]]:
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
    train_idx, _, assessment_idx = split_cluster_indices(
        len(enriched["y_list"]),
        seed=args.seed,
        train_fraction=args.train_fraction,
        tuning_fraction=args.tuning_fraction,
    )
    dataset, age_standardization = apply_training_age_standardization(dataset, stays, train_idx)
    assessment_data = data_utils.subset_cluster_data(dataset, assessment_idx)
    assessment_design, _ = design_frame(assessment_data)
    basis_dimension = int(np.asarray(assessment_data["B_fit_list"][0]).shape[1])
    metadata = {
        "data_summary": data_summary,
        "basis_dimension": basis_dimension,
        "basis_Tmax": float(basis_spec.Tmax),
        "age_standardization": age_standardization,
        "assessment_stays": int(len(assessment_idx)),
    }
    return assessment_data, assessment_design, metadata


def dynamic_signal_table(
    design: pd.DataFrame,
    beta: np.ndarray,
    tau: float,
    lambda_b: float,
    index_windows: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    x_all = design.loc[:, X_PREDICTORS].to_numpy(dtype=float)
    fitted_all = x_all @ beta
    rows: List[Dict[str, object]] = []
    stay_rows: List[Dict[str, object]] = []

    for index_hours in index_windows:
        records: List[Dict[str, float]] = []
        pop_losses: List[float] = []
        update_losses: List[float] = []
        for start, end in contiguous_stay_slices(design):
            y = y_all[start:end]
            t = t_all[start:end]
            fitted = fitted_all[start:end]
            index_idx, late_idx = strict_split_indices(t, index_hours)
            if index_idx.size == 0 or late_idx.size == 0:
                continue
            residual = y - fitted
            b_hat = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
            y_index = y[index_idx]
            y_late = y[late_idx]
            pop_loss = float(np.mean(check_loss(residual[late_idx], tau)))
            updated_loss = float(np.mean(check_loss(residual[late_idx] - b_hat, tau)))
            pop_losses.append(pop_loss)
            update_losses.append(updated_loss)
            records.append(
                {
                    "stay_index": float(design["stay_index"].iat[start]),
                    "stay_id": float(design["stay_id"].iat[start]),
                    "index_hours": float(index_hours),
                    "admission_window_q10": empirical_check_quantile(y_index, tau),
                    "admission_window_map_below65_fraction": float(np.mean(y_index < 65.0)),
                    "later_map_below65_fraction": float(np.mean(y_late < 65.0)),
                    "any_later_map_below65": float(np.any(y_late < 65.0)),
                    "updated_offset": float(b_hat),
                    "updated_vulnerability_score": float(-b_hat),
                    "index_obs": float(index_idx.size),
                    "late_obs": float(late_idx.size),
                    "population_loss": pop_loss,
                    "updated_loss": updated_loss,
                }
            )

        if not records:
            continue
        local = pd.DataFrame(records)
        stay_rows.extend(local.to_dict("records"))
        y_true = local["any_later_map_below65"].to_numpy(dtype=float)
        q20, q80 = np.quantile(local["admission_window_q10"].to_numpy(dtype=float), [0.20, 0.80])
        vulnerable = local[local["admission_window_q10"] <= q20]
        resilient = local[local["admission_window_q10"] >= q80]
        pop = float(np.mean(pop_losses))
        update = float(np.mean(update_losses))
        rows.append(
            {
                "index_hours": float(index_hours),
                "n_stays": int(local.shape[0]),
                "event_rate_any_later_map_below65": float(np.mean(y_true)),
                "median_index_obs": float(local["index_obs"].median()),
                "median_late_obs": float(local["late_obs"].median()),
                "auc_admission_window_q10_low_is_risk": auc_score(y_true, -local["admission_window_q10"].to_numpy(dtype=float)),
                "auc_updated_vulnerability": auc_score(y_true, local["updated_vulnerability_score"].to_numpy(dtype=float)),
                "vulnerable_quintile_any_later_map_below65": float(vulnerable["any_later_map_below65"].mean()),
                "resilient_quintile_any_later_map_below65": float(resilient["any_later_map_below65"].mean()),
                "vulnerable_quintile_later_map_below65_fraction": float(
                    vulnerable["later_map_below65_fraction"].mean()
                ),
                "resilient_quintile_later_map_below65_fraction": float(
                    resilient["later_map_below65_fraction"].mean()
                ),
                "risk_difference_any_later_map_below65": float(
                    vulnerable["any_later_map_below65"].mean() - resilient["any_later_map_below65"].mean()
                ),
                "population_check_loss": pop,
                "updated_check_loss": update,
                "loss_reduction_percent": float(100.0 * (pop - update) / pop),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(stay_rows)


def stay_level_frame(
    design: pd.DataFrame,
    beta: np.ndarray,
    tau: float,
    lambda_b: float,
    index_hours: float,
) -> pd.DataFrame:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    x_all = design.loc[:, X_PREDICTORS].to_numpy(dtype=float)
    fitted_all = x_all @ beta
    rows: List[Dict[str, object]] = []
    for start, end in contiguous_stay_slices(design):
        y = y_all[start:end]
        t = t_all[start:end]
        fitted = fitted_all[start:end]
        index_idx, late_idx = strict_split_indices(t, index_hours)
        if index_idx.size == 0 or late_idx.size == 0:
            continue
        residual = y - fitted
        b_hat = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
        y_index = y[index_idx]
        y_late = y[late_idx]
        rows.append(
            {
                "stay_index": int(design["stay_index"].iat[start]),
                "stay_id": int(design["stay_id"].iat[start]),
                "admission_window_q10": empirical_check_quantile(y_index, tau),
                "index_mean": float(np.mean(y_index)),
                "admission_window_map_below65_fraction": float(np.mean(y_index < 65.0)),
                "index_obs": int(index_idx.size),
                "late_obs": int(late_idx.size),
                "later_q10": empirical_check_quantile(y_late, tau),
                "later_map_below65_fraction": float(np.mean(y_late < 65.0)),
                "any_later_map_below65": float(np.any(y_late < 65.0)),
                "updated_offset": float(b_hat),
                "updated_vulnerability_score": float(-b_hat),
                "population_check_loss": float(np.mean(check_loss(residual[late_idx], tau))),
                "updated_check_loss": float(np.mean(check_loss(residual[late_idx] - b_hat, tau))),
                "age_z": float(x_all[start, 1]),
                "male": float(x_all[start, 2]),
                "emergency_or_urgent": float(x_all[start, 3]),
            }
        )
    return pd.DataFrame(rows)


def attach_outcomes(stay_df: pd.DataFrame, features_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    merge_key = "stay_id" if "stay_id" in features.columns else "stay_index"
    if merge_key not in stay_df.columns:
        raise KeyError(f"Outcome merge key {merge_key!r} is absent from the assessment stay frame.")
    outcome_cols = [
        merge_key,
        "hadm_id",
        "hospital_mortality",
        "icu_mortality",
        "icu_los_days",
        "icu_los_after24h_days",
        "icu_los_ge3_days",
    ]
    outcome_cols = [col for col in outcome_cols if col in features.columns]
    return stay_df.merge(features.loc[:, outcome_cols], on=merge_key, how="left", validate="one_to_one")


def risk_surface_table(stay_df: pd.DataFrame) -> pd.DataFrame:
    q_bins = [-np.inf, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, np.inf]
    q_labels = ["<55", "55-60", "60-65", "65-70", "70-75", "75-80", ">=80"]
    low_bins = [-1e-9, 0.0, 0.10, 0.25, 0.50, 1.0]
    low_labels = ["0", "0-0.10", "0.10-0.25", "0.25-0.50", ">0.50"]
    df = stay_df.copy()
    df["admission_window_q10_bin"] = pd.cut(df["admission_window_q10"], bins=q_bins, labels=q_labels)
    df["admission_window_low_fraction_bin"] = pd.cut(
        df["admission_window_map_below65_fraction"],
        bins=low_bins,
        labels=low_labels,
        include_lowest=True,
    )
    grouped = (
        df.groupby(["admission_window_low_fraction_bin", "admission_window_q10_bin"], observed=False)
        .agg(
            stays=("stay_id", "size"),
            admission_window_q10_median=("admission_window_q10", "median"),
            admission_window_low_fraction_mean=("admission_window_map_below65_fraction", "mean"),
            later_map_below65_fraction=("later_map_below65_fraction", "mean"),
            any_later_map_below65=("any_later_map_below65", "mean"),
            hospital_mortality=("hospital_mortality", "mean"),
        )
        .reset_index()
    )
    return grouped


def decile_summary(stay_df: pd.DataFrame) -> pd.DataFrame:
    df = stay_df.copy()
    df["vulnerability_decile"] = pd.qcut(df["admission_window_q10"].rank(method="first"), 10, labels=False) + 1
    rows: List[Dict[str, object]] = []
    for decile, local in df.groupby("vulnerability_decile", sort=True):
        n = int(local.shape[0])
        later = local["later_map_below65_fraction"].to_numpy(dtype=float)
        hosp = local["hospital_mortality"].to_numpy(dtype=float)
        icu = local["icu_mortality"].to_numpy(dtype=float)
        los3 = local["icu_los_ge3_days"].to_numpy(dtype=float)
        any_later = local["any_later_map_below65"].to_numpy(dtype=float)
        rows.append(
            {
                "decile": int(decile),
                "stays": n,
                "admission_window_q10_median": float(local["admission_window_q10"].median()),
                "later_map_below65_fraction": float(np.mean(later)),
                "later_map_below65_fraction_se": float(np.std(later, ddof=1) / np.sqrt(n)),
                "any_later_map_below65": float(np.mean(any_later)),
                "any_later_map_below65_se": binomial_se(any_later),
                "hospital_mortality": float(np.mean(hosp)),
                "hospital_mortality_se": binomial_se(hosp),
                "icu_mortality": float(np.mean(icu)),
                "icu_mortality_se": binomial_se(icu),
                "icu_los_ge3_days": float(np.mean(los3)),
                "icu_los_ge3_days_se": binomial_se(los3),
                "icu_los_days_median": float(local["icu_los_days"].median()),
            }
        )
    return pd.DataFrame(rows)


def binomial_se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    p = float(np.mean(values))
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / values.size))


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def plot_dynamic_signal(dynamic: pd.DataFrame, output_stem: Path) -> None:
    hours = dynamic["index_hours"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.25))

    ax = axes[0]
    ax.plot(
        hours,
        dynamic["auc_admission_window_q10_low_is_risk"],
        color="#20639B",
        marker="o",
        linewidth=1.8,
        label="Admission window q10",
    )
    ax.plot(
        hours,
        dynamic["auc_updated_vulnerability"],
        color="#D1495B",
        marker="s",
        linewidth=1.8,
        label="Profiled offset",
    )
    ax.set_xlabel("Index window end (hours)")
    ax.set_ylabel("AUC for any later MAP <65")
    ax.set_ylim(0.55, 0.90)
    ax.set_xticks(hours)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "A")

    ax = axes[1]
    ax.plot(
        hours,
        100.0 * dynamic["vulnerable_quintile_any_later_map_below65"],
        color="#D1495B",
        marker="o",
        linewidth=1.9,
        label="Lowest admission window q10 quintile",
    )
    ax.plot(
        hours,
        100.0 * dynamic["resilient_quintile_any_later_map_below65"],
        color="#2A9D8F",
        marker="o",
        linewidth=1.9,
        label="Highest admission window q10 quintile",
    )
    ax.fill_between(
        hours,
        100.0 * dynamic["resilient_quintile_any_later_map_below65"],
        100.0 * dynamic["vulnerable_quintile_any_later_map_below65"],
        color="#D1495B",
        alpha=0.08,
    )
    ax.set_xlabel("Index window end (hours)")
    ax.set_ylabel("Any later MAP <65 (%)")
    ax.set_xticks(hours)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="center right")
    add_panel_label(ax, "B")

    ax = axes[2]
    ax2 = ax.twinx()
    ax2.bar(
        hours,
        dynamic["n_stays"].to_numpy(dtype=float) / 1000.0,
        width=0.7,
        color="0.86",
        alpha=0.8,
        label="Usable stays",
        zorder=0,
    )
    ax.plot(
        hours,
        dynamic["loss_reduction_percent"],
        color="#2F4858",
        marker="D",
        linewidth=1.9,
        label="Profiled offset",
        zorder=3,
    )
    ax.set_xlabel("Index window end (hours)")
    ax.set_ylabel("Check loss reduction (%)")
    ax2.set_ylabel("Usable stays (thousands)")
    ax.set_xticks(hours)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_ylim(0, max(15.0, 1.15 * float(dynamic["loss_reduction_percent"].max())))
    ax2.set_ylim(0, 1.2 * float(dynamic["n_stays"].max()) / 1000.0)
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "C")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_stem)


def plot_risk_surface(surface: pd.DataFrame, deciles: pd.DataFrame, output_stem: Path) -> None:
    q_order = ["<55", "55-60", "60-65", "65-70", "70-75", "75-80", ">=80"]
    low_order = ["0", "0-0.10", "0.10-0.25", "0.25-0.50", ">0.50"]
    heat = (
        surface.pivot(index="admission_window_low_fraction_bin", columns="admission_window_q10_bin", values="any_later_map_below65")
        .reindex(index=low_order, columns=q_order)
        .to_numpy(dtype=float)
    )
    counts = (
        surface.pivot(index="admission_window_low_fraction_bin", columns="admission_window_q10_bin", values="stays")
        .reindex(index=low_order, columns=q_order)
        .fillna(0)
        .to_numpy(dtype=int)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    masked = np.ma.masked_where((~np.isfinite(heat)) | (counts < 25), 100.0 * heat)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(q_order)))
    ax.set_xticklabels(q_order)
    ax.set_yticks(np.arange(len(low_order)))
    ax.set_yticklabels(low_order)
    ax.set_xlabel("Admission window MAP q10 (mmHg)")
    ax.set_ylabel("Admission window MAP <65 fraction")
    ax.set_title("Probability of any later MAP <65")
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            if counts[row, col] >= 25 and np.isfinite(heat[row, col]):
                ax.text(
                    col,
                    row,
                    f"{100.0 * heat[row, col]:.0f}%\nn={counts[row, col]}",
                    ha="center",
                    va="center",
                    color="white" if heat[row, col] > 0.55 else "black",
                    fontsize=7,
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Any later MAP <65 (%)")
    add_panel_label(ax, "A")

    ax = axes[1]
    x = deciles["admission_window_q10_median"].to_numpy(dtype=float)
    series = [
        (
            "Later MAP <65 burden",
            100.0 * deciles["later_map_below65_fraction"].to_numpy(dtype=float),
            100.0 * deciles["later_map_below65_fraction_se"].to_numpy(dtype=float),
            "#20639B",
            "o",
        ),
        (
            "Hospital mortality",
            100.0 * deciles["hospital_mortality"].to_numpy(dtype=float),
            100.0 * deciles["hospital_mortality_se"].to_numpy(dtype=float),
            "#D1495B",
            "s",
        ),
        (
            "ICU LOS >=3 days",
            100.0 * deciles["icu_los_ge3_days"].to_numpy(dtype=float),
            100.0 * deciles["icu_los_ge3_days_se"].to_numpy(dtype=float),
            "#2A9D8F",
            "^",
        ),
    ]
    for label, values, se, color, marker in series:
        ax.plot(x, values, color=color, marker=marker, linewidth=1.8, label=label)
        ax.fill_between(x, values - 1.96 * se, values + 1.96 * se, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Median admission window MAP q10 by decile (mmHg)")
    ax.set_ylabel("Assessment rate or burden (%)")
    ax.set_ylim(0, 50)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Clinical gradient across admission window q10 deciles")
    add_panel_label(ax, "B")

    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_stem)


def select_representative_trajectories(stay_df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    df = stay_df.copy()
    df["decile"] = pd.qcut(df["admission_window_q10"].rank(method="first"), 10, labels=False) + 1
    targets = [
        ("Most vulnerable decile", 1),
        ("Middle decile", 5),
        ("Least vulnerable decile", 10),
    ]
    rows: List[Dict[str, object]] = []
    for label, decile in targets:
        local = df[(df["decile"] == decile) & (df["index_obs"] >= 6) & (df["late_obs"] >= 4)].copy()
        if local.empty:
            local = df[df["decile"] == decile].copy()
        med_index = float(local["admission_window_q10"].median())
        med_later = float(local["later_map_below65_fraction"].median())
        med_obs = float((local["index_obs"] + local["late_obs"]).median())
        score = (
            np.square((local["admission_window_q10"] - med_index) / max(local["admission_window_q10"].std(ddof=0), 1.0))
            + np.square(
                (local["later_map_below65_fraction"] - med_later)
                / max(local["later_map_below65_fraction"].std(ddof=0), 0.05)
            )
            + 0.10 * np.square(((local["index_obs"] + local["late_obs"]) - med_obs) / max(med_obs, 1.0))
        )
        selected = local.iloc[int(np.argmin(score.to_numpy(dtype=float)))].copy()
        selected["trajectory_label"] = label
        rows.append(selected.to_dict())
    out = pd.DataFrame(rows)
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    x_all = design.loc[:, X_PREDICTORS].to_numpy(dtype=float)
    slices = contiguous_stay_slices(design)
    trajectory_rows: List[Dict[str, object]] = []
    for _, selected in out.iterrows():
        stay_index = int(selected["stay_index"])
        start, end = slices[stay_index]
        for local_index in range(start, end):
            trajectory_rows.append(
                {
                    "trajectory_label": selected["trajectory_label"],
                    "stay_index": stay_index,
                    "stay_id": int(selected["stay_id"]),
                    "time_hours": float(t_all[local_index]),
                    "map_value": float(y_all[local_index]),
                    "index_flag": bool(t_all[local_index] <= 12.0),
                    "population_q10": float(x_all[local_index] @ selected["beta"]),
                    "updated_q10": float(x_all[local_index] @ selected["beta"] + selected["updated_offset"]),
                    "admission_window_q10": float(selected["admission_window_q10"]),
                    "later_map_below65_fraction": float(selected["later_map_below65_fraction"]),
                    "hospital_mortality": float(selected.get("hospital_mortality", np.nan)),
                }
            )
    return pd.DataFrame(trajectory_rows)


def plot_representative_trajectories(trajectory_df: pd.DataFrame, output_stem: Path) -> None:
    labels = trajectory_df["trajectory_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(labels), figsize=(10.6, 3.2), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        local = trajectory_df[trajectory_df["trajectory_label"] == label].sort_values("time_hours")
        ax.axhspan(30, 65, color="#F7D6D0", alpha=0.45, linewidth=0)
        ax.axhline(65, color="#B23A48", linestyle="--", linewidth=1.0)
        ax.axvline(12, color="0.35", linestyle=":", linewidth=1.0)
        ax.plot(local["time_hours"], local["map_value"], color="0.45", linewidth=0.9, alpha=0.75)
        index = local["index_flag"].to_numpy(dtype=bool)
        ax.scatter(
            local.loc[index, "time_hours"],
            local.loc[index, "map_value"],
            color="#20639B",
            s=18,
            label="0-12 h" if label == labels[0] else None,
            zorder=3,
        )
        ax.scatter(
            local.loc[~index, "time_hours"],
            local.loc[~index, "map_value"],
            color="#576574",
            s=18,
            label="12-24 h" if label == labels[0] else None,
            zorder=3,
        )
        ax.plot(
            local["time_hours"],
            local["population_q10"],
            color="#2A9D8F",
            linestyle="--",
            linewidth=1.2,
            label="Population q10" if label == labels[0] else None,
        )
        ax.plot(
            local["time_hours"],
            local["updated_q10"],
            color="#D1495B",
            linestyle="-",
            linewidth=1.2,
            label="Profiled-offset q10" if label == labels[0] else None,
        )
        first = local.iloc[0]
        ax.set_title(
            f"{label}\nq10 {first['admission_window_q10']:.1f}, later <65 {100.0 * first['later_map_below65_fraction']:.0f}%"
        )
        ax.set_xlabel("Hours since ICU admission")
        ax.set_xlim(-0.5, 24.5)
        ax.set_ylim(30, 135)
        ax.grid(axis="y", color="0.88", linewidth=0.7)
    axes[0].set_ylabel("MAP (mmHg)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=4)
    fig.tight_layout()
    save_figure(fig, output_stem)


def binned_trajectory_atlas(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    n_bins: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = stay_df.sort_values("admission_window_q10", ascending=True)["stay_index"].to_numpy(dtype=int)
    lookup = stay_df.set_index("stay_index")
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    slices = contiguous_stay_slices(design)
    bins = np.linspace(0.0, 24.0, n_bins + 1)
    atlas = np.full((order.size, n_bins), np.nan, dtype=float)
    for row, stay_index in enumerate(order):
        start, end = slices[int(stay_index)]
        y = y_all[start:end]
        t = t_all[start:end]
        bin_index = np.searchsorted(bins, t, side="right") - 1
        valid = (bin_index >= 0) & (bin_index < n_bins)
        for col in np.unique(bin_index[valid]):
            atlas[row, int(col)] = float(np.median(y[valid & (bin_index == col)]))
    admission_window_q10 = lookup.loc[order, "admission_window_q10"].to_numpy(dtype=float)
    later_burden = lookup.loc[order, "later_map_below65_fraction"].to_numpy(dtype=float)
    mortality = lookup.loc[order, "hospital_mortality"].to_numpy(dtype=float)
    return atlas, bins, admission_window_q10, later_burden, mortality


def clinical_yield_tables(stay_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = (
        stay_df.sort_values(
            ["admission_window_q10", "stay_index"],
            ascending=[True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )
    df["later_low_count"] = df["later_map_below65_fraction"] * df["late_obs"]
    df["any_later_low_count"] = df["any_later_map_below65"]
    n = df.shape[0]
    ranks = np.arange(1, n + 1, dtype=float)
    denominator = {
        "later_low_observations": float(df["later_low_count"].sum()),
        "any_later_low_stays": float(df["any_later_low_count"].sum()),
        "hospital_deaths": float(df["hospital_mortality"].sum()),
    }
    curve = pd.DataFrame(
        {
            "flagged_percent": 100.0 * ranks / n,
            "later_low_observations_captured": np.cumsum(df["later_low_count"]) / denominator["later_low_observations"],
            "any_later_low_stays_captured": np.cumsum(df["any_later_low_count"])
            / denominator["any_later_low_stays"],
            "hospital_deaths_captured": np.cumsum(df["hospital_mortality"]) / denominator["hospital_deaths"],
        }
    )
    threshold_rows: List[Dict[str, object]] = []
    for percent in [5, 10, 20, 30, 50]:
        k = int(np.ceil(n * percent / 100.0))
        local = df.iloc[:k]
        threshold_rows.append(
            {
                "flagged_percent": percent,
                "flagged_stays": int(k),
                "admission_window_q10_max": float(local["admission_window_q10"].max()),
                "later_map_below65_burden": float(local["later_map_below65_fraction"].mean()),
                "any_later_map_below65": float(local["any_later_map_below65"].mean()),
                "hospital_mortality": float(local["hospital_mortality"].mean()),
                "later_low_observations_captured": float(local["later_low_count"].sum())
                / denominator["later_low_observations"],
                "hospital_deaths_captured": float(local["hospital_mortality"].sum())
                / denominator["hospital_deaths"],
            }
        )
    return curve, pd.DataFrame(threshold_rows)


def plot_clinical_yield(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    curve: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_stem: Path,
) -> None:
    atlas, bins, admission_window_q10, later_burden, mortality = binned_trajectory_atlas(stay_df, design)
    decile_edges = [int(round(atlas.shape[0] * frac / 10.0)) for frac in range(11)]
    decile_centers = [(decile_edges[i] + decile_edges[i + 1]) / 2.0 for i in range(10)]
    decile_labels = []
    for i in range(10):
        local_q10 = admission_window_q10[decile_edges[i] : decile_edges[i + 1]]
        decile_labels.append(f"D{i + 1}\n{np.median(local_q10):.1f}")

    fig = plt.figure(figsize=(11.4, 5.8))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.75, 0.055, 1.0],
        height_ratios=[1.0, 0.9],
        wspace=0.55,
        hspace=0.42,
    )

    ax = fig.add_subplot(grid[:, 0])
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#efefef")
    norm = TwoSlopeNorm(vmin=45.0, vcenter=65.0, vmax=105.0)
    image = ax.imshow(
        np.ma.masked_invalid(atlas),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[bins[0], bins[-1], atlas.shape[0], 0],
    )
    ax.axvline(12.0, color="black", linestyle=":", linewidth=1.0)
    for edge in decile_edges[1:-1]:
        ax.axhline(edge, color="white", linewidth=0.7, alpha=0.75)
    ax.set_xlabel("Hours since ICU admission")
    ax.set_ylabel("Assessment stays ordered by admission window MAP q10")
    ax.set_yticks(decile_centers)
    ax.set_yticklabels(decile_labels)
    ax.set_title("Trajectory atlas ordered by admission window lower tail MAP")
    ax.text(12.2, 0.04 * atlas.shape[0], "12 h split", fontsize=8, color="black")
    add_panel_label(ax, "A")

    cax = fig.add_subplot(grid[:, 1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_title("MAP\n(mmHg)", fontsize=8, pad=8)

    ax = fig.add_subplot(grid[0, 2])
    xs = curve["flagged_percent"].to_numpy(dtype=float)
    ax.plot(xs, 100.0 * curve["later_low_observations_captured"], color="#20639B", linewidth=2.0)
    ax.plot(xs, 100.0 * curve["any_later_low_stays_captured"], color="#2A9D8F", linewidth=2.0)
    ax.plot(xs, 100.0 * curve["hospital_deaths_captured"], color="#D1495B", linewidth=2.0)
    ax.plot([0, 100], [0, 100], color="0.65", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Flagged lowest admission window q10 stays (%)")
    ax.set_ylabel("Captured outcome burden (%)")
    ax.set_title("Risk concentration")
    ax.grid(axis="both", color="0.9", linewidth=0.7)
    ax.legend(
        ["MAP<65 observations", "Any later MAP<65 stay", "Hospital deaths", "Random flagging"],
        frameon=False,
        loc="lower right",
    )
    add_panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, 2])
    x = thresholds["flagged_percent"].to_numpy(dtype=float)
    ax.plot(
        x,
        100.0 * thresholds["any_later_map_below65"],
        color="#2A9D8F",
        marker="o",
        linewidth=1.8,
        label="Any later MAP<65",
    )
    ax.plot(
        x,
        100.0 * thresholds["later_map_below65_burden"],
        color="#20639B",
        marker="s",
        linewidth=1.8,
        label="Later MAP<65 burden",
    )
    ax.plot(
        x,
        100.0 * thresholds["hospital_mortality"],
        color="#D1495B",
        marker="^",
        linewidth=1.8,
        label="Hospital mortality",
    )
    ax.set_xlabel("Flagged lowest admission window q10 stays (%)")
    ax.set_ylabel("Rate or burden in flagged group (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(3, 58)
    ax.set_xticks(x)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_title("Clinical yield of admission window flagging")
    ax.text(x[-1] + 1.0, 100.0 * thresholds["any_later_map_below65"].iloc[-1], "Any later\nMAP<65", color="#2A9D8F", va="center", fontsize=8)
    ax.text(x[-1] + 1.0, 100.0 * thresholds["later_map_below65_burden"].iloc[-1], "MAP<65\nburden", color="#20639B", va="center", fontsize=8)
    ax.text(x[-1] + 1.0, 100.0 * thresholds["hospital_mortality"].iloc[-1], "Hospital\nmortality", color="#D1495B", va="center", fontsize=8)
    add_panel_label(ax, "C")

    save_figure(fig, output_stem)


def normal_ci_for_difference(p1: float, n1: int, p0: float, n0: int) -> Tuple[float, float]:
    se = np.sqrt(max(p1 * (1.0 - p1), 0.0) / max(n1, 1) + max(p0 * (1.0 - p0), 0.0) / max(n0, 1))
    diff = p1 - p0
    return float(diff - 1.96 * se), float(diff + 1.96 * se)


def subgroup_signal_table(stay_df: pd.DataFrame) -> pd.DataFrame:
    df = stay_df.copy()
    median_index_obs = float(df["index_obs"].median())
    subgroup_defs = [
        ("Overall", np.ones(df.shape[0], dtype=bool)),
        ("Age <65 y", df["age_years"].to_numpy(dtype=float) < 65.0),
        ("Age >=65 y", df["age_years"].to_numpy(dtype=float) >= 65.0),
        ("Female", df["male"].to_numpy(dtype=float) < 0.5),
        ("Male", df["male"].to_numpy(dtype=float) >= 0.5),
        ("Elective/nonurgent", df["emergency_or_urgent"].to_numpy(dtype=float) < 0.5),
        ("Emergency/urgent", df["emergency_or_urgent"].to_numpy(dtype=float) >= 0.5),
        (f"Index obs <{int(median_index_obs)}", df["index_obs"].to_numpy(dtype=float) < median_index_obs),
        (f"Index obs >={int(median_index_obs)}", df["index_obs"].to_numpy(dtype=float) >= median_index_obs),
    ]
    rows: List[Dict[str, object]] = []
    for label, mask in subgroup_defs:
        local = df.loc[mask].copy()
        if local.shape[0] < 100:
            continue
        q20, q80 = np.quantile(local["admission_window_q10"].to_numpy(dtype=float), [0.20, 0.80])
        vulnerable = local[local["admission_window_q10"] <= q20]
        resilient = local[local["admission_window_q10"] >= q80]
        p_v = float(vulnerable["any_later_map_below65"].mean())
        p_r = float(resilient["any_later_map_below65"].mean())
        ci_low, ci_high = normal_ci_for_difference(p_v, vulnerable.shape[0], p_r, resilient.shape[0])
        pop = float(local["population_check_loss"].mean())
        update = float(local["updated_check_loss"].mean())
        rows.append(
            {
                "subgroup": label,
                "stays": int(local.shape[0]),
                "any_later_map_below65": float(local["any_later_map_below65"].mean()),
                "auc_admission_window_q10_low_is_risk": auc_score(
                    local["any_later_map_below65"].to_numpy(dtype=float),
                    -local["admission_window_q10"].to_numpy(dtype=float),
                ),
                "vulnerable_quintile_any_later_map_below65": p_v,
                "resilient_quintile_any_later_map_below65": p_r,
                "risk_difference_any_later_map_below65": float(p_v - p_r),
                "risk_difference_ci_low": ci_low,
                "risk_difference_ci_high": ci_high,
                "population_check_loss": pop,
                "updated_check_loss": update,
                "loss_reduction_percent": float(100.0 * (pop - update) / pop),
            }
        )
    return pd.DataFrame(rows)


def model_gain_deciles(stay_df: pd.DataFrame) -> pd.DataFrame:
    df = stay_df.copy()
    df["admission_window_q10_decile"] = pd.qcut(df["admission_window_q10"].rank(method="first"), 10, labels=False) + 1
    df["check_loss_improvement"] = df["population_check_loss"] - df["updated_check_loss"]
    total_net_improvement = float(df["check_loss_improvement"].sum())
    rows: List[Dict[str, object]] = []
    for decile, local in df.groupby("admission_window_q10_decile", sort=True):
        pop = float(local["population_check_loss"].mean())
        update = float(local["updated_check_loss"].mean())
        net = float(local["check_loss_improvement"].sum())
        rows.append(
            {
                "decile": int(decile),
                "stays": int(local.shape[0]),
                "admission_window_q10_median": float(local["admission_window_q10"].median()),
                "later_map_below65_fraction": float(local["later_map_below65_fraction"].mean()),
                "population_check_loss": pop,
                "updated_check_loss": update,
                "loss_reduction_percent": float(100.0 * (pop - update) / pop),
                "net_check_loss_improvement": net,
                "share_of_total_net_improvement": float(net / total_net_improvement),
            }
        )
    out = pd.DataFrame(rows)
    out["cumulative_share_of_total_net_improvement"] = out["share_of_total_net_improvement"].cumsum()
    return out


def outcome_association_table(stay_df: pd.DataFrame) -> pd.DataFrame:
    df = stay_df.copy()
    covariates = df[["age_z", "male", "emergency_or_urgent"]].to_numpy(dtype=float)
    scores = [
        ("10 mmHg lower admission window q10", -df["admission_window_q10"].to_numpy(dtype=float)),
        ("10 mmHg higher offset vulnerability", df["updated_vulnerability_score"].to_numpy(dtype=float)),
    ]
    outcomes = [
        ("Any later MAP <65", "any_later_map_below65"),
        ("Hospital mortality", "hospital_mortality"),
        ("ICU mortality", "icu_mortality"),
        ("ICU LOS >=3 days", "icu_los_ge3_days"),
    ]
    rows: List[Dict[str, object]] = []
    for outcome_label, outcome_col in outcomes:
        y = df[outcome_col].to_numpy(dtype=float)
        for score_label, score in scores:
            assoc = logistic_or_per_10(y, score, covariates)
            rows.append(
                {
                    "outcome": outcome_label,
                    "score": score_label,
                    "events": int(np.sum(y > 0.5)),
                    "stays": int(y.size),
                    "auc": auc_score(y, score),
                    "adjusted_or_per_10": assoc["odds_ratio_per_10"],
                    "ci_low": assoc["ci_low"],
                    "ci_high": assoc["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def plot_subgroup_signal(subgroups: pd.DataFrame, output_stem: Path) -> None:
    df = subgroups.reset_index(drop=True).copy()
    row_labels = [f"{row.subgroup} (n={int(row.stays):,})" for row in df.itertuples(index=False)]
    metric_specs = [
        ("Event rate\n%", "any_later_map_below65", 100.0, "{:.1f}"),
        ("AUC", "auc_admission_window_q10_low_is_risk", 1.0, "{:.3f}"),
        ("Risk difference\n(pp)", "risk_difference_any_later_map_below65", 100.0, "{:.1f}"),
        ("Loss reduction\n(%)", "loss_reduction_percent", 1.0, "{:.1f}"),
    ]
    values = np.column_stack(
        [df[col].to_numpy(dtype=float) * multiplier for _, col, multiplier, _ in metric_specs]
    )
    normalized = np.zeros_like(values, dtype=float)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        if hi > lo:
            normalized[:, col_idx] = (col - lo) / (hi - lo)
        else:
            normalized[:, col_idx] = 0.5

    fig, ax = plt.subplots(figsize=(9.7, 4.7))
    ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metric_specs)))
    ax.set_xticklabels([spec[0] for spec in metric_specs])
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_xticks(np.arange(-0.5, len(metric_specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(df.shape[0]):
        for j, (_, _, _, fmt) in enumerate(metric_specs):
            text_color = "white" if normalized[i, j] > 0.58 else "#22313f"
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", color=text_color, fontweight="bold")

    ax.axhline(0.5, color="#22313f", linewidth=1.2)
    ax.set_title("Primary 12 hour admission window lower tail signal across assessment subgroups", pad=34)
    ax.text(
        -0.5,
        df.shape[0] + 0.35,
        "Color is scaled within each column; numbers are the original metric values.",
        ha="left",
        va="center",
        color="0.35",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_model_gain(gain: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    x = gain["decile"].to_numpy(dtype=int)
    xlabels = [f"D{d}\n{q:.1f}" for d, q in zip(x, gain["admission_window_q10_median"])]
    overall_reduction = 100.0 * (
        gain["population_check_loss"].mul(gain["stays"]).sum()
        - gain["updated_check_loss"].mul(gain["stays"]).sum()
    ) / gain["population_check_loss"].mul(gain["stays"]).sum()

    ax = axes[0]
    ax.plot(x, gain["population_check_loss"], color="0.45", marker="o", linewidth=1.8, label="Population")
    ax.plot(x, gain["updated_check_loss"], color="#20639B", marker="s", linewidth=1.8, label="Profiled offset")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Stay level later check loss")
    ax.set_xlabel("Admission window q10 decile (median q10)")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.legend(frameon=False)
    add_panel_label(ax, "A")

    ax = axes[1]
    ax.bar(x, gain["loss_reduction_percent"], color="#2A9D8F", alpha=0.88)
    ax.axhline(overall_reduction, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Check loss reduction (%)")
    ax.set_xlabel("Admission window q10 decile (median q10)")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    add_panel_label(ax, "B")

    ax = axes[2]
    ax.bar(x, 100.0 * gain["share_of_total_net_improvement"], color="#D1495B", alpha=0.85)
    ax.plot(x, 100.0 * gain["cumulative_share_of_total_net_improvement"], color="#2F4858", marker="o", linewidth=1.8)
    ax.axhline(100.0, color="0.65", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Share of net improvement (%)")
    ax.set_xlabel("Admission window q10 decile (median q10)")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    add_panel_label(ax, "C")

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, output_stem)


def write_dynamic_signal_tex(dynamic: pd.DataFrame, path: Path) -> None:
    rows = dynamic[dynamic["index_hours"].isin([1.0, 3.0, 6.0, 12.0, 18.0])]
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Exploratory dynamic-window summaries of the admission-window lower-tail MAP signal.}",
        "\\label{tab:dynamic_signal_metrics}",
        "\\centering",
        "\\begin{tabular}{rrrrrr}",
        "\\hline",
        "Window (h) & Stays & AUC & Low q10 risk & High q10 risk & Loss reduction\\\\",
        "\\hline",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{float(row['index_hours']):.0f} & {int(row['n_stays'])} & "
            f"{float(row['auc_admission_window_q10_low_is_risk']):.3f} & "
            f"{100.0 * float(row['vulnerable_quintile_any_later_map_below65']):.1f}\\% & "
            f"{100.0 * float(row['resilient_quintile_any_later_map_below65']):.1f}\\% & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Rows use strict clock-time splits requiring at least four observations by the window end and at least one later observation. Low and high q10 tail groups include all stays at or below the empirical 20th percentile and at or above the empirical 80th percentile, respectively. Boundary ties are retained, so these tail groups need not contain exactly 20\\% of stays and are not identical to the mutually exclusive primary q10 strata.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_clinical_yield_tex(thresholds: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Exploratory fixed-capacity risk-concentration summaries based on admission-window q10.}",
        "\\label{tab:clinical_yield_thresholds}",
        "\\centering",
        "\\begin{tabular}{rrrrrrr}",
        "\\hline",
        "Selected fraction & Stays & q10 cutoff & Any later MAP $<$65 & Later MAP $<$65 & Hosp. mortality & Deaths captured\\\\",
        "\\hline",
    ]
    for _, row in thresholds.iterrows():
        lines.append(
            f"{float(row['flagged_percent']):.0f}\\% & {int(row['flagged_stays'])} & "
            f"{float(row['admission_window_q10_max']):.1f} & "
            f"{100.0 * float(row['any_later_map_below65']):.1f}\\% & "
            f"{100.0 * float(row['later_map_below65_burden']):.1f}\\% & "
            f"{100.0 * float(row['hospital_mortality']):.1f}\\% & "
            f"{100.0 * float(row['hospital_deaths_captured']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Each flagged group contains exactly $\\lceil Np\\rceil$ stays after sorting by admission-window MAP 0.10 quantile, where $p$ is the nominal flagged fraction; ties at the cutoff are resolved reproducibly by ascending stay index. These fixed-capacity groups can therefore differ from cutpoint-defined tail groups that retain all boundary ties. Later MAP $<$65 is the mean later observation-level burden within the flagged group. Deaths captured is the fraction of all assessment hospital deaths contained in the flagged group.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_subgroup_signal_tex(subgroups: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Exploratory subgroup summaries of the admission-window lower-tail MAP signal.}",
        "\\label{tab:subgroup_signal}",
        "\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Subgroup & Stays & Event rate & AUC & Risk difference & Loss reduction\\\\",
        "\\hline",
    ]
    for _, row in subgroups.iterrows():
        lines.append(
            f"{row['subgroup']} & {int(row['stays'])} & "
            f"{100.0 * float(row['any_later_map_below65']):.1f}\\% & "
            f"{float(row['auc_admission_window_q10_low_is_risk']):.3f} & "
            f"{100.0 * float(row['risk_difference_any_later_map_below65']):.1f}\\% & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Risk difference compares the lowest and highest admission-window q10 quintiles within each subgroup for any later recorded MAP below 65 mmHg. Loss reduction is relative to the population component and uses the primary profiled offset rule. No multiplicity adjustment was applied.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_gain_tex(gain: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Where the primary profiled offset improves later lower-tail prediction.}",
        "\\label{tab:model_gain_deciles}",
        "\\centering",
        "\\begin{tabular}{rrrrrrr}",
        "\\hline",
        "Decile & Stays & Adm. q10 & Later MAP $<$65 & Pop. loss & Offset loss & Reduction\\\\",
        "\\hline",
    ]
    for _, row in gain.iterrows():
        lines.append(
            f"{int(row['decile'])} & {int(row['stays'])} & {float(row['admission_window_q10_median']):.1f} & "
            f"{100.0 * float(row['later_map_below65_fraction']):.1f}\\% & "
            f"{float(row['population_check_loss']):.3f} & {float(row['updated_check_loss']):.3f} & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Deciles are ordered from lowest to highest admission window MAP 0.10 quantile. Later MAP $<$65 is the mean later observation level burden.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outcome_associations_tex(associations: pd.DataFrame, path: Path) -> None:
    score_labels = {
        "10 mmHg lower admission window q10": "Lower adm. q10",
        "10 mmHg higher offset vulnerability": "Offset vulnerability",
    }
    outcome_labels = {
        "Any later MAP <65": "Any later MAP $<$65",
        "ICU LOS >=3 days": "ICU LOS $\\ge$3 days",
    }
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Covariate adjusted associations between admission window lower tail MAP summaries and assessment outcomes.}",
        "\\label{tab:outcome_associations}",
        "\\centering",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Outcome & Score & Events & AUC & Adjusted OR & 95\\% CI\\\\",
        "\\hline",
    ]
    for _, row in associations.iterrows():
        lines.append(
            f"{outcome_labels.get(row['outcome'], row['outcome'])} & "
            f"{score_labels.get(row['score'], row['score'])} & {int(row['events'])} & "
            f"{float(row['auc']):.3f} & {float(row['adjusted_or_per_10']):.2f} & "
            f"{float(row['ci_low']):.2f}--{float(row['ci_high']):.2f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{AUCs are unadjusted score-only discrimination summaries. Adjusted odds ratios are per 10 mmHg and are from logistic regressions adjusted for age, sex, and emergency or urgent admission. The admission-window q10 score is coded so that larger values mean lower admission-window MAP 0.10 quantile; offset vulnerability is coded so that larger values mean a higher $-\\widehat b_i$. Associations are descriptive risk-concentration summaries under observed care.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_metrics(dynamic: pd.DataFrame, deciles: pd.DataFrame, surface: pd.DataFrame) -> Dict[str, object]:
    dyn12 = dynamic.iloc[(dynamic["index_hours"] - 12.0).abs().argsort()[:1]].iloc[0]
    low_decile = deciles[deciles["decile"] == 1].iloc[0]
    high_decile = deciles[deciles["decile"] == 10].iloc[0]
    dense_surface = surface[surface["stays"] >= 25].copy()
    max_cell = dense_surface.iloc[int(dense_surface["any_later_map_below65"].argmax())]
    min_cell = dense_surface.iloc[int(dense_surface["any_later_map_below65"].argmin())]
    return {
        "dynamic_12h_auc_admission_window_q10": float(dyn12["auc_admission_window_q10_low_is_risk"]),
        "dynamic_12h_auc_updated_vulnerability": float(dyn12["auc_updated_vulnerability"]),
        "dynamic_12h_risk_difference_any_later_low_percent": float(
            100.0 * dyn12["risk_difference_any_later_map_below65"]
        ),
        "dynamic_12h_loss_reduction_percent": float(dyn12["loss_reduction_percent"]),
        "most_vs_least_vulnerable_decile": {
            "admission_window_q10_medians": [float(low_decile["admission_window_q10_median"]), float(high_decile["admission_window_q10_median"])],
            "later_map_below65_burden_percent": [
                float(100.0 * low_decile["later_map_below65_fraction"]),
                float(100.0 * high_decile["later_map_below65_fraction"]),
            ],
            "any_later_map_below65_percent": [
                float(100.0 * low_decile["any_later_map_below65"]),
                float(100.0 * high_decile["any_later_map_below65"]),
            ],
            "hospital_mortality_percent": [
                float(100.0 * low_decile["hospital_mortality"]),
                float(100.0 * high_decile["hospital_mortality"]),
            ],
        },
        "surface_extreme_cells_with_at_least_25_stays": {
            "highest_any_later_low": max_cell.to_dict(),
            "lowest_any_later_low": min_cell.to_dict(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced empirical figures for the ICU MAP manuscript.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument(
        "--include-diagnostic-trajectories",
        action="store_true",
        help="Also write the optional representative trajectory diagnostic plot and CSV.",
    )
    args = parser.parse_args()

    set_plot_style()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    beta = np.asarray(results["coefficients"]["baseline_tau_0.10"], dtype=float)
    lambda_b = float(results["settings"]["selected_lambda_baseline_update"])

    _, assessment_design, metadata = load_assessment_design(args)
    dynamic, dynamic_stay_level = dynamic_signal_table(
        assessment_design,
        beta=beta,
        tau=args.tau,
        lambda_b=lambda_b,
        index_windows=[1, 2, 3, 4, 6, 8, 10, 12, 16, 18],
    )
    stay_df = stay_level_frame(assessment_design, beta, tau=args.tau, lambda_b=lambda_b, index_hours=12.0)
    stay_df = attach_outcomes(stay_df, args.features_csv)
    age_info = metadata["age_standardization"]
    stay_df["age_years"] = (
        stay_df["age_z"].to_numpy(dtype=float) * float(age_info["age_sd_training_split"])
        + float(age_info["age_mean_training_split"])
    )
    stay_df["beta"] = [beta for _ in range(stay_df.shape[0])]
    surface = risk_surface_table(stay_df)
    deciles = decile_summary(stay_df)
    trajectories = None
    if args.include_diagnostic_trajectories:
        trajectories = select_representative_trajectories(stay_df, assessment_design)
    yield_curve, yield_thresholds = clinical_yield_tables(stay_df)
    subgroups = subgroup_signal_table(stay_df)
    gain = model_gain_deciles(stay_df)
    associations = outcome_association_table(stay_df)

    dynamic.to_csv(args.artifact_dir / "enhanced_dynamic_signal_metrics.csv", index=False)
    public_export_frame(dynamic_stay_level).to_csv(args.artifact_dir / "enhanced_dynamic_signal_stay_level.csv", index=False)
    public_export_frame(stay_df.drop(columns=["beta"])).to_csv(
        args.artifact_dir / "enhanced_12h_stay_level_features.csv",
        index=False,
    )
    surface.to_csv(args.artifact_dir / "enhanced_risk_surface.csv", index=False)
    deciles.to_csv(args.artifact_dir / "enhanced_admission_window_q10_decile_summary.csv", index=False)
    if trajectories is not None:
        trajectories.to_csv(args.artifact_dir / "enhanced_representative_trajectories.csv", index=False)
    yield_curve.to_csv(args.artifact_dir / "enhanced_clinical_yield_curve.csv", index=False)
    yield_thresholds.to_csv(args.artifact_dir / "enhanced_clinical_yield_thresholds.csv", index=False)
    subgroups.to_csv(args.artifact_dir / "enhanced_subgroup_signal.csv", index=False)
    gain.to_csv(args.artifact_dir / "enhanced_model_gain_deciles.csv", index=False)
    associations.to_csv(args.artifact_dir / "enhanced_outcome_associations.csv", index=False)

    plot_dynamic_signal(dynamic, args.artifact_dir / "enhanced_dynamic_signal_plot")
    plot_risk_surface(surface, deciles, args.artifact_dir / "enhanced_risk_surface_plot")
    if trajectories is not None:
        plot_representative_trajectories(trajectories, args.artifact_dir / "enhanced_representative_trajectories_plot")
    plot_clinical_yield(
        stay_df,
        assessment_design,
        yield_curve,
        yield_thresholds,
        args.artifact_dir / "enhanced_clinical_yield_plot",
    )
    plot_subgroup_signal(subgroups, args.artifact_dir / "enhanced_subgroup_signal_plot")
    plot_model_gain(gain, args.artifact_dir / "enhanced_model_gain_plot")

    write_dynamic_signal_tex(dynamic, args.artifact_dir / "enhanced_dynamic_signal_metrics.tex")
    write_clinical_yield_tex(yield_thresholds, args.artifact_dir / "enhanced_clinical_yield_thresholds.tex")
    write_subgroup_signal_tex(subgroups, args.artifact_dir / "enhanced_subgroup_signal.tex")
    write_model_gain_tex(gain, args.artifact_dir / "enhanced_model_gain_deciles.tex")
    write_outcome_associations_tex(associations, args.artifact_dir / "enhanced_outcome_associations.tex")

    artifacts = {
        "dynamic_signal_plot": str(args.artifact_dir / "enhanced_dynamic_signal_plot.pdf"),
        "risk_surface_plot": str(args.artifact_dir / "enhanced_risk_surface_plot.pdf"),
        "clinical_yield_plot": str(args.artifact_dir / "enhanced_clinical_yield_plot.pdf"),
        "subgroup_signal_plot": str(args.artifact_dir / "enhanced_subgroup_signal_plot.pdf"),
        "model_gain_plot": str(args.artifact_dir / "enhanced_model_gain_plot.pdf"),
        "dynamic_signal_metrics_csv": str(args.artifact_dir / "enhanced_dynamic_signal_metrics.csv"),
        "risk_surface_csv": str(args.artifact_dir / "enhanced_risk_surface.csv"),
        "decile_summary_csv": str(args.artifact_dir / "enhanced_admission_window_q10_decile_summary.csv"),
        "clinical_yield_curve_csv": str(args.artifact_dir / "enhanced_clinical_yield_curve.csv"),
        "clinical_yield_thresholds_csv": str(args.artifact_dir / "enhanced_clinical_yield_thresholds.csv"),
        "subgroup_signal_csv": str(args.artifact_dir / "enhanced_subgroup_signal.csv"),
        "model_gain_deciles_csv": str(args.artifact_dir / "enhanced_model_gain_deciles.csv"),
        "outcome_associations_csv": str(args.artifact_dir / "enhanced_outcome_associations.csv"),
    }
    if trajectories is not None:
        artifacts["representative_trajectories_plot"] = str(
            args.artifact_dir / "enhanced_representative_trajectories_plot.pdf"
        )
        artifacts["representative_trajectories_csv"] = str(
            args.artifact_dir / "enhanced_representative_trajectories.csv"
        )

    summary = {
        "status": "ok",
        "metadata": metadata,
        "tau": args.tau,
        "lambda_b": lambda_b,
        "metrics": summary_metrics(dynamic, deciles, surface),
        "artifacts": artifacts,
    }
    (args.artifact_dir / "enhanced_empirical_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
