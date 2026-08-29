# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) penalty-selection heatmap -> assets/figures/heatmap/plot_composition.py -> param inherit
# (b) offset-correlation comparison -> assets/figures/BarComparison/plot_comparison_Trajectory.py -> param inherit
# (c) offset-RMSE comparison -> assets/figures/BarComparison/plot_comparison_Trajectory.py -> param inherit
# (d) landmark trend -> assets/figures/LineTrend/plot_trend.py -> param inherit
# (e) source and weighting comparison -> assets/figures/BarComparison/plot_comparison_Trajectory.py -> param inherit
# (f) density comparison -> assets/figures/BarComparison/plot_comparison_Trajectory.py -> param inherit
# (g) fold trend -> assets/figures/LineTrend/plot_trend.py -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PAPER_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = PAPER_ROOT / "recovery_20260822"
ARCHIVE_PUBLIC_ROOT = RECOVERY_ROOT / "github_release" / "icu-map-quantile-updating"
ARCHIVE_MANUSCRIPT_ROOT = (
    RECOVERY_ROOT
    / "revision_sim_benchmark_20260827"
    / "WileyNJDv5_Template"
)
PRIVATE_LAYOUT = ARCHIVE_PUBLIC_ROOT.exists() and ARCHIVE_MANUSCRIPT_ROOT.exists()
PUBLIC_ROOT = ARCHIVE_PUBLIC_ROOT if PRIVATE_LAYOUT else PAPER_ROOT
MANUSCRIPT_ROOT = ARCHIVE_MANUSCRIPT_ROOT if PRIVATE_LAYOUT else PAPER_ROOT
FIGURE_ROOT = MANUSCRIPT_ROOT / "figures" if PRIVATE_LAYOUT else PAPER_ROOT / "supplement_aggregate"
TABLE_ROOT = MANUSCRIPT_ROOT / "tables" if PRIVATE_LAYOUT else PAPER_ROOT / "supplement_aggregate"

SCENARIO_ORDER = [
    "ideal_large_dense",
    "ideal_small_dense",
    "ideal_large_sparse",
    "serial_dependence",
    "discrete_map_rounding",
    "informative_monitoring",
    "cluster_size_informative",
    "common_time_misspecified",
    "level_plus_shape",
    "treatment_feedback",
    "transient_nonpersistent",
    "null_serial",
    "weak_level",
    "heavy_tail_t3",
]
SCENARIO_LABEL = [
    "Large dense",
    "Small dense",
    "Large sparse",
    "Serial dependence",
    "Integer rounding",
    "Informative monitoring",
    "Cluster size association",
    "Omitted common time",
    "Persistent level with shape",
    "Treatment feedback",
    "Transient displacement",
    "Serial null",
    "Weak persistent level",
    "Heavy tailed residual",
]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def load_simulation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    penalties = pd.read_csv(
        PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_penalty_summary.csv"
    )
    summary = pd.read_csv(PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_summary.csv")
    penalties = penalties.loc[penalties["method"].eq("tuned_level")].copy()
    return penalties, summary


def plot_simulation_diagnostics() -> None:
    penalties, summary = load_simulation_data()
    penalty_values = [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
    matrix = (
        penalties.pivot(index="scenario_key", columns="selected_penalty", values="selected_proportion")
        .reindex(index=SCENARIO_ORDER, columns=penalty_values)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (14, 7) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise RuntimeError("Penalty selection frequencies must form a 14 by 7 probability matrix")

    methods = summary.loc[
        summary["method"].isin(["unpenalized_level", "tuned_level"])
    ].copy()
    unpen = methods.loc[methods["method"].eq("unpenalized_level")].set_index("scenario_key")
    tuned = methods.loc[methods["method"].eq("tuned_level")].set_index("scenario_key")
    corr_unpen = unpen.reindex(SCENARIO_ORDER)["offset_correlation_mean"].to_numpy(dtype=float)
    corr_tuned = tuned.reindex(SCENARIO_ORDER)["offset_correlation_mean"].to_numpy(dtype=float)
    rmse_unpen = unpen.reindex(SCENARIO_ORDER)["offset_rmse_mean"].to_numpy(dtype=float)
    rmse_tuned = tuned.reindex(SCENARIO_ORDER)["offset_rmse_mean"].to_numpy(dtype=float)
    rmse_ratio = rmse_tuned / rmse_unpen

    source = penalties.copy()
    source.to_csv(TABLE_ROOT / "supp_penalty_selection_source.csv", index=False)
    pd.DataFrame(
        {
            "scenario_key": SCENARIO_ORDER,
            "scenario_label": SCENARIO_LABEL,
            "unpenalized_offset_correlation": corr_unpen,
            "tuned_offset_correlation": corr_tuned,
            "unpenalized_offset_rmse": rmse_unpen,
            "tuned_offset_rmse": rmse_tuned,
            "tuned_to_unpenalized_rmse_ratio": rmse_ratio,
        }
    ).to_csv(TABLE_ROOT / "supp_offset_recovery_source.csv", index=False)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 145 * mm))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.0, 1.0], wspace=0.28)
    y = np.arange(len(SCENARIO_ORDER))

    ax_a = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("selection", SEQUENTIAL)
    image = ax_a.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax_a.set_xticks(np.arange(len(penalty_values)))
    ax_a.set_xticklabels(["0", "0.03", "0.10", "0.30", "1", "3", "10"])
    ax_a.set_xlabel(r"Selected $\lambda_b$")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(SCENARIO_LABEL, fontsize=6.2)
    ax_a.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value >= 0.05:
                color = "white" if value >= 0.55 else BLACK
                ax_a.text(j, i, f"{100*value:.0f}", ha="center", va="center", fontsize=5.2, color=color)
    for boundary in (2.5, 6.5, 9.5):
        ax_a.axhline(boundary, color="white", linewidth=1.4)
    colorbar = fig.colorbar(image, ax=ax_a, orientation="horizontal", fraction=0.055, pad=0.10)
    colorbar.set_label("Selection frequency", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.5)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    valid = np.isfinite(corr_unpen) & np.isfinite(corr_tuned)
    for index in np.flatnonzero(valid):
        ax_b.plot([corr_unpen[index], corr_tuned[index]], [index, index], color="#C7C7C7", linewidth=0.7, zorder=1)
    ax_b.scatter(corr_unpen[valid], y[valid], s=16, facecolor="white", edgecolor=GREY, linewidth=0.8, label="Unpenalized", zorder=2)
    ax_b.scatter(corr_tuned[valid], y[valid], s=17, color=CATEGORICAL[0], marker="o", label="Tuned", zorder=3)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_xlabel("Correlation with true level")
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=6.5, handletextpad=0.3)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    ax_c.axvline(1.0, color="#B8B8B8", linestyle="--", linewidth=0.7)
    ax_c.scatter(rmse_ratio, y, s=18, color=CATEGORICAL[0], zorder=3)
    ax_c.set_xlim(0.0, 1.08)
    ax_c.set_xlabel("Tuned / unpenalized RMSE")
    ax_c.tick_params(axis="y", left=False, labelleft=False)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.22, right=0.985, top=0.92, bottom=0.13)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, str(FIGURE_ROOT / "supp_simulation_tuning_diagnostics"))
    plt.close(fig)


def plot_empirical_validation() -> None:
    sensitivity = pd.read_csv(PUBLIC_ROOT / "empirical_aggregate" / "sensitivity_analysis.csv")
    weighted = pd.read_csv(PUBLIC_ROOT / "empirical_aggregate" / "stay_weighted_population_sensitivity.csv")
    sources = pd.read_csv(PUBLIC_ROOT / "source_sensitivity" / "map_source_sensitivity.csv")
    density = pd.read_csv(PUBLIC_ROOT / "empirical_aggregate" / "measurement_density_sensitivity.csv")
    folds = pd.read_csv(TABLE_ROOT / "supp_fold_stability_source.csv")

    landmark = sensitivity.loc[sensitivity["tau"].eq(0.10)].sort_values("index_hours")
    if landmark["index_hours"].tolist() != [6.0, 12.0, 18.0]:
        raise RuntimeError("Expected 6, 12, and 18 hour landmark results")

    weight_row = weighted.loc[weighted["model"].eq("Stay-weighted population + level update")].iloc[0]
    weight_base = weighted.loc[weighted["model"].eq("Stay-weighted population")].iloc[0]
    comparison_rows = [
        {
            "label": "Stay weighted training",
            "estimate": 100 * weight_row["paired_reduction_vs_stay_weighted_population"] / weight_base["assessment_loss"],
            "low": 100 * weight_row["ci_low"] / weight_base["assessment_loss"],
            "high": 100 * weight_row["ci_high"] / weight_base["assessment_loss"],
        }
    ]
    for _, row in sources.iterrows():
        comparison_rows.append(
            {
                "label": row["source_label"],
                "estimate": row["relative_loss_reduction_percent"],
                "low": 100 * row["paired_reduction_ci_low"] / row["population_loss"],
                "high": 100 * row["paired_reduction_ci_high"] / row["population_loss"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLE_ROOT / "supp_source_weighting_figure_source.csv", index=False)

    folds = folds.copy()
    folds["profiled_reduction_percent"] = 100 * (folds["population_loss"] - folds["profiled_loss"]) / folds["population_loss"]
    folds["calibrated_q10_reduction_percent"] = 100 * (folds["population_loss"] - folds["q10_loss"]) / folds["population_loss"]
    folds.to_csv(TABLE_ROOT / "supp_fold_reduction_figure_source.csv", index=False)

    mm = 1 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(183 * mm, 132 * mm))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    ax_a.plot(
        landmark["index_hours"],
        landmark["loss_reduction_percent"],
        color=CATEGORICAL[0],
        marker="o",
        markersize=4,
        linewidth=1.2,
    )
    ax_a.set_xticks([6, 12, 18])
    ax_a.set_xlabel("Landmark (hours)")
    ax_a.set_ylabel("Profiled loss reduction (%)")
    ax_a.set_ylim(0, 18)
    panel_label(ax_a, "a")

    y_b = np.arange(comparison.shape[0])
    ax_b.errorbar(
        comparison["estimate"],
        y_b,
        xerr=np.vstack(
            [comparison["estimate"] - comparison["low"], comparison["high"] - comparison["estimate"]]
        ),
        fmt="o",
        color=CATEGORICAL[0],
        ecolor="#6F6F6F",
        elinewidth=0.8,
        capsize=2,
        markersize=4,
    )
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(comparison["label"], fontsize=6.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Profiled loss reduction (%)")
    ax_b.set_xlim(0, 19)
    panel_label(ax_b, "b")

    density_plot = density.copy()
    y_c = np.arange(density_plot.shape[0])
    colors = [CATEGORICAL[0]] * 3 + [CATEGORICAL[3]] * 3
    markers = ["o"] * 3 + ["s"] * 3
    for idx, row in density_plot.iterrows():
        ax_c.scatter(
            row["loss_reduction_percent"],
            idx,
            color=colors[idx],
            marker=markers[idx],
            s=20,
            zorder=3,
        )
    labels = [f"{row.density_measure.replace(' observations', '')}: {row.stratum}" for row in density_plot.itertuples()]
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(labels, fontsize=6.5)
    ax_c.invert_yaxis()
    ax_c.axvline(0, color="#B8B8B8", linewidth=0.7)
    ax_c.set_xlabel("Profiled loss reduction (%)")
    ax_c.set_xlim(0, 19)
    panel_label(ax_c, "c")

    ax_d.plot(
        folds["fold"],
        folds["profiled_reduction_percent"],
        color=CATEGORICAL[0],
        marker="o",
        markersize=3.8,
        linewidth=1.1,
        label="Profiled offset",
    )
    ax_d.plot(
        folds["fold"],
        folds["calibrated_q10_reduction_percent"],
        color=CATEGORICAL[3],
        marker="s",
        markersize=3.8,
        linewidth=1.1,
        label="Calibrated q10",
    )
    ax_d.set_xticks([1, 2, 3, 4, 5])
    ax_d.set_xlabel("Outer fold")
    ax_d.set_ylabel("Reduction versus population (%)")
    ax_d.set_ylim(8, 18)
    ax_d.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.5, handlelength=1.4)
    panel_label(ax_d, "d")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.35, alpha=0.45)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.92, bottom=0.11, wspace=0.42, hspace=0.48)
    save_cns_figure(fig, str(FIGURE_ROOT / "supp_empirical_method_validation"))
    plt.close(fig)


def main() -> None:
    plot_simulation_diagnostics()
    plot_empirical_validation()
    print(f"Wrote expanded supplement figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
