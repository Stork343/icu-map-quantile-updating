import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


BLUE = "#2B6C9E"
TEAL = "#2F9C95"
RED = "#C94C5A"
DARK = "#2F3A45"
GRAY = "#6B7280"
LIGHT_BLUE = "#E8F1F8"
LIGHT_TEAL = "#E6F4F1"
LIGHT_RED = "#F8E8EA"
LIGHT_GRAY = "#F3F4F6"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: Path, *, preserve_size: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if preserve_size else {"bbox_inches": "tight"}
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=600, **save_kwargs)
    plt.close(fig)


def box(ax: plt.Axes, xy: Tuple[float, float], wh: Tuple[float, float], text: str, fc: str, ec: str) -> None:
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2.0,
        xy[1] + wh[1] / 2.0,
        text,
        ha="center",
        va="center",
        color=DARK,
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = GRAY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def draw_mini_map_stream(ax: plt.Axes) -> None:
    x0, y0, width, height = 0.06, 0.46, 0.28, 0.31
    panel = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#CBD5E1",
        facecolor="white",
    )
    ax.add_patch(panel)
    xs = np.linspace(x0 + 0.025, x0 + width - 0.025, 17)
    ys = np.array([0.75, 0.70, 0.66, 0.72, 0.63, 0.58, 0.61, 0.68, 0.70, 0.64, 0.62, 0.66, 0.71, 0.73, 0.67, 0.65, 0.69])
    ys = y0 + 0.055 + (ys - ys.min()) / (ys.max() - ys.min()) * (height - 0.11)
    ax.plot(xs, ys, color=BLUE, linewidth=1.5, marker="o", markersize=2.6, clip_on=False)
    split = x0 + width * 0.52
    ax.plot([split, split], [y0 + 0.035, y0 + height - 0.035], color=RED, linestyle=":", linewidth=1.2)
    ax.text(split + 0.006, y0 + height - 0.045, "12 h", color=RED, fontsize=7.5, va="top")
    ax.text(
        (x0 + split) / 2.0,
        y0 + 0.014,
        "Index-window\nobservations",
        ha="center",
        va="bottom",
        linespacing=1.0,
        color=GRAY,
        fontsize=6.8,
    )
    ax.text(
        (split + x0 + width) / 2.0,
        y0 + 0.014,
        "Later\nassessment",
        ha="center",
        va="bottom",
        linespacing=1.0,
        color=GRAY,
        fontsize=6.8,
    )
    ax.text(x0 + width / 2.0, y0 + height + 0.035, "Observed ICU MAP stream", ha="center", color=DARK, fontweight="bold")


def plot_workflow(output_stem: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(10.2, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.95,
        "Split-window lower-tail MAP prediction protocol",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        va="top",
    )

    draw_mini_map_stream(ax)

    box(
        ax,
        (0.42, 0.67),
        (0.21, 0.12),
        "Training split\nfit population q10 component",
        LIGHT_BLUE,
        BLUE,
    )
    box(
        ax,
        (0.42, 0.45),
        (0.21, 0.12),
        "Tuning split\nselect quadratic penalty",
        LIGHT_TEAL,
        TEAL,
    )
    box(
        ax,
        (0.42, 0.23),
        (0.21, 0.12),
        "Assessment split\nscore the frozen rule",
        LIGHT_GRAY,
        "#9CA3AF",
    )
    box(
        ax,
        (0.72, 0.56),
        (0.22, 0.13),
        "Profiled offset\nestimate scalar $\\hat b_i$",
        LIGHT_RED,
        RED,
    )
    box(
        ax,
        (0.72, 0.30),
        (0.22, 0.13),
        "Later evaluation\nstay-level ordinary check loss",
        "white",
        DARK,
    )

    arrow(ax, (0.34, 0.62), (0.42, 0.73), BLUE)
    arrow(ax, (0.34, 0.58), (0.42, 0.51), TEAL)
    arrow(ax, (0.34, 0.54), (0.42, 0.29), GRAY)
    arrow(ax, (0.63, 0.73), (0.72, 0.63), BLUE)
    arrow(ax, (0.63, 0.51), (0.72, 0.62), TEAL)
    arrow(ax, (0.83, 0.56), (0.83, 0.43), RED)

    ax.text(0.515, 0.61, "candidate model set", ha="center", color=GRAY, fontsize=7.5)
    ax.text(0.835, 0.49, "split-window prediction", ha="center", color=GRAY, fontsize=7.5)
    ax.text(
        0.06,
        0.20,
        "$\\rho_{0.10}(u)=u\\{0.10-I(u<0)\\}$",
        fontsize=9,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    ax.text(
        0.06,
        0.09,
        "The reported loss is averaged within each assessment stay,\nthen averaged across stays.",
        fontsize=8.2,
        color=GRAY,
        linespacing=1.25,
    )

    save(fig, output_stem)


def plot_subgroup_forest(subgroups_csv: Path, output_stem: Path) -> None:
    """Plot the four prespecified exploratory subgroup metrics.

    Only the low-versus-high q10 risk difference has a source interval. The
    other panels therefore show point estimates without invented uncertainty.
    """
    set_style()
    df = pd.read_csv(subgroups_csv)
    df = df.reset_index(drop=True)
    required = [
        "subgroup",
        "stays",
        "any_later_map_below65",
        "auc_admission_window_q10_low_is_risk",
        "risk_difference_any_later_map_below65",
        "risk_difference_ci_low",
        "risk_difference_ci_high",
        "loss_reduction_percent",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Subgroup source data are missing required columns: {missing}")
    if df.empty or df["subgroup"].duplicated().any():
        raise ValueError("Subgroup source data must contain unique, nonempty rows")
    numeric = df.loc[:, required[1:]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Subgroup source data contain missing or non-finite values")
    if str(df.loc[0, "subgroup"]) != "Overall":
        raise ValueError("The Overall row must be first so reference lines are reproducible")

    labels = [f"{row.subgroup} (n={int(row.stays):,})" for row in df.itertuples(index=False)]
    y = np.arange(df.shape[0])[::-1]

    width_mm = 183.0
    height_mm = 108.0
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.00, 0.92, 1.18, 1.04], "wspace": 0.28},
    )

    event = 100.0 * df["any_later_map_below65"].to_numpy(dtype=float)
    auc = df["auc_admission_window_q10_low_is_risk"].to_numpy(dtype=float)
    risk = 100.0 * df["risk_difference_any_later_map_below65"].to_numpy(dtype=float)
    lo = 100.0 * df["risk_difference_ci_low"].to_numpy(dtype=float)
    hi = 100.0 * df["risk_difference_ci_high"].to_numpy(dtype=float)
    loss = df["loss_reduction_percent"].to_numpy(dtype=float)
    colors = np.array([DARK] + [BLUE] * (df.shape[0] - 1), dtype=object)
    markers = np.array(["D"] + ["o"] * (df.shape[0] - 1), dtype=object)

    def scatter_rows(ax: plt.Axes, values: np.ndarray) -> None:
        for yi, value, color, marker in zip(y, values, colors, markers):
            ax.scatter(value, yi, s=23, color=color, marker=marker, zorder=3, linewidths=0)

    panel_specs = [
        {
            "values": event,
            "title": "a  Later-event rate",
            "xlabel": "Any later MAP < 65 mmHg\n(event rate, %)",
            "xlim": (34.0, 60.0),
            "xticks": [35, 45, 55],
        },
        {
            "values": auc,
            "title": "b  q10 discrimination",
            "xlabel": "AUC",
            "xlim": (0.775, 0.815),
            "xticks": [0.78, 0.80, 0.81],
        },
        {
            "values": risk,
            "title": "c  Risk separation",
            "xlabel": "Low-high q10 event-rate difference\n(percentage points)",
            "xlim": (58.0, 76.0),
            "xticks": [60, 65, 70, 75],
        },
        {
            "values": loss,
            "title": "d  Predictive gain",
            "xlabel": "Profiled-offset loss reduction\n(%; positive favors offset rule)",
            "xlim": (7.0, 16.0),
            "xticks": [8, 10, 12, 14, 16],
        },
    ]

    for ax, spec in zip(axes, panel_specs):
        scatter_rows(ax, spec["values"])
        ax.axvline(float(spec["values"][0]), color=TEAL, linestyle="--", linewidth=0.85, zorder=1)
        ax.set_title(spec["title"], loc="left", fontweight="bold", color=DARK, pad=5)
        ax.set_xlabel(spec["xlabel"], labelpad=4)
        ax.set_xlim(*spec["xlim"])
        ax.set_xticks(spec["xticks"])
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", width=0.65, length=2.4, color="#6B7280")
        for separator in [7.5, 5.5, 3.5, 1.5]:
            ax.axhline(separator, color="#E5E7EB", linewidth=0.55, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#BFC5CC")

    axes[2].hlines(y, lo, hi, color="#8D99A6", linewidth=1.05, zorder=2)
    scatter_rows(axes[2], risk)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].tick_params(axis="y", pad=3)

    fig.suptitle(
        "Exploratory subgroup consistency of within-stay lower-tail MAP persistence",
        x=0.025,
        y=0.978,
        ha="left",
        fontsize=8.6,
        fontweight="bold",
        color=DARK,
    )
    fig.text(
        0.025,
        0.925,
        "Assessment-split estimates for a 12 h admission window. Lower q10 is coded as higher risk; panel c alone shows approximate normal 95% CIs; dashed lines mark overall estimates.",
        ha="left",
        va="top",
        fontsize=5.8,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.285, right=0.985, bottom=0.19, top=0.82)

    save(fig, output_stem, preserve_size=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("../paper/empirical"))
    parser.add_argument(
        "--subgroups-csv",
        type=Path,
        help="Optional subgroup source CSV; defaults to ARTIFACT_DIR/enhanced_subgroup_signal.csv",
    )
    parser.add_argument(
        "--subgroup-output-stem",
        type=Path,
        help="Optional Figure 6 output stem; defaults to ARTIFACT_DIR/enhanced_subgroup_signal_plot",
    )
    parser.add_argument("--only", choices=["all", "workflow", "subgroup"], default="all")
    args = parser.parse_args()
    artifact_dir = args.artifact_dir
    if args.only in {"all", "workflow"}:
        plot_workflow(artifact_dir / "enhanced_workflow_schematic")
    if args.only in {"all", "subgroup"}:
        plot_subgroup_forest(
            args.subgroups_csv or artifact_dir / "enhanced_subgroup_signal.csv",
            args.subgroup_output_stem or artifact_dir / "enhanced_subgroup_signal_plot",
        )


if __name__ == "__main__":
    main()
