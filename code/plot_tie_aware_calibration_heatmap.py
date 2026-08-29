# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-b) tie-aware calibration heatmaps -> assets/figures/heatmap/plot_comparison.py -> param inherit
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
    "savefig.dpi": 600,
})

def save_cns_figure(fig, filename):
    """Export a vector PDF and a 600 dpi submission PNG."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=600)


import argparse
from pathlib import Path

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


MODEL_ORDER = ("population", "primary_level_update", "raw_q10", "calibrated_q10")
MODEL_LABELS = {
    "population": "Population rule",
    "primary_level_update": "Profiled offset",
    "raw_q10": "Raw q10",
    "calibrated_q10": "Calibrated q10",
}


def signed_bracket_departure(lower: np.ndarray, upper: np.ndarray, tau: float) -> np.ndarray:
    """Signed distance of [lower, upper] from tau; zero when the bracket contains tau."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Calibration bracket endpoints must have the same shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Calibration bracket endpoints must be finite")
    if np.any((lower < 0.0) | (upper > 1.0) | (lower > upper)):
        raise ValueError("Calibration brackets must satisfy 0 <= L <= U <= 1")
    return np.where(lower > tau, lower - tau, np.where(upper < tau, upper - tau, 0.0))


def calibration_matrix(
    detail: pd.DataFrame,
    scope: str,
    aggregation: str,
    tau: float,
) -> np.ndarray:
    source = detail.loc[
        (detail["scope"] == scope)
        & detail["model"].isin(MODEL_ORDER)
        & detail["group"].astype(str).str.startswith("decile_")
    ].copy()
    source["decile"] = source["group"].str.replace("decile_", "", regex=False).astype(int)
    lower_column = f"p_y_lt_q_{aggregation}"
    upper_column = f"p_y_le_q_{aggregation}"
    rows = []
    for model in MODEL_ORDER:
        local = source.loc[source["model"] == model].sort_values("decile")
        if local["decile"].tolist() != list(range(1, 11)):
            raise ValueError(f"Model {model} does not contain exactly deciles 1 through 10")
        rows.append(
            signed_bracket_departure(
                local[lower_column].to_numpy(dtype=float),
                local[upper_column].to_numpy(dtype=float),
                tau,
            )
        )
    matrix = 100.0 * np.vstack(rows)
    if matrix.shape != (4, 10):
        raise ValueError(f"Expected a 4 by 10 calibration matrix, found {matrix.shape}")
    return matrix


def plot_calibration_heatmap(
    detail: pd.DataFrame,
    output_stem: Path,
    scope: str = "nested_5fold_internal_crossfit",
    tau: float = 0.10,
) -> None:
    stay_equal = calibration_matrix(detail, scope, "stay_equal", tau)
    observation_weighted = calibration_matrix(detail, scope, "observation_weighted", tau)
    weighting_difference = observation_weighted - stay_equal

    width_mm = 183.0
    height_mm = 94.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.035),
        height_ratios=(1.0, 1.0),
        left=0.18,
        right=0.90,
        bottom=0.15,
        top=0.95,
        hspace=0.56,
        wspace=0.05,
    )
    absolute_axis = fig.add_subplot(grid[0, 0])
    difference_axis = fig.add_subplot(grid[1, 0])
    absolute_colorbar_axis = fig.add_subplot(grid[0, 1])
    difference_colorbar_axis = fig.add_subplot(grid[1, 1])

    absolute_cmap = LinearSegmentedColormap.from_list("calibration_departure", DIVERGING)
    absolute_limit = 22.0
    absolute_norm = TwoSlopeNorm(vmin=-absolute_limit, vcenter=0.0, vmax=absolute_limit)
    absolute_image = absolute_axis.imshow(
        stay_equal,
        aspect="auto",
        cmap=absolute_cmap,
        norm=absolute_norm,
        interpolation="nearest",
    )
    absolute_axis.set_title("a  Equal weighting by stay", loc="left", fontweight="bold", pad=5)
    absolute_axis.set_yticks(np.arange(len(MODEL_ORDER)))
    absolute_axis.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    absolute_axis.set_xticks(np.arange(10))
    absolute_axis.set_xticklabels([])
    absolute_axis.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    absolute_axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    absolute_axis.grid(which="minor", color="white", linewidth=1.0)
    absolute_axis.tick_params(which="minor", bottom=False, left=False)
    absolute_axis.tick_params(axis="both", length=0)
    for spine in absolute_axis.spines.values():
        spine.set_visible(False)

    absolute_colorbar = fig.colorbar(
        absolute_image,
        cax=absolute_colorbar_axis,
        ticks=[-20, -10, 0, 10, 20],
    )
    absolute_colorbar.set_label(r"$D_{0.10}$ (percentage points)", rotation=270, labelpad=13)
    absolute_colorbar.outline.set_linewidth(0.6)

    difference_limit = np.ceil(np.max(np.abs(weighting_difference)) * 10.0) / 10.0
    if difference_limit <= 0.0:
        difference_limit = 0.1
    difference_cmap = LinearSegmentedColormap.from_list(
        "weighting_difference",
        [CATEGORICAL[4], DIVERGING[1], CATEGORICAL[3]],
    )
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit,
        vcenter=0.0,
        vmax=difference_limit,
    )
    difference_image = difference_axis.imshow(
        weighting_difference,
        aspect="auto",
        cmap=difference_cmap,
        norm=difference_norm,
        interpolation="nearest",
    )
    difference_axis.set_title(
        "b  Difference under observation weighting",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    difference_axis.set_yticks(np.arange(len(MODEL_ORDER)))
    difference_axis.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    difference_axis.set_xticks(np.arange(10))
    difference_axis.set_xticklabels([str(value) for value in range(1, 11)])
    difference_axis.set_xlabel("Decile of fitted quantile")
    difference_axis.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    difference_axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    difference_axis.grid(which="minor", color="white", linewidth=1.0)
    difference_axis.tick_params(which="minor", bottom=False, left=False)
    difference_axis.tick_params(axis="both", length=0)
    for spine in difference_axis.spines.values():
        spine.set_visible(False)

    difference_ticks = [-difference_limit, 0.0, difference_limit]
    difference_colorbar = fig.colorbar(
        difference_image,
        cax=difference_colorbar_axis,
        ticks=difference_ticks,
    )
    difference_colorbar.ax.set_yticklabels([f"{value:.1f}" for value in difference_ticks])
    difference_colorbar.set_label(r"Difference in $D_{0.10}$ (percentage points)", rotation=270, labelpad=13)
    difference_colorbar.outline.set_linewidth(0.6)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, str(output_stem))
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tie-aware calibration departures by model and prediction decile.")
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--scope", default="nested_5fold_internal_crossfit")
    parser.add_argument("--tau", type=float, default=0.10)
    args = parser.parse_args()
    plot_calibration_heatmap(
        pd.read_csv(args.detail_csv),
        args.output_stem,
        scope=args.scope,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
