import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = (
    "informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress",
)
SETTING_LABELS = (
    "Informative recovery", "Weaker recovery", "Information limit", "Serial dependence and integer MAP",
)
HORIZONS = (4.0, 8.0, 12.0)
COVERAGE = (4.0, 8.0, 12.0)
CONTRASTS = (
    ("full_geometry", "diagonal_profile"),
    ("horizon_geometry", "diagonal_profile"),
    ("full_geometry", "horizon_geometry"),
)
METHODS = ("full_geometry", "horizon_geometry")
SELECTION_MODES = ("joint", "direct")
SCOPES = ("reference", "assessment")
COLUMNS = (
    "setting", "selection_mode", "scope", "horizon_hours", "coverage_hours", "first", "second", "replicates",
    "mean_difference", "mcse", "pointwise_low", "pointwise_high",
    "simultaneous_low", "simultaneous_high",
    "mean_stays_per_replicate", "minimum_stays_per_replicate", "total_stay_evaluations",
)
STYLE = {
    "full_geometry": {"label": "Full geometry minus diagonal", "color": "#0072B2",
                      "marker": "o", "linestyle": "-", "markerfacecolor": "#0072B2"},
    "horizon_geometry": {"label": "Horizon geometry minus diagonal", "color": "#D55E00",
                         "marker": "s", "linestyle": "--", "markerfacecolor": "white"},
}


def read_results(path):
    frame = pd.read_csv(path)
    if set(frame.columns) != set(COLUMNS):
        raise ValueError("Unexpected geometry mechanism summary columns")
    keys = ["setting", "selection_mode", "scope", "horizon_hours", "coverage_hours", "first", "second"]
    expected = {(setting, mode, scope, horizon, coverage, first, second)
                for setting, mode, scope, horizon, coverage, (first, second)
                in product(SETTINGS, SELECTION_MODES, SCOPES, HORIZONS, COVERAGE, CONTRASTS)}
    observed = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or observed != expected or len(frame) != 432:
        raise ValueError("The figure requires every prespecified setting and comparison")
    numeric = frame[list(COLUMNS[7:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (frame["replicates"] == 200).all():
        raise ValueError("All figure cells require 200 finite replicate summaries")
    if (frame["mcse"] < 0).any():
        raise ValueError("Monte Carlo standard errors must be nonnegative")
    if not ((frame["minimum_stays_per_replicate"] >= 1)
            & (frame["minimum_stays_per_replicate"] <= frame["mean_stays_per_replicate"])).all():
        raise ValueError("Every conditional risk cell must contain evaluation stays")
    if not np.allclose(frame["mean_stays_per_replicate"] * frame["replicates"],
                       frame["total_stay_evaluations"], rtol=1e-12, atol=1e-9):
        raise ValueError("Conditional evaluation counts do not match their replicate average")
    if not ((frame["simultaneous_low"] <= frame["pointwise_low"])
            & (frame["pointwise_low"] <= frame["mean_difference"])
            & (frame["mean_difference"] <= frame["pointwise_high"])
            & (frame["pointwise_high"] <= frame["simultaneous_high"])).all():
        raise ValueError("The supplied uncertainty intervals are inconsistent")
    for _, group in frame.groupby(["setting", "selection_mode", "scope", "horizon_hours", "coverage_hours"]):
        contrast = group.set_index(["first", "second"])["mean_difference"]
        if not np.isclose(contrast.loc[CONTRASTS[0]],
                          contrast.loc[CONTRASTS[1]] + contrast.loc[CONTRASTS[2]],
                          rtol=1e-10, atol=1e-12):
            raise ValueError("Paired mean differences violate their exact contrast identity")
        if any(group[column].nunique() != 1 for column in COLUMNS[-3:]):
            raise ValueError("The comparisons within a cell do not share evaluation stays")
    return frame


def configure_style():
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"),
                          fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })


def check_layout(fig, axes, plotted):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_count = 0
    font_sizes = []
    hidden_labels = set()
    for axis in axes.flat:
        for coordinate, bounds in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in coordinate.get_major_ticks():
                if not bounds[0] <= tick.get_loc() <= bounds[1]:
                    hidden_labels.update((tick.label1, tick.label2))
    text_boxes = []
    for item in fig.findobj(match=Text):
        if item in hidden_labels or not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < canvas.x0 - 0.5 or box.y0 < canvas.y0 - 0.5 or \
                box.x1 > canvas.x1 + 0.5 or box.y1 > canvas.y1 + 0.5:
            raise RuntimeError(f"Figure text extends beyond the page: {item.get_text()}")
        font_sizes.append(float(item.get_fontsize()))
        text_boxes.append((item.get_text(), box))
        text_count += 1
    for index, (first_text, first_box) in enumerate(text_boxes):
        for second_text, second_box in text_boxes[index + 1:]:
            width = min(first_box.x1, second_box.x1) - max(first_box.x0, second_box.x0)
            height = min(first_box.y1, second_box.y1) - max(first_box.y0, second_box.y0)
            if width > 0.5 and height > 0.5:
                raise RuntimeError(f"Figure text overlaps: {first_text} and {second_text}")
    if min(font_sizes) < 9:
        raise RuntimeError("Figure text is smaller than nine points")
    limits = [tuple(axis.get_ylim()) for axis in axes.flat]
    if len(set(limits)) != 1:
        raise RuntimeError("Figure panels do not share an identical vertical scale")
    low, high = limits[0]
    if not ((plotted["pointwise_low"] > low)
            & (plotted["pointwise_high"] < high)).all():
        raise RuntimeError("A supplied uncertainty interval lies outside the panel")
    return {"text_items_checked": text_count, "minimum_font_points": min(font_sizes),
            "all_panel_y_limits": [low, high], "text_outside_canvas": 0,
            "text_overlaps": 0}


def plot_results(frame, output, selection_mode, scope):
    configure_style()
    plotted = frame[(frame["selection_mode"] == selection_mode) & (frame["scope"] == scope)
                    & (frame["second"] == "diagonal_profile")
                    & frame["first"].isin(METHODS)].copy()
    order = ["setting", "horizon_hours", "first", "coverage_hours"]
    plotted = plotted.sort_values(order).reset_index(drop=True)
    if len(plotted) != 72:
        raise RuntimeError("The figure must display 72 paired risk estimates")
    minimum = min(0.0, float(plotted["pointwise_low"].min()))
    maximum = max(0.0, float(plotted["pointwise_high"].max()))
    padding = max((maximum - minimum) * 0.09, 0.002)
    limits = minimum - padding, maximum + padding
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 7.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.11, top=0.84,
                        wspace=0.25, hspace=0.78)
    for row, (setting, setting_label) in enumerate(zip(SETTINGS, SETTING_LABELS)):
        for column, horizon in enumerate(HORIZONS):
            axis = axes[row, column]
            axis.axhline(0, color="#6B7280", linewidth=0.7, zorder=1)
            for method in reversed(METHODS):
                group = plotted[(plotted["setting"] == setting)
                                & (plotted["horizon_hours"] == horizon)
                                & (plotted["first"] == method)].sort_values("coverage_hours")
                np.testing.assert_array_equal(group["coverage_hours"].to_numpy(), COVERAGE)
                style = STYLE[method]
                mean = group["mean_difference"].to_numpy()
                interval = np.vstack([mean - group["pointwise_low"].to_numpy(),
                                      group["pointwise_high"].to_numpy() - mean])
                artists = axis.errorbar(group["coverage_hours"], mean, yerr=interval,
                                        color=style["color"], marker=style["marker"],
                                        linestyle=style["linestyle"], markerfacecolor=style["markerfacecolor"],
                                        markersize=4.1 if method == "full_geometry" else 5.1,
                                        markeredgewidth=0.8, linewidth=0.9, elinewidth=0.7,
                                        capsize=2, capthick=0.7, zorder=3 if method == "full_geometry" else 2)
                np.testing.assert_array_equal(artists.lines[0].get_xdata(), group["coverage_hours"])
                np.testing.assert_array_equal(artists.lines[0].get_ydata(), mean)
                segments = np.asarray(artists.lines[2][0].get_segments())
                np.testing.assert_allclose(segments[:, 0, 1], group["pointwise_low"], rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(segments[:, 1, 1], group["pointwise_high"], rtol=1e-12, atol=1e-14)
            axis.set_xlim(3, 13)
            axis.set_ylim(*limits)
            axis.set_xticks(COVERAGE, ["4", "8", "12"])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.tick_params(axis="both", length=3, pad=2, labelbottom=True)
            axis.set_title(f"{int(horizon)} hour forecast", pad=4)
            axis.text(0.025, 0.95, chr(ord("A") + row * 3 + column),
                      transform=axis.transAxes, va="top", ha="left", fontsize=9,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3}, zorder=5)
        position = axes[row, 0].get_position()
        fig.text(position.x0, position.y1 + 0.040, setting_label, fontsize=9,
                 fontweight="bold", ha="left", va="bottom")
    legend = [Line2D([0], [0], color=STYLE[method]["color"], marker=STYLE[method]["marker"],
                     linestyle=STYLE[method]["linestyle"], markerfacecolor=STYLE[method]["markerfacecolor"],
                     markersize=4.5, linewidth=0.9, label=STYLE[method]["label"])
              for method in METHODS]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.984),
               frameon=False, ncol=2, columnspacing=1.8, handlelength=2)
    fig.text(0.56, 0.045, "Early sampling coverage interval (hours)",
             ha="center", va="center", fontsize=9)
    fig.text(0.025, 0.48, "Paired check loss difference (mmHg)",
             rotation=90, ha="center", va="center", fontsize=9)
    checks = check_layout(fig, axes, plotted)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "Early coverage and forecast horizon",
                                 "Subject": f"Paired {scope} risk contrasts with {selection_mode} selection across 200 independent replicates"})
    plotted.to_csv(output.with_suffix(".csv"), index=False, float_format="%.17g")
    report = {"panels": 12, "source_rows_checked": len(frame), "plotted_rows": len(plotted),
              "plotted_estimates_and_interval_endpoints_checked": 216,
              "replicates_per_cell": 200, "figure_size_inches": [7.0, 7.1],
              "uncertainty": "supplied pointwise 95 percent intervals",
              "font_family": "Times New Roman", "visual_inspection": False,
              "source_filename": "paired_summary.csv", "selection_mode": selection_mode,
              "scope": scope, **checks}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description="Plot verified coverage and forecast horizon risk contrasts")
    parser.add_argument("--input", type=Path, default=ROOT / ".review-work/mechanism/public/paired_summary.csv")
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="joint")
    parser.add_argument("--scope", choices=SCOPES, default="reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        suffix = "" if (args.selection_mode, args.scope) == ("joint", "reference") else f"_{args.selection_mode}_{args.scope}"
        args.output = ROOT / f"statistics-in-medicine-paper/manuscript/figures/geometry_mechanism{suffix}.pdf"
    if args.output.suffix.lower() != ".pdf":
        raise ValueError("The publication figure requires a PDF output path")
    report = plot_results(read_results(args.input), args.output, args.selection_mode, args.scope)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
