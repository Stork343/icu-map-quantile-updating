from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / ".review-work" / "sim_figure_20261024"
CACHE = WORK / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
tempfile.tempdir = str(CACHE)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from lxml import etree
import numpy as np
import pandas as pd
import pymupdf
import svgutils.transform as svg

from manuscript_plot_style import HORIZONTAL_GRID, SERIES_STYLES, add_panel_label, set_plot_style

QA_SCRIPTS = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", Path(__file__).resolve().parent / "figure_qa"))
assert QA_SCRIPTS.is_dir(), f"Figure QA directory is unavailable: {QA_SCRIPTS}"
sys.path.insert(0, str(QA_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


WIDTH_MM = 183.0
HEIGHT_MM = 122.0
MM_TO_PT = 72 / 25.4
RECORD_STYLES = {
    "A": {**SERIES_STYLES["blue"], "markersize": 3.2, "markeredgewidth": 0.5},
    "B": {**SERIES_STYLES["red"], "markersize": 2.8, "markeredgewidth": 0.5},
}
COLORS = {key: style["color"] for key, style in RECORD_STYLES.items()}
MARKERS = {key: style["marker"] for key, style in RECORD_STYLES.items()}
LINE_WIDTH = SERIES_STYLES["blue"]["linewidth"]
INK = "black"
GREY = "0.35"
LIGHT = HORIZONTAL_GRID["color"]
NAMES = {"A": "Record 1", "B": "Record 2"}


def write_json(path: Path, contents: dict) -> None:
    path.write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n")


def load_data(directory: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((directory / "method_figure_data.json").read_text())
    observations = pd.read_csv(directory / "method_figure_observations.csv")
    profiles = pd.read_csv(directory / "method_figure_horizon_profiles.csv")
    curves = pd.read_csv(directory / "method_figure_quantile_curves.csv")
    assert manifest["illustration_type"] == "Simulation illustration"
    assert manifest["setting"] == "informative_recovery" and manifest["replicate"] == 1
    assert manifest["allocation_counts"] == {"F": 360, "T": 120, "A": 120}
    assert [row["synthetic_stay_index"] for row in manifest["examples"]] == [22, 27]
    assert manifest["tau"] == 0.1 and manifest["landmark_hours"] == 12
    assert max(manifest["assessment_replay_maximum_errors"].values()) <= 1e-12
    records = []
    for example in manifest["examples"]:
        key = example["example"]
        index = observations[(observations.example == key) & (observations.window == "index")]
        np.testing.assert_array_equal(index.time_hours, example["index_time"])
        np.testing.assert_allclose(index.map_mmhg, example["index_y"], rtol=0, atol=2e-14)
        assert len(index) == example["index_count"]
        profile = profiles[profiles.example == key]
        assert len(profile) == 81 and np.all(np.diff(profile.horizon_hours) > 0)
        for knot in example["knots"]:
            row = profile[np.isclose(profile.horizon_hours, knot["horizon_hours"])].iloc[0]
            for column in ("lambda_level", "lambda_slope", "level_coefficient", "slope_coefficient"):
                np.testing.assert_allclose(row[column], knot[column], rtol=0, atol=2e-14)
        fixed = curves[(curves.example == key) & (curves.horizon_hours == 12)]
        knot = profile[profile.horizon_hours == 12].iloc[0]
        expected = (example["population_mmhg"] + knot.level_coefficient
                    + knot.slope_coefficient * (fixed.time_hours.to_numpy() - 12) / 12)
        np.testing.assert_allclose(fixed.direct_prediction_mmhg, expected, rtol=0, atol=3e-14)
        np.testing.assert_allclose(fixed.level_prediction_mmhg,
                                   fixed.population_mmhg + fixed.level_contribution_mmhg,
                                   rtol=0, atol=3e-14)
        assert np.ptp(profile.lambda_slope) < 1e-5
        records.append({"record": NAMES[key], "synthetic_stay_index": example["synthetic_stay_index"],
                        "index_observations": len(index), "future_curve_points": len(fixed[fixed.time_hours >= 12]),
                        "profile_horizons": len(profile), "fixed_horizon_hours": 12,
                        "population_mmhg": example["population_mmhg"],
                        "level_at_12_mmhg": knot.level_coefficient,
                        "slope_at_24_mmhg": knot.slope_coefficient,
                        "q10_at_24_mmhg": float(expected[-1]),
                        "lambda_slope_range": [profile.lambda_slope.min(), profile.lambda_slope.max()]})
    qa = {"status": "PASS", "records": records,
          "selection_rule": manifest["selection_rule"],
          "selection_inputs": manifest["selection_inputs"],
          "assessment_replay_maximum_errors": manifest["assessment_replay_maximum_errors"],
          "example_curve_replay_maximum_error": manifest["example_curve_replay_maximum_error"]}
    return manifest, observations, profiles, curves, qa


def panel_axes(fig: plt.Figure, left_mm: float, top_mm: float, width_mm: float, gid: str) -> plt.Axes:
    axis = fig.add_axes([left_mm / WIDTH_MM, (HEIGHT_MM - top_mm - 32) / HEIGHT_MM,
                         width_mm / WIDTH_MM, 32 / HEIGHT_MM])
    axis.set_gid(gid)
    axis.grid(**HORIZONTAL_GRID)
    return axis


def make_figure(manifest: dict, observations: pd.DataFrame, profiles: pd.DataFrame,
                curves: pd.DataFrame) -> tuple[plt.Figure, list[plt.Axes]]:
    set_plot_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    axes = [panel_axes(fig, left, top, width, name)
            for top, names in ((14, ("panel_a1", "panel_b1", "panel_c1")),
                               (67, ("panel_a2", "panel_b2", "panel_c2")))
            for left, width, name in zip((12, 65, 118), (39, 39, 59), names)]
    a1, b1, c1, a2, b2, c2 = axes

    for axis, label, title in zip(axes, "ABCDEF", (
        "Record 1 early MAP", "Level penalty", "Record 1 forecast",
        "Record 2 early MAP", "Slope penalty", "Record 2 forecast",
    )):
        axis.set_title(title)
        add_panel_label(axis, label)

    for axis, key, example in zip((a1, a2), ("A", "B"), manifest["examples"]):
        early = observations[(observations.example == key) & (observations.window == "index")]
        axis.set(xlim=(0, 12.5), ylim=(52, 95), xticks=(0, 4, 8, 12), yticks=(55, 65, 75, 85, 95))
        axis.set_ylabel("MAP (mmHg)")
        axis.axhline(example["population_mmhg"], color=GREY, lw=LINE_WIDTH, linestyle=(0, (4, 2)), zorder=1)
        axis.plot(early.time_hours, early.map_mmhg, **RECORD_STYLES[key], zorder=2)
        axis.axvline(12, color=GREY, lw=0.8, linestyle=(0, (2, 2)), zorder=1)
        axis.text(0.15, 0.96, f"n = {len(early)}, {example['index_coverage_hours']:g} h window",
                  transform=axis.transAxes, fontsize=8, ha="left", va="top", color=INK)
    a2.set_xlabel("Time after admission (h)", labelpad=4)

    for axis, column, title, formula in (
        (b1, "lambda_level", "Level penalty", r"$\lambda_{\mathrm{level}}(D)$"),
        (b2, "lambda_slope", "Slope penalty", r"$\lambda_{\mathrm{slope}}(D)$"),
    ):
        axis.set(xlim=(3.7, 12.3), ylim=(0.026, 0.11), xticks=(4, 8, 12), yticks=(0.03, 0.05, 0.07, 0.09, 0.11))
        axis.set_ylabel(formula, fontsize=9, labelpad=4)
        axis.axhline(0.03, color=LIGHT, lw=0.7, zorder=1)
        for key in ("A", "B"):
            data = profiles[profiles.example == key]
            knots = np.flatnonzero(data.horizon_hours.isin((4.0, 8.0, 12.0)).to_numpy())
            axis.plot(data.horizon_hours, data[column], **RECORD_STYLES[key], markevery=knots,
                      zorder=6 if key == "A" else 5)
    b2.text(0.05, 0.62, "Both records", transform=b2.transAxes,
             fontsize=8, color=GREY, va="top")
    b2.text(0.05, 0.51, r"$\lambda_{\mathrm{slope}}\approx0.0300$", transform=b2.transAxes,
             fontsize=9, color=GREY, va="top")
    b2.set_xlabel("Requested horizon D (h)", labelpad=4)

    for axis, key in zip((c1, c2), ("A", "B")):
        data = curves[(curves.example == key) & (curves.horizon_hours == 12) & (curves.time_hours >= 12)]
        axis.set(xlim=(11.6, 24.4), ylim=(52, 95), xticks=(12, 16, 20, 24), yticks=(55, 65, 75, 85, 95))
        axis.set_ylabel("MAP (mmHg)")
        axis.axvline(12, ymax=0.8, color=GREY, lw=0.8, linestyle=(0, (2, 2)), zorder=1)
        axis.plot(data.time_hours, data.population_mmhg, color=GREY, linewidth=LINE_WIDTH,
                  linestyle=(0, (4, 2)), zorder=1)
        axis.plot(data.time_hours, data.level_prediction_mmhg, color=COLORS[key], linewidth=LINE_WIDTH,
                  linestyle=(0, (1.2, 1.7)), zorder=2)
        endpoints = np.flatnonzero(data.time_hours.isin((12, 24)).to_numpy())
        axis.plot(data.time_hours, data.direct_prediction_mmhg, **RECORD_STYLES[key],
                  markevery=endpoints, zorder=4)
        level = data.level_contribution_mmhg.iloc[0]
        slope = data.slope_contribution_mmhg.iloc[-1]
        sign = lambda value: f"{value:+.2f}".replace("-", "−")
        axis.text(0.08, 0.11, f"Level {sign(level)}; slope {sign(slope)} mmHg at 24 h",
                  transform=axis.transAxes, color=INK, fontsize=8)
        axis.text(0.96, 0.86, rf"$\widehat q_{{0.10}}(24)={data.direct_prediction_mmhg.iloc[-1]:.1f}$",
                  transform=axis.transAxes, ha="right", color=INK, fontsize=9)
    c2.set_xlabel("Time after admission (h)", labelpad=4)

    # 图例分别解释记录标识与预测组成，全部使用原生文字。
    fig.legend([Line2D([], [], **RECORD_STYLES[key], linestyle="none")
                for key in ("A", "B")],
               [NAMES[key] for key in ("A", "B")], ncol=2, frameon=False, handlelength=1.3,
               handletextpad=0.45, columnspacing=1.4, loc="center",
               bbox_to_anchor=(94 / WIDTH_MM, 1 - 111.5 / HEIGHT_MM))
    fig.legend([Line2D([], [], color=GREY, linewidth=LINE_WIDTH, linestyle=(0, (4, 2))),
                Line2D([], [], color=INK, linewidth=LINE_WIDTH, linestyle=(0, (1.2, 1.7))),
                Line2D([], [], color=INK, linewidth=LINE_WIDTH)],
               ["Population", "Population + level", "Full profile"], ncol=3, frameon=False, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.1, loc="center",
               bbox_to_anchor=(94 / WIDTH_MM, 1 - 117.7 / HEIGHT_MM))
    return fig, axes


def replay_artists(axes: list[plt.Axes], observations: pd.DataFrame,
                   profiles: pd.DataFrame, curves: pd.DataFrame) -> dict:
    for key, observed_axis, predicted_axis in zip(("A", "B"), (axes[0], axes[3]), (axes[2], axes[5])):
        early = observations[(observations.example == key) & (observations.window == "index")]
        np.testing.assert_array_equal(observed_axis.lines[1].get_xdata(), early.time_hours)
        np.testing.assert_array_equal(observed_axis.lines[1].get_ydata(), early.map_mmhg)
        fixed = curves[(curves.example == key) & (curves.horizon_hours == 12) & (curves.time_hours >= 12)]
        for line, column in zip(predicted_axis.lines[1:],
                                ("population_mmhg", "level_prediction_mmhg", "direct_prediction_mmhg")):
            np.testing.assert_array_equal(line.get_xdata(), fixed.time_hours)
            np.testing.assert_array_equal(line.get_ydata(), fixed[column])
    for axis, column in zip((axes[1], axes[4]), ("lambda_level", "lambda_slope")):
        for key, line in zip(("A", "B"), axis.lines[1:]):
            selected = profiles[profiles.example == key]
            np.testing.assert_array_equal(line.get_xdata(), selected.horizon_hours)
            np.testing.assert_array_equal(line.get_ydata(), selected[column])
    return {"status": "PASS", "observed_points": 22,
            "penalty_curve_coordinates": 324, "future_component_coordinates": 294,
            "comparison": "Every displayed artist coordinate equals its source CSV value."}


def assemble_svg(source: Path, target: Path) -> dict:
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    tree = etree.parse(str(source))
    original = tree.getroot()
    composed = svg.SVGFigure(original.attrib["width"], original.attrib["height"])
    composed.root.set("viewBox", original.attrib["viewBox"])
    composed.root.set("id", "editable-method-illustration")
    for element in original.findall("svg:defs", namespace):
        composed.root.append(deepcopy(element))
    figure = original.find("svg:g", namespace)
    assert figure is not None
    for element in figure:
        group = svg.GroupElement([deepcopy(element)], attrib={"id": f"editable-{element.get('id', 'element')}"})
        composed.append(group)
    composed.save(str(target), encoding="utf-8")
    saved = etree.parse(str(target))
    groups = [name for name in saved.xpath("//@id") if name.startswith("editable-panel_")]
    assert len(groups) == 6
    texts = saved.findall(".//svg:text", namespace)
    assert len(texts) >= 75
    assert not saved.findall(".//svg:image", namespace)
    return {"native_panel_groups": groups, "editable_text_elements": len(texts), "raster_images": 0}


def rendered_text_geometry(fig: plt.Figure) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    size = fig.bbox
    inspected = []
    for artist in fig.findobj(match=mpl.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        assert bbox.x0 >= size.x0 and bbox.y0 >= size.y0 and bbox.x1 <= size.x1 and bbox.y1 <= size.y1, artist.get_text()
        inspected.append({"text": artist.get_text(), "font_size_pt": artist.get_fontsize(),
                          "bbox_mm": [value / fig.dpi * 25.4 for value in bbox.extents]})
    assert min(item["font_size_pt"] for item in inspected) >= 5
    return {"status": "PASS", "text_items": inspected,
            "page_mm": [WIDTH_MM, HEIGHT_MM],
            "font_file": font_manager.findfont("Arial", fallback_to_default=False),
            "special_mathtext": [r"\lambda_{\mathrm{level}}(D)", r"\lambda_{\mathrm{slope}}(D)",
                                 r"\lambda_{\mathrm{slope}}\approx0.0300", r"\widehat q_{0.10}(24)"],
            "mathtext_note": "Matplotlib mathematical labels use the shared style's default mathtext fonts; actual glyph sizes are checked in PDF."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=WORK / "data")
    parser.add_argument("--output-dir", type=Path, default=WORK / "rendered")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, observations, profiles, curves, scientific_qa = load_data(args.data_dir.resolve())
    fig, axes = make_figure(manifest, observations, profiles, curves)
    scientific_qa["plotted_artist_replay"] = replay_artists(axes, observations, profiles, curves)
    base = output / "figure_method_illustration"
    text_geometry = rendered_text_geometry(fig)
    alignment = require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=("a1", "b1", "c1", "a2", "b2", "c2"),
        row_groups=(("a1", "b1", "c1"), ("a2", "b2", "c2")),
        column_groups=(("a1", "a2"), ("b1", "b2"), ("c1", "c2")),
        exemptions=({"panels": ["c1", "c2"], "checks": ["panel-width"],
                     "reason": "Prediction panels need room for the component curves and values."},),
        json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg", strict=True)
    fig.savefig(f"{base}.matplotlib.svg", metadata={"Creator": "Matplotlib"})
    fig.savefig(f"{base}.pdf", metadata={"Title": "Adaptive mixed quantile prediction", "Creator": "Matplotlib"})
    fig.savefig(f"{base}.png", dpi=600)
    svg_qa = assemble_svg(Path(f"{base}.matplotlib.svg"), Path(f"{base}.svg"))
    source_data = output / "source_data"
    source_data.mkdir(exist_ok=True)
    copied = []
    for path in sorted(args.data_dir.glob("method_figure_*")):
        shutil.copy2(path, source_data / path.name)
        copied.append(str((source_data / path.name).relative_to(output)))
    scientific_qa["source_data"] = copied
    scientific_qa["plot_data"] = {"A,D": "index observations from both records",
                                  "B,E": "all 81 horizon profiles per record",
                                  "C,F": "all future grid values at fixed D = 12 h"}
    write_json(Path(f"{base}.numeric-qa.json"), scientific_qa)
    write_json(Path(f"{base}.text-bounds.json"), text_geometry)
    write_json(Path(f"{base}.svg-qa.json"), svg_qa)
    audit_commands = {
        "source": [sys.executable, str(QA_SCRIPTS / "validate_figure.py"), str(Path(__file__).resolve()),
                   "--style-source", str(Path(__file__).with_name("manuscript_plot_style.py")), "--json"],
        "pdf-text": [sys.executable, str(QA_SCRIPTS / "audit_pdf_text.py"), f"{base}.pdf", "--min-pt", "5", "--json"],
        "collision": [sys.executable, str(QA_SCRIPTS / "audit_figure_collisions.py"), f"{base}.pdf",
                      "--json-out", f"{base}.collision-audit.json", "--overlay-pdf", f"{base}.collision-audit.pdf"],
    }
    for name, command in audit_commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if name != "collision":
            Path(f"{base}.{name}.json").write_text(result.stdout)
    pdf = pymupdf.open(f"{base}.pdf")
    page = pdf[0]
    np.testing.assert_allclose([page.rect.width, page.rect.height],
                               [WIDTH_MM * MM_TO_PT, HEIGHT_MM * MM_TO_PT], rtol=0, atol=1e-4)
    for heading in ("Record 1 early MAP", "Record 2 early MAP", "Level penalty", "Slope penalty",
                    "Record 1 forecast", "Record 2 forecast"):
        assert heading in page.get_text(), heading
    assert not page.get_images(full=True)
    report = {"dimensions_mm": [WIDTH_MM, HEIGHT_MM], "png_dpi": 600,
              "alignment_verdict": alignment["verdict"], "numeric_replay": "PASS",
              "editable_svg": svg_qa, "pdf_vector_images": len(page.get_images(full=True)),
              "dependencies": {package: importlib.metadata.version(package)
                               for package in ("matplotlib", "svgutils", "lxml", "numpy", "pandas", "pymupdf")},
              "source_data": copied,
              "source_preflight_warn_reviews": [
                  "The source preflight assigns the smallest figure text size to mathtext globally. The PDF glyph audit measures the actual mathematical glyph sizes.",
                  "PNG is the requested 600 dpi raster preview; PDF and SVG provide vector submission and editing files."
              ],
              "uncertainty": "A fixed illustrative pair of simulation records and fitted point curves. No uncertainty interval is displayed.",
              "panel_roles": {"A,D": "Observed early sampling and individual MAP history",
                              "B,E": "Component penalties as a function of the requested horizon",
                              "C,F": "Population, individual level and individual slope in future prediction"},
              "publication_style": {"font": "Arial", "axis_label_pt": 9, "tick_label_pt": 8,
                                    "panel_title_pt": 10, "panel_letter_pt": 11, "legend_pt": 8,
                                    "record_colors": COLORS, "record_markers": MARKERS,
                                    "spine_width_pt": mpl.rcParams["axes.linewidth"], "horizontal_grid": LIGHT,
                                    "horizontal_grid_width_pt": HORIZONTAL_GRID["linewidth"],
                                    "curve_width_pt": LINE_WIDTH,
                                    "marker_size_pt": {key: style["markersize"]
                                                       for key, style in RECORD_STYLES.items()},
                                    "marker_edge_width_pt": 0.5,
                                    "shared_style_module": "manuscript_plot_style.py",
                                    "reference_figures": [5, 6]}}
    write_json(Path(f"{base}.delivery.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
