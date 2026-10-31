from pathlib import Path
import argparse
import csv
import json
import math
import os
import shutil
import subprocess

import numpy as np
import pymupdf


FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 128.0
NATIVE_ORIGIN_OFFSET_MM = 24.0


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_features(observations, profiles):
    summaries = []
    for example, expected_count in (("A", 16), ("B", 10)):
        rows = [row for row in observations
                if row["example"] == example and row["window"] == "index"]
        times = np.asarray([float(row["time_hours"]) for row in rows])
        residuals = np.asarray([float(row["map_mmhg"]) - float(row["population_mmhg"])
                                for row in rows])
        assert len(rows) == expected_count
        assert np.isfinite(times).all() and np.isfinite(residuals).all()
        assert np.all(np.diff(times) >= 0) and np.all((times >= 0) & (times <= 12))
        half_count = len(rows) // 2
        assert half_count > 0
        variance = float(np.var(times, ddof=0))
        earlier = float(np.quantile(residuals[:half_count], .1, method="inverted_cdf"))
        recent = float(np.quantile(residuals[half_count:], .1, method="inverted_cdf"))
        delta = recent - earlier
        features = np.asarray([
            np.clip(np.log(len(rows) / 12.0) / np.log(4.0), -1.0, 1.0),
            np.clip(12.0 * np.var((times - 12.0) / 12.0, ddof=0) - 1.0, -1.0, 1.0),
            np.clip(delta / 10.0, -1.0, 1.0),
        ])
        retained = [row for row in profiles if row["example"] == example]
        assert sorted(float(row["horizon_hours"]) for row in retained) == [4.0, 8.0, 12.0]
        np.testing.assert_allclose(
            [[float(row[key]) for key in ("count", "time_spread", "q10_change")]
             for row in retained],
            np.tile(features, (len(retained), 1)), rtol=0, atol=1e-12,
        )
        summaries.append({
            "example": example,
            "count": len(rows),
            "half_count": half_count,
            "recent_half_count": len(rows) - half_count,
            "mean_time_hours": float(np.mean(times)),
            "time_variance_hours2": variance,
            "q10_earlier": earlier,
            "q10_recent": recent,
            "q10_delta_mmHg": delta,
            "variance_label": f"{variance:.2f}",
            "q10_earlier_label": f"{earlier:.3f}".replace("-", "−"),
            "q10_recent_label": f"{recent:.3f}".replace("-", "−"),
            "q10_delta_label": f"{delta:+.3f}".replace("-", "−"),
            "count_feature": float(features[0]),
            "time_variance_feature": float(features[1]),
            "q10_change_feature": float(features[2]),
        })
    return summaries


def point_segment_distance(point, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point[0] - first[0], point[1] - first[1])
    position = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length_squared
    position = min(1.0, max(0.0, position))
    return math.hypot(point[0] - first[0] - position * dx,
                      point[1] - first[1] - position * dy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert data_dir != output_dir
    temp_dir = output_dir / "tex-temp"
    temp_dir.mkdir(exist_ok=True)
    observations = read_rows(data_dir / "introduction_observations.csv")
    forecasts = read_rows(data_dir / "introduction_forecast.csv")
    profiles = read_rows(data_dir / "introduction_profiles.csv")
    metadata = json.loads((data_dir / "introduction_data.json").read_text(encoding="utf-8"))
    assert metadata["replicate"] == 4 and metadata["tau"] == .1
    assert metadata["landmark_hours"] == 12 and metadata["horizons_hours"] == [4, 8, 12]
    assert metadata["profile_diagnostics"]["all_certified"]
    feature_summaries = summarize_features(observations, profiles)
    write_rows(output_dir / "introduction_feature_summary.csv",
               list(feature_summaries[0]), feature_summaries)
    early_rows, future_rows = {}, {}
    for example, expected_count in (("A", 16), ("B", 10)):
        early = [
            {"time": row["time_hours"], "map": row["map_mmhg"]}
            for row in observations if row["example"] == example and row["window"] == "index"
        ]
        future = [
            {"time": row["time_hours"], "population": row["population_mmhg"],
             "profile": row["direct_prediction_mmhg"]}
            for row in forecasts
            if row["example"] == example and float(row["horizon_hours"]) == 12.0
        ]
        assert len(early) == expected_count and len(future) == 49
        assert float(future[0]["time"]) == 12 and float(future[-1]["time"]) == 24
        early_rows[example], future_rows[example] = early, future
        write_rows(output_dir / f"introduction_{example}_early.csv", ["time", "map"], early)
        write_rows(output_dir / f"introduction_{example}_forecast.csv", ["time", "population", "profile"], future)

    engine = shutil.which("lualatex")
    assert engine is not None, "lualatex is required"
    environment = os.environ.copy()
    environment["TMPDIR"] = str(temp_dir)
    with (output_dir / "compile_stdout.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", "-jobname=introduction_overview", str(source)],
            cwd=output_dir, env=environment, check=True, stdout=stream, stderr=subprocess.STDOUT,
        )
    document = pymupdf.open(output_dir / "introduction_overview.pdf")
    assert len(document) == 1
    page = document[0]
    assert abs(page.rect.width - FIGURE_WIDTH_MM * 72 / 25.4) < .2
    assert abs(page.rect.height - FIGURE_HEIGHT_MM * 72 / 25.4) < .2
    (output_dir / "introduction_overview.svg").write_text(
        page.get_svg_image(text_as_path=True), encoding="utf-8"
    )
    page.get_pixmap(dpi=600).save(output_dir / "introduction_overview.png")
    spans = [
        span for block in page.get_text("dict")["blocks"] if "lines" in block
        for line in block["lines"] for span in line["spans"]
    ]
    outside = [span for span in spans if not page.rect.contains(pymupdf.Rect(span["bbox"]))]
    assert not outside, outside
    assert not page.get_images(), "The PDF must contain native vector elements"
    layout_panels = []
    for row in read_rows(output_dir / "introduction_plot_geometry.csv"):
        bounds = [float(row[key].removesuffix("pt")) * 72 / 72.27
                  for key in ("left_pt", "bottom_pt", "right_pt", "top_pt")]
        bounds[1] += NATIVE_ORIGIN_OFFSET_MM * 72 / 25.4
        bounds[3] += NATIVE_ORIGIN_OFFSET_MM * 72 / 25.4
        layout_panels.append({"id": row["panel"], "bbox_pt": bounds})
    panels = {panel["id"]: panel for panel in layout_panels}
    for identifier, heading in (("earlyA", "History A"), ("earlyB", "History B")):
        heading_spans = [span for span in spans if span["text"] == heading]
        assert len(heading_spans) == 1
        origin = heading_spans[0]["origin"]
        panels[identifier]["panel_label_anchor_pt"] = [origin[0], page.rect.height - origin[1]]
    early_a, early_b = panels["earlyA"]["bbox_pt"], panels["earlyB"]["bbox_pt"]
    assert max(abs(early_a[index] - early_b[index]) for index in (0, 2)) < .1
    assert abs((early_a[3] - early_a[1]) - (early_b[3] - early_b[1])) < .1

    # 根据 PDF 原生路径检查全部数据位置，记录缩略图不显示坐标轴。
    drawings = page.get_drawings()
    trace_errors = {}

    def check_trace(identifier, rows, ordinate, domain):
        bounds = panels[identifier]["bbox_pt"]
        xmin, xmax, ymin, ymax = domain
        points = [
            (bounds[0] + (float(row["time"]) - xmin) / (xmax - xmin) * (bounds[2] - bounds[0]),
             page.rect.height - bounds[1] - (float(row[ordinate]) - ymin) / (ymax - ymin) * (bounds[3] - bounds[1]))
            for row in rows
        ]
        expected_box = [min(point[0] for point in points), min(point[1] for point in points),
                        max(point[0] for point in points), max(point[1] for point in points)]
        errors = []
        for drawing in drawings:
            if max(abs(drawing["rect"][index] - expected_box[index]) for index in range(4)) > .03:
                continue
            segments = [(item[1], item[2]) for item in drawing["items"] if item[0] == "l"]
            if not segments:
                continue
            errors.append(max(min(point_segment_distance(point, first, second)
                                  for first, second in segments) for point in points))
        assert errors and min(errors) < .03, (identifier, ordinate, errors)
        return min(errors)

    for example, identifier in (("A", "earlyA"), ("B", "earlyB")):
        trace_errors[identifier] = check_trace(identifier, early_rows[example], "map", (0, 12, 60, 90))
        trace_errors["future" + example] = check_trace("futureQuantiles", future_rows[example], "profile", (12, 24, 62, 76))
    trace_errors["monitorRecord"] = check_trace("monitorRecord", early_rows["A"], "map", (8, 12, 60, 90))
    observation_time_errors = {}
    observation_time_counts = {}

    def check_time_rug(identifier, rows, baseline_fraction, mark_size):
        bounds = panels[identifier]["bbox_pt"]
        baseline = page.rect.height - bounds[1] - baseline_fraction * (bounds[3] - bounds[1])
        expected_times = [bounds[0] + float(row["time"]) / 12 * (bounds[2] - bounds[0])
                          for row in rows]
        tick_height = 2 * mark_size * 72 / 72.27
        tick_positions = [
            drawing["rect"].x0 for drawing in drawings
            if drawing["rect"].width < .01 and abs(drawing["rect"].height - tick_height) < .2
            and abs((drawing["rect"].y0 + drawing["rect"].y1) / 2 - baseline) < .02
            and bounds[0] - .02 <= drawing["rect"].x0 <= bounds[2] + .02
        ]
        assert len(tick_positions) == len(expected_times), (identifier, len(tick_positions), len(expected_times))
        maximum_time_error = max(min(abs(expected - observed) for observed in tick_positions)
                                 for expected in expected_times)
        assert maximum_time_error < .02, (identifier, maximum_time_error)
        return maximum_time_error, len(tick_positions)

    for example, identifier in (("A", "earlyA"), ("B", "earlyB")):
        maximum_time_error, count = check_time_rug(identifier, early_rows[example], 0.0, 1.3)
        observation_time_errors[example] = maximum_time_error
        observation_time_counts[example] = count
    feature_time_errors = {}
    feature_time_counts = {}
    for example, identifier in (("A", "featureTimeA"), ("B", "featureTimeB")):
        maximum_time_error, count = check_time_rug(identifier, early_rows[example], .5, 1.8)
        feature_time_errors[example] = maximum_time_error
        feature_time_counts[example] = count
    layout = {
        "schema_version": 1,
        "backend": "Native TikZ record geometry measured at compilation and checked against PDF data paths",
        "figure": {"width_pt": page.rect.width, "height_pt": page.rect.height},
        "panels": layout_panels,
        "column_groups": [
            {"id": "record_miniatures", "panels": ["earlyA", "earlyB"]},
            {"id": "feature_time_rugs", "panels": ["featureTimeA", "featureTimeB"]},
        ],
        "exemptions": [
            {"panels": ["monitorRecord", "futureQuantiles"], "checks": ["all"],
             "reason": "The bedside monitor repeats the observed 8 to 12 hour record; the forecast report depicts a different time and MAP domain"}
        ],
    }
    (output_dir / "introduction_overview.layout.json").write_text(
        json.dumps(layout, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "native_bounds_mm": [0, -NATIVE_ORIGIN_OFFSET_MM, FIGURE_WIDTH_MM, 104],
        "pdf_origin_offset_mm": [0, NATIVE_ORIGIN_OFFSET_MM],
        "early_observations": {key: len(value) for key, value in early_rows.items()},
        "forecast_points": {key: len(value) for key, value in future_rows.items()},
        "monitor_repeated_early_observations": 16,
        "observation_time_mark_counts": observation_time_counts,
        "maximum_pdf_observation_time_error_pt": max(observation_time_errors.values()),
        "feature_time_mark_counts": feature_time_counts,
        "maximum_pdf_feature_time_error_pt": max(feature_time_errors.values()),
        "feature_summary_file": "introduction_feature_summary.csv",
        "feature_summaries": feature_summaries,
        "feature_definition": {
            "count": "Number of early MAP observations",
            "time_variance_hours2": "Observation time variance with denominator m_i",
            "q10_earlier_and_recent": "Inverse empirical CDF residual quantiles in the earlier and recent chronological count halves",
            "q10_delta_mmHg": "Recent half residual q10 minus earlier half residual q10",
            "model_transformations": "Clipped log count and normalized time variance and residual quantile change divided by 10 mmHg",
        },
        "synthetic_stay_indices": [597, 148],
        "forecast_horizon_hours": 12,
        "source_rows": "Frozen informative recovery replicate 4 under seed 20261022",
        "forecast_definition": "direct population + level + slope prediction",
        "record_miniature_domains": {"time_hours": [0, 12], "map_mmhg": [60, 90]},
        "forecast_miniature_domains": {"time_hours": [12, 24], "map_mmhg": [62, 76]},
        "native_icons": ["ICU bed and patient", "MAP monitor", "layered records", "population group", "forecast report",
                         "observed reading count arrays and actual time rugs and rank-half residual q10 summaries", "learned and constant candidate cards",
                         "held out MAP assessment card with stay weighted loss and strict and weak probabilities"],
        "schematic_controls": "Separate positive level and slope penalty functions in one joint quantile profile; slider positions are symbolic",
        "component_glyphs": "Constant level and linear time basis, jointly fitted for each new stay",
        "window_ribbons": "Fixed 4, 8 and 12 hour forecast windows beginning at a common landmark",
        "history_features": "Count and temporal spread and chronological rank-half residual lower quantile change jointly inform both controls",
        "independent_tuning_object": "Stay grouped threefold CV selects learned penalty functions or exact constant penalties using later MAP loss at three horizons",
        "dashed_parameter_connections": "The selected rule determines both component penalties",
        "population_covariates": "Clinical population component uses age proxy and sex and admission type",
        "assessment_card": "Observed later MAP gives stay weighted check loss and the displayed bracket [P(Y<q), P(Y<=q)] for the frozen direct forecast q",
        "maximum_pdf_data_path_error_pt": max(trace_errors.values()),
        "pdf_data_path_errors_pt": trace_errors,
        "intervals": "none",
        "axes_visible": 0,
        "pdf_text_spans": len(spans),
        "pdf_minimum_glyph_size_pt": min(span["size"] for span in spans),
        "pdf_raster_images": len(page.get_images()),
        "pdf_text_outside_page": len(outside),
        "rendered_record_miniature_alignment": "PASS",
        "svg_text": "glyph outlines for portable mathematical rendering; editable text retained in PDF and native TikZ",
        "visual_inspection": "not performed",
    }
    (output_dir / "introduction_overview.validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
