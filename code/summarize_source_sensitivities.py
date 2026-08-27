import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


SOURCE_LABELS = {
    "invasive": "Invasive MAP only",
    "noninvasive": "Non-invasive MAP only",
}


def _model_row(payload: Dict[str, object], model_name: str) -> Dict[str, object]:
    rows = payload["model_comparison_rows"]
    for row in rows:
        if row["model"] == model_name:
            return row
    raise KeyError(f"Model row not found: {model_name}")


def source_summary(source_root: Path, source: str) -> Dict[str, object]:
    result_path = source_root / source / "split_window_mixed_effects_results.json"
    metadata_path = source_root / source / "cache" / "mimic_map_full_cache_metadata.json"
    with result_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    with metadata_path.open("r", encoding="utf-8") as stream:
        cache_metadata = json.load(stream)

    cohort = payload["cohort_summary"]
    split = payload["split"]
    settings = payload["settings"]
    paired = payload["selected_vs_baseline_paired_loss_reduction"]
    baseline = _model_row(payload, "Baseline covariates quantile")
    raw_q10 = _model_row(payload, "Admission window q10 carry forward")
    updated = _model_row(payload, "Selected baseline-adjusted random-intercept update")
    relative_reduction = 100.0 * (
        float(baseline["assessment_loss"]) - float(updated["assessment_loss"])
    ) / float(baseline["assessment_loss"])

    return {
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "cache_schema_version": cache_metadata.get("cache_schema_version"),
        "analytic_stays": int(cohort["analytic_icu_stays"]),
        "map_observations": int(cohort["total_map_observations"]),
        "split_window_eligible_stays": int(cohort["split_window_eligible_stays"]),
        "train_stays": int(split["train_stays"]),
        "tuning_stays": int(split["tuning_stays"]),
        "assessment_stays": int(split["assessment_stays"]),
        "selected_lambda": float(settings["selected_lambda_baseline_update"]),
        "population_loss": float(baseline["assessment_loss"]),
        "updated_loss": float(updated["assessment_loss"]),
        "paired_reduction": float(paired["mean"]),
        "paired_reduction_se": float(paired["se"]),
        "paired_reduction_ci_low": float(paired["ci95"][0]),
        "paired_reduction_ci_high": float(paired["ci95"][1]),
        "relative_loss_reduction_percent": relative_reduction,
        "raw_q10_loss": float(raw_q10["assessment_loss"]),
        "admission_q10_auc_later_map_below65": float(payload["auc"]["admission_window_q10_low_is_risk"]),
        "admission_later_q10_correlation": float(
            payload["diagnostic_correlations"]["admission_window_q10_vs_later_q10"]
        ),
    }


def write_tex(rows: List[Dict[str, object]], output: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{MAP-source-restricted sensitivity analyses.}",
        r"\label{tab:map_source_sensitivity}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"MAP source & Analytic stays & Split-window stays & Assessment stays & $\lambda_b$ & Population loss & Updated loss & Reduction, \% & AUC\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['source_label']} & {int(row['analytic_stays']):,} & "
            f"{int(row['split_window_eligible_stays']):,} & {int(row['assessment_stays']):,} & "
            f"{float(row['selected_lambda']):.2g} & {float(row['population_loss']):.4f} & "
            f"{float(row['updated_loss']):.4f} & "
            f"{float(row['relative_loss_reduction_percent']):.1f} & "
            f"{float(row['admission_q10_auc_later_map_below65']):.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize{Each row uses a source-specific eligibility cohort and an independently generated stay split. The rows therefore assess within-source robustness and are not paired device comparisons. Reduction is relative to the source-specific population quantile model; AUC uses lower admission-window q10 as the risk score for any later recorded MAP below 65 mmHg.}",
            r"\end{table}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize invasive-only and non-invasive-only recovery analyses.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [source_summary(args.source_root, source) for source in ("invasive", "noninvasive")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "map_source_sensitivity.csv"
    json_path = args.output_dir / "map_source_sensitivity.json"
    tex_path = args.output_dir / "map_source_sensitivity.tex"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"sources": rows}, indent=2), encoding="utf-8")
    write_tex(rows, tex_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
