"""Generate aggregate tables for the expanded Statistics in Medicine supplement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = PAPER_ROOT / "recovery_20260822"
ARCHIVE_PUBLIC_ROOT = RECOVERY_ROOT / "github_release" / "icu-map-quantile-updating"
ARCHIVE_MANUSCRIPT_ROOT = PAPER_ROOT / "manuscript"
PRIVATE_LAYOUT = ARCHIVE_PUBLIC_ROOT.exists() and ARCHIVE_MANUSCRIPT_ROOT.exists()
PUBLIC_ROOT = ARCHIVE_PUBLIC_ROOT if PRIVATE_LAYOUT else PAPER_ROOT
MANUSCRIPT_ROOT = ARCHIVE_MANUSCRIPT_ROOT if PRIVATE_LAYOUT else PAPER_ROOT
TABLE_ROOT = MANUSCRIPT_ROOT / "tables" if PRIVATE_LAYOUT else PAPER_ROOT / "supplement_aggregate"
PRIVATE_VALIDATION = RECOVERY_ROOT / "validation_extensions" if PRIVATE_LAYOUT else None


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

SCENARIO_LABEL = {
    "ideal_large_dense": "Large dense",
    "ideal_small_dense": "Small dense",
    "ideal_large_sparse": "Large sparse",
    "serial_dependence": "Serial dependence",
    "discrete_map_rounding": "Integer rounding",
    "informative_monitoring": "Informative monitoring",
    "cluster_size_informative": "Cluster size association",
    "common_time_misspecified": "Omitted common time",
    "level_plus_shape": "Persistent level with shape",
    "treatment_feedback": "Treatment feedback",
    "transient_nonpersistent": "Transient displacement",
    "null_serial": "Serial null",
    "weak_level": "Weak persistent level",
    "heavy_tail_t3": "Heavy tailed residual",
}

SCENARIO_GROUP = {
    "ideal_large_dense": "Sample size and density",
    "ideal_small_dense": "Sample size and density",
    "ideal_large_sparse": "Sample size and density",
    "serial_dependence": "Observation process",
    "discrete_map_rounding": "Observation process",
    "informative_monitoring": "Observation process",
    "cluster_size_informative": "Observation process",
    "common_time_misspecified": "Trajectory structure",
    "level_plus_shape": "Trajectory structure",
    "treatment_feedback": "Trajectory structure",
    "transient_nonpersistent": "Signal controls",
    "null_serial": "Signal controls",
    "weak_level": "Signal controls",
    "heavy_tail_t3": "Signal controls",
}


def tex(value: object) -> str:
    text = str(value)
    phrase_replacements = {
        "non-informative": "noninformative",
        "sample-size": "sample size",
        "Gaussian-copula": "Gaussian copula",
        "15-minute": "15 minute",
        "stay-specific": "stay specific",
        "level-plus-slope": "level plus slope",
        "MAP-raising": "MAP raising",
        "index-only": "index period",
        "noise-only": "noise",
        "lower-tail": "lower tail",
        "Student-t3": "Student t3",
        "Population-only": "Population",
        "slope-only": "slope",
        "Non-invasive MAP only": "Noninvasive MAP",
        "only two to five": "two to five",
        "Cluster size is associated with the latent stay level even though observation times are otherwise noninformative.":
            "Cluster size is associated with the latent stay level under noninformative observation times.",
        "favoring a correctly targeted level plus slope rule": "favoring a targeted level plus slope update",
        "favoring a correctly targeted level plus slope update": "favoring a targeted level plus slope update",
        "An early low lower tail triggers a later MAP raising intervention whose effect decays over time.":
            "An early low lower tail triggers a later intervention that raises MAP and decays over time.",
        "No persistent level, index shift, or random shape is present; serial residual dependence tests whether tuning suppresses noise offsets.":
            "Serial residual dependence tests whether tuning suppresses updates when persistent level, index shift, and random shape are absent.",
        "No persistent level, index shift, or random shape is present; serial residual dependence tests whether tuning suppresses noise updates.":
            "Serial residual dependence tests whether tuning suppresses updates when persistent level, index shift, and random shape are absent.",
        "A small but persistent stay specific lower tail level": "A small persistent stay specific lower tail level",
        "Penalized stay specific level update": "Penalized stay specific level",
        "Penalized stay specific slope update": "Penalized stay specific slope",
        "Penalized stay specific level + slope update": "Penalized stay specific level plus slope",
    }
    for old, new in phrase_replacements.items():
        text = text.replace(old, new)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("--", " ")


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return pattern.format(float(value))


def write(name: str, lines: list[str]) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    (TABLE_ROOT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_design_table() -> None:
    design = pd.read_csv(PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_design.csv").set_index(
        "scenario_key"
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.14\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\centering\arraybackslash}p{0.06\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.43\textwidth}}",
        r"\caption{Complete ADEMP mechanism specification.}\label{tab:supp_ademp_design}\\",
        r"\toprule",
        r"Group & Mechanism & Stays & Index/later counts & Defining feature\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Group & Mechanism & Stays & Index/later counts & Defining feature\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{5}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for key in SCENARIO_ORDER:
        row = design.loc[key]
        counts = f"{row['index_count_range']} / {row['late_count_range']}"
        lines.append(
            f"{tex(SCENARIO_GROUP[key])} & {tex(SCENARIO_LABEL[key])} & "
            f"{int(row['n_stays'])} & {tex(counts)} & {tex(row['description'])}\\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    write("supp_ademp_design.tex", lines)


def paired_lookup(frame: pd.DataFrame, key: str, method_a: str, method_b: str) -> pd.Series:
    rows = frame.loc[
        frame["scenario_key"].eq(key)
        & frame["method_a"].eq(method_a)
        & frame["method_b"].eq(method_b)
    ]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Missing paired result for {key}: {method_a} - {method_b}")
    return rows.iloc[0]


def diff_cell(row: pd.Series) -> str:
    return f"{fmt(row['loss_difference_mean'], 4, True)} ({fmt(row['loss_difference_mcse'], 4)})"


def generate_performance_table() -> None:
    paired = pd.read_csv(
        PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_paired_loss_comparisons.csv"
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.19\textwidth}*{4}{>{\centering\arraybackslash}p{0.185\textwidth}}}",
        r"\caption{Complete paired Monte Carlo loss comparisons.}\label{tab:supp_ademp_performance}\\",
        r"\toprule",
        r"Mechanism & Level minus population & Calibrated q10 minus population & Level minus calibrated q10 & Level with slope minus level\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Mechanism & Level minus population & Calibrated q10 minus population & Level minus calibrated q10 & Level with slope minus level\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    pairs = [
        ("tuned_level", "population"),
        ("affine_calibrated_q10", "population"),
        ("tuned_level", "affine_calibrated_q10"),
        ("tuned_level_slope", "tuned_level"),
    ]
    for key in SCENARIO_ORDER:
        cells = [diff_cell(paired_lookup(paired, key, a, b)) for a, b in pairs]
        lines.append(f"{tex(SCENARIO_LABEL[key])} & " + " & ".join(cells) + r"\\")
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{Entries are Monte Carlo means with MCSE in parentheses across 200 independent replicates. Negative values favor the first rule named in the column heading.}",
            r"\endgroup",
        ]
    )
    write("supp_ademp_performance.tex", lines)


def monte_carlo_mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def generate_diagnostic_table() -> None:
    diagnostics = pd.read_csv(
        PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_scenario_diagnostics.csv"
    )
    penalties = pd.read_csv(
        PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_penalty_summary.csv"
    )
    summary = pd.read_csv(PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_summary.csv")
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.18\textwidth}*{2}{>{\centering\arraybackslash}p{0.09\textwidth}}>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}}",
        r"\caption{Realized simulation diagnostics and penalty behavior.}\label{tab:supp_ademp_diagnostics}\\",
        r"\toprule",
        r"Mechanism & Index count & Later count & Count level correlation & Error lag 1 & Modal $\lambda_b$ (frequency) & Offset correlation\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Mechanism & Index count & Later count & Count level correlation & Error lag 1 & Modal $\lambda_b$ (frequency) & Offset correlation\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for key in SCENARIO_ORDER:
        d = diagnostics.loc[diagnostics["scenario_key"].eq(key)]
        p = penalties.loc[
            penalties["scenario_key"].eq(key) & penalties["method"].eq("tuned_level")
        ].sort_values(["selected_count", "selected_penalty"], ascending=[False, True])
        modal = p.iloc[0]
        s = summary.loc[
            summary["scenario_key"].eq(key) & summary["method"].eq("tuned_level")
        ].iloc[0]
        count_corr = d["total_count_latent_level_correlation"].mean()
        lag_corr = d["observed_error_lag1_correlation"].mean()
        offset_corr = s["offset_correlation_mean"]
        lines.append(
            f"{tex(SCENARIO_LABEL[key])} & {fmt(d['mean_index_count'].mean(), 2)} & "
            f"{fmt(d['mean_late_count'].mean(), 2)} & {fmt(count_corr, 3)} & "
            f"{fmt(lag_corr, 3)} & {fmt(modal['selected_penalty'], 2)} "
            f"({100 * modal['selected_proportion']:.1f}\\%) & {fmt(offset_corr, 3)}\\\\"
        )

    primary = pd.read_csv(
        PUBLIC_ROOT / "simulation_ademp_v2" / "ademp_v2_method_replicates.csv"
    )
    extended = pd.read_csv(
        PUBLIC_ROOT / "transient_grid_sensitivity" / "ademp_v2_method_replicates.csv"
    )

    def transient_differences(frame: pd.DataFrame) -> pd.Series:
        use = frame.loc[
            frame["scenario_key"].eq("transient_nonpersistent")
            & frame["method"].isin(["population", "tuned_level"]),
            ["group_id", "method", "loss"],
        ]
        wide = use.pivot(index="group_id", columns="method", values="loss")
        return wide["tuned_level"] - wide["population"]

    primary_diff = transient_differences(primary)
    extended_diff = transient_differences(extended)
    paired_change = extended_diff.sort_index() - primary_diff.sort_index()
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\textit{Transient penalty grid sensitivity}}\\",
            r"Primary penalty grid & \multicolumn{3}{r}{Level minus population} & "
            + rf"\multicolumn{{3}}{{l}}{{{fmt(primary_diff.mean(), 6, True)} ({fmt(primary_diff.std(ddof=1) / np.sqrt(primary_diff.size), 6)})}}\\",
            r"Extended penalty grid & \multicolumn{3}{r}{Level minus population} & "
            + rf"\multicolumn{{3}}{{l}}{{{fmt(extended_diff.mean(), 6, True)} ({fmt(extended_diff.std(ddof=1) / np.sqrt(extended_diff.size), 6)})}}\\",
            r"Extended minus primary & \multicolumn{3}{r}{Paired change} & "
            + rf"\multicolumn{{3}}{{l}}{{{fmt(paired_change.mean(), 6, True)} ({fmt(paired_change.std(ddof=1) / np.sqrt(paired_change.size), 6)})}}\\",
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{Counts, correlations, and treatment fractions are means across 200 replicates. Blank offset correlations occur when the true persistent level has zero variance. Every method completed 200 replicates per mechanism. The primary penalty grid is $\{0,0.03,0.10,0.30,1,3,10\}$; the extended grid adds 30, 100, and 300. Transient sensitivity values are paired loss differences with MCSE in parentheses.}",
            r"\endgroup",
        ]
    )
    write("supp_ademp_diagnostics.tex", lines)


def generate_clinical_table() -> None:
    strata = pd.read_csv(PUBLIC_ROOT / "empirical_aggregate" / "clinical_outcome_strata.csv")
    associations = pd.read_csv(
        PUBLIC_ROOT / "empirical_aggregate" / "enhanced_outcome_associations.csv"
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.18\textwidth}*{7}{>{\centering\arraybackslash}p{0.10\textwidth}}}",
        r"\caption{Clinical outcome gradients and adjusted associations.}\label{tab:supp_clinical_evidence}\\",
        r"\toprule",
        r"Vulnerability stratum & Stays & q10 & Later MAP $<65$ & Any later MAP $<65$ & Hospital death & ICU death & ICU stay $\ge3$ days\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Vulnerability stratum & Stays & q10 & Later MAP $<65$ & Any later MAP $<65$ & Hospital death & ICU death & ICU stay $\ge3$ days\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in strata.iterrows():
        lines.append(
            f"{tex(row['stratum'])} & {int(row['stays'])} & {row['admission_window_q10_median']:.1f} & "
            f"{100*row['later_map_below65_fraction']:.1f}\\% & {100*row['any_later_map_below65']:.1f}\\% & "
            f"{100*row['hospital_mortality']:.1f}\\% & {100*row['icu_mortality']:.1f}\\% & "
            f"{100*row['icu_los_ge3_days']:.1f}\\%\\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{8}{l}{\textit{Adjusted logistic associations}}\\",
            r"Outcome and score & Events & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{Odds ratio per 10 mmHg} & \multicolumn{2}{c}{95\% interval}\\",
        ]
    )
    for _, row in associations.iterrows():
        outcome = str(row["outcome"]).replace("Any later MAP <65", "Any later MAP below 65")
        outcome = outcome.replace("ICU LOS >=3 days", "ICU stay at least 3 days")
        label = f"{outcome}: {row['score']}"
        lines.append(
            rf"{tex(label)} & {int(row['events'])} & \multicolumn{{2}}{{c}}{{{row['auc']:.3f}}} & "
            rf"\multicolumn{{2}}{{c}}{{{row['adjusted_or_per_10']:.2f}}} & "
            rf"\multicolumn{{2}}{{c}}{{{row['ci_low']:.2f} to {row['ci_high']:.2f}}}\\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{Strata order admission window q10 from high to low. Regressions adjust for age, sex, and emergency or urgent admission. Larger scores in the association panel denote lower q10 or larger negative profiled offsets.}",
            r"\endgroup",
        ]
    )
    write("supp_clinical_evidence.tex", lines)


def generate_empirical_sensitivity_table() -> None:
    sensitivity = pd.read_csv(PUBLIC_ROOT / "empirical_aggregate" / "sensitivity_analysis.csv")
    weighted = pd.read_csv(
        PUBLIC_ROOT / "empirical_aggregate" / "stay_weighted_population_sensitivity.csv"
    )
    sources = pd.read_csv(PUBLIC_ROOT / "source_sensitivity" / "map_source_sensitivity.csv")
    density = pd.read_csv(
        PUBLIC_ROOT / "empirical_aggregate" / "measurement_density_sensitivity.csv"
    )
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}rrrrr}",
        r"\caption{Empirical sensitivity to target, monitoring, weighting, and MAP source.}\label{tab:supp_empirical_sensitivity}\\",
        r"\toprule",
        r"Analysis & Setting & Stays & $\lambda_b$ & Population loss & Profiled loss & Reduction\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Analysis & Setting & Stays & $\lambda_b$ & Population loss & Profiled loss & Reduction\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in sensitivity.iterrows():
        setting = f"Quantile {row['tau']:.2f}, landmark {int(row['index_hours'])} h"
        lines.append(
            f"Quantile and landmark & {tex(setting)} & {int(row['n_stays'])} & {row['lambda_b']:.2f} & "
            f"{row['population_assessment_loss']:.4f} & {row['updated_assessment_loss']:.4f} & "
            f"{row['loss_reduction_percent']:.1f}\\%\\\\"
        )
    row = weighted.loc[weighted["model"].eq("Stay-weighted population + level update")].iloc[0]
    base = weighted.loc[weighted["model"].eq("Stay-weighted population")].iloc[0]
    lines.append(
        f"Training weight & Equal total weight per stay & {int(row['n_stays'])} & {row['selected_lambda']:.2f} & "
        f"{base['assessment_loss']:.4f} & {row['assessment_loss']:.4f} & {row['loss_reduction_percent']:.1f}\\%\\\\"
    )
    for _, row in sources.iterrows():
        lines.append(
            f"MAP source & {tex(row['source_label'])} & {int(row['assessment_stays'])} & {row['selected_lambda']:.2f} & "
            f"{row['population_loss']:.4f} & {row['updated_loss']:.4f} & "
            f"{row['relative_loss_reduction_percent']:.1f}\\%\\\\"
        )
    for _, row in density.iterrows():
        setting = f"{row['density_measure']}, {row['stratum']} (median {row['median_index_obs']:.0f}/{row['median_late_obs']:.0f})"
        lines.append(
            f"Measurement density & {tex(setting)} & {int(row['stays'])} & 0.03 & "
            f"{row['population_assessment_loss']:.4f} & {row['updated_assessment_loss']:.4f} & "
            f"{row['loss_reduction_percent']:.1f}\\%\\\\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{The measurement density setting reports median index/later counts in parentheses. Every row refits or reuses the population component and selects the stated penalty before assessment according to its analysis protocol.}",
            r"\endgroup",
        ]
    )
    write("supp_empirical_sensitivity.tex", lines)


def generate_candidate_table() -> None:
    random_effects = pd.read_csv(
        PUBLIC_ROOT / "empirical_aggregate" / "random_effect_structure_comparison.csv"
    )
    model_loss = pd.read_csv(PUBLIC_ROOT / "validation_aggregate" / "model_loss_summary.csv")
    paired = pd.read_csv(PUBLIC_ROOT / "validation_aggregate" / "paired_comparator_results.csv")
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}*{3}{>{\centering\arraybackslash}p{0.10\textwidth}}}",
        r"\caption{Stay specific structures and complete scalar rule comparisons.}\label{tab:supp_candidate_evaluation}\\",
        r"\toprule",
        r"Scope & Rule & Penalty (level, slope) or calibration & Stays & Mean loss & Difference\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Scope & Rule & Penalty (level, slope) or calibration & Stays & Mean loss & Difference\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        r"\multicolumn{6}{l}{\textit{Historical structure comparison; difference versus population}}\\",
    ]
    pop_loss = float(random_effects.iloc[0]["assessment_loss"])
    for _, row in random_effects.iterrows():
        penalty = ""
        if np.isfinite(row["lambda_intercept"]):
            penalty = f"$({row['lambda_intercept']:.2f},{row['lambda_slope']:.2f})$"
        lines.append(
            f"Historical split & {tex(row['model'])} & {penalty} & {int(row['n_stays'])} & "
            f"{row['assessment_loss']:.4f} & {row['assessment_loss'] - pop_loss:+.4f}\\\\"
        )

    lines.append(r"\midrule\multicolumn{6}{l}{\textit{Nested fivefold scalar comparison; difference versus calibrated q10}}\\")
    scope = "nested_5fold_internal_crossfit"
    selected_models = [
        "population",
        "raw_q10",
        "primary_level_update",
        "calibrated_last",
        "calibrated_min",
        "calibrated_mean",
        "calibrated_below65_burden",
        "calibrated_q10",
    ]
    display = {
        "population": "Population quantile component",
        "raw_q10": "Raw admission window q10",
        "primary_level_update": "Primary profiled offset",
        "calibrated_last": "Calibrated last MAP",
        "calibrated_min": "Calibrated minimum MAP",
        "calibrated_mean": "Calibrated mean MAP",
        "calibrated_below65_burden": "Calibrated fraction below 65",
        "calibrated_q10": "Calibrated admission window q10",
    }
    for model in selected_models:
        row = model_loss.loc[model_loss["scope"].eq(scope) & model_loss["model"].eq(model)].iloc[0]
        if model == "calibrated_q10":
            difference = "Reference"
        else:
            pr = paired.loc[
                paired["scope"].eq(scope)
                & paired["candidate"].eq(model)
                & paired["reference"].eq("calibrated_q10")
            ].iloc[0]
            difference = f"{pr['paired_difference_candidate_minus_reference']:+.4f}"
        lines.append(
            f"Nested cross fitting & {tex(display[model])} & Inner tuning & {int(row['n_stays'])} & "
            f"{row['mean_stay_level_check_loss']:.4f} & {difference}\\\\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{Historical structure comparisons use the original assessment split. Nested results contain one out of fold prediction for every eligible stay. Differences in the nested panel equal candidate loss minus calibrated q10 loss.}",
            r"\endgroup",
        ]
    )
    write("supp_candidate_evaluation.tex", lines)


def generate_fold_table() -> None:
    precomputed_source = PUBLIC_ROOT / "supplement_aggregate" / "supp_fold_stability_source.csv"
    use_precomputed = PRIVATE_VALIDATION is None
    if use_precomputed:
        if not precomputed_source.exists():
            raise FileNotFoundError(
                "Public regeneration requires supplement_aggregate/supp_fold_stability_source.csv"
            )
        aggregate_rows = pd.read_csv(precomputed_source).to_dict(orient="records")
    else:
        predictions = pd.read_csv(PRIVATE_VALIDATION / "nested_crossfit_stay_level_predictions.csv")
        aggregate_rows = []
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{*{8}{>{\centering\arraybackslash}p{0.115\textwidth}}}",
        r"\caption{Fold specific tuning and outer assessment results.}\label{tab:supp_fold_stability}\\",
        r"\toprule",
        r"Fold & $\lambda_b$ & q10 intercept & q10 slope & Tuning q10 loss & Population loss & Profiled loss & Calibrated q10 loss\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Fold & $\lambda_b$ & q10 intercept & q10 slope & Tuning q10 loss & Population loss & Profiled loss & Calibrated q10 loss\\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    if not use_precomputed:
        for fold in range(1, 6):
            metadata_path = (
                PRIVATE_VALIDATION
                / "work"
                / "nested_crossfit"
                / f"fold_{fold}"
                / "fold_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            calibration = metadata["calibrations"]["calibrated_q10"]
            outer = predictions.loc[predictions["outer_fold"].eq(fold)]
            aggregate_rows.append(
                {
                    "fold": fold,
                    "lambda_b": float(metadata["selected_lambda_b"]),
                    "intercept": float(calibration["intercept"]),
                    "slope": float(calibration["slope"]),
                    "tuning_loss": float(calibration["tuning_mean_stay_level_check_loss"]),
                    "population_loss": float(outer["loss_population"].mean()),
                    "profiled_loss": float(outer["loss_primary_level_update"].mean()),
                    "q10_loss": float(outer["loss_calibrated_q10"].mean()),
                }
            )
    for values in aggregate_rows:
        fold = int(values["fold"])
        lines.append(
            f"{fold} & {values['lambda_b']:.2f} & {values['intercept']:.3f} & {values['slope']:.3f} & "
            f"{values['tuning_loss']:.4f} & {values['population_loss']:.4f} & "
            f"{values['profiled_loss']:.4f} & {values['q10_loss']:.4f}\\\\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize{Each fold fits the population component on 36,186 stays, selects the penalty and affine summary on 12,062 inner tuning stays, and scores about 12,062 outer stays.}",
            r"\endgroup",
        ]
    )
    write("supp_fold_stability.tex", lines)
    pd.DataFrame(aggregate_rows).to_csv(TABLE_ROOT / "supp_fold_stability_source.csv", index=False)


def main() -> None:
    generate_design_table()
    generate_performance_table()
    generate_diagnostic_table()
    generate_clinical_table()
    generate_empirical_sensitivity_table()
    generate_candidate_table()
    generate_fold_table()
    print(f"Wrote expanded supplement tables to {TABLE_ROOT}")


if __name__ == "__main__":
    main()
