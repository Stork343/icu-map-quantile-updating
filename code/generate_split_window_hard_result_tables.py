import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import split_window_data as data_utils
from run_split_window_mixed_effects_analysis import (
    apply_training_age_standardization,
    se_mean,
    split_cluster_indices,
)
from split_window_analysis_core import check_loss, design_frame, empirical_check_quantile, profiled_intercept
from split_window_data import _safe_read_frame, build_dataset_from_cache


X_PREDICTORS = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
DIRECT_IDENTIFIER_COLUMNS = ("subject_id", "hadm_id", "stay_id")


def public_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in DIRECT_IDENTIFIER_COLUMNS if col in df.columns])


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


def load_split_designs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, _ = build_dataset_from_cache(
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
    tuning_data = data_utils.subset_cluster_data(dataset, tuning_idx)
    assessment_data = data_utils.subset_cluster_data(dataset, assessment_idx)
    tuning_design, _ = design_frame(tuning_data)
    assessment_design, _ = design_frame(assessment_data)
    metadata = {
        "data_summary": data_summary,
        "age_standardization": age_standardization,
        "split": {
            "train_stays": int(train_idx.size),
            "tuning_stays": int(tuning_idx.size),
            "assessment_stays": int(assessment_idx.size),
        },
    }
    return tuning_design, assessment_design, metadata


def stay_level_prediction_frame(
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
        offset = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
        y_index = y[index_idx]
        y_late = y[late_idx]
        pop_pred_late = fitted[late_idx]
        updated_pred_late = pop_pred_late + offset
        rows.append(
            {
                "stay_index": int(design["stay_index"].iat[start]),
                "stay_id": int(design["stay_id"].iat[start]),
                "index_obs": int(index_idx.size),
                "late_obs": int(late_idx.size),
                "admission_window_q10": empirical_check_quantile(y_index, tau),
                "index_map_below65_fraction": float(np.mean(y_index < 65.0)),
                "later_q10": empirical_check_quantile(y_late, tau),
                "later_map_below65_fraction": float(np.mean(y_late < 65.0)),
                "any_later_map_below65": float(np.any(y_late < 65.0)),
                "population_q10_median": float(np.median(pop_pred_late)),
                "updated_q10_median": float(np.median(updated_pred_late)),
                "updated_offset": float(offset),
                "updated_vulnerability_score": float(-offset),
                "population_check_loss": float(np.mean(check_loss(y_late - pop_pred_late, tau))),
                "updated_check_loss": float(np.mean(check_loss(y_late - updated_pred_late, tau))),
                "population_below_fraction": float(np.mean(y_late < pop_pred_late)),
                "updated_below_fraction": float(np.mean(y_late < updated_pred_late)),
                "population_below_count": int(np.sum(y_late < pop_pred_late)),
                "updated_below_count": int(np.sum(y_late < updated_pred_late)),
            }
        )
    return pd.DataFrame(rows)


def weighted_later_arrays(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    predictor_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    predictor_lookup = stay_df.set_index("stay_index")[predictor_column].to_dict()
    y_parts: List[np.ndarray] = []
    predictor_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []
    for start, end in contiguous_stay_slices(design):
        stay_index = int(design["stay_index"].iat[start])
        if stay_index not in predictor_lookup:
            continue
        t = t_all[start:end]
        y = y_all[start:end]
        _, late_idx = strict_split_indices(t, 12.0)
        if late_idx.size == 0:
            continue
        y_late = y[late_idx]
        predictor_value = float(predictor_lookup[stay_index])
        y_parts.append(y_late)
        predictor_parts.append(np.full(y_late.size, predictor_value, dtype=float))
        w_parts.append(np.full(y_late.size, 1.0 / y_late.size, dtype=float))
    return np.concatenate(y_parts), np.concatenate(predictor_parts), np.concatenate(w_parts)


def weighted_later_matrix(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    predictor_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    predictor_lookup = stay_df.set_index("stay_index").loc[:, predictor_columns].to_dict("index")
    y_parts: List[np.ndarray] = []
    x_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []
    for start, end in contiguous_stay_slices(design):
        stay_index = int(design["stay_index"].iat[start])
        if stay_index not in predictor_lookup:
            continue
        t = t_all[start:end]
        y = y_all[start:end]
        _, late_idx = strict_split_indices(t, 12.0)
        if late_idx.size == 0:
            continue
        y_late = y[late_idx]
        predictor_values = np.asarray([predictor_lookup[stay_index][col] for col in predictor_columns], dtype=float)
        x_row = np.r_[1.0, predictor_values]
        y_parts.append(y_late)
        x_parts.append(np.repeat(x_row[None, :], y_late.size, axis=0))
        w_parts.append(np.full(y_late.size, 1.0 / y_late.size, dtype=float))
    return np.concatenate(y_parts), np.vstack(x_parts), np.concatenate(w_parts)


def fit_calibrated_predictor(
    tuning_df: pd.DataFrame,
    tuning_design: pd.DataFrame,
    tau: float,
    predictor_column: str,
) -> Dict[str, object]:
    y, predictor, weights = weighted_later_arrays(tuning_df, tuning_design, predictor_column)
    n_stays = float(tuning_df.shape[0])

    def objective(par: np.ndarray) -> float:
        pred = float(par[0]) + float(par[1]) * predictor
        return float(np.sum(weights * check_loss(y - pred, tau)) / n_stays)

    starts = [
        np.array([0.0, 1.0], dtype=float),
        np.array([empirical_check_quantile(y, tau), 0.0], dtype=float),
        np.array([10.0, 0.8], dtype=float),
    ]
    best = None
    for x0 in starts:
        fit = minimize(
            objective,
            x0=x0,
            method="Powell",
            options={"xtol": 1e-8, "ftol": 1e-8, "maxiter": 10000, "disp": False},
        )
        if best is None or float(fit.fun) < float(best.fun):
            best = fit
    if best is None or not np.all(np.isfinite(best.x)):
        raise RuntimeError(f"Unable to fit calibrated predictor: {predictor_column}.")
    return {
        "predictor_column": predictor_column,
        "intercept": float(best.x[0]),
        "slope": float(best.x[1]),
        "tuning_loss": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }


def fit_calibrated_model(
    tuning_df: pd.DataFrame,
    tuning_design: pd.DataFrame,
    tau: float,
    predictor_columns: Sequence[str],
) -> Dict[str, object]:
    y, x, weights = weighted_later_matrix(tuning_df, tuning_design, predictor_columns)
    n_stays = float(tuning_df.shape[0])

    def objective(par: np.ndarray) -> float:
        pred = x @ par
        return float(np.sum(weights * check_loss(y - pred, tau)) / n_stays)

    starts = [
        np.r_[empirical_check_quantile(y, tau), np.zeros(len(predictor_columns))],
        np.r_[0.0, np.ones(len(predictor_columns)) / max(len(predictor_columns), 1)],
    ]
    if "admission_window_q10" in predictor_columns:
        start = np.zeros(1 + len(predictor_columns), dtype=float)
        start[0] = 18.688921950131046
        start[1 + list(predictor_columns).index("admission_window_q10")] = 0.6722711876790879
        starts.append(start)

    best = None
    for x0 in starts:
        fit = minimize(
            objective,
            x0=x0,
            method="Powell",
            options={"xtol": 1e-8, "ftol": 1e-8, "maxiter": 20000, "disp": False},
        )
        if best is None or float(fit.fun) < float(best.fun):
            best = fit
    if best is None or not np.all(np.isfinite(best.x)):
        raise RuntimeError(f"Unable to fit calibrated model: {predictor_columns}.")
    return {
        "predictor_columns": list(predictor_columns),
        "coefficients": [float(v) for v in best.x],
        "tuning_loss": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }


def calibrated_losses(
    stay_df: pd.DataFrame,
    intercept: float,
    slope: float,
    tau: float,
) -> pd.DataFrame:
    out = stay_df.copy()
    pred = intercept + slope * out["admission_window_q10"].to_numpy(dtype=float)
    out["calibrated_admission_q10_prediction"] = pred
    out["calibrated_admission_q10_loss"] = np.nan
    out["calibrated_admission_q10_below_fraction"] = np.nan
    return out


def add_calibrated_loss_from_design(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    intercept: float,
    slope: float,
    tau: float,
    predictor_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    df = stay_df.copy()
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    predictor_lookup = df.set_index("stay_index")[predictor_column].to_dict()
    losses: Dict[int, float] = {}
    coverages: Dict[int, float] = {}
    below_counts: Dict[int, int] = {}
    for start, end in contiguous_stay_slices(design):
        stay_index = int(design["stay_index"].iat[start])
        if stay_index not in predictor_lookup:
            continue
        t = t_all[start:end]
        y = y_all[start:end]
        _, late_idx = strict_split_indices(t, 12.0)
        if late_idx.size == 0:
            continue
        pred = float(intercept + slope * float(predictor_lookup[stay_index]))
        y_late = y[late_idx]
        losses[stay_index] = float(np.mean(check_loss(y_late - pred, tau)))
        coverages[stay_index] = float(np.mean(y_late < pred))
        below_counts[stay_index] = int(np.sum(y_late < pred))
    df[f"{output_prefix}_prediction"] = [
        float(intercept + slope * value) for value in df[predictor_column].to_numpy(dtype=float)
    ]
    df[f"{output_prefix}_loss"] = df["stay_index"].map(losses).astype(float)
    df[f"{output_prefix}_below_fraction"] = df["stay_index"].map(coverages).astype(float)
    df[f"{output_prefix}_below_count"] = df["stay_index"].map(below_counts).astype(float)
    return df


def add_calibrated_model_loss_from_design(
    stay_df: pd.DataFrame,
    design: pd.DataFrame,
    calibration: Dict[str, object],
    tau: float,
    output_prefix: str,
) -> pd.DataFrame:
    df = stay_df.copy()
    predictor_columns = list(calibration["predictor_columns"])
    coef = np.asarray(calibration["coefficients"], dtype=float)
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    predictor_lookup = df.set_index("stay_index").loc[:, predictor_columns].to_dict("index")
    losses: Dict[int, float] = {}
    coverages: Dict[int, float] = {}
    below_counts: Dict[int, int] = {}
    predictions: Dict[int, float] = {}
    for start, end in contiguous_stay_slices(design):
        stay_index = int(design["stay_index"].iat[start])
        if stay_index not in predictor_lookup:
            continue
        t = t_all[start:end]
        y = y_all[start:end]
        _, late_idx = strict_split_indices(t, 12.0)
        if late_idx.size == 0:
            continue
        x = np.r_[1.0, [predictor_lookup[stay_index][col] for col in predictor_columns]]
        pred = float(x @ coef)
        y_late = y[late_idx]
        losses[stay_index] = float(np.mean(check_loss(y_late - pred, tau)))
        coverages[stay_index] = float(np.mean(y_late < pred))
        below_counts[stay_index] = int(np.sum(y_late < pred))
        predictions[stay_index] = pred
    df[f"{output_prefix}_prediction"] = df["stay_index"].map(predictions).astype(float)
    df[f"{output_prefix}_loss"] = df["stay_index"].map(losses).astype(float)
    df[f"{output_prefix}_below_fraction"] = df["stay_index"].map(coverages).astype(float)
    df[f"{output_prefix}_below_count"] = df["stay_index"].map(below_counts).astype(float)
    return df


def loss_summary(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), se_mean(arr)


def paired_loss_difference(
    stay_df: pd.DataFrame,
    candidate_column: str,
    reference_column: str = "calibrated_admission_q10_loss",
) -> Dict[str, float]:
    diff = stay_df[candidate_column].to_numpy(dtype=float) - stay_df[reference_column].to_numpy(dtype=float)
    mean, se = loss_summary(diff)
    return {
        "paired_diff_vs_calibrated_q10": mean,
        "paired_diff_vs_calibrated_q10_se": se,
        "paired_diff_vs_calibrated_q10_ci_lower": mean - 1.96 * se,
        "paired_diff_vs_calibrated_q10_ci_upper": mean + 1.96 * se,
    }


def comparator_rows(
    stay_df: pd.DataFrame,
    admission_q10_calibration: Dict[str, object],
    updated_q10_calibration: Dict[str, object],
    hybrid_calibrations: Sequence[Tuple[str, str, Dict[str, object], str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model_specs = [
        ("Baseline covariates quantile", "none", "population_check_loss"),
        ("Admission window q10 carry forward", "raw admission window q10", "admission_q10_loss"),
        ("Primary penalized stay-specific level update", "profiled level offset", "updated_check_loss"),
        ("Calibrated admission window q10", "tuning-calibrated affine q10", "calibrated_admission_q10_loss"),
        ("Calibrated primary level update", "tuning-calibrated level-updated q10", "calibrated_updated_q10_loss"),
    ]
    model_specs.extend((model, info, f"{prefix}_loss") for model, info, _, prefix in hybrid_calibrations)
    for model, info, column in model_specs:
        mean, se = loss_summary(stay_df[column].to_numpy(dtype=float))
        row = {
            "model": model,
            "index_information": info,
            "assessment_loss": mean,
            "assessment_loss_se": se,
            "n_stays": int(stay_df.shape[0]),
        }
        if column != "calibrated_admission_q10_loss":
            row.update(paired_loss_difference(stay_df, column))
        if column == "calibrated_admission_q10_loss":
            row["calibration_intercept"] = float(admission_q10_calibration["intercept"])
            row["calibration_slope"] = float(admission_q10_calibration["slope"])
            row["calibration_tuning_loss"] = float(admission_q10_calibration["tuning_loss"])
        if column == "calibrated_updated_q10_loss":
            row["calibration_intercept"] = float(updated_q10_calibration["intercept"])
            row["calibration_slope"] = float(updated_q10_calibration["slope"])
            row["calibration_tuning_loss"] = float(updated_q10_calibration["tuning_loss"])
        for hybrid_model, _, calibration, prefix in hybrid_calibrations:
            if model == hybrid_model:
                row["calibration_tuning_loss"] = float(calibration["tuning_loss"])
                row["calibration_coefficients"] = ";".join(f"{v:.6g}" for v in calibration["coefficients"])
        rows.append(row)
    return rows


def add_raw_admission_q10_loss(stay_df: pd.DataFrame, design: pd.DataFrame, tau: float) -> pd.DataFrame:
    df = stay_df.copy()
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    q_lookup = df.set_index("stay_index")["admission_window_q10"].to_dict()
    losses: Dict[int, float] = {}
    below: Dict[int, float] = {}
    for start, end in contiguous_stay_slices(design):
        stay_index = int(design["stay_index"].iat[start])
        if stay_index not in q_lookup:
            continue
        t = t_all[start:end]
        y = y_all[start:end]
        _, late_idx = strict_split_indices(t, 12.0)
        if late_idx.size == 0:
            continue
        pred = float(q_lookup[stay_index])
        y_late = y[late_idx]
        losses[stay_index] = float(np.mean(check_loss(y_late - pred, tau)))
        below[stay_index] = float(np.mean(y_late < pred))
    df["admission_q10_loss"] = df["stay_index"].map(losses).astype(float)
    df["admission_q10_below_fraction"] = df["stay_index"].map(below).astype(float)
    return df


def quantile_calibration_rows(stay_df: pd.DataFrame, tau: float) -> List[Dict[str, object]]:
    df = stay_df.copy()
    df["updated_q10_quintile"] = pd.qcut(df["updated_q10_median"].rank(method="first"), 5, labels=False) + 1
    labels = {
        1: "Lowest updated q10",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Highest updated q10",
    }
    rows: List[Dict[str, object]] = []

    def summarize(local: pd.DataFrame, label: str) -> Dict[str, object]:
        later_obs = int(local["late_obs"].sum())
        return {
            "stratum": label,
            "stays": int(local.shape[0]),
            "later_observations": later_obs,
            "updated_q10_median": float(local["updated_q10_median"].median()),
            "population_below_fraction_observation": float(local["population_below_count"].sum() / later_obs),
            "updated_below_fraction_observation": float(local["updated_below_count"].sum() / later_obs),
            "population_below_fraction_stay_mean": float(local["population_below_fraction"].mean()),
            "updated_below_fraction_stay_mean": float(local["updated_below_fraction"].mean()),
            "later_map_below65_fraction": float(local["later_map_below65_fraction"].mean()),
            "target": float(tau),
        }

    rows.append(summarize(df, "Overall"))
    for quintile in range(1, 6):
        rows.append(summarize(df[df["updated_q10_quintile"] == quintile], labels[quintile]))
    return rows


def calibration_curve_rows(
    stay_df: pd.DataFrame,
    tau: float,
    groups: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = [
        ("Population QR", "population_q10_median", "population_below_count"),
        ("Primary level update", "updated_q10_median", "updated_below_count"),
        ("Calibrated update", "calibrated_updated_q10_prediction", "calibrated_updated_q10_below_count"),
        ("Calibrated q10", "calibrated_admission_q10_prediction", "calibrated_admission_q10_below_count"),
    ]
    curve_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    for model, prediction_col, below_count_col in models:
        ranked = stay_df[prediction_col].rank(method="first")
        decile = pd.qcut(ranked, groups, labels=False) + 1
        ace_terms: List[float] = []
        weighted_abs = 0.0
        total_obs = float(stay_df["late_obs"].sum())
        for level in range(1, groups + 1):
            local = stay_df.loc[decile == level]
            if local.empty:
                continue
            later_obs = float(local["late_obs"].sum())
            below_fraction = float(local[below_count_col].sum() / later_obs)
            abs_error = abs(below_fraction - tau)
            ace_terms.append(abs_error)
            weighted_abs += (later_obs / total_obs) * abs_error
            curve_rows.append(
                {
                    "model": model,
                    "decile": int(level),
                    "stays": int(local.shape[0]),
                    "later_observations": int(later_obs),
                    "predicted_q10_median": float(local[prediction_col].median()),
                    "observed_below_fraction": below_fraction,
                    "target": float(tau),
                }
            )
        error_rows.append(
            {
                "model": model,
                "groups": int(groups),
                "ace": float(np.mean(ace_terms)),
                "wce": float(weighted_abs),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(error_rows)


def write_calibration_error_tex(rows: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Decile-level later-window quantile calibration error in the assessment split.}",
        "\\label{tab:calibration_error}",
        "\\centering",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Model & ACE & WCE\\\\",
        "\\hline",
    ]
    for row in rows.to_dict("records"):
        lines.append(f"{row['model']} & {float(row['ace']):.3f} & {float(row['wce']):.3f}\\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{ACE is the unweighted mean absolute deviation between the observed below-prediction fraction and the nominal $\\tau=0.10$ target across deciles of predicted q10. WCE weights the same absolute deviations by the number of later observations in each decile.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_calibration_curve(curve: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Population QR": "0.40",
        "Primary level update": "#1f77b4",
        "Calibrated update": "#ff7f0e",
        "Calibrated q10": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in colors:
        local = curve[curve["model"] == model]
        ax.plot(
            local["decile"],
            100.0 * local["observed_below_fraction"],
            marker="o",
            lw=1.8,
            label=model,
            color=colors[model],
        )
    ax.axhline(10.0, color="black", lw=0.9, ls="--", label="Nominal 10%")
    ax.set_xlabel("Decile of predicted MAP 0.10 quantile")
    ax.set_ylabel("Later MAP below fitted q10 (%)")
    ax.set_xlim(1, 10)
    ax.set_ylim(0, max(24.0, 100.0 * float(curve["observed_below_fraction"].max()) + 2.0))
    ax.set_xticks(np.arange(1, 11))
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def density_sensitivity_rows(stay_df: pd.DataFrame) -> List[Dict[str, object]]:
    df = stay_df.copy()
    rows: List[Dict[str, object]] = []

    def add_rows(column: str, label: str) -> None:
        values = df[column].to_numpy(dtype=float)
        q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        masks = [
            ("Low", values <= q1),
            ("Middle", (values > q1) & (values <= q2)),
            ("High", values > q2),
        ]
        for level, mask in masks:
            local = df.loc[mask].copy()
            if local.empty:
                continue
            pop = float(local["population_check_loss"].mean())
            update = float(local["updated_check_loss"].mean())
            rows.append(
                {
                    "density_measure": label,
                    "stratum": level,
                    "stays": int(local.shape[0]),
                    "median_index_obs": float(local["index_obs"].median()),
                    "median_late_obs": float(local["late_obs"].median()),
                    "any_later_map_below65": float(local["any_later_map_below65"].mean()),
                    "population_assessment_loss": pop,
                    "updated_assessment_loss": update,
                    "loss_reduction_percent": float(100.0 * (pop - update) / pop),
                    "updated_below_fraction_observation": float(
                        local["updated_below_count"].sum() / local["late_obs"].sum()
                    ),
                }
            )

    add_rows("index_obs", "Index observations")
    add_rows("late_obs", "Later observations")
    return rows


def write_calibration_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Empirical later-window quantile calibration of population and updated 0.10 quantile predictions.}",
        "\\label{tab:quantile_calibration}",
        "\\centering",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Predicted updated q10 stratum & Stays & Later obs. & Updated q10 & Pop. below & Updated below & Later MAP $<$65\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['stratum']} & {int(row['stays'])} & {int(row['later_observations'])} & "
            f"{float(row['updated_q10_median']):.1f} & "
            f"{100.0 * float(row['population_below_fraction_observation']):.1f}\\% & "
            f"{100.0 * float(row['updated_below_fraction_observation']):.1f}\\% & "
            f"{100.0 * float(row['later_map_below65_fraction']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Below fractions are observation-level proportions of later MAP measurements below the fitted 0.10 quantile; the nominal target is 10\\%. Strata are quintiles of the updated predicted 0.10 quantile in the assessment split, with lower updated q10 indicating greater lower-tail vulnerability.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibrated_comparator_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    adm_calib = next(row for row in rows if row["model"] == "Calibrated admission window q10")
    upd_calib = next(row for row in rows if row["model"] == "Calibrated primary level update")
    hybrid_intercept = next(row for row in rows if row["model"] == "Hybrid q10 + level offset")
    hybrid_low_burden = next(row for row in rows if row["model"] == "q10 + index MAP<65 burden")

    def fmt_delta(value: float) -> str:
        return "0.0000" if abs(float(value)) < 0.00005 else f"{float(value):.4f}"

    def paired_ci(row: Dict[str, object]) -> str:
        return (
            f"{fmt_delta(float(row['paired_diff_vs_calibrated_q10']))} "
            f"(descriptive interval {fmt_delta(float(row['paired_diff_vs_calibrated_q10_ci_lower']))} "
            f"to {fmt_delta(float(row['paired_diff_vs_calibrated_q10_ci_upper']))})"
        )

    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Assessment check loss for the primary level update and tuning-calibrated admission-window q10 comparators.}",
        "\\label{tab:calibrated_comparator}",
        "\\centering",
        "\\begin{tabular}{llrr}",
        "\\hline",
        "Model & Index window information used & Assessment loss & SE\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['index_information']} & "
            f"{float(row['assessment_loss']):.4f} & {float(row['assessment_loss_se']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Calibrated rows fit an affine transformation on tuning stays by stay-averaged later check loss, then evaluate that fixed transformation on assessment stays. For admission-window q10, the fitted transformation was "
            f"$\\widehat q= {float(adm_calib['calibration_intercept']):.2f} + {float(adm_calib['calibration_slope']):.3f} q_{{0.10,\\mathrm{{index}}}}$; "
            "for the updated q10 it was "
            f"$\\widehat q= {float(upd_calib['calibration_intercept']):.2f} + {float(upd_calib['calibration_slope']):.3f} \\widehat q_{{0.10,\\mathrm{{updated}}}}$.",
            "Hybrid rows are tuning-calibrated multivariable affine rules using only index-window summaries and the profiled stay-specific level offset.",
            f"Paired mean loss differences versus calibrated admission-window q10 were {paired_ci(hybrid_intercept)} for q10 plus the level offset and {paired_ci(hybrid_low_burden)} for q10 plus index MAP below 65 burden.",
            "}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_density_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Measurement-density sensitivity of the primary stay-specific level update in the assessment split.}",
        "\\label{tab:measurement_density_sensitivity}",
        "\\centering",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Density measure & Stratum & Stays & Index obs. & Later obs. & Update reduction & Updated below\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['density_measure']} & {row['stratum']} & {int(row['stays'])} & "
            f"{float(row['median_index_obs']):.0f} & {float(row['median_late_obs']):.0f} & "
            f"{float(row['loss_reduction_percent']):.1f}\\% & "
            f"{100.0 * float(row['updated_below_fraction_observation']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Index and later observation strata are empirical thirds of the corresponding observation count in assessment stays. Updated below is the observation-level fraction of later MAP values below the updated fitted 0.10 quantile; the nominal target is 10\\%.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate additional hard-result diagnostics for the split-window MAP analysis.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--index-hours", type=float, default=12.0)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    beta = np.asarray(results["coefficients"]["baseline_tau_0.10"], dtype=float)
    lambda_b = float(results["settings"]["selected_lambda_baseline_update"])

    tuning_design, assessment_design, metadata = load_split_designs(args)
    tuning_df = stay_level_prediction_frame(tuning_design, beta, args.tau, lambda_b, args.index_hours)
    assessment_df = stay_level_prediction_frame(assessment_design, beta, args.tau, lambda_b, args.index_hours)
    for df in (tuning_df, assessment_df):
        df["log_index_obs"] = np.log(df["index_obs"].to_numpy(dtype=float))
    assessment_df = add_raw_admission_q10_loss(assessment_df, assessment_design, args.tau)
    admission_q10_calibration = fit_calibrated_predictor(
        tuning_df,
        tuning_design,
        args.tau,
        predictor_column="admission_window_q10",
    )
    updated_q10_calibration = fit_calibrated_predictor(
        tuning_df,
        tuning_design,
        args.tau,
        predictor_column="updated_q10_median",
    )
    hybrid_specs = [
        (
            "Hybrid q10 + level offset",
            "q10 and profiled level offset",
            ["admission_window_q10", "updated_offset"],
            "calibrated_hybrid_q10_offset",
        ),
        (
            "Hybrid q10 + level offset + count",
            "q10, profiled level offset, log index count",
            ["admission_window_q10", "updated_offset", "log_index_obs"],
            "calibrated_hybrid_q10_offset_count",
        ),
        (
            "q10 + index MAP<65 burden",
            "q10 and index MAP<65 fraction",
            ["admission_window_q10", "index_map_below65_fraction"],
            "calibrated_q10_lowburden",
        ),
    ]
    hybrid_calibrations: List[Tuple[str, str, Dict[str, object], str]] = []
    for model, info, columns, prefix in hybrid_specs:
        calibration = fit_calibrated_model(tuning_df, tuning_design, args.tau, columns)
        hybrid_calibrations.append((model, info, calibration, prefix))
    assessment_df = add_calibrated_loss_from_design(
        assessment_df,
        assessment_design,
        intercept=float(admission_q10_calibration["intercept"]),
        slope=float(admission_q10_calibration["slope"]),
        tau=args.tau,
        predictor_column="admission_window_q10",
        output_prefix="calibrated_admission_q10",
    )
    assessment_df = add_calibrated_loss_from_design(
        assessment_df,
        assessment_design,
        intercept=float(updated_q10_calibration["intercept"]),
        slope=float(updated_q10_calibration["slope"]),
        tau=args.tau,
        predictor_column="updated_q10_median",
        output_prefix="calibrated_updated_q10",
    )
    for _, _, calibration, prefix in hybrid_calibrations:
        assessment_df = add_calibrated_model_loss_from_design(
            assessment_df,
            assessment_design,
            calibration=calibration,
            tau=args.tau,
            output_prefix=prefix,
        )

    calibration_rows = quantile_calibration_rows(assessment_df, args.tau)
    comparator_table_rows = comparator_rows(
        assessment_df,
        admission_q10_calibration,
        updated_q10_calibration,
        hybrid_calibrations,
    )
    calibration_curve, calibration_error = calibration_curve_rows(assessment_df, args.tau, groups=10)
    density_rows = density_sensitivity_rows(assessment_df)

    public_export_frame(assessment_df).to_csv(args.artifact_dir / "hard_result_stay_level_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.artifact_dir / "quantile_calibration.csv", index=False)
    pd.DataFrame(comparator_table_rows).to_csv(args.artifact_dir / "calibrated_comparator.csv", index=False)
    calibration_curve.to_csv(args.artifact_dir / "quantile_calibration_curve.csv", index=False)
    calibration_error.to_csv(args.artifact_dir / "quantile_calibration_error.csv", index=False)
    pd.DataFrame(density_rows).to_csv(args.artifact_dir / "measurement_density_sensitivity.csv", index=False)

    write_calibration_tex(calibration_rows, args.artifact_dir / "quantile_calibration.tex")
    write_calibrated_comparator_tex(comparator_table_rows, args.artifact_dir / "calibrated_comparator.tex")
    write_calibration_error_tex(calibration_error, args.artifact_dir / "quantile_calibration_error.tex")
    plot_calibration_curve(calibration_curve, args.artifact_dir / "quantile_calibration_plot")
    write_density_tex(density_rows, args.artifact_dir / "measurement_density_sensitivity.tex")

    payload = {
        "status": "complete",
        "tau": float(args.tau),
        "index_hours": float(args.index_hours),
        "lambda_b": lambda_b,
        "metadata": metadata,
        "calibrated_admission_q10": admission_q10_calibration,
        "calibrated_updated_q10": updated_q10_calibration,
        "hybrid_calibrations": [
            {"model": model, "index_information": info, "prefix": prefix, **calibration}
            for model, info, calibration, prefix in hybrid_calibrations
        ],
        "quantile_calibration": calibration_rows,
        "quantile_calibration_error": calibration_error.to_dict("records"),
        "calibrated_comparator": comparator_table_rows,
        "measurement_density_sensitivity": density_rows,
        "artifacts": {
            "stay_level_predictions_csv": str(args.artifact_dir / "hard_result_stay_level_predictions.csv"),
            "quantile_calibration_tex": str(args.artifact_dir / "quantile_calibration.tex"),
            "quantile_calibration_error_tex": str(args.artifact_dir / "quantile_calibration_error.tex"),
            "quantile_calibration_plot": str(args.artifact_dir / "quantile_calibration_plot.pdf"),
            "calibrated_comparator_tex": str(args.artifact_dir / "calibrated_comparator.tex"),
            "measurement_density_sensitivity_tex": str(args.artifact_dir / "measurement_density_sensitivity.tex"),
        },
    }
    (args.artifact_dir / "split_window_hard_results.json").write_text(
        json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
