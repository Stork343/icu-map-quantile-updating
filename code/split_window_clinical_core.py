import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

import split_window_data as data_utils
from split_window_analysis_core import (
    auc_score,
    check_loss,
    design_frame,
    empirical_check_quantile,
    fit_common_quantile,
    profiled_intercept,
    split_indices_for_hours,
    tune_lambda,
    validation_losses,
)
from split_window_data import _safe_read_frame, build_dataset_from_cache


def validation_losses_generic(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    update: str,
    lambda_b: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    stay_index = design["stay_index"].to_numpy(dtype=np.int64)
    x_all = design.loc[:, predictor_cols].to_numpy(dtype=float)
    fitted_all = x_all @ gamma
    starts = np.r_[0, np.flatnonzero(np.diff(stay_index)) + 1]
    ends = np.r_[starts[1:], stay_index.size]
    losses: List[float] = []
    offsets: List[float] = []
    for start, end in zip(starts, ends):
        y = y_all[start:end]
        t = t_all[start:end]
        fitted = fitted_all[start:end]
        index_idx, late_idx = split_indices_for_hours(t, index_hours)
        if update == "none":
            late_resid = y[late_idx] - fitted[late_idx]
            b_hat = 0.0
        elif update == "profiled":
            residual = y - fitted
            b_hat = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
            late_resid = residual[late_idx] - b_hat
        elif update == "admission_window_q10":
            b_hat = empirical_check_quantile(y[index_idx], tau)
            late_resid = y[late_idx] - b_hat
        elif update == "index_mean":
            b_hat = float(np.mean(y[index_idx]))
            late_resid = y[late_idx] - b_hat
        elif update == "index_resid_mean":
            residual = y - fitted
            b_hat = float(np.mean(residual[index_idx]))
            late_resid = residual[late_idx] - b_hat
        else:
            raise ValueError(f"Unknown update: {update}")
        losses.append(float(np.mean(check_loss(late_resid, tau))))
        offsets.append(float(b_hat))
    return np.asarray(losses, dtype=float), np.asarray(offsets, dtype=float)


def tune_lambda_generic(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    rows = []
    best = None
    for lam in lambda_grid:
        losses, _ = validation_losses_generic(
            design,
            predictor_cols,
            gamma,
            tau=tau,
            index_hours=index_hours,
            update="profiled",
            lambda_b=float(lam),
        )
        row = {
            "lambda_b": float(lam),
            "updated_validation_loss": float(np.mean(losses)),
            "updated_validation_loss_se": float(np.std(losses, ddof=1) / np.sqrt(losses.size)),
            "n_stays": int(losses.size),
        }
        rows.append(row)
        if best is None or row["updated_validation_loss"] < best["updated_validation_loss"]:
            best = row
    assert best is not None
    return {"best": best, "grid": rows}


def stay_level_features(
    dataset: Dict[str, object],
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_b: float,
) -> pd.DataFrame:
    enriched = data_utils.ensure_cluster_lists(dataset)
    cluster_ids = np.asarray(enriched["cluster_ids"], dtype=np.int64)
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    stay_index_all = design["stay_index"].to_numpy(dtype=np.int64)
    x_all = design.loc[:, predictor_cols].to_numpy(dtype=float)
    fitted_all = x_all @ gamma
    starts = np.r_[0, np.flatnonzero(np.diff(stay_index_all)) + 1]
    ends = np.r_[starts[1:], stay_index_all.size]
    rows: List[Dict[str, object]] = []
    for start, end in zip(starts, ends):
        stay_index = int(stay_index_all[start])
        y = y_all[start:end]
        t = t_all[start:end]
        index_idx, late_idx = split_indices_for_hours(t, index_hours)
        fitted = fitted_all[start:end]
        residual = y - fitted
        offset = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
        y_index = y[index_idx]
        y_late = y[late_idx]
        rows.append(
            {
                "stay_index": stay_index,
                "stay_id": int(cluster_ids[stay_index]),
                "admission_window_q10": empirical_check_quantile(y_index, tau),
                "index_mean": float(np.mean(y_index)),
                "admission_window_map_below65_fraction": float(np.mean(y_index < 65.0)),
                "later_q10": empirical_check_quantile(y_late, tau),
                "later_map_below65_fraction": float(np.mean(y_late < 65.0)),
                "any_later_map_below65": float(np.any(y_late < 65.0)),
                "updated_offset": float(offset),
                "updated_vulnerability_score": float(-offset),
            }
        )
    return pd.DataFrame(rows)


def outcome_frame(data_root: Path, stay_ids: Sequence[int]) -> pd.DataFrame:
    stay_set = set(int(x) for x in stay_ids)
    icu_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]
    icu = pd.read_csv(data_root / "icu" / "icustays.csv.gz", compression="gzip", usecols=icu_cols)
    icu = icu[icu["stay_id"].isin(stay_set)].copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    adm_cols = ["subject_id", "hadm_id", "deathtime", "hospital_expire_flag"]
    adm = pd.read_csv(data_root / "hosp" / "admissions.csv.gz", compression="gzip", usecols=adm_cols)
    adm["deathtime"] = pd.to_datetime(adm["deathtime"], errors="coerce")

    out = icu.merge(adm, on=["subject_id", "hadm_id"], how="left")
    out["hospital_mortality"] = out["hospital_expire_flag"].astype(float)
    out["icu_mortality"] = (
        out["deathtime"].notna()
        & out["intime"].notna()
        & out["outtime"].notna()
        & (out["deathtime"] >= out["intime"])
        & (out["deathtime"] <= out["outtime"])
    ).astype(float)
    out["icu_los_days"] = pd.to_numeric(out["los"], errors="coerce")
    out["icu_los_ge3_days"] = (out["icu_los_days"] >= 3.0).astype(float)
    out["icu_los_after24h_days"] = np.maximum(out["icu_los_days"] - 1.0, 0.0)
    return out[
        [
            "stay_id",
            "hadm_id",
            "hospital_mortality",
            "icu_mortality",
            "icu_los_days",
            "icu_los_after24h_days",
            "icu_los_ge3_days",
        ]
    ]


def logistic_or_per_10(y: np.ndarray, score: np.ndarray, covariates: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    score10 = np.asarray(score, dtype=float) / 10.0
    x = np.column_stack([np.ones(y.size), score10, covariates])
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(50):
        p = expit(x @ beta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        z = x @ beta + (y - p) / w
        xtw = x.T * w
        h = xtw @ x
        g = xtw @ z
        try:
            beta_new = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(h + 1e-6 * np.eye(h.shape[0]), g, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    p = expit(x @ beta)
    w = np.clip(p * (1.0 - p), 1e-8, None)
    h = (x.T * w) @ x
    cov = np.linalg.pinv(h)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    log_or = float(beta[1])
    return {
        "odds_ratio_per_10": float(np.exp(log_or)),
        "ci_low": float(np.exp(log_or - 1.96 * se)),
        "ci_high": float(np.exp(log_or + 1.96 * se)),
        "log_or": log_or,
        "se": se,
    }


def risk_strata_table(features: pd.DataFrame) -> List[Dict[str, object]]:
    df = features.copy()
    df["admission_window_q10_vulnerability_score"] = -df["admission_window_q10"]
    cuts = np.quantile(df["admission_window_q10_vulnerability_score"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    groups = np.searchsorted(cuts, df["admission_window_q10_vulnerability_score"].to_numpy(), side="right")
    labels = ["Lowest vulnerability", "Second", "Third", "Fourth", "Highest vulnerability"]
    rows: List[Dict[str, object]] = []
    for level in range(5):
        local = df.iloc[np.where(groups == level)[0]]
        rows.append(
            {
                "stratum": labels[level],
                "stays": int(local.shape[0]),
                "updated_offset_median": float(local["updated_offset"].median()),
                "admission_window_q10_median": float(local["admission_window_q10"].median()),
                "later_map_below65_fraction": float(local["later_map_below65_fraction"].mean()),
                "any_later_map_below65": float(local["any_later_map_below65"].mean()),
                "hospital_mortality": float(local["hospital_mortality"].mean()),
                "icu_mortality": float(local["icu_mortality"].mean()),
                "icu_los_median": float(local["icu_los_days"].median()),
                "icu_los_iqr_low": float(local["icu_los_days"].quantile(0.25)),
                "icu_los_iqr_high": float(local["icu_los_days"].quantile(0.75)),
                "icu_los_ge3_days": float(local["icu_los_ge3_days"].mean()),
            }
        )
    return rows


def write_comparator_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Split window validation across comparator models.}",
        "\\label{tab:model_comparison}",
        "\\centering",
        "\\begin{tabular}{llrr}",
        "\\hline",
        "Model & Index window information used & Validation loss & SE\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['index_information']} & "
            f"{float(row['validation_loss']):.4f} & {float(row['validation_loss_se']):.4f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Validation loss is the mean stay level ordinary check loss on later observations at $\\tau=0.10$. The selected model uses baseline covariates plus a ridge penalized stay specific offset estimated from index observations; the spline rows show candidate common time trajectory extensions.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outcome_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Later hypotension and clinical outcomes by admission window lower tail vulnerability quintile.}",
        "\\label{tab:clinical_outcome_strata}",
        "\\centering",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Vulnerability stratum & Stays & Admission window q10 & Later MAP $<$65 & Hosp. mortality & ICU LOS & ICU LOS $\\ge$3d\\\\",
        "\\hline",
    ]
    for row in rows:
        los = f"{float(row['icu_los_median']):.1f}"
        lines.append(
            f"{row['stratum']} & {int(row['stays'])} & {float(row['admission_window_q10_median']):.1f} & "
            f"{100.0 * float(row['later_map_below65_fraction']):.1f}\\% & "
            f"{100.0 * float(row['hospital_mortality']):.1f}\\% & "
            f"{los} & {100.0 * float(row['icu_los_ge3_days']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Strata are quintiles of $-$admission window MAP 0.10 quantile, so higher strata correspond to lower admission window lower tail MAP. Boundary ties are assigned to the next higher-vulnerability stratum. Later MAP $<$65 is the mean within stay fraction of later recorded MAP values below 65 mmHg. ICU LOS is the median ICU length of stay in days.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selected_sensitivity_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Sensitivity of the selected admission window updated mixed effects offset model.}",
        "\\label{tab:sensitivity_analysis}",
        "\\centering",
        "\\begin{tabular}{rrrrrr}",
        "\\hline",
        "$\\tau$ & Index window (h) & $\\lambda_b$ & Pop. loss & Updated loss & Reduction\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{float(row['tau']):.2f} & {float(row['index_hours']):.0f} & {float(row['lambda_b']):.2g} & "
            f"{float(row['population_validation_loss']):.4f} & {float(row['updated_validation_loss']):.4f} & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Population loss is the corresponding baseline-covariate quantile validation loss. Updated loss adds the admission window profiled mixed effects offset. The primary analysis selected $\\lambda_b=0.10$, which is used across these sensitivity checks.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_validation_sensitivity(rows: Sequence[Dict[str, object]], output_stem: Path) -> None:
    df = pd.DataFrame(rows)
    labels = []
    pop = []
    update = []
    reductions = []
    for row in rows:
        if abs(float(row["tau"]) - 0.10) < 1e-12:
            label = f"{int(float(row['index_hours']))}h"
        else:
            label = rf"$\tau={float(row['tau']):.2f}$"
        labels.append(label)
        pop.append(float(row["population_validation_loss"]))
        update.append(float(row["updated_validation_loss"]))
        reductions.append(float(row["loss_reduction_percent"]))
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, pop, width, label="Population", color="0.72")
    ax.bar(x + width / 2, update, width, label="Updated mixed effects", color="#1f77b4")
    for xi, yi, red in zip(x, update, reductions):
        ax.text(xi + width / 2, yi + 0.03, f"{red:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Validation check loss")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clinical_vulnerability(rows: Sequence[Dict[str, object]], output_stem: Path) -> None:
    labels = ["Lowest", "Second", "Third", "Fourth", "Highest"]
    later_low = [100.0 * float(r["later_map_below65_fraction"]) for r in rows]
    mort = [100.0 * float(r["hospital_mortality"]) for r in rows]
    los3 = [100.0 * float(r["icu_los_ge3_days"]) for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
    panels = [
        (later_low, "Later MAP <65 (%)", "#1f77b4"),
        (mort, "Hospital mortality (%)", "#d62728"),
        (los3, "ICU LOS >=3 days (%)", "#2ca02c"),
    ]
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.bar(x, values, color=color, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparator and clinical outcome analyses for the ordinary penalized stay-level MAP update.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("statistics-in-medicine-paper/data/mimic-iv-3.1"))
    parser.add_argument("--results-json", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_results.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("statistics-in-medicine-paper/manuscript/WileyNJDv5_Template/tables"))
    parser.add_argument("--output", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_clinical_comparator_results.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_comparator_work"))
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    tau = 0.10
    index_hours = 12.0
    lambda_grid = [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, _, _ = build_dataset_from_cache(obs, stays, fit_stays=args.fit_stays, seed=args.seed, analysis_hours=24.0)
    design, full_predictors = design_frame(dataset)

    existing = json.loads(args.results_json.read_text())
    full_gamma = np.asarray(existing["coefficients"]["tau_0.10"], dtype=float)
    x_predictors = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
    r_script = Path(__file__).with_name("fit_quantile_common.R")
    baseline_fits: Dict[float, np.ndarray] = {}
    for tau_fit in [0.05, 0.10, 0.20]:
        baseline_fits[tau_fit] = fit_common_quantile(
            design,
            x_predictors,
            tau=tau_fit,
            work_dir=args.work_dir / f"baseline_tau_{tau_fit:.2f}",
            r_script=r_script,
            force_refit=False,
        )
    baseline_gamma = baseline_fits[0.10]

    base_tuned = tune_lambda_generic(
        design,
        x_predictors,
        baseline_gamma,
        tau=tau,
        index_hours=index_hours,
        lambda_grid=lambda_grid,
    )
    full_tuned = tune_lambda(dataset, full_gamma, tau=tau, index_hours=index_hours, lambda_grid=lambda_grid)
    admission_window_q10_losses, _ = validation_losses_generic(design, full_predictors, full_gamma, tau, index_hours, "admission_window_q10")
    index_mean_losses, _ = validation_losses_generic(design, full_predictors, full_gamma, tau, index_hours, "index_mean")
    full_unpen_losses, _ = validation_losses_generic(
        design, full_predictors, full_gamma, tau, index_hours, "profiled", lambda_b=0.0
    )
    sensitivity_rows: List[Dict[str, object]] = []
    sensitivity_lambda = float(base_tuned["best"]["lambda_b"])
    for tau_fit in [0.05, 0.10, 0.20]:
        windows = [6.0, 12.0, 18.0] if abs(tau_fit - 0.10) < 1e-12 else [12.0]
        for window in windows:
            gamma_fit = baseline_fits[tau_fit]
            pop_losses, _ = validation_losses_generic(design, x_predictors, gamma_fit, tau_fit, window, "none")
            update_losses, _ = validation_losses_generic(
                design,
                x_predictors,
                gamma_fit,
                tau_fit,
                window,
                "profiled",
                lambda_b=sensitivity_lambda,
            )
            pop_loss = float(np.mean(pop_losses))
            updated_loss = float(np.mean(update_losses))
            sensitivity_rows.append(
                {
                    "tau": float(tau_fit),
                    "index_hours": float(window),
                    "lambda_b": sensitivity_lambda,
                    "population_validation_loss": pop_loss,
                    "population_validation_loss_se": float(np.std(pop_losses, ddof=1) / np.sqrt(pop_losses.size)),
                    "updated_validation_loss": updated_loss,
                    "updated_validation_loss_se": float(np.std(update_losses, ddof=1) / np.sqrt(update_losses.size)),
                    "n_stays": int(update_losses.size),
                    "loss_reduction_percent": float(100.0 * (pop_loss - updated_loss) / pop_loss),
                }
            )

    def loss_row(model: str, index_information: str, losses: np.ndarray) -> Dict[str, object]:
        return {
            "model": model,
            "index_information": index_information,
            "validation_loss": float(np.mean(losses)),
            "validation_loss_se": float(np.std(losses, ddof=1) / np.sqrt(losses.size)),
            "n_stays": int(losses.size),
        }

    full_pop_losses, full_update_losses = validation_losses(dataset, full_gamma, tau, index_hours, float(full_tuned["best"]["lambda_b"]))
    base_pop_losses, _ = validation_losses_generic(design, x_predictors, baseline_gamma, tau, index_hours, "none")
    base_update_losses, _ = validation_losses_generic(
        design,
        x_predictors,
        baseline_gamma,
        tau,
        index_hours,
        "profiled",
        lambda_b=float(base_tuned["best"]["lambda_b"]),
    )
    comparator_rows = [
        loss_row("Baseline covariates quantile", "none", base_pop_losses),
        loss_row("Admission window q10 carry forward", "raw admission window q10", admission_window_q10_losses),
        loss_row("Population spline quantile", "none", full_pop_losses),
        loss_row("Spline + unpenalized update", "index residual q10", full_unpen_losses),
        loss_row("Spline + penalized random-intercept update", "profiled intercept", full_update_losses),
        loss_row("Selected baseline-adjusted random-intercept update", "profiled intercept", base_update_losses),
    ]

    features = stay_level_features(
        dataset,
        design,
        x_predictors,
        baseline_gamma,
        tau=tau,
        index_hours=index_hours,
        lambda_b=float(base_tuned["best"]["lambda_b"]),
    )
    outcomes = outcome_frame(args.data_root, features["stay_id"].to_numpy(dtype=np.int64))
    features = features.merge(outcomes, on="stay_id", how="left")
    features = features.dropna(subset=["hospital_mortality", "icu_los_days"])
    strata_rows = risk_strata_table(features)

    cov = design.groupby("stay_index", sort=True)[["age_z", "male", "emergency_or_urgent"]].first().reset_index(drop=True)
    features = features.sort_values("stay_index").reset_index(drop=True)
    cov = cov.iloc[features["stay_index"].to_numpy(dtype=int)].to_numpy(dtype=float)
    y_mort = features["hospital_mortality"].to_numpy(dtype=float)
    outcome_metrics = {
        "hospital_mortality_rate": float(np.mean(y_mort)),
        "icu_mortality_rate": float(np.mean(features["icu_mortality"].to_numpy(dtype=float))),
        "icu_los_median": float(np.median(features["icu_los_days"].to_numpy(dtype=float))),
        "hospital_mortality_auc_updated_vulnerability": auc_score(y_mort, features["updated_vulnerability_score"].to_numpy(dtype=float)),
        "hospital_mortality_auc_admission_window_q10_low_is_risk": auc_score(y_mort, -features["admission_window_q10"].to_numpy(dtype=float)),
        "hospital_mortality_auc_later_low_fraction": auc_score(
            y_mort, features["later_map_below65_fraction"].to_numpy(dtype=float)
        ),
        "adjusted_or_hospital_mortality_per_10mmhg_lower_admission_window_q10": logistic_or_per_10(
            y_mort, -features["admission_window_q10"].to_numpy(dtype=float), cov
        ),
        "adjusted_or_hospital_mortality_per_10mmhg_higher_updated_vulnerability": logistic_or_per_10(
            y_mort, features["updated_vulnerability_score"].to_numpy(dtype=float), cov
        ),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparator_rows).to_csv(args.artifact_dir / "split_window_mixed_effects_comparator_models.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.artifact_dir / "split_window_mixed_effects_sensitivity.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(args.artifact_dir / "split_window_mixed_effects_clinical_outcome_strata.csv", index=False)
    features.to_csv(args.artifact_dir / "split_window_mixed_effects_stay_level_clinical_features.csv", index=False)
    write_comparator_tex(comparator_rows, args.artifact_dir / "split_window_mixed_effects_model_comparison.tex")
    write_selected_sensitivity_tex(sensitivity_rows, args.artifact_dir / "split_window_mixed_effects_sensitivity.tex")
    write_outcome_tex(strata_rows, args.artifact_dir / "split_window_mixed_effects_clinical_outcome_strata.tex")
    plot_validation_sensitivity(sensitivity_rows, args.artifact_dir / "split_window_mixed_effects_validation_sensitivity_plot")
    plot_clinical_vulnerability(strata_rows, args.artifact_dir / "split_window_mixed_effects_clinical_vulnerability_plot")

    payload = {
        "analysis_type": "split_window_mixed_effects_clinical_comparator_analysis",
        "status": "complete",
        "settings": {
            "tau": tau,
            "index_hours": index_hours,
            "lambda_grid": lambda_grid,
            "selected_lambda_spline_update": float(full_tuned["best"]["lambda_b"]),
            "selected_lambda_baseline_update": float(base_tuned["best"]["lambda_b"]),
        },
        "comparator_rows": comparator_rows,
        "sensitivity_rows": sensitivity_rows,
        "clinical_outcome_strata": strata_rows,
        "outcome_metrics": outcome_metrics,
        "artifacts": {
            "model_comparison_tex": str(args.artifact_dir / "split_window_mixed_effects_model_comparison.tex"),
            "clinical_outcome_strata_tex": str(args.artifact_dir / "split_window_mixed_effects_clinical_outcome_strata.tex"),
            "clinical_vulnerability_plot": str(args.artifact_dir / "split_window_mixed_effects_clinical_vulnerability_plot.pdf"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
