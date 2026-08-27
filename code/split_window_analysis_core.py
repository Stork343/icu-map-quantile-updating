import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import split_window_data as data_utils
from split_window_data import _safe_read_frame, build_dataset_from_cache


def check_loss(residual: np.ndarray, tau: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    return residual * (tau - (residual < 0.0).astype(float))


def empirical_check_quantile(values: np.ndarray, tau: float) -> float:
    """Return the lower-endpoint empirical quantile that minimizes check loss.

    For sorted values x_(1) <= ... <= x_(n), this is x_(ceil(n * tau)).
    When n * tau is an integer, the finite-sample check-loss minimizer is an
    interval and this convention deterministically selects its lower endpoint.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("empirical_check_quantile requires at least one value")
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must be strictly between 0 and 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("empirical_check_quantile requires finite values")
    ordered = np.sort(values)
    one_based_rank = int(np.ceil(float(tau) * ordered.size))
    one_based_rank = int(np.clip(one_based_rank, 1, ordered.size))
    return float(ordered[one_based_rank - 1])


def design_frame(dataset: Dict[str, object]) -> Tuple[pd.DataFrame, List[str]]:
    enriched = data_utils.ensure_cluster_lists(dataset)
    y = np.asarray(enriched["y"], dtype=float)
    x_long = np.asarray(enriched["X_long"], dtype=float)
    b_long = np.asarray(enriched["B_fit_long"], dtype=float)
    cluster_ids = np.asarray(enriched["cluster_ids"], dtype=np.int64)
    t_list = [np.asarray(t, dtype=float) for t in enriched["t_list"]]
    n_list = [len(t) for t in t_list]
    stay_index = np.repeat(np.arange(len(n_list), dtype=np.int64), n_list)
    stay_id = np.repeat(cluster_ids, n_list)
    time_hours = np.concatenate(t_list)
    index_flag = time_hours <= 12.0
    late_flag = ~index_flag

    data = {
        "y": y,
        "stay_index": stay_index,
        "stay_id": stay_id,
        "time_hours": time_hours,
        "index_flag": index_flag.astype(int),
        "late_flag": late_flag.astype(int),
    }
    x_names = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
    b_names = [f"basis_{j + 1}" for j in range(b_long.shape[1])]
    for j, name in enumerate(x_names):
        data[name] = x_long[:, j]
    for j, name in enumerate(b_names):
        data[name] = b_long[:, j]
    return pd.DataFrame(data), x_names + b_names


def fit_common_quantile(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    tau: float,
    work_dir: Path,
    r_script: Path,
    force_refit: bool = False,
    observation_weights: np.ndarray | None = None,
) -> np.ndarray:
    work_dir.mkdir(parents=True, exist_ok=True)
    design_csv = work_dir / "ordinary_update_design.csv"
    coef_csv = work_dir / f"ordinary_common_tau{tau:.2f}.csv"
    cols = ["y", "stay_index", "stay_id", "time_hours", "index_flag", "late_flag", *predictor_cols]
    fit_design = design.loc[:, cols].copy()
    if observation_weights is not None:
        weights = np.asarray(observation_weights, dtype=float).reshape(-1)
        if weights.size != fit_design.shape[0]:
            raise ValueError("observation_weights must match the number of design rows")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("observation_weights must be finite and strictly positive")
        fit_design["fit_weight"] = weights
    fit_design.to_csv(design_csv, index=False)
    subprocess.run(
        ["Rscript", str(r_script), str(design_csv), f"{tau:.12g}", str(coef_csv)],
        check=True,
    )
    coef_df = pd.read_csv(coef_csv)
    coef_map = dict(zip(coef_df["term"], coef_df["estimate"]))
    return np.asarray([coef_map[name] for name in predictor_cols], dtype=float)


def equal_stay_observation_weights(design: pd.DataFrame) -> np.ndarray:
    """Give every stay total fitting weight one."""
    if "stay_index" not in design.columns:
        raise ValueError("design must contain stay_index")
    stay_index = design["stay_index"].to_numpy(dtype=np.int64)
    counts = pd.Series(stay_index).value_counts(sort=False)
    weights = 1.0 / pd.Series(stay_index).map(counts).to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("Unable to construct equal-stay observation weights")
    return weights


def split_indices_for_hours(t: np.ndarray, index_hours: float, min_index_obs: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    return data_utils._within_stay_split_indices(
        np.asarray(t, dtype=float),
        index_fraction=0.5,
        index_hours=float(index_hours),
        min_index_obs=int(min_index_obs),
    )


def profiled_intercept(residual_index: np.ndarray, tau: float, lambda_b: float) -> float:
    r = np.asarray(residual_index, dtype=float).reshape(-1)
    if r.size == 0:
        return 0.0
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must be strictly between 0 and 1")
    if not np.all(np.isfinite(r)):
        raise ValueError("profiled_intercept requires finite residuals")
    if lambda_b < 0.0:
        raise ValueError("lambda_b must be nonnegative")
    if lambda_b == 0.0:
        return empirical_check_quantile(r, tau)
    lam = float(lambda_b)
    n = float(r.size)
    sorted_r = np.sort(r)

    # First check differentiable intervals between adjacent residual knots.
    for k in range(sorted_r.size + 1):
        candidate = (tau * n - float(k)) / (2.0 * lam)
        lower = sorted_r[k - 1] if k > 0 else -np.inf
        upper = sorted_r[k] if k < sorted_r.size else np.inf
        if lower < candidate < upper:
            return float(candidate)

    # If no interval contains the zero derivative, the unique optimum is a
    # residual knot. Check whether zero lies in that knot's subgradient.
    for knot in np.unique(sorted_r):
        n_less = int(np.searchsorted(sorted_r, knot, side="left"))
        n_less_equal = int(np.searchsorted(sorted_r, knot, side="right"))
        subgradient_left = float(n_less) - tau * n + 2.0 * lam * float(knot)
        subgradient_right = float(n_less_equal) - tau * n + 2.0 * lam * float(knot)
        tolerance = 1e-12 * max(1.0, n, abs(2.0 * lam * float(knot)))
        if subgradient_left <= tolerance and subgradient_right >= -tolerance:
            return float(knot)

    raise RuntimeError("Unable to locate the convex profiled-intercept minimizer")


def validation_losses(
    dataset: Dict[str, object],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    enriched = data_utils.ensure_cluster_lists(dataset)
    x = np.asarray(enriched["X"], dtype=float)
    b_list = [np.asarray(b, dtype=float) for b in enriched["B_fit_list"]]
    y_list = [np.asarray(y, dtype=float) for y in enriched["y_list"]]
    t_list = [np.asarray(t, dtype=float) for t in enriched["t_list"]]
    p = x.shape[1]
    beta = gamma[:p]
    theta = gamma[p:]
    population_losses: List[float] = []
    update_losses: List[float] = []

    for i, (y_i, t_i, b_i) in enumerate(zip(y_list, t_list, b_list)):
        index_idx, late_idx = split_indices_for_hours(t_i, index_hours)
        fitted = float(x[i] @ beta) + b_i @ theta
        residual = y_i - fitted
        pop_late = check_loss(residual[late_idx], tau)
        b_hat = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
        update_late = check_loss(residual[late_idx] - b_hat, tau)
        population_losses.append(float(np.mean(pop_late)))
        update_losses.append(float(np.mean(update_late)))
    return np.asarray(population_losses, dtype=float), np.asarray(update_losses, dtype=float)


def tune_lambda(
    dataset: Dict[str, object],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    rows = []
    best = None
    for lam in lambda_grid:
        pop_losses, update_losses = validation_losses(dataset, gamma, tau, index_hours, lam)
        row = {
            "tau": float(tau),
            "index_hours": float(index_hours),
            "lambda_b": float(lam),
            "population_validation_loss": float(np.mean(pop_losses)),
            "population_validation_loss_se": float(np.std(pop_losses, ddof=1) / np.sqrt(pop_losses.size)),
            "updated_validation_loss": float(np.mean(update_losses)),
            "updated_validation_loss_se": float(np.std(update_losses, ddof=1) / np.sqrt(update_losses.size)),
            "n_stays": int(update_losses.size),
        }
        row["loss_reduction_percent"] = float(
            100.0 * (row["population_validation_loss"] - row["updated_validation_loss"]) / row["population_validation_loss"]
        )
        rows.append(row)
        if best is None or row["updated_validation_loss"] < best["updated_validation_loss"]:
            best = row
    assert best is not None
    return {"best": best, "grid": rows}


def admission_window_q10_strata(dataset: Dict[str, object], index_hours: float = 12.0) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    enriched = data_utils.ensure_cluster_lists(dataset)
    y_list = [np.asarray(y, dtype=float) for y in enriched["y_list"]]
    t_list = [np.asarray(t, dtype=float) for t in enriched["t_list"]]
    x = np.asarray(enriched["X"], dtype=float)
    records = []
    for i, (y_i, t_i) in enumerate(zip(y_list, t_list)):
        index_idx, late_idx = split_indices_for_hours(t_i, index_hours)
        y_index = y_i[index_idx]
        y_late = y_i[late_idx]
        if y_index.size == 0 or y_late.size == 0:
            continue
        admission_window_q10 = empirical_check_quantile(y_index, 0.10)
        late_q10 = empirical_check_quantile(y_late, 0.10)
        index_slope = float(np.polyfit(t_i[index_idx], y_index, deg=1)[0]) if y_index.size >= 2 else 0.0
        later_slope = float(np.polyfit(t_i[late_idx], y_late, deg=1)[0]) if y_late.size >= 2 else 0.0
        records.append(
            {
                "admission_window_q10": admission_window_q10,
                "later_q10": late_q10,
                "admission_window_low_fraction": float(np.mean(y_index < 65.0)),
                "later_low_fraction": float(np.mean(y_late < 65.0)),
                "any_later_low": float(np.any(y_late < 65.0)),
                "index_slope": index_slope,
                "later_slope": later_slope,
                "age_z": float(x[i, 1]),
                "male": float(x[i, 2]),
                "emergency_or_urgent": float(x[i, 3]),
            }
        )
    df = pd.DataFrame(records)
    cuts = np.quantile(df["admission_window_q10"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    strata = np.searchsorted(cuts, df["admission_window_q10"].to_numpy(), side="right")
    labels = ["Lowest quintile", "Second quintile", "Third quintile", "Fourth quintile", "Highest quintile"]
    rows: List[Dict[str, object]] = []
    for level in range(5):
        local = df.iloc[np.where(strata == level)[0]]
        rows.append(
            {
                "stratum": labels[level],
                "stays": int(local.shape[0]),
                "admission_window_q10_median": float(local["admission_window_q10"].median()),
                "later_q10_median": float(local["later_q10"].median()),
                "later_map_below65_fraction": float(local["later_low_fraction"].mean()),
                "any_later_map_below65": float(local["any_later_low"].mean()),
            }
        )
    y_true = df["any_later_low"].to_numpy(dtype=float)
    aucs = {
        "admission_window_q10_low_is_risk": auc_score(y_true, -df["admission_window_q10"].to_numpy(dtype=float)),
        "admission_window_low_fraction": auc_score(y_true, df["admission_window_low_fraction"].to_numpy(dtype=float)),
        "age": auc_score(y_true, df["age_z"].to_numpy(dtype=float)),
        "male": auc_score(y_true, df["male"].to_numpy(dtype=float)),
        "emergency_or_urgent": auc_score(y_true, df["emergency_or_urgent"].to_numpy(dtype=float)),
        "index_later_q10_correlation": float(np.corrcoef(df["admission_window_q10"], df["later_q10"])[0, 1]),
        "index_later_low_fraction_correlation": float(
            np.corrcoef(df["admission_window_low_fraction"], df["later_low_fraction"])[0, 1]
        ),
        "index_later_slope_correlation": float(np.corrcoef(df["index_slope"], df["later_slope"])[0, 1]),
    }
    return rows, aucs


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    pos = y > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(~pos))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    rank_sum_pos = float(np.sum(ranks[pos]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_model_comparison_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Split window validation under the ordinary check loss.}",
        "\\label{tab:model_comparison}",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Model & Stay-specific update & Validation loss & SE & Selected $\\lambda_b$\\\\",
        "\\hline",
    ]
    for row in rows:
        lam = "--" if row.get("lambda_b") is None else f"{float(row['lambda_b']):.2g}"
        lines.append(
            f"{row['model']} & {row['stay_update']} & {float(row['validation_loss']):.4f} & "
            f"{float(row['validation_loss_se']):.4f} & {lam}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Validation loss is the mean stay level ordinary check loss on later observations. For the mixed effects offset model, the stay specific offset is estimated from index window observations only. Lower values indicate better lower tail prediction.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Sensitivity of the admission window updated mixed effects offset model.}",
        "\\label{tab:sensitivity_analysis}",
        "\\centering",
        "\\begin{tabular}{rrrrrr}",
        "\\hline",
        "$\\tau$ & Index window (h) & Selected $\\lambda_b$ & Pop. loss & Updated loss & Reduction\\\\",
        "\\hline",
    ]
    for row in rows:
        pop_loss = float(row["population_validation_loss"])
        updated_loss = float(row["updated_validation_loss"])
        reduction = float(row["loss_reduction_percent"])
        lines.append(
            f"{float(row['tau']):.2f} & {float(row['index_hours']):.0f} & {float(row['lambda_b']):.2g} & "
            f"{pop_loss:.4f} & {updated_loss:.4f} & {reduction:.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Population loss is the corresponding population-only spline quantile validation loss. Updated loss uses the admission window profiled mixed effects offset. Reduction is the percentage decrease in validation loss.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_strata_tex(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Later hypotension burden by quintile of index MAP 0.10 quantile.}",
        "\\label{tab:admission_q10_strata}",
        "\\centering",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Admission window MAP 0.10 quantile & Stays & Admission window q10 & Later q10 & Later MAP $<$65 & Any later MAP $<$65\\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['stratum']} & {int(row['stays'])} & {float(row['admission_window_q10_median']):.1f} & "
            f"{float(row['later_q10_median']):.1f} & {float(row['later_map_below65_fraction']):.3f} & "
            f"{float(row['any_later_map_below65']):.3f}\\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Admission window q10 and later q10 are median stay level MAP 0.10 quantiles in mmHg within each stratum. Later MAP $<$65 is the mean within stay fraction of later recorded MAP values below 65 mmHg.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final ordinary-check-loss penalized stay-level update analysis for MIMIC-IV MAP.")
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--fit-stays", type=int, default=0, help="Maximum stays to sample; 0 includes all eligible stays")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--output", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_results.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("statistics-in-medicine-paper/manuscript/WileyNJDv5_Template/tables"))
    parser.add_argument("--work-dir", type=Path, default=Path("statistics-in-medicine-paper/code/split_window_mixed_effects_work"))
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, basis_spec = build_dataset_from_cache(
        obs,
        stays,
        fit_stays=args.fit_stays,
        seed=args.seed,
        analysis_hours=24.0,
    )
    design, predictors = design_frame(dataset)
    r_script = Path(__file__).with_name("fit_quantile_common.R")
    lambda_grid = [0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]

    tau_values = [0.05, 0.10, 0.20]
    index_windows = [6.0, 12.0, 18.0]
    fits: Dict[str, List[float]] = {}
    sensitivity_rows: List[Dict[str, object]] = []
    lambda_grids: Dict[str, object] = {}

    for tau in tau_values:
        gamma = fit_common_quantile(
            design,
            predictors,
            tau=tau,
            work_dir=args.work_dir,
            r_script=r_script,
            force_refit=args.force_refit,
        )
        fits[f"tau_{tau:.2f}"] = gamma.tolist()
        windows = index_windows if abs(tau - 0.10) < 1e-12 else [12.0]
        for index_hours in windows:
            tuned = tune_lambda(dataset, gamma, tau=tau, index_hours=index_hours, lambda_grid=lambda_grid)
            sensitivity_rows.append(tuned["best"])
            lambda_grids[f"tau_{tau:.2f}_index_{index_hours:.0f}"] = tuned["grid"]

    main_row = next(row for row in sensitivity_rows if abs(row["tau"] - 0.10) < 1e-12 and abs(row["index_hours"] - 12.0) < 1e-12)
    model_rows = [
        {
            "model": "Population spline quantile",
            "stay_update": "none",
            "validation_loss": main_row["population_validation_loss"],
            "validation_loss_se": main_row["population_validation_loss_se"],
            "lambda_b": None,
        },
        {
            "model": "Admission window updated mixed effects offset quantile",
            "stay_update": "intercept",
            "validation_loss": main_row["updated_validation_loss"],
            "validation_loss_se": main_row["updated_validation_loss_se"],
            "lambda_b": main_row["lambda_b"],
        },
    ]
    strata_rows, aucs = admission_window_q10_strata(dataset, index_hours=12.0)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(model_rows, args.artifact_dir / "split_window_mixed_effects_model_comparison.csv")
    write_csv(sensitivity_rows, args.artifact_dir / "split_window_mixed_effects_sensitivity.csv")
    write_csv(strata_rows, args.artifact_dir / "split_window_mixed_effects_admission_window_q10_strata.csv")
    write_model_comparison_tex(model_rows, args.artifact_dir / "split_window_mixed_effects_model_comparison.tex")
    write_sensitivity_tex(sensitivity_rows, args.artifact_dir / "split_window_mixed_effects_sensitivity.tex")
    write_strata_tex(strata_rows, args.artifact_dir / "split_window_mixed_effects_admission_window_q10_strata.tex")

    payload = {
        "analysis_type": "split_window_mixed_effects_analysis",
        "status": "complete",
        "runtime_seconds": time.time() - t0,
        "data_summary": data_summary,
        "basis": {
            "Tmax": basis_spec.Tmax,
            "knots": basis_spec.knots,
            "basis_dimension": basis_spec.L,
            "include_intercept": basis_spec.include_intercept,
            "center_basis": basis_spec.center_basis,
            "scale_basis": basis_spec.scale_basis,
        },
        "settings": {
            "seed": args.seed,
            "fit_stays_requested": args.fit_stays,
            "lambda_grid": lambda_grid,
            "tau_values": tau_values,
            "index_windows_tau_010": index_windows,
            "common_fit": "ordinary quantile regression via quantreg::rq.fit(method='fn')",
            "stay_offset": "ordinary check-loss profiled intercept from index observations",
        },
        "model_comparison_rows": model_rows,
        "sensitivity_rows": sensitivity_rows,
        "lambda_grids": lambda_grids,
        "admission_window_q10_strata": strata_rows,
        "auc": aucs,
        "coefficients": fits,
        "artifacts": {
            "model_comparison_tex": str(args.artifact_dir / "split_window_mixed_effects_model_comparison.tex"),
            "sensitivity_tex": str(args.artifact_dir / "split_window_mixed_effects_sensitivity.tex"),
            "admission_window_q10_strata_tex": str(args.artifact_dir / "split_window_mixed_effects_admission_window_q10_strata.tex"),
            "model_comparison_csv": str(args.artifact_dir / "split_window_mixed_effects_model_comparison.csv"),
            "sensitivity_csv": str(args.artifact_dir / "split_window_mixed_effects_sensitivity.csv"),
            "admission_window_q10_strata_csv": str(args.artifact_dir / "split_window_mixed_effects_admission_window_q10_strata.csv"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
