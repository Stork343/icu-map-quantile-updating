#!/usr/bin/env python3
"""Re-fit the prespecified landmark sensitivity rows on horizon-specific queues.

The base cache contains all stays with at least eight retained MAP observations.
For each requested landmark, the fixed-clock rule is applied first and the
existing stay-level split is then intersected with that eligible queue.  The
population component, penalty, and assessment are consequently fit on the
corresponding horizon-specific groups.  No later outcome is used to construct
the queue or the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import split_window_data as data_utils
from run_split_window_mixed_effects_analysis import (
    apply_training_age_standardization,
    se_mean,
    split_cluster_indices,
    tune_lambda_generic,
)
from split_window_analysis_core import check_loss, design_frame, fit_common_quantile, profiled_intercept
from split_window_data import _safe_read_frame, build_dataset_from_cache


X_PREDICTORS = ["x_intercept", "age_z", "male", "emergency_or_urgent"]
DEFAULT_LAMBDA_GRID = (0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
DEFAULT_SPECS = ((0.05, 12.0), (0.10, 6.0), (0.10, 12.0), (0.10, 18.0), (0.20, 12.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contiguous_slices(design: pd.DataFrame) -> Iterable[Tuple[int, int]]:
    stay_index = design["stay_index"].to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(stay_index)) + 1]
    ends = np.r_[starts[1:], stay_index.size]
    return zip(starts, ends)


def horizon_eligible_indices(
    dataset: Dict[str, object],
    index_hours: float,
    min_total_obs: int = 8,
    min_index_obs: int = 4,
    min_late_obs: int = 1,
) -> np.ndarray:
    """Return base-dataset indices satisfying one strict-clock horizon."""
    enriched = data_utils.ensure_cluster_lists(dataset)
    qualified: List[int] = []
    for position, (y_i, t_i) in enumerate(zip(enriched["y_list"], enriched["t_list"])):
        if len(y_i) < int(min_total_obs):
            continue
        index_idx, late_idx = data_utils.clock_split_indices(
            np.asarray(t_i, dtype=float),
            index_hours=index_hours,
            min_index_obs=min_index_obs,
            min_late_obs=min_late_obs,
        )
        if index_idx.size and late_idx.size:
            qualified.append(position)
    return np.asarray(qualified, dtype=np.int64)


def preserve_existing_assignments(
    dataset: Dict[str, object],
    seed: int,
    train_fraction: float,
    tuning_fraction: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, int]]:
    """Preserve the historical 12-hour split and assign only new stays once.

    The historical analysis was split after filtering to the 12-hour eligible
    cohort.  A direct permutation of the larger all-stay cache would move
    those existing stays between groups.  We therefore reproduce that split
    for the historical IDs, then apply one independent deterministic
    permutation to the remaining base-cache IDs; every horizon reuses this
    assignment before applying its own eligibility filter.
    """
    enriched = data_utils.ensure_cluster_lists(dataset)
    n_stays = len(enriched["y_list"])
    old_indices = horizon_eligible_indices(dataset, 12.0)
    old_set = set(old_indices.tolist())
    old_positions = split_cluster_indices(
        old_indices.size,
        seed=seed,
        train_fraction=train_fraction,
        tuning_fraction=tuning_fraction,
    )
    assignment = np.full(n_stays, -1, dtype=np.int8)
    for group, positions in enumerate(old_positions):
        assignment[old_indices[positions]] = group
    new_indices = np.asarray([i for i in range(n_stays) if i not in old_set], dtype=np.int64)
    new_positions = split_cluster_indices(
        new_indices.size,
        seed=seed + 1,
        train_fraction=train_fraction,
        tuning_fraction=tuning_fraction,
    )
    for group, positions in enumerate(new_positions):
        assignment[new_indices[positions]] = group
    if np.any(assignment < 0):
        raise RuntimeError("Every base-cache stay must receive one fixed split assignment")
    split = tuple(np.flatnonzero(assignment == group).astype(np.int64) for group in range(3))
    return split, {
        "historical_12h_stays": int(old_indices.size),
        "new_base_stays": int(new_indices.size),
        "base_stays": int(n_stays),
        "historical_split_seed": int(seed),
        "new_split_seed": int(seed + 1),
    }


def paired_assessment_losses(
    design: pd.DataFrame,
    predictor_cols: Sequence[str],
    gamma: np.ndarray,
    tau: float,
    index_hours: float,
    lambda_b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return population/update losses and their explicitly paired stay IDs."""
    y_all = design["y"].to_numpy(dtype=float)
    t_all = design["time_hours"].to_numpy(dtype=float)
    stay_ids_all = design["stay_id"].to_numpy(dtype=np.int64)
    fitted_all = design.loc[:, predictor_cols].to_numpy(dtype=float) @ gamma
    population: List[float] = []
    updated: List[float] = []
    stay_ids: List[int] = []
    for start, end in _contiguous_slices(design):
        y = y_all[start:end]
        t = t_all[start:end]
        index_idx, late_idx = data_utils.clock_split_indices(
            t,
            index_hours=index_hours,
            min_index_obs=4,
            min_late_obs=1,
        )
        if index_idx.size == 0 or late_idx.size == 0:
            continue
        residual = y - fitted_all[start:end]
        offset = profiled_intercept(residual[index_idx], tau=tau, lambda_b=lambda_b)
        population.append(float(np.mean(check_loss(residual[late_idx], tau))))
        updated.append(float(np.mean(check_loss(residual[late_idx] - offset, tau))))
        local_ids = stay_ids_all[start:end]
        if not np.all(local_ids == local_ids[0]):
            raise RuntimeError("A design stay contains multiple stay IDs")
        if int(local_ids[0]) in stay_ids:
            raise RuntimeError("A stay was paired more than once")
        stay_ids.append(int(local_ids[0]))
    return np.asarray(population, dtype=float), np.asarray(updated, dtype=float), np.asarray(stay_ids, dtype=np.int64)


def write_table7(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Sensitivity of the profiled stay specific level update under horizon specific fitting and tuning.}",
        "\\label{tab:robustness_summary}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Analysis & Setting & $\\lambda_b$ & Population loss & Offset rule loss & Reduction, \\%\\\\",
        "\\midrule",
    ]
    for row in rows:
        setting = rf"$\tau={float(row['tau']):.2f}$, {int(float(row['index_hours']))} h"
        lines.append(
            f"Quantile and window & {setting} & {float(row['lambda_b']):.2g} & "
            f"{float(row['population_assessment_loss']):.4f} & {float(row['updated_assessment_loss']):.4f} & "
            f"{float(row['loss_reduction_percent']):.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\footnotesize{Each row uses the same upstream cache and one fixed stay-level partition. "
            "Eligibility is horizon specific: at least 8 total observations, 4 observations at or before the "
            "landmark, and 1 observation after it. The population component and penalty are re-fitted and selected "
            "within the corresponding horizon-specific training and tuning groups; assessment losses are paired "
            "by stay on the corresponding horizon-specific assessment group.}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one_spec(
    dataset: Dict[str, object],
    stays: pd.DataFrame,
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tau: float,
    index_hours: float,
    lambda_grid: Sequence[float],
    work_dir: Path,
    r_script: Path,
    force_refit: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    eligible = horizon_eligible_indices(dataset, index_hours)
    eligible_set = set(eligible.tolist())
    train_idx = np.asarray([i for i in split[0] if int(i) in eligible_set], dtype=np.int64)
    tuning_idx = np.asarray([i for i in split[1] if int(i) in eligible_set], dtype=np.int64)
    assessment_idx = np.asarray([i for i in split[2] if int(i) in eligible_set], dtype=np.int64)
    if min(train_idx.size, tuning_idx.size, assessment_idx.size) < 20:
        raise RuntimeError(f"Horizon {index_hours:g} has too few qualified stays in one split")

    standardized, age_info = apply_training_age_standardization(dataset, stays, train_idx)
    train_design, _ = design_frame(data_utils.subset_cluster_data(standardized, train_idx))
    tuning_design, _ = design_frame(data_utils.subset_cluster_data(standardized, tuning_idx))
    assessment_design, _ = design_frame(data_utils.subset_cluster_data(standardized, assessment_idx))
    gamma = fit_common_quantile(
        train_design,
        X_PREDICTORS,
        tau=tau,
        work_dir=work_dir,
        r_script=r_script,
        force_refit=force_refit,
    )
    tuning = tune_lambda_generic(
        tuning_design,
        X_PREDICTORS,
        gamma,
        tau=tau,
        index_hours=index_hours,
        lambda_grid=lambda_grid,
    )
    selected_lambda = float(tuning["best"]["lambda_b"])
    population, updated, stay_ids = paired_assessment_losses(
        assessment_design,
        X_PREDICTORS,
        gamma,
        tau=tau,
        index_hours=index_hours,
        lambda_b=selected_lambda,
    )
    if population.size != assessment_idx.size or np.unique(stay_ids).size != stay_ids.size:
        raise RuntimeError("Horizon assessment pairing does not cover each qualified stay exactly once")
    population_loss = float(np.mean(population))
    updated_loss = float(np.mean(updated))
    row = {
        "tau": float(tau),
        "index_hours": float(index_hours),
        "lambda_b": selected_lambda,
        "population_assessment_loss": population_loss,
        "population_assessment_loss_se": se_mean(population),
        "updated_assessment_loss": updated_loss,
        "updated_assessment_loss_se": se_mean(updated),
        "loss_reduction_percent": float(100.0 * (population_loss - updated_loss) / population_loss),
        "n_base_stays": int(len(data_utils.ensure_cluster_lists(dataset)["y_list"])),
        "n_eligible_stays": int(eligible.size),
        "train_stays": int(train_idx.size),
        "tuning_stays": int(tuning_idx.size),
        "assessment_stays": int(assessment_idx.size),
        "minimum_total_observations": 8,
        "minimum_index_observations": 4,
        "minimum_later_observations": 1,
        "split_rule": "t<=h index; t>h later",
        "age_definition": "icu_proxy",
    }
    detail = {
        "row": row,
        "age_standardization": age_info,
        "lambda_grid": tuning["grid"],
        "population_coefficients": gamma.tolist(),
        "paired_stay_count": int(stay_ids.size),
    }
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-cache", type=Path, required=True)
    parser.add_argument("--stays-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--tuning-fraction", type=float, default=0.20)
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["age"]["age_definition"] != "icu_proxy":
        raise RuntimeError("Table 7 requires the approved icu_proxy age definition")
    if config["landmark"]["split"] != "t <= h is index and t > h is later; observations exactly at h belong to index":
        raise RuntimeError("Table 7 configuration does not contain the approved strict-clock rule")
    obs = _safe_read_frame(args.obs_cache)
    stays = _safe_read_frame(args.stays_cache)
    dataset, data_summary, basis_spec = build_dataset_from_cache(
        obs,
        stays,
        fit_stays=0,
        seed=args.seed,
        analysis_hours=24.0,
        qualification_index_hours=None,
        min_total_obs=8,
        min_index_obs=4,
        min_late_obs=1,
    )
    enriched = data_utils.ensure_cluster_lists(dataset)
    split, assignment_info = preserve_existing_assignments(
        dataset,
        seed=args.seed,
        train_fraction=args.train_fraction,
        tuning_fraction=args.tuning_fraction,
    )
    lambda_grid = tuple(float(x) for x in DEFAULT_LAMBDA_GRID)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    r_script = Path(__file__).with_name("fit_quantile_common.R")
    rows: List[Dict[str, object]] = []
    details: List[Dict[str, object]] = []
    for tau, horizon in DEFAULT_SPECS:
        print(f"Fitting tau={tau:.2f}, horizon={horizon:g} h", flush=True)
        row, detail = run_one_spec(
            dataset,
            stays,
            split,
            tau=tau,
            index_hours=horizon,
            lambda_grid=lambda_grid,
            work_dir=args.work_dir / f"tau_{tau:.2f}_h{int(horizon)}",
            r_script=r_script,
            force_refit=args.force_refit,
        )
        rows.append(row)
        details.append(detail)
        print(
            f"  n={row['assessment_stays']}, lambda={row['lambda_b']:g}, reduction={row['loss_reduction_percent']:.2f}%",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "landmark_table7.csv", index=False)
    write_table7(rows, args.output_dir / "robustness_summary.tex")
    payload = {
        "status": "complete",
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "settings": {
            "seed": int(args.seed),
            "train_fraction": float(args.train_fraction),
            "tuning_fraction": float(args.tuning_fraction),
            "lambda_grid": list(lambda_grid),
            "specifications": [[float(tau), float(horizon)] for tau, horizon in DEFAULT_SPECS],
            "queue_source": "all schema 4 cache stays with at least 8 total MAP observations",
            "split_rule": "t<=h index; t>h later",
            "minimum_total_observations": 8,
            "minimum_index_observations": 4,
            "minimum_later_observations": 1,
            "assignment": "one fixed stay-level split before horizon-specific eligibility filtering",
        },
        "data_summary": data_summary,
        "basis": {"Tmax": basis_spec.Tmax, "knots": basis_spec.knots, "basis_dimension": basis_spec.L},
        "fixed_split": {
            "train_stays": int(split[0].size),
            "tuning_stays": int(split[1].size),
            "assessment_stays": int(split[2].size),
            **assignment_info,
        },
        "rows": rows,
        "details": details,
        "artifacts": {
            "csv": str(args.output_dir / "landmark_table7.csv"),
            "tex": str(args.output_dir / "robustness_summary.tex"),
        },
    }
    (args.output_dir / "landmark_table7_results.json").write_text(
        json.dumps(data_utils.to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(data_utils.to_serializable({"status": payload["status"], "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
