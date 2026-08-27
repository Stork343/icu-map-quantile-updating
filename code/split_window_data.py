import csv
import gzip
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BasisSpec:
    Tmax: float = 24.0
    knots: Tuple[float, ...] = (8.0, 16.0)
    scale_basis: bool = True
    include_intercept: bool = False
    center_basis: bool = True

    @property
    def L(self) -> int:
        base = 4 if self.include_intercept else 3
        return base + len(self.knots)


@dataclass
class MimicPaths:
    root_dir: Path
    icustays_path: Path
    patients_path: Path
    admissions_path: Path
    d_items_path: Path
    chartevents_path: Path

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "MimicPaths":
        root = Path(root_dir)
        return cls(
            root_dir=root,
            icustays_path=root / "icu" / "icustays.csv.gz",
            patients_path=root / "hosp" / "patients.csv.gz",
            admissions_path=root / "hosp" / "admissions.csv.gz",
            d_items_path=root / "icu" / "d_items.csv.gz",
            chartevents_path=root / "icu" / "chartevents.csv.gz",
        )


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def _safe_read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    fallback = path.with_suffix(".csv.gz")
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(path)


def _parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _emergency_indicator(admission_type: str) -> float:
    label = admission_type.upper()
    return float("EMER" in label or "URGENT" in label)


def locate_map_itemids(d_items_path: str | Path) -> Dict[str, int]:
    candidates: Dict[str, int] = {}
    with gzip.open(d_items_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label") or "").strip()
            itemid = int(row["itemid"])
            if label == "Arterial Blood Pressure mean":
                candidates["abpm"] = itemid
            elif label == "Non Invasive Blood Pressure mean":
                candidates["nbpm"] = itemid
    if "abpm" not in candidates or "nbpm" not in candidates:
        raise ValueError("Unable to locate invasive and noninvasive MAP itemids in d_items.")
    return candidates


def _load_patients(patients_path: str | Path) -> Dict[int, Dict[str, Any]]:
    patients: Dict[int, Dict[str, Any]] = {}
    with gzip.open(patients_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = int(row["subject_id"])
            patients[subject_id] = {
                "gender": row.get("gender", ""),
                "anchor_age": float(row.get("anchor_age") or 0.0),
            }
    return patients


def _load_admissions(admissions_path: str | Path) -> Dict[int, Dict[str, Any]]:
    admissions: Dict[int, Dict[str, Any]] = {}
    with gzip.open(admissions_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hadm_id = int(row["hadm_id"])
            admissions[hadm_id] = {
                "admission_type": row.get("admission_type", ""),
                "race": row.get("race", ""),
                "hospital_expire_flag": row.get("hospital_expire_flag", ""),
            }
    return admissions


def _load_icu_stays(
    icustays_path: str | Path,
    patients: Dict[int, Dict[str, Any]],
    admissions: Dict[int, Dict[str, Any]],
    analysis_hours: float,
    keep_one_stay_per_subject: bool,
) -> Dict[int, Dict[str, Any]]:
    candidate_stays: List[Dict[str, Any]] = []
    with gzip.open(icustays_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = int(row["subject_id"])
            hadm_id = int(row["hadm_id"])
            stay_id = int(row["stay_id"])
            patient = patients.get(subject_id)
            admission = admissions.get(hadm_id)
            if patient is None or admission is None:
                continue
            if patient["anchor_age"] < 18:
                continue
            intime = _parse_time(row["intime"])
            outtime = _parse_time(row["outtime"])
            if intime is None or outtime is None or outtime <= intime:
                continue
            window_end = min(outtime, intime + timedelta(hours=analysis_hours))
            if window_end <= intime:
                continue
            candidate_stays.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "stay_id": stay_id,
                    "intime": intime,
                    "outtime": outtime,
                    "window_end": window_end,
                    "gender": patient["gender"],
                    "age": patient["anchor_age"],
                    "admission_type": admission["admission_type"],
                }
            )

    if keep_one_stay_per_subject:
        first_stay: Dict[int, Dict[str, Any]] = {}
        for stay in sorted(candidate_stays, key=lambda item: (item["subject_id"], item["intime"])):
            first_stay.setdefault(stay["subject_id"], stay)
        candidate_stays = list(first_stay.values())

    return {stay["stay_id"]: stay for stay in candidate_stays}


def raw_truncated_cubic_spline_basis(t: np.ndarray, spec: BasisSpec) -> np.ndarray:
    ts = np.asarray(t, dtype=float) / spec.Tmax
    cols: List[np.ndarray] = []
    if spec.include_intercept:
        cols.append(np.ones_like(ts))
    cols.extend([ts, ts**2, ts**3])
    for knot in spec.knots:
        ks = knot / spec.Tmax
        cols.append(np.maximum(ts - ks, 0.0) ** 3)
    return np.column_stack(cols)


def basis_scales(spec: BasisSpec, n_grid: int = 500) -> np.ndarray:
    tgrid = np.linspace(0.0, spec.Tmax, n_grid)
    basis = raw_truncated_cubic_spline_basis(tgrid, spec)
    scales = np.sqrt(np.mean(basis**2, axis=0))
    if spec.include_intercept:
        scales[0] = 1.0
    return np.clip(scales, 1e-8, None)


def basis_center(spec: BasisSpec, n_grid: int = 500) -> np.ndarray:
    tgrid = np.linspace(0.0, spec.Tmax, n_grid)
    basis = raw_truncated_cubic_spline_basis(tgrid, spec)
    if spec.scale_basis:
        basis = basis / basis_scales(spec, n_grid=n_grid)
    return np.mean(basis, axis=0)


def spline_basis(t: np.ndarray, spec: BasisSpec) -> np.ndarray:
    basis = raw_truncated_cubic_spline_basis(t, spec)
    if spec.scale_basis:
        basis = basis / basis_scales(spec)
    if spec.center_basis:
        basis = basis - basis_center(spec)[None, :]
    return basis


def make_basis_list(t_list: Sequence[np.ndarray], spec: BasisSpec) -> List[np.ndarray]:
    return [spline_basis(np.asarray(t, dtype=float), spec) for t in t_list]


def build_cluster_dataset(
    y_list: Sequence[Sequence[float] | np.ndarray],
    t_list: Sequence[Sequence[float] | np.ndarray],
    X: np.ndarray,
    basis_spec: BasisSpec,
    cluster_ids: Optional[Sequence[Any]] = None,
    weights_list: Optional[Sequence[Sequence[float] | np.ndarray]] = None,
) -> Dict[str, object]:
    if len(y_list) != len(t_list):
        raise ValueError("y_list and t_list must have the same number of clusters.")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y_list):
        raise ValueError("X must be a 2D array with one row per cluster.")

    processed_y: List[np.ndarray] = []
    processed_t: List[np.ndarray] = []
    processed_w: List[np.ndarray] = []
    for y_i, t_i in zip(y_list, t_list):
        y_arr = np.asarray(y_i, dtype=float).reshape(-1)
        t_arr = np.asarray(t_i, dtype=float).reshape(-1)
        if y_arr.size != t_arr.size:
            raise ValueError("Within each cluster, y and t must have the same length.")
        if y_arr.size == 0:
            raise ValueError("Each cluster must contain at least one observation.")
        order = np.argsort(t_arr)
        processed_y.append(y_arr[order])
        processed_t.append(t_arr[order])

    if weights_list is None:
        processed_w = [np.ones_like(y_arr, dtype=float) for y_arr in processed_y]
    else:
        if len(weights_list) != len(y_list):
            raise ValueError("weights_list must match y_list length.")
        for w_i, y_i, t_i in zip(weights_list, y_list, t_list):
            w_arr = np.asarray(w_i, dtype=float).reshape(-1)
            y_ref = np.asarray(y_i, dtype=float).reshape(-1)
            t_ref = np.asarray(t_i, dtype=float).reshape(-1)
            if w_arr.size != y_ref.size or y_ref.size != t_ref.size:
                raise ValueError("Each weights_list element must match cluster observation length.")
            processed_w.append(w_arr[np.argsort(t_ref)])

    basis_list = make_basis_list(processed_t, basis_spec)
    ids: List[np.ndarray] = []
    y_long: List[np.ndarray] = []
    t_long: List[np.ndarray] = []
    x_long: List[np.ndarray] = []
    b_long: List[np.ndarray] = []
    w_long: List[np.ndarray] = []
    for idx, (y_i, t_i, B_i, w_i) in enumerate(zip(processed_y, processed_t, basis_list, processed_w)):
        n_i = y_i.size
        ids.append(np.full(n_i, idx, dtype=int))
        y_long.append(y_i)
        t_long.append(t_i)
        x_long.append(np.tile(X[idx], (n_i, 1)))
        b_long.append(B_i)
        w_long.append(w_i)

    cluster_ids_arr = np.arange(len(processed_y), dtype=int) if cluster_ids is None else np.asarray(list(cluster_ids))
    if cluster_ids_arr.shape[0] != len(processed_y):
        raise ValueError("cluster_ids must match the number of clusters.")

    return {
        "y": np.concatenate(y_long),
        "id": np.concatenate(ids),
        "time": np.concatenate(t_long),
        "X": X,
        "X_long": np.vstack(x_long),
        "B_fit_long": np.vstack(b_long),
        "B_fit_list": basis_list,
        "y_list": processed_y,
        "t_list": processed_t,
        "w_list": processed_w,
        "obs_weights": np.concatenate(w_long),
        "cluster_ids": cluster_ids_arr,
        "basis_spec": basis_spec,
    }


def ensure_cluster_lists(data: Dict[str, object]) -> Dict[str, object]:
    if "y_list" in data and "t_list" in data and "cluster_ids" in data:
        return data

    ids = np.asarray(data["id"], dtype=int)
    y = np.asarray(data["y"], dtype=float)
    time_vec = np.asarray(data["time"], dtype=float)
    weights = np.asarray(data.get("obs_weights", np.ones_like(y)), dtype=float)
    n_clusters = int(ids.max()) + 1 if ids.size else 0
    y_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    for i in range(n_clusters):
        mask = ids == i
        order = np.argsort(time_vec[mask])
        y_list.append(y[mask][order])
        t_list.append(time_vec[mask][order])
        w_list.append(weights[mask][order])

    out = dict(data)
    out["y_list"] = y_list
    out["t_list"] = t_list
    out["w_list"] = w_list
    out["obs_weights"] = np.concatenate(w_list) if w_list else np.array([], dtype=float)
    out["cluster_ids"] = np.arange(n_clusters, dtype=int)
    return out


def subset_cluster_data(data: Dict[str, object], cluster_index: Sequence[int]) -> Dict[str, object]:
    enriched = ensure_cluster_lists(data)
    y_list = enriched["y_list"]
    t_list = enriched["t_list"]
    X = np.asarray(enriched["X"], dtype=float)
    cluster_ids = np.asarray(enriched["cluster_ids"])
    w_list = enriched.get("w_list")
    indices = list(cluster_index)
    sub_w = [w_list[idx] for idx in indices] if w_list is not None else None
    return build_cluster_dataset(
        [y_list[idx] for idx in indices],
        [t_list[idx] for idx in indices],
        X[indices],
        enriched["basis_spec"],
        cluster_ids=cluster_ids[indices],
        weights_list=sub_w,
    )


def _within_stay_split_indices(
    t_i: np.ndarray,
    index_fraction: float,
    index_hours: float,
    min_index_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_i = int(t_i.size)
    if n_i <= 1:
        return np.array([0], dtype=int), np.array([0], dtype=int)
    half_n = int(np.ceil(max(min(index_fraction, 0.95), 0.05) * n_i))
    half_n = int(np.clip(half_n, 1, n_i - 1))
    idx_half = np.arange(half_n, dtype=int)
    idx_time = np.where(t_i <= index_hours)[0]
    if idx_time.size >= min_index_obs and idx_time.size < n_i:
        index_idx = idx_time.astype(int)
    else:
        index_idx = idx_half
    if index_idx.size >= n_i:
        index_idx = np.arange(n_i - 1, dtype=int)
    late_idx = np.setdiff1d(np.arange(n_i, dtype=int), index_idx, assume_unique=True)
    if late_idx.size == 0:
        late_idx = np.array([n_i - 1], dtype=int)
        index_idx = np.arange(n_i - 1, dtype=int)
    return index_idx, late_idx


def build_dataset_from_cache(
    obs: pd.DataFrame,
    stays: pd.DataFrame,
    fit_stays: int,
    seed: int,
    analysis_hours: float,
) -> Tuple[Dict[str, object], Dict[str, object], BasisSpec]:
    obs = obs.copy()
    stays = stays.copy()
    counts = obs.groupby("stay_id").agg(
        obs_count=("map_value", "size"),
        index_count=("time_hours", lambda x: int(np.sum(np.asarray(x) <= 12.0))),
        late_count=("time_hours", lambda x: int(np.sum(np.asarray(x) > 12.0))),
        any_low=("map_value", lambda x: bool(np.any(np.asarray(x) < 65.0))),
    )
    usable_ids = counts[(counts["index_count"] >= 4) & (counts["late_count"] >= 1)].index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed + 17)
    if fit_stays > 0 and usable_ids.size > fit_stays:
        selected_ids = np.asarray(sorted(rng.choice(usable_ids, size=fit_stays, replace=False).tolist()), dtype=np.int64)
    else:
        selected_ids = np.asarray(sorted(usable_ids.tolist()), dtype=np.int64)
    if selected_ids.size < 20:
        raise RuntimeError("Too few usable stays for fitting. Increase the available MIMIC-IV cache.")

    obs_sel = obs[obs["stay_id"].isin(selected_ids)].sort_values(["stay_id", "time_hours"])
    stays_sel = stays[stays["stay_id"].isin(selected_ids)].set_index("stay_id").loc[selected_ids].reset_index()
    age = stays_sel["age"].to_numpy(dtype=float)
    age_sd = float(np.std(age, ddof=1)) if age.size > 1 else 1.0
    if age_sd <= 0:
        age_sd = 1.0
    age_z = (age - float(np.mean(age))) / age_sd
    X = np.column_stack(
        [
            np.ones(selected_ids.size),
            age_z,
            stays_sel["male"].to_numpy(dtype=float),
            stays_sel["emergency_or_urgent"].to_numpy(dtype=float),
        ]
    )

    y_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    for stay_id in selected_ids:
        local = obs_sel[obs_sel["stay_id"] == stay_id]
        y_list.append(local["map_value"].to_numpy(dtype=float))
        t_list.append(local["time_hours"].to_numpy(dtype=float))

    basis_spec = BasisSpec(Tmax=float(analysis_hours), knots=(8.0, 16.0), scale_basis=True, include_intercept=False, center_basis=True)
    dataset = build_cluster_dataset(y_list, t_list, X, basis_spec, cluster_ids=selected_ids)
    selected_counts = counts.loc[selected_ids]
    summary = {
        "fit_stays_requested": int(fit_stays),
        "fit_stays": int(selected_ids.size),
        "fit_observations": int(obs_sel.shape[0]),
        "obs_per_stay_median": float(np.median(selected_counts["obs_count"])),
        "obs_per_stay_iqr": [
            float(np.quantile(selected_counts["obs_count"], 0.25)),
            float(np.quantile(selected_counts["obs_count"], 0.75)),
        ],
        "index_obs_per_stay_median": float(np.median(selected_counts["index_count"])),
        "late_obs_per_stay_median": float(np.median(selected_counts["late_count"])),
        "stay_fraction_with_any_map_below_65": float(np.mean(selected_counts["any_low"])),
        "map_below_65_observation_fraction": float(np.mean(obs_sel["map_value"].to_numpy(dtype=float) < 65.0)),
    }
    return dataset, summary, basis_spec
