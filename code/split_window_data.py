import csv
import gzip
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


AGE_DEFINITION_CHOICES = ("anchor_age", "icu_proxy")


def _parse_optional_anchor_year(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"anchor_year must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValueError(f"anchor_year must be an integer, got {value!r}")
    year = int(numeric)
    if year < 1900 or year > 2300:
        raise ValueError(f"anchor_year is outside the supported shifted-year range: {year}")
    return year


def _parse_anchor_age(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("anchor_age is missing")
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"anchor_age must be numeric, got {value!r}") from exc
    if not math.isfinite(age) or age < 0.0:
        raise ValueError(f"anchor_age must be a finite nonnegative value, got {value!r}")
    return age


def derive_age_fields(
    anchor_age: Any,
    anchor_year: Any,
    intime: datetime | str,
    age_definition: str = "anchor_age",
) -> Dict[str, Any]:
    """Return recorded anchor age and an optional ICU-time age proxy.

    ``anchor_age`` remains the default until the age definition is approved at
    design gate 1. The proxy follows the MIMIC convention of adding the shifted
    ICU admission year difference to the age in the anchor year. A value of 91
    remains a high-age code and is never interpreted as a recovered true age.
    """
    if age_definition not in AGE_DEFINITION_CHOICES:
        raise ValueError(f"age_definition must be one of {AGE_DEFINITION_CHOICES}")
    age = _parse_anchor_age(anchor_age)
    parsed_anchor_year = _parse_optional_anchor_year(anchor_year)
    parsed_intime = _parse_time(intime) if isinstance(intime, str) else intime
    if parsed_intime is None:
        raise ValueError("intime is required to construct ICU admission age")
    proxy = None
    year_delta = None
    if parsed_anchor_year is not None:
        year_delta = int(parsed_intime.year - parsed_anchor_year)
        proxy = float(age + year_delta)
    elif age_definition == "icu_proxy":
        raise ValueError("anchor_year is required when age_definition='icu_proxy'")
    selected_age = age if age_definition == "anchor_age" else float(proxy)
    return {
        "anchor_age": age,
        "anchor_year": parsed_anchor_year,
        "age_at_icu_proxy": proxy,
        "age_year_delta": year_delta,
        "high_age_coded": bool(age == 91.0),
        "age": selected_age,
        "age_definition": age_definition,
    }


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
            try:
                anchor_age = _parse_anchor_age(row.get("anchor_age"))
                anchor_year = _parse_optional_anchor_year(row.get("anchor_year"))
            except ValueError as exc:
                raise ValueError(f"Invalid age provenance for subject_id={subject_id}: {exc}") from exc
            patients[subject_id] = {
                "gender": row.get("gender", ""),
                "anchor_age": anchor_age,
                "anchor_year": anchor_year,
                "anchor_year_group": row.get("anchor_year_group", ""),
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
    age_definition: str = "anchor_age",
) -> Dict[int, Dict[str, Any]]:
    if age_definition not in AGE_DEFINITION_CHOICES:
        raise ValueError(f"age_definition must be one of {AGE_DEFINITION_CHOICES}")
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
            intime = _parse_time(row["intime"])
            outtime = _parse_time(row["outtime"])
            if intime is None or outtime is None or outtime <= intime:
                continue
            try:
                age_fields = derive_age_fields(
                    patient.get("anchor_age"),
                    patient.get("anchor_year"),
                    intime,
                    age_definition=age_definition,
                )
            except ValueError as exc:
                raise ValueError(f"Invalid age provenance for stay_id={stay_id}: {exc}") from exc
            if age_fields["age"] < 18:
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
                    **age_fields,
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


def clock_split_indices(
    t_i: np.ndarray,
    index_hours: float,
    min_index_obs: int = 0,
    min_late_obs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split one stay strictly by clock time at ``index_hours``.

    Observations at the landmark belong to the index window. A failed minimum
    count returns two empty arrays so callers cannot silently substitute a
    count split.
    """
    times = np.asarray(t_i, dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("clock_split_indices requires finite times")
    if not np.isfinite(float(index_hours)):
        raise ValueError("index_hours must be finite")
    if min_index_obs < 0 or min_late_obs < 0:
        raise ValueError("minimum observation counts must be nonnegative")
    index_idx = np.flatnonzero(times <= float(index_hours)).astype(int)
    late_idx = np.flatnonzero(times > float(index_hours)).astype(int)
    if index_idx.size < int(min_index_obs) or late_idx.size < int(min_late_obs):
        return np.array([], dtype=int), np.array([], dtype=int)
    return index_idx, late_idx


def count_split_indices(
    t_i: np.ndarray,
    index_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a nonempty ordered stay by observation count.

    This function is explicit because it is a different estimand from a
    clock-time landmark. A stay with one record cannot supply both windows.
    """
    times = np.asarray(t_i, dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("count_split_indices requires finite times")
    n_i = int(times.size)
    if n_i <= 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    half_n = int(np.ceil(max(min(index_fraction, 0.95), 0.05) * n_i))
    half_n = int(np.clip(half_n, 1, n_i - 1))
    return np.arange(half_n, dtype=int), np.arange(half_n, n_i, dtype=int)


def _within_stay_split_indices(
    t_i: np.ndarray,
    index_fraction: float,
    index_hours: float,
    min_index_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible name for the explicit count split."""
    del index_hours, min_index_obs
    return count_split_indices(t_i, index_fraction=index_fraction)


def build_dataset_from_cache(
    obs: pd.DataFrame,
    stays: pd.DataFrame,
    fit_stays: int,
    seed: int,
    analysis_hours: float,
    qualification_index_hours: Optional[float] = 12.0,
    min_total_obs: int = 8,
    min_index_obs: int = 4,
    min_late_obs: int = 1,
) -> Tuple[Dict[str, object], Dict[str, object], BasisSpec]:
    obs = obs.copy()
    stays = stays.copy()
    required_age_columns = {"anchor_age", "anchor_year", "age_at_icu_proxy", "high_age_coded", "age_definition"}
    missing_age_columns = required_age_columns.difference(stays.columns)
    if missing_age_columns:
        raise RuntimeError(
            "Cached stay data lack the age provenance fields "
            f"{sorted(missing_age_columns)}. Rebuild the cache with build_revised_mimic_map_cache.py."
        )
    age_definitions = sorted(set(stays["age_definition"].dropna().astype(str)))
    if age_definitions != ["anchor_age"] and age_definitions != ["icu_proxy"]:
        raise RuntimeError(f"Cached stay data contain inconsistent age definitions: {age_definitions}")
    if qualification_index_hours is not None and not np.isfinite(float(qualification_index_hours)):
        raise ValueError("qualification_index_hours must be finite or None")
    if min_total_obs < 0 or min_index_obs < 0 or min_late_obs < 0:
        raise ValueError("minimum observation counts must be nonnegative")
    counts = obs.groupby("stay_id").agg(
        obs_count=("map_value", "size"),
        any_low=("map_value", lambda x: bool(np.any(np.asarray(x) < 65.0))),
    )
    if qualification_index_hours is None:
        counts["index_count"] = counts["obs_count"]
        counts["late_count"] = 0
        usable_ids = counts[counts["obs_count"] >= int(min_total_obs)].index.to_numpy(dtype=np.int64)
    else:
        counts["index_count"] = obs.groupby("stay_id")["time_hours"].apply(
            lambda x: int(np.sum(np.asarray(x) <= float(qualification_index_hours)))
        )
        counts["late_count"] = obs.groupby("stay_id")["time_hours"].apply(
            lambda x: int(np.sum(np.asarray(x) > float(qualification_index_hours)))
        )
        usable_ids = counts[
            (counts["obs_count"] >= int(min_total_obs))
            & (counts["index_count"] >= int(min_index_obs))
            & (counts["late_count"] >= int(min_late_obs))
        ].index.to_numpy(dtype=np.int64)
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
        "age_definition": age_definitions[0],
        "qualification_index_hours": (
            None if qualification_index_hours is None else float(qualification_index_hours)
        ),
        "minimum_index_observations": int(min_index_obs),
        "minimum_later_observations": int(min_late_obs),
        "minimum_total_observations": int(min_total_obs),
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
