import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import split_window_data as data_utils


MAP_SOURCE_CHOICES = ("combined", "invasive", "noninvasive")
CACHE_SCHEMA_VERSION = 3
REQUIRED_SOURCE_FILES = (
    "hosp/admissions.csv.gz",
    "hosp/patients.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz",
    "icu/icustays.csv.gz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_manifest(data_root: Path) -> Dict[str, Dict[str, object]]:
    checksum_path = data_root / "SHA256SUMS.txt"
    checksums: Dict[str, str] = {}
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                checksums[fields[1].replace("\\", "/")] = fields[0]
    manifest: Dict[str, Dict[str, object]] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = data_root / Path(relative)
        if not path.exists():
            raise FileNotFoundError(path)
        manifest[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256_from_distribution_manifest": checksums.get(relative),
        }
    return manifest


def _write_frame(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False)
        return fallback


def _selected_stays(data_root: Path, analysis_hours: float, candidate_stays: int, seed: int) -> pd.DataFrame:
    paths = data_utils.MimicPaths.from_root(data_root)
    patients = data_utils._load_patients(paths.patients_path)
    admissions = data_utils._load_admissions(paths.admissions_path)
    candidates = data_utils._load_icu_stays(
        paths.icustays_path,
        patients,
        admissions,
        analysis_hours=analysis_hours,
        keep_one_stay_per_subject=True,
    )
    stays = pd.DataFrame(list(candidates.values()))
    if stays.empty:
        raise RuntimeError("No adult first ICU stays were found in MIMIC-IV.")
    stays = stays.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    if candidate_stays > 0 and candidate_stays < stays.shape[0]:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(stays.index.to_numpy(), size=candidate_stays, replace=False))
        stays = stays.iloc[keep].sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    stays["male"] = (stays["gender"].astype(str).str.upper() == "M").astype(float)
    stays["emergency_or_urgent"] = stays["admission_type"].map(data_utils._emergency_indicator).astype(float)
    return stays


def _aggregate_map_buckets(obs: pd.DataFrame) -> pd.DataFrame:
    """Apply bucket-level source priority and average the retained source rows."""
    required = {
        "stay_id",
        "charttime",
        "time_hours",
        "time_bucket",
        "source_priority",
        "itemid",
        "map_value",
    }
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"MAP bucket input is missing columns: {sorted(missing)}")
    if obs.empty:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "charttime",
                "time_hours",
                "time_bucket",
                "itemid",
                "map_value",
                "map_source",
                "bucket_measurement_count",
            ]
        )

    ordered = obs.sort_values(["stay_id", "time_bucket", "source_priority", "charttime"]).copy()
    bucket_priority = ordered.groupby(["stay_id", "time_bucket"], sort=False)["source_priority"].transform("min")
    selected = ordered.loc[ordered["source_priority"].eq(bucket_priority)].copy()
    aggregated = selected.groupby(["stay_id", "time_bucket"], as_index=False, sort=False).agg(
        charttime=("charttime", "mean"),
        time_hours=("time_hours", "mean"),
        source_priority=("source_priority", "first"),
        itemid=("itemid", "first"),
        map_value=("map_value", "mean"),
        bucket_measurement_count=("map_value", "size"),
    )
    aggregated["map_source"] = np.where(aggregated["source_priority"].eq(0), "invasive", "noninvasive")
    aggregated = aggregated.sort_values(["stay_id", "time_hours", "time_bucket"]).reset_index(drop=True)
    return aggregated[
        [
            "stay_id",
            "charttime",
            "time_hours",
            "time_bucket",
            "itemid",
            "map_value",
            "map_source",
            "bucket_measurement_count",
        ]
    ]


def _time_bucket_from_datetimes(
    charttime: pd.Series,
    intime: pd.Series,
    duplicate_window_minutes: int,
) -> np.ndarray:
    if duplicate_window_minutes <= 0:
        return charttime.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    elapsed_ns = (charttime - intime).to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    bucket_width_ns = np.int64(duplicate_window_minutes) * np.int64(60_000_000_000)
    return np.floor_divide(elapsed_ns, bucket_width_ns).astype(np.int64)


def _read_map_observations(
    data_root: Path,
    stays: pd.DataFrame,
    analysis_hours: float,
    duplicate_window_minutes: int,
    map_source: str,
    chunksize: int,
    progress_every: int,
) -> pd.DataFrame:
    if map_source not in MAP_SOURCE_CHOICES:
        raise ValueError(f"map_source must be one of {MAP_SOURCE_CHOICES}")
    paths = data_utils.MimicPaths.from_root(data_root)
    itemids = data_utils.locate_map_itemids(paths.d_items_path)
    invasive_itemid = int(itemids["abpm"])
    noninvasive_itemid = int(itemids["nbpm"])
    if map_source == "invasive":
        map_itemids = {invasive_itemid}
    elif map_source == "noninvasive":
        map_itemids = {noninvasive_itemid}
    else:
        map_itemids = {invasive_itemid, noninvasive_itemid}
    stay_ids = set(stays["stay_id"].astype(int).tolist())
    intime = stays.set_index("stay_id")["intime"]
    window_end = stays.set_index("stay_id")["window_end"]
    frames = []
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]

    reader = pd.read_csv(paths.chartevents_path, compression="gzip", usecols=usecols, chunksize=chunksize)
    for chunk_id, chunk in enumerate(reader, start=1):
        local = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(map_itemids)
            & chunk["valuenum"].notna()
        ].copy()
        if local.empty:
            continue
        local["map_value"] = pd.to_numeric(local["valuenum"], errors="coerce")
        local = local[(local["map_value"] >= 20.0) & (local["map_value"] <= 200.0)].copy()
        if local.empty:
            continue
        local["charttime"] = pd.to_datetime(local["charttime"], errors="coerce")
        local["intime"] = local["stay_id"].map(intime)
        local["window_end"] = local["stay_id"].map(window_end)
        local["time_hours"] = (local["charttime"] - local["intime"]).dt.total_seconds() / 3600.0
        local = local[
            (local["time_hours"] >= 0.0)
            & (local["time_hours"] <= analysis_hours)
            & (local["charttime"] <= local["window_end"])
        ].copy()
        if local.empty:
            continue
        local["source_priority"] = np.where(local["itemid"].astype(int) == invasive_itemid, 0, 1)
        local["time_bucket"] = _time_bucket_from_datetimes(
            local["charttime"],
            local["intime"],
            duplicate_window_minutes,
        )
        frames.append(
            local[
                [
                    "stay_id",
                    "charttime",
                    "time_hours",
                    "time_bucket",
                    "source_priority",
                    "itemid",
                    "map_value",
                ]
            ]
        )
        if progress_every > 0 and chunk_id % progress_every == 0:
            print(f"Processed {chunk_id} chartevents chunks; retained {sum(len(x) for x in frames)} MAP rows")

    if not frames:
        raise RuntimeError("No eligible MAP observations were found.")
    obs = pd.concat(frames, ignore_index=True)
    return _aggregate_map_buckets(obs)


def build_map_cache(
    data_root: Path,
    cache_dir: Path,
    seed: int,
    candidate_stays: int,
    analysis_hours: float,
    min_obs_per_stay: int,
    duplicate_window_minutes: int,
    map_source: str,
    chunksize: int,
    progress_every: int,
    refresh: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    obs_path = cache_dir / "mimic_map_observations.parquet"
    stays_path = cache_dir / "mimic_map_stays.parquet"
    manifest_path = cache_dir / "mimic_map_full_cache_metadata.json"
    expected_manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "seed": int(seed),
        "candidate_stays": int(candidate_stays),
        "analysis_hours": float(analysis_hours),
        "min_obs_per_stay": int(min_obs_per_stay),
        "duplicate_window_minutes": int(duplicate_window_minutes),
        "map_source": str(map_source),
        "source_files": _source_file_manifest(data_root),
        "algorithm_files": {
            Path(__file__).name: _sha256(Path(__file__)),
            "split_window_data.py": _sha256(Path(__file__).with_name("split_window_data.py")),
        },
    }
    if not refresh and (obs_path.exists() or obs_path.with_suffix(".csv.gz").exists()) and (
        stays_path.exists() or stays_path.with_suffix(".csv.gz").exists()
    ):
        if not manifest_path.exists():
            raise RuntimeError(
                f"Cache files exist without a compatible provenance manifest at {manifest_path}. "
                "Use --refresh-cache or a new cache directory."
            )
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"expected": value, "cached": cached_manifest.get(key)}
            for key, value in expected_manifest.items()
            if cached_manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Cache provenance does not match the requested analysis settings: "
                f"{json.dumps(mismatches, ensure_ascii=False)}. Use --refresh-cache or a new cache directory."
            )
        obs = data_utils._safe_read_frame(obs_path)
        stays = data_utils._safe_read_frame(stays_path)
        required_obs_columns = {
            "stay_id",
            "charttime",
            "time_hours",
            "time_bucket",
            "itemid",
            "map_value",
            "map_source",
            "bucket_measurement_count",
        }
        missing_obs_columns = required_obs_columns.difference(obs.columns)
        if missing_obs_columns:
            raise RuntimeError(
                f"Cached observation schema is missing {sorted(missing_obs_columns)}. "
                "Use --refresh-cache or a new cache directory."
            )
    else:
        stays = _selected_stays(data_root, analysis_hours, candidate_stays, seed)
        obs = _read_map_observations(
            data_root,
            stays,
            analysis_hours,
            duplicate_window_minutes,
            map_source,
            chunksize,
            progress_every,
        )
        counts = obs.groupby("stay_id").size()
        retained_ids = counts[counts >= min_obs_per_stay].index.astype(int)
        obs = obs[obs["stay_id"].isin(retained_ids)].copy()
        stays = stays[stays["stay_id"].isin(retained_ids)].copy()
        _write_frame(obs, obs_path)
        _write_frame(stays, stays_path)

    counts = obs.groupby("stay_id").agg(
        obs_count=("map_value", "size"),
        index_count=("time_hours", lambda x: int(np.sum(np.asarray(x) <= 12.0))),
        late_count=("time_hours", lambda x: int(np.sum(np.asarray(x) > 12.0))),
    )
    split_eligible = int(((counts["index_count"] >= 4) & (counts["late_count"] >= 1)).sum())
    metadata = {
        "analysis_type": "revised_mimic_map_cache",
        **expected_manifest,
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "icu_window_rule": "intime <= charttime <= min(outtime, intime + analysis_hours)",
        "bucket_source_rule": "invasive priority within bucket; otherwise noninvasive",
        "bucket_aggregation": "arithmetic mean within selected source and bucket",
        "retained_cache_stays": int(stays.shape[0]),
        "retained_cache_observations": int(obs.shape[0]),
        "split_window_eligible_stays_12h": split_eligible,
        "map_below_65_observation_fraction": float(np.mean(obs["map_value"].to_numpy(dtype=float) < 65.0)),
        "obs_cache": str(obs_path if obs_path.exists() else obs_path.with_suffix(".csv.gz")),
        "stays_cache": str(stays_path if stays_path.exists() else stays_path.with_suffix(".csv.gz")),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data_utils.to_serializable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return obs, stays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable MIMIC-IV MAP cache for split-window analyses.")
    parser.add_argument("--data-root", type=Path, default=Path("statistics-in-medicine-paper/data/mimic-iv-3.1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("statistics-in-medicine-paper/code/mimic_full_cache"))
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("statistics-in-medicine-paper/code/mimic_full_cache/mimic_map_full_cache_metadata.json"),
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--candidate-stays", type=int, default=100000)
    parser.add_argument("--analysis-hours", type=float, default=24.0)
    parser.add_argument("--min-obs", type=int, default=8)
    parser.add_argument("--duplicate-minutes", type=int, default=5)
    parser.add_argument("--map-source", choices=MAP_SOURCE_CHOICES, default="combined")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    _, _, meta = build_map_cache(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        seed=args.seed,
        candidate_stays=args.candidate_stays,
        analysis_hours=args.analysis_hours,
        min_obs_per_stay=args.min_obs,
        duplicate_window_minutes=args.duplicate_minutes,
        map_source=args.map_source,
        chunksize=args.chunksize,
        progress_every=args.progress_every,
        refresh=args.refresh_cache,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(data_utils.to_serializable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data_utils.to_serializable(meta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
