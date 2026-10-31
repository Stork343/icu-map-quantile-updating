from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from run_adaptive_mixed_simulation import saved_job, write_json
from verify_adaptive_mixed_simulation import verify


AGGREGATE_FILES = (
    "methods_replicates.csv", "paired_replicates.csv", "selections_replicates.csv", "method_summary.csv",
    "paired_summary.csv", "selection_frequencies.csv", "stump_selection_frequencies.csv", "verification_replicates.csv",
)
REPLICATE_FILES = (
    "population_coefficients.csv", "methods.csv", "paired.csv", "selections.csv", "tuning.csv",
    "calibration.csv", "synthetic_latent_diagnostics.csv", "assessment_stay_metrics.parquet",
)


def assert_portable(value: object) -> None:
    if isinstance(value, str) and Path(value).is_absolute():
        raise ValueError("Public metadata contains an absolute filesystem path")
    if isinstance(value, dict):
        for item in value.values():
            assert_portable(item)
    if isinstance(value, list):
        for item in value:
            assert_portable(item)


def write_portable_json(path: Path, value: object) -> None:
    assert_portable(value)
    write_json(path, value)


def check_csv(path: Path) -> None:
    frame = pd.read_csv(path)
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        for value in frame[column].dropna().unique():
            assert_portable(str(value))


def export(source: Path, destination: Path, include_synthetic: bool) -> dict:
    if destination.exists():
        raise FileExistsError("The public export destination must be new")
    verification = verify(source)
    destination.mkdir(parents=True)
    configuration = json.loads((source / "design.json").read_text())
    write_portable_json(destination / "design.json", configuration)
    environment = json.loads((source / "environment.json").read_text())
    environment["source"], environment["core"] = "run_adaptive_mixed_simulation.py", "adaptive_profile_core.py"
    write_portable_json(destination / "environment.json", environment)
    for name in AGGREGATE_FILES:
        check_csv(source / name)
        shutil.copyfile(source / name, destination / name)
    completions = []
    candidate_rows = []
    execution = configuration["execution"]
    for setting in execution["settings"]:
        for replicate in range(1, execution["replicates"] + 1):
            relative = Path("replicates") / setting / f"rep_{replicate:04d}"
            original, public = source / relative, destination / relative
            public.mkdir(parents=True)
            for name in REPLICATE_FILES:
                if name.endswith(".csv"):
                    check_csv(original / name)
                shutil.copyfile(original / name, public / name)
            for name in ("started.json", "complete.json"):
                metadata = saved_job(json.loads((original / name).read_text()))
                write_portable_json(public / name, metadata)
                if name == "complete.json":
                    completions.append(metadata)
            resolution_path = original / "resolution_summary.json"
            if resolution_path.exists():
                resolution = json.loads(resolution_path.read_text())
                resolution["audit_scope"] = "source_verified_public_metadata"
                resolution["raw_failure_records_included"] = False
                write_portable_json(public / "resolution_summary.json", resolution)
            for horizon in execution["horizons"]:
                name = f"frozen_h{int(horizon)}.json"
                frozen = json.loads((original / name).read_text())
                write_portable_json(public / name, frozen)
                for method, rule in frozen["rules"].items():
                    spec = rule["spec"]
                    if spec["family"] == "adaptive":
                        kind = "geometry" if spec["gamma"] == 0 else "conditioned" if spec["theta"] == 0 else "contextual"
                    elif spec["basis"] == "level" and spec["lambda_level"] == 0:
                        kind = "original_q10"
                    elif spec["basis"] == "summary":
                        kind = "original_" + spec["summary"]
                    else:
                        kind = "original_" + spec["basis"]
                    candidate_rows.append({"setting": setting, "horizon": horizon, "method": method,
                                           "candidate_type": kind, "calibrated": rule["calibrated"]})
            if include_synthetic:
                shutil.copyfile(original / "synthetic_records.npz", public / "synthetic_records.npz")
    write_portable_json(destination / "completion_manifest.json", {
        "expected_replicates": len(completions), "completed_replicates": len(completions),
        "resolved_numerical_attempts": verification["resolved_numerical_attempts"],
        "unresolved_failures": verification["unresolved_failures"],
        "replicates": completions,
    })
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frequencies = candidate_frame.groupby(["setting", "horizon", "method", "candidate_type", "calibrated"],
                                                     sort=False).size().rename("count").reset_index()
    candidate_totals = candidate_frequencies.groupby(["setting", "horizon", "method"])["count"].transform("sum")
    if not (candidate_totals == execution["replicates"]).all():
        raise RuntimeError("Candidate type frequencies excluded a replicate")
    candidate_frequencies["frequency"] = candidate_frequencies["count"] / candidate_totals
    candidate_frequencies.to_csv(destination / "candidate_type_frequencies.csv", index=False)
    selections = pd.read_csv(destination / "selections_replicates.csv")
    stump = selections.loc[selections["method"] == "stump_original"]
    stump_frequencies = stump.groupby(["setting", "horizon", "stump_feature"], sort=False).size().rename("count").reset_index()
    stump_totals = stump_frequencies.groupby(["setting", "horizon"])["count"].transform("sum")
    if not (stump_totals == execution["replicates"]).all():
        raise RuntimeError("Stump feature frequencies excluded a replicate")
    stump_frequencies["frequency"] = stump_frequencies["count"] / stump_totals
    stump_frequencies.to_csv(destination / "stump_feature_frequencies.csv", index=False)
    public_verification = verify(destination)
    write_portable_json(destination / "verification.json", public_verification)
    result = {"status": "complete", "replicates": len(completions), "synthetic_records_included": include_synthetic,
              "restricted_clinical_records_included": False, "absolute_filesystem_paths": 0,
              "resolved_numerical_attempts": verification["resolved_numerical_attempts"],
              "unresolved_failures": verification["unresolved_failures"],
              "source_verification": verification, "public_verification": public_verification}
    write_portable_json(destination / "export_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.destination.resolve(), args.include_synthetic), indent=2))


if __name__ == "__main__":
    main()
