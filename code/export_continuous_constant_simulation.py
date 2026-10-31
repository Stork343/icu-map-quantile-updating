import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from run_adaptive_mixed_simulation import SETTINGS
from run_continuous_constant_simulation import COMPARISON_COUNT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified = json.loads((args.source / "verification.json").read_text())
    if (verified["status"] != "complete" or verified["completed_replicates"] != 800
            or verified["comparison_count"] != COMPARISON_COUNT
            or not verified["all_profiles_certified"] or not verified["all_history_coefficients_zero"]):
        raise RuntimeError("The public control export requires all 800 verified replicates")
    methods = pd.read_csv(args.source / "method_summary.csv")
    paired = pd.read_csv(args.source / "paired_summary.csv")
    if len(methods) != 240 or len(paired) != 192 or not (paired.replicates == 200).all():
        raise RuntimeError("The aggregate control design is incomplete")
    sources = []
    for name in ("environment.json", "verification.json", "method_summary.csv", "paired_summary.csv",
                 "methods_replicates.csv", "paired_replicates.csv", "replicate_manifest.csv",
                 "optimizer_replicates.csv", "optimizer_summary.csv", "validation_scores.csv", "penalty_replicates.csv"):
        sources.append((args.source / name, Path(name)))
    for setting in SETTINGS:
        for replicate in range(1, 201):
            relative = Path("replicates") / setting / f"rep_{replicate:04d}"
            directory = args.source / relative
            frozen = json.loads((directory / "frozen.json").read_text())
            fits = [frozen["control"]["continuous_fit"]] + [row["fit"] for row in frozen["control"]["validation_rows"]]
            if any(np.any(np.asarray(fit["params"])[:, 3:] != 0) for fit in fits):
                raise RuntimeError("An exported constant control uses history coefficients")
            for name in ("frozen.json", "methods.csv", "paired.csv", "assessment_metrics.parquet", "complete.json"):
                sources.append((directory / name, relative / name))
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for source, relative in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix in (".json", ".csv"):
            text = source.read_text()
            if any(value in text for value in ("/Users/", "/home/", ".review-work/")):
                raise RuntimeError(f"A local path appears in a public control record: {relative}")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append({"file": relative.as_posix(), "bytes": source.stat().st_size})
    design = json.loads((args.source / "design.json").read_text())
    design["source"] = "component_learning/simulation"
    (args.output / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    files.append({"file": "design.json", "bytes": (args.output / "design.json").stat().st_size})
    manifest = {"status": "passed", "replicates": 800, "paired_comparisons": 192,
                "methods": len(methods), "individual_clinical_records": 0,
                "history_coefficients_fixed_at_zero": True, "files": files}
    (args.output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
