from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")


def verify(output):
    rows = []
    for setting in SETTINGS:
        for number in range(1, 201):
            path = output / "replicates" / setting / f"rep_{number:04d}" / "frozen.json"
            learning = json.loads(path.read_text())["learning"]
            rows.append({"setting": setting, "replicate": number,
                "selected_kind": learning["selected_kind"], "ridge_eta": learning["best_continuous_eta"]})
    keys = ["setting", "replicate"]
    expected = pd.DataFrame(rows).set_index(keys).sort_index()
    observed = pd.read_csv(output / "selection_replicates.csv").set_index(keys).sort_index()
    if not expected.index.equals(observed.index) or not expected.selected_kind.equals(observed.selected_kind):
        raise ValueError("Exported component branch decisions differ from frozen fits")
    np.testing.assert_allclose(expected.ridge_eta, observed.ridge_eta, rtol=0, atol=0)
    fields = ["setting", "selected_kind", "ridge_eta"]
    expected_counts = pd.DataFrame(rows).groupby(fields).size().rename("count").sort_index()
    observed_counts = pd.read_csv(output / "selection_frequencies.csv").set_index(fields)["count"].sort_index()
    if not expected_counts.equals(observed_counts):
        raise ValueError("Exported component selection frequencies differ from frozen fits")
    report = {"status": "passed", "frozen_selections_checked": len(rows),
        "frequency_records_checked": len(expected_counts)}
    (output / "selection_export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
