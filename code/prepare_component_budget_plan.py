from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


SETTINGS = ("informative_recovery", "weaker_recovery", "information_limit", "serial_integer_stress")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.source / "replicates"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting in SETTINGS:
        for number in range(1, 201):
            frozen = json.loads((directory / setting / f"rep_{number:04d}/frozen.json").read_text())
            learning = frozen["learning"]
            fits = [learning["best_continuous"]]
            fits += [row["fit"] for row in learning["validation_rows"] if "fit" in row]
            rows.append({"setting": setting, "replicate": number,
                         "fullfit_optimizer_success": learning["best_continuous"]["optimizer_success"],
                         "selected_kind": learning["selected_kind"], "eta": learning["best_continuous_eta"],
                         "learning_seconds": learning["seconds"],
                         "sum_fit_seconds": sum(fit["seconds"] for fit in fits),
                         "fullfit_seconds": learning["best_continuous"]["seconds"],
                         "fullfit_projected_gradient_norm": learning["best_continuous"]["projected_gradient_norm"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "optimization_budget_inventory.csv", index=False)
    selected = frame[~frame.fullfit_optimizer_success].copy()
    selected["selection_reason"] = "all_full_fits_reaching_iteration_limit"
    controls = (frame[frame.fullfit_optimizer_success]
                .sort_values("replicate").groupby(["setting", "selected_kind"], sort=False).head(1).copy())
    controls["selection_reason"] = "first_successful_full_fit_in_each_setting_and_selected_branch"
    jobs = pd.concat([selected, controls], ignore_index=True).sort_values(["setting", "replicate"])
    cv_controls = frame.sort_values("replicate").groupby(["setting", "fullfit_optimizer_success", "selected_kind"]).head(1)
    cv_keys = set(zip(cv_controls.setting, cv_controls.replicate))
    jobs["redo_cv"] = [(row.setting, row.replicate) in cv_keys for row in jobs.itertuples()]
    assert len(jobs) == 211 and len(selected) == 203 and jobs.redo_cv.sum() == 15
    jobs.to_csv(output / "optimization_budget_jobs.csv", index=False)
    report = {"selection_uses_assessment_or_reference_outcomes": False,
              "iteration_budgets": [300, 500], "deterministic_starts": 2,
              "selected_jobs": len(jobs), "all_fullfit_iteration_limit_jobs": len(selected),
              "fullfit_iteration_limit_selected_continuous": int((selected.selected_kind == "continuous").sum()),
              "successful_control_jobs": len(controls),
              "complete_cv_controls": int(jobs.redo_cv.sum()),
              "cv_control_rule": "First replicate in each setting, original fullfit termination status, and originally selected branch stratum.",
              "original_complete_learning_seconds_selected_total": float(jobs.learning_seconds.sum()),
              "original_complete_learning_seconds_selected_median": float(jobs.learning_seconds.median()),
              "original_fullfit_seconds_selected_median": float(jobs.fullfit_seconds.median()),
              "approximate_8x_iteration_budget_seconds_serial": float(jobs.sum_fit_seconds.sum() * 8),
              "jobs_by_setting_and_branch": jobs.groupby(["setting", "selected_kind"]).size().reset_index(name="count").to_dict("records")}
    (output / "optimization_budget_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
