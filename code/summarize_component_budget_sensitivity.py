from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


KEYS = ["setting", "replicate", "scope", "horizon_hours", "method"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, results, output = args.source, args.results, args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((results / "budget_execution_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["launched_jobs"] == 211
    jobs = pd.read_csv(args.jobs)
    objective, risk, cv, cv_risk = [], [], [], []
    for job in jobs.to_dict("records"):
        directory = results / "budget_results" / job["setting"]
        report = json.loads((directory / f"rep_{job['replicate']:04d}.json").read_text())
        assert report["status"] == "complete"
        for row in report["results"]:
            identity = {"setting": job["setting"], "replicate": job["replicate"], "budget": row["budget"]}
            assert row["all_profiles_certified"]
            if row["mode"] == "fullfit_fixed_cv":
                objective.append({**identity, "original_optimizer_success": report["original_fullfit_optimizer_success"],
                                  "original_objective": report["original_fullfit_objective"],
                                  "objective": row["fullfit_objective"], "objective_change": row["fullfit_objective_change"],
                                  "optimizer_success": row["fullfit_optimizer_success"],
                                  "projected_gradient_norm": row["fullfit_projected_gradient_norm"]})
                risk.extend({**identity, **values} for values in row["risk"])
            else:
                cv.append({**identity, "original_selected_kind": report["original_selected_kind"],
                           "selected_kind": row["selected_kind"], "selected_kind_changed": row["selected_kind_changed"],
                           "original_eta": report["original_eta"], "eta": row["eta"],
                           "eta_changed": row["eta_changed"], "fullfit_objective_change": row["fullfit_objective_change"]})
                cv_risk.extend({**identity, **values} for values in row["risk"])
    objectives, risks = pd.DataFrame(objective), pd.DataFrame(risk)
    cv_frame, cv_risks = pd.DataFrame(cv), pd.DataFrame(cv_risk)
    assert len(objectives) == 422 and len(cv_frame) == 30
    objectives.to_csv(output / "budget_objective_changes.csv", index=False)
    cv_frame.to_csv(output / "budget_cv_controls.csv", index=False)
    cv_risks.to_csv(output / "budget_cv_control_risks.csv", index=False)
    original = pd.read_csv(source / "methods_replicates.csv").set_index(KEYS).sort_index()
    contrasts = json.loads((source / "design.json").read_text())["contrasts"]
    comparisons, risk_changes = [], []
    for budget, selected in risks.groupby("budget"):
        selected = selected.set_index(KEYS).sort_index()
        assert not selected.index.duplicated().any()
        updated = original.copy()
        updated.loc[selected.index, ["mean_loss", "p_lt", "p_le"]] = selected[["mean_loss", "p_lt", "p_le"]]
        difference = updated.mean_loss - original.mean_loss
        changes = difference.rename("difference").reset_index()
        for key, group in changes.groupby(["setting", "scope", "horizon_hours", "method"]):
            if not group.method.iloc[0].startswith(("selected_", "continuous_")):
                continue
            assert len(group) == 200
            risk_changes.append({"budget": budget, **dict(zip(["setting", "scope", "horizon_hours", "method"], key)),
                                 "replicates": 200, "mean_risk_change": float(group.difference.mean()),
                                 "maximum_absolute_replicate_change": float(group.difference.abs().max()),
                                 "changed_replicates": int((group.difference.abs() > 1e-10).sum())})
        wide = updated.mean_loss.unstack("method")
        for first, second in contrasts:
            values = (wide[first] - wide[second]).rename("difference").reset_index()
            for key, group in values.groupby(["setting", "scope", "horizon_hours"]):
                assert len(group) == 200 and group.replicate.nunique() == 200
                mean, se = float(group.difference.mean()), float(group.difference.std(ddof=1) / np.sqrt(200))
                within = t.ppf(1 - .05 / (2 * 192), 199) * se
                both = t.ppf(1 - .05 / (2 * 384), 199) * se
                comparisons.append({"budget": budget, **dict(zip(["setting", "scope", "horizon_hours"], key)),
                                    "first": first, "second": second, "replicates": 200,
                                    "mean_difference": mean, "mcse": se,
                                    "within_budget_192_low": mean - within,
                                    "within_budget_192_high": mean + within,
                                    "both_budgets_384_low": mean - both,
                                    "both_budgets_384_high": mean + both})
    comparison_frame, change_frame = pd.DataFrame(comparisons), pd.DataFrame(risk_changes)
    assert len(comparison_frame) == 384
    comparison_frame.to_csv(output / "budget_paired_summary.csv", index=False)
    change_frame.to_csv(output / "budget_risk_changes.csv", index=False)
    headline = comparison_frame[(comparison_frame.scope == "reference")
                               & (comparison_frame["first"] == "selected_direct")
                               & (comparison_frame["second"] == "constant_direct")
                               & (comparison_frame.setting != "information_limit")]
    controls = cv_risks.set_index(["budget", *KEYS])
    control_original = original.loc[controls.index.droplevel("budget")].copy()
    control_differences = controls.mean_loss.to_numpy() - control_original.mean_loss.to_numpy()
    controls = controls.assign(mean_loss_change=control_differences)
    controls.reset_index().to_csv(output / "budget_cv_control_risks.csv", index=False)
    report = {"status": "complete", "optimization_budget_sensitivity": True,
              "new_independent_simulation_datasets": 0, "original_100_iteration_results_modified": False,
              "fullfit_records": 211, "iteration_limit_fullfit_records": 203,
              "complete_cv_records": 15, "iteration_budgets": [300, 500],
              "all_profiles_certified": True,
              "objective_changes": objectives.groupby("budget").agg(
                  minimum_change=("objective_change", "min"), median_change=("objective_change", "median"),
                  maximum_change=("objective_change", "max"), optimizer_success=("optimizer_success", "sum")).reset_index().to_dict("records"),
              "cv_changes": cv_frame.groupby("budget").agg(
                  eta_changed=("eta_changed", "sum"), selected_kind_changed=("selected_kind_changed", "sum")).reset_index().to_dict("records"),
              "direct_reference_headline": headline.to_dict("records"),
              "all_18_direct_reference_means_negative": bool((headline.mean_difference < 0).all()),
              "all_18_direct_reference_384_intervals_negative": bool((headline.both_budgets_384_high < 0).all()),
              "maximum_absolute_direct_selected_mean_risk_shift": float(change_frame[(change_frame.scope == "reference")
                  & (change_frame.method == "selected_direct")].mean_risk_change.abs().max())}
    (output / "budget_sensitivity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "direct_reference_headline"}, indent=2))


if __name__ == "__main__":
    main()
