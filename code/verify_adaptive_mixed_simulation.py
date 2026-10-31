from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_adaptive_mixed_simulation import CONTRASTS, METHODS, SETTINGS, TAU, design_configuration, write_json


MATCHING_FIELDS = ("setting", "setting_index", "replicate", "seed", "n_stays", "reference_stays", "horizons")


def match_saved_inputs(work: Path, snapshot: Path) -> dict:
    with np.load(work / "synthetic_records.npz", allow_pickle=False) as final_data:
        with np.load(snapshot / "synthetic_records.npz", allow_pickle=False) as early_data:
            if set(final_data.files) != set(early_data.files):
                raise RuntimeError("The earlier and accepted datasets contain different arrays")
            for name in final_data.files:
                np.testing.assert_array_equal(final_data[name], early_data[name])
            arrays_checked = sorted(final_data.files)
    final_coefficients = pd.read_csv(work / "population_coefficients.csv").set_index("term")
    early_coefficients = pd.read_csv(snapshot / "population_coefficients.csv").set_index("term")
    terms = ["intercept", "age_z", "male", "urgent"]
    final_beta = final_coefficients.loc[terms, "estimate"].to_numpy()
    early_beta = early_coefficients.loc[terms, "estimate"].to_numpy()
    np.testing.assert_allclose(final_beta, early_beta, rtol=0, atol=1e-12)
    for frame in (final_coefficients, early_coefficients):
        if frame["relative_objective_gap"].max() > 1e-7 or frame["dual_stationarity"].max() > 1e-6:
            raise RuntimeError("A common-component fit lacks its strict certificate")
    final_configuration = json.loads((work.parents[2] / "design.json").read_text())
    early_configuration = json.loads((snapshot.parents[2] / "design.json").read_text())
    excluded = {"qp_settings", "profile_polish_settings"}
    final_science = {key: value for key, value in final_configuration["prespecified_protocol"].items() if key not in excluded}
    early_science = {key: value for key, value in early_configuration["prespecified_protocol"].items() if key not in excluded}
    if final_science != early_science or final_configuration["execution"] != early_configuration["execution"]:
        raise RuntimeError("The scientific configuration differs between attempts")
    return {"arrays_checked": arrays_checked, "all_arrays_exactly_equal": True,
            "allocation_arrays": ["fitting_indices", "tuning_indices", "assessment_indices"],
            "population_coefficients_exactly_equal": bool(np.array_equal(final_beta, early_beta)),
            "maximum_coefficient_difference": float(np.max(np.abs(final_beta - early_beta))),
            "coefficient_comparison_atol": 1e-12, "scientific_configuration_equal": True,
            "numerical_settings_excluded_from_scientific_comparison": sorted(excluded)}


def verify_attempt_resolution(work: Path, complete: dict) -> tuple[int, int]:
    failures = sorted(work.glob("failure*.json"))
    summary_path = work / "resolution_summary.json"
    if not summary_path.exists():
        if failures:
            raise RuntimeError(f"An unresolved numerical attempt exists at {work}")
        return 0, 0
    summary = json.loads(summary_path.read_text())
    count = int(summary["resolved_numerical_attempts"])
    if count < 1 or summary["unresolved_failures"] != 0 or summary["matching_fields"] != list(MATCHING_FIELDS):
        raise RuntimeError("Numerical attempt resolution metadata is inconsistent")
    if len(summary["failed_attempt_metadata"]) != count:
        raise RuntimeError("Numerical attempt resolution count is inconsistent")
    for failed in summary["failed_attempt_metadata"]:
        if any(failed[key] != complete[key] for key in MATCHING_FIELDS):
            raise RuntimeError("An earlier numerical attempt used different data or scientific configuration")
    accepted = summary["accepted_completion_metadata"]
    if any(accepted[key] != complete[key] for key in (*MATCHING_FIELDS, "derived_seeds", "diagnostics")):
        raise RuntimeError("Resolution metadata does not match the accepted computation")
    early_started = summary["early_started_metadata"]
    if any(early_started[key] != complete[key] for key in (*MATCHING_FIELDS, "derived_seeds")):
        raise RuntimeError("The earlier input snapshot uses different seeds or scientific configuration")
    data_matching = summary["data_matching"]
    if not data_matching["all_arrays_exactly_equal"] or not data_matching["scientific_configuration_equal"]:
        raise RuntimeError("Data or scientific configuration matching did not pass")
    if summary["audit_scope"] == "private_input_records":
        private = json.loads((work / "resolution.json").read_text())
        if len(failures) != count or private["failures"] != [json.loads(path.read_text()) for path in failures]:
            raise RuntimeError("The private original numerical attempt records differ from the audit")
        if private["accepted_completion"] != complete:
            raise RuntimeError("The accepted completion differs from its numerical attempt audit")
        snapshot = Path(private["early_input_snapshot"])
        if match_saved_inputs(work, snapshot) != data_matching:
            raise RuntimeError("The saved input comparison differs from the numerical attempt audit")
        return count, count
    if summary["audit_scope"] == "source_verified_public_metadata":
        if failures or (work / "resolution.json").exists() or summary["raw_failure_records_included"]:
            raise RuntimeError("The public numerical attempt audit contains undeclared private records")
        return count, 0
    raise RuntimeError("Unknown numerical attempt audit scope")


def verify_replicate(work: Path, expected: dict) -> dict:
    complete = json.loads((work / "complete.json").read_text())
    if complete["status"] != "complete" or any(complete[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Replicate completion does not match its scheduled configuration at {work}")
    started = json.loads((work / "started.json").read_text())
    if started["derived_seeds"] != complete["derived_seeds"]:
        raise RuntimeError(f"Replicate seeds changed between start and completion at {work}")
    sequence = np.random.SeedSequence([complete["seed"], complete["setting_index"], complete["replicate"]])
    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(3)]
    if complete["derived_seeds"] != dict(zip(("dataset", "split", "reference"), seeds)):
        raise RuntimeError(f"Derived seeds do not match the frozen master-seed rule at {work}")
    resolved_attempts, source_rechecks = verify_attempt_resolution(work, complete)
    coefficient = pd.read_csv(work / "population_coefficients.csv")
    if len(coefficient) != 4 or set(coefficient["solver"]) != {"br"}:
        raise RuntimeError(f"Unexpected common component fit at {work}")
    if not np.all(np.isfinite(coefficient.select_dtypes(include=["number"]).to_numpy())):
        raise RuntimeError(f"Nonfinite common-component certificate at {work}")
    if coefficient["relative_objective_gap"].max() > 1e-7 or coefficient["dual_stationarity"].max() > 1e-6:
        raise RuntimeError(f"Uncertified common component at {work}")
    diagnostics = complete["diagnostics"]
    if diagnostics["qp_fits"] <= 0 or not np.all(np.isfinite(list(diagnostics.values()))):
        raise RuntimeError(f"Missing or nonfinite profile certificates at {work}")
    if diagnostics["max_projected_kkt_violation"] > 5e-7 or diagnostics["max_stationarity_violation"] > 1e-10:
        raise RuntimeError(f"Uncertified profile at {work}")
    if diagnostics["max_relative_duality_gap"] > 1e-8 and diagnostics["max_duality_gap"] > 1e-7:
        raise RuntimeError(f"Profile duality gap exceeds declared tolerances at {work}")
    methods = pd.read_csv(work / "methods.csv")
    paired = pd.read_csv(work / "paired.csv")
    tuning = pd.read_csv(work / "tuning.csv")
    calibration = pd.read_csv(work / "calibration.csv")
    selections = pd.read_csv(work / "selections.csv")
    stays = pd.read_parquet(work / "assessment_stay_metrics.parquet")
    horizons = expected["horizons"]
    if len(methods) != len(horizons) * len(METHODS) * 2 or len(paired) != len(horizons) * len(CONTRASTS) * 2:
        raise RuntimeError(f"Missing selected-rule results at {work}")
    if len(tuning) != len(horizons) * 200 or len(calibration) != len(horizons) * 102:
        raise RuntimeError(f"The complete tuning grid was not retained at {work}")
    if calibration["max_equality_residual"].max() > 1e-6:
        raise RuntimeError(f"Calibration equality residual exceeds tolerance at {work}")
    if np.any(methods["p_lt"] > methods["p_le"] + 1e-13):
        raise RuntimeError(f"Strict and weak tail probabilities are reversed at {work}")
    if np.any((methods[["p_lt", "p_le"]] < 0) | (methods[["p_lt", "p_le"]] > 1)):
        raise RuntimeError(f"Tail probabilities lie outside their range at {work}")
    expected_violation = np.maximum.reduce([methods["p_lt"].to_numpy() - TAU,
                                           TAU - methods["p_le"].to_numpy(), np.zeros(len(methods))])
    np.testing.assert_allclose(methods["violation"], expected_violation, rtol=0, atol=1e-13)
    for horizon in horizons:
        local = tuning.loc[tuning["horizon"] == horizon]
        if local["name"].nunique() != 200:
            raise RuntimeError(f"Duplicate or missing tuning rules at {work} horizon {horizon}")
        selected = selections.loc[selections["horizon"] == horizon].set_index("method")
        np.testing.assert_allclose(selected.loc["expanded_selected", "tuning_loss"], local["tuning_loss"].min(), rtol=0, atol=1e-12)
        original = local.loc[local["family"] == "original", "tuning_loss"].min()
        np.testing.assert_allclose(selected.loc["original_selected", "tuning_loss"], original, rtol=0, atol=1e-12)
        if selected.loc["stump_original", "tuning_loss"] > original + 1e-12:
            raise RuntimeError(f"The stump omitted its constant original-family candidate at {work}")
        assessment = stays.loc[stays["horizon"] == horizon]
        observed = methods.loc[(methods["horizon"] == horizon) & (methods["scope"] == "assessment")].set_index("method")
        for method in METHODS:
            group = assessment.loc[assessment["method"] == method]
            if group["synthetic_stay_index"].nunique() != complete["n_assessment"] or len(group) != complete["n_assessment"]:
                raise RuntimeError(f"Assessment stays are missing or duplicated at {work}")
            for metric in ("loss", "p_lt", "p_le"):
                np.testing.assert_allclose(observed.loc[method, metric], group[metric].mean(), rtol=0, atol=1e-12)
        loss = assessment.pivot(index="synthetic_stay_index", columns="method", values="loss")
        for first, second in CONTRASTS:
            displayed = paired.loc[(paired["horizon"] == horizon) & (paired["scope"] == "assessment")
                                   & (paired["first"] == first) & (paired["second"] == second)]
            if len(displayed) != 1:
                raise RuntimeError(f"A paired comparison is missing or duplicated at {work}")
            difference = loss[first] - loss[second]
            np.testing.assert_allclose(displayed["difference"].iat[0], difference.mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(displayed["within_sample_se"].iat[0], difference.std(ddof=1) / np.sqrt(len(difference)),
                                       rtol=0, atol=1e-12)
    return {"setting": expected["setting"], "replicate": expected["replicate"], "qp_fits": diagnostics["qp_fits"],
            "runtime_seconds": complete["runtime_seconds"], "max_projected_kkt_violation": diagnostics["max_projected_kkt_violation"],
            "max_relative_duality_gap": diagnostics["max_relative_duality_gap"],
            "resolved_numerical_attempts": resolved_attempts, "resolved_attempt_source_input_rechecks": source_rechecks}


def verify(output: Path) -> dict:
    configuration = json.loads((output / "design.json").read_text())
    if configuration["prespecified_protocol"] != design_configuration():
        raise RuntimeError("The stored scientific protocol differs from the current prespecified protocol")
    execution = configuration["execution"]
    rows = []
    for setting in execution["settings"]:
        for replicate in range(1, execution["replicates"] + 1):
            work = output / "replicates" / setting / f"rep_{replicate:04d}"
            expected = {"setting": setting, "setting_index": SETTINGS.index(setting), "replicate": replicate,
                        "seed": execution["seed"], "n_stays": execution["n_stays"],
                        "reference_stays": execution["reference_stays"], "horizons": execution["horizons"]}
            rows.append(verify_replicate(work, expected))
    methods = pd.read_csv(output / "methods_replicates.csv")
    paired = pd.read_csv(output / "paired_replicates.csv")
    summary = pd.read_csv(output / "method_summary.csv")
    paired_summary = pd.read_csv(output / "paired_summary.csv")
    for row in summary.itertuples(index=False):
        group = methods.loc[(methods["setting"] == row.setting) & (methods["horizon"] == row.horizon)
                            & (methods["method"] == row.method) & (methods["scope"] == row.scope)]
        if len(group) != execution["replicates"]:
            raise RuntimeError("Method summary excluded a replicate")
        for metric in ("loss", "p_lt", "p_le", "violation"):
            np.testing.assert_allclose(getattr(row, metric), group[metric].mean(), rtol=0, atol=1e-12)
            if len(group) > 1:
                np.testing.assert_allclose(getattr(row, f"{metric}_mcse"), group[metric].std(ddof=1) / np.sqrt(len(group)), rtol=0, atol=1e-12)
        np.testing.assert_allclose(row.mean_bracket_violation, max(row.p_lt - TAU, TAU - row.p_le, 0), rtol=0, atol=1e-12)
    for row in paired_summary.itertuples(index=False):
        group = paired.loc[(paired["setting"] == row.setting) & (paired["horizon"] == row.horizon)
                           & (paired["first"] == row.first) & (paired["second"] == row.second) & (paired["scope"] == row.scope)]
        if len(group) != execution["replicates"]:
            raise RuntimeError("Paired summary excluded a replicate")
        np.testing.assert_allclose(row.mean_difference, group["difference"].mean(), rtol=0, atol=1e-12)
        if len(group) > 1:
            np.testing.assert_allclose(row.mcse, group["difference"].std(ddof=1) / np.sqrt(len(group)), rtol=0, atol=1e-12)
    frequencies = pd.read_csv(output / "selection_frequencies.csv")
    for _, group in frequencies.groupby(["setting", "horizon", "method"]):
        if group["count"].sum() != execution["replicates"]:
            raise RuntimeError("Selection frequencies excluded a replicate")
        np.testing.assert_allclose(group["frequency"].sum(), 1, rtol=0, atol=1e-12)
    pd.DataFrame(rows).to_csv(output / "verification_replicates.csv", index=False)
    resolved_attempts = int(sum(row["resolved_numerical_attempts"] for row in rows))
    source_rechecks = int(sum(row["resolved_attempt_source_input_rechecks"] for row in rows))
    manifest = json.loads((output / "completion_manifest.json").read_text())
    if manifest["expected_replicates"] != len(rows) or manifest["completed_replicates"] != len(rows):
        raise RuntimeError("The completion manifest has an incorrect replicate denominator")
    manifest.pop("failures", None)
    manifest["resolved_numerical_attempts"], manifest["unresolved_failures"] = resolved_attempts, 0
    write_json(output / "completion_manifest.json", manifest)
    result = {"status": "passed", "replicates_checked": len(rows), "qp_fits_checked": int(sum(row["qp_fits"] for row in rows)),
              "resolved_numerical_attempts": resolved_attempts, "unresolved_failures": 0,
              "resolved_attempt_source_input_rechecks": source_rechecks,
              "checks": ["frozen configuration and seeds", "complete candidate menu", "actual R fitting certificates",
                         "profile certificates", "probability mass brackets", "stay-paired assessment calculations",
                         "all-replicate Monte Carlo summaries", "selection frequency denominators",
                         "resolved numerical attempt provenance"],
              "visual_inspection": False, "forecast_clipping": False}
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
