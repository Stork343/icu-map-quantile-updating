import argparse
import json
import os
import pickle
from pathlib import Path
from statistics import NormalDist
import sys
import subprocess

import numpy as np
import pandas as pd


INPUTS = None
OUTPUT = None

HORIZONS = (4, 8, 12)
PARTITIONS = tuple(f"fold_{number}" for number in range(1, 6))
CONTRASTS = (
    ("continuous_direct", "constant_diagonal_direct"),
    ("continuous_affine", "constant_diagonal_affine"),
    ("continuous_direct", "calibrated_q10"),
    ("continuous_affine", "calibrated_q10"),
    ("continuous_affine", "stump_original"),
)


def paired_summary(difference):
    difference = np.asarray(difference, dtype=float)
    assert len(difference) > 1 and np.isfinite(difference).all()
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    leave_one = (difference.sum() - difference) / (len(difference) - 1)
    return {
        "n_stays": len(difference), "mean_difference": mean, "se": se,
        "pointwise_ci95_low": mean - 1.96 * se,
        "pointwise_ci95_high": mean + 1.96 * se,
        "candidate_better_fraction": float(np.mean(difference < 0)),
        "median_difference": float(np.median(difference)),
        "minimum_leave_one_stay_out_mean": float(leave_one.min()),
        "maximum_leave_one_stay_out_mean": float(leave_one.max()),
    }


def load_records(partition, horizon):
    path = INPUTS.component_clinical_root / partition / f"assessment_h{horizon}.csv"
    frame = pd.read_csv(path)
    assert not frame.global_stay_index.duplicated().any()
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    assert (frame.late_obs >= 1).all()
    return frame


def compare_saved_summaries(frames):
    saved = pd.read_csv(INPUTS.component_public_root / "clinical_model_summary.csv")
    complete_frames = dict(frames)
    for horizon in HORIZONS:
        complete_frames[("nested_pooled", horizon)] = pd.concat(
            [frames[(partition, horizon)] for partition in PARTITIONS], ignore_index=True)
    rebuilt = []
    for (partition, horizon), frame in complete_frames.items():
        scope = "temporal_2017_2022" if partition == "temporal" else partition
        for column in frame:
            if not column.startswith("loss_"):
                continue
            model = column.removeprefix("loss_")
            lower = frame[f"p_lt_{model}"].mean()
            upper = frame[f"p_le_{model}"].mean()
            row = saved[(saved.scope == scope) & (saved.horizon_hours == horizon) & (saved.model == model)]
            assert len(row) == 1
            actual = np.array([len(frame), frame[column].mean(), lower, upper,
                               max(lower - .1, .1 - upper, 0)])
            expected = row[["n_stays", "mean_loss", "p_lt", "p_le", "bracket_violation"]].iloc[0].to_numpy(float)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
            rebuilt.append((partition, horizon, model))
    return len(rebuilt)


def nested_stability(frames):
    rows, dropped, pooled_rows = [], [], []
    for horizon in HORIZONS:
        ids = np.concatenate([frames[(partition, horizon)].global_stay_index for partition in PARTITIONS])
        assert len(np.unique(ids)) == len(ids)
        assert len(ids) == {4: 59969, 8: 60241, 12: 60310}[horizon]
        for first, second in CONTRASTS:
            differences = []
            for partition in PARTITIONS:
                frame = frames[(partition, horizon)]
                difference = frame[f"loss_{first}"].to_numpy() - frame[f"loss_{second}"].to_numpy()
                differences.append(difference)
                rows.append({"partition": partition, "horizon_hours": horizon,
                             "first": first, "second": second,
                             "interval_scope": "conditional paired within an assessment fold",
                             **paired_summary(difference)})
            combined = np.concatenate(differences)
            pooled_rows.append({"horizon_hours": horizon, "first": first, "second": second,
                "n_stays": len(combined), "mean_difference": float(combined.mean()),
                "folds_with_negative_difference": sum(float(value.mean()) < 0 for value in differences),
                "minimum_fold_mean": min(float(value.mean()) for value in differences),
                "maximum_fold_mean": max(float(value.mean()) for value in differences),
                "minimum_omitting_fold_mean": min(float(np.concatenate(
                    [value for position, value in enumerate(differences) if position != index]).mean())
                    for index in range(len(PARTITIONS))),
                "maximum_omitting_fold_mean": max(float(np.concatenate(
                    [value for position, value in enumerate(differences) if position != index]).mean())
                    for index in range(len(PARTITIONS))),
                "summary_scope": "descriptive out-of-fold aggregation without procedure confidence intervals"})
            for index, partition in enumerate(PARTITIONS):
                retained = np.concatenate([value for position, value in enumerate(differences) if position != index])
                dropped.append({"omitted_assessment_fold": partition, "horizon_hours": horizon,
                                "first": first, "second": second, "n_retained_stays": len(retained),
                                "mean_difference": float(retained.mean()),
                                "summary_scope": "saved forecasts retained without retraining"})
    pd.DataFrame(rows).to_csv(OUTPUT / "clinical_nested_fold_pairs.csv", index=False)
    pd.DataFrame(dropped).to_csv(OUTPUT / "clinical_nested_leave_one_fold_out.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUTPUT / "clinical_nested_stability_summary.csv", index=False)
    return pooled_rows


def temporal_intervals(frames):
    critical = NormalDist().inv_cdf(1 - .05 / (2 * len(HORIZONS)))
    rows = []
    for first, second in CONTRASTS:
        for horizon in HORIZONS:
            frame = frames[("temporal", horizon)]
            difference = frame[f"loss_{first}"].to_numpy() - frame[f"loss_{second}"].to_numpy()
            summary = paired_summary(difference)
            rows.append({"horizon_hours": horizon, "first": first, "second": second,
                         **summary, "joint_ci95_low": summary["mean_difference"] - critical * summary["se"],
                         "joint_ci95_high": summary["mean_difference"] + critical * summary["se"],
                         "joint_family": "three horizons within the stated contrast",
                         "joint_method": "two-sided Bonferroni normal", "critical_value": critical,
                         "interval_scope": "conditional risk difference of frozen fitted rules"})
    pd.DataFrame(rows).to_csv(OUTPUT / "clinical_temporal_joint_intervals.csv", index=False)
    return rows


def clinical_design(cached):
    dataset, stays = cached["dataset"], cached["stays"]
    cohort = stays.set_index("stay_id").loc[dataset["cluster_ids"]].reset_index()
    assert len(cohort) == 60310 and cohort.subject_id.nunique() == 60310
    assert not cohort.stay_id.duplicated().any()
    report = []
    for partition in PARTITIONS + ("temporal",):
        path = INPUTS.adaptive_clinical_root / f"horizon_12/{'nested/' if partition != 'temporal' else ''}{partition}/private_partition_config.json"
        config = json.loads(path.read_text())
        arrays = [np.asarray(config[key], dtype=int) for key in ("fit_indices", "tune_indices", "assessment_indices")]
        assert len(np.unique(np.concatenate(arrays))) == sum(map(len, arrays))
        ages = cohort.age.iloc[arrays[0]].to_numpy(float)
        np.testing.assert_allclose(ages.mean(), config["age_standardization"]["age_mean_training_split"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ages.std(ddof=1), config["age_standardization"]["age_sd_training_split"], rtol=0, atol=1e-12)
        for horizon in HORIZONS:
            other_path = INPUTS.adaptive_clinical_root / f"horizon_{horizon}/{'nested/' if partition != 'temporal' else ''}{partition}/private_partition_config.json"
            other = json.loads(other_path.read_text())
            for key, indices in zip(("fit_indices", "tune_indices", "assessment_indices"), arrays):
                np.testing.assert_array_equal(other[key], indices)
        report.append({"partition": partition, "fitting_stays": len(arrays[0]),
                       "tuning_stays": len(arrays[1]), "assessment_stays": len(arrays[2]),
                       "disjoint_allocations": True, "fitting_only_age_standardization_replayed": True,
                       "same_assignments_across_horizons": True})
    pd.DataFrame(report).to_csv(OUTPUT / "clinical_partition_checks.csv", index=False)
    return cohort, report


def lqmm_influence(frames, cached, cohort):
    current = frames[("temporal", 12)]
    references = {
        31: INPUTS.slope31_root / "temporal/assessment_horizon_12.csv",
        61: INPUTS.slope61_root / "temporal/assessment_horizon_12.csv",
    }
    saved_pairs = pd.read_csv(INPUTS.adaptive_public_root / "clinical_temporal_paired_comparisons.csv")
    saved_precision = pd.read_csv(INPUTS.adaptive_public_root / "clinical_quadrature61_temporal_paired_comparisons.csv")
    selected_indices = current.global_stay_index.to_numpy(int)
    local = cohort.iloc[selected_indices].copy().reset_index(drop=True)
    all_counts = np.asarray([np.sum(np.asarray(times) <= 12)
                             for times in cached["dataset"]["t_list"]])
    count_threshold = float(np.median(all_counts))
    assert count_threshold == 13.0
    counts, quantiles = [], []
    for index in selected_indices:
        times = np.asarray(cached["dataset"]["t_list"][index])
        early = np.asarray(cached["dataset"]["y_list"][index])[times <= 12]
        counts.append(len(early))
        quantiles.append(np.sort(early)[int(np.ceil(len(early) * .1)) - 1])
    local["index_count"] = counts
    local["index_q10"] = quantiles
    labels = {
        "age": np.where(local.age < 65, "under_65", "65_or_older"),
        "sex": local.gender.to_numpy(),
        "admission": np.where(local.emergency_or_urgent == 1, "emergency_or_urgent", "other_admission"),
        "index_count": np.where(local.index_count < count_threshold, "fewer_than_13", "13_or_more"),
        "index_q10_quintile": pd.qcut(local.index_q10, q=5, labels=False, duplicates="drop").to_numpy(),
    }
    influence_rows, group_rows, deletion_rows, interaction_rows = [], [], [], []
    for nodes, path in references.items():
        reference = pd.read_csv(path)
        assert not reference.global_stay_index.duplicated().any()
        reference = reference.set_index("global_stay_index").loc[selected_indices]
        np.testing.assert_array_equal(reference.late_obs, current.late_obs)
        assert len(reference) == len(current) == 22005
        for model in ("random_slope_lqmm", "calibrated_random_slope_lqmm"):
            difference = current.loss_vector_direct.to_numpy() - reference[f"loss_{model}"].to_numpy()
            row = {"nodes_per_dimension": nodes, "first": "vector_direct", "second": model,
                   "horizon_hours": 12, **paired_summary(difference)}
            original = saved_pairs[(saved_pairs.horizon_hours == 12) & (saved_pairs.candidate == "vector_direct") & (saved_pairs.reference == model)]
            if nodes == 61:
                original = saved_precision[(saved_precision.horizon_hours == 12) & (saved_precision.reference == model + "_61")]
            assert len(original) == 1
            np.testing.assert_allclose(row["mean_difference"], original.mean_difference.iloc[0], rtol=0, atol=2e-12)
            influence_rows.append(row)
            high = local.index_count.to_numpy() >= count_threshold
            high_mean, low_mean = float(difference[high].mean()), float(difference[~high].mean())
            interaction = high_mean - low_mean
            interaction_se = float(np.sqrt(difference[high].var(ddof=1) / high.sum()
                                           + difference[~high].var(ddof=1) / (~high).sum()))
            interaction_rows.append({"nodes_per_dimension": nodes, "first": "vector_direct", "second": model,
                "horizon_hours": 12, "count_threshold": count_threshold,
                "n_high_count_stays": int(high.sum()), "n_low_count_stays": int((~high).sum()),
                "high_count_mean_difference": high_mean, "low_count_mean_difference": low_mean,
                "mean_difference": interaction, "se": interaction_se,
                "ci95_low": interaction - 1.96 * interaction_se,
                "ci95_high": interaction + 1.96 * interaction_se,
                "comparison_family": "index_count_interactions_exploratory",
                "difference_definition": "high count paired contrast minus low count paired contrast"})
            order = np.argsort(difference)
            for fraction in (.001, .01, .05):
                count = int(np.ceil(len(difference) * fraction))
                for side, retained in (("most_favorable", order[count:]), ("least_favorable", order[:-count])):
                    deletion_rows.append({"nodes_per_dimension": nodes, "second": model,
                        "excluded_fraction": fraction, "excluded_stays": count, "excluded_side": side,
                        "n_retained_stays": len(retained), "retained_mean_difference": float(difference[retained].mean()),
                        "summary_scope": "outcome-ranked deletion diagnostic changes target composition"})
            for dimension, values in labels.items():
                for label in np.unique(values):
                    mask = values == label
                    group_rows.append({"nodes_per_dimension": nodes, "second": model,
                        "grouping": dimension, "group": str(label), **paired_summary(difference[mask]),
                        "group_mean_difference": float(difference[mask].mean()),
                        "n_remaining_stays": int((~mask).sum()),
                        "mean_omitting_group": float(difference[~mask].mean()),
                        "summary_scope": "descriptive subgroup and saved-rule group omission"})
    pd.DataFrame(influence_rows).to_csv(OUTPUT / "clinical_lqmm_influence.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT / "clinical_lqmm_subgroups.csv", index=False)
    pd.DataFrame(deletion_rows).to_csv(OUTPUT / "clinical_lqmm_outcome_rank_deletions.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(OUTPUT / "clinical_lqmm_count_interactions.csv", index=False)
    return influence_rows, group_rows, {"threshold": count_threshold,
        "origin": "median number of index observations among all 60310 eligible stays",
        "inputs": "all retained observation times at or before the 12 hour landmark",
        "temporal_cohort_median": float(local.index_count.median()),
        "analysis_role": "additional descriptive grouping in this audit"}


def multiplicity_audit():
    sources = (
        ("original_complete_family_temporal", INPUTS.original_complete_paired,
         "complete saved 12 hour original-family comparisons including the published random-intercept comparator"),
        ("adaptive_horizon_temporal", INPUTS.adaptive_public_root / "clinical_temporal_paired_comparisons.csv",
         "complete saved adaptive-family candidates against five references across three horizons"),
        ("component_learning_temporal", INPUTS.component_public_root / "clinical_temporal_paired.csv",
         "complete saved learned and constant component comparisons against six original references and matched branches across three horizons"),
        ("quadrature61_temporal", INPUTS.adaptive_public_root / "clinical_quadrature61_temporal_paired_comparisons.csv",
         "all six saved quadrature sensitivity comparisons across three horizons and two random-slope forms"),
        ("index_count_interactions_exploratory", OUTPUT / "clinical_lqmm_count_interactions.csv",
         "four additional count interactions from two quadrature orders and two random-slope forms"),
    )
    tables, metadata = [], []
    for family, path, definition in sources:
        frame = pd.read_csv(path).rename(columns={"candidate": "first", "reference": "second"})
        assert frame[["mean_difference", "se"]].notna().all().all()
        if "horizon_hours" not in frame:
            frame["horizon_hours"] = 12
        frame["comparison_family"] = family
        frame["source_role"] = family
        metadata.append({"comparison_family": family, "source_role": family,
                         "comparisons": len(frame), "definition": definition})
        tables.append(frame)
    original = OUTPUT / "clinical_all_family_pointwise.csv"
    destination = OUTPUT / "clinical_all_family_holm.csv"
    pd.concat(tables, ignore_index=True).to_csv(original, index=False)
    environment = os.environ.copy()
    environment["TMPDIR"] = "."
    subprocess.run(["Rscript", "--vanilla", str(Path(__file__).with_name("adjust_clinical_families.R")),
                    str(original), str(destination)],
                   check=True, env=environment, cwd=OUTPUT)
    result = pd.read_csv(destination)
    target = result[(result.comparison_family == "adaptive_horizon_temporal")
                    & (result.horizon_hours == 12) & (result["first"] == "vector_direct")
                    & (result["second"] == "random_slope_lqmm")]
    assert len(target) == 1
    (OUTPUT / "clinical_comparison_families.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cleaned = target.astype(object).where(pd.notna(target), None)
    return {"families": metadata, "profile_minus_random_slope_12h": cleaned.to_dict("records")[0]}


def main():
    global INPUTS, OUTPUT
    parser = argparse.ArgumentParser(description="Audit saved eligible-stay clinical comparisons and frozen-rule stability")
    parser.add_argument("--component-clinical-root", type=Path, required=True)
    parser.add_argument("--adaptive-clinical-root", type=Path, required=True)
    parser.add_argument("--cached-dataset", type=Path, required=True)
    parser.add_argument("--component-public-root", type=Path, required=True)
    parser.add_argument("--adaptive-public-root", type=Path, required=True)
    parser.add_argument("--original-complete-paired", type=Path, required=True)
    parser.add_argument("--slope31-root", type=Path, required=True)
    parser.add_argument("--slope61-root", type=Path, required=True)
    parser.add_argument("--project-code", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    INPUTS = parser.parse_args()
    for key, value in vars(INPUTS).items():
        setattr(INPUTS, key, value.resolve())
    OUTPUT = INPUTS.output_root
    sys.path.insert(0, str(INPUTS.project_code))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {(partition, horizon): load_records(partition, horizon)
              for partition in PARTITIONS + ("temporal",) for horizon in HORIZONS}
    summary_cells = compare_saved_summaries(frames)
    stability = nested_stability(frames)
    temporal = temporal_intervals(frames)
    with INPUTS.cached_dataset.open("rb") as stream:
        cached = pickle.load(stream)
    cohort, partitions = clinical_design(cached)
    lqmm, groups, count_rule = lqmm_influence(frames, cached, cohort)
    multiplicity = multiplicity_audit()
    diagnostics = pd.read_csv(INPUTS.component_public_root / "clinical_optimization_diagnostics.csv")
    full_tuning = diagnostics[diagnostics.phase == "full_tuning"]
    report = {"status": "passed", "unique_subjects": int(cohort.subject_id.nunique()),
        "source_summary_cells_recomputed": summary_cells,
        "partition_checks": partitions,
        "nested_stability": stability,
        "temporal_joint_intervals": temporal,
        "lqmm_influence": lqmm,
        "count_grouping_rule": count_rule,
        "multiplicity_audit": multiplicity,
        "lqmm_subgroup_omission_range": [min(row["mean_omitting_group"] for row in groups),
                                         max(row["mean_omitting_group"] for row in groups)],
        "learning_optimizer_diagnostics": {"all_saved_fits": len(diagnostics),
            "optimizer_stops_without_success": int((~diagnostics.optimizer_success).sum()),
            "full_tuning_fits": len(full_tuning),
            "all_selected_full_tuning_optimizers_successful": bool(full_tuning.optimizer_success.all()),
            "maximum_selected_projected_gradient": float(full_tuning.projected_gradient_norm.max())},
        "uncertainty_policy": "No pooled OOF interval is presented as an interval for a retrained procedure. Temporal joint intervals cover three horizons within each contrast by Bonferroni.",
        "selection_policy": "All five stated contrasts, all folds, all horizons, both quadrature orders and both direct/calibrated LQMM forms are retained."}
    (OUTPUT / "clinical_analysis_sufficiency.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    protected = {"subject_id", "hadm_id", "stay_id", "global_stay_index", "prediction", "predictions"}
    manifest = {"status": "passed", "files": {}, "clinical_rows_exported": 0,
                "protected_identifier_columns": 0, "input_access": "credentialed clinical inputs supplied by command-line paths",
                "independent_unit": "eligible first adult ICU stay", "unique_subjects": 60310}
    for path in sorted(OUTPUT.glob("*.csv")):
        frame = pd.read_csv(path)
        assert not protected.intersection(frame.columns), path.name
        manifest["files"][path.name] = {"aggregate_rows": len(frame), "columns": list(frame.columns)}
    for path in sorted(OUTPUT.glob("*.json")):
        content = path.read_text()
        assert "/Users/" not in content and "/tmp/" not in content, path.name
        json.loads(content)
    (OUTPUT / "clinical_public_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "summary_cells_recomputed": summary_cells,
                      "nested_direct_vs_constant": [row for row in stability if row["second"] == "constant_diagonal_direct"],
                      "temporal_direct_vs_constant": [row for row in temporal if row["second"] == "constant_diagonal_direct"],
                      "lqmm_influence": lqmm,
                      "lqmm_subgroup_omission_range": report["lqmm_subgroup_omission_range"]}, indent=2))


if __name__ == "__main__":
    main()
