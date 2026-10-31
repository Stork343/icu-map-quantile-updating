from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


TABLES = (
    "supp_ademp_design.tex", "supp_ademp_performance.tex",
    "supp_ademp_diagnostics.tex", "supp_clinical_evidence.tex",
    "supp_empirical_sensitivity.tex", "supp_candidate_evaluation.tex",
    "supp_fold_stability.tex",
)
FIGURES = ("supp_simulation_tuning_diagnostics", "supp_empirical_method_validation")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "config/analysis_config.json",
        "empirical_aggregate/enhanced_outcome_associations.csv",
        "supplement_aggregate/supp_fold_stability_source.csv",
        *[f"empirical_aggregate/{stem}.{ext}"
          for stem in ("introduction_overview", "enhanced_workflow_schematic")
          for ext in ("svg", "pdf", "png")],
        "code/figure_specs/introduction_overview.tex",
        "code/build_introduction_overview.py",
        "code/plot_method_illustration.py",
        "code/manuscript_plot_style.py",
        *[f"supplement_aggregate/{name}" for name in TABLES],
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing public package inputs: {missing}")
    config = json.loads((root / "config/analysis_config.json").read_text())
    if config["age"]["age_definition"] != "icu_proxy":
        raise ValueError("The package must use the approved ICU admission age proxy")
    if config["temporal_assignment"]["rule"] != "aligned_full_containment":
        raise ValueError("Unexpected temporal allocation in public configuration")
    if [config["landmark"][key] for key in (
        "minimum_total_observations", "minimum_index_observations",
        "minimum_later_observations",
    )] != [8, 4, 1]:
        raise ValueError("Unexpected landmark qualification in public configuration")

    verification_directory = root / "local_runs"
    verification_directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="icu-public-check-", dir=verification_directory) as tmp:
        dest = Path(tmp) / "package"
        # 复制公开程序和聚合输入。
        for name in ("code", "config", "empirical_aggregate", "source_sensitivity",
                     "validation_aggregate", "simulation_ademp_v2",
                     "transient_grid_sensitivity", "supplement_aggregate"):
            shutil.copytree(root / name, dest / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # 删除待生成文件，随后执行实际程序。
        for name in TABLES:
            (dest / "supplement_aggregate" / name).unlink()
        for stem in FIGURES:
            for ext in ("pdf", "png"):
                (dest / "supplement_aggregate" / f"{stem}.{ext}").unlink(missing_ok=True)
        input_paths = [f for name in ("config", "empirical_aggregate", "source_sensitivity",
                                     "validation_aggregate", "simulation_ademp_v2",
                                     "transient_grid_sensitivity")
                       for f in (dest / name).rglob("*")
                       if f.is_file() and f.suffix in (".csv", ".json")]
        input_paths.append(dest / "supplement_aggregate/supp_fold_stability_source.csv")
        input_contents = {f: f.read_bytes() for f in input_paths}
        env = dict(os.environ, MPLBACKEND="Agg")
        for script in ("generate_expanded_supplement_tables.py",
                       "plot_expanded_supplement_figures.py"):
            subprocess.run([sys.executable, str(dest / "code" / script)],
                           cwd=dest, env=env, check=True)
        for path, content in input_contents.items():
            if path.read_bytes() != content:
                raise AssertionError(f"Regeneration changed an input: {path.relative_to(dest)}")
        for name in TABLES:
            actual = (dest / "supplement_aggregate" / name).read_bytes()
            expected = (root / "supplement_aggregate" / name).read_bytes()
            if actual != expected:
                raise AssertionError(f"Regenerated table differs from supplied table: {name}")
        for stem in FIGURES:
            for ext, signature in (("pdf", b"%PDF-"), ("png", b"\x89PNG\r\n\x1a\n")):
                data = (dest / "supplement_aggregate" / f"{stem}.{ext}").read_bytes()
                if len(data) < 1000 or not data.startswith(signature):
                    raise AssertionError(f"Invalid regenerated figure: {stem}.{ext}")
        print("PASS: seven tables match; two PDF/PNG figures regenerated; "
              "configuration and editable figure sources present.")


if __name__ == "__main__":
    main()
