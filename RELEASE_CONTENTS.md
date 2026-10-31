# Complete reproducibility materials

The repository contains the analysis programs and aggregate results for version 1.1.0. The complete `reproducibility_code.zip` release asset also includes the simulation replicate directories and retained fitted inputs used in the reported numerical checks.

Download and extract the complete package with GitHub CLI:

```bash
mkdir -p local_runs/complete_archive
gh release download v1.1.0 --repo Stork343/icu-map-quantile-updating --pattern reproducibility_code.zip --dir local_runs/complete_archive
python -m zipfile -e local_runs/complete_archive/reproducibility_code.zip local_runs/complete_archive
```

Use the commands in the extracted package's `README.md` for the complete simulation checks. The repository's aggregate integration check runs with:

```bash
python code/verify_public_package.py
```

MIMIC-IV reproduction requires credentialed access to the original data. Public clinical files contain aggregate quantities.
