# Fitted rules for the assessment size experiment

The `risk_inference/replicates/` directory supplies independently fitted coefficients and frozen prediction rules for 200 replicates in each of the large dense, omitted common time pattern and informative observation count settings. Each replicate includes its coefficients, prediction rules and paired interval results. Completion records accompany the numerical files.

These 600 fitted rule sets are the inputs to `code/run_frozen_risk_coverage.py`. The command in `analysis_stability/README.md` reproduces the assessment size experiment from this package. `risk_inference/frozen_input_reconstruction_validation.json` records the verification against all 1,800 original paired interval results.

The original `code/run_risk_inference_simulation.py` generates fitted models for all fourteen study mechanisms. Its design and aggregate outputs are supplied in `risk_inference/`.
