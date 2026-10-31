# Recorded numerical environments

`published_quantile.json` records the environment used for the 321-node published LQMM fits. The synthetic risk-inference calculations used the versions in `risk_inference.json`. These calculations used Python 3.12.1, NumPy 1.26.4, pandas 2.2.2, SciPy 1.14.0, and R 4.3.2.

`empirical_family.json` records the complete-family calculations, which used Python 3.12.9, NumPy 2.3.4, pandas 2.3.3, SciPy 1.16.2, OSQP 1.1.0, and R 4.3.2. Its solver settings and partition compute time are included in the same file.

The Python dependency pins in the package retain the archived environment and add OSQP 1.1.0. The R packages used were quantreg 6.1, data.table 1.17.8, and lqmm 1.5.8. Clinical reproduction requires the credentialed source records specified in the repository README. The public aggregate outputs and synthetic simulation files can be read directly.
