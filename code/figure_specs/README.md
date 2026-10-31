# Research overview and component illustration

## Figure 1

`introduction_overview.tex` contains the editable TikZ drawing. `build_introduction_overview.py` combines the drawing with the supplied numerical records and exports a vector PDF, SVG and 600 dpi PNG at 183 by 128 mm. LuaLaTeX is required.

```bash
python code/build_introduction_overview.py \
  --source code/figure_specs/introduction_overview.tex \
  --data-dir empirical_aggregate/introduction_figure_source \
  --output-dir local_runs/figure_01
```

The numerical panels use informative recovery replicate 4 under seed `20261022`, with synthetic assessment records 597 and 148. The drawing displays their early observations and fixed 12 hour forecasts. Its feature band reports the reading counts, observation time variances, and residual q10 changes calculated from those records. The two penalty controls denote the jointly fitted level and slope components.

The source records can be regenerated with:

```bash
NUMBA_NUM_THREADS=1 python code/prepare_introduction_figure_data.py \
  --source component_learning/simulation \
  --output local_runs/figure_01_data
```

## Figure 2

`plot_method_illustration.py` draws the six quantitative panels from informative recovery replicate 1 under seed `20261022`. The synthetic assessment records are 22 and 27. The source CSV files and fitted parameters are supplied in `empirical_aggregate/method_figure_source/`.

```bash
python code/plot_method_illustration.py \
  --data-dir empirical_aggregate/method_figure_source \
  --output-dir local_runs/figure_02
```

To regenerate the source data from the frozen simulation output, run:

```bash
NUMBA_NUM_THREADS=1 python code/prepare_method_figure_data.py \
  --source component_learning/simulation \
  --output local_runs/figure_02_data
```

Figures 2, 5 and 6 import `set_plot_style()` and `add_panel_label()` from `code/manuscript_plot_style.py`. This module also supplies their series colors, line widths and horizontal grid settings. Figure 2 uses blue circles of 3.2 pt and red squares of 2.8 pt, with marker edge widths of 0.5 pt. Its vector page measures 183 by 122 mm. Matplotlib renders the curves and mathematical labels, and svgutils preserves editable panel groups and text.

The figure programs compare displayed coordinates with their source values and report PDF dimensions, text bounds and panel geometry alongside the exports. The checks in `code/figure_qa/` are distributed with the source package.
