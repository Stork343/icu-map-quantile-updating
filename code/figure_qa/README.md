# Figure checks

The figure programs call these checks when producing the publication exports. `validate_figure.py` reads plotting source and its shared style module. `audit_pdf_text.py` checks font sizes and text bounds in the exported PDF. Panel geometry is checked by `audit_panel_alignment.py`, and `audit_figure_collisions.py` measures intersections in the PDF.

Run the source and PDF text checks from the extracted reproducibility directory:

```bash
python code/figure_qa/validate_figure.py code/plot_method_illustration.py \
  --style-source code/manuscript_plot_style.py --json
python code/figure_qa/audit_pdf_text.py \
  empirical_aggregate/enhanced_workflow_schematic.pdf --min-pt 5 --json
```

The figure output directories contain machine readable reports for the source values and panel coordinates. These numerical checks accompany the PDF, SVG and PNG exports.
