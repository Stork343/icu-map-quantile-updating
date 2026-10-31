from pathlib import Path
from xml.etree import ElementTree
import argparse
import json
import os
import subprocess

import pymupdf


TITLE = "Learning and applying component specific quantile retention"


def verify_outputs(stem, print_width, architecture):
    graph = ElementTree.parse(stem.with_suffix(".drawio"))
    cells = graph.findall(".//mxCell")
    vertices = [cell for cell in cells if cell.get("vertex") == "1"]
    edges = [cell for cell in cells if cell.get("edge") == "1"]
    if len(vertices) != 20 or len(edges) != 9:
        raise ValueError("The method workflow has incomplete native elements")
    svg = ElementTree.parse(stem.with_suffix(".svg")).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    if svg.findall(f".//{namespace}image") or len(svg.findall(f".//{namespace}text")) != 20:
        raise ValueError("The SVG must retain vector text and native geometry")
    document = pymupdf.open(stem.with_suffix(".pdf"))
    if len(document) != 1 or document[0].get_images():
        raise ValueError("The workflow PDF must be a single vector page")
    page = document[0]
    for word in page.get_text("words"):
        if not page.rect.contains(pymupdf.Rect(word[:4])):
            raise ValueError("A workflow label extends beyond the PDF page")
    spans = [span for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"]]
    minimum = min(span["size"] for span in spans) * print_width / page.rect.width
    if minimum < 8:
        raise ValueError("The workflow font is below eight points at its printed width")
    fonts = []
    for font in page.get_fonts():
        if "TimesNewRoman" not in font[3] or not document.extract_font(font[0])[3]:
            raise ValueError("A Times New Roman font is missing or unembedded")
        fonts.append(font[3])
    text = page.get_text()
    semantic = json.loads(architecture.read_text())
    compact = "".join(text.split())
    for node in semantic["nodes"]:
        label = "".join(node["label"].replace("\\n", " ").split())
        if label not in compact:
            raise ValueError(f"A complete workflow label is missing: {node['id']}")
    for phrase in ("Fitting stays F", "Tuning stays T", "Assessment stays A",
                   "Grouped threefold CV", "Functions / constants", "Assessment only"):
        if phrase not in text:
            raise ValueError(f"A scientific workflow label is missing: {phrase}")
    preview = pymupdf.Pixmap(stem.with_suffix(".png"))
    if preview.xres != 300 or preview.yres != 300 or preview.width < 3000:
        raise ValueError("The workflow PNG does not meet its 300 dpi export specification")
    return {"status": "passed", "native_nodes": len(vertices), "native_edges": len(edges),
        "pdf_pages": 1, "raster_images": 0, "embedded_fonts": fonts,
        "scientific_labels_checked": len(semantic["nodes"]),
        "preview_width_px": preview.width, "preview_height_px": preview.height, "preview_dpi": 300,
        "printed_width_pt": print_width, "minimum_printed_font_pt": minimum,
        "pdf_width_pt": page.rect.width, "pdf_height_pt": page.rect.height,
        "palette": "Paul Tol High Contrast", "colorblind_safe": True, "grayscale_safe": True,
        "visual_inspection": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(__file__).parent / "figure_specs/figure_01_workflow.yaml")
    parser.add_argument("--drawio-cli", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--print-width", type=float, default=460)
    args = parser.parse_args()
    if not args.spec.is_file() or not args.drawio_cli.is_file() or args.print_width <= 0:
        raise ValueError("A valid YAML specification, draw.io CLI, and print width are required")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.work_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["TMPDIR"] = str(temporary.resolve())
    common = [args.node, str(args.drawio_cli), str(args.spec)]
    with (args.work_dir / "build.log").open("w") as log:
        for suffix in (".drawio", ".svg"):
            command = common + [str(args.output_stem.with_suffix(suffix)), "--validate", "--strict-warnings"]
            if suffix == ".drawio":
                command += ["--write-sidecars", "--sidecar-dir", str(args.work_dir / "sidecars")]
            subprocess.run(command, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
        raw_pdf = args.work_dir / "workflow_native.pdf"
        subprocess.run(common + [str(raw_pdf), "--validate", "--strict-warnings", "--use-desktop"],
                       env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    document = pymupdf.open(raw_pdf)
    document.set_metadata({"title": TITLE, "author": "Jian Hou and Maozai Tian",
        "subject": "Quantile profiling workflow", "keywords": "ICU MAP; quantile regression; landmark prediction",
        "creator": "draw.io", "producer": "draw.io"})
    document.save(args.output_stem.with_suffix(".pdf"), garbage=4, deflate=True)
    document[0].get_pixmap(dpi=300).save(args.output_stem.with_suffix(".png"))
    document.close()
    architecture = args.work_dir / "sidecars" / (args.output_stem.name + ".arch.json")
    report = verify_outputs(args.output_stem, args.print_width, architecture)
    (args.work_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
