#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marker-pdf>=1.10.2",
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
# ]
# ///
"""
marker-pdf: Convert PDF to Markdown
https://github.com/datalab-to/marker

Uses Surya OCR for text detection + recognition, layout analysis, and
table/equation extraction. Supports all languages.
"""

import sys
import time
import json
from pathlib import Path


def convert(pdf_path: str, output_dir: str):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[marker-pdf] Converting: {pdf_path.name}")
    print(f"[marker-pdf] Output: {output_dir}")

    # Load models (surya OCR, layout detection, table recognition)
    print("[marker-pdf] Loading models...")
    t0 = time.time()
    models = create_model_dict()
    model_time = time.time() - t0
    print(f"[marker-pdf] Models loaded in {model_time:.1f}s")

    # Convert
    print("[marker-pdf] Converting PDF...")
    t0 = time.time()
    converter = PdfConverter(artifact_dict=models)
    rendered = converter(str(pdf_path))
    text, ext, images = text_from_rendered(rendered)
    convert_time = time.time() - t0
    print(f"[marker-pdf] Conversion done in {convert_time:.1f}s")

    # Save markdown
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    # Save images
    for img_name, img in images.items():
        img_path = output_dir / img_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)

    # Save timing metadata
    total_time = model_time + convert_time
    meta = {
        "tool": "marker-pdf",
        "input": pdf_path.name,
        "model_load_time": round(model_time, 2),
        "conversion_time": round(convert_time, 2),
        "total_time": round(total_time, 2),
        "output_chars": len(text),
        "output_words": len(text.split()),
        "images_extracted": len(images),
    }
    (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))

    print(f"[marker-pdf] Output: {md_path}")
    print(
        f"[marker-pdf] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run run_marker.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/marker"
    convert(pdf, out)
