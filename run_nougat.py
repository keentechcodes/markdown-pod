#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nougat-ocr>=0.1.17",
#     "albumentations>=1.0.0,<1.4.4",
#     "transformers>=4.25.1,<=4.38.2",
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
#     "pypdf>=3.1.0",
#     "pypdfium2>=4.0.0,<5",
# ]
# ///
"""
Nougat: Neural Optical Understanding for Academic Documents
https://github.com/facebookresearch/nougat

Encoder-decoder transformer trained on academic papers.
Outputs Mathpix Markdown (.mmd) with LaTeX math and tables.
Model: 0.1.0-base (~1.5GB)

Uses the Python API directly (not CLI) to avoid '__main__' module issues.
nougat-ocr's own setup.py pins transformers<=4.38.2 and albumentations<=1.4.24,
so we let it resolve its own dependency versions.
"""

import sys
import os
import re
import time
import json
import logging
from functools import partial
from pathlib import Path

# Suppress albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nougat-runner")


def convert(pdf_path: str, output_dir: str):
    import torch
    from torch.utils.data import ConcatDataset

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nougat] Converting: {pdf_path.name}")
    print(f"[nougat] Output: {output_dir}")
    print(f"[nougat] CUDA available: {torch.cuda.is_available()}")

    t0 = time.time()

    try:
        from nougat import NougatModel
        from nougat.utils.dataset import LazyDataset
        from nougat.utils.device import move_to_device, default_batch_size
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.postprocessing import markdown_compatible
    except Exception as e:
        error_msg = f"Failed to import nougat: {e}"
        print(f"[nougat] ERROR: {error_msg}")
        meta = {
            "tool": "nougat",
            "input": pdf_path.name,
            "total_time": round(time.time() - t0, 2),
            "error": error_msg,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    # Download and load model
    model_tag = "0.1.0-base"
    print(f"[nougat] Loading model {model_tag} (downloads ~1.5GB on first run)...")

    try:
        checkpoint = get_checkpoint(None, model_tag=model_tag)
        model = NougatModel.from_pretrained(checkpoint)
        use_cuda = torch.cuda.is_available()
        model = move_to_device(model, bf16=use_cuda, cuda=use_cuda)
        model.eval()
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"[nougat] ERROR: {error_msg}")
        meta = {
            "tool": "nougat",
            "input": pdf_path.name,
            "total_time": round(time.time() - t0, 2),
            "error": error_msg,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    model_time = time.time() - t0
    print(f"[nougat] Model loaded in {model_time:.1f}s")

    # Prepare dataset from PDF
    print("[nougat] Processing PDF pages...")
    t1 = time.time()

    try:
        dataset = LazyDataset(
            pdf_path,
            partial(model.encoder.prepare_input, random_padding=False),
            None,  # all pages
        )
    except Exception as e:
        error_msg = f"Failed to read PDF: {e}"
        print(f"[nougat] ERROR: {error_msg}")
        meta = {
            "tool": "nougat",
            "input": pdf_path.name,
            "model": model_tag,
            "model_load_time": round(model_time, 2),
            "total_time": round(time.time() - t0, 2),
            "error": error_msg,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    batch_size = default_batch_size() if torch.cuda.is_available() else 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    # Run inference
    predictions = []
    page_num = 0

    for sample, is_last_page in dataloader:
        model_output = model.inference(
            image_tensors=sample,
            early_stopping=False,  # --no-skipping
        )
        for j, output in enumerate(model_output["predictions"]):
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            else:
                output = markdown_compatible(output)
                predictions.append(output)
            print(f"[nougat]   Page {page_num} done")

    convert_time = time.time() - t1

    # Combine output
    text = "".join(predictions).strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        error_msg = "No text extracted from PDF"
        print(f"[nougat] WARNING: {error_msg}")

    # Save as .mmd and .md
    mmd_path = output_dir / f"{pdf_path.stem}.mmd"
    mmd_path.write_text(text, encoding="utf-8")
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    total_time = model_time + convert_time

    # Save timing metadata
    meta = {
        "tool": "nougat",
        "input": pdf_path.name,
        "model": model_tag,
        "model_load_time": round(model_time, 2),
        "conversion_time": round(convert_time, 2),
        "total_time": round(total_time, 2),
        "output_chars": len(text),
        "output_words": len(text.split()),
        "pages": page_num,
    }
    (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))

    print(f"[nougat] Output: {md_path}")
    print(
        f"[nougat] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run run_nougat.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/nougat"
    convert(pdf, out)
