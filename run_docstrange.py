#!/usr/bin/env python3
"""
DocStrange: PDF-to-Markdown converter by NanoNets
https://github.com/NanoNets/docstrange

Uses the Nanonets-OCR-s ~4B VLM (Qwen2.5-VL-3B based) for local GPU inference.
Outputs markdown with HTML tables and LaTeX equations.

Run via dedicated venv created by setup_runpod.sh:
    .venvs/docstrange/bin/python run_docstrange.py <pdf> [output_dir]

Dependencies (installed by setup_runpod.sh):
    torch>=2.0.0, docstrange>=1.1.0, pdf2image, Pillow
    System: poppler-utils, pandoc
"""

import sys
import time
import json
import re
from pathlib import Path


def _extract_images_from_pdf(pdf_path: str, images_dir: str, min_size: int = 50):
    """Extract embedded images from a PDF using PyMuPDF.

    Returns dict mapping page_number (1-indexed) -> list of image info dicts.
    Skips tiny images (icons, bullets, decorative) below min_size pixels.
    """
    import fitz

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_images = {}
    seen_xrefs = set()  # avoid duplicates across pages

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        page_images[page_num + 1] = []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                if width < min_size or height < min_size:
                    continue

                image_filename = f"page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                image_path = images_dir / image_filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                page_images[page_num + 1].append(
                    {
                        "filename": image_filename,
                        "width": width,
                        "height": height,
                    }
                )
            except Exception as e:
                print(
                    f"  [warn] Image extraction failed (page {page_num + 1}, xref {xref}): {e}"
                )

    doc.close()
    return page_images


def _replace_img_tags(markdown: str, page_images: dict) -> tuple:
    """Replace <img>description</img> tags with ![description](images/file) refs.

    Strategy: for each page, match the Nth <img> tag to the Nth extracted image.
    Documents typically list images in reading order, matching PyMuPDF's extraction order.

    Returns (updated_markdown, images_matched_count).
    """
    img_tag_pattern = re.compile(r"<img>(.*?)</img>", re.DOTALL)
    page_header_pattern = re.compile(r"^## Page (\d+)", re.MULTILINE)

    page_splits = list(page_header_pattern.finditer(markdown))
    if not page_splits:
        # No page headers — treat entire document as page 1
        page_splits_ranges = [(1, 0, len(markdown))]
    else:
        page_splits_ranges = []
        for i, m in enumerate(page_splits):
            page_num = int(m.group(1))
            start = m.end()
            end = (
                page_splits[i + 1].start()
                if i + 1 < len(page_splits)
                else len(markdown)
            )
            page_splits_ranges.append((page_num, start, end))

    # Collect all replacements with absolute positions
    replacements = []
    matched = 0

    for page_num, start, end in page_splits_ranges:
        page_content = markdown[start:end]
        extracted = page_images.get(page_num, [])

        for img_idx, img_match in enumerate(img_tag_pattern.finditer(page_content)):
            description = img_match.group(1).strip()
            abs_start = start + img_match.start()
            abs_end = start + img_match.end()

            if img_idx < len(extracted):
                img_info = extracted[img_idx]
                replacement = f"![{description}](images/{img_info['filename']})"
                matched += 1
            else:
                # No matching extracted image — keep as italic description
                replacement = f"*[Image: {description}]*"

            replacements.append((abs_start, abs_end, replacement))

    # Apply in reverse order to preserve positions
    result = markdown
    for abs_start, abs_end, replacement in reversed(replacements):
        result = result[:abs_start] + replacement + result[abs_end:]

    return result, matched


def _apply_optimizations():
    """Apply performance optimizations and verbose logging to docstrange.

    1. Lower PDF rasterization DPI from 300 to 150 (~2-4x fewer pixels,
       20-40% speedup, sufficient quality for most printed documents).
    2. Monkey-patch the model loader to use SDPA attention (built into PyTorch,
       ~30-40% speedup on attention computation, no extra deps).
    3. Add per-page progress logging (docstrange is silent during processing).
    """
    optimizations = []

    # Optimization 1: Lower DPI (300 -> 150)
    try:
        from docstrange.config import InternalConfig

        InternalConfig.pdf_image_dpi = 150  # default is 300
        optimizations.append("DPI=150 (was 300)")
    except (ImportError, AttributeError):
        pass

    # Optimization 2: SDPA attention on model load
    try:
        import docstrange.pipeline.nanonets_processor as np_module

        _original_init = np_module.NanonetsDocumentProcessor._initialize_models

        def _patched_init(self, cache_dir=None):
            from transformers import (
                AutoTokenizer,
                AutoProcessor,
                AutoModelForImageTextToText,
            )
            from docstrange.pipeline.model_downloader import ModelDownloader

            model_downloader = ModelDownloader(cache_dir)
            model_path = model_downloader.get_model_path("nanonets-ocr")
            # The model is stored in a subdirectory named Nanonets-OCR-ss
            actual_model_path = model_path / "Nanonets-OCR-ss"
            if not actual_model_path.exists():
                # Fall back to the model_path itself
                actual_model_path = model_path

            self.model = AutoModelForImageTextToText.from_pretrained(
                str(actual_model_path),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                attn_implementation="sdpa",  # PyTorch built-in SDPA
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(actual_model_path), local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(
                str(actual_model_path), local_files_only=True
            )

        np_module.NanonetsDocumentProcessor._initialize_models = _patched_init
        optimizations.append("SDPA attention")
    except (ImportError, AttributeError) as e:
        print(f"[docstrange] SDPA patch failed: {e}")

    # Patch 3: Per-page verbose progress logging
    # docstrange is completely silent during page-by-page OCR processing.
    # Wrap _extract_text_with_nanonets to print progress per page.
    try:
        import docstrange.pipeline.nanonets_processor as np_module

        _original_extract = (
            np_module.NanonetsDocumentProcessor._extract_text_with_nanonets
        )
        _page_counter = {"current": 0, "total": 0}

        def _verbose_extract(self, image_path, max_new_tokens=4096):
            import time as _time

            _page_counter["current"] += 1
            n = _page_counter["current"]
            total = _page_counter["total"]
            total_str = f"/{total}" if total else ""
            print(
                f"[docstrange]   Page {n}{total_str}...",
                end="",
                flush=True,
            )
            t = _time.time()
            result = _original_extract(self, image_path, max_new_tokens)
            elapsed = _time.time() - t
            chars = len(result) if result else 0
            print(f" done ({elapsed:.1f}s, {chars:,} chars)")
            return result

        np_module.NanonetsDocumentProcessor._extract_text_with_nanonets = (
            _verbose_extract
        )
        # Store the counter ref so we can set total from convert()
        _apply_optimizations._page_counter = _page_counter
        optimizations.append("verbose progress")
    except (ImportError, AttributeError) as e:
        print(f"[docstrange] Verbose patch failed: {e}")

    # Patch 4: Wrap _convert_pdf_to_images to count total pages
    try:
        import docstrange.gpu_processor as gpu_mod

        _original_convert = gpu_mod.GPUProcessor._convert_pdf_to_images

        def _verbose_convert(self, pdf_path):
            result = _original_convert(self, pdf_path)
            total = len(result) if result else 0
            if hasattr(_apply_optimizations, "_page_counter"):
                _apply_optimizations._page_counter["total"] = total
                _apply_optimizations._page_counter["current"] = 0
            print(
                f"[docstrange] Rasterized {total} pages (DPI={getattr(InternalConfig, 'pdf_image_dpi', 300)})"
            )
            return result

        gpu_mod.GPUProcessor._convert_pdf_to_images = _verbose_convert
    except (ImportError, AttributeError):
        pass

    return optimizations


def convert(pdf_path: str, output_dir: str):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[docstrange] Converting: {pdf_path.name}")
    print(f"[docstrange] Output: {output_dir}")

    t0 = time.time()

    try:
        import torch

        print(f"[docstrange] CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("[docstrange] torch not available")

    try:
        from docstrange import DocumentExtractor
    except ImportError as e:
        error_msg = f"Failed to import docstrange: {e}"
        print(f"[docstrange] ERROR: {error_msg}")
        meta = {
            "tool": "docstrange",
            "input": pdf_path.name,
            "total_time": round(time.time() - t0, 2),
            "error": error_msg,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    # Apply performance optimizations before creating the extractor
    opts = _apply_optimizations()
    if opts:
        print(f"[docstrange] Optimizations: {', '.join(opts)}")

    # Use GPU local mode for fair benchmark comparison
    # Falls back to cloud if GPU fails
    use_gpu = False
    try:
        import torch

        use_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    mode = "GPU local" if use_gpu else "cloud API"
    print(f"[docstrange] Mode: {mode}")
    print(f"[docstrange] Loading extractor (downloads ~6GB model on first GPU run)...")

    try:
        if use_gpu:
            extractor = DocumentExtractor(gpu=True)
        else:
            extractor = DocumentExtractor()
    except Exception as e:
        error_msg = f"Failed to create extractor: {e}"
        print(f"[docstrange] ERROR: {error_msg}")
        # Try cloud fallback
        print("[docstrange] Falling back to cloud API...")
        try:
            extractor = DocumentExtractor()
            mode = "cloud API (fallback)"
        except Exception as e2:
            error_msg = f"Both GPU and cloud failed: GPU={e}, Cloud={e2}"
            print(f"[docstrange] ERROR: {error_msg}")
            meta = {
                "tool": "docstrange",
                "input": pdf_path.name,
                "total_time": round(time.time() - t0, 2),
                "error": error_msg,
                "output_chars": 0,
                "output_words": 0,
            }
            (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
            return None

    model_time = time.time() - t0
    print(f"[docstrange] Extractor ready in {model_time:.1f}s")

    # Convert PDF to markdown
    print("[docstrange] Processing PDF...")
    t1 = time.time()

    try:
        result = extractor.extract(str(pdf_path))
        text = result.extract_markdown()
    except Exception as e:
        error_msg = f"Extraction failed: {e}"
        print(f"[docstrange] ERROR: {error_msg}")
        meta = {
            "tool": "docstrange",
            "input": pdf_path.name,
            "model_load_time": round(model_time, 2),
            "total_time": round(time.time() - t0, 2),
            "error": error_msg,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    convert_time = time.time() - t1

    if not text:
        text = ""
        print("[docstrange] WARNING: No text extracted from PDF")

    # Extract images from PDF and replace <img> tags with markdown refs
    images_extracted = 0
    images_matched = 0
    try:
        images_dir = str(output_dir / "images")
        print("[docstrange] Extracting images from PDF...")
        page_images = _extract_images_from_pdf(str(pdf_path), images_dir)
        images_extracted = sum(len(imgs) for imgs in page_images.values())

        if images_extracted > 0:
            text, images_matched = _replace_img_tags(text, page_images)
            print(
                f"[docstrange] Extracted {images_extracted} images, "
                f"matched {images_matched} to <img> tags"
            )
        else:
            print("[docstrange] No embedded images found in PDF")
    except Exception as e:
        print(f"[docstrange] Image extraction failed (non-fatal): {e}")

    # Save markdown
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    total_time = model_time + convert_time

    # Save timing metadata
    meta = {
        "tool": "docstrange",
        "input": pdf_path.name,
        "model": "nanonets/Nanonets-OCR-s (~4B VLM, Qwen2.5-VL-3B based)",
        "mode": mode,
        "model_load_time": round(model_time, 2),
        "conversion_time": round(convert_time, 2),
        "total_time": round(total_time, 2),
        "output_chars": len(text),
        "output_words": len(text.split()) if text else 0,
        "images_extracted": images_extracted,
        "images_matched": images_matched,
    }
    (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))

    print(f"[docstrange] Output: {md_path}")
    print(
        f"[docstrange] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: .venvs/docstrange/bin/python run_docstrange.py <pdf_path> [output_dir]"
        )
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/docstrange"
    convert(pdf, out)
