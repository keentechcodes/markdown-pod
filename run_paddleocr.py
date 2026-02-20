#!/usr/bin/env python3
"""
PaddleOCR with PP-StructureV3
https://github.com/PaddlePaddle/PaddleOCR

Multilingual OCR toolkit with document structure analysis.
PP-StructureV3 outputs markdown directly with table extraction.
Uses PaddlePaddle framework (separate from PyTorch ecosystem).

Run via dedicated venv created by setup_runpod.sh:
    .venvs/paddleocr/bin/python run_paddleocr.py <pdf> [output_dir]

Dependencies (installed by setup_runpod.sh):
    paddlepaddle>=3.0.0 (CPU from PyPI, or GPU from Paddle index if reachable),
    paddleocr>=3.0.0, Pillow, pymupdf
"""

import sys
import time
import json
from pathlib import Path


def check_paddle():
    """Verify paddlepaddle is available (should be pre-installed by setup_runpod.sh)."""
    try:
        import paddle

        gpu_ok = paddle.device.is_compiled_with_cuda()
        print(f"[paddleocr] PaddlePaddle {paddle.__version__} (GPU: {gpu_ok})")
        return True
    except ImportError:
        print("[paddleocr] ERROR: PaddlePaddle not installed!")
        print("[paddleocr] Run setup_runpod.sh first to create the paddleocr venv.")
        return False


def convert(pdf_path: str, output_dir: str):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paddleocr] Converting: {pdf_path.name}")
    print(f"[paddleocr] Output: {output_dir}")

    # Check paddlepaddle is available (pre-installed by setup_runpod.sh)
    if not check_paddle():
        meta = {
            "tool": "paddleocr",
            "input": pdf_path.name,
            "error": "PaddlePaddle not installed. Run setup_runpod.sh first.",
            "total_time": 0,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    # Try PP-StructureV3 first (best quality), fall back to basic OCR
    try:
        return _convert_ppstructurev3(pdf_path, output_dir)
    except Exception as e:
        print(f"[paddleocr] PP-StructureV3 failed: {e}")
        print("[paddleocr] Falling back to basic PaddleOCR...")
        return _convert_basic(pdf_path, output_dir)


def _convert_ppstructurev3(pdf_path: Path, output_dir: Path):
    """Use PP-StructureV3 for structured markdown output."""
    from paddleocr import PPStructureV3
    import paddle

    # Use GPU if available, fall back to CPU
    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    print(f"[paddleocr] Loading PP-StructureV3 pipeline (device={device})...")
    t0 = time.time()
    pipeline = PPStructureV3(device=device)
    model_time = time.time() - t0
    print(f"[paddleocr] Pipeline loaded in {model_time:.1f}s")

    print("[paddleocr] Processing PDF...")
    t0 = time.time()
    output = pipeline.predict(input=str(pdf_path))

    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    text = pipeline.concatenate_markdown_pages(markdown_list)
    convert_time = time.time() - t0
    print(f"[paddleocr] Conversion done in {convert_time:.1f}s")

    # Save markdown
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    # Save images
    for item in markdown_images:
        if item:
            for img_rel_path, image in item.items():
                file_path = output_dir / img_rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)

    total_time = model_time + convert_time
    _save_meta(
        output_dir,
        pdf_path,
        text,
        model_time,
        convert_time,
        total_time,
        "PP-StructureV3",
    )

    print(f"[paddleocr] Output: {md_path}")
    print(
        f"[paddleocr] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


def _convert_basic(pdf_path: Path, output_dir: Path):
    """Fallback: basic PaddleOCR text extraction per page."""
    from paddleocr import PaddleOCR
    import fitz

    print("[paddleocr] Loading basic PaddleOCR...")
    t0 = time.time()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    model_time = time.time() - t0
    print(f"[paddleocr] OCR loaded in {model_time:.1f}s")

    # Convert PDF to images, then OCR each
    print("[paddleocr] Processing PDF pages...")
    t0 = time.time()

    doc = fitz.open(str(pdf_path))
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 144 DPI
        pix = page.get_pixmap(matrix=mat)

        temp_path = output_dir / f"_temp_page_{page_num}.png"
        pix.save(str(temp_path))

        result = ocr.predict(str(temp_path))
        page_lines = []
        for res in result:
            # PaddleOCR 3.x result objects support dict-like access.
            # rec_texts contains the recognized text strings.
            rec_texts = None
            try:
                rec_texts = res["rec_texts"]
            except (KeyError, TypeError):
                if hasattr(res, "rec_texts"):
                    rec_texts = res.rec_texts

            if rec_texts:
                for line in rec_texts:
                    page_lines.append(line)
            else:
                # Last resort: try to get any text from the result
                print(
                    f"[paddleocr] WARNING: Could not extract rec_texts from page {page_num + 1}"
                )

        all_text.append(f"## Page {page_num + 1}\n\n" + "\n".join(page_lines))

        if temp_path.exists():
            temp_path.unlink()

    doc.close()
    convert_time = time.time() - t0

    text = "\n\n---\n\n".join(all_text)

    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    total_time = model_time + convert_time
    _save_meta(
        output_dir,
        pdf_path,
        text,
        model_time,
        convert_time,
        total_time,
        "PaddleOCR-basic",
    )

    print(f"[paddleocr] Output: {md_path}")
    print(
        f"[paddleocr] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


def _save_meta(
    output_dir, pdf_path, text, model_time, convert_time, total_time, variant
):
    meta = {
        "tool": "paddleocr",
        "variant": variant,
        "input": pdf_path.name,
        "model_load_time": round(model_time, 2),
        "conversion_time": round(convert_time, 2),
        "total_time": round(total_time, 2),
        "output_chars": len(text),
        "output_words": len(text.split()),
    }
    (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run run_paddleocr.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/paddleocr"
    convert(pdf, out)
