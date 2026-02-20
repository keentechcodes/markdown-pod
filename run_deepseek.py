#!/usr/bin/env python3
"""
DeepSeek-OCR: Contexts Optical Compression
https://github.com/deepseek-ai/DeepSeek-OCR

3B MoE vision-language model for document understanding.
Achieves 97% OCR precision at 10x compression.
Requires: CUDA + flash-attn (GPU only).

Run via dedicated venv created by setup_runpod.sh:
    .venvs/deepseek/bin/python run_deepseek.py <pdf> [output_dir]

Dependencies (installed by setup_runpod.sh):
    torch>=2.0.0, transformers==4.46.3, flash-attn>=2.7.3,
    Pillow, pymupdf, accelerate, safetensors, einops, addict, easydict
"""

import os
import sys
import time
import json
import re
import gc
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_grounding_tags(
    text: str, source_image, images_dir: str, page_idx: int
) -> tuple:
    """Process DeepSeek-OCR grounding tags: extract images, clean markdown.

    The model outputs inline tags like:
        <|ref|>image<|/ref|><|det|>[[155, 150, 713, 216]]<|/det|>
        <|ref|>title<|/ref|><|det|>[[155, 150, 713, 216]]<|/det|>

    For 'image' tags: crop the region from the source page image, save it,
    and replace the tag with a markdown image reference ![](images/...).
    For all other tags: strip them (they're redundant with the markdown).

    Coordinates are normalized to 0-999 (top-left=0,0, bottom-right=999,999).
    """
    import ast

    img_width, img_height = source_image.size
    os.makedirs(images_dir, exist_ok=True)

    # Find all grounding tag pairs
    tag_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(tag_pattern, text, re.DOTALL)

    img_count = 0
    for full_match, label, coords_str in matches:
        label = label.strip().lower()

        if label == "image":
            # Try to crop the figure from the source image
            try:
                coords_list = ast.literal_eval(coords_str.strip())
                # coords_list is [[x1, y1, x2, y2]] or [[x1,y1,x2,y2], ...]
                if coords_list and len(coords_list[0]) == 4:
                    bbox = coords_list[0]
                    # Convert normalized coords (0-999) to pixel coords
                    x1 = int(bbox[0] / 999 * img_width)
                    y1 = int(bbox[1] / 999 * img_height)
                    x2 = int(bbox[2] / 999 * img_width)
                    y2 = int(bbox[3] / 999 * img_height)

                    # Clamp to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_width, x2), min(img_height, y2)

                    if x2 > x1 and y2 > y1:
                        cropped = source_image.crop((x1, y1, x2, y2))
                        img_filename = f"page{page_idx}_{img_count}.jpg"
                        cropped.save(os.path.join(images_dir, img_filename), quality=95)
                        text = text.replace(full_match, f"![](images/{img_filename})\n")
                        img_count += 1
                        continue
            except Exception as e:
                print(f"  [warn] Failed to extract image: {e}")

            # If cropping failed, just remove the tag
            text = text.replace(full_match, "")
        else:
            # Non-image tags: strip entirely (redundant with markdown formatting)
            text = text.replace(full_match, "")

    # Clean up whitespace from tag removal
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), img_count


def strip_grounding_tags(text: str) -> str:
    """Simple fallback: remove all grounding tags without image extraction."""
    text = re.sub(r"<\|ref\|>[^<]*<\|/ref\|>", "", text)
    text = re.sub(r"<\|det\|>[^<]*<\|/det\|>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def check_flash_attn():
    """Check if flash-attn is available (should be pre-installed by setup_runpod.sh)."""
    try:
        import flash_attn  # noqa: F401

        print(f"[deepseek-ocr] flash-attn {flash_attn.__version__} available")
        return True
    except ImportError:
        print(
            "[deepseek-ocr] flash-attn NOT available, will use eager attention (slower)"
        )
        return False


def pdf_to_images(pdf_path: str, dpi: int = 150):
    """Convert PDF pages to PIL images."""
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def convert(pdf_path: str, output_dir: str):
    import torch

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[deepseek-ocr] Converting: {pdf_path.name}")
    print(f"[deepseek-ocr] Output: {output_dir}")
    print(f"[deepseek-ocr] CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[deepseek-ocr] ERROR: CUDA is required for DeepSeek-OCR")
        meta = {
            "tool": "deepseek-ocr",
            "input": pdf_path.name,
            "error": "CUDA not available",
            "total_time": 0,
            "output_chars": 0,
            "output_words": 0,
        }
        (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        return None

    # Check flash-attn (should be pre-installed by setup_runpod.sh)
    has_flash_attn = check_flash_attn()

    from transformers import AutoModel, AutoTokenizer

    # Convert PDF to images
    print("[deepseek-ocr] Converting PDF to images...")
    images = pdf_to_images(str(pdf_path))
    total_pages = len(images)
    print(f"[deepseek-ocr] {total_pages} pages")

    # Load model (official way from README)
    print("[deepseek-ocr] Loading model (downloads ~6GB on first run)...")
    model_name = "deepseek-ai/DeepSeek-OCR"

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # DeepSeek-OCR only supports flash_attention_2 or eager (NOT sdpa).
    # flash_attention_2 gives ~2-3x speedup over eager but requires the
    # flash-attn package to be installed (built by setup_runpod.sh).
    if has_flash_attn:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "eager"
    print(f"[deepseek-ocr] Using attention implementation: {attn_impl}")

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "_attn_implementation": attn_impl,
    }

    # Load model in bfloat16 to fit in 20GB VRAM
    # float32 would use ~19GB just for model weights, causing OOM on inference.
    # bfloat16 uses ~9.5GB, leaving room for inference activations.
    # Note: the previous "masked_scatter_ Half vs Float" error was from float16,
    # NOT bfloat16. bfloat16 has the same exponent range as float32 and works
    # correctly with eager attention.
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # load directly in bf16 to halve peak memory
        **model_kwargs,
    )
    dtype = torch.bfloat16
    print(f"[deepseek-ocr] Using dtype: {dtype} (fits in 20GB VRAM)")
    model = model.eval().cuda()
    model_time = time.time() - t0
    print(f"[deepseek-ocr] Model loaded in {model_time:.1f}s")

    # Process each page
    print(f"[deepseek-ocr] Processing {total_pages} pages...")
    t0 = time.time()
    all_results = []

    temp_dir = output_dir / "temp_images"
    temp_dir.mkdir(exist_ok=True)
    images_dir = str(output_dir / "images")
    total_images_extracted = 0

    for idx, img in enumerate(images):
        page_start = time.time()
        print(f"[deepseek-ocr]   Page {idx + 1}/{total_pages}...", end="", flush=True)

        temp_img_path = temp_dir / f"page_{idx:03d}.jpg"
        img.save(temp_img_path, quality=95)

        prompt = "<image>\n<|grounding|>Convert the document to markdown."

        try:
            with torch.no_grad():
                # eval_mode=True is REQUIRED to get a return value.
                # Without it, infer() streams output to stdout via TextStreamer
                # and returns None. With eval_mode=True, it decodes output_ids
                # and returns the text as a string.
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(temp_img_path),
                    output_path=str(output_dir),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                    test_compress=False,
                    eval_mode=True,
                )

            if result is None:
                raise RuntimeError("Model returned None")

            raw_text = result if isinstance(result, str) else str(result)

            # Process grounding tags: extract images + clean markdown
            cleaned_text, img_count = process_grounding_tags(
                raw_text, img, images_dir, page_idx=idx
            )
            all_results.append(cleaned_text)
            total_images_extracted += img_count

            page_time = time.time() - page_start
            imgs_msg = f", {img_count} images" if img_count else ""
            print(f" done ({page_time:.1f}s{imgs_msg})")

        except Exception as e:
            print(f" ERROR: {e}")
            all_results.append(f"[Error on page {idx + 1}: {e}]")

        if temp_img_path.exists():
            temp_img_path.unlink()

    convert_time = time.time() - t0

    # Cleanup temp dir
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    if total_images_extracted:
        print(
            f"[deepseek-ocr] Extracted {total_images_extracted} images to {images_dir}"
        )

    # Combine results (grounding tags already processed per-page)
    text = "\n\n---\n\n".join(all_results)

    # Save output
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")

    total_time = model_time + convert_time

    # Save timing metadata
    meta = {
        "tool": "deepseek-ocr",
        "input": pdf_path.name,
        "model": "deepseek-ai/DeepSeek-OCR",
        "model_load_time": round(model_time, 2),
        "conversion_time": round(convert_time, 2),
        "total_time": round(total_time, 2),
        "output_chars": len(text),
        "output_words": len(text.split()),
        "pages": total_pages,
        "images_extracted": total_images_extracted,
    }
    (output_dir / "timing.json").write_text(json.dumps(meta, indent=2))

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[deepseek-ocr] Output: {md_path}")
    print(
        f"[deepseek-ocr] {len(text):,} chars | {len(text.split()):,} words | {total_time:.1f}s total"
    )
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run run_deepseek.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/deepseek"
    convert(pdf, out)
