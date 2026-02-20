# OCR Benchmark on RunPod GPU

## Goal

Compare 4 PDF-to-Markdown OCR tools on a RunPod GPU instance, using the same
test PDF (`Sample-PDF2.pdf`, 5 pages). Evaluate quality, speed, and output
structure.

## Tools Under Test

| # | Tool | Model | VRAM Needed | Key Strength |
|---|------|-------|-------------|--------------|
| 1 | **marker-pdf** | surya-ocr (layout+OCR) | ~4-6 GB | Best all-around: tables, math, code, multi-language |
| 2 | **nougat** | nougat-0.1.0-base | ~6 GB | Academic papers, LaTeX math, structured output |
| 3 | **deepseek-ocr** | DeepSeek-OCR (3B MoE) | ~8-10 GB | Document understanding, 10x compression, high accuracy |
| 4 | **paddleocr** | PP-StructureV3 | ~4-6 GB | Table extraction, multilingual, lightweight |

## Recommended RunPod Setup

- **GPU**: A40 (48GB) or A100 (40GB) - all 4 tools fit comfortably
  - Budget option: RTX 4090 (24GB) - still enough for all tools
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Disk**: 50 GB (for model downloads)
- **Estimated cost**: ~$0.50-1.00 for the full benchmark run

## Dependency Isolation Strategy

Each tool has conflicting dependency requirements:
- `marker-pdf` requires `transformers>=4.56.1`
- `deepseek-ocr` requires `transformers==4.46.3`
- `nougat-ocr` has its own pinned dependency tree
- `paddleocr` uses PaddlePaddle (entirely different framework)

**Solution**: Use `uv run` with PEP 723 inline script metadata. Each converter
script declares its own isolated dependencies in a `# /// script` header block.
`uv` creates a separate cached environment per script automatically.

## Scripts

### `benchmark.sh`
Main entrypoint. Runs all 4 tools sequentially, collects timing and output.

### Individual converter scripts (PEP 723 isolated):
- `run_marker.py` - marker-pdf conversion
- `run_nougat.py` - nougat-ocr conversion
- `run_deepseek.py` - deepseek-ocr conversion
- `run_paddleocr.py` - paddleocr conversion

### `compare_results.py`
Compares all outputs side-by-side: character count, word count, structure
detection (tables, headings, lists, math), and timing.

## Output Structure

```
markdown-pod/
  RUNPOD_PLAN.md          # This file
  benchmark.sh            # Main runner script
  run_marker.py           # marker-pdf (isolated deps)
  run_nougat.py           # nougat-ocr (isolated deps)
  run_deepseek.py         # deepseek-ocr (isolated deps)
  run_paddleocr.py        # paddleocr (isolated deps)
  compare_results.py      # Compare all outputs
  results/                # Created at runtime
    marker/
    nougat/
    deepseek/
    paddleocr/
    benchmark_report.md
```

## Quick Start (on RunPod)

```bash
# 1. Upload test PDF and scripts to the pod
# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Run the benchmark
chmod +x benchmark.sh
./benchmark.sh /path/to/Sample-PDF2.pdf
```

## Notes

- All scripts use `trust_remote_code=True` for HuggingFace models
- First run downloads models (~20 GB total across all tools)
- Subsequent runs use cached models
- Each script exits cleanly with timing info written to results/
