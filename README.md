# markdown-pod

PDF-to-Markdown OCR benchmark suite for converting textbooks and documents into clean, structured markdown. Designed for vector database ingestion (Milvus).

## Overview

markdown-pod provides a unified interface to compare multiple state-of-the-art OCR tools on PDF documents. It benchmarks quality, speed, and structure preservation to help you choose the best converter for your use case.

**Use Case**: Convert large textbooks (e.g., 481-page medical texts) into markdown for semantic search and RAG applications.

## Tools Included

| Tool | Model | VRAM | Best For | Speed |
|------|-------|------|----------|-------|
| **marker-pdf** | surya-ocr | 4-6 GB | All-around: tables, math, multi-language | Fast |
| **nougat** | nougat-0.1.0-base | 6 GB | Academic papers, LaTeX math | Medium |
| **deepseek-ocr** | DeepSeek-OCR (3B MoE) | 8-10 GB | Document understanding, high accuracy | Slow |
| **paddleocr** | PP-StructureV3 | 4-6 GB | Table extraction, multilingual | Medium |
| **docstrange** | Nanonets-OCR-s (~4B VLM) | 8-10 GB | Rich descriptions, highest char output | Slow |

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create isolated environments for tools with complex dependencies
chmod +x setup_runpod.sh
./setup_runpod.sh
```

This creates:
- `.venvs/deepseek/` - torch, transformers==4.46.3, flash-attn
- `.venvs/paddleocr/` - paddlepaddle, paddleocr
- `.venvs/docstrange/` - torch, docstrange

marker-pdf and nougat use `uv run` with inline dependencies (PEP 723) and don't need dedicated venvs.

### 2. Run Benchmark

```bash
# Single PDF
./benchmark.sh /path/to/document.pdf

# Multiple PDFs
./benchmark.sh doc1.pdf doc2.pdf

# Custom output directory
./benchmark.sh document.pdf --results-dir ./my_results
```

### 3. Compare Results

The benchmark generates:
```
results/
  marker/document/document.md
  nougat/document/document.md
  deepseek/document/document.md
  paddleocr/document/document.md
  docstrange/document/document.md
  benchmark_report.md      # Side-by-side comparison
  system_info.txt          # GPU, Python versions
```

## Individual Tool Usage

### marker-pdf (uv run)

```bash
uv run run_marker.py input.pdf output_dir/
```

### nougat (uv run)

```bash
uv run run_nougat.py input.pdf output_dir/
```

### DeepSeek-OCR (venv)

```bash
.venvs/deepseek/bin/python run_deepseek.py input.pdf output_dir/
```

### PaddleOCR (venv)

```bash
.venvs/paddleocr/bin/python run_paddleocr.py input.pdf output_dir/
```

### DocStrange (venv)

```bash
.venvs/docstrange/bin/python run_docstrange.py input.pdf output_dir/
```

## Textbook Processing

For optimized textbook conversion with enhanced hierarchy preservation:

```bash
.venvs/docstrange/bin/python run_docstrange_textbook.py textbook.pdf output/ \
  --format markdown \
  --dpi 200 \
  --tokens 8192
```

Options:
- `--format` - Output format: `markdown`, `json`, `html`
- `--dpi` - PDF rasterization DPI (150, 200, 300)
- `--tokens` - Max output tokens per page (default: 8192)
- `--no-hierarchy` - Disable heading normalization

Features:
- Custom OCR prompt for educational content
- Preserves chapter/section hierarchy
- Extracts and matches embedded images
- Outputs page number annotations
- Optimized for vector DB ingestion

## Sample Benchmark Results

On a 19-page medical document (Bates' Guide - Anus, Rectum, Prostate):

| Tool | Time | Chars | Words | Headings | Tables | Lists |
|------|------|-------|-------|----------|--------|-------|
| marker-pdf | 23s | 31,770 | 4,111 | 53 | 7 | 94 |
| nougat | 224s | 21,194 | 3,082 | 41 | 0 | 131 |
| deepseek-ocr | 526s | 29,424 | 4,050 | 62 | 0 | 1 |
| paddleocr | 180s | 28,026 | 3,943 | 19 | 0 | 0 |
| docstrange | 775s | 53,357 | 8,144 | 34 | 0 | 84 |

**Recommendation**: marker-pdf for best speed/structure balance, docstrange for maximum content extraction.

## Output Structure

Each converter produces:

```
output_dir/
  document.md           # Markdown output
  images/               # Extracted images (if applicable)
  timing.json           # Performance metrics
```

## RunPod Deployment

For cloud GPU processing, see [RUNPOD_PLAN.md](RUNPOD_PLAN.md) for:
- Recommended GPU instances (A40, A100, RTX 4090)
- Docker image configuration
- Cost estimates (~$0.50-1.00 per benchmark run)

## Project Structure

```
markdown-pod/
  benchmark.sh              # Main benchmark runner
  setup_runpod.sh           # Environment setup script
  run_marker.py             # marker-pdf converter
  run_nougat.py             # nougat-ocr converter
  run_deepseek.py           # DeepSeek-OCR converter
  run_paddleocr.py          # PaddleOCR converter
  run_docstrange.py         # DocStrange converter
  run_docstrange_textbook.py # Enhanced textbook processor
  compare_results.py        # Results comparison generator
  RUNPOD_PLAN.md            # Cloud deployment guide
```

## Requirements

### System

- Linux (Ubuntu 22.04 recommended)
- NVIDIA GPU with CUDA 12.x
- 24GB+ VRAM for all tools
- 50GB+ disk space for models

### Python

- Python 3.11+
- uv package manager

### System Packages (for docstrange)

```bash
apt-get install poppler-utils pandoc
```

## License

Individual tools have their own licenses. See respective repositories:
- [marker-pdf](https://github.com/datalab-to/marker)
- [nougat](https://github.com/facebookresearch/nougat)
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [DocStrange](https://github.com/NanoNets/docstrange)
