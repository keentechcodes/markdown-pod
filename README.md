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

## Benchmark Results

Tested on a 19-page medical textbook sample (Bates' Guide to Physical Examination, Chapters 15-16) containing tables, clinical images, multi-column layouts, bullet lists, and dense medical terminology. GPU: RTX 4000 Ada 20GB.

### Performance

| Tool | Time | Chars | Words | Images | Tables | Lists | Bold |
|------|------|-------|-------|--------|--------|-------|------|
| marker-pdf | **23s** | 31,770 | 4,111 | 25 | 7 | 94 | 80 |
| docstrange | 775s | 53,357 | 8,144 | 33 | 0* | 84 | 72 |
| deepseek-ocr | 526s | 29,424 | 4,050 | 24 | 0* | 1 | 0 |
| nougat | 224s | 21,194 | 3,082 | 0 | 0 | 131 | 13 |
| paddleocr | 180s | 28,026 | 3,943 | 0 | 0 | 0 | 0 |

*DeepSeek and DocStrange output HTML `<table>` elements (not markdown pipe tables), so the structure counter reports 0.

### Quality Comparison (Top 3)

#### 1st: DocStrange

Best overall output for medical education content.

- **Tables**: Semantic HTML with `<thead>`, `<tbody>`, `colspan`, `<strong>` inside cells. The BPH scoring questionnaire and Systemic Disorders tables are perfectly structured.
- **Images**: Extracts native embedded images from PDF at original resolution. AI-generated alt-text descriptions (e.g., `![A diagram showing an anal fissure...](images/page3_img1.jpeg)`) provide context that's genuinely useful for accessibility and study.
- **Lists**: 84 properly formatted bullet items with correct nesting (sublists for ligaments, tendons under extra-articular structures).
- **Bold**: 72 elements. Clinical terms, table titles, and section labels correctly emphasized.
- **Hierarchy**: Clean `#` > `##` > `###` heading structure.
- **Weaknesses**: Slowest tool (775s). Leaves `<page_number>` tags in output. Some italic formatting glitches on quoted text.

#### 2nd: marker-pdf

Best speed-to-quality ratio. The pragmatic choice.

- **Speed**: 33x faster than DocStrange, 23x faster than DeepSeek.
- **Tables**: Markdown pipe tables that render correctly. Handles multi-line cells with `<br>`. 7 tables detected.
- **Images**: 25 native embedded images extracted at original resolution.
- **Lists**: 94 bullet items, the most of any tool. Proper indentation for nested sublists.
- **Bold**: 80 elements, the most of any tool.
- **Weaknesses**: Pipe tables can't handle very complex cell layouts as cleanly as HTML. Some `<sup>l</sup>` artifacts from converting the textbook's bullet markers.

#### 3rd: DeepSeek-OCR

Highest raw OCR text accuracy, but poor structure preservation.

- **Text accuracy**: Best of all tools. Medical terminology, drug names, clinical measurements all correct. Uses LaTeX for math notation (`\(\leq 7\)`, `\(< 6\)` weeks).
- **Tables**: Produces HTML `<table>` elements. Content is correct but flat structure (no `<thead>`/`<tbody>`).
- **Images**: 24 images cropped from page rasters using model-predicted bounding box coordinates.
- **Lists**: **Critical weakness.** Only 1 list item detected. Bullet lists get concatenated into single long lines (e.g., all risk factors for osteoporosis smashed into one paragraph). For a medical textbook that is ~40% bullet-pointed clinical criteria, this makes the output significantly harder to read.
- **Bold**: No formatting preservation. Everything is plain text or headings.
- **Headings**: Flat hierarchy. Uses `##` for almost everything including "OR" between clinical examples.

### Tools Not Recommended

- **Nougat**: Hallucinated on non-academic content (repeated "Clean-Clean-Clean..." on page 1 of Sample-PDF2). Trained on arXiv papers; struggles with clinical textbook formatting. No image extraction.
- **PaddleOCR**: Plain text dump only. No tables, lists, bold, or image extraction. PP-StructureV3 (which adds structure) requires `paddlepaddle-gpu` from a Chinese package index that's unreachable from most cloud providers.

### Recommendation

| Use Case | Tool | Why |
|----------|------|-----|
| Medical education / MediFact | **DocStrange** | Best tables, image descriptions, lists, and bold formatting for study material |
| Production pipeline (speed matters) | **marker-pdf** | 33x faster with 90% of the quality. Best speed/structure balance |
| Maximum text accuracy | **DeepSeek-OCR** | Best raw OCR but unusable list/bold formatting for structured content |

## Visualization

Generate visual comparison charts from benchmark results:

```bash
uv run visualize_results.py results_BATES_1/
```

Outputs:
- `timing_comparison.png` - Processing time bar chart
- `content_extraction.png` - Characters/words extracted
- `structure_detection.png` - Structural elements comparison
- `radar_comparison.png` - Overall quality radar/spider chart
- `speed_vs_quality.png` - Speed vs content scatter plot

See `results_BATES_1/` for sample output from the Bates medical textbook benchmark.

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
  visualize_results.py      # Chart generation
  results_BATES_1/          # Sample benchmark results
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
