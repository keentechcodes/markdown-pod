#!/usr/bin/env bash
# OCR Benchmark Runner
# Runs all 4 PDF-to-Markdown tools.
#
# marker-pdf and nougat: Run via `uv run` (PEP 723 inline deps)
# deepseek-ocr and paddleocr: Run via dedicated venvs (flash-attn / paddlepaddle-gpu
#   require special install procedures that don't work with PEP 723)
#
# Usage:
#   ./benchmark.sh /path/to/document.pdf
#   ./benchmark.sh /path/to/doc1.pdf /path/to/doc2.pdf
#   ./benchmark.sh /path/to/*.pdf --results-dir /workspace/results
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - NVIDIA GPU with CUDA (for deepseek-ocr and paddleocr-gpu)
#   - Run setup_runpod.sh first to create tool venvs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENVS_DIR="$SCRIPT_DIR/.venvs"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pdf_path> [pdf_path2 ...] [--results-dir DIR]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Sample-PDF2.pdf"
    echo "  $0 /path/to/Sample-PDF2.pdf /path/to/BATES.pdf"
    echo "  $0 /path/to/*.pdf --results-dir /workspace/results"
    exit 1
fi

# Parse args: collect PDFs and optional --results-dir
PDFS=()
RESULTS_DIR="$SCRIPT_DIR/results"
while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) PDFS+=("$(realpath "$1")"); shift ;;
    esac
done

if [ ${#PDFS[@]} -eq 0 ]; then
    echo "ERROR: No PDF files provided"
    exit 1
fi

echo "============================================================"
echo "  OCR Benchmark Suite"
echo "============================================================"
echo "  PDFs:    ${PDFS[*]}"
echo "  Results: $RESULTS_DIR"
echo "  Date:    $(date)"
echo ""

# Check prerequisites
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

for pdf in "${PDFS[@]}"; do
    if [ ! -f "$pdf" ]; then
        echo "ERROR: PDF not found: $pdf"
        exit 1
    fi
done

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. DeepSeek-OCR requires CUDA."
    echo ""
fi

mkdir -p "$RESULTS_DIR"

# Save system info
{
    echo "# Benchmark System Info"
    echo "Date: $(date -Iseconds)"
    for pdf in "${PDFS[@]}"; do
        echo "PDF: $pdf ($(du -h "$pdf" | cut -f1))"
    done
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "## GPU"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        echo ""
    fi
    echo "## Python"
    python3 --version
    echo ""
    echo "## uv"
    uv --version
} > "$RESULTS_DIR/system_info.txt"

run_tool_uv() {
    # Run a tool using `uv run` (PEP 723 inline deps)
    local tool_name="$1"
    local script="$2"
    local pdf_file="$3"
    local pdf_basename
    pdf_basename="$(basename "$pdf_file" .pdf)"
    local output_dir="$RESULTS_DIR/$tool_name/$pdf_basename"
    local log_file="$RESULTS_DIR/$tool_name/$pdf_basename/run.log"

    echo "------------------------------------------------------------"
    echo "  Running: $tool_name on $pdf_basename  [via uv run]"
    echo "------------------------------------------------------------"

    mkdir -p "$output_dir"
    local start_time
    start_time=$(date +%s)

    # Capture full output to log file AND display to console
    if uv run "$SCRIPT_DIR/$script" "$pdf_file" "$output_dir" 2>&1 | tee "$log_file"; then
        local end_time
        end_time=$(date +%s)
        echo ""
        echo "  $tool_name ($pdf_basename) completed in $((end_time - start_time))s"
    else
        local end_time
        end_time=$(date +%s)
        echo ""
        echo "  $tool_name ($pdf_basename) FAILED after $((end_time - start_time))s"
        echo "  Full log: $log_file"
    fi
    echo ""
}

run_tool_venv() {
    # Run a tool using a pre-created venv
    local tool_name="$1"
    local script="$2"
    local pdf_file="$3"
    local venv_dir="$VENVS_DIR/$tool_name"
    local pdf_basename
    pdf_basename="$(basename "$pdf_file" .pdf)"
    local output_dir="$RESULTS_DIR/$tool_name/$pdf_basename"
    local log_file="$RESULTS_DIR/$tool_name/$pdf_basename/run.log"

    echo "------------------------------------------------------------"
    echo "  Running: $tool_name on $pdf_basename  [via venv]"
    echo "------------------------------------------------------------"

    mkdir -p "$output_dir"

    if [ ! -d "$venv_dir" ]; then
        echo "  ERROR: Venv not found at $venv_dir"
        echo "  Run setup_runpod.sh first to create tool environments."
        # Write error timing.json
        cat > "$output_dir/timing.json" <<EOF
{"tool": "$tool_name", "input": "$(basename "$pdf_file")", "error": "Venv not found. Run setup_runpod.sh first.", "total_time": 0, "output_chars": 0, "output_words": 0}
EOF
        echo "Venv not found at $venv_dir" > "$log_file"
        return 1
    fi

    local start_time
    start_time=$(date +%s)

    # Capture full output to log file AND display to console
    if "$venv_dir/bin/python" "$SCRIPT_DIR/$script" "$pdf_file" "$output_dir" 2>&1 | tee "$log_file"; then
        local end_time
        end_time=$(date +%s)
        echo ""
        echo "  $tool_name ($pdf_basename) completed in $((end_time - start_time))s"
    else
        local end_time
        end_time=$(date +%s)
        echo ""
        echo "  $tool_name ($pdf_basename) FAILED after $((end_time - start_time))s"
        echo "  Full log: $log_file"
    fi
    echo ""
}

# Run each tool on each PDF (sequentially to avoid GPU memory conflicts)
for pdf in "${PDFS[@]}"; do
    pdf_basename="$(basename "$pdf" .pdf)"
    echo ""
    echo "============================================================"
    echo "  Processing: $pdf_basename"
    echo "============================================================"

    # marker & nougat: simple deps, use uv run
    run_tool_uv   "marker"    "run_marker.py"    "$pdf"
    run_tool_uv   "nougat"    "run_nougat.py"    "$pdf"

    # deepseek, paddleocr, docstrange: need special packages, use venvs
    run_tool_venv "deepseek"    "run_deepseek.py"    "$pdf"
    run_tool_venv "paddleocr"   "run_paddleocr.py"   "$pdf"
    run_tool_venv "docstrange"  "run_docstrange.py"  "$pdf"
done

# Generate comparison report
echo "------------------------------------------------------------"
echo "  Generating comparison report"
echo "------------------------------------------------------------"
uv run "$SCRIPT_DIR/compare_results.py" "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Benchmark Complete!"
echo "============================================================"
echo "  Results: $RESULTS_DIR"
echo "  Report:  $RESULTS_DIR/benchmark_report.md"
echo ""
echo "  Output files:"
for tool in marker nougat deepseek paddleocr docstrange; do
    if [ -d "$RESULTS_DIR/$tool" ]; then
        echo "  $tool:"
        find "$RESULTS_DIR/$tool" -name "*.md" -not -name "timing.json" | while read -r md_file; do
            size=$(wc -c < "$md_file")
            echo "    $(basename "$(dirname "$md_file")")/$(basename "$md_file") ($size bytes)"
        done
    fi
done
