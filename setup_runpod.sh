#!/usr/bin/env bash
# RunPod Setup Script
# Creates dedicated virtual environments for tools with complex deps.
#
# Usage:
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
#
# This creates:
#   .venvs/deepseek/  - venv with torch, transformers==4.46.3, flash-attn, etc.
#   .venvs/paddleocr/ - venv with paddlepaddle-gpu, paddleocr, etc.
#
# marker-pdf and nougat use uv run (PEP 723) and don't need this.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENVS_DIR="$SCRIPT_DIR/.venvs"
SETUP_LOG="$SCRIPT_DIR/setup.log"

# Log everything to both console and log file
exec > >(tee -a "$SETUP_LOG") 2>&1
echo "Setup started at $(date -Iseconds)"
echo ""

echo "============================================================"
echo "  RunPod Environment Setup"
echo "============================================================"
echo ""

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "[setup] uv installed: $(uv --version)"
else
    echo "[setup] uv already installed: $(uv --version)"
fi

# 2. System info
echo ""
echo "[setup] System Info:"
echo "  Python: $(python3 --version 2>&1)"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
    echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
fi
python3 -c "import torch; print(f'  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, {torch.version.cuda})')" 2>/dev/null || echo "  PyTorch: not found in system python"

mkdir -p "$VENVS_DIR"

# ============================================================
# 3. DeepSeek-OCR venv
# ============================================================
echo ""
echo "============================================================"
echo "  Creating DeepSeek-OCR environment"
echo "============================================================"

DEEPSEEK_VENV="$VENVS_DIR/deepseek"

if [ -d "$DEEPSEEK_VENV" ] && "$DEEPSEEK_VENV/bin/python" -c "import torch; import transformers" 2>/dev/null; then
    echo "[deepseek] Venv already exists and looks good"
else
    echo "[deepseek] Creating venv at $DEEPSEEK_VENV"
    rm -rf "$DEEPSEEK_VENV"
    uv venv "$DEEPSEEK_VENV" --python 3.11

    echo "[deepseek] Installing base dependencies..."
    uv pip install --python "$DEEPSEEK_VENV/bin/python" \
        "torch>=2.0.0" \
        "torchvision>=0.15.0" \
        "transformers==4.46.3" \
        "tokenizers==0.20.3" \
        "Pillow>=10.0.0" \
        "pymupdf>=1.24.0" \
        "accelerate>=0.25.0" \
        "safetensors>=0.4.0" \
        "einops>=0.7.0" \
        "addict>=2.4.0" \
        "easydict>=1.13"

    echo "[deepseek] Building flash-attn (this takes 5-15 minutes)..."
    echo "[deepseek] (if it fails, DeepSeek will use eager attention — slower but works)"
    # flash-attn must be built against the torch that's already installed.
    # It needs setuptools at build time but doesn't declare it as a build dep.
    # uv venv doesn't install pip/setuptools, so we install setuptools first.
    uv pip install --python "$DEEPSEEK_VENV/bin/python" setuptools wheel 2>&1 | tail -5
    export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4000 Ada
    export MAX_JOBS=4

    if uv pip install --python "$DEEPSEEK_VENV/bin/python" \
        "flash-attn>=2.7.3" --no-build-isolation 2>&1 | tail -20; then
        echo "[deepseek] flash-attn installed!"
    else
        echo "[deepseek] WARNING: flash-attn build failed."
        echo "[deepseek] DeepSeek-OCR will use eager attention with float32 (slower but functional)."
    fi

    echo "[deepseek] Verifying..."
    "$DEEPSEEK_VENV/bin/python" -c "
import torch
print(f'  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import transformers
print(f'  transformers: {transformers.__version__}')
try:
    import flash_attn
    print(f'  flash-attn: {flash_attn.__version__}')
except ImportError:
    print('  flash-attn: NOT INSTALLED (will use eager attention)')
"
fi

# ============================================================
# 4. PaddleOCR venv
# ============================================================
echo ""
echo "============================================================"
echo "  Creating PaddleOCR environment"
echo "============================================================"

PADDLE_VENV="$VENVS_DIR/paddleocr"

if [ -d "$PADDLE_VENV" ] && "$PADDLE_VENV/bin/python" -c "import paddle; import paddleocr" 2>/dev/null; then
    echo "[paddleocr] Venv already exists and looks good"
else
    echo "[paddleocr] Creating venv at $PADDLE_VENV"
    rm -rf "$PADDLE_VENV"
    uv venv "$PADDLE_VENV" --python 3.11

    # paddlepaddle-gpu >=3.0 is ONLY on Paddle's Chinese index (unreachable from RunPod).
    # paddlepaddle >=3.0 (CPU) IS on standard PyPI and works with paddleocr 3.x.
    # CPU inference is slower but produces the same quality output.
    #
    # Strategy: try GPU from Paddle index first, fall back to CPU 3.x from PyPI.

    PADDLE_INSTALLED=false

    # Strategy 1: GPU cu124 from Paddle index (fastest if reachable)
    # --index-strategy unsafe-best-match: allow Paddle's index to provide paddlepaddle-gpu
    # even though PyPI also has an older version. Without this flag, uv refuses to use
    # the Paddle index version to prevent dependency confusion attacks.
    echo "[paddleocr] Trying: paddlepaddle-gpu 3.0.0 (cu124) from Paddle index..."
    if uv pip install --python "$PADDLE_VENV/bin/python" \
        "paddlepaddle-gpu==3.0.0" \
        --index-strategy unsafe-best-match \
        --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/cu124/" \
        2>&1 | tail -10; then
        PADDLE_INSTALLED=true
        echo "[paddleocr] paddlepaddle-gpu 3.0.0 (cu124) installed!"
    fi

    # Strategy 2: CPU paddlepaddle 3.0.0 from PyPI (always available, slower)
    if [ "$PADDLE_INSTALLED" = false ]; then
        echo "[paddleocr] GPU install failed (Paddle Chinese index likely unreachable)."
        echo "[paddleocr] Installing CPU paddlepaddle 3.0.0 from PyPI..."
        echo "[paddleocr] NOTE: CPU inference is slower but output quality is identical."
        if uv pip install --python "$PADDLE_VENV/bin/python" \
            "paddlepaddle==3.0.0" 2>&1 | tail -10; then
            PADDLE_INSTALLED=true
            echo "[paddleocr] paddlepaddle 3.0.0 CPU installed from PyPI!"
        fi
    fi

    if [ "$PADDLE_INSTALLED" = false ]; then
        echo "[paddleocr] ERROR: All paddlepaddle install strategies failed!"
        echo "[paddleocr] This should not happen — paddlepaddle 3.0.0 is on PyPI."
    fi

    echo "[paddleocr] Installing paddleocr and other deps..."
    # paddleocr 3.0.0 is the first 3.x version with PP-StructureV3 markdown output.
    # Using 3.0.0 to match paddlepaddle 3.0.0 for compatibility.
    uv pip install --python "$PADDLE_VENV/bin/python" \
        "paddleocr>=3.0.0" \
        "Pillow>=10.0.0" \
        "pymupdf>=1.24.0"

    # PP-StructureV3 requires the paddlex[ocr] extra for full functionality.
    # Without this, PPStructureV3() raises DependencyError.
    echo "[paddleocr] Installing paddlex[ocr] extra for PP-StructureV3..."
    if uv pip install --python "$PADDLE_VENV/bin/python" \
        "paddlex[ocr]" 2>&1 | tail -10; then
        echo "[paddleocr] paddlex[ocr] installed!"
    else
        echo "[paddleocr] WARNING: paddlex[ocr] install failed."
        echo "[paddleocr] PP-StructureV3 won't work, will fall back to basic OCR."
    fi

    echo "[paddleocr] Verifying..."
    "$PADDLE_VENV/bin/python" -c "
import paddle
print(f'  paddle: {paddle.__version__} (GPU: {paddle.device.is_compiled_with_cuda()})')
import paddleocr
print(f'  paddleocr: OK')
"
fi

# ============================================================
# 5. DocStrange venv
# ============================================================
echo ""
echo "============================================================"
echo "  Creating DocStrange environment"
echo "============================================================"

DOCSTRANGE_VENV="$VENVS_DIR/docstrange"

# System dependencies: poppler-utils (for pdf2image) and pandoc
echo "[docstrange] Installing system dependencies (poppler-utils, pandoc)..."
apt-get update -qq && apt-get install -y -qq poppler-utils pandoc 2>&1 | tail -3

if [ -d "$DOCSTRANGE_VENV" ] && "$DOCSTRANGE_VENV/bin/python" -c "import docstrange" 2>/dev/null; then
    echo "[docstrange] Venv already exists and looks good"
else
    echo "[docstrange] Creating venv at $DOCSTRANGE_VENV"
    rm -rf "$DOCSTRANGE_VENV"
    uv venv "$DOCSTRANGE_VENV" --python 3.11

    echo "[docstrange] Installing torch + docstrange..."
    # docstrange needs numpy<2 and torch for GPU mode.
    # Install torch first, then docstrange (which pins numpy<2).
    uv pip install --python "$DOCSTRANGE_VENV/bin/python" \
        "torch>=2.0.0" \
        "torchvision>=0.15.0"

    uv pip install --python "$DOCSTRANGE_VENV/bin/python" \
        "docstrange>=1.1.0" \
        "pdf2image>=1.17.0" \
        "Pillow>=10.0.0" \
        "pymupdf>=1.24.0"

    echo "[docstrange] Verifying..."
    "$DOCSTRANGE_VENV/bin/python" -c "
import docstrange
print(f'  docstrange: OK')
try:
    import torch
    print(f'  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except ImportError:
    print('  torch: NOT INSTALLED')
"
fi

# ============================================================
# 6. Warm up nougat via uv (pre-cache the environment)
# ============================================================
echo ""
echo "============================================================"
echo "  Pre-caching nougat uv environment"
echo "============================================================"
# Clear any stale cached environments for nougat so uv rebuilds with
# the current PEP 723 deps (e.g., after pinning pypdfium2 or transformers).
echo "[nougat] Clearing stale uv cached environments..."
rm -rf ~/.cache/uv/environments-v2/run-nougat-* 2>/dev/null || true
echo "[nougat] Resolving and caching dependencies..."
# Run the script with no args — it prints usage and exits(1), but uv caches deps.
uv run "$SCRIPT_DIR/run_nougat.py" 2>/dev/null || true
echo "[nougat] Environment cached"

# ============================================================
# 6. Warm up marker via uv (pre-cache the environment)
# ============================================================
echo ""
echo "============================================================"
echo "  Pre-caching marker uv environment"
echo "============================================================"
echo "[marker] Clearing stale uv cached environments..."
rm -rf ~/.cache/uv/environments-v2/run-marker-* 2>/dev/null || true
echo "[marker] Resolving and caching dependencies..."
uv run "$SCRIPT_DIR/run_marker.py" 2>/dev/null || true
echo "[marker] Environment cached"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  Environments:"
echo "    marker:     uv run (PEP 723 cached)"
echo "    nougat:     uv run (PEP 723 cached)"
echo "    deepseek:   $DEEPSEEK_VENV"
echo "    paddleocr:  $PADDLE_VENV"
echo "    docstrange: $DOCSTRANGE_VENV"
echo ""
echo "  Next steps:"
echo "    1. Copy your test PDF to this directory"
echo "    2. Run: ./benchmark.sh test_pdf/Sample-PDF2.pdf"
echo ""
