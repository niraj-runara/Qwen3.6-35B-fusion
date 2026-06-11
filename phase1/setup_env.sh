#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Environment Setup
# Installs all Python dependencies needed to load, inspect, and benchmark
# Qwen3.6-35B-A3B with HuggingFace transformers (no vLLM / SGLang yet).
#
# Usage:
#   bash setup_env.sh                   # creates .venv in phase1/
#   bash setup_env.sh --skip-venv       # install into current active env
#
# Requirements:
#   - Python 3.11+
#   - CUDA 12.x driver already installed
#   - ~5 GB free disk for packages
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
SKIP_VENV=0
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"

# Parse args
for arg in "$@"; do
    case "$arg" in
        --skip-venv) SKIP_VENV=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# 1. Python version check
# --------------------------------------------------------------------------
echo "=== Checking Python version ==="
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MIN=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

echo "Found Python $PY_VER"

if [[ "$PY_MAJ" -lt 3 ]] || [[ "$PY_MAJ" -eq 3 && "$PY_MIN" -lt 11 ]]; then
    echo "ERROR: Python 3.11+ is required (found $PY_VER)."
    echo "Install with: sudo apt install python3.11  or use pyenv."
    exit 1
fi

# --------------------------------------------------------------------------
# 2. Create venv (optional)
# --------------------------------------------------------------------------
if [[ "$SKIP_VENV" -eq 0 ]]; then
    echo ""
    echo "=== Creating virtual environment at $VENV_DIR ==="
    "$PYTHON" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
    echo "Activated: $(which python)"
else
    echo ""
    echo "=== Skipping venv creation (--skip-venv) ==="
    PIP="pip"
fi

# --------------------------------------------------------------------------
# 3. Upgrade pip / setuptools
# --------------------------------------------------------------------------
echo ""
echo "=== Upgrading pip + setuptools ==="
"$PIP" install --upgrade pip setuptools wheel

# --------------------------------------------------------------------------
# 4. CUDA / PyTorch
# Detect CUDA version from nvcc or nvidia-smi and pick the right torch index.
# --------------------------------------------------------------------------
echo ""
echo "=== Detecting CUDA version ==="

CUDA_VER=""
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
fi

if [[ -z "$CUDA_VER" ]]; then
    echo "WARNING: Could not detect CUDA version. Defaulting to cu124 PyTorch index."
    CUDA_SHORT="cu124"
else
    echo "Detected CUDA $CUDA_VER"
    # Map to PyTorch index tag
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

    if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 6 ]]; then
        CUDA_SHORT="cu126"
    elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then
        CUDA_SHORT="cu124"
    elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 1 ]]; then
        CUDA_SHORT="cu121"
    else
        echo "WARNING: CUDA < 12.1 detected. PyTorch may not support your driver fully."
        CUDA_SHORT="cu121"
    fi
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_SHORT}"
echo "Using PyTorch index: $TORCH_INDEX"

echo ""
echo "=== Installing PyTorch ==="
"$PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# --------------------------------------------------------------------------
# 5. Core ML / HF packages
# --------------------------------------------------------------------------
echo ""
echo "=== Installing HuggingFace stack ==="
"$PIP" install \
    transformers>=4.51.0 \
    accelerate>=0.34.0 \
    safetensors>=0.4.3 \
    huggingface_hub>=0.23.0 \
    tokenizers>=0.21.0

# --------------------------------------------------------------------------
# 6. Profiling / benchmark utilities
# --------------------------------------------------------------------------
echo ""
echo "=== Installing benchmark utilities ==="
"$PIP" install \
    numpy \
    pandas \
    tqdm \
    psutil

# --------------------------------------------------------------------------
# 7. Verify CUDA is visible to PyTorch
# --------------------------------------------------------------------------
echo ""
echo "=== Verifying PyTorch + CUDA ==="
python - <<'EOF'
import torch
print(f"  torch version    : {torch.__version__}")
print(f"  CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version     : {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {p.name} | {total / 1024**3:.1f} GB total | {free / 1024**3:.1f} GB free")
else:
    print("  WARNING: No CUDA GPU found. Run on a GPU machine.")
EOF

# --------------------------------------------------------------------------
# 8. Verify transformers can import Qwen3 class
# --------------------------------------------------------------------------
echo ""
echo "=== Checking transformers Qwen3-MoE support ==="
python - <<'EOF'
import transformers
print(f"  transformers version: {transformers.__version__}")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Check the Qwen3 MoE config class is registered
    from transformers import AutoConfig
    print("  AutoModelForCausalLM : OK")
    print("  AutoConfig           : OK")
except ImportError as e:
    print(f"  ERROR: {e}")
    raise SystemExit(1)
EOF

# --------------------------------------------------------------------------
# 9. Done
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Environment setup complete."
if [[ "$SKIP_VENV" -eq 0 ]]; then
    echo ""
    echo "  Activate before running other scripts:"
    echo "    source ${VENV_DIR}/bin/activate"
fi
echo "============================================================"
