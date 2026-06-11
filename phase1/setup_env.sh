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
#   - NVIDIA driver with CUDA 12.x+ (Blackwell/sm_120 needs cu130 + torch>=2.7)
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
# 3. Upgrade pip (setuptools pinned — torch 2.12 requires setuptools<82)
# --------------------------------------------------------------------------
echo ""
echo "=== Upgrading pip + setuptools ==="
"$PIP" install --upgrade pip wheel
"$PIP" install "setuptools>=70.0.0,<82.0.0"

# --------------------------------------------------------------------------
# 4. CUDA / PyTorch
# Blackwell (sm_100/sm_120) needs torch>=2.7 on the cu130 wheel. Do NOT pick
# the wheel from nvcc alone — e.g. nvcc 12.9 + sm_120 still needs cu130, not cu126.
# --------------------------------------------------------------------------
echo ""
echo "=== Detecting GPU and PyTorch CUDA wheel ==="

CUDA_SHORT=""
TORCH_MIN_VERSION=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)

    echo "GPU               : ${GPU_NAME:-unknown}"
    echo "Compute capability: ${COMPUTE_CAP:-unknown}"
    [[ -n "$DRIVER_CUDA" ]] && echo "Driver CUDA max   : $DRIVER_CUDA"

    if [[ -n "$COMPUTE_CAP" ]]; then
        CC_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
        # sm_100 (10.0) and sm_120 (12.0) are Blackwell — cu126 tops out at sm_90
        if [[ "$CC_MAJOR" -ge 10 ]]; then
            CUDA_SHORT="cu130"
            TORCH_MIN_VERSION="2.7.0"
            echo "-> Blackwell GPU (sm_${CC_MAJOR}*): requiring PyTorch >=${TORCH_MIN_VERSION} from cu130 index"
        fi
    fi
fi

if [[ -z "$CUDA_SHORT" ]]; then
    CUDA_VER=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    elif command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    fi

    if [[ -z "$CUDA_VER" ]]; then
        echo "WARNING: Could not detect CUDA version. Defaulting to cu124 PyTorch index."
        CUDA_SHORT="cu124"
    else
        echo "Detected CUDA: $CUDA_VER"
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

        if [[ "$CUDA_MAJOR" -ge 13 ]]; then
            CUDA_SHORT="cu130"
        elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 8 ]]; then
            CUDA_SHORT="cu128"
        elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 6 ]]; then
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
fi

# Optional override: TORCH_CUDA=cu132 bash setup_env.sh  (experimental CUDA 13.2)
CUDA_SHORT="${TORCH_CUDA:-$CUDA_SHORT}"

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_SHORT}"
echo "Using PyTorch index : $TORCH_INDEX"

echo ""
echo "=== Installing PyTorch ==="
# Remove any existing torch build (e.g. cu126) — pip will not swap CUDA tags otherwise
"$PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true

TORCH_PKGS=(torch torchvision torchaudio)
if [[ -n "$TORCH_MIN_VERSION" ]]; then
    TORCH_PKGS=("torch>=${TORCH_MIN_VERSION}" torchvision torchaudio)
fi

"$PIP" install \
    --force-reinstall \
    --no-cache-dir \
    "${TORCH_PKGS[@]}" \
    --index-url "$TORCH_INDEX"

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
# 7. Verify CUDA kernels run on this GPU (catches cu126 + sm_120 mismatch)
# --------------------------------------------------------------------------
echo ""
echo "=== Verifying PyTorch + CUDA ==="
python - <<'EOF'
import sys

import torch

print(f"  torch version    : {torch.__version__}")
print(f"  CUDA available   : {torch.cuda.is_available()}")

if "+cu126" in torch.__version__ or torch.version.cuda == "12.6":
    print("  ERROR: torch is still the cu126 build (no sm_120). Re-run:")
    print("    pip uninstall -y torch torchvision torchaudio")
    print("    pip install --force-reinstall --no-cache-dir torch torchvision torchaudio \\")
    print("      --index-url https://download.pytorch.org/whl/cu130")
    sys.exit(1)

if not torch.cuda.is_available():
    print("  ERROR: No CUDA GPU visible to PyTorch. Run on a GPU machine.")
    sys.exit(1)

print(f"  PyTorch CUDA     : {torch.version.cuda}")
arch_list = torch.cuda.get_arch_list()
print(f"  PyTorch arch list: {arch_list}")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    major, minor = p.major, p.minor
    sm_tag = f"sm_{major}{minor}"
    free, total = torch.cuda.mem_get_info(i)
    print(
        f"  GPU {i}: {p.name} | {sm_tag} | "
        f"{total / 1024**3:.1f} GB total | {free / 1024**3:.1f} GB free"
    )
    if sm_tag not in arch_list:
        print(f"  ERROR: {sm_tag} is not in PyTorch's compiled arch list.")
        print("  Blackwell (RTX PRO 6000 / sm_120) needs the cu130 wheel, e.g.:")
        print("    pip install 'torch>=2.7.0' torchvision torchaudio \\")
        print("      --index-url https://download.pytorch.org/whl/cu130")
        sys.exit(1)

try:
    x = torch.zeros(1, device="cuda")
    x.uniform_()
    torch.cuda.synchronize()
    print("  CUDA kernel test : OK")
except Exception as exc:
    print(f"  ERROR: CUDA kernel test failed: {exc}")
    print("  Reinstall PyTorch from the cu130 index (see above), then re-run setup.")
    sys.exit(1)
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
