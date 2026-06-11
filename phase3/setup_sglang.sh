#!/usr/bin/env bash
# Phase 3 — Install SGLang for Blackwell (sm_120) + CUDA 13 / cu130 PyTorch.
#
# Default pip install pulls sgl-kernel built for CUDA 12 → fails with:
#   libnvrtc.so.13: cannot open shared object file
#   [sgl_kernel] CRITICAL: Could not load any common_ops library!
#
# Usage (from repo root):
#   source phase1/.venv/bin/activate
#   bash phase3/setup_sglang.sh
#
# Then before benchmark:
#   source phase3/env_sglang.sh   # optional, sets LD_LIBRARY_PATH for libnvrtc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIP="${PIP:-pip}"
PYTHON="${PYTHON:-python3}"

echo "=== Phase 3 SGLang setup (Blackwell / cu130) ==="
echo "Python: $($PYTHON --version 2>&1)"

TORCH_CUDA=$("$PYTHON" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)
if [[ -z "$TORCH_CUDA" ]]; then
  echo "ERROR: torch not installed. Run: bash phase1/setup_env.sh"
  exit 1
fi
echo "PyTorch CUDA: $TORCH_CUDA"

CUDA_MAJOR="${TORCH_CUDA%%.*}"
if [[ "$CUDA_MAJOR" -lt 13 ]]; then
  echo "WARNING: PyTorch is not cu130 ($TORCH_CUDA). Blackwell needs cu130 — run phase1/setup_env.sh first."
fi

# CUDA runtime libs (libnvrtc.so.13, libcudart.so.13) for sgl-kernel
for d in \
  /usr/local/cuda/lib64 \
  /usr/local/cuda-13.0/targets/x86_64-linux/lib \
  /usr/local/cuda-13.0/targets/sbsa-linux/lib \
  /usr/local/cuda/targets/x86_64-linux/lib; do
  if [[ -f "$d/libnvrtc.so.13" || -L "$d/libnvrtc.so" ]]; then
    export CUDA_LIB_DIR="$d"
    break
  fi
done

echo ""
echo "=== Installing sglang (cu130 index) ==="
"$PIP" install --upgrade "sglang[all]>=0.5.10" \
  --extra-index-url "https://download.pytorch.org/whl/cu130"

echo ""
echo "=== Installing sgl-kernel cu130 wheel (required on Blackwell) ==="
"$PIP" uninstall -y sgl-kernel 2>/dev/null || true
"$PIP" install --upgrade sgl-kernel \
  --extra-index-url "https://sgl-project.github.io/whl/cu130/sgl-kernel"

echo ""
echo "=== Verify import ==="
if [[ -n "${CUDA_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "LD_LIBRARY_PATH includes: $CUDA_LIB_DIR"
fi

"$PYTHON" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import sglang
print('sglang', sglang.__version__)
import sgl_kernel
print('sgl_kernel OK')
from sglang import Engine
print('Engine import OK')
"

# Write helper env file for future shells
ENV_FILE="$SCRIPT_DIR/env_sglang.sh"
{
  echo "# Source before Phase 3 benchmarks: source phase3/env_sglang.sh"
  if [[ -n "${CUDA_LIB_DIR:-}" ]]; then
    echo "export LD_LIBRARY_PATH=\"${CUDA_LIB_DIR}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
  fi
  echo "export TRITON_PTXAS_PATH=\"\${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}\""
} > "$ENV_FILE"

echo ""
echo "=== Done ==="
echo "Before running benchmarks:"
echo "  source phase3/env_sglang.sh"
echo "  python phase3/benchmark_sglang.py --model-dir /data/Qwen3.6-35B-A3B-bf16 --check-logits"
