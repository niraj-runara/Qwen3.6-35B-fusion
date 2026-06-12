#!/usr/bin/env bash
# Phase 3 — Install SGLang 0.5.x for Blackwell (sm_120) + CUDA 13 / cu130 PyTorch.
#
# SGLang 0.5.11+ uses sglang-kernel (0.4.x), NOT the legacy sgl-kernel (0.3.x).
# Wrong package → undefined symbol c10_cuda_check_implementation.
#
# Official order (CUDA 13): torch first → sglang → sglang-kernel cu130 wheel.
#
# Usage (from repo root):
#   source phase1/.venv/bin/activate
#   bash phase3/setup_sglang.sh
#   source phase3/env_sglang.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIP="${PIP:-pip}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
KERNEL_INDEX="https://docs.sglang.ai/whl/cu130/"

# sglang 0.5.13 pins these; read from sglang if already installed
SGLANG_VERSION="${SGLANG_VERSION:-0.5.13}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
KERNEL_VERSION="${KERNEL_VERSION:-0.4.3}"

echo "=== Phase 3 SGLang setup (Blackwell / cu130) ==="
echo "Python: $($PYTHON --version 2>&1)"

VENV_SITE=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")

# libnvrtc.so.13 — usually in pip nvidia-* wheels, not system CUDA
_cuda_ld_path() {
  local parts=()
  for d in \
    "${VENV_SITE}/nvidia/cuda_nvrtc/lib" \
    "${VENV_SITE}/nvidia/cuda_runtime/lib" \
    "${VENV_SITE}/nvidia/cu13/lib" \
    "${VENV_SITE}/torch/lib" \
    /usr/local/cuda/lib64; do
    [[ -d "$d" ]] && parts+=("$d")
  done
  (IFS=:; echo "${parts[*]}")
}

echo ""
echo "=== Step 1/4: PyTorch ${TORCH_VERSION} (cu130) ==="
"$PIP" uninstall -y sgl-kernel sglang-kernel 2>/dev/null || true
"$PIP" install --force-reinstall --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  "torchvision" \
  "torchaudio" \
  --index-url "$TORCH_INDEX"

echo ""
echo "=== Step 2/4: sglang[all] ${SGLANG_VERSION} (PyPI, not torch index) ==="
"$PIP" install --upgrade "sglang[all]==${SGLANG_VERSION}"

# Resolve pinned kernel version from installed sglang metadata
RESOLVED_KERNEL=$("$PYTHON" -c "
import importlib.metadata as m
for req in m.requires('sglang') or []:
    if req.startswith('sglang-kernel'):
        print(req.split('==')[1].split(',')[0])
        break
" 2>/dev/null || true)
if [[ -n "$RESOLVED_KERNEL" ]]; then
  KERNEL_VERSION="$RESOLVED_KERNEL"
fi
echo "sglang-kernel version: ${KERNEL_VERSION}"

echo ""
echo "=== Step 3/4: sglang-kernel ${KERNEL_VERSION} (cu130 wheel) ==="
"$PIP" uninstall -y sglang-kernel sgl-kernel 2>/dev/null || true

ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  WHEEL_ARCH="manylinux2014_aarch64"
else
  WHEEL_ARCH="manylinux2014_x86_64"
fi

WHEEL_URL="https://github.com/sgl-project/whl/releases/download/v${KERNEL_VERSION}/sglang_kernel-${KERNEL_VERSION}+cu130-cp310-abi3-${WHEEL_ARCH}.whl"
echo "Wheel: $WHEEL_URL"
if ! "$PIP" install --force-reinstall --no-cache-dir "$WHEEL_URL"; then
  echo "Direct wheel failed; trying cu130 index..."
  "$PIP" install --force-reinstall --no-cache-dir \
    "sglang-kernel==${KERNEL_VERSION}" \
    --index-url "$KERNEL_INDEX"
fi

echo ""
echo "=== Step 4/4: CUDA runtime libs + verify ==="
"$PIP" install --upgrade nvidia-cuda-nvrtc nvidia-cuda-runtime
CUDA_LD=$(_cuda_ld_path)
export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=${CUDA_LD}"

"$PYTHON" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import sglang
print('sglang', sglang.__version__)
import importlib.metadata as m
print('sglang-kernel', m.version('sglang-kernel'))
# legacy package must NOT shadow sglang-kernel
try:
    m.version('sgl-kernel')
    print('WARNING: legacy sgl-kernel still installed — pip uninstall sgl-kernel')
except m.PackageNotFoundError:
    pass
import sgl_kernel
print('sgl_kernel import OK')
from sglang import Engine
print('Engine import OK')
"

ENV_FILE="$SCRIPT_DIR/env_sglang.sh"
cat > "$ENV_FILE" <<EOF
# Source before Phase 3: source phase3/env_sglang.sh
export LD_LIBRARY_PATH="${CUDA_LD}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export TRITON_PTXAS_PATH="\${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
EOF

echo ""
echo "=== Done ==="
echo "  source phase3/env_sglang.sh"
echo "  python phase3/benchmark_sglang.py --model-dir /data/Qwen3.6-35B-A3B-bf16 --check-logits"
