#!/usr/bin/env bash
# Launch weight-fused Qwen3.6 on SGLang with native fusion plugin (Phase 5).
#
# Usage:
#   bash phase5/setup_fusion_plugin.sh   # once
#   bash phase5/launch_server_fused.sh

set -euo pipefail

VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cuda_nvrtc/lib:${VENV_SITE}/nvidia/cuda_runtime/lib:${VENV_SITE}/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SGLANG_PLUGINS="${SGLANG_PLUGINS:-qwen_fusion_native}"
export SGLANG_FUSION="${SGLANG_FUSION:-1}"
export FUSION_VARIANT="${FUSION_VARIANT:-V2}"

MODEL_DIR="${MODEL_DIR:-/data/Qwen3.6-35B-A3B-bf16-fused}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"

if ! python -c "from importlib.metadata import entry_points; assert 'qwen_fusion_native' in [e.name for e in entry_points(group='sglang.srt.plugins')]" 2>/dev/null; then
  echo "ERROR: qwen_fusion_native plugin not installed."
  echo "  bash phase5/setup_fusion_plugin.sh"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: MODEL_DIR not found: $MODEL_DIR"
  exit 1
fi

echo "============================================================"
echo "Phase 5 — SGLang native fused server"
echo "============================================================"
echo "  Model   : $MODEL_DIR"
echo "  Plugins : $SGLANG_PLUGINS  SGLANG_FUSION=$SGLANG_FUSION"
echo "  Variant : $FUSION_VARIANT"
echo "  Port    : $PORT"
echo "============================================================"

exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP" \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --trust-remote-code
