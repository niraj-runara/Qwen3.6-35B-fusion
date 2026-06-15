#!/usr/bin/env bash
# Launch fused Qwen3.6 on local SGLang clone with Site-1 kernel fusion (Phase 6).
#
# Usage:
#   source phase6/env.sh
#   bash phase6/launch_server.sh
#
# Optional:
#   FUSED_DIR=/data/Qwen3.6-35B-A3B-bf16-fused PORT=30000 bash phase6/launch_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"

FUSED_DIR="${FUSED_DIR:-/data/Qwen3.6-35B-A3B-bf16-fused}"
VANILLA_DIR="${VANILLA_DIR:-/data/Qwen3.6-35B-A3B-bf16}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"

if [[ ! -d "$FUSED_DIR" ]]; then
  echo "ERROR: FUSED_DIR not found: $FUSED_DIR"
  exit 1
fi

bash "${SCRIPT_DIR}/sync_fused_ckpt_metadata.sh" \
  VANILLA_DIR="$VANILLA_DIR" FUSED_DIR="$FUSED_DIR" || true

export SGLANG_QWEN_FUSION=1

echo "============================================================"
echo "Phase 6 — SGLang fused kernel server"
echo "============================================================"
echo "  Model   : $FUSED_DIR"
echo "  Host    : $HOST"
echo "  Port    : $PORT"
echo "  TP      : $TP"
echo "  Context : $CONTEXT_LENGTH"
echo "  Fusion  : --enable-qwen-fusion"
echo ""
echo "Health: curl http://127.0.0.1:${PORT}/health"
echo "Benchmark (separate terminal):"
echo "  source phase6/env.sh"
echo "  python phase6/benchmark_sglang_fused_kernel.py --check-logits"
echo "============================================================"

exec python -m sglang.launch_server \
  --model-path "$FUSED_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP" \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --trust-remote-code \
  --enable-qwen-fusion
