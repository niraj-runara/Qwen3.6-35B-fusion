#!/usr/bin/env bash
# Launch vanilla Qwen3.6-35B-A3B on SGLang (Phase 3).
#
# Usage:
#   bash phase3/launch_server.sh
#   MODEL_DIR=/data/Qwen3.6-35B-A3B-bf16 PORT=30000 bash phase3/launch_server.sh
#
# Requires: pip install "sglang[all]"  (see phase3/README.md)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/Qwen3.6-35B-A3B-bf16}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: MODEL_DIR not found: $MODEL_DIR"
  echo "Run phase1/download_model.sh first."
  exit 1
fi

if ! python -c "import sglang" 2>/dev/null; then
  echo "ERROR: sglang is not installed."
  echo "  pip install \"sglang[all]\""
  exit 1
fi

echo "============================================================"
echo "Phase 3 — SGLang vanilla server"
echo "============================================================"
echo "  Model : $MODEL_DIR"
echo "  Host  : $HOST"
echo "  Port  : $PORT"
echo "  TP    : $TP"
echo "  dtype : bfloat16"
echo ""
echo "Health: curl http://127.0.0.1:${PORT}/health"
echo "Benchmark (separate terminal):"
echo "  python phase3/benchmark_sglang.py --backend http --server-url http://127.0.0.1:${PORT}"
echo "============================================================"

exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION" \
  --trust-remote-code
