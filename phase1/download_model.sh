#!/usr/bin/env bash
# Download Qwen3.6-35B-A3B from HuggingFace
#
# Usage:
#   bash download_model.sh
#   MODEL_DIR=/nvme/models bash download_model.sh

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_DIR="${MODEL_DIR:-/data/Qwen3.6-35B-A3B-bf16}"

# Create destination folder if it doesn't exist
mkdir -p "$MODEL_DIR"

echo "Downloading $MODEL_ID -> $MODEL_DIR"

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: 'hf' not found. Install with: pip install -U huggingface_hub" >&2
    exit 1
fi

hf download \
    "$MODEL_ID" \
    --local-dir "$MODEL_DIR"

echo "Done. Model saved to $MODEL_DIR"
