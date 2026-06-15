#!/usr/bin/env bash
# Copy HF/SGLang metadata from vanilla ckpt into fused ckpt (weights-only export).
#
# Usage:
#   source phase1/.venv/bin/activate
#   bash phase6/sync_fused_ckpt_metadata.sh
#
# Optional:
#   VANILLA_DIR=/data/Qwen3.6-35B-A3B-bf16 \
#   FUSED_DIR=/data/Qwen3.6-35B-A3B-bf16-fused \
#   bash phase6/sync_fused_ckpt_metadata.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VANILLA_DIR="${VANILLA_DIR:-/data/Qwen3.6-35B-A3B-bf16}"
FUSED_DIR="${FUSED_DIR:-/data/Qwen3.6-35B-A3B-bf16-fused}"

export PYTHONPATH="${REPO_ROOT}/phase5${PYTHONPATH:+:${PYTHONPATH}}"

python -c "
from patch_sglang_kernel_fusion import sync_fused_config_architectures
import os
v, f = os.environ['VANILLA_DIR'], os.environ['FUSED_DIR']
if sync_fused_config_architectures(v, f):
    print('Metadata synced into', f)
else:
    print('No changes (missing paths or already up to date)')
" VANILLA_DIR="$VANILLA_DIR" FUSED_DIR="$FUSED_DIR"
