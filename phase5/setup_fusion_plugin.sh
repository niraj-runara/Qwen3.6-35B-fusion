#!/usr/bin/env bash
# Install Phase 5 native SGLang fusion plugin + sync fused checkpoint metadata.
#
# Usage:
#   source phase1/.venv/bin/activate
#   bash phase5/setup_fusion_plugin.sh
#
# Optional:
#   VANILLA_DIR=/data/Qwen3.6-35B-A3B-bf16 \
#   FUSED_DIR=/data/Qwen3.6-35B-A3B-bf16-fused \
#   bash phase5/setup_fusion_plugin.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP="${PIP:-pip}"
VANILLA_DIR="${VANILLA_DIR:-/data/Qwen3.6-35B-A3B-bf16}"
FUSED_DIR="${FUSED_DIR:-/data/Qwen3.6-35B-A3B-bf16-fused}"

echo "=== Installing qwen-sglang-fusion-native plugin (editable) ==="
"$PIP" install -e "$SCRIPT_DIR"

echo ""
echo "=== Verify entry point ==="
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='sglang.srt.plugins')
names = [e.name for e in eps]
assert 'qwen_fusion_native' in names, names
print('sglang.srt.plugins: qwen_fusion_native OK')
"

echo ""
echo "=== Fused checkpoint config (copy vanilla metadata for SGLang) ==="
VANILLA_DIR="$VANILLA_DIR" FUSED_DIR="$FUSED_DIR" python -c "
from patch_sglang_kernel_fusion import sync_fused_config_architectures
import os
v, f = os.environ['VANILLA_DIR'], os.environ['FUSED_DIR']
if sync_fused_config_architectures(v, f):
    print('  updated')
else:
    print(f'  skipped (paths missing or already OK): {v} / {f}')
"

echo ""
echo "Enable for benchmarks / server:"
echo "  export SGLANG_PLUGINS=qwen_fusion_native"
echo "  export SGLANG_FUSION=1"
