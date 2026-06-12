#!/usr/bin/env bash
# Install the SGLang fusion plugin (editable) into the active venv.
#
# Usage:
#   source phase1/.venv/bin/activate
#   bash phase4/setup_fusion_plugin.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP="${PIP:-pip}"

echo "=== Installing qwen-sglang-fusion plugin (editable) ==="
"$PIP" install -e "$SCRIPT_DIR"

echo ""
echo "=== Verify entry point ==="
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='sglang.srt.plugins')
names = [e.name for e in eps]
assert 'qwen_fusion' in names, names
print('sglang.srt.plugins: qwen_fusion OK')
"

echo ""
echo "Enable for benchmarks / server:"
echo "  export SGLANG_PLUGINS=qwen_fusion"
echo "  export SGLANG_FUSION=1"
