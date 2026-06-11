#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Download Qwen3.6-35B-A3B from HuggingFace
#
# Downloads the full BF16 checkpoint and verifies the result.
#
# Usage:
#   bash download_model.sh                        # downloads to MODEL_DIR default
#   MODEL_DIR=/nvme/models bash download_model.sh # custom path
#   bash download_model.sh --verify-only          # skip download, just verify
#
# Environment variables:
#   MODEL_DIR      Where to save the checkpoint (default: /data/Qwen3.6-35B-A3B-bf16)
#   HF_TOKEN       HuggingFace token (optional; needed if repo requires auth)
#   HF_HUB_CACHE   Override HF cache dir (optional)
#
# The script:
#   1. Checks huggingface_hub / huggingface-cli is installed
#   2. Downloads the full repo (all safetensors shards + config files)
#   3. Verifies:
#       - model.safetensors.index.json is present
#       - number of shards matches the index
#       - spot-checks key tensor names in the index
#       - prints total checkpoint size on disk
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
MODEL_DIR="${MODEL_DIR:-/data/Qwen3.6-35B-A3B-bf16}"
VERIFY_ONLY=0
HF_TOKEN="${HF_TOKEN:-}"

for arg in "$@"; do
    case "$arg" in
        --verify-only) VERIFY_ONLY=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Model ID  : $MODEL_ID"
echo "  Local dir : $MODEL_DIR"
echo "  Mode      : $([ $VERIFY_ONLY -eq 1 ] && echo 'verify only' || echo 'download + verify')"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 1. Check huggingface-cli is available
# --------------------------------------------------------------------------
echo "=== Checking huggingface-cli ==="
if ! command -v huggingface-cli &>/dev/null; then
    echo "huggingface-cli not found. Installing huggingface_hub..."
    pip install --upgrade huggingface_hub
fi

HF_CLI_VER=$(huggingface-cli --version 2>&1 | head -1)
echo "  $HF_CLI_VER"

# --------------------------------------------------------------------------
# 2. Check available disk space
# --------------------------------------------------------------------------
echo ""
echo "=== Checking disk space ==="

# Parent dir of MODEL_DIR (create if needed for df)
PARENT_DIR=$(dirname "$MODEL_DIR")
mkdir -p "$PARENT_DIR"

AVAIL_KB=$(df -k "$PARENT_DIR" | awk 'NR==2 {print $4}')
AVAIL_GB=$(echo "scale=1; $AVAIL_KB / 1048576" | bc 2>/dev/null || echo "unknown")
echo "  Available at $PARENT_DIR: ${AVAIL_GB} GB"

# Qwen3.6-35B-A3B BF16 is ~70 GB; warn if less than 80 GB free
if [[ "$AVAIL_KB" != "unknown" ]] && [[ "$AVAIL_KB" -lt 83886080 ]]; then
    echo "  WARNING: Less than 80 GB free. Download may fail."
    echo "  Required: ~70 GB for model weights + a few hundred MB for configs."
    read -r -p "  Continue anyway? [y/N] " yn
    if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 3. Download
# --------------------------------------------------------------------------
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    echo ""
    echo "=== Downloading $MODEL_ID ==="
    echo "  Destination : $MODEL_DIR"
    echo "  This will download ~70 GB. Progress is shown per file."
    echo ""

    TOKEN_ARG=""
    if [[ -n "$HF_TOKEN" ]]; then
        TOKEN_ARG="--token $HF_TOKEN"
        echo "  Using HF_TOKEN for authentication."
    fi

    huggingface-cli download \
        "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        $TOKEN_ARG

    echo ""
    echo "Download finished."
fi

# --------------------------------------------------------------------------
# 4. Verify the checkpoint
# --------------------------------------------------------------------------
echo ""
echo "=== Verifying checkpoint at $MODEL_DIR ==="

# 4a. Directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Directory does not exist: $MODEL_DIR"
    exit 1
fi
echo "  [OK] Directory exists"

# 4b. Index file present
INDEX_FILE="$MODEL_DIR/model.safetensors.index.json"
if [[ ! -f "$INDEX_FILE" ]]; then
    echo "ERROR: model.safetensors.index.json not found in $MODEL_DIR"
    echo "  Expected a sharded safetensors checkpoint."
    exit 1
fi
echo "  [OK] model.safetensors.index.json present"

# 4c. Count shards listed in index vs. files on disk
echo ""
echo "  Counting shards..."
python3 - <<PYEOF
import json, os, sys

model_dir = "$MODEL_DIR"
index_path = os.path.join(model_dir, "model.safetensors.index.json")

with open(index_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]

# Unique shard filenames referenced in the index
expected_shards = sorted(set(weight_map.values()))
n_expected = len(expected_shards)

# Shards actually on disk
missing = []
for shard in expected_shards:
    shard_path = os.path.join(model_dir, shard)
    if not os.path.exists(shard_path):
        missing.append(shard)

n_on_disk = n_expected - len(missing)

print(f"  Shards expected : {n_expected}")
print(f"  Shards on disk  : {n_on_disk}")

if missing:
    print(f"  MISSING shards  : {missing}")
    sys.exit(1)
else:
    print(f"  [OK] All {n_expected} shards present")

# Print total size
total_bytes = sum(
    os.path.getsize(os.path.join(model_dir, f))
    for f in os.listdir(model_dir)
    if f.endswith(".safetensors")
)
print(f"  Total safetensors size: {total_bytes / 1024**3:.2f} GB")

# Spot-check tensor names: expect these key patterns for Qwen3-MoE
required_patterns = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
]

print("")
print("  Spot-checking expected tensor names:")
found_all = True
all_keys = set(weight_map.keys())
for pat in required_patterns:
    found = pat in all_keys
    status = "[OK]" if found else "[MISSING]"
    print(f"    {status}  {pat}")
    if not found:
        found_all = False

# Also check MoE expert tensors
moe_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
found_moe = moe_key in all_keys
status = "[OK]" if found_moe else "[MISSING]"
print(f"    {status}  {moe_key}")
if not found_moe:
    found_all = False

# Count how many expert layers are present in layer 0
expert_keys = [k for k in all_keys if k.startswith("model.layers.0.mlp.experts.")]
expert_ids = sorted(set(
    int(k.split("model.layers.0.mlp.experts.")[1].split(".")[0])
    for k in expert_keys
))
print(f"")
print(f"  Experts found in layer 0: {len(expert_ids)} (ids 0..{max(expert_ids) if expert_ids else 'none'})")

if not found_all:
    print("")
    print("  ERROR: Some expected tensors are missing. The download may be incomplete or this is a different model.")
    sys.exit(1)

print("")
print("  [OK] All spot-checks passed")
PYEOF

# 4d. Config files
echo ""
echo "  Checking config files:"
for f in config.json tokenizer_config.json tokenizer.json generation_config.json; do
    if [[ -f "$MODEL_DIR/$f" ]]; then
        echo "    [OK]  $f"
    else
        echo "    [MISSING]  $f  (may be optional)"
    fi
done

# 4e. Print model config summary
echo ""
echo "  Model config summary:"
python3 - <<PYEOF
import json, os

cfg_path = os.path.join("$MODEL_DIR", "config.json")
if not os.path.exists(cfg_path):
    print("  config.json not found — skipping.")
    exit()

with open(cfg_path) as f:
    cfg = json.load(f)

fields = [
    ("model_type",              "model_type"),
    ("num_hidden_layers",       "num_hidden_layers"),
    ("hidden_size",             "hidden_size"),
    ("num_attention_heads",     "num_attention_heads"),
    ("num_key_value_heads",     "num_key_value_heads"),
    ("num_experts",             "num_experts"),
    ("num_experts_per_tok",     "num_experts_per_tok"),
    ("intermediate_size",       "moe_intermediate_size"),
    ("vocab_size",              "vocab_size"),
    ("torch_dtype",             "torch_dtype"),
]

for label, key in fields:
    val = cfg.get(key, "—")
    print(f"    {label:<28}: {val}")
PYEOF

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Verification complete. Checkpoint is ready at:"
echo "  $MODEL_DIR"
echo ""
echo "  Next: run phase1 inspection + benchmark"
echo "    python run_phase1.py --all --model-dir $MODEL_DIR"
echo "============================================================"
