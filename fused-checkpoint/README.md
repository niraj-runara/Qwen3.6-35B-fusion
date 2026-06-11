# Fused Checkpoint

This folder contains the export script that produces a **weight-fused** version of the vanilla Qwen3.6-35B-A3B checkpoint.

Weight fusion absorbs each RMSNorm's gamma vector into the downstream linear weights offline, so at inference time the norm becomes a trivial divide-by-rms with no gamma multiplication overhead.

---

## Prerequisites

1. Vanilla checkpoint at `/data/Qwen3.6-35B-A3B-bf16` (`bash phase1/download_model.sh`)
2. Phase 1 correctness oracle — **required for `--check`**:
   ```bash
   source phase1/.venv/bin/activate
   python phase1/run_phase1.py --reference
   ```
   Produces `phase1/outputs/reference_logits.pt`
3. ~70 GB free disk for the fused output
4. Same venv active: `source phase1/.venv/bin/activate`

---

## Step 1 — Dry run (verify correctness, no disk write)

Always do this first. It fuses the model in memory and checks the output logits match the Phase 1 oracle before writing anything to disk.

```bash
cd fused-checkpoint

python export_fused_weights.py --dry-run --check
```

Expected output:
```
[correctness] max |logit diff| = 0.00xxxx  (threshold: 0.01)
oracle top-1 id = XXXXX  |  fused top-1 id = XXXXX  |  match = True
PASS = True
[dry-run] Skipping save.
```

If `PASS = False`, do not proceed — the weight absorption has a bug.

---

## Step 2 — Export

Once the dry run passes, export the fused checkpoint to disk:

```bash
python export_fused_weights.py --check
```

This will:
1. Load the vanilla BF16 model (~70 GB, takes a few minutes)
2. Fuse site 1 per layer type: `linear_attn.in_proj_*` or `self_attn.q/k/v_proj`
3. Fuse site 2: `mlp.experts.gate_up_proj`, router, shared expert weights
4. Verify all `*layernorm.weight` tensors are ~0 (Qwen3.5 norm scale = `1 + weight`)
5. Run the correctness check against the Phase 1 oracle
6. Save the fused checkpoint to `/data/Qwen3.6-35B-A3B-bf16-fused`

**Custom paths:**
```bash
python export_fused_weights.py \
    --src /your/path/Qwen3.6-35B-A3B-bf16 \
    --dst /your/path/Qwen3.6-35B-A3B-bf16-fused \
    --check
```

---

## Step 3 — Verify the output

After export, confirm the fused checkpoint looks right:

```bash
# Check the index and shard count
python -c "
import json
with open('/data/Qwen3.6-35B-A3B-bf16-fused/model.safetensors.index.json') as f:
    idx = json.load(f)
print('Tensors in index:', len(idx['weight_map']))
print('Unique shards:', len(set(idx['weight_map'].values())))
"

# Spot-check that a norm weight is now ~0 (effective scale 1 + weight = 1)
python -c "
from safetensors.torch import load_file
import glob
shards = sorted(glob.glob('/data/Qwen3.6-35B-A3B-bf16-fused/*.safetensors'))
t = load_file(shards[0], device='cpu')
key = 'model.layers.0.input_layernorm.weight'
if key in t:
    w = t[key]
    print(f'{key}: mean={w.float().mean():.6f}  std={w.float().std():.6f}  (expect ~0)')
"
```

---

## What changed vs the vanilla checkpoint

| Tensor | Before | After |
|---|---|---|
| `layers.{i}.input_layernorm.weight` | learned scale (1 + weight) | all 0.0 |
| `linear_attention` layers: `linear_attn.in_proj_{qkv,z,b,a}.weight` | W | W × gamma |
| `full_attention` layers: `self_attn.{q,k,v}_proj.weight` | W | W × gamma |
| `layers.{i}.post_attention_layernorm.weight` | learned scale (1 + weight) | all 0.0 |
| `layers.{i}.mlp.experts.gate_up_proj` | W (gate+up fused `[E,2I,H]`) | W × gamma |
| `layers.{i}.mlp.gate.weight` (router) | W | W × gamma |
| `layers.{i}.mlp.shared_expert.{gate,up}_proj.weight` | W | W × gamma |
| `layers.{i}.mlp.shared_expert_gate.weight` | W | W × gamma |

All other tensors (embeddings, `o_proj`, `down_proj`, etc.) are unchanged.

---

## Next

Once the fused checkpoint is on disk, go to Phase 2 to benchmark it:

```bash
python phase2/benchmark_fused_vs_unfused.py \
    --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \
    --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \
    --mode checkpoints \
    --site all
```
