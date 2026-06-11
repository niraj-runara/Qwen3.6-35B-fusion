# Fused Checkpoint

This folder contains the export script that produces a **weight-fused** version of the vanilla Qwen3.6-35B-A3B checkpoint.

Weight fusion absorbs each RMSNorm's gamma vector into the downstream linear weights offline, so at inference time the norm becomes a trivial divide-by-rms with no gamma multiplication overhead.

---

## Prerequisites

- Phase 1 complete: vanilla checkpoint downloaded at `/data/Qwen3.6-35B-A3B-bf16`
- Phase 1 correctness oracle generated: `phase1/outputs/reference_logits.pt`
- ~70 GB free disk for the fused checkpoint output
- Same venv as Phase 1 active: `source phase1/.venv/bin/activate`

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
2. Absorb `input_layernorm` gamma into `q_proj`, `k_proj`, `v_proj` for all 64 layers
3. Absorb `post_attention_layernorm` gamma into every expert's `gate_proj` and `up_proj`
4. Verify all norm gammas are ~1.0
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

# Spot-check that a norm gamma is now 1.0
python -c "
from safetensors.torch import load_file
import torch, glob
shards = sorted(glob.glob('/data/Qwen3.6-35B-A3B-bf16-fused/*.safetensors'))
t = load_file(shards[0], device='cpu')
key = 'model.layers.0.input_layernorm.weight'
if key in t:
    w = t[key]
    print(f'{key}: mean={w.mean():.6f}  std={w.std():.6f}  (expect mean=1.0, std=0.0)')
"
```

---

## What changed vs the vanilla checkpoint

| Tensor | Before | After |
|---|---|---|
| `layers.{i}.input_layernorm.weight` | gamma values (not all 1.0) | all 1.0 |
| `layers.{i}.self_attn.q_proj.weight` | W | W × gamma |
| `layers.{i}.self_attn.k_proj.weight` | W | W × gamma |
| `layers.{i}.self_attn.v_proj.weight` | W | W × gamma |
| `layers.{i}.post_attention_layernorm.weight` | gamma values (not all 1.0) | all 1.0 |
| `layers.{i}.mlp.experts.{j}.gate_proj.weight` | W | W × gamma |
| `layers.{i}.mlp.experts.{j}.up_proj.weight` | W | W × gamma |

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
