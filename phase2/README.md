# Phase 2 — Fused vs Unfused Benchmark

Compare **vanilla** vs **weight-fused + kernel-fused** Qwen3.6-35B-A3B at each fusion site.

- **Unfused arm:** vanilla checkpoint → `RMSNorm(x)` → `Linear(x)`
- **Fused arm:** fused checkpoint → `FusedRMSNormLinear` V2 (`linear(x) / rms(x)`, γ absorbed offline)

Metrics match `benchmark_reference.py`: median/p99 latency, throughput, peak GPU memory, numerical equivalence.

---

## Fusion sites

Each decoder layer has **two** places where we fuse “normalize, then linear” into one faster path.

### Site 1 — `attn` (before attention)

| | |
|---|---|
| **Norm** | `input_layernorm` |
| **Linears (export fuses all)** | `linear_attention` layers: `linear_attn.in_proj_{qkv,z,b,a}` |
| | `full_attention` layers: `self_attn.{q,k,v}_proj` |
| **Benchmark uses** | One representative linear — largest input projection (`in_proj_qkv` or `q_proj`) |

```
hidden → input_layernorm → token-mixer projections → attention
```

### Site 2 — `moe` (before MoE FFN)

| | |
|---|---|
| **Norm** | `post_attention_layernorm` |
| **Weights (export fuses all)** | `mlp.experts.gate_up_proj`, router `gate.weight`, `shared_expert` gate/up, `shared_expert_gate` |
| **Benchmark uses** | Gate half of `mlp.experts.gate_up_proj[0]` (representative slice) |

```
hidden → post_attention_layernorm → router + experts + shared expert → MoE output
```

Both sites use the same fusion math; only the **location in the layer** and **matrix sizes** differ.

---

## Prerequisites

1. Vanilla checkpoint: `/data/Qwen3.6-35B-A3B-bf16`
2. Fused checkpoint: `/data/Qwen3.6-35B-A3B-bf16-fused` (from `fused-checkpoint/export_fused_weights.py`)
3. Phase 1 venv with **cu130** PyTorch (Blackwell / sm_120)

```bash
cd /path/to/qwen3.6-35B-fusion
source phase1/.venv/bin/activate
```

---

## How to run

### Smoke test (recommended first)

```bash
python phase2/benchmark_fused_vs_unfused.py \
    --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \
    --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \
    --site all \
    --test-load
```

### Full sweep (both sites, 9 shape configs)

```bash
python phase2/benchmark_fused_vs_unfused.py \
    --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \
    --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \
    --site all
```

Defaults: **layer 0** (`linear_attention`), **variant V2** (CUDA stream overlap).

### Other useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--layer-idx` | `0` | Which decoder layer to extract (`3` = first `full_attention` layer) |
| `--site` | `attn` | `attn`, `moe`, or `all` |
| `--variant` | `V2` | `V1` = sequential PyTorch; `V2` = stream overlap |
| `--no-save` | off | Skip writing CSV to `phase2/results/` |

### Loading behavior (96 GB GPU)

Checkpoints are loaded **one at a time** (~70 GB each): load → extract one layer → free. Only the small fusion-site modules stay on GPU for timing. This is **not** a full-model forward benchmark.

`load_mode: full` in the CSV means the **entire checkpoint file** was loaded to extract layer weights — not that the full 35B model was timed end-to-end.

### E2E prefill (full model, separate script)

`benchmark_e2e_prefill.py` mirrors **phase3**’s shape grid and CSV schema but runs through native HuggingFace `AutoModelForCausalLM` instead of SGLang. Checkpoints are still loaded **sequentially** (unfused sweep → free → fused sweep).

```bash
# Smoke test
python phase2/benchmark_e2e_prefill.py \
    --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \
    --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \
    --test-load

# Full sweep (Site-1 kernel V2 on fused ckpt)
python phase2/benchmark_e2e_prefill.py \
    --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \
    --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \
    --variant V2
```

| Flag | Meaning |
|------|---------|
| `--no-kernel` | Fused arm uses weight-fused ckpt only (no runtime `FusedRMSNormLinear` patch) |
| `--check-logits` | Top-1 token check vs Phase 1 oracle on unfused ckpt |
| `--no-save` | Skip CSV under `phase2/results/` |

E2E speedup is **diluted** vs the microbenchmark above — fusion is only Site 1 (attn input projections) in this script; MoE Site 2 still uses stock HF forward. Compare with `phase3/benchmark_sglang.py` for vanilla SGLang prefill baseline.

Checkpoints load **one at a time** (~65 GB each on a 96 GB GPU). The script uses a context manager that strips accelerate hooks and drops all references before loading the next checkpoint. After unfused teardown you should see `[after free] GPU 0: ~0.x GB` — not ~64 GB.

---

## Results — RTX PRO 6000 Blackwell (2026-06-11)

**Run:** `benchmark_20260611T183156Z` · layer **0** · variant **V2** · `checkpoints+kernel`

GPU: NVIDIA RTX PRO 6000 Blackwell · PyTorch 2.12.0+cu130 · hidden **2048**

Raw CSVs: `phase2/results/benchmark_20260611T183156Z_qwen3_attn_V2_layer0.csv` and `..._moe_...csv`

### Site 1 — `attn` (`input_layernorm` → `in_proj_qkv`, out_dim 8192)

From `benchmark_20260611T183156Z_qwen3_attn_V2_layer0.csv`. Latency/speedup: 4 dp; `cosine_sim`: truncated to 4 dp (never rounded to 1.0000); `kl_divergence`: scientific notation, 4 dp mantissa.

| batch | seq_len | nonfused_median_ms | fused_median_ms | speedup | nonfused_p99_ms | fused_p99_ms | max_abs_diff | cosine_sim | kl_divergence |
|------:|--------:|-------------------:|----------------:|--------:|----------------:|-------------:|-------------:|-----------:|--------------:|
| 1 | 128 | 0.1098 | 0.0929 | 1.1817 | 0.1234 | 0.1085 | 0.0625 | 0.9999 | 1.8530e-05 |
| 1 | 512 | 0.1533 | 0.0973 | 1.5750 | 0.1617 | 0.1065 | 0.0625 | 0.9999 | 2.3378e-05 |
| 1 | 2048 | 0.2817 | 0.2644 | 1.0654 | 0.2904 | 0.2734 | 0.0625 | 0.9999 | 2.2103e-05 |
| 8 | 128 | 0.2015 | 0.1523 | 1.3231 | 0.2127 | 0.1610 | 0.0625 | 0.9999 | 2.5118e-05 |
| 8 | 512 | 0.5328 | 0.5239 | 1.0170 | 0.5414 | 0.5320 | 0.1250 | 0.9999 | 2.2243e-05 |
| 8 | 2048 | 2.2592 | 2.0702 | 1.0913 | 2.3036 | 2.0888 | 0.1250 | 0.9999 | 2.2842e-05 |
| 32 | 128 | 0.5324 | 0.5023 | 1.0599 | 0.5376 | 0.5237 | 0.1250 | 0.9999 | 2.3469e-05 |
| 32 | 512 | 2.1888 | 2.0697 | 1.0575 | 2.2985 | 2.0814 | 0.1250 | 0.9999 | 2.3282e-05 |
| 32 | 2048 | 8.9380 | 8.5482 | 1.0456 | 9.3513 | 8.6066 | 0.1250 | 0.9999 | 2.2382e-05 |

Best attn speedup in this run: **1.5750** at batch=1, seq=512.

### Site 2 — `moe` (`post_attention_layernorm` → expert-0 gate, out_dim 512)

From `benchmark_20260611T183156Z_qwen3_moe_V2_layer0.csv`. Same formatting as attn table above.

| batch | seq_len | nonfused_median_ms | fused_median_ms | speedup | nonfused_p99_ms | fused_p99_ms | max_abs_diff | cosine_sim | kl_divergence |
|------:|--------:|-------------------:|----------------:|--------:|----------------:|-------------:|-------------:|-----------:|--------------:|
| 1 | 128 | 0.0952 | 0.0947 | 1.0050 | 0.1107 | 0.1048 | 0.0156 | 0.9999 | 8.9276e-07 |
| 1 | 512 | 0.0997 | 0.0945 | 1.0554 | 0.1105 | 0.1072 | 0.0156 | 0.9999 | 8.4567e-07 |
| 1 | 2048 | 0.1079 | 0.0893 | 1.2081 | 0.1190 | 0.0994 | 0.0156 | 0.9999 | 6.3802e-07 |
| 8 | 128 | 0.1054 | 0.0933 | 1.1301 | 0.1212 | 0.1043 | 0.0156 | 0.9999 | 8.5375e-07 |
| 8 | 512 | 0.1512 | 0.0984 | 1.5371 | 0.1598 | 0.1060 | 0.0156 | 0.9999 | 6.3074e-07 |
| 8 | 2048 | 0.9611 | 0.5014 | 1.9169 | 0.9741 | 0.5091 | 0.0156 | 0.9999 | 7.3108e-07 |
| 32 | 128 | 0.1503 | 0.0985 | 1.5257 | 0.1577 | 0.1083 | 0.0156 | 0.9999 | 6.3300e-07 |
| 32 | 512 | 0.9611 | 0.5018 | 1.9153 | 0.9693 | 0.5135 | 0.0156 | 0.9999 | 7.3418e-07 |
| 32 | 2048 | 4.0819 | 2.1862 | 1.8671 | 4.1139 | 2.1970 | 0.0156 | 0.9999 | 6.3325e-07 |

Best moe speedup in this run: **1.9169** at batch=8, seq=2048.

### Takeaways

- **Numerics:** `cosine_sim` ≈ 0.9999 (not exactly 1); `kl_divergence` ≈ 1e-05 (attn) or 1e-07 (moe).
- **MoE site** shows stronger speedups than **attn** on this layer at larger token counts.
- **Attn site** peaks at speedup **1.5750** (batch=1, seq=512); minimum **1.0170** (batch=8, seq=512).
- Results are for **layer 0 only** (`linear_attention`). Re-run with `--layer-idx 3` for a `full_attention` layer.

---

## CSV columns

| Column | Meaning |
|--------|---------|
| `fusion_point` | `attn` or `moe` |
| `benchmark_mode` | `checkpoints+kernel` (vanilla vs fused ckpt + V2) |
| `variant` | `V1` or `V2` kernel |
| `load_mode` | `full` = whole checkpoint loaded to extract layer weights |
| `fused_median_ms` / `nonfused_median_ms` | Median forward time for that fusion pair |
| `speedup` | `nonfused / fused` (>1 = fused faster) |
| `fused_throughput` | tokens/sec for the micro-op |
| `fused_peak_mem_mb` | Peak GPU memory during fused forward |
| `max_abs_diff` | Max elementwise difference (fused vs unfused output) |
| `cosine_sim` / `kl_divergence` | Distribution similarity of outputs |

---

## Files

| File | Purpose |
|------|---------|
| `benchmark_fused_vs_unfused.py` | Per-site microbenchmark (one decoder layer) |
| `benchmark_e2e_prefill.py` | Full-model HF prefill (fused vs unfused) |
| `patch_hf_kernel_fusion.py` | Site-1 runtime kernel patch for HF decoder layers |
| `fusion_bf16.py` | `FusedRMSNormLinear` V1/V2 modules |
| `results/*.csv` | Timestamped benchmark outputs |

---

## Next

- Benchmark other layers: `--layer-idx 3` (full attention), `7`, `11`, …
- Run E2E prefill and compare with Phase 3 SGLang baseline on the same GPU
- Phase 4: fused ckpt + kernel hooks in SGLang for serving speedup
