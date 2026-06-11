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

---

## Results — RTX PRO 6000 Blackwell (2026-06-11)

**Run:** `benchmark_20260611T183156Z` · layer **0** · variant **V2** · `checkpoints+kernel`

GPU: NVIDIA RTX PRO 6000 Blackwell · PyTorch 2.12.0+cu130 · hidden **2048**

Raw CSVs: `phase2/results/benchmark_20260611T183156Z_qwen3_attn_V2_layer0.csv` and `..._moe_...csv`

### Site 1 — `attn` (`input_layernorm` → `in_proj_qkv`, out_dim 8192)

| batch | seq_len | unfused ms | fused ms | speedup | max \|diff\| | cosine sim |
|------:|--------:|-----------:|---------:|--------:|-------------:|-----------:|
| 1 | 128 | 0.110 | 0.093 | **1.18×** | 0.0625 | 1.0000 |
| 1 | 512 | 0.153 | 0.097 | **1.58×** | 0.0625 | 1.0000 |
| 1 | 2048 | 0.282 | 0.264 | **1.07×** | 0.0625 | 1.0000 |
| 8 | 128 | 0.202 | 0.152 | **1.32×** | 0.0625 | 1.0000 |
| 8 | 512 | 0.533 | 0.524 | **1.02×** | 0.1250 | 1.0000 |
| 8 | 2048 | 2.259 | 2.070 | **1.09×** | 0.1250 | 1.0000 |
| 32 | 128 | 0.532 | 0.502 | **1.06×** | 0.1250 | 1.0000 |
| 32 | 512 | 2.189 | 2.070 | **1.06×** | 0.1250 | 1.0000 |
| 32 | 2048 | 8.938 | 8.548 | **1.05×** | 0.1250 | 1.0000 |

Best attn speedup: **1.58×** at batch=1, seq=512. Gains are modest when the matmul dominates (large batch × seq).

### Site 2 — `moe` (`post_attention_layernorm` → expert-0 gate, out_dim 512)

| batch | seq_len | unfused ms | fused ms | speedup | max \|diff\| | cosine sim |
|------:|--------:|-----------:|---------:|--------:|-------------:|-----------:|
| 1 | 128 | 0.095 | 0.095 | **1.01×** | 0.0156 | 1.0000 |
| 1 | 512 | 0.100 | 0.095 | **1.06×** | 0.0156 | 1.0000 |
| 1 | 2048 | 0.108 | 0.089 | **1.21×** | 0.0156 | 1.0000 |
| 8 | 128 | 0.105 | 0.093 | **1.13×** | 0.0156 | 1.0000 |
| 8 | 512 | 0.151 | 0.098 | **1.54×** | 0.0156 | 1.0000 |
| 8 | 2048 | 0.961 | 0.501 | **1.92×** | 0.0156 | 1.0000 |
| 32 | 128 | 0.150 | 0.099 | **1.53×** | 0.0156 | 1.0000 |
| 32 | 512 | 0.961 | 0.502 | **1.92×** | 0.0156 | 1.0000 |
| 32 | 2048 | 4.082 | 2.186 | **1.87×** | 0.0156 | 1.0000 |

Best moe speedup: **~1.9×** at batch≥8, seq≥512. Smaller gate matrix → norm overhead was a larger fraction of unfused time.

### Takeaways

- **Numerics:** cosine similarity ≈ 1.0 and tiny KL on all shapes — fusion is faithful at the op level.
- **MoE site** shows stronger speedups than **attn** on this layer, especially at larger token counts.
- **Attn site** peaks around **1.5×** for small batch prefill-like shapes; approaches **~1.05×** when compute-bound.
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
| `benchmark_fused_vs_unfused.py` | Main benchmark script |
| `fusion_bf16.py` | `FusedRMSNormLinear` V1/V2 modules |
| `results/*.csv` | Timestamped benchmark outputs |

---

## Next

- Benchmark other layers: `--layer-idx 3` (full attention), `7`, `11`, …
- Full-model prefill/decode (no kernel patch yet): `phase1/run_phase1.py --benchmark` on each checkpoint separately
- Phase 3+: integrate fused path into SGLang / vLLM for end-to-end serving speedup
