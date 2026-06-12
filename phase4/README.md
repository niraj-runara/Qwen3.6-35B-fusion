# Phase 4 — Fused Qwen3.6 on SGLang

Weight-fused checkpoint + **Site-1 kernel fusion** (same fast-path as Phase 2 HF E2E), benchmarked against **Phase 3** vanilla SGLang CSV.

| Phase | Backend | Checkpoint | Kernel | CSV `benchmark_mode` |
|-------|---------|------------|--------|----------------------|
| 3 | SGLang | vanilla | none | `sglang-vanilla` |
| **4** | SGLang | **fused** | **Site-1 (default)** | `sglang-fused` |

**Metrics** (same schema as Phase 2/3):

- `fused_median_ms`, `fused_p99_ms`, `fused_throughput` — this run (fused ckpt)
- `nonfused_median_ms`, … — copied from Phase 3 baseline CSV
- `speedup` = `nonfused_median_ms / fused_median_ms`

---

## Prerequisites

1. Phase 3 engine baseline CSV in `phase3/results/`
2. Fused checkpoint: `/data/Qwen3.6-35B-A3B-bf16-fused` (`fused-checkpoint/export_fused_weights.py`)
3. Phase 1 + Phase 3 SGLang env (`phase3/setup_sglang.sh`)
4. Fusion plugin:

```bash
source phase1/.venv/bin/activate
bash phase4/setup_fusion_plugin.sh
```

---

## How fusion is applied

SGLang runs the model in a **scheduler subprocess**. Patching from the benchmark process does not work — we use an official **SGLang plugin** (`SGLANG_PLUGINS=qwen_fusion`) that hooks `post_load_weights` and reuses `phase2/patch_hf_kernel_fusion.py` (Site-1: one `x/rms(x)` per layer, stock linears).

```bash
export SGLANG_PLUGINS=qwen_fusion
export SGLANG_FUSION=1          # 0 = fused weights only (--no-kernel)
export FUSION_VARIANT=V2        # optional
export SGLANG_FUSION_SITE2=0    # 1 = experimental MoE patch (usually slower)
```

---

## Benchmark (engine — recommended)

```bash
source phase1/.venv/bin/activate
export SGLANG_PLUGINS=qwen_fusion
export SGLANG_FUSION=1

python phase4/benchmark_sglang_fused.py \
    --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused \
    --baseline-csv phase3/results/benchmark_20260612T001855Z_sglang-vanilla_prefill_na_full.csv \
    --check-logits
```

Omit `--baseline-csv` to auto-pick the latest `phase3/results/benchmark_*_sglang-vanilla_*.csv`.

**Weights only** (no runtime kernel patch):

```bash
python phase4/benchmark_sglang_fused.py \
    --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused \
    --no-kernel
```

---

## HTTP server (optional)

```bash
bash phase4/launch_server_fused.sh

python phase4/benchmark_sglang_fused.py \
    --backend http \
    --server-url http://127.0.0.1:30000 \
    --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused
```

---

## Output

`phase4/results/benchmark_<timestamp>_sglang-fused_prefill_V2_full.csv`

Compare `speedup` column to Phase 2 HF E2E — expect larger gains here if SGLang’s fused path avoids extra norm work.

---

## Files

| File | Purpose |
|------|---------|
| `setup_fusion_plugin.sh` | `pip install -e phase4/` (registers `qwen_fusion` plugin) |
| `sglang_fusion_plugin.py` | `post_load_weights` hook |
| `patch_sglang_kernel_fusion.py` | Resolve `model.layers`, apply Phase 2 Site-1 patch |
| `benchmark_sglang_fused.py` | 9-shape prefill sweep + speedup vs Phase 3 |
| `launch_server_fused.sh` | Fused model HTTP server |
