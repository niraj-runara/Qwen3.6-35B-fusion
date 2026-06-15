# SGLang local changes (Phase 6)

Private edits to the sibling clone at `/sglang` (pinned **v0.5.13**).  
This repo does not vendor SGLang; use `source phase6/env.sh` so `PYTHONPATH` points at `/sglang/python` (PyPI `sglang` + `sglang-kernel` stay installed; local clone overrides Python sources only).

**Status:** M0–M4 **done**. Fusion is wired into the forward path when `--enable-qwen-fusion` + fused ckpt + `tp=1` + not quantized. E2E prefill ~**1.00×** vs Phase 3 (correctness good; 3% speed target not met). See [Results](#results) and `README.md`.

To snapshot edits after re-cloning:

```bash
cd /sglang && git diff > /Qwen3.6-35B-fusion/phase6/patches/sglang-phase6.patch
```

---

## All files changed (summary)

### New files in `/sglang`

| File | One-liner |
|------|-----------|
| `python/sglang/srt/layers/qwen_fusion_utils.py` | Fused-ckpt detection, `use_qwen_fusion()`, residual-only `prepare_attn`, shared RMS state, post-load logging |
| `python/sglang/srt/layers/fused_rmsnorm_gemm.py` | Site-1 op API: `y = linear(x,W)/rms(x)`, `RmsNormGemmState`, pair helper, custom-op + CUDA-graph bypass |
| `python/sglang/srt/layers/triton_ops/fused_rmsnorm_gemm.py` | Triton RMS + divide kernels; torch fallback during CUDA graph capture |

### Modified files in `/sglang`

| File | One-liner |
|------|-----------|
| `python/sglang/srt/environ.py` | `SGLANG_QWEN_FUSION = EnvBool(False)` |
| `python/sglang/srt/server_args.py` | `enable_qwen_fusion`, `--enable-qwen-fusion` CLI, env wiring |
| `python/sglang/srt/model_executor/model_runner.py` | `init_qwen_fusion_after_load()` after `loader.load_model()` (**primary hook**) |
| `python/sglang/srt/model_loader/loader.py` | Same post-load hook (dummy / sharded loaders) |
| `python/sglang/srt/model_loader/utils.py` | Same hook on legacy `post_load_weights()` path |
| `python/sglang/srt/layers/communicator.py` | `prepare_attn`: fusion branch skips `input_layernorm`, stashes `RmsNormGemmState` |
| `python/sglang/srt/models/qwen3_5.py` | Fused `in_proj_qkvz`/`in_proj_ba`/`qkv_proj`; alt-stream overlap preserved |

### New files in this repo (`phase6/`)

| File | One-liner |
|------|-----------|
| `env.sh` | Venv + `PYTHONPATH=/sglang/python` dev environment |
| `test_fused_rmsnorm_gemm.py` | M1: op math vs Phase 2 `Site1RmsState` |
| `test_cuda_graph_fusion.py` | M3: CUDAGraph capture/replay smoke test |
| `benchmark_sglang_fused_kernel.py` | M4: 9-shape prefill vs Phase 3 CSV |
| `launch_server.sh` | Optional HTTP server with `--enable-qwen-fusion` |
| `sync_fused_ckpt_metadata.sh` | Copy vanilla HF metadata into fused ckpt dir |

---

## Fused checkpoint & γ (how SGLang “knows”)

SGLang does **not** read a config flag like `gamma_in_weights: true`.

1. **Export** (`fused-checkpoint/export_fused_weights.py`): multiply `(1 + input_layernorm.weight)` into downstream linear weights; **zero** norm weights.
2. **Detection** (Phase 6): layer-0 `input_layernorm.weight.abs().max() < 0.15` → `fused_ckpt=True`.
3. **Forward** (when `use_qwen_fusion()`): skip `input_layernorm`; run `y = W_fused @ x / rms(x)` — no `(1+γ)` at runtime.

With fusion **off** on a fused ckpt, stock path still works (~correct) because norm weights ≈ 0 and γ is already in `W_fused`; you just don’t get the optimized kernel layout.

---

## Testing pyramid

| Level | Command | What it proves |
|-------|---------|----------------|
| Op math | `python phase6/test_fused_rmsnorm_gemm.py` | Kernel matches Phase 2 `Site1RmsState` (not full model) |
| CUDA graphs | `python phase6/test_cuda_graph_fusion.py` | Fused op replays correctly in `CUDAGraph` |
| E2E token | `python phase6/benchmark_sglang_fused_kernel.py --check-logits` | Top-1 next token vs Phase 1 oracle |
| E2E speed | same without `--check-logits` | 9-shape prefill vs Phase 3 CSV |

Also confirm at load time:

```text
Qwen Site-1 fusion: flag=on layers=64 (...) layer0_norm_max=0.00xx fused_ckpt=True active=True
```

---

## M0 — Flag, fused-ckpt detection, post-load probe (done)

### Flag wiring

| Source | Effect |
|--------|--------|
| `--enable-qwen-fusion` | Sets `ServerArgs.enable_qwen_fusion = True` |
| `SGLANG_QWEN_FUSION=1` | Same via `_handle_qwen_fusion_env()` |

### `use_qwen_fusion()` → stock path when

| Condition | Stock path |
|-----------|------------|
| `enable_qwen_fusion == False` | ✓ |
| Layer 0 `input_layernorm.weight` not ~0 | ✓ |
| `tp_size > 1` | ✓ |
| `quantization` set | ✓ |

---

## M1 — `fused_rmsnorm_gemm` kernel (done)

### Math

```text
rms(x) = sqrt(mean(x²) + eps)     per token row
y      = linear(x, W) / rms(x)    W already includes offline γ; no Gemma (1+weight)
```

### Implementation

- **GEMM:** `F.linear` / `torch.mm` (cuBLAS).
- **RMS + divide:** Triton in eager mode; torch during `get_is_capture_mode()`.
- **Shared RMS:** `RmsNormGemmState.begin(x)` once; `project()` / `fused_rmsnorm_gemm_pair()` for paired projections.

```bash
source phase6/env.sh && python phase6/test_fused_rmsnorm_gemm.py
```

---

## M2 — Forward wiring (done)

### Forward flow

```text
prepare_attn (qwen_fusion_prepare_attn)
  └─ residual add only (no GemmaRMSNorm)
  └─ RmsNormGemmState.begin(raw hidden_states)  [FP32 scratch via get_qwen_fusion_rms_buffer]
       └─ linear-attn: fused_rmsnorm_gemm on in_proj_qkvz + in_proj_ba (shared rms; alt-stream overlap)
       └─ full-attn: fused_rmsnorm_gemm on qkv_proj
  └─ MoE + attention unchanged (stock SGLang)
```

### Stock-path fallbacks

- Quantized models (`quant_format` set)
- TP scatter / `input_scattered`
- MoE allreduce–norm fusion (`_sglang_needs_allreduce_fusion`)

---

## M3 — CUDA graph capture (done)

Triton RMS/divide are not CUDA-graph replay safe. Fix:

- Capture mode: torch RMS (`copy_` into scratch) + in-place `div_` with dtype cast
- Optional `raw_out` for `torch.mm`/`addmm` `out=`
- Custom op bypassed when `get_is_capture_mode()`

```bash
source phase6/env.sh
python phase6/test_cuda_graph_fusion.py
python phase6/test_fused_rmsnorm_gemm.py   # regression
```

**Not done:** `nsys` profile confirming kernel-count reduction (optional).

---

## M4 — Benchmark harness (done)

| File | Description |
|------|-------------|
| `phase6/benchmark_sglang_fused_kernel.py` | 9-shape prefill; `benchmark_mode=sglang-fused-kernel` |
| `phase6/launch_server.sh` | Optional HTTP server |

Default backend: **`engine`** (in-process, like Phase 4/5). Passes `enable_qwen_fusion=True` to `sglang.Engine`.

```bash
source phase6/env.sh
python phase6/benchmark_sglang_fused_kernel.py --check-logits
```

Optional HTTP: `bash phase6/launch_server.sh` then `--backend http`.

**Note:** `nf_med` columns come from a **saved Phase 3 CSV**, not a same-session vanilla run — see [Results](#results).

---

## Results

E2E prefill (engine backend, fused kernel vs Phase 3 baseline CSV): **~1.00× ±1%** on all nine shapes. Best ~**1.01×** (1×128, 1×512, 8×128); a few shapes ~**0.98–0.99×** (within run-to-run noise).

| Success gate | Target | Outcome |
|--------------|--------|---------|
| Correctness (top-1) | PASS | **PASS** (`--check-logits`) |
| M1/M3 unit tests | PASS | **PASS** |
| E2E ≥ 1.03× @ 1×512 | 3% faster | **~1.01×** (not met) |
| Large shapes ≥ 1.00× | no regression | **~1.00×** (met) |

**Interpretation:** Site-1 kernel is in the hot path and correct, but attention + MoE dominate prefill; saving norm materialization on input projections does not move E2E much. Same story as Phase 5 (~1.0× with hooks). Layer-0 microbench (~1.58×) does not translate to full-model E2E.

Site-2 MoE fusion in SGLang is **out of scope** — microbench looks better but HF E2E experiments were neutral/negative; `FusedMoE` is already optimized.

---

## M5 — Remaining (optional)

- [ ] Export `phase6/patches/sglang-phase6.patch` for reproducibility
- [ ] `nsys` one-layer profile (kernel count vs stock)
- [ ] Same-session A/B benchmark mode (vanilla + fused in one run)
- [ ] Layer-0 microbench inside SGLang-loaded module

---

## Quick reference

```bash
source phase6/env.sh

# Serve with fusion
python -m sglang.launch_server \
  --model-path /data/Qwen3.6-35B-A3B-bf16-fused \
  --dtype bfloat16 --tp-size 1 --trust-remote-code \
  --enable-qwen-fusion

python phase6/test_fused_rmsnorm_gemm.py
python phase6/test_cuda_graph_fusion.py
python phase6/benchmark_sglang_fused_kernel.py --check-logits
```

**Fused ckpt metadata:** run `bash phase6/sync_fused_ckpt_metadata.sh` once (weights-only export lacks VLM processor files).
