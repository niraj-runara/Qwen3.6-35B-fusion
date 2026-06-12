# Qwen3.6-35B-A3B RMSNorm + Linear Fusion

Offline weight fusion and runtime kernel fusion for **Qwen/Qwen3.6-35B-A3B**, benchmarked on HuggingFace eager inference and SGLang serving.

**Target hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB) · PyTorch 2.11+ cu130 · SGLang 0.5.13 · `sglang-kernel` 0.4.3+cu130

**Checkpoints (GPU host):**

| Path | Description |
|------|-------------|
| `/data/Qwen3.6-35B-A3B-bf16` | Vanilla BF16 from HuggingFace |
| `/data/Qwen3.6-35B-A3B-bf16-fused` | Weight-fused export (γ absorbed into downstream linears) |

---

## What we fuse

Each decoder layer has two **sites** where RMSNorm feeds a linear:

| Site | Location | Norm | Downstream linears |
|------|----------|------|-------------------|
| **1 — attn** | Before attention | `input_layernorm` | Linear-attn: `in_proj_{qkv,z,b,a}` · Full-attn: `q/k/v_proj` |
| **2 — moe** | Before MoE FFN | `post_attention_layernorm` | Router, expert `gate_up_proj`, shared expert |

**Math (γ absorbed offline into weights):**

```
unfused:  out = W @ ((x / rms(x)) * γ)
fused:    out = W_fused @ (x / rms(x))     where W_fused = W * γ
```

**Two layers of optimization:**

1. **Weight fusion** (`fused-checkpoint/`) — absorb γ at export; norm weights → ~0.
2. **Kernel fusion** (`phase2/`+) — at runtime, one `x/rms(x)` per layer (Site-1) instead of separate norm + linear.

Site-2 **runtime** patch is experimental; the fused checkpoint already bakes γ into MoE weights, so E2E defaults to stock MoE after Site-1.

---

## Repository layout

```
phase1/              Vanilla HF baseline + correctness oracle
fused-checkpoint/    Export weight-fused checkpoint
phase2/              HF microbench + E2E prefill (fused vs unfused)
phase3/              SGLang vanilla prefill baseline
phase4/              SGLang + fused ckpt + HF plugin (qwen_fusion)
phase5/              SGLang + fused ckpt + native layer plugin (qwen_fusion_native)
benchmark_reference.py   Shared metrics, CSV schema, shape sweep
qwen3_moe_layers.py      Layer-type helpers
```

Activate the venv once:

```bash
source phase1/.venv/bin/activate
```

Per-phase setup and commands: see each `phase*/README.md`.

---

## Phase overview

| Phase | Backend | Checkpoint | Runtime kernel | CSV `benchmark_mode` | Role |
|-------|---------|------------|----------------|----------------------|------|
| **1** | HF | vanilla | none | — | Load model, correctness oracle (`reference_logits.pt`) |
| **export** | HF | vanilla → fused | none | — | Offline γ absorption |
| **2** | HF | both | Site-1 V2 (Site-2 opt-in) | `checkpoints+kernel`, `hf-e2e` | **Where fusion helps** |
| **3** | SGLang | vanilla | none | `sglang-vanilla` | Serving baseline |
| **4** | SGLang | fused | HF `patch_hf_kernel_fusion` plugin | `sglang-fused` | Plugin on HF decoder forward |
| **5** | SGLang | fused | Native `Qwen3_5*DecoderLayer` hooks | `sglang-fused-native` | Plugin on SGLang layer graph |

Benchmark shapes (all full-model prefill phases): `(batch, seq_len)` ∈ {1,8,32} × {128,512,2048}, warmup 50 / measure 200.

**Speedup** = `nonfused_median_ms / fused_median_ms` (>1.0 = fused is faster).

---

## Results summary (Blackwell, 2026-06)

### Phase 2 — HuggingFace (the meaningful win)

| Benchmark | Site-1 speedup (typical) | Notes |
|-----------|--------------------------|-------|
| Layer microbench (attn) | **1.05–1.58×** | Best at small seq (e.g. 1×512 → 1.58×) |
| Layer microbench (moe) | **1.0–1.9×** | Large at high seq in isolation only |
| **E2E prefill** | **~1.0–1.06×** | Site-1 fast-path; logits match oracle (cos ~0.997–0.999) |

HF eager path has separate Python ops and extra memory traffic — fusion removes real overhead, but E2E gain is modest because attention + MoE dominate.

### Phase 3 — SGLang vanilla

~**3–7× faster** than HF E2E at the same shapes (e.g. ~65 ms vs ~220 ms median at batch=1, seq=128). SGLang already uses optimized CUDA norms, GEMMs, FusedMoE, and flash attention.

### Phase 4 — SGLang + fused (HF plugin)

| Mode | Small batch | Large shapes |
|------|-------------|--------------|
| Site-1 kernel (`SGLANG_FUSION=1`) | **~0.82–0.85×** (slower) | ~1.00× |
| Weights only (`--no-kernel`) | **~0.98×** | ~1.00× |

Replacing SGLang forwards with HF `patch_decoder_layer` adds Python overhead and bypasses native kernels — **do not use for inference**.

### Phase 5 — SGLang + fused (native layer plugin)

| Regime | Speedup vs Phase 3 |
|--------|-------------------|
| Small batch | **~0.97×** |
| Mid / large | **~1.00×** |

Native hooks (skip `input_layernorm`, `linear(x)/rms(x)` on projections) avoid the Phase 4 regression but **do not beat** vanilla SGLang. Python-level fusion is slightly worse than stock norm+GEMM at small batch; parity at large shapes.

**Site-2 on Phase 5:** not implemented for native layers; `--site2` only affects HF fallback. Even if wired, Phase 2 E2E showed Site-2 runtime patch is usually slower than stock MoE with fused weights.

---

## Key learnings

### 1. Weight fusion is correct; kernel fusion is stack-dependent

- Fused checkpoint matches Phase 1 oracle (top-1 token, tight logit diffs).
- On **HF eager**, Site-1 kernel fusion gives a small but real E2E gain (~1–6%).
- On **SGLang**, fused weights alone are **parity** (~1.0×); runtime kernel plugins do not help.

### 2. SGLang does not do our fusion — and does not need to for speed

SGLang runs `input_layernorm` → projection as **separate** optimized kernels (`GemmaRMSNorm`, fast GEMM). It fuses other things (residual+norm, QK norm+RoPE, FusedMoE Triton) — not our specific `W_fused @ (x/rms(x))` with γ absorbed offline.

With fused weights, stock SGLang path is mathematically equivalent (`norm.weight ≈ 0` → divide-by-rms only) and already fast enough that skipping the norm kernel does not show up in benchmarks.

### 3. Plugin hooks hit a ceiling

| Approach | Outcome |
|----------|---------|
| Phase 4 HF forward replacement | Clear loss (~15–20% small batch) |
| Phase 4 weights only | Parity |
| Phase 5 native Python Site-1 | Parity / tiny loss |

Plugins cannot match a single fused CUDA norm+GEMM inside SGLang’s layer code.

### 4. Fused checkpoint + SGLang needs metadata sync

`save_pretrained` writes text-only config; SGLang expects full multimodal metadata. Phase 4/5 copy from vanilla (not weight shards):

`config.json`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer_config.json`, etc.

### 5. SGLang on Blackwell — env pitfalls

- Use **`sglang-kernel` 0.4.x** (cu130), not legacy `sgl-kernel` 0.3.x.
- Set `LD_LIBRARY_PATH` to pip `nvidia/cuda_nvrtc`, `cuda_runtime`, `cu13` libs.
- `context_length=65536` for benchmarks (full 262k OOMs on 96 GB).
- Install: `bash phase3/setup_sglang.sh`

### 6. What would be needed for SGLang speedup

A **custom fused CUDA/Triton op** (`(x @ W) / rms(x)`) wired into SGLang’s `qwen3_5.py` call sites — fork or upstream PR, not an external plugin. Realistic E2E gain even then: **~1–5%** (norm is a thin slice vs attention + MoE).

---

## Recommendations

| Use case | Recommendation |
|----------|----------------|
| **HF / eager inference** | Fused ckpt + Phase 2 Site-1 V2 kernel |
| **SGLang serving** | Vanilla ckpt, or fused ckpt **without** kernel plugin (`--no-kernel` / `SGLANG_FUSION=0`) |
| **Site-2 runtime** | Skip for E2E; microbench only |
| **Phase 4 HF plugin** | Avoid for production |
| **Phase 5 native plugin** | Research / parity check only |

---

## Quick start (GPU host)

```bash
# 1. Env + model
bash phase1/setup_env.sh
source phase1/.venv/bin/activate
bash phase1/download_model.sh

# 2. Oracle + fused export
python phase1/run_phase1.py --reference
cd fused-checkpoint && python export_fused_weights.py --check

# 3. HF fusion benchmark
python phase2/benchmark_e2e_prefill.py --check-logits

# 4. SGLang baseline + fused experiments
bash phase3/setup_sglang.sh
python phase3/benchmark_sglang.py

bash phase4/setup_fusion_plugin.sh   # HF plugin
bash phase5/setup_fusion_plugin.sh   # native plugin (optional)
```

---

## Further reading

- [phase1/README.md](phase1/README.md) — baseline & oracle
- [fused-checkpoint/README.md](fused-checkpoint/README.md) — weight export
- [phase2/README.md](phase2/README.md) — fusion sites, microbench tables
- [phase3/README.md](phase3/README.md) — SGLang setup & vanilla benchmark
- [phase4/README.md](phase4/README.md) — HF plugin on SGLang
- [phase5/README.md](phase5/README.md) — native SGLang layer hooks
