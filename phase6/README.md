# Phase 6 — Fused Norm+GEMM Inside SGLang

**Goal:** Implement Site-1 fusion as a **real kernel** in SGLang’s Qwen3.5 forward path: `(x @ W_fused) / rms(x)` instead of separate `GemmaRMSNorm` + GEMM.

**Status (2026-06):** M0–M4 **complete**. Correctness verified; E2E prefill ~**1.00×** vs Phase 3 (3% target not met). Implementation details → **[SGLang_changes.md](./SGLang_changes.md)**.

**Why Phase 6:** Phases 4–5 proved plugins cannot beat stock SGLang (~1.0×). Phase 2 layer-0 microbench shows **~1.58× attn / ~1.9× MoE** when fusion hits the hot path on HF — Phase 6 put that inside the engine.

**Non-goals:** Site-2 MoE kernel fusion, TP>1, FP8/quant paths, upstream SGLang PR.

---

## Success criteria vs outcomes

| Gate | Target | Outcome |
|------|--------|---------|
| Correctness | Top-1 vs Phase 1 oracle | **PASS** (`--check-logits`) |
| Op math | vs Phase 2 `Site1RmsState` | **PASS** (`test_fused_rmsnorm_gemm.py`) |
| CUDA graphs | Graph replay safe | **PASS** (`test_cuda_graph_fusion.py`) |
| E2E prefill @ 1×512 | **≥ 1.03×** | **~1.01×** (not met) |
| Large shapes | **≥ 1.00×** | **~1.00×** (met) |
| Layer-0 microbench in SGLang | ~1.58× attn | **Not run** (optional M5) |

E2E ~1.00× after a working kernel means the bottleneck is attention/MoE, not norm+linear — documented and expected.

---

## Architecture

### Vanilla SGLang path

```text
LayerCommunicator.prepare_attn
  └─ gemma_fused_add_rmsnorm(x, residual)  →  h = (x/rms) * (1+γ)
       └─ in_proj_qkvz(h) / qkv_proj(h)     →  attention ...
```

### Phase 6 fused path (active when `use_qwen_fusion()`)

```text
LayerCommunicator.prepare_attn
  └─ residual add only (no norm write)
       └─ fused_rmsnorm_gemm(raw, W_fused)   →  W @ x / rms(x)
            └─ attention + MoE               →  stock SGLang
```

**γ is in `W_fused`** from `fused-checkpoint/export_fused_weights.py`. SGLang **infers** fused ckpt (norm weights ≈ 0); no special config field. Details in [SGLang_changes.md § Fused checkpoint](./SGLang_changes.md#fused-checkpoint--γ-how-sglang-knows).

### Call sites

| Layer type | File | Projections |
|------------|------|-------------|
| Linear-attn | `sglang/.../qwen3_5.py` → `_forward_input_proj` | `in_proj_qkvz`, `in_proj_ba` |
| Full-attn | same → `forward_prepare_native` / `self_attention` | `qkv_proj` |
| Norm boundary | `sglang/.../communicator.py` → `prepare_attn` | skip `input_layernorm` |

MoE stays **stock** (`FusedMoE` + `post_attention_layernorm` with γ already in expert weights from export).

---

## Setup — local SGLang clone

Clone upstream once; override sources via `PYTHONPATH` (no `pip install -e` required — avoids Rust/gRPC rebuild issues):

```bash
# Sibling clone @ v0.5.13
git clone https://github.com/sgl-project/sglang.git /sglang
cd /sglang && git checkout v0.5.13

# Phase 3 venv already has pip sglang + sglang-kernel
source phase6/env.sh   # sets PYTHONPATH=/sglang/python
```

```text
/sglang/                          # private edits (see SGLang_changes.md)
/Qwen3.6-35B-fusion/phase6/
  SGLang_changes.md               # file-by-file change log
  env.sh                          # dev environment
  test_fused_rmsnorm_gemm.py
  test_cuda_graph_fusion.py
  benchmark_sglang_fused_kernel.py
  launch_server.sh
  sync_fused_ckpt_metadata.sh
  patches/                        # optional: git diff snapshot
```

**Once:** `bash phase6/sync_fused_ckpt_metadata.sh` (VLM processor metadata for fused ckpt).

---

## Milestones

### M0 — Flag & fused-ckpt detection — done

- [x] Local clone @ v0.5.13 + `phase6/env.sh`
- [x] `SGLANG_QWEN_FUSION` / `--enable-qwen-fusion`
- [x] Fused ckpt heuristic: layer-0 `input_layernorm.weight.max() < 0.15`
- [x] Post-load log: `fused_ckpt=True active=True`

### M1 — Reference kernel — done

- [x] `fused_rmsnorm_gemm` + Triton RMS/divide
- [x] `phase6/test_fused_rmsnorm_gemm.py` vs Phase 2

### M2 — Forward wiring — done

- [x] `prepare_attn` fusion branch
- [x] GDN + full-attn projections wired
- [x] Shared RMS + alt-stream overlap for paired projections
- [x] `--check-logits` PASS

### M3 — CUDA graphs — done

- [x] Graph-capture safe (torch fallback + `raw_out`)
- [x] `phase6/test_cuda_graph_fusion.py`
- [ ] `nsys` kernel-count profile (optional)

### M4 — Benchmark — done

- [x] `phase6/benchmark_sglang_fused_kernel.py` (`benchmark_mode=sglang-fused-kernel`)
- [x] 9-shape prefill sweep vs Phase 3 CSV
- [x] Results: ~1.00× E2E

### M5 — Hardening (optional)

- [x] TP>1 / quant → stock fallback (implemented in `use_qwen_fusion()`)
- [ ] Export `phase6/patches/sglang-phase6.patch`
- [ ] Same-session A/B benchmark
- [ ] Layer-0 microbench inside SGLang

---

## Testing checklist

### Correctness

- [x] Phase 1 oracle top-1 (`benchmark_sglang_fused_kernel.py --check-logits`)
- [x] Per-op: fused vs `Site1RmsState` (`test_fused_rmsnorm_gemm.py`)
- [x] CUDA graph replay (`test_cuda_graph_fusion.py`)
- [ ] Full logit diff vs vanilla (optional; top-1 sufficient for gate)

### Performance

- [x] 9-shape prefill sweep (M4)
- [ ] Layer-0 isolated projection in SGLang (optional)
- [ ] Same-session vanilla comparison (baseline CSV is from a prior Phase 3 run)

---

## Kernel design notes

### Math

```text
rms(x) = sqrt(mean(x²) + eps)
y      = (x @ W_fused) / rms(x)        γ absorbed offline; no (1+weight) at runtime
```

### Shared RMS (linear-attn layers)

`in_proj_qkvz` and `in_proj_ba` share one `rms(x)` per layer via `RmsNormGemmState` / `fused_rmsnorm_gemm_pair`.

### Fallback matrix

| Condition | Behavior |
|-----------|----------|
| `SGLANG_QWEN_FUSION=0` / no `--enable-qwen-fusion` | Stock SGLang |
| Vanilla ckpt (norm weight not ~0) | Stock |
| `tp_size > 1` | Stock |
| Quantized model | Stock |

---

## Quick reference

```bash
source phase6/env.sh

# Benchmark (engine backend — stop any server on :30000 first)
python phase6/benchmark_sglang_fused_kernel.py --check-logits

# Or serve + HTTP benchmark
bash phase6/launch_server.sh
python phase6/benchmark_sglang_fused_kernel.py --backend http --check-logits

# Unit tests
python phase6/test_fused_rmsnorm_gemm.py
python phase6/test_cuda_graph_fusion.py
```

---

## Relationship to earlier phases

| Phase | Role |
|-------|------|
| **export** | Fused checkpoint (`W_fused`, norm ≈ 0) |
| **1** | Correctness oracle |
| **2** | Math reference + layer-0 speedup targets |
| **3** | E2E baseline CSV |
| **4–5** | Anti-patterns (plugins/hooks ~1.0× E2E) |
| **6** | In-engine kernel (this work) |

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06 | Site-1 only in SGLang | MoE already `FusedMoE`; Site-2 hurt/neutral on HF E2E |
| 2026-06 | Local clone via `PYTHONPATH` | No upstream PR; avoid editable-install Rust build |
| 2026-06 | Triton + cuBLAS first | Match SGLang patterns; fast iteration |
| 2026-06 | E2E ~1.00× accepted | Kernel correct; bottleneck elsewhere |
