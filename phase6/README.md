# Phase 6 — Fused Norm+GEMM Inside SGLang

**Goal:** Implement Site-1 fusion as a **real kernel** in SGLang’s Qwen3.5 forward path, so deployment uses `(x @ W_fused) / rms(x)` in one fused op instead of separate `GemmaRMSNorm` + GEMM.

**Why Phase 6:** Phases 4–5 proved plugins cannot beat stock SGLang (~1.0×). Phase 2 layer-0 microbench shows **~1.58× attn / ~1.9× MoE** when fusion is actually in the hot path — but only on HF. Phase 6 targets that gap inside the engine.

**Non-goals (initial scope):** Site-2 MoE kernel fusion, TP>1, FP8/quant paths, CUDA-graph-unfriendly Python hooks, Pie/inferlets.

---

## Success criteria

| Gate | Target |
|------|--------|
| Correctness | Top-1 token + logits vs Phase 1 oracle on fused ckpt (`atol ≤ 0.01`) |
| E2E prefill median | **≥ 1.03×** vs Phase 3 vanilla at batch=1, seq=512 (stretch: 1.05×) |
| Large shapes | **≥ 1.00×** (no regression at 32×2048) |
| Layer-0 microbench | Match or beat Phase 2 attn Site-1 (~1.58× at 1×512) inside SGLang-loaded layer |

If E2E stays at ~1.00× after a fused kernel lands, stop — the bottleneck is elsewhere (attention/MoE), not norm+linear.

---

## Architecture (what changes)

### Current SGLang path (vanilla ckpt)

```text
LayerCommunicator.prepare_attn
  └─ gemma_fused_add_rmsnorm(x, residual)  →  h = (x/rms) * (1+γ)
       └─ in_proj_qkvz(h) / qkv_proj(h)     →  attention ...
```

### Target path (fused ckpt, Phase 6)

```text
LayerCommunicator.prepare_attn
  └─ fused_add_residual_only(x, residual)   →  raw x (no norm write)
       └─ fused_rmsnorm_gemm(raw, W_fused)   →  one kernel: W @ x / rms(x)
            └─ attention ...                  →  stock SGLang (unchanged)
```

**γ is already in `W_fused`** from `fused-checkpoint/export_fused_weights.py`. The kernel must **not** apply `(1+weight)` from `input_layernorm` when `weight ≈ 0`.

### Call sites (Qwen3.6 / SGLang 0.5.x)

| Layer type | File (upstream) | Replace |
|------------|-----------------|---------|
| Linear-attn | `python/sglang/srt/models/qwen3_5.py` → `Qwen3_5GatedDeltaNet._forward_input_proj` | `in_proj_qkvz`, `in_proj_ba` |
| Full-attn | same file → `Qwen3_5AttentionDecoderLayer.self_attention` | `qkv_proj` |
| Norm boundary | `python/sglang/srt/layers/communicator.py` → `prepare_attn` | Skip `input_layernorm` when fusion enabled |

MoE stays **stock**: `post_attention_layernorm` + `FusedMoE` (Site-2 not worth it per Phase 2 E2E).

---

## Repo strategy — local clone (no upstream PR)

You do **not** need a GitHub fork or PR to `sgl-project/sglang`. Clone upstream once, edit locally, install editable — keep it private to this project.

```bash
# Sibling to this repo (example layout)
cd /path/to/parent
git clone https://github.com/sgl-project/sglang.git
cd sglang && git checkout v0.5.13

# Install over pip SGLang (in phase1 venv)
pip install -e /path/to/sglang/python
pip install sglang-kernel==0.4.3+cu130
```

```text
/path/to/
  sglang/                         # local clone @ v0.5.13 — your private edits
    python/sglang/srt/layers/fused_rmsnorm_gemm.py   # NEW
    python/sglang/srt/models/qwen3_5.py              # wire fused path
    python/sglang/srt/layers/communicator.py         # prepare_attn branch

  qwen3.6-35B-fusion/
    phase6/
      README.md                   # this plan
      patches/                    # optional: export your sglang diff for reproducibility
      benchmark_sglang_fused_kernel.py   # (later)
```

**Optional:** store `phase6/patches/*.patch` (`git diff` from your clone) so you can re-apply after re-cloning — no need to submodule or open-source the SGLang changes.

**Do not** `pip install sglang` from PyPI after `pip install -e` — the editable clone replaces it.

---

## Implementation plan (milestones)

### M0 — Spike & environment (1–2 days)

- [ ] Clone `sgl-project/sglang` at **v0.5.13** (match `phase3/setup_sglang.sh`); work on a local branch.
- [ ] Confirm layer class names on GPU: `Qwen3_5LinearDecoderLayer`, `Qwen3_5AttentionDecoderLayer` (64 layers).
- [ ] Add env flag: `SGLANG_QWEN_FUSION=1` (server arg + `ServerArgs`).
- [ ] Detect fused ckpt: `input_layernorm.weight.abs().max() < 0.15` on layer 0 (reuse Phase 5 check).

**Deliverable:** Local clone builds; vanilla Qwen3.6 loads; flag is a no-op.

---

### M1 — Reference kernel (Python/Triton prototype) (3–5 days)

Implement **correctness-first** op outside the hot path:

```python
# fused_rmsnorm_gemm(x, weight, bias=None, eps=1e-6) -> y
# y = F.linear(x, weight) / rms(x)   # mathematically matches Phase 2 Site1RmsState
```

**Steps:**

1. Triton kernel in `sglang/srt/layers/triton/` (follow `fused_moe_triton` patterns) **or** epilogue on existing GEMM.
2. Unit test: random `x`, `W`, compare to `phase2/fusion_bf16.Site1RmsState.project`.
3. Unit test: load one `in_proj_qkvz` weight from fused ckpt; max diff vs HF reference < 1e-2 BF16.

**Start shapes:** `[T, 4096] @ [4096, K]` — token-major 2D (SGLang linear-attn layout).

**Deliverable:** `tests/test_fused_rmsnorm_gemm.py` green on Blackwell.

---

### M2 — Wire into Qwen3.5 projections (3–5 days)

**`Qwen3_5GatedDeltaNet._forward_input_proj`:**

```python
if use_qwen_fusion:
    qkvz = fused_rmsnorm_gemm(hidden_states, self.in_proj_qkvz.weight, eps=...)
    ba   = fused_rmsnorm_gemm(hidden_states, self.in_proj_ba.weight, eps=...)
    # preserve alt_stream overlap: rms(x) shared across both (like Site1RmsState V2)
else:
    ... stock ...
```

**`Qwen3_5AttentionDecoderLayer.self_attention`:**

```python
if use_qwen_fusion:
    qkv = fused_rmsnorm_gemm(hidden_states, self.qkv_proj.weight, eps=...)
else:
    qkv, _ = self.qkv_proj(hidden_states)
```

**`prepare_attn` (when fusion on):**

- Keep residual add path (`gemma_fused_add_rmsnorm` **without** norm multiply, or add-only + pass raw `hidden_states`).
- Do **not** materialize normed tensor to global memory.

**Deliverable:** `python -m sglang.launch_server --model-path ...-fused` + greedy decode matches oracle.

---

### M3 — CUDA graphs & perf (3–5 days)

SGLang captures piecewise CUDA graphs. Fused op must be:

- [ ] Graph-capture safe (fixed addresses, no CPU sync in forward).
- [ ] Compatible with `disable_piecewise_cuda_graph` fallback.
- [ ] Registered in `MultiPlatformOp` pattern if needed (see `GemmaRMSNorm`).

**Profile:** `nsys` one layer forward — confirm one fewer kernel launch vs stock (norm + GEMM → fused).

**Deliverable:** Phase 3-equivalent benchmark script; median latency table.

---

### M4 — Benchmark & compare (1–2 days)

Add `phase6/benchmark_sglang_fused_kernel.py` (copy phase5 harness):

| Compare against | CSV / baseline |
|-----------------|----------------|
| Phase 3 vanilla | `phase3/results/benchmark_*_sglang-vanilla_*.csv` |
| Phase 5 native plugin | `phase5/results/benchmark_*_sglang-fused-native_*.csv` |
| Phase 2 layer-0 (optional) | microbench single projection inside SGLang |

Record `benchmark_mode=sglang-fused-kernel`.

---

### M5 — Hardening (ongoing)

- [ ] TP=1 only documented; TP>1 → fall back to stock path.
- [ ] Quant / FP8 paths → fall back (no fused kernel).
- [ ] `language_model_only`, multimodal load unchanged.
- [ ] Document clone + `pip install -e` in `phase6/SETUP.md`.
- [ ] Optional: save `phase6/patches/` diff for reproducibility.

---

## Kernel design notes

### Math (must match Phase 2)

```text
rms(x) = sqrt(mean(x²) + eps)          per token row
y      = (x @ W_fused) / rms(x)        no γ — absorbed offline
```

Gemma norm `(1+weight)` must be **skipped** when fusion is active.

### Shared RMS across paired projections (V2)

Linear-attn runs `in_proj_qkvz` and `in_proj_ba` on the **same** `hidden_states`. Compute `rms(x)` **once** per layer forward (Phase 2 `Site1RmsState` V2 stream overlap). Kernel API:

```python
fused_rmsnorm_gemm_pair(x, W_a, W_b, eps, rms_state=None)
```

### Why Triton first

- SGLang already ships Triton MoE kernels — matches project conventions.
- Faster iteration than `sglang-kernel` C++/CUDA rebuild cycle.
- Promote to `sglang-kernel` CUDA only if Triton leaves performance on the table.

### Fallback matrix

| Condition | Behavior |
|-----------|----------|
| `SGLANG_QWEN_FUSION=0` | Stock SGLang |
| Vanilla ckpt (norm weight not ~0) | Stock SGLang |
| `tp_size > 1` | Stock (v1) |
| Quantized model | Stock |
| Triton kernel unsupported dtype | Stock |

---

## Testing checklist

### Correctness

- [ ] Phase 1 oracle top-1 on reference prompt (engine backend).
- [ ] `max |logit diff| < 0.01` vs Phase 3 vanilla on same prompt (may differ slightly in last bits; top-1 must match).
- [ ] Per-layer: fused vs `Site1RmsState` on random tensors, all layer types (linear + full).

### Performance

- [ ] 9-shape prefill sweep (same as phase3).
- [ ] Layer-0 isolated projection timing (optional microbench hook).
- [ ] Memory: no extra large activation for normed `h` (bandwidth win is the hypothesis).

---

## Risk register

| Risk | Mitigation |
|------|------------|
| E2E gain < 3% even with fused kernel | Expected from Phase 3/5; document result; no need to contribute upstream |
| CUDA graph breaks | `disable_piecewise_cuda_graph` fallback; test capture mode early |
| SGLang version drift | Pin v0.5.13; document diff for 0.5.14+ |
| `MergedColumnParallelLinear` weight layout | Unit test weight shapes against live module |
| Multimodal load / config | Reuse phase4/5 metadata sync; no change to vision path |

---

## Relationship to earlier phases

| Phase | Role in Phase 6 |
|-------|-----------------|
| **1** | Correctness oracle |
| **export** | Fused checkpoint (`W_fused`, norm ≈ 0) |
| **2** | Math reference (`fusion_bf16`, layer-0 speedup target) |
| **3** | E2E baseline to beat |
| **4** | Anti-pattern (HF forward replacement) |
| **5** | Anti-pattern (Python native hooks); reuse weight check + benchmark harness |

---

## Quick reference commands (once implemented)

```bash
# Editable install from local clone
pip install -e /path/to/sglang/python

# Serve fused model with kernel fusion
export SGLANG_QWEN_FUSION=1
python -m sglang.launch_server \
    --model-path /data/Qwen3.6-35B-A3B-bf16-fused \
    --dtype bfloat16 --tp-size 1 --context-length 65536 \
    --trust-remote-code

# Benchmark
python phase6/benchmark_sglang_fused_kernel.py \
    --baseline-csv phase3/results/benchmark_*_sglang-vanilla_*.csv \
    --check-logits
```

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06 | Site-1 only in SGLang kernel | MoE already FusedMoE; Site-2 hurts E2E on HF |
| 2026-06 | Local SGLang clone, not Pie/plugins | Fusion must live in model forward / CUDA; no upstream PR |
| 2026-06 | Triton before custom CUDA | Match SGLang MoE patterns; faster iteration |
| 2026-06 | Success bar ≥1.03× E2E | Phase 5 parity proves <3% is noise without real kernel |

---

## Next action

Start **M0**: clone SGLang @ v0.5.13, add `SGLANG_QWEN_FUSION` flag (no-op), verify load on `/data/Qwen3.6-35B-A3B-bf16-fused` with Phase 3 env.
