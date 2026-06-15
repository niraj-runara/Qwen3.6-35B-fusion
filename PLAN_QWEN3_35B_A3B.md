# Plan: Weight + Kernel Fusion for Qwen3.6-35B-A3B on SGLang

**Target model:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)  
**Reference work:** Kimi-K2.6 weight + kernel fusion (this repo)  
**Inference engine:** SGLang (vs vLLM used for Kimi)

---

## Architecture Overview (Qwen3.6-35B-A3B)

Before planning work, understand what we are operating on.

| Property | Value |
|---|---|
| Total parameters | ~35B |
| Active parameters per token | ~3B |
| Layers | 64 |
| Hidden size (h) | 4096 |
| Attention | GQA — 32 Q heads, 8 KV heads, head_dim = 128 |
| MoE | 128 total experts, 8 active per token |
| Expert FFN size | 1536 intermediate per expert |
| Shared experts | None |
| Norm type | RMSNorm (same as Kimi) |
| Activation | SiLU + gating (SwiGLU-style) |
| Dtype (HF release) | BF16 |

**Key difference from Kimi-K2.6:**  
Kimi uses MLA (Multi-head Latent Attention) with a latent KV bottleneck — this drove the 3 specific fusion sites in this repo. Qwen3.6-35B-A3B uses **standard GQA**, so the attention fusion is simpler (no q_a/kv_a latent projections). The dominant fusion opportunity shifts to the **MoE FFN** instead.

**Fusion sites for Qwen3.6-35B-A3B:**

| Site | Modules | Fused op |
|---|---|---|
| 1 | `input_layernorm` + `q_proj`, `k_proj`, `v_proj` | `fused_qkv(x)` = linear(x) / rms(x) |
| 2 | `post_attention_layernorm` + expert `gate_proj` + `up_proj` (×128) | `fused_moe_gate_up(x)` = linear(x) / rms(x) per dispatch |

Site 1 is a direct port of this repo's existing `FusedRMSNormNVFP4Linear`.  
Site 2 is the harder problem — MoE dispatch means the fused op must run after expert routing, inside the expert forward.

---

## Phase 1 — Vanilla Model from HF (No Changes)

**Goal:** Clean working baseline. Know exactly what the model does, what its output looks like, and its performance before we touch anything.

### 1.1 Environment Setup

- Python 3.11+, CUDA 12.x
- `pip install transformers accelerate safetensors torch`
- Confirm GPU memory: the model is ~70 GB in BF16; fits on a single RTX Pro 6000 (96 GB GDDR7) with ~18–22 GB headroom for KV cache and activations. Would NOT fit on a single A100-80G (only 80 GB — not enough headroom). No TP needed for this GPU.
- **Note on bandwidth:** RTX Pro 6000 uses GDDR7 (~1.8 TB/s) vs A100 HBM2e (~2 TB/s). Decode is memory-bandwidth-bound, so benchmark numbers here will not directly compare to A100 runs.
- Note: this phase uses HF `transformers` directly, no vLLM/SGLang yet

### 1.2 Download the Checkpoint

```bash
huggingface-cli download Qwen/Qwen3.6-35B-A3B \
    --local-dir /data/Qwen3.6-35B-A3B-bf16
```

Verify the download:
- Check `model.safetensors.index.json` is present
- Count shards and confirm total size
- Spot-check a few tensor names to confirm it matches the expected architecture

### 1.3 Load and Inspect

- Load with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto")`
- Print the full model tree: confirm layer names follow `model.layers.{i}.self_attn.q_proj`, `model.layers.{i}.mlp.experts.{j}.gate_proj`, etc.
- Check norm weights: `model.layers.0.input_layernorm.weight` — these should be non-trivial (not all 1.0) confirming weight fusion has not been applied yet
- Note exact module class names for later patching (e.g. `Qwen3MoeSparseMoeBlock`, `Qwen3MoeAttention`)

### 1.4 Correctness Reference

Run a short forward pass on a fixed prompt. Record:
- Logits for the first token (will be used as the correctness oracle in Phase 2)
- Top-5 token predictions

Save these to a file. Every subsequent phase must reproduce them within `atol=1e-2`.

### 1.5 Baseline Performance Benchmark

Use `transformers` with `torch.compile` disabled (clean baseline):
- Measure prefill latency at batch=1, seq_len=[512, 1024, 2048]
- Measure throughput (tokens/sec) at batch=8
- Profile with `torch.profiler` to identify the top 3 time-consuming ops

**Deliverables:** a clean BF16 checkpoint at `/data/Qwen3.6-35B-A3B-bf16`, logit reference output, baseline latency numbers.

---

## Phase 2 — Add Algo (Weight Fusion + Kernel Fusion) to Vanilla

**Goal:** Produce a weight-fused checkpoint and a runtime-patched forward that is provably faster and numerically equivalent to Phase 1.

This phase uses HF `transformers` as the backend — no SGLang yet. This keeps the delta small: only the fusion logic is new.

### 2.1 Weight Fusion (Offline, on BF16 Checkpoint)

**What it does:**  
Absorb `layernorm.weight` (gamma) into the downstream linear weight matrix.  
`W_new = W * gamma`  
Set `layernorm.weight = ones` (gamma ≈ 1) so the RMSNorm becomes a pure divide-by-rms operation.

**Math:**  
`layernorm(x) = (x / rms(x)) * gamma`  
`linear(layernorm(x)) = (W * gamma) @ (x / rms(x)) = W_new @ (x / rms(x))`  
So `W_new @ x / rms(x)` is numerically identical.

**Script to write: `export_fused_weights_qwen3.py`**

Site 1 — absorb `input_layernorm` into `q_proj`, `k_proj`, `v_proj`:
```
for each layer i:
    gamma = model.layers[i].input_layernorm.weight   # shape [h]
    model.layers[i].self_attn.q_proj.weight *= gamma
    model.layers[i].self_attn.k_proj.weight *= gamma
    model.layers[i].self_attn.v_proj.weight *= gamma
    model.layers[i].input_layernorm.weight = ones
```

Site 2 — absorb `post_attention_layernorm` into each expert's `gate_proj` and `up_proj`:
```
for each layer i:
    gamma = model.layers[i].post_attention_layernorm.weight   # shape [h]
    for each expert j in model.layers[i].mlp.experts:
        expert.gate_proj.weight *= gamma
        expert.up_proj.weight   *= gamma
    model.layers[i].post_attention_layernorm.weight = ones
```

Save the modified model as a new checkpoint:
`/data/Qwen3.6-35B-A3B-bf16-fused`

**Correctness gate:** Load fused checkpoint, run same prompt from Phase 1, diff logits. Must pass `atol=1e-2`. If it fails, the weight absorption script has a bug.

### 2.2 Kernel Fusion (Runtime Patch on Fused Checkpoint)

**What it does:**  
Replace `layernorm(x) → linear(x)` with `linear(x) / rms(x)` (since gamma=1 after weight fusion).  
The existing `FusedRMSNormNVFP4Linear` class in `fusion/nvfp4_fused_rmsnorm_linear.py` implements exactly this — it can be reused as-is for BF16 (the quantized NVFP4 path is a detail of the linear module passed in, not the wrapper itself).

**Script to write: `patch_qwen3_layers.py`**

Site 1 — patch attention input norm:
- Zero out `input_layernorm` in the decoder forward (set weight to ones, it's already fused)
- Replace the decoder's forward to pass raw `x` directly into `q_proj`, `k_proj`, `v_proj` through a `FusedRMSNormNVFP4Linear` wrapper

Site 2 — patch MoE FFN:
- More complex: the fused op must run per-expert after routing, not before
- Two sub-options:
  - **2.2a (simpler):** Patch the expert class's forward: replace `gate_proj(x)` with `gate_proj(x) / rms(x)` where rms is computed once and shared for `up_proj` too
  - **2.2b (optimal):** Fuse the rms computation with both `gate_proj` and `up_proj` in a single kernel pass (compute rms once, apply to both outputs) — this is a new CUDA kernel

For the initial version, implement 2.2a first (Python-level fusion, no new kernel). Profile to see if 2.2b is worth the CUDA kernel work.

**Correctness gate:** Same logit diff test as 2.1. Both weight-only and weight+kernel paths must pass.

**Performance benchmark:**  
Compare prefill and decode latency:
1. Vanilla BF16 (Phase 1)
2. Weight-fused BF16 (Phase 2.1 only)
3. Weight-fused + kernel-fused BF16 (Phase 2.2)

Metric: tokens/sec, GPU utilization, memory bandwidth. Profile with `torch.profiler` to quantify the norm overhead eliminated.

**Deliverables:** `export_fused_weights_qwen3.py`, `patch_qwen3_layers.py`, fused checkpoint at `/data/Qwen3.6-35B-A3B-bf16-fused`, correctness + speedup numbers.

---

## Phase 3 — Run Vanilla with SGLang

**Goal:** Get the unmodified Qwen3.6-35B-A3B running correctly and fast under SGLang before touching SGLang internals. Know where SGLang's model implementation lives.

### 3.1 SGLang Environment

```bash
pip install "sglang[all]"
# or from source for the right version:
git clone https://github.com/sgl-project/sglang
pip install -e "sglang[all]"
```

Check SGLang's Qwen3-MoE support:
- Look for `sglang/srt/models/qwen3_moe.py` or similar
- Confirm it maps to the same `Qwen3MoeForCausalLM` model class
- Note the module hierarchy — it will differ from HF `transformers` (SGLang has its own linear/attention wrappers)

### 3.2 Launch Vanilla Inference

```bash
python -m sglang.launch_server \
    --model-path /data/Qwen3.6-35B-A3B-bf16 \
    --tp 1 \
    --dtype bfloat16 \
    --port 30000
```

Confirm:
- Server starts without errors
- Tokenizer and model load correctly
- Run the same reference prompt from Phase 1 through the SGLang HTTP API
- Diff logits against Phase 1 baseline (some numerical diff expected due to different kernel implementations; aim for top-1 token match)

### 3.3 Understand SGLang's Model Loading

This is research work — read the code, don't write any yet.

Key files to read in SGLang:
- `sglang/srt/models/qwen3_moe.py` — the SGLang Qwen3 MoE model implementation
- How SGLang registers model classes (`EntryClass`, `AutoModelForCausalLM` mapping)
- Where post-load hooks can be inserted (look for `model_runner.py`, `server_args.py`)
- Whether SGLang has a plugin/general_plugins mechanism like vLLM does

Answer these questions:
1. What class wraps the attention layers? (vLLM equivalent of `MultiHeadLatentAttentionWrapper`)
2. What is the SGLang equivalent of `VLLM_PLUGINS`? (How does vLLM's `general_plugins` entry point work, and does SGLang have something analogous?)
3. Does SGLang use `torch.compile`? If so, does patching the forward post-compile work?

### 3.4 SGLang Baseline Benchmark

Use SGLang's built-in benchmark:
```bash
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompt 200 \
    --request-rate 10 \
    --input-len 512 \
    --output-len 128
```

Record: TTFT (time-to-first-token), TPOT (time-per-output-token), throughput (req/s).  
This is the number to beat in Phase 4.

**Deliverables:** SGLang server running vanilla model, confirmed logit match, baseline latency/throughput numbers, annotated notes on where to insert the fusion hook.

---

## Phase 4 — Add Algo (Weight + Kernel Fusion) to SGLang

**Goal:** Apply the same weight-fused checkpoint and runtime kernel fusion inside SGLang, matching or exceeding Phase 2 speedups at production inference scale.

This is the most complex phase. The key challenge is that SGLang uses its own model implementations with different layer hierarchies than HF `transformers`.

### 4.1 Adapt Weight-Fused Checkpoint for SGLang

The weight-fused checkpoint from Phase 2 (`/data/Qwen3.6-35B-A3B-bf16-fused`) should load into SGLang without modification — the checkpoint is just safetensors with the same key names. Verify:

- Load the fused checkpoint into SGLang
- Run the reference prompt
- Confirm logits match Phase 2's fused-checkpoint result
- If there is a mismatch, diff the weight key names between HF `transformers` and SGLang (they sometimes differ)

### 4.2 Write the SGLang Fusion Plugin/Hook

SGLang does not have a `general_plugins` system identical to vLLM's, but it does expose post-load hooks through the `model_runner`. The approach is:

**Option A — SGLang model override:**  
Subclass SGLang's `Qwen3MoeForCausalLM` and override the decoder layer forward. Register the subclass as the model implementation via SGLang's model registry. This is the cleanest approach if SGLang supports custom model registration.

**Option B — Post-load monkey patch (mirrors this repo's vLLM plugin):**  
After SGLang loads the model, iterate over `model.layers`, apply the same `patch_qwen3_layers.py` logic from Phase 2 but targeting SGLang's module hierarchy. Inject via a startup script or SGLang's `--load-format` hook.

Preference: Option B first (mirrors what works in `deploy_lib/apply_fusion.py` for vLLM), then refactor to Option A if SGLang's module system resists it.

**File to write: `sglang_fusion_plugin/apply_fusion_sglang.py`**

Structure mirrors `kimi-fused-nvfp4-vllm/deploy_lib/apply_fusion.py`:
```
apply_qwen3_kernel_fusion(model)
  -> resolve_decoder_layers(model)   # find model.layers
  -> assert_weight_fused(model)      # verify gamma ~ 1
  -> patch_decoder_input_norm(layer) # site 1: bypass input_layernorm
  -> patch_moe_experts(layer)        # site 2: per-expert rms fusion
```

The `FusedRMSNormNVFP4Linear` class from `fusion/nvfp4_fused_rmsnorm_linear.py` is reused directly — it does not depend on vLLM internals, only on `nn.Module` and `torch`.

### 4.3 MoE Expert Fusion Detail

This is the new work not present in the Kimi repo. For each expert in `mlp.experts`:

**Before:**
```
def forward(x):
    gate = gate_proj(x)        # [T, 1536]
    up   = up_proj(x)          # [T, 1536]
    return down_proj(silu(gate) * up)
```

**After (weight-fused, kernel-fused):**
```
def forward(x):
    # gamma absorbed into weights; kernel divides by rms(x)
    gate = FusedRMSNormLinear(gate_proj)(x)   # gate_proj(x) / rms(x)
    up   = FusedRMSNormLinear(up_proj)(x)     # up_proj(x) / rms(x)
    return down_proj(silu(gate) * up)
```

**Optimization note:** `rms(x)` is computed twice (once for gate, once for up). A V2 variant should compute rms once and pass it to both. This requires a small extension to `FusedRMSNormNVFP4Linear` — a `shared_rms` forward mode where the rms tensor is pre-computed and passed in.

### 4.4 SGLang Serving Configuration

Equivalent of `deploy_vllm.sh` but for SGLang:

```bash
SGLANG_FUSION=1 \
FUSION_VARIANT=V2 \
python -m sglang.launch_server \
    --model-path /data/Qwen3.6-35B-A3B-bf16-fused \
    --tp 1 \
    --dtype bfloat16 \
    --model-impl qwen3_moe_fused \   # if using Option A
    --port 30000
```

Or via monkey-patch wrapper around `launch_server` (Option B).

### 4.5 Correctness Validation

Same three-level check as Phase 2:
1. Weight-fused checkpoint in SGLang: logits match Phase 2.1
2. Weight-fused + kernel-fused in SGLang: logits match Phase 2.2
3. Run a longer generation (200 tokens) and compare full output against vanilla SGLang Phase 3

Acceptable tolerance: `atol=1e-2` on logits, exact token match for greedy decoding.

### 4.6 Final Benchmark

Run the same SGLang bench from Phase 3 but with the fused model:

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompt 200 \
    --request-rate 10 \
    --input-len 512 \
    --output-len 128
```

Compare TTFT, TPOT, throughput (req/s) against Phase 3 baseline.  
Also benchmark Phase 2 (HF+fusion) vs Phase 4 (SGLang+fusion) to confirm SGLang's batching advantage at high concurrency.

**Deliverables:** `sglang_fusion_plugin/`, fused SGLang server running, final benchmark table comparing all 4 phases.

**Status (2026-06):** Implemented as `phase4/` with plugin `qwen_fusion` (HF `patch_hf_kernel_fusion` on `post_load_weights`). Correctness passes; **no E2E speedup** — HF forward replacement regresses ~15–20% at small batch vs Phase 3. Fused weights only (`--no-kernel`) ≈ parity (~0.98–1.00×). Fused ckpt needs vanilla metadata sync (`config.json`, `preprocessor_config.json`, etc.) for SGLang multimodal load.

---

## Phase 5 — Native SGLang Layer Fusion (Experimental)

**Goal:** Apply Site-1 fusion **inside SGLang’s own layer graph** (`Qwen3_5LinearDecoderLayer` / `Qwen3_5AttentionDecoderLayer`) instead of replacing forwards with the HF `Qwen3_5MoeDecoderLayer` patch from Phase 4. Keep Phase 4 frozen as the HF-plugin baseline; iterate in `phase5/`.

**Motivation:** Phase 4 proved that swapping in HF decoder `forward` bypasses SGLang’s optimized norm/GEMM/attention/MoE stack. Phase 5 tests whether hooking the **native** path — skip `input_layernorm`, fuse projections only — recovers parity or yields a small gain.

### 5.1 Approach (vs Phase 4)

| | Phase 4 | Phase 5 |
|---|---------|---------|
| Target layers | HF `Qwen3_5MoeDecoderLayer` (if present) | Native `Qwen3_5*DecoderLayer` |
| Plugin entry | `qwen_fusion` | `qwen_fusion_native` |
| Patch scope | Whole decoder `forward` (Phase 2) | `LayerCommunicator.prepare_attn` + input projections |
| MoE (Site 2) | HF `--site2` only (not on native path) | Stock `post_attention_layernorm` + `FusedMoE` |

**Site-1 native hooks (per layer, after weight load):**

1. **`prepare_attn`** — skip `input_layernorm`; keep residual bookkeeping; compute shared `rms(x)` via Phase 2 `Site1RmsState`.
2. **Linear-attn layers** — patch `Qwen3_5GatedDeltaNet._forward_input_proj`: `in_proj_qkvz`, `in_proj_ba` as `linear(x) / rms(x)`.
3. **Full-attn layers** — patch `self_attention`: `qkv_proj` as `linear(x) / rms(x)`.
4. **Fallback** — stock `prepare_attn` on TP scatter, quant, or allreduce-fusion paths.

Weight-fused checkpoint and Phase 3 baseline CSV comparison unchanged from Phase 4.

### 5.2 Implementation layout

```
phase5/
  patch_sglang_native_fusion.py   # Native layer Site-1 hooks
  patch_sglang_kernel_fusion.py   # Layer resolution; native-first, HF fallback
  sglang_fusion_native_plugin.py  # post_load_weights → qwen_fusion_native
  benchmark_sglang_fused.py       # CSV benchmark_mode: sglang-fused-native
  setup_fusion_plugin.sh
  launch_server_fused.sh
```

Install and run:

```bash
bash phase5/setup_fusion_plugin.sh
export SGLANG_PLUGINS=qwen_fusion_native
export SGLANG_FUSION=1

python phase5/benchmark_sglang_fused.py \
    --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused \
    --baseline-csv phase3/results/benchmark_*_sglang-vanilla_prefill_na_full.csv \
    --check-logits
```

Scheduler logs should show `native=N` (e.g. 40 layers) in the fusion plugin line.

### 5.3 Correctness

Same gates as Phase 4:

- Fused checkpoint loads in SGLang (metadata synced from vanilla).
- `--check-logits` vs Phase 1 oracle (top-1 token match).
- No regression vs Phase 4 weights-only path on numerics.

### 5.4 Expected outcome (hypothesis)

Native hooks avoid HF forward overhead → at least match Phase 4 `--no-kernel` (~1.0×), possibly small Site-1 gain from skipping `input_layernorm` kernel launches.

### 5.5 Actual results (2026-06, Blackwell)

| Regime | Speedup vs Phase 3 vanilla |
|--------|---------------------------|
| Small batch (1×128–512) | **~0.97×** |
| Mid / large (8×2048, 32×*) | **~1.00×** |

**Conclusion:** Native Python Site-1 fixes the Phase 4 regression but **does not beat** stock SGLang. SGLang already runs fast separate norm + GEMM kernels; skipping norm and doing `matmul` + `/ rms` in Python is parity or slightly worse. Site-2 on native MoE was **not implemented** and is not expected to help (Phase 2 E2E: stock MoE + fused weights is faster than Site-2 runtime wrappers).

### 5.6 Future work (out of scope for Phase 5 plugins)

Meaningful SGLang speedup would require a **custom fused CUDA/Triton op** (`(x @ W) / rms(x)`) wired into a **local SGLang clone** (`sglang/srt/models/qwen3_5.py`), not external plugins. Realistic E2E gain even then: **~1–5%**.

**Deliverables:** `phase5/` plugin + benchmark CSVs under `phase5/results/`; comparison table vs Phase 3, Phase 4 (HF plugin), Phase 4 weights-only.

---

## Summary Table

| Phase | Backend | Fusion | Checkpoint | CSV mode | Goal |
|---|---|---|---|---|---|
| 1 | HF transformers | None | Vanilla BF16 | — | Baseline + correctness oracle |
| 2 | HF transformers | Weight + Kernel | BF16-fused | `hf-e2e` | Prove fusion correct + modest E2E gain (~1.0–1.06×) |
| 3 | SGLang | None | Vanilla BF16 | `sglang-vanilla` | SGLang baseline (~3–7× faster than HF E2E) |
| 4 | SGLang | Weight + HF kernel plugin | BF16-fused | `sglang-fused` | Fused serving via `qwen_fusion` — **no speedup; HF patch slower** |
| 5 | SGLang | Weight + **native** layer hooks | BF16-fused | `sglang-fused-native` | Native Site-1 in SGLang graph — **parity only (~1.0×)** |

### Recommendations (post Phase 5)

| Use case | Use |
|----------|-----|
| HF eager inference | Fused ckpt + Phase 2 Site-1 V2 |
| SGLang production | Vanilla ckpt, or fused ckpt **without** kernel plugin |
| Phase 4 / 5 plugins | Research only — not for production inference |
| Further SGLang gains | Local clone + fused norm+GEMM kernel in `qwen3_5.py` |

See [README.md](README.md) for full results and learnings across all phases.

---

## Phase 6 — Fused kernel inside SGLang (planned)

Implementation plan for a real fused norm+GEMM op in a **local SGLang clone** (`qwen3_5.py` / Triton). No upstream PR required. Plugins (Phases 4–5) are insufficient.

**Details:** [phase6/README.md](phase6/README.md)

---



