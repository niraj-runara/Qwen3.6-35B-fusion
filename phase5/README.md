# Phase 5 — Native SGLang layer fusion

Experimental follow-on to **Phase 4**. Same fused checkpoint and Phase 3 baseline comparison, but Site-1 fusion is wired into **SGLang’s native** `Qwen3_5*DecoderLayer` forwards instead of replacing them with the HF `patch_hf_kernel_fusion` path.

| Phase | Fusion hook | Plugin | CSV `benchmark_mode` |
|-------|-------------|--------|----------------------|
| 4 | HF `Qwen3_5MoeDecoderLayer` forward | `qwen_fusion` | `sglang-fused` |
| **5** | Native `Qwen3_5LinearDecoderLayer` / `Qwen3_5AttentionDecoderLayer` | `qwen_fusion_native` | `sglang-fused-native` |

Phase 4 is unchanged — use it for the HF-plugin baseline. Phase 5 is where native-layer experiments live.

---

## What native fusion does

On each decoder layer (after weight load, via `post_load_weights` plugin):

1. **`LayerCommunicator.prepare_attn`** — skip `input_layernorm`; keep residual bookkeeping.
2. **Projections** — `linear(x) / rms(x)` via Phase 2 `Site1RmsState` on:
   - `Qwen3_5GatedDeltaNet`: `in_proj_qkvz`, `in_proj_ba`
   - `Qwen3_5AttentionDecoderLayer`: `qkv_proj`
3. **MoE** — stock `post_attention_layernorm` + `FusedMoE` (weights already fused offline).

Falls back to stock `prepare_attn` on TP scatter / quant / allreduce-fusion paths. HF `Qwen3_5MoeDecoderLayer` layers still use Phase 2 patch if encountered.

---

## Prerequisites

Same as Phase 4:

1. Phase 3 baseline CSV in `phase3/results/`
2. Fused checkpoint: `/data/Qwen3.6-35B-A3B-bf16-fused`
3. `source phase1/.venv/bin/activate` + Phase 3 SGLang env

```bash
bash phase5/setup_fusion_plugin.sh
```

---

## Benchmark

```bash
cd /path/to/qwen3.6-35B-fusion
source phase1/.venv/bin/activate
export LD_LIBRARY_PATH="$(python -c "import site; s=site.getsitepackages()[0]; print(':'.join([s+'/nvidia/cuda_nvrtc/lib', s+'/nvidia/cuda_runtime/lib', s+'/nvidia/cu13/lib']))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SGLANG_PLUGINS=qwen_fusion_native
export SGLANG_FUSION=1

python phase5/benchmark_sglang_fused.py \
    --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused \
    --vanilla-dir /data/Qwen3.6-35B-A3B-bf16 \
    --baseline-csv phase3/results/benchmark_20260612T001855Z_sglang-vanilla_prefill_na_full.csv \
    --check-logits
```

**Weights only** (no runtime patch): add `--no-kernel`.

Check scheduler logs for `native=N` in the fusion plugin line to confirm native patching.

---

## HTTP server

```bash
bash phase5/launch_server_fused.sh
```

---

## Output

`phase5/results/benchmark_<timestamp>_sglang-fused-native_prefill_V2_full.csv`

Compare speedup vs Phase 3 vanilla and vs Phase 4 `sglang-fused` results.

---

## Files

| File | Purpose |
|------|---------|
| `setup_fusion_plugin.sh` | `pip install -e phase5/` → `qwen_fusion_native` |
| `sglang_fusion_native_plugin.py` | `post_load_weights` hook |
| `patch_sglang_native_fusion.py` | Native layer Site-1 patch |
| `patch_sglang_kernel_fusion.py` | Layer resolution + native/HF dispatch |
| `benchmark_sglang_fused.py` | 9-shape prefill sweep |
| `launch_server_fused.sh` | HTTP server with native plugin |
