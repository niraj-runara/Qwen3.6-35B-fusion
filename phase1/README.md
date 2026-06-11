# Phase 1 — Vanilla Baseline

Load, inspect, and benchmark **Qwen3.6-35B-A3B** straight from HuggingFace with zero modifications.  
No fusion, no quantization — just a clean working baseline and a correctness oracle for Phase 2+.

---

## Scripts

| Script | What it does |
|---|---|
| `setup_env.sh` | Creates a venv and installs all Python dependencies |
| `download_model.sh` | Downloads the ~70 GB BF16 checkpoint from HuggingFace and verifies it |
| `run_phase1.py` | Inspects the model, captures the correctness oracle, runs the baseline benchmark |

Run them in this order.

---

## Step 1 — Environment Setup

```bash
bash setup_env.sh
```

This will:
- Check Python ≥ 3.11 is available
- Create a `.venv/` inside `phase1/`
- Auto-detect your CUDA version and install the matching PyTorch build
- Install `transformers`, `accelerate`, `safetensors`, `huggingface_hub`
- Print GPU name and free memory as a final sanity check

**Activate the venv before running anything else:**
```bash
source .venv/bin/activate
```

**Already have your own env?** Skip venv creation:
```bash
bash setup_env.sh --skip-venv
```

---

## Step 2 — Download the Model

```bash
bash download_model.sh
```

Downloads `Qwen/Qwen3.6-35B-A3B` to `/data/Qwen3.6-35B-A3B-bf16` (~70 GB).

**Custom download path:**
```bash
MODEL_DIR=/nvme/models/Qwen3.6-35B-A3B bash download_model.sh
```

**Private / gated model? Pass your HF token:**
```bash
HF_TOKEN=hf_xxx bash download_model.sh
```

**Already downloaded? Just verify the checkpoint:**
```bash
bash download_model.sh --verify-only
```

The verify step checks:
- `model.safetensors.index.json` is present
- All shards listed in the index are on disk
- Key tensor names exist (`q_proj`, `k_proj`, `v_proj`, `input_layernorm`, expert `gate_proj`)
- Prints the architecture summary from `config.json`

> **Disk space:** Requires ~80 GB free. The script will warn you if there isn't enough before starting.

---

## Step 3 — Run Phase 1

The run script has three independent tasks. Run them all at once:

```bash
python run_phase1.py --all
```

Or run each task individually:

```bash
# Task 1.3 — inspect module tree, class names, norm weight stats
python run_phase1.py --inspect

# Task 1.4 — capture correctness oracle (logits + top-5 predictions)
python run_phase1.py --reference

# Task 1.5 — baseline latency benchmark + torch.profiler trace
python run_phase1.py --benchmark
```

**Custom model path** (if you didn't use the default):
```bash
python run_phase1.py --all --model-dir /nvme/models/Qwen3.6-35B-A3B
```

**Custom benchmark config:**
```bash
python run_phase1.py --benchmark --seq-lens 256 512 1024 2048 --batch-size 4
```

---

## What Each Task Does

### `--inspect` (Task 1.3)
- Loads the model and prints the full module tree
- Lists the exact class names for each structural role (attention, MoE block, expert, norm) — needed before writing any patch in Phase 2
- Checks `input_layernorm.weight` and `post_attention_layernorm.weight` stats — confirms gamma ≠ 1, i.e. weight fusion has **not** been applied yet

### `--reference` (Task 1.4)
- Runs a fixed prompt through the model
- Saves the full logit vector and top-5 next-token predictions
- **These files are the correctness oracle.** Every Phase 2, 3, and 4 result must match them within `atol=1e-2`

### `--benchmark` (Task 1.5)
- Measures prefill latency at `seq_len = [512, 1024, 2048]` (configurable)
- Measures single-token decode latency with a live KV cache
- Runs `torch.profiler` on a 512-token forward pass and prints the top-15 CUDA ops by self-time
- This is the number to beat after fusion is applied in Phase 2

---

## Outputs

All outputs are written to `phase1/outputs/`:

| File | Created by | Contents |
|---|---|---|
| `module_tree.txt` | `--inspect` | Full `name → classname` listing for every module |
| `module_classes.json` | `--inspect` | Unique class names grouped by role (attention, expert, norm, …) |
| `norm_weights.json` | `--inspect` | Mean / std / min / max of norm weights for layer 0 and last layer |
| `reference_logits.pt` | `--reference` | `torch.Tensor` of shape `[vocab_size]` — the logit oracle |
| `reference_top5.json` | `--reference` | Human-readable top-5 predictions with token strings and logit values |
| `benchmark_results.json` | `--benchmark` | Latency table (prefill + decode) and top profiler ops |
| `profiler_trace/` | `--benchmark` | TensorBoard-compatible `torch.profiler` trace |

---

## Hardware

| Requirement | Value |
|---|---|
| GPU | Single RTX Pro 6000 (96 GB) or equivalent ≥ 80 GB VRAM |
| Model size | ~70 GB in BF16 |
| Headroom for KV cache + activations | ~18–22 GB on RTX Pro 6000 |
| CUDA | 12.x |
| Tensor parallelism | Not needed (`--tp 1`) |

> **Why not 2× A100-80G?**  
> Each A100 has 80 GB. The 70 GB model barely fits on one with almost no KV cache headroom.  
> A single 96 GB card is the cleaner setup for this phase.

---

## Troubleshooting

**`CUDA out of memory` on load**  
The model is 70 GB. If your GPU has less than 80 GB free, it will OOM. Check free memory:
```bash
nvidia-smi
```

**`huggingface-cli: command not found`**  
The download script will install `huggingface_hub` automatically, but if your venv isn't active it may install into the wrong env. Activate first:
```bash
source .venv/bin/activate
bash download_model.sh
```

**`transformers` can't find the Qwen3-MoE config**  
Upgrade transformers — Qwen3-MoE support requires ≥ 4.51.0:
```bash
pip install --upgrade transformers
```

**Download is slow / keeps timing out**  
Use `HF_HUB_ENABLE_HF_TRANSFER=1` for faster multi-part downloads:
```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 bash download_model.sh
```

**Benchmark runs but profiler trace is empty**  
Make sure CUDA activity profiling is enabled — it requires the model to actually run on GPU. Check `nvidia-smi` to confirm the process is using the GPU.
