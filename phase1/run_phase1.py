#!/usr/bin/env python3
"""
Phase 1 — Load, Inspect, Correctness Reference, and Baseline Benchmark
for Qwen3.6-35B-A3B (vanilla BF16 from HuggingFace).

Sub-tasks (run independently or all at once):
  --inspect     1.3  Load model and print full module tree, class names, norm weights
  --reference   1.4  Run fixed prompt, save logits + top-5 as correctness oracle
  --benchmark   1.5  Prefill/decode latency sweep + torch.profiler top-ops
  --all               Run all three in sequence

Usage examples:
  python run_phase1.py --all
  python run_phase1.py --inspect
  python run_phase1.py --reference
  python run_phase1.py --benchmark
  python run_phase1.py --all --model-dir /nvme/Qwen3.6-35B-A3B-bf16
  python run_phase1.py --benchmark --seq-lens 512 1024 --batch-size 4

Outputs (written to ./outputs/):
  module_tree.txt          Full printed model tree (--inspect)
  norm_weights.json        Layer 0 + 63 norm weight stats (--inspect)
  module_classes.json      Unique class names per module type (--inspect)
  reference_logits.pt      Logit tensor for the oracle prompt (--reference)
  reference_top5.json      Human-readable top-5 predictions (--reference)
  benchmark_results.json   Latency table (--benchmark)
  profiler_trace/          torch.profiler trace dir (--benchmark)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "/data/Qwen3.6-35B-A3B-bf16"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Fixed oracle prompt — must stay identical across all phases
REFERENCE_PROMPT = (
    "The key difference between a mixture-of-experts model and a dense model is"
)

# Benchmark sweep config
DEFAULT_SEQ_LENS = [512, 1024, 2048]
DEFAULT_BATCH_SIZE = 1
WARMUP_RUNS = 2
TIMED_RUNS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_and_tokenizer(model_dir: str):
    """Load Qwen3.6-35B-A3B in BF16. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[load] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[load] Loading model (BF16, device_map=auto) — this may take a few minutes...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # Disable torch.compile — we want a clean eager baseline
        attn_implementation="eager",
    )
    elapsed = time.time() - t0
    print(f"[load] Model loaded in {elapsed:.1f}s")
    model.eval()
    return model, tokenizer


def _get_primary_device(model) -> torch.device:
    """Return the device of the first parameter (where most of the model lives)."""
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Task 1.3 — Inspect
# ---------------------------------------------------------------------------

def run_inspect(model_dir: str) -> None:
    print("\n" + "=" * 60)
    print("TASK 1.3 — Model Inspection")
    print("=" * 60)

    _ensure_output_dir()
    model, tokenizer = _load_model_and_tokenizer(model_dir)

    # ------------------------------------------------------------------
    # 1. Full module tree
    # ------------------------------------------------------------------
    print("\n--- Full module tree (first 80 lines) ---")
    tree_lines = []
    for name, module in model.named_modules():
        line = f"{name:80s}  {type(module).__name__}"
        tree_lines.append(line)

    for line in tree_lines[:80]:
        print(" ", line)
    if len(tree_lines) > 80:
        print(f"  ... ({len(tree_lines) - 80} more lines — see outputs/module_tree.txt)")

    tree_path = OUTPUT_DIR / "module_tree.txt"
    tree_path.write_text("\n".join(tree_lines))
    print(f"\n[saved] Full tree -> {tree_path}")

    # ------------------------------------------------------------------
    # 2. Unique module class names, grouped by structural role
    # ------------------------------------------------------------------
    role_patterns = {
        "decoder_layer": "model.layers.",
        "attention":     ".self_attn",
        "mlp_block":     ".mlp",
        "expert":        ".mlp.experts.",
        "norm":          "layernorm",
    }
    class_map: dict[str, set[str]] = {r: set() for r in role_patterns}

    for name, module in model.named_modules():
        name_lower = name.lower()
        for role, pat in role_patterns.items():
            if pat.lower() in name_lower:
                class_map[role].add(type(module).__name__)

    print("\n--- Module class names by role ---")
    class_map_serializable = {k: sorted(v) for k, v in class_map.items()}
    for role, classes in class_map_serializable.items():
        print(f"  {role:<20}: {classes}")

    (OUTPUT_DIR / "module_classes.json").write_text(
        json.dumps(class_map_serializable, indent=2)
    )
    print(f"[saved] Module classes -> {OUTPUT_DIR / 'module_classes.json'}")

    # ------------------------------------------------------------------
    # 3. Norm weight stats — confirm gamma != 1 (not yet fused)
    # ------------------------------------------------------------------
    print("\n--- Norm weight stats (input_layernorm + post_attention_layernorm) ---")
    print("  Expect: mean != 1.0 and std != 0.0 (weights not yet fused)")
    print(f"  {'Layer':<8} {'Norm':<35} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  {'gamma~1?':>9}")

    layers = model.model.layers
    norm_stats = []

    # Check first and last layer
    for idx in [0, len(layers) - 1]:
        layer = layers[idx]
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            norm = getattr(layer, norm_name, None)
            if norm is None or not hasattr(norm, "weight"):
                continue
            w = norm.weight.float()
            mean = w.mean().item()
            std  = w.std().item()
            wmin = w.min().item()
            wmax = w.max().item()
            gamma_one = abs(mean - 1.0) < 1e-3 and std < 1e-3

            status = "YES (already fused?)" if gamma_one else "no  (unfused, correct)"
            row = {
                "layer": idx,
                "norm": norm_name,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(wmin, 6),
                "max": round(wmax, 6),
                "gamma_approx_one": gamma_one,
            }
            norm_stats.append(row)
            print(f"  {idx:<8} {norm_name:<35} {mean:>8.4f} {std:>8.4f} {wmin:>8.4f} {wmax:>8.4f}  {status}")

    (OUTPUT_DIR / "norm_weights.json").write_text(json.dumps(norm_stats, indent=2))
    print(f"[saved] Norm stats -> {OUTPUT_DIR / 'norm_weights.json'}")

    # ------------------------------------------------------------------
    # 4. Architecture summary from model config
    # ------------------------------------------------------------------
    cfg = model.config
    print("\n--- Architecture summary ---")
    attrs = [
        "model_type", "num_hidden_layers", "hidden_size",
        "num_attention_heads", "num_key_value_heads",
        "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "vocab_size", "torch_dtype",
        "rms_norm_eps",
    ]
    for attr in attrs:
        val = getattr(cfg, attr, "—")
        print(f"  {attr:<30}: {val}")

    # ------------------------------------------------------------------
    # 5. GPU memory after load
    # ------------------------------------------------------------------
    print("\n--- GPU memory after model load ---")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            print(f"  GPU {i} ({props.name}): {alloc:.2f} GB allocated | "
                  f"{reserved:.2f} GB reserved | {total:.2f} GB total")

    print("\n[DONE] Inspection complete.")


# ---------------------------------------------------------------------------
# Task 1.4 — Correctness Reference
# ---------------------------------------------------------------------------

def run_reference(model_dir: str) -> None:
    print("\n" + "=" * 60)
    print("TASK 1.4 — Correctness Reference")
    print("=" * 60)

    _ensure_output_dir()
    model, tokenizer = _load_model_and_tokenizer(model_dir)
    device = _get_primary_device(model)

    print(f"\nPrompt: \"{REFERENCE_PROMPT}\"")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    inputs = tokenizer(REFERENCE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    print(f"Input tokens: {seq_len}")

    # ------------------------------------------------------------------
    # Forward pass (no grad, no sampling)
    # ------------------------------------------------------------------
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Logits for the LAST input token position (prediction of next token)
    logits = outputs.logits[0, -1, :]   # shape: [vocab_size]
    logits_cpu = logits.float().cpu()

    # ------------------------------------------------------------------
    # Top-5 predictions
    # ------------------------------------------------------------------
    top5_vals, top5_ids = torch.topk(logits_cpu, k=5)
    top5_tokens = [tokenizer.decode([tid.item()]) for tid in top5_ids]

    print("\n--- Top-5 next-token predictions ---")
    top5_records = []
    for rank, (tok, tid, val) in enumerate(
        zip(top5_tokens, top5_ids.tolist(), top5_vals.tolist()), start=1
    ):
        record = {
            "rank": rank,
            "token_id": tid,
            "token_str": tok,
            "logit": round(val, 4),
        }
        top5_records.append(record)
        print(f"  #{rank}  id={tid:7d}  logit={val:8.3f}  repr={repr(tok)}")

    # ------------------------------------------------------------------
    # Save oracle files
    # ------------------------------------------------------------------
    logits_path = OUTPUT_DIR / "reference_logits.pt"
    torch.save(logits_cpu, logits_path)
    print(f"\n[saved] Logit tensor ({logits_cpu.shape}) -> {logits_path}")

    top5_path = OUTPUT_DIR / "reference_top5.json"
    meta = {
        "prompt": REFERENCE_PROMPT,
        "model_dir": model_dir,
        "input_token_count": seq_len,
        "atol_threshold": 0.01,
        "top5": top5_records,
        "note": (
            "Logit tensor saved to reference_logits.pt. "
            "Phase 2+ must match top-1 token and pass "
            "torch.allclose(atol=0.01) against this tensor."
        ),
    }
    top5_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[saved] Top-5 reference -> {top5_path}")

    print("\n[DONE] Correctness reference saved.")
    print(f"       Oracle top-1 token: {repr(top5_tokens[0])} (id={top5_ids[0].item()})")


# ---------------------------------------------------------------------------
# Task 1.5 — Baseline Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model_dir: str,
    seq_lens: list[int],
    batch_size: int,
) -> None:
    print("\n" + "=" * 60)
    print("TASK 1.5 — Baseline Performance Benchmark")
    print("=" * 60)
    print(f"Seq lens   : {seq_lens}")
    print(f"Batch size : {batch_size}")
    print(f"Warmup     : {WARMUP_RUNS} runs | Timed: {TIMED_RUNS} runs")

    _ensure_output_dir()
    model, tokenizer = _load_model_and_tokenizer(model_dir)
    device = _get_primary_device(model)

    results = []

    # ------------------------------------------------------------------
    # Helper: build a dummy input of given (batch, seq_len)
    # ------------------------------------------------------------------
    def _make_inputs(b: int, s: int) -> dict:
        # Use the tokenizer's pad token id or 0 as dummy
        pad_id = tokenizer.pad_token_id or 0
        ids = torch.full((b, s), pad_id, dtype=torch.long, device=device)
        mask = torch.ones((b, s), dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": mask}

    # ------------------------------------------------------------------
    # Prefill latency sweep
    # ------------------------------------------------------------------
    print("\n--- Prefill latency (full forward pass, no generation) ---")
    print(f"  {'batch':>5} {'seq_len':>8} {'mean_ms':>10} {'std_ms':>8} {'tok/s':>10}")

    for seq_len in seq_lens:
        inputs = _make_inputs(batch_size, seq_len)

        # Warmup
        for _ in range(WARMUP_RUNS):
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()

        # Timed
        times_ms = []
        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)

        mean_ms = sum(times_ms) / len(times_ms)
        std_ms  = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
        toks_per_sec = (batch_size * seq_len) / (mean_ms / 1000)

        row = {
            "task": "prefill",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "tokens_per_sec": round(toks_per_sec, 1),
        }
        results.append(row)
        print(f"  {batch_size:>5} {seq_len:>8} {mean_ms:>10.2f} {std_ms:>8.2f} {toks_per_sec:>10.1f}")

    # ------------------------------------------------------------------
    # Single-token decode latency (autoregressive step simulation)
    # We pass a context of 512 tokens + 1 new token as input, measure
    # the cost of one additional forward step with past_key_values.
    # ------------------------------------------------------------------
    print("\n--- Single-token decode latency (with KV cache) ---")

    DECODE_CTX = 512
    ctx_inputs = _make_inputs(batch_size, DECODE_CTX)

    print(f"  Building KV cache from {DECODE_CTX} context tokens...")
    with torch.no_grad():
        ctx_out = model(**ctx_inputs, use_cache=True)
    past_kv = ctx_out.past_key_values
    torch.cuda.synchronize()

    # Decode step: one new token
    new_token = torch.full((batch_size, 1), tokenizer.pad_token_id or 0,
                           dtype=torch.long, device=device)

    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = model(input_ids=new_token, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()

    # Timed
    decode_times_ms = []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=new_token, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
        decode_times_ms.append((time.perf_counter() - t0) * 1000)

    mean_dec = sum(decode_times_ms) / len(decode_times_ms)
    std_dec  = (sum((t - mean_dec) ** 2 for t in decode_times_ms) / len(decode_times_ms)) ** 0.5
    dec_tps  = batch_size / (mean_dec / 1000)  # tokens/sec (1 token per step)

    row = {
        "task": "decode_step",
        "batch_size": batch_size,
        "context_len": DECODE_CTX,
        "mean_ms": round(mean_dec, 2),
        "std_ms": round(std_dec, 2),
        "tokens_per_sec": round(dec_tps, 1),
    }
    results.append(row)
    print(f"  {'batch':>5} {'ctx_len':>8} {'mean_ms':>10} {'std_ms':>8} {'tok/s':>10}")
    print(f"  {batch_size:>5} {DECODE_CTX:>8} {mean_dec:>10.2f} {std_dec:>8.2f} {dec_tps:>10.1f}")

    # ------------------------------------------------------------------
    # GPU memory snapshot
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            results.append({
                "task": "gpu_memory",
                "gpu_id": i,
                "allocated_gb": round(alloc, 3),
                "reserved_gb": round(reserved, 3),
            })

    # ------------------------------------------------------------------
    # torch.profiler — identify top ops
    # ------------------------------------------------------------------
    print("\n--- Profiling forward pass (seq_len=512, batch=1) ---")
    print("  Running torch.profiler (this adds ~30s overhead)...")

    from torch.profiler import profile, record_function, ProfilerActivity

    prof_inputs = _make_inputs(1, 512)
    prof_dir = OUTPUT_DIR / "profiler_trace"
    prof_dir.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
    ) as prof:
        for _ in range(3):
            with record_function("forward_pass"):
                with torch.no_grad():
                    _ = model(**prof_inputs)
            prof.step()

    # Print top-15 ops by CUDA time
    print("\n  Top 15 ops by CUDA self time:")
    print(f"  {'Name':<55} {'CUDA self':>12} {'calls':>6}")
    top_events = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=15
    )
    # Print just the table (it's already formatted)
    for line in top_events.splitlines()[:20]:
        print("  " + line)

    top_ops = []
    for ev in sorted(prof.key_averages(), key=lambda e: e.self_cuda_time_total, reverse=True)[:15]:
        top_ops.append({
            "name": ev.key,
            "cuda_self_us": ev.self_cuda_time_total,
            "count": ev.count,
        })
    results.append({"task": "profiler_top_ops", "top_ops": top_ops})

    print(f"  [saved] Profiler trace -> {prof_dir}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = OUTPUT_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] Benchmark results -> {out_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"  {'Task':<20} {'Batch':>5} {'Seq':>6} {'Mean ms':>9} {'Tok/s':>9}")
    for row in results:
        if row["task"] not in ("prefill", "decode_step"):
            continue
        seq = row.get("seq_len") or row.get("context_len", "-")
        print(f"  {row['task']:<20} {row['batch_size']:>5} {str(seq):>6} "
              f"{row['mean_ms']:>9.2f} {row['tokens_per_sec']:>9.1f}")

    print("\n[DONE] Benchmark complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: inspect, reference, and benchmark Qwen3.6-35B-A3B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        help="Path to the BF16 HuggingFace checkpoint",
    )
    p.add_argument("--inspect",   action="store_true", help="Run task 1.3: model inspection")
    p.add_argument("--reference", action="store_true", help="Run task 1.4: correctness reference")
    p.add_argument("--benchmark", action="store_true", help="Run task 1.5: baseline benchmark")
    p.add_argument("--all",       action="store_true", help="Run all tasks (1.3 + 1.4 + 1.5)")
    p.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=DEFAULT_SEQ_LENS,
        metavar="N",
        help="Sequence lengths for prefill benchmark",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for benchmark",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Default to --all if nothing specified
    if not any([args.inspect, args.reference, args.benchmark, args.all]):
        print("No task specified. Use --inspect, --reference, --benchmark, or --all.")
        print("Run with --help for usage.")
        sys.exit(1)

    run_inspect_flag   = args.all or args.inspect
    run_reference_flag = args.all or args.reference
    run_benchmark_flag = args.all or args.benchmark

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"ERROR: model dir not found: {model_dir}")
        print("Run download_model.sh first, or pass --model-dir <path>.")
        sys.exit(1)

    print(f"Model dir : {model_dir}")
    print(f"Output dir: {OUTPUT_DIR}")

    if run_inspect_flag:
        run_inspect(model_dir)

    if run_reference_flag:
        run_reference(model_dir)

    if run_benchmark_flag:
        run_benchmark(model_dir, args.seq_lens, args.batch_size)

    print("\n" + "=" * 60)
    print("Phase 1 complete. Outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
