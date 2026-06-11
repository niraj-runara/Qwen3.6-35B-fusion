"""
Phase 1 — Layer-level vanilla baseline benchmark.

Benchmarks the unfused norm→linear operation for each Qwen3.6-35B-A3B
fusion site using the same metric set as benchmark_reference.py:
  - Latency  : median ms, p99 ms (200 timed runs)
  - Memory   : peak GPU memory allocated per forward pass (MB)
  - Throughput: tokens / second

This is the "unfused" half of the Phase 2 comparison.  Running this
produces a CSV in the benchmark_reference format so Phase 1 and Phase 2
results are directly comparable.

Fusion sites (Qwen3_5MoeDecoderLayer — same class as Qwen3.6-35B-A3B):
  attn  input_layernorm + token-mixer input projection
        linear_attention (3 of 4 layers): linear_attn.in_proj_qkv
        full_attention   (1 of 4 layers): self_attn.q_proj
  moe   post_attention_layernorm + gate half of mlp.experts.gate_up_proj[expert_idx]
        (Qwen3_5MoeExperts stores gate+up as one fused [E, 2*I, H] tensor)

Usage
─────
  # Default: benchmark both sites, layer 0
  python benchmark_layer.py --model-dir /data/Qwen3.6-35B-A3B-bf16

  # Single site
  python benchmark_layer.py --model-dir /data/Qwen3.6-35B-A3B-bf16 --site attn

  # Different layer
  python benchmark_layer.py --model-dir /data/Qwen3.6-35B-A3B-bf16 --layer-idx 32

  # Smoke test (one forward pass, no timing sweep)
  python benchmark_layer.py --model-dir /data/Qwen3.6-35B-A3B-bf16 --test-load
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sys-path setup — benchmark_reference.py lives at repo root
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_reference import (   # noqa: E402
    ShapeResult,
    measure_latency,
    measure_peak_memory,
    print_summary_table,
    save_csv,
    WARMUP_ITERS,
    MEASURE_ITERS,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DTYPE  = torch.bfloat16
DEVICE = "cuda:0"

_BATCH_SEQ_PAIRS = [
    (1,   128),
    (1,   512),
    (1,  2048),
    (8,   128),
    (8,   512),
    (8,  2048),
    (32,  128),
    (32,  512),
    (32, 2048),
]

_RESULTS_DIR = _THIS_DIR / "outputs" / "benchmark_layer"

# Qwen3.6-35B-A3B decoder layer types (config.layer_types, 3:1 hybrid stack)
_LAYER_LINEAR = "linear_attention"
_LAYER_FULL   = "full_attention"
_DECODER_CLS  = "Qwen3_5MoeDecoderLayer"


# ---------------------------------------------------------------------------
# Unfused benchmark wrapper
# ---------------------------------------------------------------------------

class _UnfusedNormLinear(nn.Module):
    """norm(x) → linear(x) — the plain unfused path."""

    def __init__(self, norm: nn.Module, linear: nn.Module):
        super().__init__()
        self.norm   = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.norm(x))
        return out[0] if isinstance(out, tuple) else out


class _ExpertGateProj(nn.Module):
    """
    Gate projection for one expert in Qwen3_5MoeExperts.

    Experts store gate and up weights together in gate_up_proj [E, 2*I, H];
    the gate matrix is the first I rows for each expert.
    """

    def __init__(self, gate_up_proj: torch.Tensor, expert_idx: int):
        super().__init__()
        intermediate = gate_up_proj.shape[1] // 2
        self.weight = nn.Parameter(
            gate_up_proj[expert_idx, :intermediate, :].detach().clone(),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_model(model_dir: str) -> nn.Module:
    from transformers import AutoModelForCausalLM
    print(f"\nLoading model from {model_dir} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return model


def _extract_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    layer = copy.deepcopy(model.model.layers[layer_idx])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    layer = layer.to(device=DEVICE, dtype=DTYPE).eval()

    if type(layer).__name__ != _DECODER_CLS:
        raise TypeError(
            f"Expected {_DECODER_CLS} at model.layers[{layer_idx}], "
            f"got {type(layer).__name__}"
        )
    return layer


# ---------------------------------------------------------------------------
# Build benchmark modules — Qwen3_5MoeDecoderLayer fusion sites
# ---------------------------------------------------------------------------

def _attn_input_proj(layer: nn.Module) -> nn.Module:
    """Token-mixer input linear for this layer's hybrid attention type."""
    if layer.layer_type == _LAYER_LINEAR:
        return layer.linear_attn.in_proj_qkv
    if layer.layer_type == _LAYER_FULL:
        return layer.self_attn.q_proj
    raise ValueError(f"Unknown layer_type={layer.layer_type!r}")


def _moe_gate_proj(layer: nn.Module, expert_idx: int) -> nn.Module:
    """Gate projection for one routed expert (Qwen3_5MoeExperts.gate_up_proj)."""
    gate_up = layer.mlp.experts.gate_up_proj
    num_experts = gate_up.shape[0]
    if not 0 <= expert_idx < num_experts:
        raise IndexError(f"expert_idx={expert_idx} out of range (num_experts={num_experts})")
    return _ExpertGateProj(gate_up, expert_idx)


def _build_attn_module(layer: nn.Module) -> _UnfusedNormLinear:
    return _UnfusedNormLinear(layer.input_layernorm, _attn_input_proj(layer))


def _build_moe_module(layer: nn.Module, expert_idx: int = 0) -> _UnfusedNormLinear:
    return _UnfusedNormLinear(
        layer.post_attention_layernorm,
        _moe_gate_proj(layer, expert_idx),
    )


def _attn_site_label(layer: nn.Module) -> str:
    if layer.layer_type == _LAYER_LINEAR:
        return "input_layernorm + linear_attn.in_proj_qkv"
    return "input_layernorm + self_attn.q_proj"


# ---------------------------------------------------------------------------
# Benchmark one site
# ---------------------------------------------------------------------------

def _benchmark_site(
    module: nn.Module,
    site_label: str,
    layer_idx: int,
    hidden_dim: int,
) -> list[ShapeResult]:
    results: list[ShapeResult] = []

    module = module.to(DEVICE, dtype=DTYPE).eval()

    for batch, seq_len in _BATCH_SEQ_PAIRS:
        print(f"\n{'=' * 60}")
        print(f"[{site_label}  layer={layer_idx}]  batch={batch}  seq={seq_len}  hidden={hidden_dim}")
        print(f"{'=' * 60}")

        x = torch.randn(batch, seq_len, hidden_dim, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            sample_out = module(x)
        out_dim = sample_out.shape[-1]

        result = ShapeResult(
            batch=batch, seq_len=seq_len,
            hidden=hidden_dim, out_dim=out_dim,
        )

        print("  Measuring latency (unfused) ...")
        # Store in nonfused_latencies (this IS the unfused baseline)
        result.nonfused_latencies = measure_latency(
            module, x, warmup=WARMUP_ITERS, measure=MEASURE_ITERS
        )
        # Mirror to fused_latencies so ShapeResult properties work
        # (no fused model here; speedup will show 1.0x)
        result.fused_latencies = result.nonfused_latencies[:]

        print("  Measuring peak memory ...")
        result.nonfused_peak_mem_mb = measure_peak_memory(module, x)
        result.fused_peak_mem_mb    = result.nonfused_peak_mem_mb

        print(
            f"\n  Latency (median ms) : {result.nonfused_median_ms:.3f}"
            f"  p99: {result.nonfused_p99_ms:.3f}"
        )
        print(f"  Throughput (tok/s)  : {result.nonfused_throughput:,.0f}")
        print(f"  Peak mem (MB)       : {result.nonfused_peak_mem_mb:.1f}")

        del x, sample_out
        gc.collect()
        torch.cuda.empty_cache()

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_vanilla_table(results: list[ShapeResult], site_label: str) -> None:
    header = (
        f"{'batch':>5} {'seq':>5} {'hidden':>7} {'out_dim':>8} "
        f"{'median_ms':>10} {'p99_ms':>9} {'tok/s':>10} {'mem_MB':>8}"
    )
    print(f"\n{'=' * len(header)}")
    print(f"VANILLA BASELINE — {site_label}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.batch:>5} {r.seq_len:>5} {r.hidden:>7} {r.out_dim:>8} "
            f"{r.nonfused_median_ms:>10.3f} {r.nonfused_p99_ms:>9.3f} "
            f"{r.nonfused_throughput:>10,.0f} {r.nonfused_peak_mem_mb:>8.1f}"
        )
    print(f"{'=' * len(header)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: layer-level unfused baseline benchmark for Qwen3.6-35B-A3B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/Qwen3.6-35B-A3B-bf16"),
        help="Path to the vanilla BF16 checkpoint",
    )
    p.add_argument(
        "--site",
        choices=("attn", "moe", "all"),
        default="all",
        help="Which fusion site(s) to benchmark",
    )
    p.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Decoder layer index to extract",
    )
    p.add_argument(
        "--device",
        default=DEVICE,
        help="CUDA device",
    )
    p.add_argument(
        "--test-load",
        action="store_true",
        help="Smoke test: load + one forward pass only",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV to phase1/outputs/benchmark_layer/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        sys.exit(1)

    print("=" * 60)
    print("Qwen3.6-35B-A3B  Phase 1  Layer Baseline Benchmark")
    print("=" * 60)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Site    : {args.site}")
    print(f"  Layer   : {args.layer_idx}")
    print(f"  Warmup  : {WARMUP_ITERS}  |  Measure: {MEASURE_ITERS}")

    global DEVICE
    DEVICE = args.device

    model = _load_model(args.model_dir)
    layer = _extract_layer(model, args.layer_idx)
    hidden_size = layer.hidden_size

    print(f"  Layer type : {layer.layer_type}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attn site  : {_attn_site_label(layer)}")
    print(f"  MoE site   : post_attention_layernorm + mlp.experts.gate_up_proj[0] gate half")

    sites = ["attn", "moe"] if args.site == "all" else [args.site]

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------
    if args.test_load:
        for site in sites:
            module = _build_attn_module(layer) if site == "attn" else _build_moe_module(layer)
            module = module.to(DEVICE, dtype=DTYPE).eval()
            x = torch.randn(1, 128, hidden_size, device=DEVICE, dtype=DTYPE)
            with torch.no_grad():
                y = module(x)
            print(f"[{site}] Forward OK — output shape: {tuple(y.shape)}")
        print("\n(--test-load passed. Re-run without --test-load for full sweep.)")
        return

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------
    all_results: dict[str, list[ShapeResult]] = {}

    for site in sites:
        print(f"\n{'#' * 60}")
        print(f"# Site: {site}")
        print(f"{'#' * 60}")

        module = _build_attn_module(layer) if site == "attn" else _build_moe_module(layer)
        results = _benchmark_site(
            module,
            site_label=site,
            layer_idx=args.layer_idx,
            hidden_dim=hidden_size,
        )
        all_results[site] = results
        _print_vanilla_table(results, site_label=site)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if not args.no_save:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for site, results in all_results.items():
            fname = f"vanilla_baseline_{ts}_layer{args.layer_idx}_{site}.csv"
            path  = str(_RESULTS_DIR / fname)
            save_csv(
                results,
                path,
                run_metadata={
                    "run_timestamp_utc": run_ts,
                    "benchmark_mode":    "vanilla-unfused",
                    "fusion_point":      site,
                    "variant":           "n/a",
                    "load_mode":         "full",
                    "device":            args.device,
                },
            )
            print(f"[saved] {path}")

    print("\nPhase 1 layer benchmark complete.")


if __name__ == "__main__":
    main()
