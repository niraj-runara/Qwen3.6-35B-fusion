"""
Phase 2 — Benchmark: Fused vs Unfused Qwen3.6-35B-A3B (no inference engine)

Compares vanilla vs weight-fused checkpoints at each fusion site, using the
same metric set as benchmark_reference.py:
  - Latency  : median ms, p99 ms over 200 timed runs
  - Memory   : peak GPU memory allocated per forward pass (MB)
  - Numerical: max |diff|, cosine similarity, KL divergence

Arms (both checkpoints required — no runtime weight fusion):
  Unfused  vanilla checkpoint  → norm(x) → linear(x)
  Fused    fused checkpoint     → FusedRMSNormLinear(V1/V2): linear_fused(x) / rms(x)
                                   (γ absorbed offline in export_fused_weights.py)

Loads each full checkpoint sequentially, extracts one decoder layer, then frees
the rest (~70 GB). Only the two small layer slices stay on GPU for the benchmark.

Fusion sites (Qwen3_5MoeDecoderLayer):
  attn   input_layernorm → linear_attn.in_proj_qkv  OR  self_attn.q_proj
  moe    post_attention_layernorm → gate half of mlp.experts.gate_up_proj[0]

Usage
─────
  # Smoke test
  python benchmark_fused_vs_unfused.py \\
      --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \\
      --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \\
      --site attn --test-load

  # Full sweep (both sites, all shapes)
  python benchmark_fused_vs_unfused.py \\
      --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \\
      --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \\
      --site all

  # List module paths (no load)
  python benchmark_fused_vs_unfused.py \\
      --unfused-dir /data/Qwen3.6-35B-A3B-bf16 --print-keys
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
# ---------------------------------------------------------------------------
# Repo root on sys.path so we can import benchmark_reference and fusion_bf16
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_REPO_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import the reusable benchmark primitives from benchmark_reference.py
from benchmark_reference import (   # noqa: E402
    ShapeResult,
    measure_latency,
    measure_peak_memory,
    measure_numerical_equivalence,
    print_summary_table,
    save_csv,
    WARMUP_ITERS,
    MEASURE_ITERS,
    NUMERICAL_ITERS,
)

# Import BF16 fusion modules
from fusion_bf16 import VARIANTS, build_fused_module  # noqa: E402
from qwen3_moe_layers import (      # noqa: E402
    attn_input_proj,
    attn_site_label,
    moe_gate_proj,
    validate_decoder_layer,
)


# ---------------------------------------------------------------------------
# Qwen3-specific configuration
# ---------------------------------------------------------------------------

DTYPE  = torch.bfloat16
DEVICE = "cuda:0"

# Shape sweep — (batch, seq_len, hidden_dim)
# hidden_dim = 4096 for Qwen3.6-35B-A3B; kept as a variable so it auto-adjusts
# if a different layer's hidden dim is used.
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

# ---------------------------------------------------------------------------
# Benchmark wrapper modules (site-specific)
# ---------------------------------------------------------------------------

class _UnfusedNormLinear(nn.Module):
    """
    Baseline: apply RMSNorm then nn.Linear.
    Mirrors benchmark_reference._UnfusedNormLinear but for BF16 standard norm.
    """
    def __init__(self, norm: nn.Module, linear: nn.Module):
        super().__init__()
        self.norm   = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.norm(x))
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Model / layer loading
# ---------------------------------------------------------------------------

def _print_gpu_mem(label: str) -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  [{label}] GPU {i}: {alloc:.2f} / {total:.2f} GB allocated")


def _load_full_model(model_dir: str, label: str) -> nn.Module:
    from transformers import AutoModelForCausalLM
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")
    print(f"\nLoading {label} from {model_dir} ...")
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
    print(f"  {label} loaded in {time.time() - t0:.0f}s")
    _print_gpu_mem(label)
    return model


def _extract_layer(model: nn.Module, layer_idx: int, device: str) -> nn.Module:
    """Deep-copy one decoder layer onto `device` and free the full model."""
    layer = copy.deepcopy(model.model.layers[layer_idx])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    layer = layer.to(device=device, dtype=DTYPE).eval()
    validate_decoder_layer(layer, layer_idx)
    return layer


def _load_decoder_layer(
    model_dir: str,
    layer_idx: int,
    device: str,
    label: str,
) -> nn.Module:
    """
    Load a full checkpoint, extract one decoder layer, then drop the rest.

    Only one ~70 GB model is on GPU at a time — safe for a single RTX Pro 6000.
  """
    model = _load_full_model(model_dir, label)
    print(f"Extracting layer {layer_idx} from {label} (freeing full model) ...")
    return _extract_layer(model, layer_idx, device)


def print_model_keys(model_dir: str) -> None:
    """Print all module paths from the safetensors index (no weight load)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    paths = sorted({k.rsplit(".", 1)[0] for k in weight_map})
    print(f"\nModule paths in {index_path} ({len(paths)} total):\n")
    for p in paths[:60]:
        print(f"  {p}")
    if len(paths) > 60:
        print(f"  ... ({len(paths) - 60} more)")


# ---------------------------------------------------------------------------
# Build benchmark module pairs for each site
# ---------------------------------------------------------------------------

def _build_attn_pair(
    layer_unfused: nn.Module,
    layer_fused: nn.Module,
    *,
    variant: str,
) -> Tuple[nn.Module, nn.Module]:
    """Unfused: vanilla norm→linear.  Fused: fused ckpt weights + FusedRMSNormLinear."""
    unfused = _UnfusedNormLinear(
        layer_unfused.input_layernorm,
        attn_input_proj(layer_unfused),
    )
    fused = build_fused_module(
        attn_input_proj(layer_fused),
        layer_fused.input_layernorm,
        variant=variant,
    )
    return unfused, fused


def _build_moe_pair(
    layer_unfused: nn.Module,
    layer_fused: nn.Module,
    *,
    variant: str,
    expert_idx: int = 0,
) -> Tuple[nn.Module, nn.Module]:
    """Unfused: vanilla norm→gate.  Fused: fused ckpt gate + FusedRMSNormLinear."""
    unfused = _UnfusedNormLinear(
        layer_unfused.post_attention_layernorm,
        moe_gate_proj(layer_unfused, expert_idx),
    )
    fused = build_fused_module(
        moe_gate_proj(layer_fused, expert_idx),
        layer_fused.post_attention_layernorm,
        variant=variant,
    )
    return unfused, fused


# ---------------------------------------------------------------------------
# Core benchmark loop  (reuses benchmark_reference primitives)
# ---------------------------------------------------------------------------

def run_site_benchmark(
    unfused_bench: nn.Module,
    fused_bench:   nn.Module,
    hidden_dim:    int,
    site_label:    str,
    device:        str,
) -> list[ShapeResult]:
    """
    Run the full shape sweep for one fusion site.

    Uses measure_latency, measure_peak_memory, measure_numerical_equivalence
    directly from benchmark_reference — no changes to those functions.
    """
    results: list[ShapeResult] = []

    unfused_bench = unfused_bench.to(device, dtype=DTYPE).eval()
    fused_bench   = fused_bench.to(device, dtype=DTYPE).eval()

    for batch, seq_len in _BATCH_SEQ_PAIRS:
        print(f"\n{'=' * 62}")
        print(f"[{site_label}]  batch={batch}  seq={seq_len}  hidden={hidden_dim}")
        print(f"{'=' * 62}")

        x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=DTYPE)

        # Infer out_dim from one forward pass
        with torch.no_grad():
            out_nf = unfused_bench(x)
        out_dim = out_nf.shape[-1]

        result = ShapeResult(
            batch=batch, seq_len=seq_len,
            hidden=hidden_dim, out_dim=out_dim,
        )

        print("  Measuring latency (unfused) ...")
        result.nonfused_latencies = measure_latency(unfused_bench, x,
                                                    warmup=WARMUP_ITERS,
                                                    measure=MEASURE_ITERS)

        print("  Measuring latency (fused) ...")
        result.fused_latencies = measure_latency(fused_bench, x,
                                                  warmup=WARMUP_ITERS,
                                                  measure=MEASURE_ITERS)

        print("  Measuring peak memory ...")
        result.nonfused_peak_mem_mb = measure_peak_memory(unfused_bench, x)
        result.fused_peak_mem_mb    = measure_peak_memory(fused_bench,   x)

        print("  Measuring numerical equivalence ...")
        (result.max_abs_diff,
         result.cosine_sim,
         result.kl_divergence) = measure_numerical_equivalence(
            fused_bench, unfused_bench, x, n_iters=NUMERICAL_ITERS
        )

        # Per-shape summary
        print(
            f"\n  Latency (median ms):  fused={result.fused_median_ms:.3f}"
            f"  unfused={result.nonfused_median_ms:.3f}"
            f"  speedup={result.speedup:.2f}x"
        )
        print(
            f"  Latency (p99 ms):     fused={result.fused_p99_ms:.3f}"
            f"  unfused={result.nonfused_p99_ms:.3f}"
        )
        print(
            f"  Throughput (tok/s):   fused={result.fused_throughput:,.0f}"
            f"  unfused={result.nonfused_throughput:,.0f}"
        )
        print(
            f"  Peak mem (MB):        fused={result.fused_peak_mem_mb:.1f}"
            f"  unfused={result.nonfused_peak_mem_mb:.1f}"
        )
        print(f"  Numerical equivalence:")
        print(f"    max |diff|  = {result.max_abs_diff:.6f}")
        print(f"    cosine sim  = {result.cosine_sim:.6f}  (1.0 = identical)")
        print(f"    KL div      = {result.kl_divergence:.6f}  (0.0 = identical)")

        del x, out_nf
        gc.collect()
        torch.cuda.empty_cache()

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    unfused_dir: str,
    fused_dir:   str,
    *,
    sites:       list[str],
    layer_idx:   int = 0,
    variant:     str = "V2",
    device:      str = DEVICE,
) -> dict[str, list[ShapeResult]]:
    """
    Load layers, build benchmark pairs, run sweep for each requested site.

    Returns dict mapping site label → list[ShapeResult].
    """

    # ------------------------------------------------------------------
    # Load one decoder layer per checkpoint (sequential — not both 70 GB models)
    # ------------------------------------------------------------------
    layer_uf = _load_decoder_layer(unfused_dir, layer_idx, device, "unfused")
    layer_f  = _load_decoder_layer(fused_dir, layer_idx, device, "fused")
    hidden   = layer_uf.hidden_size

    print(f"\nLayer {layer_idx}: type={layer_uf.layer_type}  hidden={hidden}")
    print(f"  Attn site: {attn_site_label(layer_uf)}")
    print(f"  MoE site : post_attention_layernorm + mlp.experts.gate_up_proj[0] gate half")

    all_results: dict[str, list[ShapeResult]] = {}

    # ------------------------------------------------------------------
    # Site: attn
    # ------------------------------------------------------------------
    if "attn" in sites:
        print(f"\n{'#' * 62}")
        print(f"# Fusion site: attn  ({attn_site_label(layer_uf)})")
        print(f"{'#' * 62}")

        unfused_b, fused_b = _build_attn_pair(layer_uf, layer_f, variant=variant)
        site_results = run_site_benchmark(
            unfused_b, fused_b, hidden, site_label="attn", device=device
        )
        all_results["attn"] = site_results

        print(f"\n{'─' * 62}")
        print(f"Summary — attn site (layer {layer_idx}, variant={variant})")
        print_summary_table(site_results)

    # ------------------------------------------------------------------
    # Site: moe
    # ------------------------------------------------------------------
    if "moe" in sites:
        print(f"\n{'#' * 62}")
        print(f"# Fusion site: moe  (post_attention_layernorm + gate_up_proj[0] gate)")
        print(f"{'#' * 62}")

        unfused_b, fused_b = _build_moe_pair(
            layer_uf, layer_f, variant=variant, expert_idx=0
        )
        site_results = run_site_benchmark(
            unfused_b, fused_b, hidden, site_label="moe", device=device
        )
        all_results["moe"] = site_results

        print(f"\n{'─' * 62}")
        print(f"Summary — moe site (layer {layer_idx}, variant={variant})")
        print_summary_table(site_results)

    return all_results


# ---------------------------------------------------------------------------
# CSV / output helpers
# ---------------------------------------------------------------------------

_RESULTS_DIR = _THIS_DIR / "results"


def _save_all(
    all_results: dict[str, list[ShapeResult]],
    *,
    variant: str,
    layer_idx: int,
    device: str,
) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for site, results in all_results.items():
        fname = f"benchmark_{ts}_qwen3_{site}_{variant}_layer{layer_idx}.csv"
        path  = str(_RESULTS_DIR / fname)
        save_csv(
            results,
            path,
            run_metadata={
                "run_timestamp_utc": run_ts_str,
                "benchmark_mode":    "checkpoints+kernel",
                "fusion_point":      site,
                "variant":           variant,
                "load_mode":         "full",
                "device":            device,
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark fused vs unfused Qwen3.6-35B-A3B (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--unfused-dir",
        default=os.environ.get("MODEL_DIR", "/data/Qwen3.6-35B-A3B-bf16"),
        help="Vanilla BF16 checkpoint (always required)",
    )
    p.add_argument(
        "--fused-dir",
        default=os.environ.get("FUSED_DIR", "/data/Qwen3.6-35B-A3B-bf16-fused"),
        help="Weight-fused BF16 checkpoint (from export_fused_weights.py)",
    )
    p.add_argument(
        "--site",
        choices=("attn", "moe", "all"),
        default="attn",
        help="Fusion site to benchmark: attn | moe | all",
    )
    p.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Which decoder layer to extract for the benchmark",
    )
    p.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default="V2",
        help="Fused kernel variant: V1 (sequential) | V2 (stream overlap)",
    )
    p.add_argument(
        "--device",
        default=DEVICE,
        help="CUDA device for the benchmark",
    )
    p.add_argument(
        "--print-keys",
        action="store_true",
        help="Print all module paths from the unfused checkpoint index and exit",
    )
    p.add_argument(
        "--test-load",
        action="store_true",
        help=(
            "Smoke test: load + one forward pass only. "
            "Does NOT run the full latency / memory / numerical sweep."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving CSV results to phase2/results/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        sys.exit(1)

    print("=" * 62)
    print("Qwen3.6-35B-A3B  Fusion Benchmark  (Phase 2)")
    print("=" * 62)
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  CUDA      : {torch.version.cuda}")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  Site      : {args.site}")
    print(f"  Variant   : {args.variant}")
    print(f"  Layer idx : {args.layer_idx}")
    print(f"  Unfused   : {args.unfused_dir}")
    print(f"  Fused     : {args.fused_dir}")
    print(f"  Warmup    : {WARMUP_ITERS}  |  Measure: {MEASURE_ITERS}")

    if args.print_keys:
        print_model_keys(args.unfused_dir)
        sys.exit(0)

    sites = ["attn", "moe"] if args.site == "all" else [args.site]

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------
    if args.test_load:
        layer_uf = _load_decoder_layer(
            args.unfused_dir, args.layer_idx, args.device, "unfused"
        )
        layer_f = _load_decoder_layer(
            args.fused_dir, args.layer_idx, args.device, "fused"
        )

        hidden = layer_uf.hidden_size
        print(f"\nLayer {args.layer_idx}: type={layer_uf.layer_type}  hidden={hidden}")
        x = torch.randn(1, 128, hidden, device=args.device, dtype=DTYPE)

        for site in sites:
            if site == "attn":
                uf_b, f_b = _build_attn_pair(layer_uf, layer_f, variant=args.variant)
            else:
                uf_b, f_b = _build_moe_pair(layer_uf, layer_f, variant=args.variant)

            uf_b = uf_b.to(args.device, dtype=DTYPE).eval()
            f_b  = f_b.to(args.device,  dtype=DTYPE).eval()
            with torch.no_grad():
                y_uf = uf_b(x)
                y_f  = f_b(x)
            print(
                f"\n[{site}] Forward OK — "
                f"unfused={tuple(y_uf.shape)}  fused={tuple(y_f.shape)}"
            )
            diff = (y_f.float() - y_uf.float()).abs().max().item()
            print(f"[{site}] max |diff| = {diff:.6f}")

        print(
            "\n(--test-load passed. Re-run WITHOUT --test-load "
            "for the full latency / memory / numerical sweep.)"
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------
    all_results = run_benchmark(
        args.unfused_dir,
        args.fused_dir,
        sites=sites,
        layer_idx=args.layer_idx,
        variant=args.variant,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if not args.no_save:
        _save_all(
            all_results,
            variant=args.variant,
            layer_idx=args.layer_idx,
            device=args.device,
        )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
