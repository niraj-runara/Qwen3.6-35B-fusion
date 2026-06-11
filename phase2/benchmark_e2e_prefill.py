#!/usr/bin/env python3
"""
Phase 2 E2E — HuggingFace full-model prefill benchmark (fused vs unfused).

Mirrors phase3/benchmark_sglang.py shape grid and CSV schema, but runs through
native transformers instead of SGLang.  Loads each checkpoint sequentially (~70 GB
each) so a single 96 GB GPU is enough.

Arms
────
  nonfused  vanilla checkpoint, stock HF forward
  fused     weight-fused checkpoint + optional Site-1 kernel patch (V2 default)

Usage
─────
  # Smoke test
  python phase2/benchmark_e2e_prefill.py \\
      --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \\
      --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \\
      --test-load

  # Full sweep (sequential load: unfused → fused)
  python phase2/benchmark_e2e_prefill.py \\
      --unfused-dir /data/Qwen3.6-35B-A3B-bf16 \\
      --fused-dir   /data/Qwen3.6-35B-A3B-bf16-fused \\
      --variant V2
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_REPO_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_reference import (  # noqa: E402
    MEASURE_ITERS,
    NUMERICAL_ITERS,
    WARMUP_ITERS,
    ShapeResult,
    print_summary_table,
    save_csv,
)
from fusion_bf16 import VARIANTS  # noqa: E402
from patch_hf_kernel_fusion import apply_hf_kernel_fusion  # noqa: E402

DTYPE = torch.bfloat16

_BATCH_SEQ_PAIRS = [
    (1, 128),
    (1, 512),
    (1, 2048),
    (8, 128),
    (8, 512),
    (8, 2048),
    (32, 128),
    (32, 512),
    (32, 2048),
]

REFERENCE_PROMPT = (
    "The key difference between a mixture-of-experts model and a dense model is"
)

DEFAULT_UNFUSED_DIR = os.environ.get("MODEL_DIR", "/data/Qwen3.6-35B-A3B-bf16")
DEFAULT_FUSED_DIR = os.environ.get("FUSED_DIR", "/data/Qwen3.6-35B-A3B-bf16-fused")
DEFAULT_ORACLE = str(_REPO_ROOT / "phase1/outputs/reference_logits.pt")
_RESULTS_DIR = _THIS_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_gpu_mem(label: str) -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{label}] GPU {i}: {alloc:.2f} / {total:.2f} GB allocated")


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_text_config(model_dir: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def _model_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _make_inputs(
    tokenizer,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    input_ids = torch.full((batch, seq_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _sync_ms(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _prefill_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


def _last_token_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = _prefill_forward(model, input_ids, attention_mask)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[:, -1, :].float().cpu()


def _measure_prefill_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmup: int,
    measure: int,
) -> list[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _prefill_forward(model, input_ids, attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies: list[float] = []
        for _ in range(measure):
            latencies.append(
                _sync_ms(lambda: _prefill_forward(model, input_ids, attention_mask))
            )
    return latencies


def _measure_peak_memory_mb(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    dev = input_ids.device
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        _prefill_forward(model, input_ids, attention_mask)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(dev) / 1024**2


def _compare_logits(
    fused_logits: torch.Tensor,
    nonfused_logits: torch.Tensor,
    *,
    n_iters: int = NUMERICAL_ITERS,
) -> tuple[float, float, float]:
    """Compare last-position logits (same inputs, already on CPU)."""
    max_diffs, cosines, kls = [], [], []
    for _ in range(n_iters):
        out_f = fused_logits
        out_nf = nonfused_logits
        max_diffs.append((out_f - out_nf).abs().max().item())
        cos = F.cosine_similarity(out_f, out_nf, dim=-1).mean().item()
        cosines.append(cos)
        p = F.softmax(out_f, dim=-1).clamp(min=1e-10)
        q = F.softmax(out_nf, dim=-1).clamp(min=1e-10)
        kl = (p * (p / q).log()).sum(dim=-1).mean().item()
        kls.append(kl)
    return (
        statistics.mean(max_diffs),
        statistics.mean(cosines),
        statistics.mean(kls),
    )


def _load_model(
    model_dir: str,
    label: str,
    *,
    attn_implementation: str = "sdpa",
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")
    print(f"\nLoading {label} from {model_dir} (attn={attn_implementation}) ...")
    t0 = time.time()
    # Pin to a single GPU so sequential unfused→fused loads reuse the same
    # ~65 GB pool.  device_map="auto" can CPU-offload when it sees cached
    # memory from the previous checkpoint as "used".
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=DTYPE,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    off_device = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if off_device:
        raise RuntimeError(
            f"{label}: {len(off_device)} parameters not on CUDA "
            f"(e.g. {off_device[0]}). Free GPU memory before loading the next checkpoint."
        )
    print(f"  {label} loaded in {time.time() - t0:.0f}s")
    _print_gpu_mem(label)
    return model


def _release_model(model: nn.Module) -> None:
    """Strip accelerate hooks so the caller's ``del model`` frees GPU memory."""
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is not None:
            try:
                hook.detach_hook(module)
            except Exception:
                pass
            try:
                del module._hf_hook
            except AttributeError:
                pass
        if hasattr(module, "_old_forward"):
            del module._old_forward
    if hasattr(model, "hf_device_map"):
        model.hf_device_map = None


def _cuda_gc(label: str = "after free", *, quiet: bool = False) -> None:
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if not quiet:
        _print_gpu_mem(label)


@contextmanager
def _loaded_model(model_dir: str, label: str, *, attn_implementation: str = "sdpa"):
    """Load one full checkpoint; guaranteed GPU release on exit."""
    model = _load_model(model_dir, label, attn_implementation=attn_implementation)
    try:
        yield model
    finally:
        _release_model(model)
        del model
        _cuda_gc(label)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _benchmark_one_shape(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmup: int,
    measure: int,
    batch: int,
    seq_len: int,
) -> tuple[list[float], float, torch.Tensor | None]:
    """
    Run latency + peak-mem + logits for one shape.

    Returns (latencies, peak_mem_mb, last_token_logits).  On OOM, returns
    empty latencies, NaN peak mem, and None logits (caller records skip).
    """
    try:
        _cuda_gc(quiet=True)
        print("  Measuring prefill latency ...")
        latencies = _measure_prefill_latency(
            model, input_ids, attention_mask, warmup=warmup, measure=measure
        )

        print("  Measuring peak GPU memory ...")
        try:
            peak_mem = _measure_peak_memory_mb(model, input_ids, attention_mask)
        except Exception as exc:
            if _is_oom(exc):
                raise
            print(f"  [warn] peak memory failed: {exc}")
            peak_mem = float("nan")

        print("  Caching last-token logits for numerical compare ...")
        logits = _last_token_logits(model, input_ids, attention_mask)
        return latencies, peak_mem, logits

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_oom(exc):
            raise
        print(
            f"  [OOM] Skipping batch={batch} seq_len={seq_len} "
            f"(activation memory exceeds free GPU after ~65 GB weights)."
        )
        print(f"         {exc}")
        _cuda_gc("after OOM")
        return [], float("nan"), None


# ---------------------------------------------------------------------------
# Benchmark arms
# ---------------------------------------------------------------------------

def _run_unfused_arm(
    model_dir: str,
    tokenizer,
    *,
    attn_implementation: str,
    warmup: int,
    measure: int,
    shapes: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], float], dict[tuple[int, int], torch.Tensor]]:
    latencies: dict[tuple[int, int], list[float]] = {}
    peak_mem: dict[tuple[int, int], float] = {}
    logits_cache: dict[tuple[int, int], torch.Tensor] = {}

    with _loaded_model(model_dir, "unfused", attn_implementation=attn_implementation) as model:
        device = _model_device(model)
        for batch, seq_len in shapes:
            print(f"\n{'=' * 62}")
            print(f"[unfused]  batch={batch}  seq_len={seq_len}")
            print(f"{'=' * 62}")

            input_ids, attention_mask = _make_inputs(tokenizer, batch, seq_len, device)
            key = (batch, seq_len)

            lat, peak, logits = _benchmark_one_shape(
                model,
                input_ids,
                attention_mask,
                warmup=warmup,
                measure=measure,
                batch=batch,
                seq_len=seq_len,
            )
            latencies[key] = lat
            peak_mem[key] = peak
            if logits is not None:
                logits_cache[key] = logits

            if lat:
                print(f"  Latency (median ms): {statistics.median(lat):.4f}")
            else:
                print("  Latency (median ms): skipped (OOM)")
            print(f"  Peak mem (MB):       {peak_mem[key]:.1f}")

            del input_ids, attention_mask
            _cuda_gc(quiet=True)

    return latencies, peak_mem, logits_cache


def _run_fused_arm(
    model_dir: str,
    tokenizer,
    *,
    attn_implementation: str,
    variant: str,
    use_kernel: bool,
    warmup: int,
    measure: int,
    shapes: list[tuple[int, int]],
    unfused_logits: dict[tuple[int, int], torch.Tensor],
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], float], dict[tuple[int, int], tuple[float, float, float]]]:
    latencies: dict[tuple[int, int], list[float]] = {}
    peak_mem: dict[tuple[int, int], float] = {}
    numerical: dict[tuple[int, int], tuple[float, float, float]] = {}

    with _loaded_model(model_dir, "fused", attn_implementation=attn_implementation) as model:
        if use_kernel:
            n = apply_hf_kernel_fusion(model, variant=variant)
            print(f"  Applied Site-1 kernel fusion to {n} decoder layers (variant={variant})")
        else:
            print("  Kernel patch skipped (--no-kernel); weight-fused ckpt only")

        device = _model_device(model)
        for batch, seq_len in shapes:
            print(f"\n{'=' * 62}")
            print(f"[fused]  batch={batch}  seq_len={seq_len}")
            print(f"{'=' * 62}")

            input_ids, attention_mask = _make_inputs(tokenizer, batch, seq_len, device)
            key = (batch, seq_len)

            lat, peak, fused_logits = _benchmark_one_shape(
                model,
                input_ids,
                attention_mask,
                warmup=warmup,
                measure=measure,
                batch=batch,
                seq_len=seq_len,
            )
            latencies[key] = lat
            peak_mem[key] = peak

            if fused_logits is not None and key in unfused_logits:
                numerical[key] = _compare_logits(fused_logits, unfused_logits[key])
            else:
                numerical[key] = (float("nan"), float("nan"), float("nan"))

            if lat:
                print(f"  Latency (median ms): {statistics.median(lat):.4f}")
            else:
                print("  Latency (median ms): skipped (OOM)")
            print(f"  Peak mem (MB):       {peak_mem[key]:.1f}")
            if key in numerical and fused_logits is not None:
                md, cs, kl = numerical[key]
                print(f"  max|diff|: {md:.4f}  cosine: {cs:.6f}  KL: {kl:.2e}")

            del input_ids, attention_mask, fused_logits
            _cuda_gc(quiet=True)

    return latencies, peak_mem, numerical


def _merge_results(
    *,
    hidden: int,
    out_dim: int,
    unfused_lat: dict[tuple[int, int], list[float]],
    unfused_mem: dict[tuple[int, int], float],
    fused_lat: dict[tuple[int, int], list[float]],
    fused_mem: dict[tuple[int, int], float],
    numerical: dict[tuple[int, int], tuple[float, float, float]],
    shapes: list[tuple[int, int]],
) -> list[ShapeResult]:
    results: list[ShapeResult] = []
    for batch, seq_len in shapes:
        key = (batch, seq_len)
        r = ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=out_dim)
        r.nonfused_latencies = unfused_lat.get(key, [])
        r.fused_latencies = fused_lat.get(key, [])
        r.nonfused_peak_mem_mb = unfused_mem.get(key, float("nan"))
        r.fused_peak_mem_mb = fused_mem.get(key, float("nan"))
        if key in numerical:
            r.max_abs_diff, r.cosine_sim, r.kl_divergence = numerical[key]
        results.append(r)
    return results


def _parse_shapes(specs: list[str] | None) -> list[tuple[int, int]]:
    if not specs:
        return list(_BATCH_SEQ_PAIRS)
    out: list[tuple[int, int]] = []
    for spec in specs:
        parts = spec.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid --shapes entry {spec!r}; use BATCH,SEQ_LEN e.g. 32,2048")
        out.append((int(parts[0]), int(parts[1])))
    return out


def _smoke_test(
    unfused_dir: str,
    fused_dir: str,
    *,
    attn_implementation: str,
    variant: str,
    use_kernel: bool,
) -> None:
    tokenizer = _load_tokenizer(unfused_dir)
    batch, seq_len = 1, 128

    def _run_unfused() -> torch.Tensor:
        print("\n[test-load] Unfused forward ...")
        with _loaded_model(unfused_dir, "unfused", attn_implementation=attn_implementation) as model:
            dev = _model_device(model)
            ids, mask = _make_inputs(tokenizer, batch, seq_len, dev)
            with torch.no_grad():
                _prefill_forward(model, ids, mask)
            return _last_token_logits(model, ids, mask)

    def _run_fused(nf_logits: torch.Tensor) -> None:
        print("\n[test-load] Fused forward ...")
        with _loaded_model(fused_dir, "fused", attn_implementation=attn_implementation) as model:
            if use_kernel:
                apply_hf_kernel_fusion(model, variant=variant)
            dev = _model_device(model)
            ids, mask = _make_inputs(tokenizer, batch, seq_len, dev)
            with torch.no_grad():
                _prefill_forward(model, ids, mask)
            f_logits = _last_token_logits(model, ids, mask)
            md, cs, kl = _compare_logits(f_logits, nf_logits, n_iters=1)
            print(f"[test-load] max|diff|={md:.4f}  cosine={cs:.6f}  KL={kl:.2e}")

    nf_logits = _run_unfused()
    _run_fused(nf_logits)
    print("[test-load] PASS")


def check_logits_oracle(model_dir: str, oracle_path: Path) -> bool:
    if not oracle_path.is_file():
        print(f"[check-logits] Oracle not found: {oracle_path}")
        return False

    oracle = torch.load(oracle_path, weights_only=True).float()
    oracle_top1 = int(oracle.argmax().item())

    tokenizer = _load_tokenizer(model_dir)
    # Phase 1 oracle was captured with eager attention — match that here.
    with _loaded_model(model_dir, "oracle-check", attn_implementation="eager") as model:
        dev = _model_device(model)
        ids = tokenizer(REFERENCE_PROMPT, return_tensors="pt")["input_ids"].to(dev)
        mask = torch.ones_like(ids)
        logits = _last_token_logits(model, ids, mask)
        gen_top1 = int(logits[0].argmax().item())

    match = gen_top1 == oracle_top1
    print(
        f"\n[check-logits] oracle top-1 = {oracle_top1}  |  "
        f"hf top-1 = {gen_top1}  |  match = {match}"
    )
    return match


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 E2E: HF full-model prefill (fused vs unfused)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--unfused-dir", default=DEFAULT_UNFUSED_DIR)
    p.add_argument("--fused-dir", default=DEFAULT_FUSED_DIR)
    p.add_argument("--variant", choices=list(VARIANTS), default="V2")
    p.add_argument(
        "--no-kernel",
        action="store_true",
        help="Fused arm: weight-fused ckpt only (no Site-1 runtime kernel patch)",
    )
    p.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help=(
            "Attention backend for the benchmark sweep (default sdpa). "
            "eager OOMs on large shapes with ~65 GB weights on a 96 GB GPU."
        ),
    )
    p.add_argument(
        "--shapes",
        nargs="+",
        metavar="B,S",
        help="Subset of shapes to run, e.g. --shapes 32,2048 (default: all 9)",
    )
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    p.add_argument("--measure", type=int, default=MEASURE_ITERS)
    p.add_argument("--test-load", action="store_true", help="Smoke test only")
    p.add_argument("--check-logits", action="store_true",
                   help="Compare next-token id vs Phase 1 oracle (unfused ckpt)")
    p.add_argument("--oracle", default=DEFAULT_ORACLE,
                   help="Phase 1 reference_logits.pt (default: <repo>/phase1/outputs/...)")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        sys.exit(1)

    for d, name in ((args.unfused_dir, "unfused"), (args.fused_dir, "fused")):
        if not os.path.isdir(d):
            print(f"ERROR: {name} checkpoint not found: {d}")
            sys.exit(1)

    text_cfg = _load_text_config(args.unfused_dir)
    hidden = int(getattr(text_cfg, "hidden_size", 2048))
    out_dim = int(getattr(text_cfg, "vocab_size", 0)) or 0
    use_kernel = not args.no_kernel
    shapes = _parse_shapes(args.shapes)

    print("=" * 62)
    print("Phase 2 E2E — HF full-model prefill benchmark")
    print("=" * 62)
    print(f"  Unfused  : {args.unfused_dir}")
    print(f"  Fused    : {args.fused_dir}")
    print(f"  Kernel   : {'Site-1 ' + args.variant if use_kernel else 'disabled (weights only)'}")
    print(f"  Attn     : {args.attn_implementation}")
    print(f"  Hidden   : {hidden}")
    print(f"  Vocab    : {out_dim}")
    print(f"  Warmup   : {args.warmup}  |  Measure: {args.measure}")
    print(f"  Shapes   : {len(shapes)} configs")

    if args.test_load:
        _smoke_test(
            args.unfused_dir,
            args.fused_dir,
            attn_implementation=args.attn_implementation,
            variant=args.variant,
            use_kernel=use_kernel,
        )
        return

    if args.check_logits:
        ok = check_logits_oracle(args.unfused_dir, Path(args.oracle))
        if not ok:
            print("[check-logits] FAIL")
            sys.exit(1)
        print("[check-logits] PASS")

    tokenizer = _load_tokenizer(args.unfused_dir)

    print(f"\n{'─' * 62}")
    print("Arm 1/2 — unfused checkpoint (sequential load)")
    print(f"{'─' * 62}")
    nf_lat, nf_mem, nf_logits = _run_unfused_arm(
        args.unfused_dir,
        tokenizer,
        attn_implementation=args.attn_implementation,
        warmup=args.warmup,
        measure=args.measure,
        shapes=shapes,
    )

    print(f"\n{'─' * 62}")
    print("Arm 2/2 — fused checkpoint (sequential load)")
    print(f"{'─' * 62}")
    f_lat, f_mem, numerical = _run_fused_arm(
        args.fused_dir,
        tokenizer,
        attn_implementation=args.attn_implementation,
        variant=args.variant,
        use_kernel=use_kernel,
        warmup=args.warmup,
        measure=args.measure,
        shapes=shapes,
        unfused_logits=nf_logits,
    )

    results = _merge_results(
        hidden=hidden,
        out_dim=out_dim,
        unfused_lat=nf_lat,
        unfused_mem=nf_mem,
        fused_lat=f_lat,
        fused_mem=f_mem,
        numerical=numerical,
        shapes=shapes,
    )

    print(f"\n{'─' * 62}")
    print("Summary — HF E2E prefill (fused vs unfused)")
    print_summary_table(results)

    if not args.no_save:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        kernel_tag = args.variant if use_kernel else "weights-only"
        path = str(
            _RESULTS_DIR / f"benchmark_{ts}_hf-e2e_prefill_{kernel_tag}_full.csv"
        )
        save_csv(
            results,
            path,
            run_metadata={
                "run_timestamp_utc": run_ts,
                "benchmark_mode": "hf-e2e",
                "fusion_point": "prefill",
                "variant": kernel_tag,
                "load_mode": "full",
                "device": "cuda:0",
            },
        )
        print(f"\n[saved] {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
