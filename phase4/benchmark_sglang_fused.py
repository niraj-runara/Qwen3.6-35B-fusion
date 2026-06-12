#!/usr/bin/env python3
"""
Phase 4 — SGLang full-model prefill with weight-fused checkpoint + Site-1 kernel fusion.

Compares against Phase 3 vanilla baseline (nonfused columns) from a CSV, fills
fused_* columns and speedup in the same schema as Phase 2 E2E.

Prerequisites:
  bash phase4/setup_fusion_plugin.sh
  export SGLANG_PLUGINS=qwen_fusion
  export SGLANG_FUSION=1

Usage:
  python phase4/benchmark_sglang_fused.py \\
      --fused-dir /data/Qwen3.6-35B-A3B-bf16-fused \\
      --baseline-csv phase3/results/benchmark_*_sglang-vanilla_prefill_na_full.csv \\
      --check-logits
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import site
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

# libnvrtc.so.13 — same as phase3
_sp = site.getsitepackages()[0]
_cuda_ld = ":".join(
    os.path.join(_sp, "nvidia", sub, "lib")
    for sub in ("cuda_nvrtc", "cuda_runtime", "cu13")
    if os.path.isdir(os.path.join(_sp, "nvidia", sub, "lib"))
)
if _cuda_ld:
    _old = os.environ.get("LD_LIBRARY_PATH", "")
    if "cuda_nvrtc" not in _old:
        os.environ["LD_LIBRARY_PATH"] = f"{_cuda_ld}:{_old}" if _old else _cuda_ld

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHASE3 = _REPO_ROOT / "phase3"
for _p in (_REPO_ROOT, _PHASE3):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from benchmark_reference import (  # noqa: E402
    MEASURE_ITERS,
    WARMUP_ITERS,
    ShapeResult,
    print_summary_table,
    save_csv,
)
import benchmark_sglang as sg3  # noqa: E402

DEFAULT_FUSED_DIR = os.environ.get("FUSED_DIR", "/data/Qwen3.6-35B-A3B-bf16-fused")
DEFAULT_ORACLE = str(_REPO_ROOT / "phase1/outputs/reference_logits.pt")
DEFAULT_BASELINE_GLOB = str(_REPO_ROOT / "phase3/results/benchmark_*_sglang-vanilla_prefill_na_full.csv")
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _latest_baseline_csv(pattern: str) -> Path | None:
    matches = sorted(glob(pattern), key=os.path.getmtime)
    return Path(matches[-1]) if matches else None


def _load_baseline_rows(csv_path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """Map (batch, seq_len) -> nonfused metrics from Phase 3 CSV."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["batch"]), int(row["seq_len"]))
            out[key] = {
                "nonfused_median_ms": float(row["nonfused_median_ms"]),
                "nonfused_p99_ms": float(row["nonfused_p99_ms"]),
                "nonfused_throughput": float(row["nonfused_throughput"]),
                "nonfused_peak_mem_mb": float(row.get("nonfused_peak_mem_mb") or "nan"),
            }
    return out


def _synthetic_latencies(median: float, p99: float, n: int = 200) -> list[float]:
    """Approximate latency list so median/p99 match Phase 3 CSV aggregates."""
    k = max(1, n // 100)
    return [median] * (n - k) + [p99] * k


def _merge_baseline(results: list[ShapeResult], baseline: dict[tuple[int, int], dict]) -> None:
    for r in results:
        row = baseline.get((r.batch, r.seq_len))
        if not row:
            print(f"  [warn] no Phase 3 baseline for batch={r.batch} seq={r.seq_len}")
            continue
        r.nonfused_latencies = _synthetic_latencies(
            row["nonfused_median_ms"], row["nonfused_p99_ms"]
        )
        r.nonfused_peak_mem_mb = row["nonfused_peak_mem_mb"]


def run_fused_prefill_benchmark(
    runner,
    *,
    hidden: int,
    out_dim: int,
    warmup: int,
    measure: int,
    tokenizer,
    model_dir: str,
) -> list[ShapeResult]:
    results: list[ShapeResult] = []

    for batch, seq_len in sg3._BATCH_SEQ_PAIRS:
        print(f"\n{'=' * 62}")
        print(f"[prefill]  batch={batch}  seq_len={seq_len}  hidden={hidden}")
        print(f"{'=' * 62}")

        input_ids = sg3._make_input_ids(tokenizer, batch, seq_len)
        result = ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=out_dim)

        print("  Measuring prefill latency (fused SGLang) ...")
        result.fused_latencies = sg3._measure_prefill_latency(
            runner, input_ids, warmup=warmup, measure=measure
        )

        print("  Measuring peak GPU memory ...")
        try:
            result.fused_peak_mem_mb = sg3._measure_peak_memory_mb(runner, input_ids)
        except Exception as exc:
            print(f"  [warn] peak memory measurement failed: {exc}")
            result.fused_peak_mem_mb = float("nan")

        result.max_abs_diff = float("nan")
        result.cosine_sim = float("nan")
        result.kl_divergence = float("nan")

        med = result.fused_median_ms
        p99 = sg3._percentile(result.fused_latencies, 99)
        tput = result.fused_throughput
        print(f"\n  Latency (median ms):  {med:.4f}")
        print(f"  Latency (p99 ms):     {p99:.4f}")
        print(f"  Throughput (tok/s): {tput:,.0f}")
        print(f"  Peak mem (MB):        {result.fused_peak_mem_mb:.1f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append(result)

    return results


def _configure_fusion_env(*, enable_kernel: bool) -> None:
    os.environ["SGLANG_PLUGINS"] = "qwen_fusion"
    os.environ["SGLANG_FUSION"] = "1" if enable_kernel else "0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4: SGLang fused prefill benchmark vs Phase 3 baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fused-dir", default=DEFAULT_FUSED_DIR)
    p.add_argument(
        "--baseline-csv",
        default="",
        help="Phase 3 vanilla CSV (default: latest under phase3/results/)",
    )
    p.add_argument("--backend", choices=("engine", "http"), default="engine")
    p.add_argument("--server-url", default="http://127.0.0.1:30000")
    p.add_argument("--mem-fraction", type=float, default=0.90)
    p.add_argument("--context-length", type=int, default=sg3.DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--variant", default="V2", help="Fusion variant (env FUSION_VARIANT)")
    p.add_argument(
        "--no-kernel",
        action="store_true",
        help="Load fused weights only; skip Site-1 runtime patch",
    )
    p.add_argument(
        "--site2",
        action="store_true",
        help="Experimental Site-2 MoE patch (sets SGLANG_FUSION_SITE2=1)",
    )
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    p.add_argument("--measure", type=int, default=MEASURE_ITERS)
    p.add_argument("--check-logits", action="store_true")
    p.add_argument("--oracle", default=DEFAULT_ORACLE)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.fused_dir):
        print(f"ERROR: fused checkpoint not found: {args.fused_dir}")
        sys.exit(1)

    baseline_path = Path(args.baseline_csv) if args.baseline_csv else _latest_baseline_csv(DEFAULT_BASELINE_GLOB)
    if baseline_path is None or not baseline_path.is_file():
        print("ERROR: Phase 3 baseline CSV not found. Run phase3 benchmark first or pass --baseline-csv.")
        sys.exit(1)

    use_kernel = not args.no_kernel
    _configure_fusion_env(enable_kernel=use_kernel)
    os.environ["FUSION_VARIANT"] = args.variant
    os.environ["SGLANG_FUSION_SITE2"] = "1" if args.site2 else "0"

    text_cfg = sg3._load_text_config(args.fused_dir)
    hidden = int(getattr(text_cfg, "hidden_size", 2048))
    out_dim = int(getattr(text_cfg, "vocab_size", 0)) or 0

    kernel_desc = "disabled (fused weights only)"
    if use_kernel:
        kernel_desc = (
            f"Site-1+2 experimental ({args.variant})"
            if args.site2
            else f"Site-1 fast-path ({args.variant})"
        )

    print("=" * 62)
    print("Phase 4 — SGLang fused prefill benchmark")
    print("=" * 62)
    print(f"  Fused ckpt : {args.fused_dir}")
    print(f"  Baseline   : {baseline_path}")
    print(f"  Backend    : {args.backend}")
    print(f"  Kernel     : {kernel_desc}")
    print(f"  Plugins    : {os.environ.get('SGLANG_PLUGINS')}")
    print(f"  Context    : {args.context_length}")
    print(f"  Warmup     : {args.warmup}  |  Measure: {args.measure}")

    baseline = _load_baseline_rows(baseline_path)
    runner = None
    tokenizer = sg3._load_tokenizer(args.fused_dir)

    try:
        if args.backend == "engine":
            runner = sg3.SGLangEngineRunner(
                args.fused_dir,
                mem_fraction=args.mem_fraction,
                context_length=args.context_length,
            )
        else:
            runner = sg3.SGLangHttpRunner(args.server_url)

        if args.check_logits:
            ok = sg3.check_logits_engine(runner, Path(args.oracle))
            if not ok:
                print("[check-logits] FAIL")
                sys.exit(1)
            print("[check-logits] PASS")

        results = run_fused_prefill_benchmark(
            runner,
            hidden=hidden,
            out_dim=out_dim,
            warmup=args.warmup,
            measure=args.measure,
            tokenizer=tokenizer,
            model_dir=args.fused_dir,
        )
        _merge_baseline(results, baseline)

        print(f"\n{'─' * 62}")
        print("Summary — SGLang fused vs Phase 3 vanilla (speedup = nf_med / fused_med)")
        print_summary_table(results)

        if not args.no_save:
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            tag = "V2" if use_kernel else "weights-only"
            path = str(_RESULTS_DIR / f"benchmark_{ts}_sglang-fused_prefill_{tag}_full.csv")
            save_csv(
                results,
                path,
                run_metadata={
                    "run_timestamp_utc": run_ts,
                    "benchmark_mode": "sglang-fused",
                    "fusion_point": "prefill",
                    "variant": args.variant if use_kernel else "weights-only",
                    "load_mode": "full",
                    "device": "cuda:0",
                    "baseline_csv": str(baseline_path),
                },
            )
            print(f"\n[saved] {path}")

    finally:
        if runner is not None:
            runner.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
