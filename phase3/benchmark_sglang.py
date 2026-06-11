#!/usr/bin/env python3
"""
Phase 3 — Benchmark vanilla Qwen3.6-35B-A3B on SGLang.

Measures full-model **prefill** latency at the same (batch, seq_len) grid as
Phase 2, writing CSV rows in the same schema as benchmark_reference.py so
Phase 4 (fused SGLang) can be compared directly.

Phase 3 fills the **nonfused_*** columns (vanilla baseline).  fused_* columns
are left empty (NaN) — speedup is NaN until Phase 4 provides a fused arm.

Backends
────────
  engine  In-process sglang.Engine (no HTTP server; default)
  http    Running server from launch_server.sh

Usage
─────
  # In-process (loads model once, ~70 GB)
  python phase3/benchmark_sglang.py \\
      --model-dir /data/Qwen3.6-35B-A3B-bf16

  # Against a running server
  python phase3/benchmark_sglang.py \\
      --backend http \\
      --server-url http://127.0.0.1:30000

  # Optional: top-1 token check vs Phase 1 oracle prompt
  python phase3/benchmark_sglang.py --check-logits
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_reference import (  # noqa: E402
    ShapeResult,
    WARMUP_ITERS,
    MEASURE_ITERS,
    print_summary_table,
    save_csv,
)

# Same shape sweep as phase2/benchmark_fused_vs_unfused.py
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

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/data/Qwen3.6-35B-A3B-bf16")
DEFAULT_ORACLE = str(_REPO_ROOT / "phase1/outputs/reference_logits.pt")
# Max Phase 2/3 grid is 32×2048; avoid 262144 default KV reservation on 96 GB GPUs.
DEFAULT_CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "65536"))
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_SAMPLING_PREFILL = {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = min(int(len(sorted_data) * pct / 100), len(sorted_data) - 1)
    return sorted_data[idx]


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_text_config(model_dir: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    return text


def _make_input_ids(tokenizer, batch: int, seq_len: int) -> list[list[int]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    row = [int(pad_id)] * seq_len
    return [row[:] for _ in range(batch)]


def _sync_ms(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Engine backend
# ---------------------------------------------------------------------------

class SGLangEngineRunner:
    def __init__(
        self,
        model_dir: str,
        *,
        mem_fraction: float = 0.90,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ):
        from sglang import Engine

        self.model_dir = model_dir
        print(f"\n[engine] Loading SGLang Engine from {model_dir} ...")
        print(f"[engine] context_length={context_length}")
        t0 = time.time()
        self.engine = Engine(
            model_path=model_dir,
            tp_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            mem_fraction_static=mem_fraction,
            context_length=context_length,
            log_level="error",
        )
        print(f"[engine] Ready in {time.time() - t0:.0f}s")

    def prefill(self, input_ids: list[list[int]]) -> None:
        self.engine.generate(
            input_ids=input_ids,
            sampling_params=_SAMPLING_PREFILL,
        )

    def prefill_reference(self) -> Any:
        tok = _load_tokenizer(self.model_dir)
        ids = tok(REFERENCE_PROMPT, return_tensors="pt")["input_ids"][0].tolist()
        return self.engine.generate(
            input_ids=[ids],
            sampling_params=_SAMPLING_PREFILL,
        )

    def shutdown(self) -> None:
        if hasattr(self.engine, "shutdown"):
            self.engine.shutdown()


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

class SGLangHttpRunner:
    def __init__(self, server_url: str):
        import urllib.request

        self.base = server_url.rstrip("/")
        self._urllib = urllib.request
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=5) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"health check failed: {resp.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach SGLang at {self.base}. "
                f"Start the server: bash phase3/launch_server.sh"
            ) from exc
        print(f"\n[http] Connected to {self.base}")

    def prefill(self, input_ids: list[list[int]]) -> None:
        import urllib.request

        payload = json.dumps(
            {"input_ids": input_ids, "sampling_params": _SAMPLING_PREFILL}
        ).encode()
        req = urllib.request.Request(
            f"{self.base}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()

    def shutdown(self) -> None:
        pass


def _http_prefill_reference(runner: SGLangHttpRunner, model_dir: str) -> Any:
    import urllib.request

    tok = _load_tokenizer(model_dir)
    ids = tok(REFERENCE_PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    payload = json.dumps(
        {"input_ids": [ids], "sampling_params": _SAMPLING_PREFILL}
    ).encode()
    req = urllib.request.Request(
        f"{runner.base}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def _measure_prefill_latency(
    runner,
    input_ids: list[list[int]],
    *,
    warmup: int,
    measure: int,
) -> list[float]:
    for _ in range(warmup):
        runner.prefill(input_ids)
    latencies: list[float] = []
    for _ in range(measure):
        latencies.append(_sync_ms(lambda: runner.prefill(input_ids)))
    return latencies


def _measure_peak_memory_mb(runner, input_ids: list[list[int]]) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    runner.prefill(input_ids)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def run_prefill_benchmark(
    runner,
    *,
    hidden: int,
    out_dim: int,
    warmup: int = WARMUP_ITERS,
    measure: int = MEASURE_ITERS,
    tokenizer=None,
    model_dir: str | None = None,
) -> list[ShapeResult]:
    results: list[ShapeResult] = []

    for batch, seq_len in _BATCH_SEQ_PAIRS:
        print(f"\n{'=' * 62}")
        print(f"[prefill]  batch={batch}  seq_len={seq_len}  hidden={hidden}")
        print(f"{'=' * 62}")

        if tokenizer is None:
            if model_dir is None:
                raise ValueError("tokenizer or model_dir required")
            tokenizer = _load_tokenizer(model_dir)

        input_ids = _make_input_ids(tokenizer, batch, seq_len)

        result = ShapeResult(
            batch=batch,
            seq_len=seq_len,
            hidden=hidden,
            out_dim=out_dim,
        )

        print("  Measuring prefill latency (vanilla SGLang) ...")
        result.nonfused_latencies = _measure_prefill_latency(
            runner, input_ids, warmup=warmup, measure=measure
        )

        print("  Measuring peak GPU memory ...")
        try:
            result.nonfused_peak_mem_mb = _measure_peak_memory_mb(runner, input_ids)
        except Exception as exc:
            print(f"  [warn] peak memory measurement failed: {exc}")
            result.nonfused_peak_mem_mb = float("nan")

        # No fused arm in Phase 3
        result.max_abs_diff = float("nan")
        result.cosine_sim = float("nan")
        result.kl_divergence = float("nan")

        med = result.nonfused_median_ms
        p99 = _percentile(result.nonfused_latencies, 99)
        tput = result.nonfused_throughput
        print(f"\n  Latency (median ms):  {med:.4f}")
        print(f"  Latency (p99 ms):     {p99:.4f}")
        print(f"  Throughput (tok/s): {tput:,.0f}")
        print(f"  Peak mem (MB):        {result.nonfused_peak_mem_mb:.1f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append(result)

    return results


def check_logits_engine(runner: SGLangEngineRunner, oracle_path: Path) -> bool:
    if not oracle_path.is_file():
        print(f"[check-logits] Oracle not found: {oracle_path}")
        return False

    oracle = torch.load(oracle_path, weights_only=True).float()
    oracle_top1 = int(oracle.argmax().item())

    out = runner.prefill_reference()
    # SGLang returns a list of dicts
    item = out[0] if isinstance(out, list) else out
    meta = item.get("meta_info", item)
    output_ids = meta.get("output_ids") or meta.get("completion_tokens") or []
    if not output_ids:
        text = item.get("text", "")
        print(f"[check-logits] Generated: {text!r}")
        print("[check-logits] Could not read output token id — verify manually.")
        return False

    gen_top1 = int(output_ids[0]) if isinstance(output_ids[0], int) else int(output_ids[0][0])
    match = gen_top1 == oracle_top1
    print(f"\n[check-logits] oracle top-1 = {oracle_top1}  |  sglang top-1 = {gen_top1}  |  match = {match}")
    return match


def check_logits_http(runner: SGLangHttpRunner, model_dir: str, oracle_path: Path) -> bool:
    if not oracle_path.is_file():
        print(f"[check-logits] Oracle not found: {oracle_path}")
        return False

    oracle = torch.load(oracle_path, weights_only=True).float()
    oracle_top1 = int(oracle.argmax().item())

    item = _http_prefill_reference(runner, model_dir)
    if isinstance(item, list):
        item = item[0]
    meta = item.get("meta_info", item)
    output_ids = meta.get("output_ids") or []
    if not output_ids:
        print(f"[check-logits] Response keys: {list(item.keys())}")
        print("[check-logits] Could not read output token id — verify manually.")
        return False

    gen_top1 = int(output_ids[0])
    match = gen_top1 == oracle_top1
    print(f"\n[check-logits] oracle top-1 = {oracle_top1}  |  sglang top-1 = {gen_top1}  |  match = {match}")
    return match


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: benchmark vanilla Qwen3.6 on SGLang (prefill sweep)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="Vanilla BF16 checkpoint path")
    p.add_argument("--backend", choices=("engine", "http"), default="engine",
                   help="engine = in-process; http = running launch_server.sh")
    p.add_argument("--server-url", default="http://127.0.0.1:30000",
                   help="SGLang base URL (--backend http)")
    p.add_argument("--mem-fraction", type=float, default=0.90,
                   help="mem_fraction_static for Engine backend")
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
                   help="Max context window (default 65536 for 32×2048 grid on 96 GB)")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    p.add_argument("--measure", type=int, default=MEASURE_ITERS)
    p.add_argument("--check-logits", action="store_true",
                   help="Compare next-token id for Phase 1 reference prompt")
    p.add_argument("--oracle", default=DEFAULT_ORACLE,
                   help="Phase 1 reference_logits.pt (default: <repo>/phase1/outputs/...)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing CSV to phase3/results/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"ERROR: model dir not found: {model_dir}")
        sys.exit(1)

    text_cfg = _load_text_config(model_dir)
    hidden = int(getattr(text_cfg, "hidden_size", 2048))
    out_dim = int(getattr(text_cfg, "vocab_size", 0)) or 0

    print("=" * 62)
    print("Phase 3 — SGLang vanilla prefill benchmark")
    print("=" * 62)
    print(f"  Model    : {model_dir}")
    print(f"  Backend  : {args.backend}")
    print(f"  Context  : {args.context_length}")
    print(f"  Hidden   : {hidden}")
    print(f"  Vocab    : {out_dim}")
    print(f"  Warmup   : {args.warmup}  |  Measure: {args.measure}")
    print(f"  Shapes   : {len(_BATCH_SEQ_PAIRS)} configs (same grid as Phase 2)")

    runner = None
    tokenizer = _load_tokenizer(model_dir)

    try:
        if args.backend == "engine":
            runner = SGLangEngineRunner(
                model_dir,
                mem_fraction=args.mem_fraction,
                context_length=args.context_length,
            )
        else:
            runner = SGLangHttpRunner(args.server_url)

        if args.check_logits:
            oracle_path = Path(args.oracle)
            if args.backend == "engine":
                ok = check_logits_engine(runner, oracle_path)
            else:
                ok = check_logits_http(runner, model_dir, oracle_path)
            if not ok:
                print("[check-logits] FAIL")
                sys.exit(1)
            print("[check-logits] PASS")

        results = run_prefill_benchmark(
            runner,
            hidden=hidden,
            out_dim=out_dim,
            warmup=args.warmup,
            measure=args.measure,
            tokenizer=tokenizer,
            model_dir=model_dir,
        )

        print(f"\n{'─' * 62}")
        print("Summary — SGLang vanilla prefill (nonfused columns)")
        print_summary_table(results)

        if not args.no_save:
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            path = str(
                _RESULTS_DIR / f"benchmark_{ts}_sglang-vanilla_prefill_na_full.csv"
            )
            save_csv(
                results,
                path,
                run_metadata={
                    "run_timestamp_utc": run_ts,
                    "benchmark_mode": "sglang-vanilla",
                    "fusion_point": "prefill",
                    "variant": "n/a",
                    "load_mode": "full",
                    "device": "cuda:0",
                },
            )
            print(f"\n[saved] {path}")

    finally:
        if runner is not None:
            runner.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
