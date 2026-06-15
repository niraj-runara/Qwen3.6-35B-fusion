#!/usr/bin/env python3
"""Phase 6 / M3 — CUDA graph capture smoke test for fused_rmsnorm_gemm."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sglang.srt.layers.fused_rmsnorm_gemm import (  # noqa: E402
    fused_rmsnorm_gemm,
    fused_rmsnorm_gemm_torch,
)
from sglang.srt.layers.triton_ops.fused_rmsnorm_gemm import compute_row_rms  # noqa: E402
from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode  # noqa: E402

HIDDEN = 2048
OUT = 8192
EPS = 1e-6
ATOL = 1e-2


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().cpu())


def test_op_cuda_graph() -> None:
    m = 128
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(OUT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    rms_scratch = torch.empty((m, 1), device="cuda", dtype=torch.float32)

    static_x = torch.empty_like(x)
    static_raw = torch.empty(m, OUT, device="cuda", dtype=torch.bfloat16)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), model_capture_mode():
        static_x.copy_(x)
        compute_row_rms(static_x, EPS, out=rms_scratch)
        fused_rmsnorm_gemm(
            static_x,
            w,
            eps=EPS,
            rms=rms_scratch,
            raw_out=static_raw,
        )
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            compute_row_rms(static_x, EPS, out=rms_scratch)
            fused_rmsnorm_gemm(
                static_x,
                w,
                eps=EPS,
                rms=rms_scratch,
                raw_out=static_raw,
            )

    x2 = torch.randn_like(x)
    with torch.cuda.stream(stream):
        static_x.copy_(x2)
        with model_capture_mode():
            graph.replay()
        stream.synchronize()

    ref = fused_rmsnorm_gemm_torch(x2, w, eps=EPS)
    diff = _max_diff(static_raw, ref.view(m, OUT))
    assert diff <= ATOL, f"cuda graph replay max_diff={diff}"
    print(f"  [ok] CUDAGraph replay  max_diff={diff:.4e}")


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    print("=" * 60)
    print("Phase 6 M3 — fused_rmsnorm_gemm CUDA graph smoke test")
    print("=" * 60)
    test_op_cuda_graph()
    print("\nAll M3 graph tests passed.")


if __name__ == "__main__":
    main()
