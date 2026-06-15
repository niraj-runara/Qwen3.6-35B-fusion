#!/usr/bin/env python3
"""Phase 6 / M1 — unit tests for ``fused_rmsnorm_gemm`` vs Phase 2 ``Site1RmsState``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
_PHASE2 = _REPO / "phase2"
for _p in (_REPO, _PHASE2):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fusion_bf16 import Site1RmsState  # noqa: E402
from sglang.srt.layers.fused_rmsnorm_gemm import (  # noqa: E402
    RmsNormGemmState,
    fused_rmsnorm_gemm,
    fused_rmsnorm_gemm_pair,
    fused_rmsnorm_gemm_torch,
)

DEFAULT_FUSED_DIR = Path("/data/Qwen3.6-35B-A3B-bf16-fused")
HIDDEN = 2048
EPS = 1e-6
ATOL_BF16 = 1e-2


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().cpu())


def _site1_project_weight(
    state: Site1RmsState, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    state.begin(x)
    return state.project_weight(x, weight)


def test_random_shapes(device: torch.device) -> None:
    cases = [
        (1, 128, HIDDEN, 4096),
        (1, 512, HIDDEN, 12288),
        (8, 512, HIDDEN, 4096),
        (32, 128, HIDDEN, 2048),
    ]
    for batch, seq, h, out_dim in cases:
        x = torch.randn(batch, seq, h, device=device, dtype=torch.bfloat16)
        w = torch.randn(out_dim, h, device=device, dtype=torch.bfloat16)
        ref_state = Site1RmsState(EPS, variant="V1")
        ref = _site1_project_weight(ref_state, x, w)
        got = fused_rmsnorm_gemm(x, w, eps=EPS)
        diff = _max_diff(ref, got)
        assert diff <= ATOL_BF16, (
            f"random shape batch={batch} seq={seq} h={h} out={out_dim}: "
            f"max_diff={diff:.4e} > {ATOL_BF16}"
        )
        print(f"  [ok] random ({batch},{seq}) @ [{out_dim},{h}]  max_diff={diff:.4e}")


def test_shared_rms_pair(device: torch.device) -> None:
    x = torch.randn(4, 256, HIDDEN, device=device, dtype=torch.bfloat16)
    wa = torch.randn(8192, HIDDEN, device=device, dtype=torch.bfloat16)
    wb = torch.randn(64, HIDDEN, device=device, dtype=torch.bfloat16)

    ref_state = Site1RmsState(EPS, variant="V1")
    ref_state.begin(x)
    ref_a = ref_state.project_weight(x, wa)
    ref_b = ref_state.project_weight(x, wb)

    sgl_state = RmsNormGemmState(EPS)
    got_a, got_b = fused_rmsnorm_gemm_pair(x, wa, wb, eps=EPS, rms_state=sgl_state)

    da = _max_diff(ref_a, got_a)
    db = _max_diff(ref_b, got_b)
    assert da <= ATOL_BF16 and db <= ATOL_BF16, f"pair diff a={da} b={db}"
    print(f"  [ok] shared-rms pair  max_diff=({da:.4e}, {db:.4e})")


def test_torch_ref_matches_triton(device: torch.device) -> None:
    x = torch.randn(2, 64, HIDDEN, device=device, dtype=torch.bfloat16)
    w = torch.randn(6000, HIDDEN, device=device, dtype=torch.bfloat16)
    ref = fused_rmsnorm_gemm_torch(x, w, eps=EPS)
    got = fused_rmsnorm_gemm(x, w, eps=EPS)
    diff = _max_diff(ref, got)
    assert diff <= 1e-3, f"torch vs triton max_diff={diff}"
    print(f"  [ok] torch vs triton  max_diff={diff:.4e}")


def _load_checkpoint_weight(
    fused_dir: Path, tensor_name: str
) -> torch.Tensor | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    index_path = fused_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    import json

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map.get(tensor_name)
    if shard is None:
        return None
    shard_path = fused_dir / shard
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def test_fused_ckpt_weight(
    device: torch.device, fused_dir: Path
) -> None:
    # Linear-attn layer 0 — GDN qkv projection (export keeps HF shard names)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    w = _load_checkpoint_weight(fused_dir, name)
    if w is None:
        print(f"  [skip] could not load {name}")
        return

    w = w.to(device=device, dtype=torch.bfloat16)
    x = torch.randn(1, 512, w.shape[1], device=device, dtype=torch.bfloat16)

    ref_state = Site1RmsState(EPS, variant="V1")
    ref = _site1_project_weight(ref_state, x, w)
    got = fused_rmsnorm_gemm(x, w, eps=EPS)
    diff = _max_diff(ref, got)
    assert diff <= ATOL_BF16, f"ckpt weight max_diff={diff}"
    print(f"  [ok] fused ckpt in_proj_qkv  shape={tuple(w.shape)}  max_diff={diff:.4e}")


def main() -> None:
    p = argparse.ArgumentParser(description="M1 fused_rmsnorm_gemm unit tests")
    p.add_argument("--fused-dir", type=Path, default=DEFAULT_FUSED_DIR)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    device = torch.device("cuda")
    print("=" * 60)
    print("Phase 6 M1 — fused_rmsnorm_gemm vs Site1RmsState")
    print("=" * 60)
    print(f"  device : {device}")
    print(f"  atol   : {ATOL_BF16} (bf16)")
    print()

    test_random_shapes(device)
    test_shared_rms_pair(device)
    test_torch_ref_matches_triton(device)
    test_fused_ckpt_weight(device, args.fused_dir)

    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
