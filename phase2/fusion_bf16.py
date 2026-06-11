"""
BF16 Fused RMSNorm + Linear modules for Qwen3.6-35B-A3B.

Mirrors kimi-fused-nvfp4-vllm/fusion/nvfp4_fused_rmsnorm_linear.py from the
Kimi-K2.6 repo, but targets plain BF16 nn.Linear layers instead of NVFP4.

Math (gamma already absorbed into weights offline):
    unfused:  normed = (x / rms(x)) * gamma     <- separate RMSNorm module
              out    = W_orig @ normed
    fused:    out    = W_new  @ (x / rms(x))    <- W_new = W_orig * gamma
              equivalently: out = linear_new(x) / rms(x)

Classes
───────
  FusedRMSNormLinear      V1 — sequential: matmul then divide
  FusedRMSNormLinearV2    V2 — CUDA stream overlap: matmul and rms computed
                               concurrently on separate streams (Blackwell / H100)

Both classes take a plain nn.Linear (or any module whose forward returns a tensor)
and wrap it with the rms division.  No NVFP4 / compressed-tensors dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_forward(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call any linear layer; normalise return type to a plain tensor."""
    out = linear(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _rms(x_2d: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute per-token RMS of x_2d: shape [T, h] → [T, 1].
    Computed in FP32, cast back to input dtype.
    """
    return torch.sqrt(
        x_2d.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps
    ).to(x_2d.dtype)


# ---------------------------------------------------------------------------
# V1 — sequential (safe on all GPUs)
# ---------------------------------------------------------------------------

class FusedRMSNormLinear(nn.Module):
    """
    V1: linear(x) / rms(x)  (sequential, rms computed after matmul).

    Use when:
    - Correctness testing / debugging
    - GPUs without multi-stream overlap (older than Ampere)
    """

    def __init__(self, linear: nn.Module, h: int, eps: float):
        """
        Args:
            linear: The weight-fused nn.Linear (W_new = W * gamma).
            h:      Input hidden dimension (for documentation; not used at runtime).
            eps:    RMSNorm epsilon from the original norm module.
        """
        super().__init__()
        self.linear = linear
        self.h      = h
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d  = x.reshape(-1, x.size(-1))          # [T, h]
        raw   = _linear_forward(self.linear, x_2d)  # [T, d_out]
        out   = raw / _rms(x_2d, self.eps)          # [T, d_out]
        return out.reshape(orig_shape[:-1] + (out.size(-1),))


# ---------------------------------------------------------------------------
# V2 — CUDA stream overlap (Ampere / Hopper / Blackwell)
# ---------------------------------------------------------------------------

class FusedRMSNormLinearV2(nn.Module):
    """
    V2: rms(x) computed on a side CUDA stream while the main stream runs the
    matmul.  On compute-heavy GPUs (H100, B200, RTX Pro 6000) this overlaps
    the cheap rms reduction with the expensive matmul, hiding the rms cost.

    Use when:
    - Production benchmarking / deployment
    - GPU supports concurrent kernel execution (sm80+)
    """

    def __init__(self, linear: nn.Module, h: int, eps: float):
        super().__init__()
        self.linear = linear
        self.h      = h
        self.eps    = eps
        self._side_stream = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))

        # Main stream: matmul
        raw = _linear_forward(self.linear, x_2d)

        # Side stream: rms (overlaps with matmul on the main stream)
        with torch.cuda.stream(self._side_stream):
            rms = _rms(x_2d, self.eps)

        # Sync side stream back before the division
        torch.cuda.current_stream().wait_stream(self._side_stream)

        out = raw / rms
        return out.reshape(orig_shape[:-1] + (out.size(-1),))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

VARIANTS: dict[str, type] = {
    "V1": FusedRMSNormLinear,
    "V2": FusedRMSNormLinearV2,
}


def build_fused_module(
    linear: nn.Module,
    norm: nn.Module,
    *,
    variant: str = "V2",
) -> nn.Module:
    """
    Build a FusedRMSNormLinear from a weight-fused linear and the original
    norm (used only to read h and eps).

    The returned module replaces the (norm → linear) pair.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}; got {variant!r}")

    # Infer h from the linear's weight shape (in-features)
    if hasattr(linear, "weight"):
        h = linear.weight.shape[1]
    else:
        raise AttributeError(f"{type(linear).__name__} has no .weight")

    # Read eps from the norm
    eps = (
        getattr(norm, "variance_epsilon", None)
        or getattr(norm, "eps", None)
        or 1e-6
    )

    cls = VARIANTS[variant]
    return cls(linear, h=h, eps=float(eps))
