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


def _norm_eps(norm: nn.Module) -> float:
    return float(
        getattr(norm, "variance_epsilon", None)
        or getattr(norm, "eps", None)
        or 1e-6
    )


def fused_rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    ``x / rms(x)`` for weight-fused checkpoints (γ absorbed offline).

    Used by the E2E HF patch: one rms per layer, then stock nn.Linear modules.
    """
    shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))
    return (x_2d / _rms(x_2d, eps)).reshape(shape)


# ---------------------------------------------------------------------------
# Shared per-layer RMS (E2E / full decoder patch)
# ---------------------------------------------------------------------------

class Site1RmsState:
    """
    One rms(x) per decoder layer, shared by all Site-1 input projections.

    V2 kicks rms off on a side stream in ``begin()`` so the first matmul can
    overlap with the reduction; ``project()`` waits once before the first divide.
    """

    def __init__(self, eps: float, *, variant: str = "V2"):
        self.eps = eps
        self.variant = variant
        self._side_stream = (
            torch.cuda.Stream()
            if variant == "V2" and torch.cuda.is_available()
            else None
        )
        self._rms: torch.Tensor | None = None
        self._ready = False

    def begin(self, x: torch.Tensor) -> None:
        """Start (or compute) shared rms for this layer forward."""
        self._ready = False
        x_2d = x.reshape(-1, x.size(-1))
        if self._side_stream is not None:
            with torch.cuda.stream(self._side_stream):
                self._rms = _rms(x_2d, self.eps)
        else:
            self._rms = _rms(x_2d, self.eps)

    def _ensure_ready(self) -> None:
        if not self._ready:
            if self._side_stream is not None:
                torch.cuda.current_stream().wait_stream(self._side_stream)
            self._ready = True

    def project(self, x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
        """fused-weight linear(x) / rms(x) using the layer-shared rms."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        raw = _linear_forward(linear, x_2d)
        self._ensure_ready()
        out = raw / self._rms
        return out.reshape(orig_shape[:-1] + (out.size(-1),))

    def project_weight(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """F.linear(x, weight) / rms(x) for nn.Parameter routers / expert stacks."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        raw = torch.nn.functional.linear(x_2d, weight)
        self._ensure_ready()
        out = raw / self._rms
        return out.reshape(orig_shape[:-1] + (out.size(-1),))


# Site 1 and Site 2 use the same shared-rms helper.
SharedRmsState = Site1RmsState


class SharedRmsLinear(nn.Module):
    """Wrap one fused-weight linear; divides by the layer's shared rms state."""

    def __init__(self, linear: nn.Module, state: Site1RmsState):
        super().__init__()
        self.linear = linear
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.state.project(x, self.linear)


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
