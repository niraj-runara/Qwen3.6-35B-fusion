"""
Site-1 fusion for SGLang *native* Qwen3.5 decoder layers.

Skips ``input_layernorm`` in ``LayerCommunicator.prepare_attn`` and applies
``linear(x) / rms(x)`` on input projections (fused weights offline).

Supported layer types (SGLang 0.5.x ``qwen3_5.py``):
  - ``Qwen3_5LinearDecoderLayer``  (linear / GatedDeltaNet attn)
  - ``Qwen3_5AttentionDecoderLayer`` (full self-attn)

Falls back to stock ``prepare_attn`` on TP scatter, quant, or allreduce-fusion paths.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
_PHASE2 = _REPO / "phase2"
for _p in (_REPO, _PHASE2):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)

NATIVE_LINEAR_LAYER = "Qwen3_5LinearDecoderLayer"
NATIVE_ATTENTION_LAYER = "Qwen3_5AttentionDecoderLayer"
NATIVE_LAYER_CLASSES = frozenset({NATIVE_LINEAR_LAYER, NATIVE_ATTENTION_LAYER})


def _norm_eps(norm: nn.Module) -> float:
    return float(
        getattr(norm, "variance_epsilon", None)
        or getattr(norm, "eps", None)
        or 1e-6
    )


def _needs_stock_prepare_attn(
    comm: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    quant_format: str,
) -> bool:
    if quant_format:
        return True
    try:
        from sglang.srt.layers.dp_attention import get_attn_tp_context

        if get_attn_tp_context().input_scattered:
            return True
    except Exception:
        return True
    if (
        residual is not None
        and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")
        and hidden_states._sglang_needs_allreduce_fusion
    ):
        return True
    return False


def _patch_prepare_attn(comm: nn.Module, *, variant: str) -> None:
    if getattr(comm, "_qwen_fusion_prepare_attn_patched", False):
        return

    eps = _norm_eps(comm.input_layernorm)
    orig = comm.prepare_attn

    def fused_prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: Any,
        quant_format: str = "",
        post_residual_addition: torch.Tensor | None = None,
    ):
        if _needs_stock_prepare_attn(self, hidden_states, residual, quant_format):
            self._site1_state = None
            return orig(
                hidden_states,
                residual,
                forward_batch,
                quant_format,
                post_residual_addition,
            )

        from fusion_bf16 import Site1RmsState

        if residual is None:
            residual = hidden_states
        if post_residual_addition is not None:
            residual = residual + post_residual_addition

        self._site1_state = Site1RmsState(eps, variant=variant)
        self._site1_state.begin(hidden_states)

        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        if self.qkv_latent_func is not None:
            from sglang.srt.layers.communicator import AttentionInputs
            from sglang.srt.layers.dp_attention import get_attn_tp_context

            attn_inputs = AttentionInputs(
                hidden_states, forward_batch, self.qkv_latent_func
            )
            get_attn_tp_context().set_attn_inputs(attn_inputs)
        return hidden_states, residual

    comm.prepare_attn = types.MethodType(fused_prepare_attn, comm)
    comm._qwen_fusion_prepare_attn_patched = True
    comm._qwen_fusion_prepare_attn_orig = orig


def _is_cpu_or_npu() -> bool:
    try:
        from sglang.srt.utils import is_cpu, is_npu

        return bool(is_cpu() or is_npu())
    except ImportError:
        return False


def _patch_linear_attn_input_proj(linear_attn: nn.Module, comm: nn.Module) -> None:
    if getattr(linear_attn, "_qwen_fusion_input_proj_patched", False):
        return

    orig = linear_attn._forward_input_proj

    def fused_forward_input_proj(self, hidden_states: torch.Tensor):
        state = getattr(comm, "_site1_state", None)
        if state is None:
            return orig(hidden_states)

        try:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            from sglang.srt.server_args import get_global_server_args
        except ImportError:
            return orig(hidden_states)

        if _is_cpu_or_npu() or not get_global_server_args().disable_piecewise_cuda_graph:
            threshold = 0
        else:
            threshold = 1024

        seq_len, _ = hidden_states.shape
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < threshold
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz = state.project(hidden_states, self.in_proj_qkvz)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba = state.project(hidden_states, self.in_proj_ba)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz = state.project(hidden_states, self.in_proj_qkvz)
            projected_states_ba = state.project(hidden_states, self.in_proj_ba)
        return projected_states_qkvz, projected_states_ba

    linear_attn._forward_input_proj = types.MethodType(
        fused_forward_input_proj, linear_attn
    )
    linear_attn._qwen_fusion_input_proj_patched = True
    linear_attn._qwen_fusion_input_proj_orig = orig


def _patch_attention_self_attention(layer: nn.Module) -> None:
    if getattr(layer, "_qwen_fusion_self_attn_patched", False):
        return

    orig = layer.self_attention

    def fused_self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        state = getattr(self.layer_communicator, "_site1_state", None)
        if state is None:
            return orig(positions, hidden_states, forward_batch)

        qkv = state.project(hidden_states, self.qkv_proj)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    layer.self_attention = types.MethodType(fused_self_attention, layer)
    layer._qwen_fusion_self_attn_patched = True
    layer._qwen_fusion_self_attn_orig = orig


def patch_native_decoder_layer(layer: nn.Module, *, variant: str = "V2") -> None:
    """Apply Site-1 fusion hooks to one native SGLang Qwen3.5 decoder layer."""
    cls = type(layer).__name__
    comm = layer.layer_communicator
    _patch_prepare_attn(comm, variant=variant)

    if cls == NATIVE_LINEAR_LAYER:
        _patch_linear_attn_input_proj(layer.linear_attn, comm)
    elif cls == NATIVE_ATTENTION_LAYER:
        _patch_attention_self_attention(layer)
    else:
        raise TypeError(f"Unsupported native layer type: {cls}")

    layer._qwen_fusion_native_patched = True
