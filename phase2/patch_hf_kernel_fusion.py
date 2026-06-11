"""
Runtime kernel fusion for HuggingFace Qwen3_5MoeDecoderLayer (fused checkpoint).

Replaces Site 1 (input_layernorm → token-mixer input projections) with
FusedRMSNormLinear wrappers and skips the separate input_layernorm in the
decoder forward.  Site 2 still uses post_attention_layernorm (weight ≈ 0 on
fused ckpt) + stock MoE forward — MoE kernel fusion remains the Phase 2
microbenchmark until a full mlp patch lands in Phase 4.
"""

from __future__ import annotations

import types

import torch.nn as nn

from fusion_bf16 import build_fused_module
from qwen3_moe_layers import (
    LAYER_FULL,
    LAYER_LINEAR,
    attn_input_linears,
    validate_decoder_layer,
)


def _patch_linear_attn_site1(linear_attn: nn.Module, fused_projs: nn.ModuleList) -> None:
    """Swap in_proj_* modules for FusedRMSNormLinear (same call signature)."""
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    for name, fused in zip(names, fused_projs):
        setattr(linear_attn, name, fused)


def _patch_self_attn_site1(self_attn: nn.Module, fused_projs: nn.ModuleList) -> None:
    """Swap q/k/v_proj for FusedRMSNormLinear wrappers."""
    self_attn.q_proj = fused_projs[0]
    self_attn.k_proj = fused_projs[1]
    self_attn.v_proj = fused_projs[2]


def _patched_decoder_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    """Decoder forward that feeds raw hidden states into the patched token mixer."""
    if position_embeddings is None:
        position_embeddings = kwargs.pop("position_embeddings", None)

    residual = hidden_states
    raw = hidden_states

    if self.layer_type == LAYER_LINEAR:
        hidden_states = self.linear_attn(
            raw,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    elif self.layer_type == LAYER_FULL:
        hidden_states, _ = self.self_attn(
            raw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown layer_type={self.layer_type!r}")

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def patch_decoder_layer(layer: nn.Module, layer_idx: int, *, variant: str = "V2") -> None:
    """Patch one decoder layer in-place for Site-1 kernel fusion."""
    validate_decoder_layer(layer, layer_idx)

    fused_projs = nn.ModuleList(
        build_fused_module(proj, layer.input_layernorm, variant=variant)
        for proj in attn_input_linears(layer)
    )
    layer.register_module("_hf_fusion_site1", fused_projs)

    if layer.layer_type == LAYER_LINEAR:
        _patch_linear_attn_site1(layer.linear_attn, fused_projs)
    elif layer.layer_type == LAYER_FULL:
        _patch_self_attn_site1(layer.self_attn, fused_projs)
    else:
        raise ValueError(f"Unknown layer_type={layer.layer_type!r}")

    layer.forward = types.MethodType(_patched_decoder_forward, layer)


def apply_hf_kernel_fusion(model: nn.Module, *, variant: str = "V2") -> int:
    """
    Patch all decoder layers on a weight-fused HF model.

    Returns the number of layers patched.
    """
    layers = model.model.layers
    for i, layer in enumerate(layers):
        patch_decoder_layer(layer, i, variant=variant)
    return len(layers)
