"""
Runtime fusion for HuggingFace Qwen3_5MoeDecoderLayer (weight-fused checkpoint).

E2E strategy (fast path, default):
  Site 1 — compute ``x / rms(x)`` once per layer, feed stock token-mixer linears
           (γ already absorbed offline).  Skips ``input_layernorm``.
  Site 2 — **off by default**.  Fused ckpt already has γ in MoE weights and
           ``post_attention_layernorm.weight ≈ 0``; stock norm + MoE is correct
           and faster than wrapping every MoE linear in Python.

Optional ``--site2`` enables experimental MoE runtime patch (microbench-style;
usually slower in full-model HF forward).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_bf16 import SharedRmsLinear, SharedRmsState, _norm_eps, fused_rms_normalize
from qwen3_moe_layers import (
    LAYER_FULL,
    LAYER_LINEAR,
    validate_decoder_layer,
)

_LINEAR_ATTN_PROJS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_SELF_ATTN_PROJS = ("q_proj", "k_proj", "v_proj")


# ---------------------------------------------------------------------------
# Site 1 — one rms, stock linears (default E2E path)
# ---------------------------------------------------------------------------

def _patched_decoder_forward_site1_only(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    if position_embeddings is None:
        position_embeddings = kwargs.pop("position_embeddings", None)

    residual = hidden_states
    normed = fused_rms_normalize(hidden_states, self._hf_site1_eps)

    if self.layer_type == LAYER_LINEAR:
        hidden_states = self.linear_attn(
            normed,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    elif self.layer_type == LAYER_FULL:
        hidden_states, _ = self.self_attn(
            normed,
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


# ---------------------------------------------------------------------------
# Site 2 experimental — shared-rms wrappers (opt-in only)
# ---------------------------------------------------------------------------

def _wrap_site1_projections(layer: nn.Module, state: SharedRmsState) -> None:
    if layer.layer_type == LAYER_LINEAR:
        parent = layer.linear_attn
        names = _LINEAR_ATTN_PROJS
    elif layer.layer_type == LAYER_FULL:
        parent = layer.self_attn
        names = _SELF_ATTN_PROJS
    else:
        raise ValueError(f"Unknown layer_type={layer.layer_type!r}")

    for name in names:
        linear = getattr(parent, name)
        setattr(parent, name, SharedRmsLinear(linear, state))


def _patched_router_forward(self, hidden_states: torch.Tensor):
    state: SharedRmsState = self._shared_rms_state
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = state.project_weight(hidden_states, self.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def _patched_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    state: SharedRmsState = self._shared_rms_state
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        raw = F.linear(current_state, self.gate_up_proj[expert_idx])
        state._ensure_ready()
        fused = raw / state._rms[token_idx]
        gate, up = fused.chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _patch_moe_site2(mlp: nn.Module, state: SharedRmsState) -> None:
    se = mlp.shared_expert
    se.gate_proj = SharedRmsLinear(se.gate_proj, state)
    se.up_proj = SharedRmsLinear(se.up_proj, state)
    mlp.shared_expert_gate = SharedRmsLinear(mlp.shared_expert_gate, state)
    mlp.gate._shared_rms_state = state
    mlp.gate.forward = types.MethodType(_patched_router_forward, mlp.gate)
    mlp.experts._shared_rms_state = state
    mlp.experts.forward = types.MethodType(_patched_experts_forward, mlp.experts)


def _patched_decoder_forward_site1_and_site2(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    if position_embeddings is None:
        position_embeddings = kwargs.pop("position_embeddings", None)

    residual = hidden_states
    raw = hidden_states
    self._hf_site1_state.begin(raw)

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
    raw_moe = hidden_states
    self._hf_site2_state.begin(raw_moe)
    hidden_states = self.mlp(raw_moe)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def patch_decoder_layer(
    layer: nn.Module,
    layer_idx: int,
    *,
    variant: str = "V2",
    site2: bool = False,
) -> None:
    """Patch one decoder layer for Site-1 (+ optional Site-2) kernel fusion."""
    validate_decoder_layer(layer, layer_idx)

    if site2:
        site1 = SharedRmsState(_norm_eps(layer.input_layernorm), variant=variant)
        layer._hf_site1_state = site1
        _wrap_site1_projections(layer, site1)
        site2_state = SharedRmsState(_norm_eps(layer.post_attention_layernorm), variant=variant)
        layer._hf_site2_state = site2_state
        _patch_moe_site2(layer.mlp, site2_state)
        layer.forward = types.MethodType(_patched_decoder_forward_site1_and_site2, layer)
    else:
        layer._hf_site1_eps = _norm_eps(layer.input_layernorm)
        layer.forward = types.MethodType(_patched_decoder_forward_site1_only, layer)


def apply_hf_kernel_fusion(
    model: nn.Module,
    *,
    variant: str = "V2",
    site2: bool = False,
) -> int:
    """Patch all decoder layers on a weight-fused HF model."""
    layers = model.model.layers
    for i, layer in enumerate(layers):
        patch_decoder_layer(layer, i, variant=variant, site2=site2)
    return len(layers)
