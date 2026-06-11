"""
Qwen3.5 / Qwen3.6-35B-A3B decoder layout (transformers Qwen3_5MoeDecoderLayer).

Hybrid 3:1 stack (config.layer_types):
  linear_attention → Qwen3_5MoeGatedDeltaNet  (linear_attn)
  full_attention   → Qwen3_5MoeAttention       (self_attn)

MoE FFN: Qwen3_5MoeSparseMoeBlock with Qwen3_5MoeExperts.gate_up_proj [E, 2*I, H].
"""

from __future__ import annotations

import torch
import torch.nn as nn

LAYER_LINEAR = "linear_attention"
LAYER_FULL = "full_attention"
DECODER_CLS = "Qwen3_5MoeDecoderLayer"


def validate_decoder_layer(layer: nn.Module, layer_idx: int) -> None:
    if type(layer).__name__ != DECODER_CLS:
        raise TypeError(
            f"Expected {DECODER_CLS} at model.layers[{layer_idx}], "
            f"got {type(layer).__name__}"
        )


def attn_input_linears(layer: nn.Module) -> list[nn.Linear]:
    """All token-mixer input projections fed by input_layernorm."""
    if layer.layer_type == LAYER_LINEAR:
        la = layer.linear_attn
        return [la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]
    if layer.layer_type == LAYER_FULL:
        sa = layer.self_attn
        return [sa.q_proj, sa.k_proj, sa.v_proj]
    raise ValueError(f"Unknown layer_type={layer.layer_type!r}")


def attn_input_proj(layer: nn.Module) -> nn.Linear:
    """Representative input projection for per-site benchmarks (largest linear)."""
    if layer.layer_type == LAYER_LINEAR:
        return layer.linear_attn.in_proj_qkv
    if layer.layer_type == LAYER_FULL:
        return layer.self_attn.q_proj
    raise ValueError(f"Unknown layer_type={layer.layer_type!r}")


def attn_site_label(layer: nn.Module) -> str:
    if layer.layer_type == LAYER_LINEAR:
        return "input_layernorm + linear_attn.in_proj_qkv"
    return "input_layernorm + self_attn.q_proj"


class ExpertGateProj(nn.Module):
    """
    Gate half of Qwen3_5MoeExperts.gate_up_proj for one expert.

    gate_up_proj shape: [num_experts, 2 * intermediate, hidden]
    """

    def __init__(self, gate_up_proj: torch.Tensor, expert_idx: int):
        super().__init__()
        intermediate = gate_up_proj.shape[1] // 2
        self.weight = nn.Parameter(
            gate_up_proj[expert_idx, :intermediate, :].detach().clone(),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def moe_gate_proj(layer: nn.Module, expert_idx: int = 0) -> ExpertGateProj:
    gate_up = layer.mlp.experts.gate_up_proj
    num_experts = gate_up.shape[0]
    if not 0 <= expert_idx < num_experts:
        raise IndexError(f"expert_idx={expert_idx} out of range (num_experts={num_experts})")
    return ExpertGateProj(gate_up, expert_idx)


def moe_post_norm_weights(layer: nn.Module) -> list[torch.Tensor]:
    """
    All weights that consume post_attention_layernorm output in Qwen3_5MoeSparseMoeBlock.
    """
    mlp = layer.mlp
    weights: list[torch.Tensor] = [
        mlp.experts.gate_up_proj,
        mlp.gate.weight,
        mlp.shared_expert.gate_proj.weight,
        mlp.shared_expert.up_proj.weight,
        mlp.shared_expert_gate.weight,
    ]
    return weights
