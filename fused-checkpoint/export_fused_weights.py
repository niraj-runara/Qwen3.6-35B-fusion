#!/usr/bin/env python3
"""
Export weight-fused Qwen3.6-35B-A3B checkpoint.

Absorbs each RMSNorm's gamma vector into the downstream linear weights — offline,
before inference.  After this script the checkpoint is mathematically equivalent
to the original but norms are trivial (gamma = 1), enabling the runtime kernel
fusion in Phase 2/4.

Math
────
  RMSNorm:  normed = (x / rms(x)) * gamma
  Linear:   out    = W @ normed  =  (W * gamma) @ (x / rms(x))

  So: W_new = W * gamma  (row-wise broadcast over in-features)
       gamma_new = ones                  (norm becomes a pure divide-by-rms)

Fusion sites (all 64 decoder layers)
──────────────────────────────────────
  Site 1  input_layernorm          → self_attn.q_proj
                                      self_attn.k_proj
                                      self_attn.v_proj
  Site 2  post_attention_layernorm → mlp.experts[j].gate_proj   (MoE layers)
                                      mlp.experts[j].up_proj
                                   → mlp.gate_proj               (dense FFN layers)
                                      mlp.up_proj

Usage
─────
  # Default paths
  python export_fused_weights.py

  # Custom paths
  python export_fused_weights.py \\
      --src  /nvme/Qwen3.6-35B-A3B-bf16 \\
      --dst  /nvme/Qwen3.6-35B-A3B-bf16-fused

  # Run correctness check after export using Phase 1 oracle
  python export_fused_weights.py --check \\
      --oracle ../phase1/outputs/reference_logits.pt

  # Dry run — fuse in memory only, run correctness check, do NOT save
  python export_fused_weights.py --dry-run --check

Environment variables (override --src / --dst):
  MODEL_DIR     path to vanilla BF16 checkpoint
  FUSED_DIR     path for fused output checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SRC  = os.environ.get("MODEL_DIR",  "/data/Qwen3.6-35B-A3B-bf16")
DEFAULT_DST  = os.environ.get("FUSED_DIR",  "/data/Qwen3.6-35B-A3B-bf16-fused")

# Tolerance for correctness gate (atol on full logit vector vs oracle)
CORRECTNESS_ATOL = 1e-2

# Reference prompt — must match phase1/run_phase1.py REFERENCE_PROMPT exactly
REFERENCE_PROMPT = (
    "The key difference between a mixture-of-experts model and a dense model is"
)


# ---------------------------------------------------------------------------
# Weight fusion
# ---------------------------------------------------------------------------

def _absorb_norm_into_linear(
    norm: nn.Module,
    linears: list[nn.Module],
    *,
    set_gamma_to_one: bool = True,
) -> None:
    """
    Multiply each linear's weight by gamma in-place (FP32 precision).

    W_new[out, in] = W[out, in] * gamma[in]   (broadcast over out-features)

    norm.weight is set to all-ones afterwards so the norm becomes rms-only.
    """
    gamma = norm.weight.detach().float()    # [h]

    with torch.no_grad():
        for linear in linears:
            if not hasattr(linear, "weight"):
                raise AttributeError(
                    f"{type(linear).__name__} has no .weight — "
                    "cannot absorb norm gamma."
                )
            W = linear.weight.detach().float()   # [out, in]
            # Broadcast gamma over out dimension
            W_new = W * gamma.unsqueeze(0)        # [out, in]
            linear.weight.copy_(W_new.to(linear.weight.dtype))

        if set_gamma_to_one:
            norm.weight.fill_(1.0)


def fuse_decoder_layer(layer: nn.Module, layer_idx: int) -> dict[str, int]:
    """
    Fuse one decoder layer in-place.

    Returns a dict summarising what was fused:
      {'site1_linears': N, 'site2_experts': N, 'site2_dense': 1 or 0}
    """
    stats: dict[str, int] = {
        "site1_linears": 0,
        "site2_experts": 0,
        "site2_dense": 0,
    }

    # ------------------------------------------------------------------
    # Site 1 — input_layernorm → q_proj, k_proj, v_proj
    # ------------------------------------------------------------------
    input_norm = getattr(layer, "input_layernorm", None)
    self_attn  = getattr(layer, "self_attn", None)

    if input_norm is not None and self_attn is not None:
        qkv_linears = []
        for proj in ("q_proj", "k_proj", "v_proj"):
            m = getattr(self_attn, proj, None)
            if m is not None:
                qkv_linears.append(m)

        if qkv_linears:
            _absorb_norm_into_linear(input_norm, qkv_linears)
            stats["site1_linears"] = len(qkv_linears)
        else:
            print(f"  [layer {layer_idx}] WARNING: no q/k/v_proj found under self_attn")
    else:
        print(f"  [layer {layer_idx}] WARNING: missing input_layernorm or self_attn")

    # ------------------------------------------------------------------
    # Site 2 — post_attention_layernorm → MoE experts or dense MLP
    # ------------------------------------------------------------------
    post_norm = getattr(layer, "post_attention_layernorm", None)
    mlp       = getattr(layer, "mlp", None)

    if post_norm is None or mlp is None:
        print(f"  [layer {layer_idx}] WARNING: missing post_attention_layernorm or mlp")
        return stats

    # MoE path — mlp.experts is a ModuleList
    experts = getattr(mlp, "experts", None)
    if experts is not None and len(experts) > 0:
        for expert in experts:
            gate = getattr(expert, "gate_proj", None)
            up   = getattr(expert, "up_proj",   None)
            targets = [m for m in (gate, up) if m is not None]
            if targets:
                # Only set gamma to 1 on the LAST expert (avoid re-writing norm
                # 128 times; after the first expert the gamma is already 1)
                _absorb_norm_into_linear(
                    post_norm,
                    targets,
                    set_gamma_to_one=(stats["site2_experts"] == len(experts) - 1),
                )
                stats["site2_experts"] += 1
        # Ensure gamma is 1 in case the loop logic above missed the final write
        with torch.no_grad():
            post_norm.weight.fill_(1.0)
        return stats

    # Dense FFN path — mlp.gate_proj / mlp.up_proj directly
    gate = getattr(mlp, "gate_proj", None)
    up   = getattr(mlp, "up_proj",   None)
    targets = [m for m in (gate, up) if m is not None]
    if targets:
        _absorb_norm_into_linear(post_norm, targets)
        stats["site2_dense"] = 1
    else:
        print(f"  [layer {layer_idx}] WARNING: no gate/up_proj found in mlp")

    return stats


def fuse_all_layers(model: nn.Module) -> None:
    """Fuse every decoder layer in the model in-place."""
    layers = model.model.layers
    n = len(layers)
    print(f"\nFusing {n} decoder layers ...")
    t0 = time.time()

    total_site1 = 0
    total_site2_experts = 0
    total_site2_dense   = 0

    for i, layer in enumerate(layers):
        stats = fuse_decoder_layer(layer, i)
        total_site1          += stats["site1_linears"]
        total_site2_experts  += stats["site2_experts"]
        total_site2_dense    += stats["site2_dense"]

        if (i + 1) % 8 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} layers fused  ({time.time() - t0:.0f}s)")

    print(f"\nFusion summary:")
    print(f"  Site 1  (input_layernorm → q/k/v_proj) : {total_site1} linears fused")
    print(f"  Site 2  (post_attn_norm  → experts)    : {total_site2_experts} experts fused")
    print(f"  Site 2  (post_attn_norm  → dense FFN)  : {total_site2_dense} dense layers fused")


def verify_gamma_absorbed(model: nn.Module, atol: float = 1e-3) -> list[str]:
    """
    Return names of norms whose gamma is NOT ~ 1 after fusion.
    An empty list means all norms were absorbed successfully.
    """
    bad: list[str] = []
    for name, module in model.named_modules():
        if "layernorm" in name.lower() and hasattr(module, "weight"):
            w = module.weight.data
            if not torch.allclose(w, torch.ones_like(w), atol=atol, rtol=0):
                bad.append(name)
    return bad


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def run_correctness_check(
    model: nn.Module,
    tokenizer,
    oracle_logits_path: Path,
    device: torch.device,
) -> bool:
    """
    Run the reference prompt through the fused model and compare logits
    against the Phase 1 oracle.  Returns True if the check passes.
    """
    if not oracle_logits_path.exists():
        print(f"\n[correctness] Oracle file not found: {oracle_logits_path}")
        print("  Run phase1/run_phase1.py --reference first to generate it.")
        return False

    oracle = torch.load(oracle_logits_path, weights_only=True).float()  # [vocab_size]
    print(f"\n[correctness] Loaded oracle logits: {oracle.shape}")

    inputs = tokenizer(REFERENCE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    print(f"[correctness] Running fused model forward (prompt: {REFERENCE_PROMPT[:50]!r}...)")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)

    fused_logits = outputs.logits[0, -1, :].float().cpu()   # [vocab_size]

    max_diff = (fused_logits - oracle).abs().max().item()
    passed   = max_diff <= CORRECTNESS_ATOL

    # Top-1 token match
    oracle_top1 = oracle.argmax().item()
    fused_top1  = fused_logits.argmax().item()
    top1_match  = (oracle_top1 == fused_top1)

    print(f"\n[correctness] Results:")
    print(f"  max |logit diff| = {max_diff:.6f}  (threshold: {CORRECTNESS_ATOL})")
    print(f"  oracle top-1 id  = {oracle_top1}  |  fused top-1 id = {fused_top1}  |  match = {top1_match}")
    print(f"  PASS = {passed}")

    if not passed:
        print(
            "\n  FAIL: logit drift exceeds threshold. "
            "Check that gamma was applied in FP32 and the same prompt was used."
        )

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export weight-fused Qwen3.6-35B-A3B BF16 checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src",  default=DEFAULT_SRC,
                   help="Vanilla BF16 checkpoint (input)")
    p.add_argument("--dst",  default=DEFAULT_DST,
                   help="Fused BF16 checkpoint (output)")
    p.add_argument("--check", action="store_true",
                   help="Run correctness check after fusion (requires phase1 oracle)")
    p.add_argument("--oracle", default="../phase1/outputs/reference_logits.pt",
                   help="Path to phase1 reference_logits.pt for correctness check")
    p.add_argument("--dry-run", action="store_true",
                   help="Fuse in memory only — do NOT save checkpoint to disk")
    p.add_argument("--max-shard-size", default="5GB",
                   help="Max shard size when saving (passed to save_pretrained)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.is_dir():
        print(f"ERROR: Source checkpoint not found: {src_dir}")
        print("Run phase1/download_model.sh first.")
        sys.exit(1)

    print("=" * 60)
    print("Qwen3.6-35B-A3B  Weight Fusion Export")
    print("=" * 60)
    print(f"  Source : {src_dir}")
    print(f"  Dest   : {dst_dir}  ({'DRY RUN — not saved' if args.dry_run else 'will be written'})")

    # ------------------------------------------------------------------
    # Load vanilla model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)

    print(f"Loading model (BF16, device_map=auto) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------
    # Sanity check: norms should NOT be all-ones before fusion
    # ------------------------------------------------------------------
    sample_norm = model.model.layers[0].input_layernorm.weight.data
    if torch.allclose(sample_norm, torch.ones_like(sample_norm), atol=1e-3):
        print("\nWARNING: layers[0].input_layernorm.weight is already ~ 1.0")
        print("  The checkpoint may already be weight-fused. Proceeding anyway.")

    # ------------------------------------------------------------------
    # Fuse
    # ------------------------------------------------------------------
    fuse_all_layers(model)

    # ------------------------------------------------------------------
    # Verify all gammas absorbed
    # ------------------------------------------------------------------
    print("\nVerifying all norm gammas are ~ 1.0 after fusion ...")
    bad_norms = verify_gamma_absorbed(model)
    if bad_norms:
        print(f"  FAIL: {len(bad_norms)} norms still have non-unit gamma:")
        for n in bad_norms[:10]:
            print(f"    {n}")
        if len(bad_norms) > 10:
            print(f"    ... and {len(bad_norms) - 10} more")
        sys.exit(1)
    else:
        print(f"  OK: all {len(list(model.named_modules()))} modules checked")

    # ------------------------------------------------------------------
    # Correctness check (optional)
    # ------------------------------------------------------------------
    if args.check:
        oracle_path = Path(args.oracle)
        device = next(model.parameters()).device
        passed = run_correctness_check(model, tokenizer, oracle_path, device)
        if not passed:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] Skipping save.")
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving fused checkpoint to {dst_dir} ...")
        t_save = time.time()
        model.save_pretrained(
            str(dst_dir),
            max_shard_size=args.max_shard_size,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(dst_dir))

        # Copy config so SGLang / vLLM can load without transformers
        for fname in ("config.json", "generation_config.json"):
            src_f = src_dir / fname
            dst_f = dst_dir / fname
            if src_f.exists() and not dst_f.exists():
                import shutil
                shutil.copy2(src_f, dst_f)

        print(f"  Saved in {time.time() - t_save:.0f}s")
        print(f"\nFused checkpoint is ready at: {dst_dir}")
        print(
            "\nNext steps:"
            "\n  1. Verify with: python ../phase2/benchmark_fused_vs_unfused.py"
            "\n                       --unfused-dir {src_dir}"
            "\n                       --fused-dir   {dst_dir}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
