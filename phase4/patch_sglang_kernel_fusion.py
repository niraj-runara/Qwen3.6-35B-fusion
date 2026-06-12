"""
Apply Phase 2 Site-1 kernel fusion to a SGLang-loaded Qwen3.6 model.

Hooks into ``post_load_weights`` (scheduler subprocess) via ``sglang_fusion_plugin``.
Reuses the HF fast-path from ``phase2/patch_hf_kernel_fusion.py`` when decoder
layers are ``Qwen3_5MoeDecoderLayer`` (Transformers backend).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
_PHASE2 = _REPO / "phase2"
for _p in (_REPO, _PHASE2):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from patch_hf_kernel_fusion import patch_decoder_layer  # noqa: E402
from qwen3_moe_layers import DECODER_CLS  # noqa: E402

logger = logging.getLogger(__name__)

# Small HF metadata files SGLang needs for multimodal Qwen3.6 (not weight shards).
_VANILLA_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "configuration.json",
)


def sync_fused_config_architectures(vanilla_dir: str | Path, fused_dir: str | Path) -> bool:
    """Copy vanilla metadata into fused ckpt so SGLang loads like Phase 3 vanilla."""
    import shutil

    vanilla_dir = Path(vanilla_dir)
    fused_dir = Path(fused_dir)
    changed = False
    for fname in _VANILLA_METADATA_FILES:
        src = vanilla_dir / fname
        dst = fused_dir / fname
        if not src.is_file():
            continue
        if dst.is_file() and src.read_bytes() == dst.read_bytes():
            continue
        shutil.copy2(src, dst)
        print(f"[config] copied {fname} from vanilla -> {dst}")
        changed = True
    return changed


_LAYER_LIST_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.layers",
    "layers",
)


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    """Find the decoder ``ModuleList`` on a SGLang top-level model."""
    for path in _LAYER_LIST_PATHS:
        try:
            if hasattr(model, "get_submodule"):
                layers = model.get_submodule(path)
            else:
                cur: object = model
                for part in path.split("."):
                    cur = getattr(cur, part)
                layers = cur
            if isinstance(layers, nn.ModuleList) or (
                isinstance(layers, list) and layers and isinstance(layers[0], nn.Module)
            ):
                return layers
        except (AttributeError, ModuleNotFoundError):
            continue

    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            cls_name = type(mod[0]).__name__
            if "Decoder" in cls_name or "Moe" in cls_name:
                logger.info("Resolved decoder layers at %s (%s)", name, cls_name)
                return mod

    raise RuntimeError(
        "Could not find decoder layers on SGLang model "
        f"({type(model).__name__}). Tried: {_LAYER_LIST_PATHS}"
    )


def assert_weight_fused(layers: nn.ModuleList | list[nn.Module], *, atol: float = 0.15) -> None:
    """Sanity-check fused checkpoint: input_layernorm weights should be ~0."""
    for i, layer in enumerate(layers):
        norm = getattr(layer, "input_layernorm", None)
        if norm is None:
            continue
        w = getattr(norm, "weight", None)
        if w is None:
            continue
        peak = float(w.detach().abs().max().cpu())
        if peak > atol:
            logger.warning(
                "Layer %d input_layernorm.weight max=%.4f (expected ~0 on fused ckpt)",
                i,
                peak,
            )


def apply_sglang_kernel_fusion(
    model: nn.Module,
    *,
    variant: str = "V2",
    site2: bool = False,
    check_weights: bool = True,
) -> int:
    """Patch all decoder layers after SGLang weight load."""
    layers = resolve_decoder_layers(model)
    if check_weights:
        assert_weight_fused(layers)

    patched = 0
    for i, layer in enumerate(layers):
        if type(layer).__name__ != DECODER_CLS:
            logger.warning(
                "Skip layer %d: expected %s, got %s",
                i,
                DECODER_CLS,
                type(layer).__name__,
            )
            continue
        patch_decoder_layer(layer, i, variant=variant, site2=site2)
        patched += 1

    if patched == 0:
        raise RuntimeError(
            f"No {DECODER_CLS} layers patched — kernel fusion not applied. "
            "Model may be using a native SGLang backend without HF decoder layers."
        )

    logger.info(
        "SGLang kernel fusion: patched %d/%d layers (variant=%s, site2=%s)",
        patched,
        len(layers),
        variant,
        site2,
    )
    return patched
