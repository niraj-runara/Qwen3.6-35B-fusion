"""
SGLang plugin — native Qwen3.5 Site-1 fusion (Phase 5).

Install:
  pip install -e phase5/

Enable:
  export SGLANG_PLUGINS=qwen_fusion_native
  export SGLANG_FUSION=1
"""

from __future__ import annotations

import logging
import os

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def _post_load_apply_fusion(_result, model, model_config):
    if os.environ.get("SGLANG_FUSION", "0") not in ("1", "true", "yes"):
        return None

    variant = os.environ.get("FUSION_VARIANT", "V2")
    site2 = os.environ.get("SGLANG_FUSION_SITE2", "0") in ("1", "true", "yes")

    from patch_sglang_kernel_fusion import apply_sglang_kernel_fusion

    try:
        n = apply_sglang_kernel_fusion(
            model,
            variant=variant,
            site2=site2,
            check_weights=True,
        )
        logger.info("qwen_fusion_native: applied Site-1 fusion to %d layer(s)", n)
    except Exception:
        logger.exception("qwen_fusion_native: failed to apply kernel fusion")
        raise
    return None


def activate() -> None:
    """Called by SGLang ``load_plugins()`` via setuptools entry point."""
    HookRegistry.register(
        "sglang.srt.model_loader.utils.post_load_weights",
        _post_load_apply_fusion,
        HookType.AFTER,
    )
    logger.info("qwen_fusion_native plugin: registered post_load_weights hook")
