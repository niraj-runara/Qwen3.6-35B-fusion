"""
Benchmark: RMSNorm + Linear Fusion vs Non-Fused (NVFP4 Quantized)
PyTorch + CUDA

Usage:
    python benchmark_rmsnorm_linear_fusion.py --dir /path/to/dir [--layer-path model.layers.0.mlp]

    Expects the following layout under --dir:
        <dir>/models/fused/       — fused HuggingFace checkpoint
        <dir>/models/non-fused/   — non-fused HuggingFace checkpoint

    --benchmark-mode runtime-patch (default): load models/non-fused, baseline =
      norm→linear; fused = kimi_patch FusedRMSNormLinear* (actual kernel speedup).

    --benchmark-mode checkpoints: models/non-fused vs models/fused (offline γ
      in weights; linear-only, no CUDA fusion kernels).

    --fusion-point q_b | kv_b.  --variant V1|V2|V3 for runtime-patch.

Requirements:
    pip install torch transformers  # CUDA build
    pip install safetensors         # if checkpoints use .safetensors format
    pip install compressed-tensors  # NVFP4 checkpoints

    flash-attn is NOT required; models load with attn_implementation=eager.
"""

import argparse
import copy
import gc
import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo root on path for kimi_patch / ops / src
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
# Kimi NVFP4 checkpoints use BF16 activations / scales; FP16 causes matmul dtype errors.
DTYPE  = torch.bfloat16

# Shape sweep: (batch_size, seq_len, hidden_dim, out_dim)
SHAPE_SWEEP: List[Tuple[int, int, int, int]] = [
    (1,   128,  4096, 4096),
    (1,   512,  4096, 4096),
    (1,  2048,  4096, 4096),
    (8,   128,  4096, 4096),
    (8,   512,  4096, 4096),
    (8,  2048,  4096, 4096),
    (32,  128,  4096, 4096),
    (32,  512,  4096, 4096),
    (32, 2048,  4096, 4096),
]

WARMUP_ITERS   = 50
MEASURE_ITERS  = 200
NUMERICAL_ITERS = 10   # iterations for output collection in equivalence check

# MLA fusion sites inside self_attn (module attribute names on DeepseekV3Attention)
FUSION_POINTS = {
    "q_b": ("q_a_layernorm", "q_b_proj", "q_lora_rank"),
    "kv_b": ("kv_a_layernorm", "kv_b_proj", "kv_lora_rank"),
}


# ---------------------------------------------------------------------------
# Fusion benchmark modules (offline fused weights in checkpoint)
# ---------------------------------------------------------------------------

class _UnfusedNormLinear(nn.Module):
    """Non-fused checkpoint: RMSNorm then NVFP4 linear."""

    def __init__(self, norm: nn.Module, linear: nn.Module):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class _FusedLinearOnly(nn.Module):
    """Fused checkpoint: linear only (γ already absorbed into packed weights)."""

    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _fusion_input_dim(text_config, fusion_point: str) -> int:
    _, _, dim_attr = FUSION_POINTS[fusion_point]
    return int(getattr(text_config, dim_attr))


def _module_device(module: nn.Module) -> torch.device:
    """Device of first parameter or registered buffer (FusedRMSNormLinear uses buffers)."""
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    raise RuntimeError(f"{type(module).__name__} has no parameters or buffers")


def _resolve_self_attn(module: nn.Module) -> nn.Module:
    """Fusion helpers need DeepseekV3Attention submodules, not the wrapper."""
    if isinstance(module, _SelfAttnBenchmarkWrapper):
        return module.attn
    return module


def _build_unfused_bench(attn: nn.Module, fusion_point: str) -> nn.Module:
    attn = _resolve_self_attn(attn)
    norm_name, linear_name, _ = FUSION_POINTS[fusion_point]
    return _UnfusedNormLinear(getattr(attn, norm_name), getattr(attn, linear_name))


def _build_runtime_patch_fused(
    attn: nn.Module,
    fusion_point: str,
    device: str,
    variant: str,
) -> nn.Module:
    from kimi_patch import patch_kimi_fusion_point

    attn = _resolve_self_attn(attn)
    attn_patched = copy.deepcopy(attn)
    fused_mod = patch_kimi_fusion_point(
        attn_patched, fusion_point, device=device, variant=variant,
    )
    return fused_mod


def _build_checkpoints_pair(
    attn_nonfused: nn.Module,
    attn_fused: nn.Module,
    fusion_point: str,
) -> Tuple[nn.Module, nn.Module]:
    attn_nonfused = _resolve_self_attn(attn_nonfused)
    attn_fused = _resolve_self_attn(attn_fused)
    norm_name, linear_name, _ = FUSION_POINTS[fusion_point]
    return (
        _UnfusedNormLinear(
            getattr(attn_nonfused, norm_name),
            getattr(attn_nonfused, linear_name),
        ),
        _FusedLinearOnly(getattr(attn_fused, linear_name)),
    )


# ---------------------------------------------------------------------------
# Model loading (HuggingFace full checkpoint)
# ---------------------------------------------------------------------------

def _get_nested_attr(obj, dot_path: str):
    """
    Navigate a dot-separated attribute path, supporting integer indices.
    e.g. 'model.layers.0.mlp' → obj.model.layers[0].mlp
    """
    for part in dot_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _infer_hidden_dim(module: nn.Module) -> int:
    """
    Infer hidden_dim from the first 2-D weight in the module that looks like
    an input projection (i.e. weight.shape[-1] is the hidden dim).
    """
    for name, param in module.named_parameters():
        if param.ndim == 2 and "norm" not in name:
            hidden_dim = param.shape[-1]
            print(f"  Inferred hidden_dim={hidden_dim} from '{name}' {tuple(param.shape)}")
            return hidden_dim
    raise ValueError(
        "Could not infer hidden_dim from module parameters. "
        "Check --layer-path points to the right sub-module."
    )


def _parse_layer_idx(layer_path: str) -> int:
    parts = layer_path.split(".")
    if "layers" not in parts:
        raise ValueError(
            f"--layer-path must contain 'layers.<idx>' (got {layer_path!r})"
        )
    return int(parts[parts.index("layers") + 1])


def _gpu_max_memory(num_gpus: int | None = None, reserve_frac: float = 0.08) -> dict:
    """
    Per-GPU memory budget for accelerate device_map='auto'.
    Uses actual GPU total memory minus a small reserve (default 8%).
    Override with BENCHMARK_GPU_MEM=90GiB or BENCHMARK_CPU_MEM=128GiB.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if os.environ.get("BENCHMARK_GPU_MEM"):
        per_gpu = os.environ["BENCHMARK_GPU_MEM"]
        mem = {i: per_gpu for i in range(num_gpus)}
    else:
        mem = {}
        for i in range(num_gpus):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            usable_gib = max(1, int(total_gib * (1.0 - reserve_frac)))
            mem[i] = f"{usable_gib}GiB"
    mem["cpu"] = os.environ.get("BENCHMARK_CPU_MEM", "128GiB")
    return mem


def _print_model_device_map(model: nn.Module, label: str) -> None:
    """Summarize hf_device_map or parameter device distribution."""
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        counts: dict[str, int] = {}
        for dev in hf_map.values():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        print(f"  [{label}] hf_device_map ({len(hf_map)} modules): {dict(sorted(counts.items()))}")
        return
    counts = {}
    for p in model.parameters():
        d = str(p.device)
        counts[d] = counts.get(d, 0) + 1
    print(f"  [{label}] parameter devices: {dict(sorted(counts.items()))}")


def _extract_layer_from_full_model(
    full_model: nn.Module,
    layer_path: str,
    device: str,
) -> nn.Module:
    """
    Clone layer_path into a standalone module on `device`.

    deepcopy preserves NVFP4 CompressedLinear wrappers and weights. Deleting the
    parent model while holding only a child reference breaks accelerate hooks.
    """
    import copy

    layer = _get_nested_attr(full_model, layer_path)
    standalone = copy.deepcopy(layer)
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    standalone = standalone.to(device=device, dtype=DTYPE)
    # Return raw DeepseekV3Attention: fusion bench reads q_a_layernorm / q_b_proj directly.
    return standalone


class _SelfAttnBenchmarkWrapper(nn.Module):
    """Run DeepseekV3Attention with synthetic mask/position_ids (benchmark API is model(x))."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        # Causal 4D mask expected by DeepseekV3Attention
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
            diagonal=1,
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)
        out, _, _ = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out


def _maybe_wrap_for_benchmark(module: nn.Module) -> nn.Module:
    if module.__class__.__name__ == "DeepseekV3Attention":
        return _SelfAttnBenchmarkWrapper(module)
    return module


def _patch_config_eager_attn(config) -> None:
    """
    Kimi-K2.5 stores flash_attention_2 on vision_config (and text_config) in
    config.json. Top-level attn_implementation= does not override those nested
    fields, so patch them before from_pretrained().
    """
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    for sub in ("vision_config", "text_config"):
        sub_cfg = getattr(config, sub, None)
        if sub_cfg is not None and hasattr(sub_cfg, "_attn_implementation"):
            sub_cfg._attn_implementation = "eager"


def _load_layer_gpu(
    model_dir: str,
    layer_path: str,
    label: str,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Load only the submodule at layer_path onto GPU.

    Skips full-model 'Compressing model' (~6k modules) and ~500GB disk read;
    only quantizes and loads tensors for the target layer (~10–20 keys).
    """
    import json

    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.quantizers.auto import AutoHfQuantizer

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    t0 = time.perf_counter()
    print(f"\nLoading {label} (layer-only) from {model_dir} ...")
    print(f"  layer-path: {layer_path}  device: {device}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    _patch_config_eager_attn(config)

    layer_idx = _parse_layer_idx(layer_path)
    attn_cls = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV3Attention",
        model_dir,
        trust_remote_code=True,
    )
    module = attn_cls(config.text_config, layer_idx=layer_idx)
    module = module.to(device=device, dtype=DTYPE)

    quantizer = AutoHfQuantizer.from_config(config.quantization_config)
    quantizer._process_model_before_weight_loading(module)
    print(f"  NVFP4 wrappers ready ({time.perf_counter() - t0:.1f}s)")

    prefix = layer_path + "."
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    keys = [k for k in weight_map if k.startswith(prefix)]
    if not keys:
        raise ValueError(f"No weights under prefix {prefix!r} in {index_path}")

    shards = {weight_map[k] for k in keys}
    state = {}
    for shard in shards:
        tensors = load_file(os.path.join(model_dir, shard), device=device)
        for k in keys:
            if k in tensors:
                state[k[len(prefix):]] = tensors[k]

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    quantizer._process_model_after_weight_loading(module)
    module.eval()
    print(
        f"  Loaded {len(state)} tensors from {len(shards)} shard(s) "
        f"({time.perf_counter() - t0:.1f}s total)"
    )
    return module


def _load_hf_model_full(model_dir: str, label: str, num_gpus: int | None = None) -> nn.Module:
    """Load full NVFP4 checkpoint sharded across GPUs via device_map='auto'."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    n_gpu = num_gpus or torch.cuda.device_count()
    if n_gpu < 1:
        raise RuntimeError("full load requires at least one CUDA GPU")

    max_memory = _gpu_max_memory(n_gpu)
    t0 = time.perf_counter()
    print(f"\nLoading {label} (full model, {n_gpu} GPUs) from {model_dir} ...")
    print(f"  dtype: {DTYPE}  max_memory: {max_memory}")
    print(
        "  Expect: ~4 min 'Compressing model' (~6k modules) + disk read for all shards."
    )

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    _patch_config_eager_attn(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        dtype=DTYPE,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    _print_model_device_map(model, label)
    print(f"  Full model loaded in {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    return model


def print_model_keys(model_dir: str) -> None:
    """
    List module paths from the checkpoint index (no weight load).

    NVFP4 checkpoints use compressed-tensors; a full from_pretrained() would
    run compress_model() over thousands of layers and load ~500GB+ of weights
    just to call named_modules().  Reading model.safetensors.index.json is enough
    to discover --layer-path values.
    """
    import json

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"No {index_path}. Cannot list keys without loading the full model."
        )

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    module_paths = sorted({k.rsplit(".", 1)[0] for k in weight_map})

    print(
        f"\nParameter module paths from {index_path} ({len(module_paths)} total):"
    )
    print("(No weights loaded — derived from checkpoint tensor names.)\n")

    # Show a compact prefix tree first, then sample leaf paths
    prefixes: set[str] = set()
    for path in module_paths:
        parts = path.split(".")
        for depth in (3, 4, 5, 6):
            if len(parts) >= depth:
                prefixes.add(".".join(parts[:depth]))
    print("Common prefixes (use as --layer-path starting points):")
    for p in sorted(prefixes)[:40]:
        print(f"  {p}")
    print("\nSample leaf paths (first 40):")
    for path in module_paths[:40]:
        print(f"  {path}")
    if len(module_paths) > 40:
        print(f"  ... ({len(module_paths) - 40} more; grep the index for a prefix)")


def load_models(
    base_dir: str,
    layer_path: str,
    *,
    benchmark_mode: str = "runtime-patch",
    fusion_point: str = "q_b",
    variant: str = "V2",
    load_mode: str = "layer",
    device: str = "cuda:0",
    num_gpus: int | None = None,
) -> Tuple[nn.Module, nn.Module, int]:
    """
    Build (fused_bench, nonfused_bench, input_dim) for one fusion site.

    runtime-patch: non-fused checkpoint + kimi_patch CUDA modules (recommended).
    checkpoints:   compare models/non-fused vs models/fused on disk.
    """
    from transformers import AutoConfig

    nonfused_dir = os.path.join(base_dir, "models", "non-fused")
    fused_dir = os.path.join(base_dir, "models", "fused")

    config = AutoConfig.from_pretrained(nonfused_dir, trust_remote_code=True)
    hidden_dim = _fusion_input_dim(config.text_config, fusion_point)
    print(f"  mode={benchmark_mode}  fusion_point={fusion_point}  input_dim={hidden_dim}")

    if benchmark_mode == "runtime-patch":
        print(f"  variant={variant}  (kimi_patch on non-fused weights)")
        if load_mode == "layer":
            attn = _load_layer_gpu(nonfused_dir, layer_path, "non-fused", device)
        elif load_mode == "full":
            full = _load_hf_model_full(nonfused_dir, "non-fused", num_gpus)
            attn = _extract_layer_from_full_model(full, layer_path, device)
        else:
            raise ValueError(f"Unknown load_mode: {load_mode!r}")

        nonfused_bench = _build_unfused_bench(attn, fusion_point)
        fused_bench = _build_runtime_patch_fused(attn, fusion_point, device, variant)
        del attn
        gc.collect()
        torch.cuda.empty_cache()

    elif benchmark_mode == "checkpoints":
        print("  (offline fused weights in models/fused/, no kimi_patch)")
        if load_mode == "layer":
            attn_nf = _load_layer_gpu(nonfused_dir, layer_path, "non-fused", device)
            attn_f = _load_layer_gpu(fused_dir, layer_path, "fused", device)
        elif load_mode == "full":
            full_nf = _load_hf_model_full(nonfused_dir, "non-fused", num_gpus)
            attn_nf = _extract_layer_from_full_model(full_nf, layer_path, device)
            full_f = _load_hf_model_full(fused_dir, "fused", num_gpus)
            attn_f = _extract_layer_from_full_model(full_f, layer_path, device)
        else:
            raise ValueError(f"Unknown load_mode: {load_mode!r}")
        nonfused_bench, fused_bench = _build_checkpoints_pair(
            attn_nf, attn_f, fusion_point,
        )
    else:
        raise ValueError(
            f"Unknown benchmark_mode {benchmark_mode!r}. "
            "Choose runtime-patch or checkpoints."
        )

    print(f"  Non-fused bench: {type(nonfused_bench).__name__}")
    print(f"  Fused bench:     {type(fused_bench).__name__}")
    return fused_bench, nonfused_bench, hidden_dim


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@dataclass
class ShapeResult:
    batch:   int
    seq_len: int
    hidden:  int
    out_dim: int

    # Latency (ms)
    fused_latencies:    List[float] = field(default_factory=list)
    nonfused_latencies: List[float] = field(default_factory=list)

    # Memory (MB)
    fused_peak_mem_mb:    float = 0.0
    nonfused_peak_mem_mb: float = 0.0

    # Numerical equivalence
    max_abs_diff:   float = 0.0
    cosine_sim:     float = 0.0
    kl_divergence:  float = 0.0

    @property
    def fused_median_ms(self) -> float:
        return statistics.median(self.fused_latencies) if self.fused_latencies else float("nan")

    @property
    def nonfused_median_ms(self) -> float:
        return statistics.median(self.nonfused_latencies) if self.nonfused_latencies else float("nan")

    @property
    def fused_p99_ms(self) -> float:
        return _percentile(self.fused_latencies, 99)

    @property
    def nonfused_p99_ms(self) -> float:
        return _percentile(self.nonfused_latencies, 99)

    @property
    def speedup(self) -> float:
        if self.fused_median_ms == 0:
            return float("nan")
        return self.nonfused_median_ms / self.fused_median_ms

    @property
    def fused_throughput(self) -> float:
        """tokens / second"""
        tokens = self.batch * self.seq_len
        return tokens / (self.fused_median_ms / 1000.0)

    @property
    def nonfused_throughput(self) -> float:
        tokens = self.batch * self.seq_len
        return tokens / (self.nonfused_median_ms / 1000.0)


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _sync_time(fn, *args) -> float:
    """Run fn(*args), synchronize CUDA, return wall-clock seconds."""
    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    measure: int = MEASURE_ITERS,
) -> List[float]:
    """Returns list of per-forward-pass latencies in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(measure):
            t = _sync_time(model, x)
            latencies.append(t * 1000.0)
    return latencies


def measure_peak_memory(model: nn.Module, x: torch.Tensor) -> float:
    """Returns peak GPU memory allocated during a forward pass, in MB."""
    model.eval()
    dev = x.device
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(dev) / 1024 ** 2


def measure_numerical_equivalence(
    fused: nn.Module,
    nonfused: nn.Module,
    x: torch.Tensor,
    n_iters: int = NUMERICAL_ITERS,
) -> Tuple[float, float, float]:
    """
    Compare fused vs non-fused outputs.

    Returns:
        max_abs_diff  — max absolute elementwise difference
        cosine_sim    — mean cosine similarity across batch
        kl_divergence — mean KL divergence of softmax distributions
    """
    fused.eval()
    nonfused.eval()

    max_diffs, cosines, kls = [], [], []

    with torch.no_grad():
        for _ in range(n_iters):
            out_f  = fused(x).float()
            out_nf = nonfused(x).float()

            # Max absolute difference
            max_diffs.append((out_f - out_nf).abs().max().item())

            # Cosine similarity — flatten seq dim, compute per batch item
            f_flat  = out_f.view(out_f.size(0), -1)
            nf_flat = out_nf.view(out_nf.size(0), -1)
            cos = F.cosine_similarity(f_flat, nf_flat, dim=1).mean().item()
            cosines.append(cos)

            # KL divergence of softmax distributions
            p = F.softmax(out_f,  dim=-1).clamp(min=1e-10)
            q = F.softmax(out_nf, dim=-1).clamp(min=1e-10)
            kl = (p * (p / q).log()).sum(dim=-1).mean().item()
            kls.append(kl)

    return (
        statistics.mean(max_diffs),
        statistics.mean(cosines),
        statistics.mean(kls),
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    base_dir: str,
    layer_path: str,
    *,
    benchmark_mode: str = "runtime-patch",
    fusion_point: str = "q_b",
    variant: str = "V2",
    load_mode: str = "layer",
    device: str = "cuda:0",
    num_gpus: int | None = None,
) -> List[ShapeResult]:
    results = []

    fused_model, nonfused_model, hidden_dim = load_models(
        base_dir,
        layer_path,
        benchmark_mode=benchmark_mode,
        fusion_point=fusion_point,
        variant=variant,
        load_mode=load_mode,
        device=device,
        num_gpus=num_gpus,
    )
    target = torch.device(device)
    if _module_device(fused_model) != target:
        fused_model = fused_model.to(device, dtype=DTYPE)
    if _module_device(nonfused_model) != target:
        nonfused_model = nonfused_model.to(device, dtype=DTYPE)
    fused_model.eval()
    nonfused_model.eval()

    bench_device = device

    # Use hidden_dim from actual weights; out_dim from SHAPE_SWEEP is ignored
    # (the layer's own forward() determines output shape)
    shape_sweep = [
        (batch, seq_len, hidden_dim)
        for (batch, seq_len, _, _) in SHAPE_SWEEP
    ]

    for (batch, seq_len, hidden) in shape_sweep:
        out_dim = hidden   # placeholder for ShapeResult; real out shape is layer-defined
        print(f"\n{'='*60}")
        print(f"Shape: batch={batch}, seq={seq_len}, hidden={hidden}, out={out_dim}")
        print(f"{'='*60}")

        fused    = fused_model
        nonfused = nonfused_model

        # Input
        x = torch.randn(batch, seq_len, hidden, device=bench_device, dtype=DTYPE)

        result = ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=out_dim)

        # --- Latency ---
        print("  Measuring latency (non-fused)...")
        result.nonfused_latencies = measure_latency(nonfused, x)

        print("  Measuring latency (fused)...")
        result.fused_latencies = measure_latency(fused, x)

        # --- Peak memory ---
        print("  Measuring peak memory...")
        result.nonfused_peak_mem_mb = measure_peak_memory(nonfused, x)
        result.fused_peak_mem_mb    = measure_peak_memory(fused, x)

        # --- Numerical equivalence ---
        print("  Measuring numerical equivalence...")
        (result.max_abs_diff,
         result.cosine_sim,
         result.kl_divergence) = measure_numerical_equivalence(fused, nonfused, x)

        # Print summary for this shape
        print(f"\n  Latency (median ms):  fused={result.fused_median_ms:.3f}  non-fused={result.nonfused_median_ms:.3f}  speedup={result.speedup:.2f}x")
        print(f"  Latency (p99 ms):     fused={result.fused_p99_ms:.3f}  non-fused={result.nonfused_p99_ms:.3f}")
        print(f"  Throughput (tok/s):   fused={result.fused_throughput:,.0f}  non-fused={result.nonfused_throughput:,.0f}")
        print(f"  Peak mem (MB):        fused={result.fused_peak_mem_mb:.1f}  non-fused={result.nonfused_peak_mem_mb:.1f}")
        print(f"  Numerical equivalence:")
        print(f"    max |diff|  = {result.max_abs_diff:.6f}")
        print(f"    cosine sim  = {result.cosine_sim:.6f}  (1.0 = identical)")
        print(f"    KL div      = {result.kl_divergence:.6f}  (0.0 = identical distributions)")

        del x
        gc.collect()
        torch.cuda.empty_cache()

        results.append(result)

    return results


def print_summary_table(results: List[ShapeResult]):
    header = (
        f"{'batch':>5} {'seq':>5} {'hidden':>7} "
        f"{'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
        f"{'fused_p99':>10} {'nf_p99':>10} "
        f"{'fused_mem':>10} {'nf_mem':>10} "
        f"{'cos_sim':>9} {'kl_div':>9}"
    )
    print(f"\n{'='*len(header)}")
    print("SUMMARY TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.batch:>5} {r.seq_len:>5} {r.hidden:>7} "
            f"{r.fused_median_ms:>10.3f} {r.nonfused_median_ms:>10.3f} {r.speedup:>8.2f}x "
            f"{r.fused_p99_ms:>10.3f} {r.nonfused_p99_ms:>10.3f} "
            f"{r.fused_peak_mem_mb:>10.1f} {r.nonfused_peak_mem_mb:>10.1f} "
            f"{r.cosine_sim:>9.6f} {r.kl_divergence:>9.6f}"
        )
    print(f"{'='*len(header)}")


def _default_results_dir() -> str:
    """Directory for benchmark CSV outputs (never overwrites prior runs)."""
    return os.path.join(_BENCH_DIR, "results")


def _make_results_path(
    output_dir: str,
    *,
    benchmark_mode: str,
    fusion_point: str,
    variant: str,
    load_mode: str,
) -> str:
    """
    Unique path per run: benchmark/results/benchmark_<UTC>_<mode>_<fusion>_<variant>_<load>.csv
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    variant_tag = variant if benchmark_mode == "runtime-patch" else "checkpoints"
    base = f"benchmark_{ts}_{benchmark_mode}_{fusion_point}_{variant_tag}_{load_mode}.csv"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, base)
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return path


def save_csv(
    results: List[ShapeResult],
    path: str,
    *,
    run_metadata: dict | None = None,
):
    import csv

    meta = run_metadata or {}
    fields = [
        "run_timestamp_utc",
        "benchmark_mode",
        "fusion_point",
        "variant",
        "load_mode",
        "device",
        "batch", "seq_len", "hidden", "out_dim",
        "fused_median_ms", "nonfused_median_ms", "speedup",
        "fused_p99_ms", "nonfused_p99_ms",
        "fused_throughput", "nonfused_throughput",
        "fused_peak_mem_mb", "nonfused_peak_mem_mb",
        "max_abs_diff", "cosine_sim", "kl_divergence",
    ]
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                **meta,
                "batch": r.batch, "seq_len": r.seq_len,
                "hidden": r.hidden, "out_dim": r.out_dim,
                "fused_median_ms": r.fused_median_ms,
                "nonfused_median_ms": r.nonfused_median_ms,
                "speedup": r.speedup,
                "fused_p99_ms": r.fused_p99_ms,
                "nonfused_p99_ms": r.nonfused_p99_ms,
                "fused_throughput": r.fused_throughput,
                "nonfused_throughput": r.nonfused_throughput,
                "fused_peak_mem_mb": r.fused_peak_mem_mb,
                "nonfused_peak_mem_mb": r.nonfused_peak_mem_mb,
                "max_abs_diff": r.max_abs_diff,
                "cosine_sim": r.cosine_sim,
                "kl_divergence": r.kl_divergence,
            })
    print(f"\nResults saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm+Linear fusion vs non-fused (HuggingFace checkpoints)."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Base directory containing models/fused/ and models/non-fused/ subdirectories."
    )
    parser.add_argument(
        "--layer-path",
        default="language_model.model.layers.0.self_attn",
        help=(
            "Path to DeepseekV3Attention inside the checkpoint "
            "(default: language_model.model.layers.0.self_attn)."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("runtime-patch", "checkpoints"),
        default="runtime-patch",
        help=(
            "runtime-patch: non-fused weights + kimi_patch FusedRMSNormLinear kernels "
            "(measures real fusion speedup). "
            "checkpoints: compare models/non-fused vs models/fused on disk."
        ),
    )
    parser.add_argument(
        "--fusion-point", choices=tuple(FUSION_POINTS), default="q_b",
        help="MLA fusion site: q_b or kv_b.",
    )
    parser.add_argument(
        "--variant", choices=("V1", "V2", "V3"), default="V2",
        help="kimi_patch kernel variant (runtime-patch mode only).",
    )
    parser.add_argument(
        "--print-keys", action="store_true",
        help=(
            "Print all named modules from the non-fused model and exit. "
            "Use this to discover the correct --layer-path for your architecture."
        ),
    )
    parser.add_argument(
        "--load-mode", choices=("layer", "full"), default="layer",
        help=(
            "layer: load only --layer-path weights onto --device (~seconds). "
            "full: load entire NVFP4 checkpoint across --num-gpus GPUs "
            "(device_map=auto), then clone --layer-path onto --device."
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device for layer-only load and benchmark (default: cuda:0).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="GPUs for --load-mode full (default: all visible).",
    )
    parser.add_argument(
        "--test-load", action="store_true",
        help=(
            "Smoke test only: load weights + one forward pass, then exit. "
            "Does NOT run latency/p99/memory benchmarks (omit this flag for the full sweep)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for CSV output (default: benchmark/results/). "
            "Each run writes a new timestamped file; nothing is overwritten."
        ),
    )
    args = parser.parse_args()

    if args.print_keys:
        nonfused_dir = os.path.join(args.dir, "models", "non-fused")
        print_model_keys(nonfused_dir)
        sys.exit(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    n_gpu = torch.cuda.device_count()
    print(f"GPUs ({n_gpu}): {torch.cuda.get_device_name(0)} ...")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Base dir:   {args.dir}")
    print(f"Layer path:    {args.layer_path}")
    print(f"Benchmark mode: {args.benchmark_mode}")
    print(f"Fusion point:   {args.fusion_point}")
    if args.benchmark_mode == "runtime-patch":
        print(f"Variant:        {args.variant}")
    print(f"Load mode:      {args.load_mode}  |  Device: {args.device}")
    print(f"Warmup iters: {WARMUP_ITERS}  |  Measure iters: {MEASURE_ITERS}")

    bench_device = args.device

    if args.test_load:
        fused, nonfused, hidden = load_models(
            args.dir,
            args.layer_path,
            benchmark_mode=args.benchmark_mode,
            fusion_point=args.fusion_point,
            variant=args.variant,
            load_mode=args.load_mode,
            device=bench_device,
            num_gpus=args.num_gpus,
        )
        x = torch.randn(1, 128, hidden, device=bench_device, dtype=DTYPE)
        with torch.no_grad():
            y_f = fused(x)
            y_nf = nonfused(x)
        print(f"Forward OK — out shapes fused={tuple(y_f.shape)} non-fused={tuple(y_nf.shape)}")
        if args.load_mode == "full":
            used = [
                torch.cuda.memory_allocated(i) / 1024**3
                for i in range(torch.cuda.device_count())
            ]
            print(f"GPU memory allocated (GiB): {[round(u, 1) for u in used]}")
        print(
            "\n(--test-load: load smoke test passed. Re-run WITHOUT --test-load "
            "to run latency / p99 / memory / numerical-equivalence benchmarks.)"
        )
        sys.exit(0)

    results = run_benchmark(
        args.dir,
        args.layer_path,
        benchmark_mode=args.benchmark_mode,
        fusion_point=args.fusion_point,
        variant=args.variant,
        load_mode=args.load_mode,
        device=bench_device,
        num_gpus=args.num_gpus,
    )
    print_summary_table(results)

    output_dir = args.output_dir or _default_results_dir()
    variant = args.variant if args.benchmark_mode == "runtime-patch" else "n/a"
    out_path = _make_results_path(
        output_dir,
        benchmark_mode=args.benchmark_mode,
        fusion_point=args.fusion_point,
        variant=variant,
        load_mode=args.load_mode,
    )
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_csv(
        results,
        out_path,
        run_metadata={
            "run_timestamp_utc": run_ts,
            "benchmark_mode": args.benchmark_mode,
            "fusion_point": args.fusion_point,
            "variant": variant,
            "load_mode": args.load_mode,
            "device": args.device,
        },
    )