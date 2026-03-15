"""
Metric utilities for the multimodal regressor defined in `model.py`.

Provides:
- FLOPs counting for model efficiency
- Parameter counting
- PLCC computation
- CES (Compute Efficiency Score) calculation
"""

from typing import Optional

import math
import torch
import numpy as np
from scipy.stats import pearsonr

from torch.utils.flop_counter import FlopCounterMode

# -----------------------------------------------------------------------
# Canonical batch builder
# -----------------------------------------------------------------------
def build_canonical_batch(
    batch_size: int = 1,
    image_size: tuple = (224, 224),
    text_length: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    with_images: bool = True,
):
    """
    Build a canonical dummy batch for `MultimodalRegressor`.

    Returns:
        pixel_values : (batch_size, 3, H, W) or None
        input_ids    : (batch_size, text_length)
        attention_mask: (batch_size, text_length)
    """
    pixel_values = (
        torch.randn(batch_size, 3, image_size[0], image_size[1], device=device, dtype=dtype)
        if with_images
        else None
    )
    input_ids = torch.randint(0, 30522, (batch_size, text_length), device=device)
    attention_mask = torch.ones(batch_size, text_length, device=device, dtype=torch.long)
    return pixel_values, input_ids, attention_mask


# -----------------------------------------------------------------------
# Analytical FLOPs helpers
# -----------------------------------------------------------------------
def approx_transformer_block_flops(L: int, d: int, ffn_mult: int = 4) -> float:
    """
    Approximate FLOPs for ONE Transformer encoder layer (self-attn + FFN),
    forward pass only.

    L          : sequence length
    d          : hidden size
    ffn_mult   : FFN dimension multiplier (usually 4)
    """
    qkv  = 3.0 * L * d * d           # Q, K, V projections
    out  = 1.0 * L * d * d           # output projection
    attn = 2.0 * L * L * d           # QK^T and Attn·V
    ffn  = 2.0 * L * d * (ffn_mult * d)  # two linear layers in FFN
    return qkv + out + attn + ffn


def approx_efficientnet_b0_flops(image_size: tuple = (224, 224)) -> float:
    """
    EfficientNet-B0 FLOPs rough analytical estimate.

    Based on the well-known ~0.39 GFLOPs figure for 224×224 input.
    Scales quadratically with spatial resolution if a non-standard size
    is used.
    """
    # Reference: 390M FLOPs at 224×224
    ref_flops = 390e6
    h_ref, w_ref = 224, 224
    h, w = image_size
    scale = (h * w) / (h_ref * w_ref)
    return ref_flops * scale


def approximate_flops_multimodal_regressor(
    model,
    batch_size: int = 1,
    image_size: tuple = (224, 224),
    text_length: int = 128,
    with_images: bool = True,
) -> float:
    """
    Analytical FLOPs estimate for the current multimodal regressor.
    """
    # ---- Text branch ----
    text_cfg    = model.text_backbone.config
    text_layers = int(getattr(text_cfg, "num_hidden_layers", 6))
    text_d      = int(getattr(text_cfg, "hidden_size", 384))
    ffn_int     = int(getattr(text_cfg, "intermediate_size", 4 * text_d))
    text_ffn_mult = ffn_int // text_d
    text_flops  = text_layers * approx_transformer_block_flops(
        text_length, text_d, text_ffn_mult
    )

    # ---- Image branch ----
    image_flops = approx_efficientnet_b0_flops(image_size) if with_images else 0.0

    # ---- MLP head ----
    fused_dim   = model.image_dim + model.text_dim   # 1664
    hidden_dim  = model.hidden_dim
    bottleneck_dim = hidden_dim // 2
    head_flops  = (
        2 * fused_dim * hidden_dim        # Linear(fused_dim, hidden_dim)
        + 2 * hidden_dim * bottleneck_dim     # Linear(hidden_dim, hidden_dim // 2)
        + 2 * bottleneck_dim * model.num_outputs  # Linear(hidden_dim // 2, num_outputs)
    )

    total = text_flops + image_flops + head_flops
    return float(total * batch_size)


# -----------------------------------------------------------------------
# FLOPs measurement
# -----------------------------------------------------------------------
def measure_flops(
    model,
    batch_size: int = 1,
    image_size: tuple = (224, 224),
    text_length: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    with_images: bool = True,
) -> float:
    """
    Measure FLOPs for a single forward pass of the MultimodalRegressor.

    Uses PyTorch FlopCounterMode (torch >= 2.0) when available, otherwise
    falls back to the analytical approximation.

    Returns:
        total_flops : float
    """
    pixel_values, input_ids, attention_mask = build_canonical_batch(
        batch_size=batch_size,
        image_size=image_size,
        text_length=text_length,
        device=device,
        dtype=dtype,
        with_images=with_images,
    )

    # When pixel_values is None (text-only), the image backbone can't run;
    # skip FlopCounterMode and use the analytical path directly.
    if pixel_values is None:
        return approximate_flops_multimodal_regressor(
            model=model,
            batch_size=batch_size,
            image_size=image_size,
            text_length=text_length,
            with_images=False,
        )

    try:
        model.eval()
        with torch.no_grad():
            with FlopCounterMode(model, display=False) as flop_counter:
                _ = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        return float(flop_counter.get_total_flops())

    except Exception:
        return approximate_flops_multimodal_regressor(
            model=model,
            batch_size=batch_size,
            image_size=image_size,
            text_length=text_length,
            with_images=with_images,
        )


# -----------------------------------------------------------------------
# Parameter counting
# -----------------------------------------------------------------------
def count_params(model) -> int:
    """Total number of parameters (including frozen)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Regression metrics
# -----------------------------------------------------------------------
def compute_plcc(predictions, targets) -> float:
    """Pearson Linear Correlation Coefficient."""
    if len(predictions) < 2:
        return 0.0
    try:
        predictions = np.array(predictions).flatten()
        targets     = np.array(targets).flatten()
        plcc, _     = pearsonr(predictions, targets)
        return float(plcc) if not np.isnan(plcc) else 0.0
    except Exception:
        return 0.0
    

# -----------------------------------------------------------------------
# Efficiency scoring
# -----------------------------------------------------------------------
def geometric_resource_cost(
    params: int,
    flops: float,
    P_tgt: float = 1e9,
    F_tgt: float = 20e9,
    w: float = 0.5,
) -> float:
    """
    C = (Params / P_tgt)^w  *  (FLOPs / F_tgt)^w
    """
    return ((params / P_tgt) ** w) * ((flops / F_tgt) ** w)


def efficiency_factor(
    C: float,
    alpha: float = 0.05,
    beta: float = 2.0,
    gamma: float = 0.10,
) -> float:
    """
    E(C) = min(1 + alpha * ln(1/C), 1 + gamma)   if C <= 1
           1 / (1 + beta * ln(C))                 if C >  1
    """
    if C <= 1.0:
        return min(1.0 + alpha * math.log(1.0 / max(C, 1e-12)), 1.0 + gamma)
    else:
        return 1.0 / (1.0 + beta * math.log(C))


def ces_score(plcc: float, C: float) -> float:
    """
    Compute Efficiency Score  =  PLCC+  *  E(C)
    where PLCC+ = max(0, PLCC).
    """
    return max(0.0, float(plcc)) * efficiency_factor(C)


# -----------------------------------------------------------------------
# Composite metrics report
# -----------------------------------------------------------------------
def calculate_metrics(model, plcc: Optional[float] = None, device: str = "cuda") -> dict:
    """
    Compute and print all efficiency metrics for a MultimodalRegressor.

    Args:
        model  : MultimodalRegressor instance
        plcc   : validation PLCC (optional)
        device : device string

    Returns:
        metrics dict
    """
    print("\n" + "=" * 60)
    print("Computing Model Efficiency Metrics")
    print("=" * 60)

    total_p     = count_params(model)
    trainable_p = count_trainable_params(model)

    print("\n[Parameters]")
    print(f"  Total     : {total_p:,} ({total_p / 1e6:.2f}M)")
    print(f"  Trainable : {trainable_p:,} ({trainable_p / 1e6:.2f}M)")
    print(f"  Frozen    : {total_p - trainable_p:,} ({(total_p - trainable_p) / 1e6:.2f}M)")

    try:
        flops = measure_flops(model, device=device)
        print(f"\n[FLOPs]")
        print(f"  Total : {flops:,.0f} ({flops / 1e9:.2f}G)")
    except Exception as exc:
        print(f"\n[FLOPs] Error: {exc}")
        flops = approximate_flops_multimodal_regressor(model, 1, (224, 224), 128)
        print(f"  Approximate : {flops:,.0f} ({flops / 1e9:.2f}G)")

    C = geometric_resource_cost(params=total_p, flops=flops)
    E = efficiency_factor(C)

    print(f"\n[Efficiency]")
    print(f"  Resource Cost C     : {C:.6f}")
    print(f"  Efficiency Factor E : {E:.6f}")

    CES = None
    if plcc is not None:
        CES = ces_score(plcc, C)
        print(f"\n[Performance]")
        print(f"  PLCC  : {plcc:.4f}")
        print(f"  PLCC+ : {max(0.0, plcc):.4f}")
        print(f"  CES   : {CES:.6f}")
    else:
        print(f"\n[Performance]")
        print("  PLCC  : Not provided")

    print("=" * 60 + "\n")

    return {
        "total_params":      total_p,
        "trainable_params":  trainable_p,
        "flops":             flops,
        "resource_cost":     C,
        "efficiency_factor": E,
        "plcc":              plcc,
        "ces":               CES,
    }


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from model import MultimodalRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model= MultimodalRegressor().to(device)
    calculate_metrics(model, plcc=0.4, device=device)
