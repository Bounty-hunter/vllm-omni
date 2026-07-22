"""Shared VAE / UNet residual building blocks (fused norms, layout helpers)."""

from vllm_omni.diffusion.layers.vae.fused_groupnorm import (
    FusedGroupNormAdaSiLU,
    FusedGroupNormSiLU,
    convert_conv2d_to_channels_last,
)

__all__ = [
    "FusedGroupNormAdaSiLU",
    "FusedGroupNormSiLU",
    "convert_conv2d_to_channels_last",
]
