# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVIDIA (CUDA) optimized building blocks for HunyuanImage3 Diffusion Transformer.

This implementation provides optimized ResBlock using fused kernels:
- GroupNorm+SiLU fusion for in_layers
- Adaptive Group Normalization (AdaGN) fusion for out_layers

Performance improvements:
- Reduces kernel launches in forward pass
- Eliminates intermediate tensor materialization
- Maintains numerical precision with fp32 accumulation
"""

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.common.ops import (
    fused_group_norm_silu,
    fused_adaptive_group_norm,
)
from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import (
    Upsample,
    Downsample,
)


# ============================================================================
# Helper functions
# ============================================================================

def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels, swish=0.0, dtype=None, device=None):
    """Make a standard normalization layer."""
    factory_kwargs = {"dtype": dtype, "device": device}
    return nn.GroupNorm(
        num_channels=channels, num_groups=32, eps=1e-6, **factory_kwargs
    )


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(nn.Module):
    """NVIDIA-optimized ResBlock with fused GroupNorm+SiLU and AdaGN.

    Optimizations:
    1. in_layers: Fused GroupNorm+SiLU (single kernel instead of 5)
    2. out_layers: Fused AdaGN (GroupNorm + adaptive modulation in one pass)

    This reduces kernel launches and memory bandwidth usage significantly.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        up=False,
        down=False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}

        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels
        self.dims = dims
        self.up = up
        self.down = down

        # Input normalization (weights stored for fusion)
        self.in_norm = normalization(in_channels, **factory_kwargs)
        self.in_conv = conv_nd(
            dims,
            in_channels,
            self.out_channels,
            3,
            padding=1,
            **factory_kwargs,
        )

        # Upsampling/downsampling
        self.updown = up or down
        if up:
            from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import Upsample
            self.h_upd = Upsample(in_channels, dims, **factory_kwargs)
            self.x_upd = Upsample(in_channels, dims, **factory_kwargs)
        elif down:
            from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import Downsample
            self.h_upd = Downsample(in_channels, dims, **factory_kwargs)
            self.x_upd = Downsample(in_channels, dims, **factory_kwargs)

        # Timestep embedding projection
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels,
                **factory_kwargs,
            ),
        )

        # Output normalization (weights stored for AdaGN fusion)
        self.out_norm = normalization(self.out_channels, **factory_kwargs)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.out_conv = zero_module(
            conv_nd(
                dims,
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                **factory_kwargs,
            )
        )

        # Skip connection
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(
                dims, in_channels, self.out_channels, 1, **factory_kwargs
            )

    def forward(self, x, emb):
        """Forward pass with fused operations.

        Args:
            x: Input tensor [B, C, ...]
            emb: Timestep embedding [B, emb_channels]

        Returns:
            Output tensor [B, out_channels, ...]
        """
        # Upsampling/downsampling if needed
        if self.updown:
            h = self.h_upd(x)
            x = self.x_upd(x)
        else:
            h = x

        # === OPTIMIZATION 1: Fused GroupNorm+SiLU for in_layers ===
        h = fused_group_norm_silu(
            h,
            self.in_norm.weight,
            self.in_norm.bias,
            num_groups=32,
            eps=1e-6,
        )
        h = self.in_conv(h)

        # Timestep conditioning
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Split into scale and shift for AdaGN
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        # === OPTIMIZATION 2: Fused AdaGN for out_layers ===
        # Fuses: GroupNorm(h) * (1 + scale) + shift
        h = fused_adaptive_group_norm(
            h,
            self.out_norm.weight,
            self.out_norm.bias,
            scale,
            shift,
            num_groups=32,
            eps=1e-6,
        )

        # Remaining out_layers: SiLU + dropout + conv
        h = nn.functional.silu(h)
        h = self.dropout_layer(h)
        h = self.out_conv(h)

        return self.skip_connection(x) + h


__all__ = ["ResBlock"]
