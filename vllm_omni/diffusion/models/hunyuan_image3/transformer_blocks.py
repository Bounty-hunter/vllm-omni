# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default building blocks for HunyuanImage3 Diffusion Transformer.

This file contains the fallback/default implementation of Transformer building blocks.
Hardware-specific optimizations are in nvidia/, npu/, etc.
"""

import torch
import torch.nn as nn
from einops import rearrange


# ============================================================================
# Helper functions for building blocks
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
    """Make a standard normalization layer with optional swish activation.

    Args:
        channels: Number of input channels
        swish: Swish beta parameter (0.0 = GroupNorm only)
    """
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


# ============================================================================
# Transformer ResBlock with AdaGN support
# ============================================================================

class ResBlock(nn.Module):
    """Residual block for Diffusion Transformer with Adaptive Group Normalization.

    This block supports:
    - Standard normalization + SiLU activation
    - Adaptive Group Normalization (AdaGN) for timestep conditioning
    - Optional upsampling/downsampling
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

        # Input layers: normalization + conv
        self.in_layers = nn.Sequential(
            normalization(in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(
                dims,
                in_channels,
                self.out_channels,
                3,
                padding=1,
                **factory_kwargs,
            ),
        )

        # Upsampling/downsampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, dims, **factory_kwargs)
            self.x_upd = Upsample(in_channels, dims, **factory_kwargs)
        elif down:
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

        # Output layers: dropout + normalization + conv
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    **factory_kwargs,
                )
            ),
        )

        # Skip connection
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(
                dims, in_channels, self.out_channels, 1, **factory_kwargs
            )

    def forward(self, x, emb):
        """Forward pass with timestep conditioning.

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

        # Input layers (normalization + SiLU + conv)
        h = self.in_layers(h)

        # Timestep conditioning via AdaGN
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Split into scale and shift for AdaGN
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        # Output layers with AdaGN
        h = self.out_layers[0](h)  # normalization
        h = h * (1 + scale) + shift  # AdaGN: norm * (1 + scale) + shift
        h = self.out_layers[1](h)  # SiLU
        h = self.out_layers[2](h)  # dropout
        h = self.out_layers[3](h)  # conv

        return self.skip_connection(x) + h


# ============================================================================
# Upsampling and Downsampling blocks
# ============================================================================

class Upsample(nn.Module):
    """Upsampling block for Transformer."""

    def __init__(self, channels, dims=2, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.channels = channels
        self.dims = dims
        self.conv = conv_nd(
            dims, channels, channels, 3, padding=1, **factory_kwargs
        )

    def forward(self, x):
        if self.dims == 2:
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="nearest"
            )
        else:  # 3D
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="nearest"
            )
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling block for Transformer."""

    def __init__(self, channels, dims=2, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.channels = channels
        self.dims = dims
        self.conv = conv_nd(
            dims, channels, channels, 3, stride=2, padding=1, **factory_kwargs
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# UNet Down/Up blocks
# ============================================================================

class UNetDown(nn.Module):
    """Downsampling block with ResBlocks for UNet structure."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

        self.model.append(
            conv_nd(
                2,
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs,
            )
        )

    def forward(self, x, t, token_h, token_w):
        """Forward pass.

        Args:
            x: Input tensor [B, seq_len, C]
            t: Timestep embedding
            token_h: Height of token grid
            token_w: Width of token grid
        """
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


class UNetUp(nn.Module):
    """Upsampling block with ResBlocks for UNet structure."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
        out_norm=False,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    conv_nd(
                        2,
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),
                )
            )
        else:
            self.model.append(
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            )

    def forward(self, x, t, token_h, token_w):
        """Forward pass.

        Args:
            x: Input tensor [B, seq_len, C]
            t: Timestep embedding
            token_h: Height of token grid
            token_w: Width of token grid
        """
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


__all__ = [
    "ResBlock",
    "UNetDown",
    "UNetUp",
    "conv_nd",
    "normalization",
    "linear",
    "zero_module",
]
