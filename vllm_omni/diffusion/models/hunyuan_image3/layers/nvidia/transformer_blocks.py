# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``ResBlock`` for HunyuanImage3 -- NVIDIA CUDA + Triton implementation.

Two fusions over the default block:

* ``in_layers``: ``GroupNorm -> SiLU`` becomes one :func:`fused_group_norm_silu`
  kernel instead of two ops.
* ``out_layers``: ``GroupNorm -> adaptive modulation -> SiLU`` (i.e.
  ``SiLU(norm(h) * (1 + scale) + shift)``) becomes one
  :func:`fused_adaptive_group_norm_silu` kernel instead of four.
"""

import torch
from torch import nn

from vllm_omni.diffusion.models.hunyuan_image3.layers.common import conv_nd, normalization, zero_module
from vllm_omni.model_executor.models.common.ops import (
    fused_adaptive_group_norm_silu,
    fused_group_norm_silu,
)


class ResBlock(nn.Module):
    r"""
    A residual block that can optionally change the number of channels.
    Args:
        in_channels (`int`):
            The number of input channels.
        emb_channels (`int`):
            The number of timestep embedding channels.
        dropout (`float`):
            The rate of dropout.
        out_channels (`int`, *optional*):
            If specified, the number of output channels.
        use_conv (`bool`, *optional*):
            If True and out_channels is specified, use a spatial convolution instead of a
            smaller 1x1 convolution to change the channels in the skip connection.
        dims (`int`, *optional*):
            Determines if the signal is 1D, 2D, or 3D.
        up (`bool`, *optional*):
            If True, use this block for upsampling.
        down (`bool`, *optional*):
            If True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),  # noqa: N802
        )

        self.updown = up or down
        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels, **factory_kwargs))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),  # noqa: N802
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs)  # noqa: N802
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)  # noqa: N802

    def forward(self, x, emb) -> torch.Tensor:
        # ``in_layers``/``out_layers`` stay nn.Sequential and are indexed into
        # rather than unpacked into named submodules -- that is what keeps the
        # state_dict keys (``in_layers.0/.2``, ``out_layers.0/.3``) identical to
        # the unfused block, so checkpoints load unchanged.
        in_norm, in_conv = self.in_layers[0], self.in_layers[-1]

        # GroupNorm -> SiLU, one kernel instead of two (NCHW for the fused kernel).
        h = fused_group_norm_silu(x, in_norm.weight, in_norm.bias, num_groups=in_norm.num_groups, eps=in_norm.eps)
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        # cuDNN conv prefers channels_last; convert around the conv only.
        h = in_conv(h.to(memory_format=torch.channels_last))
        h = h.to(memory_format=torch.contiguous_format)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive GroupNorm -> SiLU: GroupNorm -> mul -> add -> silu, one
        # kernel instead of four. The ``nn.SiLU`` at ``out_layers[1]`` is
        # skipped here but stays in the Sequential, so state_dict keys are
        # unchanged.
        out_norm, out_rest = self.out_layers[0], self.out_layers[2:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = fused_adaptive_group_norm_silu(
            h,
            out_norm.weight,
            out_norm.bias,
            scale,
            shift,
            num_groups=out_norm.num_groups,
            eps=out_norm.eps,
        )
        # out_rest = Dropout, conv -> channels_last for the conv, back for the residual.
        h = out_rest(h.to(memory_format=torch.channels_last)).to(memory_format=torch.contiguous_format)

        # skip_connection may be a 1x1 conv when channels change; run it NHWC too.
        if isinstance(self.skip_connection, nn.Conv2d):
            x = self.skip_connection(x.to(memory_format=torch.channels_last)).to(memory_format=torch.contiguous_format)
        else:
            x = self.skip_connection(x)
        return x + h


__all__ = ["ResBlock"]
