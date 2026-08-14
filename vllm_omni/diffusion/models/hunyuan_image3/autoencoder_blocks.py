# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default building blocks for HunyuanImage3 autoencoder.

This file contains the fallback/default implementation of VAE building blocks.
Hardware-specific optimizations are in nvidia/, npu/, etc.
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def swish(x: Tensor) -> Tensor:
    """Swish activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    """Wrapper for gradient checkpointing."""
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(module), *inputs, use_reentrant=False
        )
    else:
        return module(*inputs)


class Conv3d(nn.Conv3d):
    """Conv3d with memory-efficient patching for large inputs."""

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunks[i]

                padded_chunks.append(
                    F.conv3d(
                        padded_chunk,
                        self.weight,
                        self.bias,
                        self.stride,
                        (0, self.padding[1], self.padding[2]),
                        self.dilation,
                        self.groups,
                    )
                )
            output = torch.cat(padded_chunks, dim=-3)
        else:
            output = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


class AttnBlock(nn.Module):
    """Attention block with torch sdpa implementation."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """Residual block with GroupNorm and Swish activation (default/fallback implementation)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class DownsampleDCAE(nn.Module):
    """Downsampling block for DCAE."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        B, C, T, H, W = x.shape
        if self.add_temporal_downsample:
            x = rearrange(x, "b c (t t2) (h h2) (w w2) -> b (c t2 h2 w2) t h w", t2=2, h2=2, w2=2)
        else:
            x = rearrange(x, "b c t (h h2) (w w2) -> b (c h2 w2) t h w", h2=2, w2=2)
        assert x.shape[1] == self.out_channels
        return x


class UpsampleDCAE(nn.Module):
    """Upsampling block for DCAE."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample

        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        B, C, T, H, W = x.shape
        if self.add_temporal_upsample:
            x = rearrange(x, "b (c t2 h2 w2) t h w -> b c (t t2) (h h2) (w w2)", t2=2, h2=2, w2=2)
        else:
            x = rearrange(x, "b (c h2 w2) t h w -> b c t (h h2) (w w2)", h2=2, w2=2)
        return x


class Encoder(nn.Module):
    """VAE Encoder."""

    def __init__(
        self,
        z_channels: int,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        resolution: int,
        in_channels: int = 3,
        use_checkpointing: bool = False,
        add_temporal_downsample: list = None,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_checkpointing = use_checkpointing

        # downsampling
        self.conv_in = Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                add_t_down = add_temporal_downsample[i_level] if add_temporal_downsample else True
                down.downsample = DownsampleDCAE(block_in, block_in, add_temporal_downsample=add_t_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block], hs[-1], use_checkpointing=self.use_checkpointing
                )
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=self.use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=self.use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=self.use_checkpointing)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """VAE Decoder."""

    def __init__(
        self,
        z_channels: int,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        resolution: int,
        out_ch: int = 3,
        use_checkpointing: bool = False,
        add_temporal_upsample: list = None,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.use_checkpointing = use_checkpointing

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                add_t_up = add_temporal_upsample[i_level - 1] if add_temporal_upsample else True
                up.upsample = UpsampleDCAE(block_in, block_in, add_temporal_upsample=add_t_up)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=self.use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=self.use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=self.use_checkpointing)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(
                    self.up[i_level].block[i_block], h, use_checkpointing=self.use_checkpointing
                )
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


__all__ = [
    "AttnBlock",
    "ResnetBlock",
    "DownsampleDCAE",
    "UpsampleDCAE",
    "Encoder",
    "Decoder",
    "Conv3d",
    "swish",
    "forward_with_checkpointing",
]
