# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ResnetBlock for the HunyuanImage3 autoencoder — NVIDIA CUDA + Triton implementation.

Split out from autoencoder.py because its ``GroupNorm -> SiLU`` pairs are served
by :func:`fused_group_norm_silu`, a single Triton kernel that falls back to
native ``F.silu(F.group_norm(...))`` when Triton is unavailable. The submodule
layout is untouched, so state_dict keys are identical to the unfused version.

``Conv3d`` is duplicated here rather than imported from autoencoder.py, which
would make the two modules import each other.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu

from ._cudnn import cudnn_settings



class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        with cudnn_settings(benchmark = True, deterministic = True):
            h = x
            h = fused_group_norm_silu(
                h, self.norm1.weight, self.norm1.bias, num_groups=self.norm1.num_groups, eps=self.norm1.eps
            )
            h = self.conv1(h)

            h = fused_group_norm_silu(
                h, self.norm2.weight, self.norm2.bias, num_groups=self.norm2.num_groups, eps=self.norm2.eps
            )
            h = self.conv2(h)

            if self.in_channels != self.out_channels:
                x = self.nin_shortcut(x)
            return x + h


__all__ = ["ResnetBlock"]
