"""Fused GroupNorm(+Ada)+SiLU shared by SD/Flux-style VAEs and UNet ResBlocks.

CUDA path uses a Triton fused kernel. Eager reshape GroupNorm+SiLU is the
native / fallback path (NPU and Triton-disable).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp

# Default on. Set VLLM_OMNI_FUSED_GN_TRITON=0 to force eager reshape fallback.
_FUSED_GN_TRITON = os.environ.get("VLLM_OMNI_FUSED_GN_TRITON", "1") != "0"


def convert_conv2d_to_channels_last(module: nn.Module) -> None:
    """Rewrite Conv2d weights in-place to channels_last (safe after weight load)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.weight.ndim == 4:
            if not m.weight.is_contiguous(memory_format=torch.channels_last):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)


def _group_norm_silu_reshape(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Eager GroupNorm+SiLU via explicit reshape reductions (reference / fallback)."""
    if x.ndim == 4:
        n, c, h, w = x.shape
        xg = x.reshape(n, num_groups, c // num_groups, h, w)
        reduce_dims = (2, 3, 4)
        view_shape = (1, -1, 1, 1)
    elif x.ndim == 5:
        n, c, t, h, w = x.shape
        xg = x.reshape(n, num_groups, c // num_groups, t, h, w)
        reduce_dims = (2, 3, 4, 5)
        view_shape = (1, -1, 1, 1, 1)
    else:
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))

    mean = xg.mean(dim=reduce_dims, keepdim=True)
    var = xg.var(dim=reduce_dims, unbiased=False, keepdim=True)
    xg = (xg - mean) * torch.rsqrt(var + eps)
    y = xg.reshape(x.shape)
    if weight is not None:
        y = y * weight.view(*view_shape)
    if bias is not None:
        y = y + bias.view(*view_shape)
    return F.silu(y)


def _group_norm_ada_silu_reshape(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if x.ndim == 4:
        n, c, h, w = x.shape
        xg = x.reshape(n, num_groups, c // num_groups, h, w)
        reduce_dims = (2, 3, 4)
        view_shape = (1, -1, 1, 1)
    elif x.ndim == 5:
        n, c, t, h, w = x.shape
        xg = x.reshape(n, num_groups, c // num_groups, t, h, w)
        reduce_dims = (2, 3, 4, 5)
        view_shape = (1, -1, 1, 1, 1)
    else:
        h = F.group_norm(x, num_groups, weight, bias, eps)
        return F.silu(h * (1.0 + scale) + shift)

    mean = xg.mean(dim=reduce_dims, keepdim=True)
    var = xg.var(dim=reduce_dims, unbiased=False, keepdim=True)
    xg = (xg - mean) * torch.rsqrt(var + eps)
    y = xg.reshape(x.shape)
    if weight is not None:
        y = y * weight.view(*view_shape)
    if bias is not None:
        y = y + bias.view(*view_shape)
    return F.silu(y * (1.0 + scale) + shift)


def _triton_gn_silu(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm_omni.diffusion.layers.vae.fused_groupnorm_triton import fused_group_norm_silu_triton

    return fused_group_norm_silu_triton(x, weight, bias, num_groups, eps, scale=scale, shift=shift)


class FusedGroupNormSiLU(nn.GroupNorm, CustomOp):
    """``silu(group_norm(x))`` — Triton fused kernel on CUDA."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.GroupNorm.__init__(
            self,
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            **factory_kwargs,
        )
        self._forward_method = CustomOp.dispatch_forward(self)
        self._use_triton = _FUSED_GN_TRITON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return _group_norm_silu_reshape(x, self.weight, self.bias, self.num_groups, self.eps)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if self._use_triton:
            try:
                return _triton_gn_silu(x, self.weight, self.bias, self.num_groups, self.eps)
            except Exception:
                return self.forward_native(x)
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps))

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class FusedGroupNormAdaSiLU(nn.GroupNorm, CustomOp):
    """``silu(group_norm(x) * (1 + scale) + shift)`` — Triton fused kernel on CUDA."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.GroupNorm.__init__(
            self,
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            **factory_kwargs,
        )
        self._forward_method = CustomOp.dispatch_forward(self)
        self._use_triton = _FUSED_GN_TRITON

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x, scale, shift)

    def forward_native(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return _group_norm_ada_silu_reshape(
            x, self.weight, self.bias, self.num_groups, self.eps, scale, shift
        )

    def forward_cuda(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if self._use_triton:
            try:
                return _triton_gn_silu(
                    x, self.weight, self.bias, self.num_groups, self.eps, scale=scale, shift=shift
                )
            except Exception:
                return self.forward_native(x, scale, shift)
        return self.forward_native(x, scale, shift)

    def forward_hip(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x, scale, shift)

    def forward_npu(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        h = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        return F.silu(h * (1.0 + scale) + shift)

    def forward_xpu(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x, scale, shift)
