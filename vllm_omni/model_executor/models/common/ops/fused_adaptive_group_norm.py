# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Adaptive Group Normalization (AdaGN) operator.

This module implements the fused AdaGN pattern commonly used in Diffusion Transformers:
    output = GroupNorm(x) * (1 + scale) + shift

Where scale and shift are conditioning signals (e.g., from timestep embeddings).
This fusion eliminates intermediate tensor materialization and reduces kernel launches.
"""

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops._dtype_utils import (
    group_norm_output_dtype,
)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _adaptive_group_norm_kernel(
        x_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        shift_ptr,
        stride_batch,
        stride_channel,
        stride_spatial,
        num_channels: tl.constexpr,
        num_groups: tl.constexpr,
        group_size: tl.constexpr,
        spatial_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused Adaptive Group Normalization kernel.

        Computes: GroupNorm(x, weight, bias) * (1 + scale) + shift in a single pass.
        """
        # Get batch and group indices
        batch_idx = tl.program_id(0)
        group_idx = tl.program_id(1)

        # Calculate starting channel for this group
        group_start_channel = group_idx * group_size

        # Compute mean and variance for this group (fp32 accumulation for stability)
        mean_acc = tl.zeros([1], dtype=tl.float32)
        var_acc = tl.zeros([1], dtype=tl.float32)

        for c_offset in range(0, group_size, BLOCK_SIZE):
            c_idx = group_start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
            mask_c = c_idx < group_start_channel + group_size

            for s_idx in range(spatial_size):
                offset = batch_idx * stride_batch + c_idx * stride_channel + s_idx * stride_spatial
                x_val = tl.load(x_ptr + offset, mask=mask_c, other=0.0)
                x_val_fp32 = x_val.to(tl.float32)

                mean_acc += tl.sum(x_val_fp32, axis=0)
                var_acc += tl.sum(x_val_fp32 * x_val_fp32, axis=0)

        # Finalize mean and variance
        group_numel = tl.cast(group_size * spatial_size, tl.float32)
        mean = mean_acc / group_numel
        var = var_acc / group_numel - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # Apply normalization, affine transform, and adaptive modulation
        for c_offset in range(0, group_size, BLOCK_SIZE):
            c_idx = group_start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
            mask_c = c_idx < group_start_channel + group_size

            # Load weight and bias for affine transform
            weight_val = tl.load(weight_ptr + c_idx, mask=mask_c, other=1.0).to(tl.float32)
            bias_val = tl.load(bias_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)

            # Load scale and shift for adaptive modulation
            scale_offset = batch_idx * num_channels + c_idx
            shift_offset = batch_idx * num_channels + c_idx
            scale_val = tl.load(scale_ptr + scale_offset, mask=mask_c, other=0.0).to(tl.float32)
            shift_val = tl.load(shift_ptr + shift_offset, mask=mask_c, other=0.0).to(tl.float32)

            for s_idx in range(spatial_size):
                offset = batch_idx * stride_batch + c_idx * stride_channel + s_idx * stride_spatial
                x_val = tl.load(x_ptr + offset, mask=mask_c, other=0.0).to(tl.float32)

                # GroupNorm: (x - mean) * rstd * weight + bias
                norm_val = (x_val - mean) * rstd * weight_val + bias_val

                # AdaGN: norm * (1 + scale) + shift
                out_val = norm_val * (1.0 + scale_val) + shift_val

                # ``tl.store`` casts to the output pointer's dtype, which is chosen
                # by the caller to match eager GroupNorm's autocast behaviour.
                tl.store(out_ptr + offset, out_val, mask=mask_c)


def fused_adaptive_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_groups: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused Adaptive Group Normalization with Triton kernel.

    Computes: GroupNorm(x, weight, bias) * (1 + scale) + shift

    This fusion is commonly used in Diffusion Transformers for timestep conditioning.

    Args:
        x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W)
        weight: Affine weight of shape (C,)
        bias: Affine bias of shape (C,)
        scale: Adaptive scale of shape (B, C) or (B, C, 1, 1) or (B, C, 1, 1, 1)
        shift: Adaptive shift of shape (B, C) or (B, C, 1, 1) or (B, C, 1, 1, 1)
        num_groups: Number of groups for GroupNorm
        eps: Epsilon for numerical stability

    Returns:
        Output tensor of same shape as x
    """
    # Check input validity
    assert x.ndim in [4, 5], f"Expected 4D or 5D input, got {x.ndim}D"
    B, C = x.shape[:2]
    assert C % num_groups == 0, f"num_channels ({C}) must be divisible by num_groups ({num_groups})"

    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        broadcast = (B, C) + (1,) * (x.ndim - 2)
        normed = F.group_norm(x, num_groups, weight, bias, eps)
        return normed * (1.0 + scale.reshape(broadcast)) + shift.reshape(broadcast)

    # Flatten spatial dimensions
    if x.ndim == 4:
        spatial_size = x.shape[2] * x.shape[3]
    else:  # 5D
        spatial_size = x.shape[2] * x.shape[3] * x.shape[4]

    # Reshape scale and shift to (B, C) if needed
    scale_2d = scale.view(B, C)
    shift_2d = shift.view(B, C)

    # Flatten x to (B, C, spatial)
    x_flat = x.view(B, C, -1)

    # Prepare output tensor with the dtype eager GroupNorm would return, so the
    # fused path stays a drop-in replacement inside autocast regions.
    out = torch.empty_like(x, dtype=group_norm_output_dtype(x))
    out_flat = out.view(B, C, -1)

    # Calculate group size
    group_size = C // num_groups

    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(min(group_size, 128))

    # Launch kernel with grid (batch_size, num_groups)
    grid = (B, num_groups)

    _adaptive_group_norm_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        scale_2d,
        shift_2d,
        stride_batch=C * spatial_size,
        stride_channel=spatial_size,
        stride_spatial=1,
        num_channels=C,
        num_groups=num_groups,
        group_size=group_size,
        spatial_size=spatial_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


__all__ = ["fused_adaptive_group_norm"]
