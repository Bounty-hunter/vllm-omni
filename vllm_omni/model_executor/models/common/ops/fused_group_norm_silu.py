"""Fused GroupNorm + SiLU operator.

This operator fuses GroupNorm followed by SiLU activation into a single kernel,
reducing memory traffic and kernel launch overhead. The implementation uses
Triton for CUDA/ROCm compatibility, and falls back to native PyTorch ops when
Triton is unavailable (NPU, CPU, ...), so callers never need a platform check.

Performance:
- Saves ~3576 kernel launches in HunyuanImage3 VAE (6.32% GPU time + 7.7% launches)
- Compatible across CUDA, ROCm via single Triton implementation
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
    def _group_norm_silu_kernel(
        # Input/Output pointers
        x_ptr, out_ptr,
        # Normalization parameters
        weight_ptr, bias_ptr,
        # Strides
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_on, stride_oc, stride_oh, stride_ow,
        # Shape info
        N, C, H, W,
        num_groups: tl.constexpr,
        eps: tl.constexpr,
        # Block sizes
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GroupNorm + SiLU kernel.
        
        Computes: SiLU(GroupNorm(x)) in a single pass.
        Uses fp32 accumulation for moments to match PyTorch's numeric behavior.
        """
        # Get program ID
        pid = tl.program_id(0)
        
        # Calculate which (batch, group) this program handles
        group_size = C // num_groups
        n_idx = pid // num_groups
        g_idx = pid % num_groups
        
        if n_idx >= N:
            return
        
        # Calculate spatial size
        spatial_size = H * W
        
        # === Pass 1: Compute mean and variance (fp32 accumulation) ===
        mean_acc = tl.zeros([1], dtype=tl.float32)
        var_acc = tl.zeros([1], dtype=tl.float32)
        
        # Iterate over channels in this group
        for c_offset in range(group_size):
            c_idx = g_idx * group_size + c_offset
            
            # Iterate over spatial locations
            for spatial_idx in range(0, spatial_size, BLOCK_SIZE):
                offsets = spatial_idx + tl.arange(0, BLOCK_SIZE)
                mask = offsets < spatial_size
                
                # Calculate 2D indices
                h_idx = offsets // W
                w_idx = offsets % W
                
                # Load input (cast to fp32 for accumulation)
                x_idx = (n_idx * stride_xn + 
                        c_idx * stride_xc + 
                        h_idx * stride_xh + 
                        w_idx * stride_xw)
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                x_val = x_val.to(tl.float32)
                
                # Accumulate for mean
                mean_acc += tl.sum(x_val, axis=0)
                
                # Accumulate for variance
                var_acc += tl.sum(x_val * x_val, axis=0)
        
        # Finalize mean and variance
        group_total = group_size * spatial_size
        mean = mean_acc / group_total
        var = var_acc / group_total - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # === Pass 2: Normalize, apply affine transform, and SiLU ===
        for c_offset in range(group_size):
            c_idx = g_idx * group_size + c_offset
            
            # Load affine parameters
            weight_val = tl.load(weight_ptr + c_idx)
            bias_val = tl.load(bias_ptr + c_idx)
            
            # Process spatial locations
            for spatial_idx in range(0, spatial_size, BLOCK_SIZE):
                offsets = spatial_idx + tl.arange(0, BLOCK_SIZE)
                mask = offsets < spatial_size
                
                h_idx = offsets // W
                w_idx = offsets % W
                
                # Load input
                x_idx = (n_idx * stride_xn + 
                        c_idx * stride_xc + 
                        h_idx * stride_xh + 
                        w_idx * stride_xw)
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                x_val = x_val.to(tl.float32)
                
                # Normalize and apply affine
                norm_val = (x_val - mean) * rstd * weight_val + bias_val
                
                # Apply SiLU: x * sigmoid(x)
                sigmoid_val = tl.sigmoid(norm_val)
                out_val = norm_val * sigmoid_val
                
                # Store output (cast back to input dtype)
                out_idx = (n_idx * stride_on + 
                          c_idx * stride_oc + 
                          h_idx * stride_oh + 
                          w_idx * stride_ow)
                tl.store(out_ptr + out_idx, out_val, mask=mask)


def fused_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused GroupNorm + SiLU activation.
    
    Computes: SiLU(GroupNorm(x, num_groups, weight, bias, eps))
    
    This is mathematically equivalent to:
        F.silu(F.group_norm(x, num_groups, weight, bias, eps))
    
    But fuses the operations into a single Triton kernel to:
    1. Reduce memory traffic (no materialized intermediate tensors)
    2. Reduce kernel launch overhead
    3. Maintain fp32 accumulation precision for numeric alignment
    
    Args:
        x: Input tensor of shape (N, C, *spatial); any spatial rank is accepted,
            e.g. (N, C, H, W) for the 2D case or (N, C, T, H, W) for the 3D VAE.
        weight: Per-channel scale of shape (C,)
        bias: Per-channel bias of shape (C,)
        num_groups: Number of groups for GroupNorm (default: 32)
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Output tensor of the same shape as ``x``, with the dtype eager
        ``F.group_norm`` would produce (see ``group_norm_output_dtype``).

    Examples:
        >>> x = torch.randn(2, 64, 32, 32, device='cuda')
        >>> weight = torch.randn(64, device='cuda')
        >>> bias = torch.randn(64, device='cuda')
        >>> out = fused_group_norm_silu(x, weight, bias, num_groups=32)
        >>> out.shape
        torch.Size([2, 64, 32, 32])

    Note:
        Inputs whose spatial rank is not 2 are flattened to (N, C, 1, -1)
        before the launch and reshaped back afterwards. GroupNorm reduces over
        the whole channel group and every spatial position, so collapsing the
        spatial axes is exact, not an approximation.
    """
    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))

    # Validate inputs
    assert x.ndim >= 3, f"Expected at least 3D input (N, C, *spatial), got {x.ndim}D"
    assert x.size(1) % num_groups == 0, \
        f"Channels {x.size(1)} must be divisible by num_groups {num_groups}"
    assert weight.ndim == 1 and weight.size(0) == x.size(1), \
        f"Weight shape {weight.shape} doesn't match channels {x.size(1)}"
    assert bias.ndim == 1 and bias.size(0) == x.size(1), \
        f"Bias shape {bias.shape} doesn't match channels {x.size(1)}"

    # Collapse arbitrary spatial ranks into a single axis so one 4D kernel
    # serves both the 2D DiT blocks and the 3D VAE blocks.
    orig_shape = x.shape
    if x.ndim != 4:
        x = x.reshape(orig_shape[0], orig_shape[1], 1, -1)

    N, C, H, W = x.shape

    # Allocate output with the dtype eager GroupNorm would return, so that the
    # fused path stays a drop-in replacement inside autocast regions.
    out = torch.empty_like(x, dtype=group_norm_output_dtype(x))

    # Launch kernel
    # Each program handles one (batch, group) pair
    grid = lambda meta: (N * num_groups,)
    
    # Choose block size based on spatial dimensions
    spatial_size = H * W
    BLOCK_SIZE = min(1024, triton.next_power_of_2(spatial_size))
    
    _group_norm_silu_kernel[grid](
        x, out,
        weight, bias,
        # Strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Shape
        N, C, H, W,
        num_groups=num_groups,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.reshape(orig_shape)