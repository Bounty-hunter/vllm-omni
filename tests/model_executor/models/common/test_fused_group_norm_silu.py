"""Unit tests for fused_group_norm_silu operator.

Tests numeric correctness against PyTorch native implementation:
    F.silu(F.group_norm(x, num_groups, weight, bias, eps))
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu

# Skip tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernels"
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("spatial_size", [(16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("num_groups", [8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_group_norm_silu_correctness(
    batch_size, channels, spatial_size, num_groups, dtype
):
    """Test numeric correctness against PyTorch native ops."""
    # Skip invalid configurations
    if channels % num_groups != 0:
        pytest.skip(f"channels {channels} not divisible by num_groups {num_groups}")
    
    H, W = spatial_size
    device = torch.device("cuda")
    eps = 1e-6
    
    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, channels, H, W, device=device, dtype=dtype)
    weight = torch.randn(channels, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    
    # Reference implementation
    ref_out = F.silu(F.group_norm(x, num_groups, weight, bias, eps))
    
    # Fused implementation
    fused_out = fused_group_norm_silu(x, weight, bias, num_groups, eps)
    
    # Check correctness
    # Use relaxed tolerance for fp16/bf16
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    else:
        rtol, atol = 1e-2, 1e-3
    
    torch.testing.assert_close(
        fused_out, ref_out, 
        rtol=rtol, atol=atol,
        msg=f"Mismatch for batch={batch_size}, C={channels}, "
            f"HW={spatial_size}, groups={num_groups}, dtype={dtype}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp32_accumulation_precision(dtype):
    """Test that fp32 accumulation matches PyTorch's behavior.
    
    GroupNorm uses fp32 for mean/variance computation regardless of input dtype.
    Our kernel should match this behavior.
    """
    batch_size, channels, H, W = 2, 64, 32, 32
    num_groups = 32
    eps = 1e-6
    device = torch.device("cuda")
    
    # Create inputs with values that expose precision differences
    torch.manual_seed(123)
    x = torch.randn(batch_size, channels, H, W, device=device, dtype=dtype) * 100
    weight = torch.ones(channels, device=device, dtype=dtype)
    bias = torch.zeros(channels, device=device, dtype=dtype)
    
    # Reference
    ref_out = F.silu(F.group_norm(x, num_groups, weight, bias, eps))
    
    # Fused
    fused_out = fused_group_norm_silu(x, weight, bias, num_groups, eps)
    
    # Should match closely even with large values
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    else:
        rtol, atol = 1e-2, 1e-2
    
    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


def test_edge_case_zero_input():
    """Test with zero input."""
    x = torch.zeros(1, 32, 16, 16, device="cuda")
    weight = torch.ones(32, device="cuda")
    bias = torch.zeros(32, device="cuda")
    
    ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
    fused_out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)
    
    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-6)


def test_edge_case_large_eps():
    """Test with large epsilon value."""
    x = torch.randn(1, 32, 16, 16, device="cuda")
    weight = torch.randn(32, device="cuda")
    bias = torch.randn(32, device="cuda")
    eps = 1e-2  # Larger than typical
    
    ref_out = F.silu(F.group_norm(x, 32, weight, bias, eps))
    fused_out = fused_group_norm_silu(x, weight, bias, 32, eps)
    
    torch.testing.assert_close(fused_out, ref_out, rtol=1e-4, atol=1e-5)


def test_backward_compatibility():
    """Test that output dtype matches input dtype."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(2, 64, 32, 32, device="cuda", dtype=dtype)
        weight = torch.randn(64, device="cuda", dtype=dtype)
        bias = torch.randn(64, device="cuda", dtype=dtype)
        
        out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)
        assert out.dtype == dtype, f"Output dtype {out.dtype} != input dtype {dtype}"
        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"


def test_invalid_channels_not_divisible():
    """Test that assertion fires when channels not divisible by num_groups."""
    x = torch.randn(1, 33, 16, 16, device="cuda")  # 33 not divisible by 32
    weight = torch.randn(33, device="cuda")
    bias = torch.randn(33, device="cuda")
    
    with pytest.raises(AssertionError, match="must be divisible by num_groups"):
        fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)


def test_invalid_weight_shape():
    """Test that assertion fires with wrong weight shape."""
    x = torch.randn(1, 64, 16, 16, device="cuda")
    weight = torch.randn(32, device="cuda")  # Wrong: should be 64
    bias = torch.randn(64, device="cuda")
    
    with pytest.raises(AssertionError, match="doesn't match channels"):
        fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)


def test_compile_fallback():
    """Test that compile mode falls back to native ops."""
    x = torch.randn(1, 32, 16, 16, device="cuda")
    weight = torch.randn(32, device="cuda")
    bias = torch.randn(32, device="cuda")
    
    # Compile the function
    compiled_fn = torch.compile(fused_group_norm_silu)
    
    # Should still work (via fallback)
    ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
    compiled_out = compiled_fn(x, weight, bias, 32, 1e-6)
    
    torch.testing.assert_close(compiled_out, ref_out, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [64, 128])
def test_hunyuan_vae_config(batch_size, channels):
    """Test with HunyuanImage3 VAE actual configuration.
    
    HunyuanImage3 ResnetBlock uses:
    - num_groups=32
    - eps=1e-6
    - Typical spatial sizes: 16x16 to 128x128 (after tiling)
    """
    spatial_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]
    
    for H, W in spatial_sizes:
        x = torch.randn(batch_size, channels, H, W, device="cuda", dtype=torch.float32)
        weight = torch.randn(channels, device="cuda")
        bias = torch.randn(channels, device="cuda")
        
        ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
        fused_out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)
        
        torch.testing.assert_close(
            fused_out, ref_out, 
            rtol=1e-5, atol=1e-6,
            msg=f"HunyuanVAE config: batch={batch_size}, C={channels}, HW=({H},{W})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
