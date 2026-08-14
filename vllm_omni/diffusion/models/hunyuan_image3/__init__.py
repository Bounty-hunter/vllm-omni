# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware-aware dispatch for HunyuanImage3 building blocks.

This module selects the appropriate implementation of building blocks
based on the current hardware platform:
- NVIDIA/CUDA: Optimized with fused kernels (nvidia/)
- NPU: Hardware-specific optimizations (npu/) [Future]
- Fallback: Default implementation

Hardware-optimized blocks:
1. VAE ResnetBlock: GroupNorm+SiLU fusion
2. Transformer ResBlock: GroupNorm+SiLU + AdaGN fusion
3. Transformer UNetDown/UNetUp: Use optimized ResBlock
"""

from vllm_omni.platforms import current_omni_platform

# ============================================================================
# VAE building blocks dispatch
# ============================================================================

if current_omni_platform.is_cuda():
    # NVIDIA: Import optimized VAE ResnetBlock
    try:
        from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import ResnetBlock
    except ImportError as e:
        import warnings
        warnings.warn(
            f"Failed to import NVIDIA-optimized VAE ResnetBlock, using default implementation. "
            f"Error: {e}",
            RuntimeWarning,
        )
        from vllm_omni.diffusion.models.hunyuan_image3.autoencoder_blocks import ResnetBlock
else:
    # Fallback: NPU, ROCm, CPU, etc.
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder_blocks import ResnetBlock

# ============================================================================
# Transformer building blocks dispatch
# ============================================================================

if current_omni_platform.is_cuda():
    # NVIDIA: Import optimized Transformer ResBlock
    try:
        from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock as TransformerResBlock
    except ImportError as e:
        import warnings
        warnings.warn(
            f"Failed to import NVIDIA-optimized Transformer ResBlock, using default implementation. "
            f"Error: {e}",
            RuntimeWarning,
        )
        from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import ResBlock as TransformerResBlock
else:
    # Fallback: NPU, ROCm, CPU, etc.
    from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import ResBlock as TransformerResBlock

# Always import Transformer UNet blocks and helpers from default
# (UNetDown/UNetUp automatically use the optimized ResBlock imported above)
from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import (
    UNetDown,
    UNetUp,
    conv_nd,
    normalization,
    linear,
    zero_module,
)

__all__ = [
    "ResnetBlock",  # VAE ResBlock (hardware-specific)
    "TransformerResBlock",  # Transformer ResBlock (hardware-specific)
]
