"""NVIDIA (CUDA) optimized building blocks for HunyuanImage3 autoencoder.

This module provides hardware-specific optimizations using fused kernels.
"""

from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import (
    AttnBlock,
    ResnetBlock,
    DownsampleDCAE,
    UpsampleDCAE,
    Encoder,
    Decoder,
)

__all__ = [
    "AttnBlock",
    "ResnetBlock",
    "DownsampleDCAE",
    "UpsampleDCAE",
    "Encoder",
    "Decoder",
]
