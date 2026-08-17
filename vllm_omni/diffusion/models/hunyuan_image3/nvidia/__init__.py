# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVIDIA (CUDA) implementations.
"""

from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import (
    ResnetBlock,
)
from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock

__all__ = ["ResnetBlock", "ResBlock"]
