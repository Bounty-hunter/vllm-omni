# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image 3 diffusion model components."""

# Platform-selected building blocks. The dispatch itself lives in blocks.py so
# that autoencoder.py and hunyuan_image3_transformer.py can import it without
# going through this package __init__ (which would be circular).
from vllm_omni.diffusion.models.hunyuan_image3.blocks import ResBlock, ResnetBlock
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)

__all__ = [
    "HunyuanImage3Pipeline",
    "HunyuanImage3Model",
    "HunyuanImage3Text2ImagePipeline",
    "ResnetBlock",
    "ResBlock",
]
