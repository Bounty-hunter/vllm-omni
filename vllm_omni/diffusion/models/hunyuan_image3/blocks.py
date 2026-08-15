# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware-aware dispatch for the HunyuanImage3 building blocks.

Selects the implementation of the two residual blocks based on the running
platform:

* ``ResnetBlock`` -- the VAE block, ``GroupNorm -> SiLU`` fusion.
* ``ResBlock``    -- the DiT block, ``GroupNorm -> SiLU`` plus AdaGN fusion.

Layout::

    autoencoder_blocks.py         default ResnetBlock
    transformer_blocks.py         default ResBlock
    nvidia/autoencoder_blocks.py  CUDA ResnetBlock
    nvidia/transformer_blocks.py  CUDA ResBlock

This module is deliberately a *leaf*: it pulls in only the block modules and
the platform probe, never ``autoencoder.py``, ``hunyuan_image3_transformer.py``
or the pipeline. That is what lets those modules import from here without a
circular import, and it means the dispatch does not depend on statement order
inside ``__init__.py``.

Note there is no ``try/except ImportError`` around the CUDA branch. A previous
version wrapped it, and when the import raised for an unrelated reason the
warning was easy to miss and every CUDA machine silently ran the default
blocks. If the CUDA blocks fail to import on a CUDA machine that is a bug, and
it should be loud. Falling back on a *non*-CUDA platform is the job of the
``else`` branch below, which is a plain, always-valid import.
"""

from vllm_omni.platforms import current_omni_platform

if current_omni_platform.is_cuda():
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import ResnetBlock
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock
else:
    # NPU, ROCm, XPU, CPU, out-of-tree: plain PyTorch blocks.
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder_blocks import ResnetBlock
    from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import ResBlock

__all__ = ["ResnetBlock", "ResBlock"]
