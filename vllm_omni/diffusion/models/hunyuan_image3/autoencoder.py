# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HunyuanImage3 Autoencoder (main model, hardware-agnostic).

This file contains the main AutoencoderKLConv3D model, which uses building blocks
(Encoder, Decoder, etc.) that are imported from hardware-specific implementations
via __init__.py dispatch.

Building blocks are located in:
- autoencoder_blocks.py (default/fallback)
- nvidia/autoencoder_blocks.py (CUDA optimized)
- npu/autoencoder_blocks.py (NPU optimized, Phase 2)
"""

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

# Import hardware-specific building blocks (dispatched in __init__.py)
from vllm_omni.diffusion.models.hunyuan_image3 import (
    Encoder,
    Decoder,
)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: DiagonalGaussianDistribution | None = None


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    """
    Autoencoder model with KL-regularized latent space based on 3D convolutions.
    
    This is the main model class (hardware-agnostic). It uses Encoder/Decoder
    building blocks that are automatically selected based on hardware platform.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int = 4,
        add_attention: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        # Encoder uses hardware-specific blocks
        self.encoder = Encoder(
            z_channels=latent_channels,
            ch=block_out_channels[0],
            ch_mult=tuple([ch // block_out_channels[0] for ch in block_out_channels]),
            num_res_blocks=layers_per_block,
            resolution=256,
            in_channels=in_channels,
            use_checkpointing=use_checkpointing,
            add_temporal_downsample=[True, True, False] if ffactor_temporal == 4 else [True, True, True],
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)

        # Decoder uses hardware-specific blocks
        self.decoder = Decoder(
            z_channels=latent_channels,
            ch=block_out_channels[0],
            ch_mult=tuple([ch // block_out_channels[0] for ch in block_out_channels]),
            num_res_blocks=layers_per_block,
            resolution=256,
            out_ch=out_channels,
            use_checkpointing=use_checkpointing,
            add_temporal_upsample=[True, True, False] if ffactor_temporal == 4 else [True, True, True],
        )

        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)

        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal

    def _set_gradient_checkpointing(self, module, value=False):
        self.encoder.use_checkpointing = value
        self.decoder.use_checkpointing = value

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> AutoencoderKLOutput | tuple:
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mean
        dec = self.decode(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec, posterior=posterior)


__all__ = [
    "AutoencoderKLConv3D",
    "DecoderOutput",
    "DiagonalGaussianDistribution",
]
