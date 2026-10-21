# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    _encode_vae_image_with_cache,
    _pil_image_preprocess_cache_key,
    _preprocess_pil_image_with_cache,
)


def test_reuses_exact_rgb_preprocessing_result():
    processor = Mock()
    processor.preprocess.side_effect = lambda _image, height, width: torch.full(
        (1, 3, height, width),
        len(processor.preprocess.mock_calls),
        dtype=torch.float32,
    )
    cache = {}
    image = PIL.Image.new("RGB", (8, 8), (1, 2, 3))

    first = _preprocess_pil_image_with_cache(image, processor, 16, 16, torch.device("cpu"), cache)
    second = _preprocess_pil_image_with_cache(image.copy(), processor, 16, 16, torch.device("cpu"), cache)

    assert second is first
    processor.preprocess.assert_called_once()


def test_cache_key_tracks_pixels_and_target_size():
    processor = Mock()
    processor.preprocess.side_effect = lambda _image, height, width: torch.empty((1, 3, height, width))
    cache = {}

    _preprocess_pil_image_with_cache(
        PIL.Image.new("RGB", (8, 8), (1, 2, 3)), processor, 16, 16, torch.device("cpu"), cache
    )
    _preprocess_pil_image_with_cache(
        PIL.Image.new("RGB", (8, 8), (3, 2, 1)), processor, 16, 16, torch.device("cpu"), cache
    )
    _preprocess_pil_image_with_cache(
        PIL.Image.new("RGB", (8, 8), (1, 2, 3)), processor, 32, 16, torch.device("cpu"), cache
    )

    assert processor.preprocess.call_count == 3


def test_non_rgb_input_is_not_cached():
    processor = Mock()
    processor.preprocess.return_value = torch.empty((1, 3, 16, 16))
    cache = {}
    image = PIL.Image.new("RGBA", (8, 8), (1, 2, 3, 4))

    _preprocess_pil_image_with_cache(image, processor, 16, 16, torch.device("cpu"), cache)
    _preprocess_pil_image_with_cache(image, processor, 16, 16, torch.device("cpu"), cache)

    assert processor.preprocess.call_count == 2
    assert not cache


def test_vae_posterior_cache_reuses_encode_but_samples_each_request():
    parameters = torch.randn(1, 8, 4, 4)
    vae = Mock()
    vae.encode.side_effect = lambda _image: SimpleNamespace(
        latent_dist=DiagonalGaussianDistribution(parameters.clone())
    )
    image = torch.empty(1, 3, 16, 16)
    pil_image = PIL.Image.new("RGB", (8, 8), (1, 2, 3))
    key = _pil_image_preprocess_cache_key(pil_image, 16, 16, torch.device("cpu"))
    cache = {}

    first_generator = torch.Generator().manual_seed(7)
    first = _encode_vae_image_with_cache(vae, image, first_generator, key, cache)
    first_reference = DiagonalGaussianDistribution(parameters).sample(torch.Generator().manual_seed(7))

    second_generator = torch.Generator().manual_seed(11)
    second = _encode_vae_image_with_cache(vae, image, second_generator, key, cache)
    second_reference = DiagonalGaussianDistribution(parameters).sample(torch.Generator().manual_seed(11))

    torch.testing.assert_close(first, first_reference, rtol=0, atol=0)
    torch.testing.assert_close(second, second_reference, rtol=0, atol=0)
    assert vae.encode.call_count == 1
    assert not torch.equal(first, second)


def test_vae_posterior_cache_is_disabled_without_exact_pil_key():
    parameters = torch.randn(1, 8, 4, 4)
    vae = Mock()
    vae.encode.side_effect = lambda _image: SimpleNamespace(
        latent_dist=DiagonalGaussianDistribution(parameters.clone())
    )
    image = torch.empty(1, 3, 16, 16)
    cache = {}

    _encode_vae_image_with_cache(vae, image, torch.Generator().manual_seed(1), None, cache)
    _encode_vae_image_with_cache(vae, image, torch.Generator().manual_seed(2), None, cache)

    assert vae.encode.call_count == 2
    assert not cache
