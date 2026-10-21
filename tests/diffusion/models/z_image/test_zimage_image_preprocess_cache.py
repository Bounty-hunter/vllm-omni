# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import PIL.Image
import torch

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
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
