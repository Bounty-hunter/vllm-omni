# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    _preprocess_pil_image_on_device,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("image", "height", "width"),
    (
        (Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)), 32, 48),
        (Image.fromarray(np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)), 32, 48),
        (Image.fromarray(np.arange(17 * 19, dtype=np.uint8).reshape(17, 19)), 32, 48),
        (Image.fromarray(np.arange(17 * 19 * 4, dtype=np.uint8).reshape(17, 19, 4)), 32, 48),
    ),
    ids=("rgb", "rgb-resized", "grayscale", "rgba"),
)
def test_pil_preprocessing_matches_diffusers(image, height, width):
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True)

    expected = image_processor.preprocess(image, height, width)
    actual = _preprocess_pil_image_on_device(image, image_processor, height, width, torch.device("cpu"))

    assert actual.shape == (1, 3, height, width)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=torch.finfo(torch.float32).eps)
    assert torch.equal(actual.to(torch.bfloat16), expected.to(torch.bfloat16))
