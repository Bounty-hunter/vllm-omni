# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 tests for the image response encoder's compression contract."""

import base64
import io

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64_with_compression,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def image():
    # Photographic-content proxy: noise-plus-gradient content compresses very
    # differently across PNG compress levels, which is what the mapping under
    # test controls.
    import numpy as np

    rng = np.random.default_rng(0)
    grad = np.linspace(0, 255, 256, dtype=np.float32)
    arr = np.clip(grad[None, :, None] + rng.integers(-24, 25, (256, 256, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_png_output_compression_maps_to_compress_level(image):
    """output_compression 100 -> compress_level 0, 1 -> compress_level 9."""

    def encode(compression):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=max(0, min(9, 9 - compression // 11)))
        return len(base64.b64encode(buffer.getvalue()))

    fast = encode_image_base64_with_compression(image, format="png", output_compression=100)
    small = encode_image_base64_with_compression(image, format="png", output_compression=1)

    # Level-0 PNG barely compresses; the mapped level-9 encode is smaller.
    assert len(small) < len(fast)
    # Both encodings must remain lossless. The level-0 fast path intentionally
    # emits filter-none scanlines, so its encoded byte length need not match
    # Pillow's encoder.
    decoded = Image.open(io.BytesIO(base64.b64decode(fast)))
    assert decoded.mode == image.mode
    assert decoded.size == image.size
    assert decoded.tobytes() == image.tobytes()
    assert len(small) == encode(1)


def test_png_fast_path_preserves_rgb_pixels(image):
    encoded = encode_image_base64_with_compression(image, format="png", output_compression=99)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))

    assert decoded.mode == "RGB"
    assert decoded.tobytes() == image.tobytes()


def test_png_non_rgb_uses_compatible_fallback(image):
    rgba = image.convert("RGBA")
    encoded = encode_image_base64_with_compression(rgba, format="png", output_compression=100)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))

    assert decoded.mode == "RGBA"
    assert decoded.tobytes() == rgba.tobytes()


def test_jpeg_output_compression_maps_to_quality(image):
    high = base64.b64decode(encode_image_base64_with_compression(image, format="jpeg", output_compression=100))
    low = base64.b64decode(encode_image_base64_with_compression(image, format="jpeg", output_compression=10))

    assert len(high) > len(low)
