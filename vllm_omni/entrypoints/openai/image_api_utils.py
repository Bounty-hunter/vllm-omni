# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Shared media utilities for OpenAI-compatible image APIs.

PUT HERE:
  - Pure media helpers with no FastAPI Request / app-state / job semantics:
    size parsing, image encode/base64, layered-layer validation.

DO NOT PUT HERE:
  - ``/v1/images*`` request/engine/job helpers — those go in ``images.helpers``.

LONGEVITY:
  - This root utils file is a **temporary shared home**.
  - TODO(#5227, P1.2): tidy up / absorb into the images family (e.g. ``images/media.py``
    or equivalent) in the P1.2 image modality PR; do not treat this file as the
    long-term owner.
  - ``images.helpers`` is the longer home for endpoint-specific logic through
    P0.2/P0.3 until P1.2 further splits it.

See ``openai/README.md`` and ``images/README.md`` (utils vs helpers, no overlap).
"""

import io
import struct
import zlib

import base64

import PIL.Image

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(2, 11)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data))


def _encode_rgb_png_level_zero(image: PIL.Image.Image) -> bytes:
    """Encode an 8-bit RGB image as a filter-none, zlib-level-0 PNG."""
    width, height = image.size
    row_bytes = width * 3
    pixels = image.tobytes()
    scanlines = bytearray((row_bytes + 1) * height)
    source = memoryview(pixels)
    target = memoryview(scanlines)
    for row in range(height):
        source_start = row * row_bytes
        target_start = row * (row_bytes + 1) + 1
        target[target_start : target_start + row_bytes] = source[source_start : source_start + row_bytes]

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=0)),
            _png_chunk(b"IEND", b""),
        )
    )


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024'). "
            f"Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def encode_image_base64(image: PIL.Image.Image) -> str:
    """Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG image as string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def prepare_image_for_output_format(image: PIL.Image.Image, format: str) -> PIL.Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = PIL.Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def encode_image_base64_with_compression(
    image: PIL.Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode a PIL image to a base64 image string.

    Args:
        image: PIL Image object.
        format: Output image format, such as "png", "jpeg", or "webp".
        output_compression: Compression level (0-100), where 100 keeps the
            best quality for lossy formats and uses the lowest PNG compression.

    Returns:
        Base64-encoded image string.
    """
    image = prepare_image_for_output_format(image, format)
    pil_format = "jpeg" if format == "jpg" else format
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))

    if pil_format == "png" and image.mode == "RGB" and save_kwargs["compress_level"] == 0:
        return base64.b64encode(_encode_rgb_png_level_zero(image)).decode("ascii")

    buffer = io.BytesIO()
    image.save(buffer, format=pil_format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def validate_layered_layers(layers: int | None) -> int | None:
    """Validate the Qwen-Image-Layered ``layers`` parameter."""
    if layers is None:
        return None
    if layers not in SUPPORTED_LAYERED_LAYERS_RANGE:
        raise ValueError(
            f"Invalid layers value {layers}. layers must be between "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.start} and "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.stop - 1} inclusive."
        )
    return layers
