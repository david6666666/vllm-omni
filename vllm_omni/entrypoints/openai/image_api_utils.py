# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible image generation API.

This module provides common helper functions for the image generation endpoint.
All functions work with plain Python types to maintain separation from the
FastAPI HTTP layer.
"""

import base64
import io
from typing import Any

import PIL.Image

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(3, 11)


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


def resolve_max_input_images(od_config: Any | None) -> int | None:
    """Resolve the maximum number of input images supported by a model."""
    if od_config is None:
        return None

    max_input_images = getattr(od_config, "max_input_images", None)
    if max_input_images is not None:
        return int(max_input_images)

    if not bool(getattr(od_config, "supports_multimodal_inputs", False)):
        return 1

    return None


def get_input_image_limit_error(input_image_count: int, max_input_images: int | None) -> str | None:
    """Return a user-facing validation error when the model limit is exceeded."""
    if max_input_images is None or input_image_count <= max_input_images:
        return None

    if max_input_images == 1:
        return "Received multiple input images. Only a single image is supported by this model."

    return (
        f"Received {input_image_count} input images. "
        f"This model supports at most {max_input_images} input images."
    )


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
