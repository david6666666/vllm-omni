# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared size normalization helpers for the Qwen-Image family."""


def normalize_qwen_image_size(height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    """Clamp dimensions to the Qwen-Image minimum valid aligned size.

    Qwen-Image packs image latents in 2x2 blocks after VAE compression, so request
    dimensions must be aligned to ``vae_scale_factor * 2``. Very small requests
    such as ``1x1`` would otherwise floor to ``0x0`` in downstream latent shape
    computations and crash reshape/view operations.
    """

    multiple_of = int(vae_scale_factor) * 2
    if multiple_of <= 0:
        raise ValueError(f"Expected positive vae_scale_factor, got {vae_scale_factor}")

    normalized_height = max(multiple_of, (int(height) // multiple_of) * multiple_of)
    normalized_width = max(multiple_of, (int(width) // multiple_of) * multiple_of)
    return normalized_height, normalized_width
