import pytest

from vllm_omni.diffusion.models.qwen_image.size_utils import (
    normalize_qwen_image_size,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (1, 1, (16, 16)),
        (15, 15, (16, 16)),
        (17, 17, (16, 16)),
        (31, 33, (16, 32)),
        (64, 80, (64, 80)),
    ],
)
def test_normalize_qwen_image_size_clamps_to_minimum_aligned_shape(height, width, expected):
    assert normalize_qwen_image_size(height, width, vae_scale_factor=8) == expected


def test_normalize_qwen_image_size_rejects_invalid_scale_factor():
    with pytest.raises(ValueError, match="positive vae_scale_factor"):
        normalize_qwen_image_size(16, 16, vae_scale_factor=0)
