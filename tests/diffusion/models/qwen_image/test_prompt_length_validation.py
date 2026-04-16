import pytest

from vllm_omni.diffusion.models.qwen_image.prompt_length_validation import (
    get_qwen_text_prompt_lengths,
    validate_qwen_text_prompt_lengths,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeAttentionMask:
    def __init__(self, lengths: list[int]):
        self._lengths = lengths

    def sum(self, dim: int):
        assert dim == 1
        return FakeTensor(self._lengths)


class FakeTensor:
    def __init__(self, values: list[int]):
        self._values = values

    def tolist(self):
        return self._values


def test_get_qwen_text_prompt_lengths():
    attention_mask = FakeAttentionMask([66, 1059, 20])
    assert get_qwen_text_prompt_lengths(attention_mask) == [66, 1059, 20]


def test_validate_qwen_text_prompt_lengths_accepts_valid_length():
    validate_qwen_text_prompt_lengths(
        [32],
        max_sequence_length=32,
        field_name="prompt",
    )


def test_validate_qwen_text_prompt_lengths_rejects_overlong_prompt():
    with pytest.raises(ValueError, match="`prompt` is too long"):
        validate_qwen_text_prompt_lengths(
            [1025],
            max_sequence_length=1024,
            field_name="prompt",
        )


def test_validate_qwen_text_prompt_lengths_mentions_negative_prompt():
    with pytest.raises(ValueError, match="`negative_prompt` is too long"):
        validate_qwen_text_prompt_lengths(
            [26, 1101],
            max_sequence_length=1024,
            field_name="negative_prompt",
        )
