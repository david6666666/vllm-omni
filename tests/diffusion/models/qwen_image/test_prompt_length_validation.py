from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.qwen_image.prompt_length_validation import (
    build_qwen_image_edit_plus_prompt_prefix,
    tokenize_and_validate_qwen_text_prompt,
    tokenize_and_validate_qwen_vl_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeProcessor:
    def __init__(self, lengths: list[int]):
        self._lengths = lengths

    def __call__(self, *, text, images, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert isinstance(text, list)
        return SimpleNamespace(attention_mask=FakeAttentionMask(self._lengths))


class FakeTokenizer:
    def __init__(self, lengths: list[int]):
        self._lengths = lengths

    def __call__(self, texts, padding, truncation, return_tensors):
        assert isinstance(texts, list)
        assert padding is True
        assert truncation is False
        assert return_tensors == "pt"
        return SimpleNamespace(attention_mask=FakeAttentionMask(self._lengths))


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


def test_build_qwen_image_edit_plus_prompt_prefix():
    assert build_qwen_image_edit_plus_prompt_prefix(0) == ""
    assert build_qwen_image_edit_plus_prompt_prefix(1) == "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    assert build_qwen_image_edit_plus_prompt_prefix(2) == (
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 2: <|vision_start|><|image_pad|><|vision_end|>"
    )


def test_tokenize_and_validate_qwen_text_prompt_accepts_valid_length():
    tokenizer = FakeTokenizer(lengths=[66])

    result = tokenize_and_validate_qwen_text_prompt(
        tokenizer,
        texts=["hello"],
        drop_idx=34,
        max_sequence_length=32,
        field_name="prompt",
    )

    assert isinstance(result, SimpleNamespace)


def test_tokenize_and_validate_qwen_text_prompt_rejects_overlong_prompt():
    tokenizer = FakeTokenizer(lengths=[1059])

    with pytest.raises(ValueError, match="`prompt` is too long"):
        tokenize_and_validate_qwen_text_prompt(
            tokenizer,
            texts=["hello"],
            drop_idx=34,
            max_sequence_length=1024,
            field_name="prompt",
        )


def test_tokenize_and_validate_qwen_vl_prompt_accepts_valid_length():
    processor = FakeProcessor(lengths=[96])

    result = tokenize_and_validate_qwen_vl_prompt(
        processor,
        texts=["hello"],
        images=None,
        drop_idx=64,
        max_sequence_length=32,
        field_name="prompt",
    )

    assert isinstance(result, SimpleNamespace)


def test_tokenize_and_validate_qwen_vl_prompt_rejects_overlong_prompt():
    processor = FakeProcessor(lengths=[1164])

    with pytest.raises(ValueError, match="`prompt` is too long"):
        tokenize_and_validate_qwen_vl_prompt(
            processor,
            texts=["hello"],
            images=None,
            drop_idx=64,
            max_sequence_length=1024,
            field_name="prompt",
        )


def test_tokenize_and_validate_qwen_vl_prompt_mentions_negative_prompt():
    processor = FakeProcessor(lengths=[90, 1165])

    with pytest.raises(ValueError, match="`negative_prompt` is too long"):
        tokenize_and_validate_qwen_vl_prompt(
            processor,
            texts=["a", "b"],
            images=None,
            drop_idx=64,
            max_sequence_length=1024,
            field_name="negative_prompt",
        )
