from typing import Any


def get_qwen_text_prompt_lengths(attention_mask: Any) -> list[int]:
    """Return token lengths for user-provided text prompts."""
    return [int(length) for length in attention_mask.sum(dim=1).tolist()]


def validate_qwen_text_prompt_lengths(
    lengths: list[int],
    *,
    max_sequence_length: int | None,
    field_name: str,
) -> None:
    """Raise when any user-provided text prompt exceeds the configured maximum."""
    if max_sequence_length is None:
        return

    for length in lengths:
        if length > max_sequence_length:
            raise ValueError(
                f"`{field_name}` is too long: {length} tokens exceeds max_sequence_length={max_sequence_length}."
            )
