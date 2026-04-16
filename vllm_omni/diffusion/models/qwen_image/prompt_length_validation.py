from typing import Any


def get_effective_qwen_prompt_lengths(attention_mask: Any, *, drop_idx: int) -> list[int]:
    """Return prompt lengths after removing the fixed system/template prefix."""
    token_lengths = attention_mask.sum(dim=1).tolist()
    return [max(int(length) - drop_idx, 0) for length in token_lengths]


def validate_qwen_prompt_lengths(
    effective_lengths: list[int],
    *,
    max_sequence_length: int | None,
    field_name: str,
) -> None:
    """Raise when any prompt length exceeds the configured maximum."""
    if max_sequence_length is None:
        return

    for effective_length in effective_lengths:
        if effective_length > max_sequence_length:
            raise ValueError(
                f"`{field_name}` is too long after Qwen image prompt expansion: "
                f"{effective_length} tokens exceeds max_sequence_length={max_sequence_length}."
            )
