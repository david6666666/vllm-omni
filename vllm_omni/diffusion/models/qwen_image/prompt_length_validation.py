from typing import Any


def build_qwen_image_edit_plus_prompt_prefix(image_count: int) -> str:
    """Build the text prefix used by Qwen image edit-plus for image placeholders."""
    if image_count <= 0:
        return ""

    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    return "".join(img_prompt_template.format(i + 1) for i in range(image_count))


def tokenize_and_validate_qwen_text_prompt(
    tokenizer: Any,
    *,
    texts: list[str],
    drop_idx: int,
    max_sequence_length: int | None,
    field_name: str,
):
    """Tokenize a text-only Qwen prompt and fail fast if it exceeds the limit."""
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    if max_sequence_length is None:
        return tokenized

    token_lengths = tokenized.attention_mask.sum(dim=1).tolist()
    effective_lengths = [max(int(length) - drop_idx, 0) for length in token_lengths]
    for effective_length in effective_lengths:
        if effective_length > max_sequence_length:
            raise ValueError(
                f"`{field_name}` is too long after Qwen image prompt expansion: "
                f"{effective_length} tokens exceeds max_sequence_length={max_sequence_length}."
            )

    return tokenized


def tokenize_and_validate_qwen_vl_prompt(
    processor: Any,
    *,
    texts: list[str],
    images: Any,
    drop_idx: int,
    max_sequence_length: int | None,
    field_name: str,
):
    """Tokenize a Qwen VL prompt and fail fast if it exceeds the requested limit."""
    model_inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    if max_sequence_length is None:
        return model_inputs

    token_lengths = model_inputs.attention_mask.sum(dim=1).tolist()
    effective_lengths = [max(int(length) - drop_idx, 0) for length in token_lengths]
    for effective_length in effective_lengths:
        if effective_length > max_sequence_length:
            raise ValueError(
                f"`{field_name}` is too long after Qwen image prompt expansion: "
                f"{effective_length} tokens exceeds max_sequence_length={max_sequence_length}."
            )

    return model_inputs
