# SPDX-License-Identifier: Apache-2.0


class InputValidationError(ValueError):
    def __init__(self, message: str = "Invalid input.") -> None:
        super().__init__(message)


def get_serialized_error_type(exc: BaseException) -> str | None:
    if isinstance(exc, InputValidationError):
        return InputValidationError.__name__
    return None


def restore_serialized_error(message: str, error_type: str | None) -> Exception:
    if error_type == InputValidationError.__name__:
        return InputValidationError(message)
    return RuntimeError(message)
