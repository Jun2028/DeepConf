from __future__ import annotations

import re
from typing import Iterable

BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}", re.IGNORECASE | re.DOTALL)
INTEGER_PATTERN = re.compile(r"-?\d+")

AIME_MIN = 0
AIME_MAX = 999


def _last_int(values: Iterable[str]) -> int | None:
    last_value: int | None = None
    for value in values:
        last_value = int(value)
    return last_value


def _validate_range(value: int | None) -> int | None:
    if value is None:
        return None
    if AIME_MIN <= value <= AIME_MAX:
        return value
    return None


def extract_answer(text: str) -> int | None:
    """Extract an integer answer from model output.

    Priority order:
    1) The last \boxed{...} expression.
    2) The last integer anywhere in the text.
    """

    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        boxed_content = boxed_matches[-1]
        boxed_int = _last_int(INTEGER_PATTERN.findall(boxed_content))
        validated = _validate_range(boxed_int)
        if validated is not None:
            return validated

    fallback_int = _last_int(INTEGER_PATTERN.findall(text))
    return _validate_range(fallback_int)