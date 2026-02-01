from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np

LOGPROB_SENTINEL_THRESHOLD = -1000.0


@dataclass(slots=True)
class ConfidenceInputs:
    token_confidences: list[float]
    group_confidences: list[float]


def token_confidence(top_logprobs: dict[str, float] | None, chosen_logprob: float | None) -> float:
    """Paper Eq. 2: negative mean of top-k token logprobs."""

    if top_logprobs:
        # DeepSeek can return sentinel values like -9999.0 for tokens outside the
        # requested top-k set. These would dominate the mean, so we filter them.
        valid_values = [
            float(value)
            for value in top_logprobs.values()
            if value is not None and float(value) > LOGPROB_SENTINEL_THRESHOLD
        ]
        if valid_values:
            return float(-np.mean(valid_values))
    if chosen_logprob is not None:
        return float(-chosen_logprob)
    return 0.0


def _extract_token_fields(token: object) -> tuple[dict[str, float] | None, float | None]:
    if hasattr(token, "top_logprobs") and hasattr(token, "logprob"):
        top = getattr(token, "top_logprobs")
        logprob = getattr(token, "logprob")
        return (top if isinstance(top, dict) else None, float(logprob) if logprob is not None else None)
    if isinstance(token, dict):
        top = token.get("top_logprobs")
        logprob = token.get("logprob")
        top_dict = top if isinstance(top, dict) else None
        logprob_value = float(logprob) if logprob is not None else None
        return top_dict, logprob_value
    return None, None


def compute_token_confidences(tokens: Sequence[object]) -> list[float]:
    confidences: list[float] = []
    for token in tokens:
        top_logprobs, chosen_logprob = _extract_token_fields(token)
        confidences.append(token_confidence(top_logprobs, chosen_logprob))
    return confidences


def compute_group_confidences(token_confidences: Sequence[float], group_size: int) -> list[float]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if not token_confidences:
        return []

    groups: list[float] = []
    for idx in range(len(token_confidences)):
        start = max(0, idx - group_size + 1)
        window = token_confidences[start : idx + 1]
        groups.append(float(np.mean(window)))
    return groups


def bottom_fraction_mean(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")
    count = max(1, ceil(fraction * len(values)))
    bottom = sorted(values)[:count]
    return float(np.mean(bottom))


def tail_mean(values: Sequence[float], tail_tokens: int) -> float:
    if tail_tokens <= 0:
        raise ValueError("tail_tokens must be positive.")
    if not values:
        return 0.0
    tail = values[-tail_tokens:]
    return float(np.mean(tail))


def compute_confidence_inputs(tokens: Sequence[object], group_size: int) -> ConfidenceInputs:
    token_confs = compute_token_confidences(tokens)
    group_confs = compute_group_confidences(token_confs, group_size=group_size)
    return ConfidenceInputs(token_confidences=token_confs, group_confidences=group_confs)


def compute_trace_confidence(
    tokens: Sequence[object],
    *,
    metric: str,
    group_size: int,
    tail_tokens: int,
    bottom_fraction: float,
) -> float:
    """Compute a trace-level confidence score using paper-aligned metrics."""

    inputs = compute_confidence_inputs(tokens, group_size=group_size)
    token_confs = inputs.token_confidences
    group_confs = inputs.group_confidences

    if metric == "avg":
        return float(np.mean(token_confs)) if token_confs else 0.0
    if metric == "bottom10_group":
        return bottom_fraction_mean(group_confs, fraction=bottom_fraction)
    if metric == "lowest_group":
        return min(group_confs) if group_confs else 0.0
    if metric == "tail":
        return tail_mean(token_confs, tail_tokens=tail_tokens)

    raise ValueError(f"Unknown confidence metric: {metric}")


def batch_trace_confidence(
    traces: Iterable[Sequence[object]],
    *,
    metric: str,
    group_size: int,
    tail_tokens: int,
    bottom_fraction: float,
) -> list[float]:
    return [
        compute_trace_confidence(
            trace,
            metric=metric,
            group_size=group_size,
            tail_tokens=tail_tokens,
            bottom_fraction=bottom_fraction,
        )
        for trace in traces
    ]
