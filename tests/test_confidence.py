import math

from src.confidence import (
    bottom_fraction_mean,
    compute_group_confidences,
    compute_token_confidences,
    compute_trace_confidence,
    tail_mean,
)


def test_token_confidence_from_top_k() -> None:
    tokens = [
        {"logprob": -3.0, "top_logprobs": {"a": -1.0, "b": -2.0}},
        {"logprob": -4.0, "top_logprobs": {"a": -2.0, "b": -2.0}},
    ]
    confidences = compute_token_confidences(tokens)
    assert confidences == [1.5, 2.0]


def test_token_confidence_fallback_to_chosen_logprob() -> None:
    tokens = [{"logprob": -3.0, "top_logprobs": {}}]
    confidences = compute_token_confidences(tokens)
    assert confidences == [3.0]


def test_token_confidence_ignores_sentinel_values() -> None:
    tokens = [
        {
            "logprob": -3.0,
            "top_logprobs": {"good": -1.0, "also_good": -2.0, "bad": -9999.0},
        }
    ]
    confidences = compute_token_confidences(tokens)
    # Mean should be computed over [-1.0, -2.0], not including -9999.0.
    assert confidences == [1.5]


def test_group_confidence_small_window() -> None:
    token_confs = [1.0, 2.0, 3.0]
    groups = compute_group_confidences(token_confs, group_size=2)
    # windows: [1], [1,2], [2,3]
    assert groups == [1.0, 1.5, 2.5]


def test_bottom_fraction_mean() -> None:
    values = [5.0, 1.0, 3.0, 2.0]
    result = bottom_fraction_mean(values, fraction=0.25)
    # ceil(0.25*4)=1 -> bottom is [1.0]
    assert result == 1.0


def test_tail_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert tail_mean(values, tail_tokens=2) == 3.5


def test_trace_confidence_lowest_group() -> None:
    tokens = [
        {"logprob": -1.0, "top_logprobs": {"a": -1.0}},
        {"logprob": -2.0, "top_logprobs": {"a": -2.0}},
        {"logprob": -4.0, "top_logprobs": {"a": -4.0}},
    ]
    # token confs: [1,2,4]
    # group_size=2 -> groups [1,1.5,3]
    result = compute_trace_confidence(
        tokens,
        metric="lowest_group",
        group_size=2,
        tail_tokens=2,
        bottom_fraction=0.1,
    )
    assert math.isclose(result, 1.0)


def test_trace_confidence_bottom10_group() -> None:
    tokens = [
        {"logprob": -1.0, "top_logprobs": {"a": -1.0}},
        {"logprob": -2.0, "top_logprobs": {"a": -2.0}},
        {"logprob": -3.0, "top_logprobs": {"a": -3.0}},
        {"logprob": -4.0, "top_logprobs": {"a": -4.0}},
    ]
    result = compute_trace_confidence(
        tokens,
        metric="bottom10_group",
        group_size=2,
        tail_tokens=2,
        bottom_fraction=0.25,
    )
    assert result == 1.0
