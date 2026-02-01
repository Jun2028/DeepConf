import pytest

from src.deepseek_client import MissingLogprobsError, parse_chat_completion


def test_parse_openai_style_logprobs() -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "hello"},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token": "he",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "he", "logprob": -0.1},
                                {"token": "ha", "logprob": -1.0},
                            ],
                        },
                        {
                            "token": "llo",
                            "logprob": -0.2,
                            "top_logprobs": {"llo": -0.2, "ll": -1.2},
                        },
                    ]
                },
            }
        ]
    }
    result = parse_chat_completion(payload, require_logprobs=True)
    assert result.text == "hello"
    assert result.finish_reason == "length"
    assert len(result.tokens) == 2
    assert result.tokens[0].top_logprobs["he"] == -0.1
    assert result.tokens[1].top_logprobs["llo"] == -0.2


def test_parse_legacy_style_logprobs() -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "hi"},
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["h", "i"],
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [
                        {"h": -0.1, "a": -1.0},
                        {"i": -0.2, "o": -1.1},
                    ],
                },
            }
        ]
    }
    result = parse_chat_completion(payload, require_logprobs=True)
    assert result.text == "hi"
    assert result.finish_reason == "stop"
    assert [tok.token for tok in result.tokens] == ["h", "i"]
    assert result.tokens[0].logprob == -0.1


def test_missing_logprobs_raises_when_required() -> None:
    payload = {"choices": [{"message": {"content": "hi"}}]}
    with pytest.raises(MissingLogprobsError):
        parse_chat_completion(payload, require_logprobs=True)


def test_missing_logprobs_allowed_when_not_required() -> None:
    payload = {"choices": [{"message": {"content": "hi"}}]}
    result = parse_chat_completion(payload, require_logprobs=False)
    assert result.text == "hi"
    assert result.tokens == []
