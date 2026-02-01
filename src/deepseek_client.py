from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass(slots=True)
class TokenLogprob:
    token: str
    logprob: float | None
    top_logprobs: dict[str, float]


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    tokens: list[TokenLogprob]
    finish_reason: str | None
    raw: dict[str, Any]


class DeepSeekClientError(RuntimeError):
    pass


class MissingLogprobsError(DeepSeekClientError):
    pass


class RetryableStatusError(DeepSeekClientError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Retryable status {status_code}: {message}")
        self.status_code = status_code


def _coerce_top_logprobs(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, list):
        result: dict[str, float] = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            token = entry.get("token")
            logprob = entry.get("logprob")
            if token is None or logprob is None:
                continue
            result[str(token)] = float(logprob)
        return result
    return {}


def _parse_logprobs(logprobs_payload: Any) -> list[TokenLogprob]:
    if not isinstance(logprobs_payload, dict):
        return []

    # OpenAI-style chat logprobs: {"content": [{"token": ..., "logprob": ..., "top_logprobs": [...]}, ...]}
    content = logprobs_payload.get("content")
    if isinstance(content, list):
        tokens: list[TokenLogprob] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", ""))
            logprob = item.get("logprob")
            logprob_value = float(logprob) if logprob is not None else None
            top_logprobs = _coerce_top_logprobs(item.get("top_logprobs"))
            tokens.append(TokenLogprob(token=token, logprob=logprob_value, top_logprobs=top_logprobs))
        return tokens

    # Legacy-style: tokens/token_logprobs/top_logprobs lists
    tokens_list = logprobs_payload.get("tokens")
    logprob_list = logprobs_payload.get("token_logprobs")
    top_list = logprobs_payload.get("top_logprobs")
    if isinstance(tokens_list, list) and isinstance(logprob_list, list):
        tokens: list[TokenLogprob] = []
        for idx, token in enumerate(tokens_list):
            logprob_value = None
            if idx < len(logprob_list) and logprob_list[idx] is not None:
                logprob_value = float(logprob_list[idx])
            top_logprobs = {}
            if isinstance(top_list, list) and idx < len(top_list):
                top_logprobs = _coerce_top_logprobs(top_list[idx])
            tokens.append(
                TokenLogprob(
                    token=str(token),
                    logprob=logprob_value,
                    top_logprobs=top_logprobs,
                )
            )
        return tokens

    return []


def parse_chat_completion(payload: dict[str, Any], *, require_logprobs: bool = True) -> ChatCompletionResult:
    """Parse a chat completion payload into a normalized result."""

    if "choices" not in payload or not isinstance(payload["choices"], list) or not payload["choices"]:
        raise DeepSeekClientError("Response payload is missing a non-empty 'choices' list.")

    choice = payload["choices"][0]
    if not isinstance(choice, dict):
        raise DeepSeekClientError("First choice is not an object.")

    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise DeepSeekClientError("Response choice is missing message.content.")

    text = str(message.get("content", ""))
    finish_reason_value = choice.get("finish_reason")
    finish_reason = str(finish_reason_value) if finish_reason_value is not None else None

    tokens = _parse_logprobs(choice.get("logprobs"))
    if require_logprobs and not tokens:
        raise MissingLogprobsError(
            "Logprobs are required but missing. Ensure logprobs=true and top_logprobs is set."
        )

    return ChatCompletionResult(text=text, tokens=tokens, finish_reason=finish_reason, raw=payload)


class DeepSeekClient:
    """Small DeepSeek chat completions client with retry logic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout_seconds: float = 60.0,
    ) -> None:
        resolved_key = api_key or os.getenv(api_key_env)
        if not resolved_key:
            raise DeepSeekClientError(
                f"Missing API key. Set {api_key_env} in the environment or pass api_key explicitly."
            )

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((requests.RequestException, RetryableStatusError)),
    )
    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        logprobs: bool,
        top_logprobs: int,
        require_logprobs: bool = True,
    ) -> ChatCompletionResult:
        url = self._chat_completions_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            # Logprobs are not supported in thinking mode; force it off.
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

        response = self._session.post(url, headers=headers, json=body, timeout=self.timeout_seconds)

        if response.status_code in {429, 500, 502, 503, 504}:
            raise RetryableStatusError(response.status_code, response.text[:200])

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise DeepSeekClientError(f"DeepSeek API error {response.status_code}: {response.text[:400]}") from exc

        payload = response.json()
        return parse_chat_completion(payload, require_logprobs=require_logprobs)
