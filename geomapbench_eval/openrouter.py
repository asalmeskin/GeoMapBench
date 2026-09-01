from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 180
    retries: int = 4
    reasoning_effort: str | None = "minimal"


class OpenRouterHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"OpenRouter HTTP {status}: {body[:1500]}")
        self.status = status
        self.body = body


class OpenRouterClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    models_endpoint = "https://openrouter.ai/api/v1/models"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required; do not put it in a source file.")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/asalmeskin/GeoMapBench",
            "X-Title": "GeoMapBench",
        }

    def complete(self, messages: list[dict[str, Any]], config: OpenRouterConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if config.reasoning_effort:
            payload["reasoning"] = {"effort": config.reasoning_effort, "exclude": True}
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(config.retries + 1):
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.endpoint, data=data, headers=self._headers(), method="POST")
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                result = json.loads(raw)
                result["_latency_seconds"] = round(time.monotonic() - started, 3)
                return result
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                wrapped = OpenRouterHTTPError(error.code, body)
                last_error = wrapped
                retryable = error.code in {408, 409, 429} or 500 <= error.code < 600
                if not retryable or attempt == config.retries:
                    raise wrapped from error
                retry_after = error.headers.get("Retry-After")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == config.retries:
                    break
                retry_after = None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 ** attempt)
            time.sleep(delay)
        raise RuntimeError(f"OpenRouter request failed after {config.retries + 1} attempts: {last_error}")

    def model_catalog(self) -> dict[str, dict[str, Any]]:
        request = urllib.request.Request(self.models_endpoint, headers=self._headers())
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {str(item.get("id")): item for item in payload.get("data", [])}


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected OpenRouter response: {response}") from error
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def finish_reason(response: dict[str, Any]) -> str | None:
    try:
        return response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def generation_failure(response: dict[str, Any], text: str) -> str | None:
    reason = finish_reason(response)
    if reason in {"length", "max_tokens"}:
        return "token_limit"
    if not text.strip():
        return "empty_response"
    return None
