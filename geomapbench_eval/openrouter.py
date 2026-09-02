from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float | None = 0.0
    max_tokens: int = 16384
    timeout_seconds: int = 240
    retries: int = 6
    reasoning_effort: str | None = None
    reasoning_enabled: bool | None = False
    request_delay_seconds: float = 0.0
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 60.0


class OpenRouterHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"OpenRouter HTTP {status}: {body[:1500]}")
        self.status = status
        self.body = body


class OpenRouterRetryExhausted(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class OpenRouterClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    models_endpoint = "https://openrouter.ai/api/v1/models"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required; do not put it in a source file.")
        self._last_request_finished = 0.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/asalmeskin/GeoMapBench",
            "X-Title": "GeoMapBench",
        }

    def _pace(self, seconds: float) -> None:
        remaining = seconds - (time.monotonic() - self._last_request_finished)
        if remaining > 0:
            print(f"[openrouter] pacing next request for {remaining:.1f}s", flush=True)
            time.sleep(remaining)

    def complete(self, messages: list[dict[str, Any]], config: OpenRouterConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.reasoning_enabled is not None:
            reasoning: dict[str, Any] = {
                "enabled": bool(config.reasoning_enabled),
                "exclude": True,
            }
            if config.reasoning_enabled and config.reasoning_effort:
                reasoning["effort"] = config.reasoning_effort
            payload["reasoning"] = reasoning
        elif config.reasoning_effort:
            payload["reasoning"] = {"effort": config.reasoning_effort, "exclude": True}

        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        last_status: int | None = None
        for attempt in range(config.retries + 1):
            self._pace(config.request_delay_seconds)
            started = time.monotonic()
            try:
                request = urllib.request.Request(
                    self.endpoint, data=data, headers=self._headers(), method="POST"
                )
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                result = json.loads(raw)
                result["_latency_seconds"] = round(time.monotonic() - started, 3)
                result["_transport_attempts"] = attempt + 1
                self._last_request_finished = time.monotonic()
                return result
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                wrapped = OpenRouterHTTPError(error.code, body)
                last_error = wrapped
                last_status = error.code
                retryable = error.code in {408, 409, 425, 429} or 500 <= error.code < 600
                retry_after = error.headers.get("Retry-After")
                if not retryable:
                    raise wrapped from error
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as error:
                last_error = error
                last_status = None
                retry_after = None
            finally:
                self._last_request_finished = time.monotonic()

            if attempt == config.retries:
                break
            if retry_after and retry_after.replace(".", "", 1).isdigit():
                delay = min(config.retry_max_seconds, float(retry_after))
            else:
                delay = min(
                    config.retry_max_seconds,
                    config.retry_base_seconds * (2.0 ** attempt),
                )
                delay *= random.uniform(0.85, 1.15)
            print(
                f"[openrouter:retry] model={config.model} attempt={attempt + 1}/{config.retries + 1} "
                f"status={last_status or 'network'} waiting={delay:.1f}s error={last_error!r}",
                flush=True,
            )
            time.sleep(delay)

        raise OpenRouterRetryExhausted(
            f"OpenRouter request failed after {config.retries + 1} attempts: {last_error}",
            status=last_status,
        )

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
