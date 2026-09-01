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
    max_tokens: int = 512
    timeout_seconds: int = 120
    retries: int = 4


class OpenRouterClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required; do not put it in a source file.")

    def complete(self, messages: list[dict[str, Any]], config: OpenRouterConfig) -> dict[str, Any]:
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/GeoMapBench", "X-Title": "GeoMapBench"}
        last_error: Exception | None = None
        for attempt in range(config.retries + 1):
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                result = json.loads(raw)
                result["_latency_seconds"] = round(time.monotonic() - started, 3)
                return result
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                retryable = error.code in {408, 409, 429} or 500 <= error.code < 600
                if not retryable:
                    raise RuntimeError(f"OpenRouter HTTP {error.code}: {body[:1500]}") from error
                last_error = error
                if attempt == config.retries:
                    break
                retry_after = getattr(error, "headers", {}).get("Retry-After") if getattr(error, "headers", None) else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 ** attempt))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == config.retries:
                    break
                time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"OpenRouter request failed after {config.retries + 1} attempts: {last_error}")


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected OpenRouter response: {response}") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content)
