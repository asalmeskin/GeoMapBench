from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import requests

from .common import atomic_write_json, sha256_file, slugify


USER_AGENT = "GeoMapRAGDataKit (https://github.com/asalmeskin/GeoMapBench; academic geospatial retrieval corpus construction)"


class CachedHTTP:
    def __init__(self, cache_root: Path, user_agent: str = USER_AGENT):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _cache_path(self, namespace: str, payload: dict[str, Any], suffix: str = ".json") -> Path:
        import hashlib

        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        directory = self.cache_root / slugify(namespace)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}{suffix}"

    def get_json(
        self,
        url: str,
        params: dict[str, Any],
        namespace: str,
        *,
        timeout: int = 120,
        max_attempts: int = 8,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        path = self._cache_path(namespace, {"method": "GET", "url": url, "params": params})
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=extra_headers,
                    timeout=(20, timeout),
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after is not None else None
                    except Exception:
                        wait = None
                    if wait is None:
                        wait = min(60.0, (2 ** min(attempt, 5)) + random.uniform(0.5, 2.0))
                    if response.status_code == 429:
                        wait = max(wait, 30.0)
                    print(f"HTTP {response.status_code} from {url}; retry {attempt}/{max_attempts} in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                    if payload["error"].get("code") == "maxlag":
                        wait = min(60.0, 3.0 * attempt + random.uniform(0.5, 2.0))
                        print(f"API maxlag from {url}; retry {attempt}/{max_attempts} in {wait:.1f}s")
                        time.sleep(wait)
                        continue
                atomic_write_json(path, payload)
                return payload
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as error:
                last_error = error
                if attempt >= max_attempts:
                    break
                wait = min(60.0, (2 ** min(attempt, 5)) + random.uniform(0.5, 2.0))
                print(f"Request error {attempt}/{max_attempts} for {url}: {error!r}; retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"GET failed after {max_attempts} attempts: {url}; last={last_error!r}") from last_error

    def post_json_rotating(
        self,
        endpoints: list[str],
        data: dict[str, Any],
        namespace: str,
        *,
        timeout: int = 120,
        attempts_per_endpoint: int = 2,
        courtesy_429_seconds: float = 30.0,
    ) -> Any:
        path = self._cache_path(namespace, {"method": "POST", "data": data, "version": 2})
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                path.unlink(missing_ok=True)

        ordered = list(endpoints)
        random.Random(path.name).shuffle(ordered)
        errors: list[str] = []
        for endpoint in ordered:
            for attempt in range(1, attempts_per_endpoint + 1):
                try:
                    response = self.session.post(endpoint, data=data, timeout=(20, timeout))
                    if response.status_code in {429, 500, 502, 503, 504}:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            wait = float(retry_after) if retry_after else None
                        except Exception:
                            wait = None
                        if wait is None:
                            wait = 2.0 + attempt + random.uniform(0.5, 1.5)
                        if response.status_code == 429:
                            wait = max(courtesy_429_seconds, wait)
                        print(f"HTTP {response.status_code} from {endpoint}; attempt {attempt}/{attempts_per_endpoint}")
                        if attempt < attempts_per_endpoint:
                            time.sleep(min(wait, 30.0))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Expected JSON object response")
                    atomic_write_json(path, payload)
                    return payload
                except Exception as error:
                    errors.append(f"{endpoint}: {error!r}")
                    if attempt < attempts_per_endpoint:
                        time.sleep(2.0 + random.uniform(0.5, 1.5))
        raise RuntimeError("All endpoints failed:\n" + "\n".join(errors[-8:]))

    def download(
        self,
        url: str,
        destination: Path,
        *,
        timeout: int = 180,
        max_attempts: int = 6,
        expected_sha256: str | None = None,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if expected_sha256 and sha256_file(destination) != expected_sha256:
                raise ValueError(f"Checksum mismatch for cached download: {destination}")
            return destination

        partial = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self.session.get(url, stream=True, timeout=(20, timeout)) as response:
                    if response.status_code in {429, 500, 502, 503, 504}:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            wait = float(retry_after) if retry_after else None
                        except Exception:
                            wait = None
                        if wait is None:
                            wait = min(60.0, (2 ** min(attempt, 5)) + random.uniform(0.5, 2.0))
                        if response.status_code == 429:
                            wait = max(wait, 30.0)
                        if attempt < max_attempts:
                            print(
                                f"Download HTTP {response.status_code} from {url}; "
                                f"retry {attempt}/{max_attempts} in {wait:.1f}s"
                            )
                            time.sleep(wait)
                            continue
                    response.raise_for_status()
                    with partial.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                partial.replace(destination)
                if expected_sha256 and sha256_file(destination) != expected_sha256:
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch after download: {url}")
                return destination
            except Exception as error:
                last_error = error
                partial.unlink(missing_ok=True)
                if attempt < max_attempts:
                    wait = min(60.0, (2 ** min(attempt, 5)) + random.uniform(0.5, 2.0))
                    print(f"Download error {attempt}/{max_attempts}: {url}: {error!r}; retrying in {wait:.1f}s")
                    time.sleep(wait)
        raise RuntimeError(f"Download failed after {max_attempts} attempts: {url}") from last_error
