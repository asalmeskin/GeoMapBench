from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from .common import sha256_file


USER_AGENT = "GeoMapBenchDataKit/1.0 (academic benchmark construction)"


def download(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and sha256_file(destination) != expected_sha256:
            raise ValueError(f"Checksum mismatch for cached file: {destination}")
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as f:
        shutil.copyfileobj(response, f)
    temporary.replace(destination)
    if expected_sha256 and sha256_file(destination) != expected_sha256:
        destination.unlink()
        raise ValueError(f"Checksum mismatch after download: {url}")
    return destination


def extract_zip(archive: Path, destination: Path) -> Path:
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    marker.write_text(sha256_file(archive) + "\n", encoding="utf-8")
    return destination


def zenodo_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return payload["files"]


def download_zenodo(
    record_id: str,
    destination: Path,
    wanted_names: Iterable[str] | None = None,
) -> list[Path]:
    wanted = set(wanted_names or [])
    outputs: list[Path] = []
    for item in zenodo_files(record_id):
        name = item["key"]
        if wanted and name not in wanted:
            continue
        url = item.get("links", {}).get("self") or item.get("links", {}).get("content")
        if not url:
            raise ValueError(f"No download URL for Zenodo file {name}")
        outputs.append(download(url, destination / name))
    missing = wanted - {p.name for p in outputs}
    if missing:
        raise FileNotFoundError(f"Zenodo record {record_id} lacks: {sorted(missing)}")
    return outputs

