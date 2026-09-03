from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import canonical_benchmark_records
from .common import atomic_json, stable_json, utc_now
from .prompts import (
    IMAGE_CONVERTER_REVISION, PROMPT_REVISION, input_asset_paths, input_document_paths,
    transport_image,
)


PREFLIGHT_REVISION = "2026-09-cache-v5-final-210"


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _fingerprint(
    benchmark_root: Path, records: list[tuple[Path, dict[str, Any]]]
) -> tuple[str, list[tuple[str, Path]], list[tuple[str, Path]]]:
    assets: dict[str, Path] = {}
    documents: dict[str, Path] = {}
    jsonl_stats: list[tuple[str, int, int]] = []
    for task_dir, record in records:
        for path in input_asset_paths(record, task_dir):
            assets[str(path)] = path
        for path in input_document_paths(record, task_dir):
            documents[str(path)] = path
    for leaf in sorted({directory.name for directory, _ in records}):
        task_dir = benchmark_root / leaf
        path = task_dir / "data_clean.jsonl"
        if not path.exists():
            path = task_dir / "data.jsonl"
        stat = path.stat()
        jsonl_stats.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    asset_stats: list[tuple[str, int, int]] = []
    for key, path in sorted(assets.items()):
        stat = path.stat()
        asset_stats.append((key, stat.st_size, stat.st_mtime_ns))
    document_stats: list[tuple[str, int, int]] = []
    for key, path in sorted(documents.items()):
        stat = path.stat()
        document_stats.append((key, stat.st_size, stat.st_mtime_ns))
    payload = {
        "revision": PREFLIGHT_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "converter_revision": IMAGE_CONVERTER_REVISION,
        "jsonl": jsonl_stats,
        "assets": asset_stats,
        "documents": document_stats,
    }
    fingerprint = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return (
        fingerprint,
        [(key, assets[key]) for key in sorted(assets)],
        [(key, documents[key]) for key in sorted(documents)],
    )


def _target_warnings(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, int]:
    findings: Counter[str] = Counter()
    for _, record in records:
        target = record.get("target") or {}
        path = str((record.get("evaluation") or {}).get("target_field") or "target.bloom_answer")
        key = path.split(".", 1)[1] if path.startswith("target.") else "bloom_answer"
        answer = target.get(key)
        if isinstance(answer, str) and answer.lower().endswith((".png", ".tif", ".tiff", ".geojson", ".json")):
            findings[str(record.get("leaf", "unknown"))] += 1
        elif isinstance(answer, dict) and any(
            isinstance(value, str) and value.lower().endswith((".png", ".tif", ".tiff", ".geojson", ".json"))
            for value in answer.values()
        ):
            findings[str(record.get("leaf", "unknown"))] += 1
    return dict(sorted(findings.items()))


def benchmark_preflight(
    benchmark_root: Path,
    *,
    cache_root: Path | None = None,
    force: bool = False,
    max_image_bytes: int = 8_000_000,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.expanduser().resolve()
    records = canonical_benchmark_records(benchmark_root)
    leaves = sorted({directory.name for directory, _ in records})
    print(
        f"[preflight] fingerprinting {len(records)} records across {len(leaves)} canonical leaves",
        flush=True,
    )
    fingerprint, assets, documents = _fingerprint(benchmark_root, records)
    cache_path = None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / "benchmark_preflight.json"
        if cache_path.exists() and not force:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("status") == "pass":
                print(
                    f"[preflight] CACHE HIT: {cached.get('image_count', 0)} assets were already validated; skipping conversion checks",
                    flush=True,
                )
                return {**cached, "cache_hit": True, "cache_path": str(cache_path)}
    print(
        f"[preflight] CACHE MISS: validating {len(assets)} unique image assets; SVG/TIFF conversions are cached",
        flush=True,
    )
    by_leaf: Counter[str] = Counter()
    for index, (path_text, path) in enumerate(assets, 1):
        data, mime, _ = transport_image(path)
        if len(data) > max_image_bytes:
            raise ValueError(f"Converted image exceeds {max_image_bytes} bytes: {path_text}")
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError(f"Unsupported transport MIME {mime}: {path_text}")
        try:
            leaf = path.relative_to(benchmark_root).parts[0]
        except ValueError:
            leaf = "external"
        by_leaf[leaf] += 1
        if index % 100 == 0 or index == len(assets):
            print(f"[preflight] {index}/{len(assets)} unique assets validated", flush=True)
    report: dict[str, Any] = {
        "status": "pass",
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "record_count": len(records),
        "leaf_count": len(leaves),
        "image_count": len(assets),
        "images_by_leaf": dict(sorted(by_leaf.items())),
        "artifact_target_warnings": _target_warnings(records),
        "portable_benchmark_hash": hashlib.sha256(stable_json({
            "records": [record for _, record in records],
            "assets": [
                (str(path.relative_to(benchmark_root)), _sha256_file(path))
                for _, path in assets
            ],
            "documents": [
                (str(path.relative_to(benchmark_root)), _sha256_file(path))
                for _, path in documents
            ],
        }).encode("utf-8")).hexdigest(),
        "cache_hit": False,
    }
    if cache_path:
        atomic_json(cache_path, report)
        report["cache_path"] = str(cache_path)
    print(f"[preflight] PASS: {len(assets)} unique assets validated", flush=True)
    if report["artifact_target_warnings"]:
        print(
            "[preflight] file-artifact targets detected; the final inline RLE/graph contracts "
            "will make these predictions locally measurable",
            flush=True,
        )
    return report
