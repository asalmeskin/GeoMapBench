from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .common import atomic_write_json, read_jsonl, sha256_file, text_hash


PROFILE_MINIMA: dict[str, dict[str, Any]] = {
    "smoke": {"total": 50, "map_image": 1},
    "standard": {"total": 12_000, "map_image": 200},
    "iclr": {
        "total": 45_000,
        "map_image": 700,
        "sources": {
            "OpenStreetMap": 12_000,
            "EPSG / PROJ": 4_000,
            "Wikidata": 5_000,
            "Wikipedia": 5_000,
            "GeoNames": 8_000,
            "World Bank": 4_000,
            "Wikimedia Commons": 300,
        },
    },
}


def validate_corpus(root: Path, *, profile: str | None = None, strict_scale: bool = False) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    corpus_path = root / "corpus.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    if not corpus_path.exists():
        return {"valid": False, "errors": [f"Missing corpus: {corpus_path}"], "warnings": []}

    records = read_jsonl(corpus_path)
    ids: set[str] = set()
    text_hashes: set[str] = set()
    sources: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    geonames_classes: Counter[str] = Counter()
    missing_media: list[str] = []
    bad_images: list[str] = []

    for index, record in enumerate(records, 1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"record {index}: missing id")
            continue
        if record_id in ids:
            errors.append(f"duplicate id: {record_id}")
        ids.add(record_id)
        source = record.get("source")
        if not isinstance(source, dict) or not source.get("name") or not source.get("license"):
            errors.append(f"{record_id}: incomplete source metadata")
            source_name = "Unknown"
        else:
            source_name = str(source["name"])
        input_obj = record.get("input")
        if not isinstance(input_obj, dict):
            errors.append(f"{record_id}: input must be an object")
            continue
        text = str(input_obj.get("text") or "")
        if len(text.strip()) < 20:
            errors.append(f"{record_id}: text is too short")
        digest = text_hash(text)
        if digest in text_hashes:
            errors.append(f"duplicate normalized text: {record_id}")
        text_hashes.add(digest)
        modality = str(input_obj.get("modality") or "unknown")
        sources[source_name] += 1
        modalities[modality] += 1
        retrieval = record.get("retrieval") or {}
        for capability in retrieval.get("capabilities") or []:
            capabilities[str(capability)] += 1
        if source_name == "GeoNames":
            feature_class = str((record.get("extra") or {}).get("feature_class") or "")
            if feature_class:
                geonames_classes[feature_class] += 1
        if modality == "map_image":
            extra = record.get("extra") or {}
            if extra.get("labels_in_pixels") is not False:
                errors.append(f"{record_id}: map image does not explicitly declare labels_in_pixels=False")
            if extra.get("coordinate_axes_in_pixels") is not False:
                errors.append(f"{record_id}: map image does not explicitly declare coordinate_axes_in_pixels=False")
            if extra.get("metric_clipping") is not True:
                errors.append(f"{record_id}: map image does not explicitly declare metric_clipping=True")
        images = input_obj.get("images") or []
        if images and not isinstance(images, list):
            errors.append(f"{record_id}: input.images must be a list")
            images = []
        for relative in images:
            path = root / str(relative)
            if not path.exists():
                missing_media.append(str(relative))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                bad_images.append(str(relative))

    if missing_media:
        errors.append(f"missing media files: {len(missing_media)}")
    if bad_images:
        errors.append(f"unreadable image assets: {len(bad_images)}")

    minima = PROFILE_MINIMA.get(profile or "", {})
    if minima:
        if len(records) < int(minima.get("total", 0)):
            message = f"scale target not met: total {len(records)} < {minima['total']}"
            (errors if strict_scale else warnings).append(message)
        if modalities.get("map_image", 0) < int(minima.get("map_image", 0)):
            message = f"scale target not met: map_image {modalities.get('map_image', 0)} < {minima['map_image']}"
            (errors if strict_scale else warnings).append(message)
        for source_name, minimum in (minima.get("sources") or {}).items():
            if sources.get(source_name, 0) < minimum:
                message = f"scale target not met: {source_name} {sources.get(source_name, 0)} < {minimum}"
                (errors if strict_scale else warnings).append(message)

        if profile == "iclr":
            required_geonames_classes = set("AHLPRSTUV")
            observed = set(geonames_classes)
            if observed != required_geonames_classes:
                message = f"GeoNames class coverage incomplete: {sorted(observed)}; expected {sorted(required_geonames_classes)}"
                (errors if strict_scale else warnings).append(message)

    report = {
        "valid": not errors,
        "record_count": len(records),
        "sources": dict(sources.most_common()),
        "modalities": dict(modalities.most_common()),
        "capabilities": dict(capabilities.most_common()),
        "geonames_feature_classes": dict(sorted(geonames_classes.items())),
        "duplicate_ids": len(records) - len(ids),
        "duplicate_text": len(records) - len(text_hashes),
        "missing_media": missing_media[:100],
        "bad_images": bad_images[:100],
        "corpus_sha256": sha256_file(corpus_path),
        "profile": profile,
        "strict_scale": strict_scale,
        "errors": errors,
        "warnings": warnings,
    }
    atomic_write_json(root / "quality_report.json", report)
    return report
