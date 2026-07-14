from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .common import N_EXAMPLES, SEEDS, read_jsonl, sha256_file


def _asset_strings(obj: Any, parent_key: str = "") -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _asset_strings(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from _asset_strings(value, parent_key)
    elif isinstance(obj, str) and (parent_key in {"images", "mask", "change_mask", "graph", "route_image", "isochrone_image"} or parent_key.endswith("_image")):
        yield obj


def validate_task(task_dir: Path, require_assets: bool = True) -> list[str]:
    errors: list[str] = []
    data_path = task_dir / "data.jsonl"
    manifest_path = task_dir / "manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        return [f"{task_dir.name}: missing data.jsonl or manifest.json"]
    records = read_jsonl(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(records) != N_EXAMPLES:
        errors.append(f"{task_dir.name}: {len(records)} records, expected {N_EXAMPLES}")
    if manifest.get("count") != len(records):
        errors.append(f"{task_dir.name}: manifest count mismatch")
    if manifest.get("sha256") != sha256_file(data_path):
        errors.append(f"{task_dir.name}: checksum mismatch")
    if manifest.get("seed") != SEEDS.get(task_dir.name):
        errors.append(f"{task_dir.name}: unexpected seed")
    ids: set[str] = set()
    for line_number, record in enumerate(records, 1):
        for field in ("id", "leaf", "seed", "group_id", "source", "input", "target", "evaluation"):
            if field not in record:
                errors.append(f"{task_dir.name}:{line_number}: missing {field}")
        if record.get("id") in ids:
            errors.append(f"{task_dir.name}:{line_number}: duplicate ID {record.get('id')}")
        ids.add(record.get("id"))
        if record.get("leaf") != task_dir.name:
            errors.append(f"{task_dir.name}:{line_number}: leaf mismatch")
        if require_assets:
            for asset in _asset_strings(record):
                if asset.startswith(("http://", "https://")):
                    continue
                candidate = (task_dir / asset).resolve()
                if task_dir.resolve() not in candidate.parents:
                    errors.append(f"{task_dir.name}:{line_number}: unsafe asset path {asset}")
                elif not candidate.exists():
                    errors.append(f"{task_dir.name}:{line_number}: missing asset {asset}")
    return errors


def validate_root(root: Path, require_all: bool = False, require_assets: bool = True) -> list[str]:
    errors: list[str] = []
    found = {p.name for p in root.iterdir() if p.is_dir() and (p / "data.jsonl").exists()} if root.exists() else set()
    if require_all:
        missing = sorted(set(SEEDS) - found)
        if missing:
            errors.append("Missing tasks: " + ", ".join(missing))
    for name in sorted(found):
        errors.extend(validate_task(root / name, require_assets=require_assets))
    return errors

