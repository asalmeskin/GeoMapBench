from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from geomapbench_data.common import N_EXAMPLES, SEEDS

from .common import digest, read_jsonl
from .prompts import build_messages


def canonical_benchmark_records(
    root: Path, *, prefer_clean: bool = True, require_complete: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load only the immutable 23-leaf release, ignoring Drive copy folders."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")
    rows: list[tuple[Path, dict[str, Any]]] = []
    for leaf in sorted(SEEDS):
        task_dir = root / leaf
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing canonical benchmark leaf: {leaf}")
        clean = task_dir / "data_clean.jsonl"
        raw = task_dir / "data.jsonl"
        path = clean if prefer_clean and clean.exists() else raw
        if not path.exists():
            raise FileNotFoundError(path)
        records = read_jsonl(path)
        if require_complete and len(records) != N_EXAMPLES:
            raise ValueError(f"{leaf}: expected {N_EXAMPLES} records, found {len(records)}")
        rows.extend((task_dir, record) for record in records)
    ids = [str(record.get("id")) for _, record in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical benchmark record IDs detected")
    return rows


def benchmark_preflight(
    root: Path, *, prefer_clean: bool = True, encode_assets: bool = True,
    max_image_bytes: int = 8_000_000,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    rows = canonical_benchmark_records(root, prefer_clean=prefer_clean)
    canonical = set(SEEDS)
    directories = sorted(path.name for path in root.iterdir() if path.is_dir())
    extras = [name for name in directories if name not in canonical]
    counts = Counter(str(record.get("leaf", directory.name)) for directory, record in rows)
    image_parts = 0
    if encode_assets:
        for task_dir, record in rows:
            messages = build_messages(
                record, task_dir, include_images=True, max_image_bytes=max_image_bytes,
            )
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    image_parts += sum(part.get("type") == "image_url" for part in content)
    return {
        "benchmark_root": str(root),
        "canonical_leaf_count": len(counts),
        "record_count": len(rows),
        "records_per_leaf": dict(sorted(counts.items())),
        "extra_directories_ignored": extras,
        "image_parts_validated": image_parts,
        "record_id_digest": digest(sorted(str(record.get("id")) for _, record in rows)),
        "valid": len(rows) == len(SEEDS) * N_EXAMPLES and set(counts.values()) == {N_EXAMPLES},
    }
