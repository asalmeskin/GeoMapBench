from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path
from typing import Any

from geomapbench_data.common import N_EXAMPLES, SEEDS

from .common import read_jsonl


@functools.lru_cache(maxsize=8)
def _canonical_benchmark_records_cached(
    root_key: str, prefer_clean: bool,
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    """Load only the 23 released leaves, ignoring Drive duplicate folders.

    This reads and JSON-parses all 2,300 canonical records on every call. It is
    invoked once per condition by ``rescore_in_place`` (via ``analyze``/
    ``compare``), and callers such as ``geoagent-eval suite`` may touch several
    conditions in one process -- uncached, that repeats the full read (often
    over a Drive FUSE mount, where it dominates wall-clock time) for no reason,
    since the on-disk benchmark never changes mid-run. Cached per (resolved
    root, prefer_clean) for the life of the process. No caller mutates a
    returned record in place, so sharing the cached list across calls is safe.
    """
    root = Path(root_key)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")
    rows: list[tuple[Path, dict[str, Any]]] = []
    for leaf in sorted(SEEDS):
        task_dir = root / leaf
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing canonical benchmark leaf: {task_dir}")
        clean = task_dir / "data_clean.jsonl"
        raw = task_dir / "data.jsonl"
        path = clean if prefer_clean and clean.exists() else raw
        if not path.exists():
            raise FileNotFoundError(path)
        records = read_jsonl(path)
        if len(records) != N_EXAMPLES:
            raise ValueError(f"{leaf}: expected {N_EXAMPLES} records, found {len(records)}")
        for record in records:
            if str(record.get("leaf", leaf)) != leaf:
                raise ValueError(f"{leaf}: record {record.get('id')} has a mismatched leaf")
            rows.append((task_dir, record))
    ids = [str(record.get("id")) for _, record in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate canonical record IDs: {duplicates[:10]}")
    return tuple(rows)


def canonical_benchmark_records(
    root: Path, *, prefer_clean: bool = True
) -> list[tuple[Path, dict[str, Any]]]:
    """Load only the 23 released leaves, ignoring Drive duplicate folders."""
    root = root.expanduser().resolve()
    return list(_canonical_benchmark_records_cached(str(root), prefer_clean))


def stable_subset(
    records: list[tuple[Path, dict[str, Any]]],
    *, per_leaf_limit: int | None, limit: int | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Choose a deterministic subset before completed IDs are removed."""
    selected = records
    if per_leaf_limit is not None:
        counts: Counter[str] = Counter()
        subset: list[tuple[Path, dict[str, Any]]] = []
        for directory, record in selected:
            leaf = str(record.get("leaf", directory.name))
            if counts[leaf] < per_leaf_limit:
                subset.append((directory, record))
                counts[leaf] += 1
        selected = subset
    if limit is not None:
        selected = selected[:limit]
    return selected
