from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geomapbench_data.bloom import BLOOM_LEVELS

from .common import append_jsonl, atomic_json, digest, read_jsonl
from .prompts import build_messages


COHORT_REVISION = "2026-09-cumulative-bloom-v1"


def _level(record: dict[str, Any]) -> str:
    return str((record.get("bloom") or {}).get("level") or "unknown")


def bloom_stratified_order(
    records: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    """Return a deterministic, nested order that preserves the legacy first row.

    The first canonical row of every leaf is kept as the migration-compatible
    seed. Remaining rows are selected greedily from the currently least-covered
    Bloom level. Rows inside each Bloom bucket use a stable content hash rather
    than source adjacency, so small cumulative cohorts are broad and repeatable.
    """
    by_leaf: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for directory, record in records:
        by_leaf[str(record.get("leaf", directory.name))].append((directory, record))

    ordered: list[tuple[Path, dict[str, Any]]] = []
    for leaf in sorted(by_leaf):
        leaf_rows = by_leaf[leaf]
        if not leaf_rows:
            continue
        chosen = [leaf_rows[0]]
        counts: Counter[str] = Counter({_level(leaf_rows[0][1]): 1})
        buckets: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        for item in leaf_rows[1:]:
            buckets[_level(item[1])].append(item)
        for level in buckets:
            buckets[level].sort(
                key=lambda item: digest({
                    "revision": COHORT_REVISION,
                    "leaf": leaf,
                    "level": level,
                    "id": str(item[1].get("id")),
                })
            )
        preferred = {level: index for index, level in enumerate(BLOOM_LEVELS.get(leaf, ())) }
        while any(buckets.values()):
            available = [level for level, values in buckets.items() if values]
            next_level = min(
                available,
                key=lambda level: (counts[level], preferred.get(level, 999), level),
            )
            chosen.append(buckets[next_level].pop(0))
            counts[next_level] += 1
        ordered.extend(chosen)
    return ordered


def cumulative_cohort(
    records: list[tuple[Path, dict[str, Any]]],
    target_per_leaf: int,
) -> list[tuple[Path, dict[str, Any]]]:
    if not 1 <= target_per_leaf <= 100:
        raise ValueError("target_per_leaf must be between 1 and 100")
    counts: Counter[str] = Counter()
    selected: list[tuple[Path, dict[str, Any]]] = []
    for directory, record in bloom_stratified_order(records):
        leaf = str(record.get("leaf", directory.name))
        if counts[leaf] < target_per_leaf:
            selected.append((directory, record))
            counts[leaf] += 1
    if set(counts.values()) != {target_per_leaf}:
        raise ValueError(f"Could not build a balanced cumulative cohort: {dict(counts)}")
    return selected


def write_cohort_manifest(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    target_per_leaf: int,
    output_root: Path,
    benchmark_content_hash: str,
) -> tuple[Path, dict[str, Any]]:
    selected = cumulative_cohort(records, target_per_leaf)
    ids = [str(record.get("id")) for _, record in selected]
    per_leaf: dict[str, Counter[str]] = defaultdict(Counter)
    for directory, record in selected:
        per_leaf[str(record.get("leaf", directory.name))][_level(record)] += 1
    manifest = {
        "revision": COHORT_REVISION,
        "benchmark_content_hash": benchmark_content_hash,
        "target_per_leaf": target_per_leaf,
        "target_record_count": len(ids),
        "selected_ids_hash": digest(ids),
        "selected_ids": ids,
        "bloom_distribution_by_leaf": {
            leaf: dict(sorted(levels.items())) for leaf, levels in sorted(per_leaf.items())
        },
    }
    history = output_root / "cohorts" / f"target_{target_per_leaf:03d}.json"
    if history.exists():
        previous = json.loads(history.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(f"Cohort manifest changed unexpectedly: {history}")
    else:
        atomic_json(history, manifest)
    active = output_root / "cohort.json"
    if active.exists():
        previous = json.loads(active.read_text(encoding="utf-8"))
        old_ids = set(previous.get("selected_ids") or [])
        if not old_ids.issubset(ids):
            raise ValueError(
                "Cumulative target cannot shrink or change prior IDs; increase --target-per-leaf"
            )
    atomic_json(active, manifest)
    return active, manifest


def _expected_prompt_hash(
    record_id: str,
    records_by_id: dict[str, tuple[Path, dict[str, Any]]],
    cache: dict[str, str],
) -> str:
    if record_id not in cache:
        directory, record = records_by_id[record_id]
        cache[record_id] = digest(build_messages(
            record,
            directory,
            contexts=None,
            include_images=True,
            max_image_bytes=8_000_000,
        ))
    return cache[record_id]


def migrate_legacy_base_outputs(
    legacy_outputs: list[Path],
    *,
    destination: Path,
    model: str,
    target_ids: set[str],
    records_by_id: dict[str, tuple[Path, dict[str, Any]]],
    prompt_hash_cache: dict[str, str],
) -> dict[str, Any]:
    """Import compatible v6 base rows and raw responses without selective retry.

    Only successful rows in the active canonical cohort are promoted to
    `responses.jsonl`. Raw API responses are also cohort-filtered. Prompt hashes
    must match the current prompt byte-for-byte, and first-seen rows win.
    """
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "responses.jsonl"
    api_path = destination / "api_responses.jsonl"
    completed = {
        str(row.get("id")) for row in (read_jsonl(result_path) if result_path.exists() else [])
        if row.get("status") == "ok"
    }
    cache_keys = {
        str(row.get("cache_key")) for row in (read_jsonl(api_path) if api_path.exists() else [])
        if row.get("cache_key")
    }
    imported_results = imported_api = rejected = 0
    source_reports: list[dict[str, Any]] = []
    for source in legacy_outputs:
        source = source.expanduser().resolve()
        source_result = source / "responses.jsonl"
        source_api = source / "api_responses.jsonl"
        before_results, before_api = imported_results, imported_api
        if source_result.exists():
            for row in read_jsonl(source_result):
                record_id = str(row.get("id"))
                if (
                    row.get("status") != "ok"
                    or row.get("condition") != "base"
                    or row.get("model") != model
                    or record_id not in target_ids
                    or record_id in completed
                    or record_id not in records_by_id
                    or row.get("prompt_hash") != _expected_prompt_hash(
                        record_id, records_by_id, prompt_hash_cache
                    )
                ):
                    rejected += 1
                    continue
                append_jsonl(result_path, row)
                completed.add(record_id)
                imported_results += 1
        if source_api.exists():
            for row in read_jsonl(source_api):
                record_id = str(row.get("id"))
                cache_key = str(row.get("cache_key") or "")
                if (
                    not cache_key
                    or cache_key in cache_keys
                    or record_id not in target_ids
                    or record_id not in records_by_id
                    or row.get("condition") != "base"
                    or row.get("model") != model
                    or not isinstance(row.get("response"), dict)
                    or row.get("prompt_hash") != _expected_prompt_hash(
                        record_id, records_by_id, prompt_hash_cache
                    )
                ):
                    rejected += 1
                    continue
                append_jsonl(api_path, row)
                cache_keys.add(cache_key)
                imported_api += 1
        source_reports.append({
            "source": str(source),
            "results_imported": imported_results - before_results,
            "api_responses_imported": imported_api - before_api,
        })
    return {
        "model": model,
        "destination": str(destination),
        "results_imported": imported_results,
        "api_responses_imported": imported_api,
        "rejected_or_duplicate_rows": rejected,
        "sources": source_reports,
    }
