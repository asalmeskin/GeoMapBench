
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CANONICAL_LEAVES: tuple[str, ...] = (
    "cartographic_symbol_recognition",
    "map_text_detection_recognition_grouping",
    "map_label_feature_anchoring",
    "dense_land_cover_labeling",
    "remote_sensing_scene_classification",
    "object_presence_counting",
    "change_localization",
    "temporal_scene_matching",
    "visual_geolocation",
    "coordinate_transformation",
    "metric_distance_computation",
    "topological_directional_reasoning",
    "spatial_graph_construction",
    "shortest_path_optimization",
    "isochrone_service_area",
    "toponym_recognition",
    "geo_entity_typing",
    "textual_spatial_relation_extraction",
    "cross_entity_comparison",
    "environmental_layer_identification",
    "population_density_estimation",
    "geologic_geomorphic_interpretation",
    "geographic_fact_reasoning",
)

EXPECTED_RECORDS_PER_LEAF = 100
CLEAN_FILENAME = "data_clean.jsonl"
METADATA_DIRNAME = "_clean_metadata"

# These are intentionally removed from each model-facing record. They remain in
# the original data.jsonl and, unless --no-provenance is used, in provenance.
TOP_LEVEL_PROVENANCE_FIELDS: tuple[str, ...] = (
    "seed",
    "group_id",
    "source",
    "attribution",
    "base_evaluation",
)

# These Bloom fields are release bookkeeping or derivable from the remaining
# fields and are therefore redundant in each of 100 records per leaf.
BLOOM_KEEP_FIELDS: tuple[str, ...] = (
    "level",
    "level_name",
    "variant",
)

# These evaluation fields duplicate record["bloom"].
EVALUATION_DROP_FIELDS: tuple[str, ...] = (
    "bloom_level",
    "bloom_level_name",
)


class CleanDataError(RuntimeError):
    """Raised when cleaning cannot be completed safely."""


def _json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise CleanDataError(
                        f"{path}: invalid JSON on line {line_no}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise CleanDataError(
                        f"{path}: line {line_no} is {type(obj).__name__}, expected object"
                    )
                record_id = obj.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise CleanDataError(f"{path}: line {line_no}: missing/invalid id")
                if record_id in seen_ids:
                    raise CleanDataError(f"{path}: duplicate id {record_id!r}")
                seen_ids.add(record_id)
                records.append(obj)
    except OSError as exc:
        raise CleanDataError(f"Could not read {path}: {exc}") from exc

    return records


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            for record in records:
                f.write(_json_bytes(record))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def _validate_bloom_record(record: dict[str, Any], leaf: str, index: int) -> None:
    prefix = f"{leaf}: record {index + 1} ({record.get('id', '?')})"

    if record.get("leaf") != leaf:
        raise CleanDataError(
            f"{prefix}: leaf={record.get('leaf')!r}, expected {leaf!r}"
        )

    input_obj = record.get("input")
    if not isinstance(input_obj, dict):
        raise CleanDataError(f"{prefix}: input must be an object")
    if not isinstance(input_obj.get("question"), str) or not input_obj["question"].strip():
        raise CleanDataError(f"{prefix}: missing Bloom input.question")

    bloom = record.get("bloom")
    if not isinstance(bloom, dict):
        raise CleanDataError(f"{prefix}: missing bloom object")
    level = bloom.get("level")
    if level not in {"R", "U", "Ap", "An", "E", "C"}:
        raise CleanDataError(f"{prefix}: invalid Bloom level {level!r}")

    target = record.get("target")
    if not isinstance(target, dict):
        raise CleanDataError(f"{prefix}: target must be an object")
    if "bloom_answer" not in target:
        raise CleanDataError(f"{prefix}: target.bloom_answer is missing")

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        raise CleanDataError(f"{prefix}: evaluation must be an object")
    if evaluation.get("target_field") not in (None, "target.bloom_answer"):
        raise CleanDataError(
            f"{prefix}: unexpected evaluation.target_field="
            f"{evaluation.get('target_field')!r}"
        )


def _clean_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (clean_record, provenance_record)."""

    input_obj = copy.deepcopy(record["input"])
    # This is the old pre-Bloom prompt retained by bloomify. The current prompt
    # is input.question, so carrying both into the model-facing view is noisy.
    input_obj.pop("base_question", None)

    bloom_obj = record["bloom"]
    clean_bloom = {
        key: copy.deepcopy(bloom_obj[key])
        for key in BLOOM_KEEP_FIELDS
        if key in bloom_obj
    }

    evaluation = copy.deepcopy(record["evaluation"])
    for key in EVALUATION_DROP_FIELDS:
        evaluation.pop(key, None)
    # Keep the explicit target path for evaluator compatibility. Add it if an
    # older Bloom file omitted it.
    evaluation.setdefault("target_field", "target.bloom_answer")

    clean: dict[str, Any] = {
        "id": record["id"],
        "leaf": record["leaf"],
        "bloom": clean_bloom,
        "input": input_obj,
        # Only the Bloom gold target is active after expansion. Keeping the old
        # target fields creates ambiguity and can expose irrelevant supervision.
        "target": {"bloom_answer": copy.deepcopy(record["target"]["bloom_answer"])},
        "evaluation": evaluation,
    }

    provenance: dict[str, Any] = {
        "id": record["id"],
        "leaf": record["leaf"],
    }

    for key in TOP_LEVEL_PROVENANCE_FIELDS:
        if key in record:
            provenance[key] = copy.deepcopy(record[key])

    # Keep everything removed from the Bloom object in provenance, especially
    # source_record_ids needed to trace constructed/multi-record examples.
    removed_bloom = {
        key: copy.deepcopy(value)
        for key, value in bloom_obj.items()
        if key not in BLOOM_KEEP_FIELDS
    }
    if removed_bloom:
        provenance["bloom_metadata"] = removed_bloom

    # Keep all old target information outside the model-facing clean file.
    old_target = {
        key: copy.deepcopy(value)
        for key, value in record["target"].items()
        if key != "bloom_answer"
    }
    if old_target:
        provenance["base_target"] = old_target

    # Capture any unknown top-level extension fields rather than silently
    # discarding them. This makes the cleaner forward-compatible.
    known_top = {
        "id",
        "leaf",
        "seed",
        "group_id",
        "source",
        "input",
        "target",
        "evaluation",
        "attribution",
        "bloom",
        "base_evaluation",
    }
    extras = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in known_top
    }
    if extras:
        provenance["extra_top_level_fields"] = extras

    return clean, provenance


def _logical_size(records: Iterable[dict[str, Any]]) -> int:
    return sum(len(_json_bytes(r)) for r in records)


def _human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def clean_leaf(
    root: Path,
    leaf: str,
    *,
    overwrite: bool,
    write_provenance: bool,
    dry_run: bool,
    expected_count: int | None,
) -> dict[str, Any]:
    task_dir = root / leaf
    src = task_dir / "data.jsonl"
    dst = task_dir / CLEAN_FILENAME
    metadata_root = root / METADATA_DIRNAME
    provenance_path = metadata_root / f"{leaf}.provenance.jsonl"

    if not task_dir.is_dir():
        raise CleanDataError(f"Missing canonical leaf directory: {task_dir}")
    if not src.is_file():
        raise CleanDataError(f"Missing source data: {src}")

    records = _read_jsonl(src)
    if expected_count is not None and len(records) != expected_count:
        raise CleanDataError(
            f"{leaf}: expected {expected_count} records, found {len(records)}"
        )

    cleaned: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    levels: Counter[str] = Counter()

    for i, record in enumerate(records):
        _validate_bloom_record(record, leaf, i)
        clean, prov = _clean_record(record)
        cleaned.append(clean)
        provenance.append(prov)
        levels[str(clean["bloom"]["level"])] += 1

    # Safety: exact ID/order preservation.
    original_ids = [r["id"] for r in records]
    clean_ids = [r["id"] for r in cleaned]
    if original_ids != clean_ids:
        raise CleanDataError(f"{leaf}: cleaner changed record IDs/order")

    original_size = _logical_size(records)
    clean_size = _logical_size(cleaned)
    reduction = 0.0 if original_size == 0 else 100.0 * (1.0 - clean_size / original_size)

    if not dry_run:
        if dst.exists() and not overwrite:
            raise CleanDataError(
                f"Refusing to overwrite existing {dst}. "
                "Use --overwrite after inspecting it, or remove that sidecar file."
            )
        if write_provenance and provenance_path.exists() and not overwrite:
            raise CleanDataError(
                f"Refusing to overwrite existing {provenance_path}. Use --overwrite."
            )

        _atomic_write_jsonl(dst, cleaned)
        if write_provenance:
            _atomic_write_jsonl(provenance_path, provenance)

        # Read it back immediately so filesystem/encoding/write errors cannot
        # silently leave a bad clean file.
        reread = _read_jsonl(dst)
        if reread != cleaned:
            raise CleanDataError(f"{leaf}: post-write verification failed for {dst}")

    result: dict[str, Any] = {
        "leaf": leaf,
        "count": len(cleaned),
        "bloom_distribution": dict(sorted(levels.items())),
        "source_file": str(src),
        "clean_file": str(dst),
        "original_logical_bytes": original_size,
        "clean_logical_bytes": clean_size,
        "reduction_percent": round(reduction, 2),
        "clean_sha256": None if dry_run else _sha256(dst),
        "provenance_file": str(provenance_path) if write_provenance else None,
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create compact data_clean.jsonl files for Bloom-expanded GeoMapBench "
            "without modifying data.jsonl or assets."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to geomapbench_100 (the directory containing the 23 leaves).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing data_clean.jsonl/provenance sidecars. Originals are still untouched.",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Do not write _clean_metadata/*.provenance.jsonl files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report expected size reduction without writing files.",
    )
    parser.add_argument(
        "--allow-non100",
        action="store_true",
        help="Allow leaves whose data.jsonl does not contain exactly 100 records.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    # Explicitly ignore Google Drive duplicate folders such as '<leaf> (1)'.
    task_like_dirs = {
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "data.jsonl").is_file()
    }
    extras = sorted(task_like_dirs - set(CANONICAL_LEAVES))

    missing = [
        leaf
        for leaf in CANONICAL_LEAVES
        if not (root / leaf / "data.jsonl").is_file()
    ]
    if missing:
        print("ERROR: missing canonical leaf data:", file=sys.stderr)
        for leaf in missing:
            print(f"  - {leaf}", file=sys.stderr)
        return 2

    if extras:
        print("Ignoring noncanonical task-like directories:")
        for name in extras:
            print(f"  - {name}")
        print()

    expected_count = None if args.allow_non100 else EXPECTED_RECORDS_PER_LEAF
    results: list[dict[str, Any]] = []

    try:
        for leaf in CANONICAL_LEAVES:
            result = clean_leaf(
                root,
                leaf,
                overwrite=args.overwrite,
                write_provenance=not args.no_provenance,
                dry_run=args.dry_run,
                expected_count=expected_count,
            )
            results.append(result)
            print(
                f"{leaf:43s} "
                f"{result['count']:3d} records | "
                f"{result['reduction_percent']:6.2f}% smaller | "
                f"Bloom {result['bloom_distribution']}"
            )
    except CleanDataError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    total_original = sum(r["original_logical_bytes"] for r in results)
    total_clean = sum(r["clean_logical_bytes"] for r in results)
    total_reduction = (
        0.0
        if total_original == 0
        else 100.0 * (1.0 - total_clean / total_original)
    )

    summary = {
        "format": "GeoMapBench Bloom model-facing clean view",
        "canonical_leaf_count": len(CANONICAL_LEAVES),
        "record_count": sum(r["count"] for r in results),
        "source_filename": "data.jsonl",
        "clean_filename": CLEAN_FILENAME,
        "original_data_modified": False,
        "cleaning_policy": {
            "kept_top_level": ["id", "leaf", "bloom", "input", "target", "evaluation"],
            "kept_bloom_fields": list(BLOOM_KEEP_FIELDS),
            "target_policy": "keep only target.bloom_answer",
            "input_policy": "keep task-specific input; remove only input.base_question",
            "evaluation_removed": list(EVALUATION_DROP_FIELDS),
            "provenance_removed_from_clean_record": list(TOP_LEVEL_PROVENANCE_FIELDS),
        },
        "original_logical_bytes": total_original,
        "clean_logical_bytes": total_clean,
        "reduction_percent": round(total_reduction, 2),
        "leaves": results,
    }

    if not args.dry_run:
        metadata_root = root / METADATA_DIRNAME
        _atomic_write_json(metadata_root / "summary.json", summary)

    print("\n" + "=" * 72)
    print("DRY RUN COMPLETE" if args.dry_run else "CLEAN DATA COMPLETE")
    print("=" * 72)
    print(f"Canonical leaves : {len(CANONICAL_LEAVES)}")
    print(f"Records          : {summary['record_count']}")
    print(f"Original size    : {_human_bytes(total_original)}")
    print(f"Clean size       : {_human_bytes(total_clean)}")
    print(f"Reduction        : {total_reduction:.2f}%")
    print("Original modified: NO")
    if not args.dry_run:
        print(f"Clean files      : <leaf>/{CLEAN_FILENAME}")
        if not args.no_provenance:
            print(f"Provenance       : {root / METADATA_DIRNAME}")
        print(f"Summary          : {root / METADATA_DIRNAME / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
