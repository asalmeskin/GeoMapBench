from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar


N_EXAMPLES = 100

# Every leaf has its own fixed seed. Keep these values immutable after a public
# benchmark release. Changing a seed creates a new benchmark version.
SEEDS: dict[str, int] = {
    "cartographic_symbol_recognition": 41001,
    "map_text_detection_recognition_grouping": 41002,
    "map_label_feature_anchoring": 41003,
    "dense_land_cover_labeling": 41004,
    "remote_sensing_scene_classification": 41005,
    "object_presence_counting": 41006,
    "change_localization": 41007,
    "temporal_scene_matching": 41008,
    "visual_geolocation": 41009,
    "coordinate_transformation": 41010,
    "metric_distance_computation": 41011,
    "topological_directional_reasoning": 41012,
    "spatial_graph_construction": 41013,
    "shortest_path_optimization": 41014,
    "isochrone_service_area": 41015,
    "toponym_recognition": 41016,
    "geo_entity_typing": 41017,
    "textual_spatial_relation_extraction": 41018,
    "cross_entity_comparison": 41019,
    "environmental_layer_identification": 41020,
    "population_density_estimation": 41021,
    "geologic_geomorphic_interpretation": 41022,
    "geographic_fact_reasoning": 41023,
}

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_sample(items: Iterable[T], n: int, seed: int, key=str) -> list[T]:
    """Sort before sampling so filesystem/API ordering cannot change a split."""
    ordered = sorted(items, key=key)
    if len(ordered) < n:
        raise ValueError(f"Need at least {n} candidates, found {len(ordered)}")
    return random.Random(seed).sample(ordered, n)


def balanced_sample(
    groups: dict[str, Sequence[T]], n: int, seed: int, key=str
) -> list[tuple[str, T]]:
    """Deterministic approximately balanced sampling over named groups."""
    names = sorted(k for k, values in groups.items() if values)
    if not names:
        raise ValueError("No non-empty groups")
    rng = random.Random(seed)
    pools = {k: sorted(groups[k], key=key)[:] for k in names}
    for values in pools.values():
        rng.shuffle(values)
    out: list[tuple[str, T]] = []
    cursor = 0
    while len(out) < n:
        name = names[cursor % len(names)]
        if pools[name]:
            out.append((name, pools[name].pop()))
        elif not any(pools.values()):
            break
        cursor += 1
    if len(out) != n:
        raise ValueError(f"Need {n} balanced candidates, found {len(out)}")
    rng.shuffle(out)
    return out


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def copy_asset(source: Path, asset_dir: Path, name: str | None = None) -> str:
    """Copy an asset and return a path relative to the task directory."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    destination = asset_dir / (name or source.name)
    shutil.copy2(source, destination)
    return destination.relative_to(asset_dir.parent).as_posix()


def base_record(
    leaf: str,
    index: int,
    source_name: str,
    source_url: str,
    license_name: str,
    group_id: str,
) -> dict[str, Any]:
    return {
        "id": f"{leaf}-{index:03d}",
        "leaf": leaf,
        "seed": SEEDS[leaf],
        "group_id": str(group_id),
        "source": {
            "name": source_name,
            "url": source_url,
            "license": license_name,
        },
    }


def finalize_task(
    output_dir: Path,
    leaf: str,
    records: list[dict[str, Any]],
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    if len(records) != N_EXAMPLES:
        raise ValueError(f"{leaf}: expected {N_EXAMPLES} records, got {len(records)}")
    ids = [r.get("id") for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{leaf}: duplicate IDs")
    task_dir = output_dir / leaf
    dataset_path = task_dir / "data.jsonl"
    write_jsonl(dataset_path, records)
    manifest = {
        "leaf": leaf,
        "count": len(records),
        "seed": SEEDS[leaf],
        "created_at": utc_now(),
        "data_file": dataset_path.name,
        "sha256": sha256_file(dataset_path),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (task_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield dictionaries recursively from irregular public JSON releases."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)

