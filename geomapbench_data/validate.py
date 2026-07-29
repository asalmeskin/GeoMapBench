from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .common import DATA_REVISION, N_EXAMPLES, SEEDS, read_jsonl, sha256_file


REVISED_TASKS = {
    "coordinate_transformation",
    "cross_entity_comparison",
    "dense_land_cover_labeling",
    "environmental_layer_identification",
    "geo_entity_typing",
    "isochrone_service_area",
    "map_label_feature_anchoring",
    "map_text_detection_recognition_grouping",
    "metric_distance_computation",
    "population_density_estimation",
    "shortest_path_optimization",
    "spatial_graph_construction",
    "topological_directional_reasoning",
}


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
    if task_dir.name in REVISED_TASKS and manifest.get("data_revision") != DATA_REVISION:
        errors.append(
            f"{task_dir.name}: expected data_revision {DATA_REVISION}, "
            f"found {manifest.get('data_revision')}"
        )
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
    errors.extend(_validate_revised_content(task_dir, records))
    return errors



def _validate_visible_rgb(path: Path) -> str | None:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            if image.mode != "RGB":
                return f"expected RGB image, found {image.mode}"
            array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return "empty image"
        spread = float(np.percentile(array, 99) - np.percentile(array, 1))
        mean = float(array.mean())
        if spread < 18:
            return f"insufficient contrast ({spread:.1f})"
        if mean < 5:
            return f"image is nearly black (mean={mean:.1f})"
        return None
    except Exception as error:
        return f"cannot inspect image: {error}"


def _validate_revised_content(task_dir: Path, records: list[dict[str, Any]]) -> list[str]:
    name = task_dir.name
    if name not in REVISED_TASKS or not records:
        return []
    errors: list[str] = []

    if name == "coordinate_transformation":
        modes = {record.get("input", {}).get("transformation_mode") for record in records}
        pairs = {
            (record.get("input", {}).get("source_crs"), record.get("input", {}).get("target_crs"))
            for record in records
        }
        if len(modes) < 6:
            errors.append(f"{name}: only {len(modes)} transformation modes")
        if len(pairs) < 8:
            errors.append(f"{name}: only {len(pairs)} source/target CRS pairs")

    elif name == "cross_entity_comparison":
        years = {record.get("target", {}).get("year") for record in records}
        indicators = {record.get("target", {}).get("indicator") for record in records}
        directions = {record.get("target", {}).get("relation") for record in records}
        if len(years) < 5:
            errors.append(f"{name}: insufficient year diversity: {sorted(years)}")
        if len(indicators) < 4:
            errors.append(f"{name}: insufficient indicator diversity: {sorted(indicators)}")
        if directions != {"higher", "lower"}:
            errors.append(f"{name}: expected higher and lower comparisons, found {sorted(directions)}")

    elif name == "environmental_layer_identification":
        answers = {record.get("target", {}).get("layer_id") for record in records}
        if len(answers) < 5:
            errors.append(f"{name}: target is still degenerate: {sorted(answers)}")

    elif name == "geo_entity_typing":
        ontology = records[0].get("input", {}).get("ontology", {})
        if set(ontology) != {"A", "H", "L", "P", "R", "S", "T", "U", "V"}:
            errors.append(f"{name}: incomplete GeoNames ontology")

    elif name == "metric_distance_computation":
        units = {record.get("target", {}).get("unit_id") for record in records}
        expected_units = {"metres", "kilometres", "miles", "nautical_miles"}
        if units != expected_units:
            errors.append(f"{name}: expected {sorted(expected_units)}, found {sorted(units, key=str)}")
        for index, record in enumerate(records, 1):
            target = record.get("target", {})
            if not isinstance(target.get("value"), (int, float)) or not isinstance(target.get("distance_m"), (int, float)):
                errors.append(f"{name}:{index}: missing numeric distance values")
                break

    elif name == "population_density_estimation":
        years = {record.get("target", {}).get("year") for record in records}
        if len(years) < 8:
            errors.append(f"{name}: insufficient year diversity: {sorted(years, key=str)}")
        if any(record.get("target", {}).get("indicator") != "EN.POP.DNST" for record in records):
            errors.append(f"{name}: unexpected indicator")

    elif name == "topological_directional_reasoning":
        if any(record.get("input", {}).get("visual_geometry") != "polygon" for record in records):
            errors.append(f"{name}: point-based or unspecified visual geometry remains")
        if any(record.get("target", {}).get("geometry_representation") != "polygon" for record in records):
            errors.append(f"{name}: target does not declare polygon representation")

    elif name in {"spatial_graph_construction", "shortest_path_optimization"}:
        for index, record in enumerate(records, 1):
            images = record.get("input", {}).get("images", [])
            if len(images) != 1 or Path(images[0]).suffix.lower() != ".png":
                errors.append(f"{name}:{index}: input must be one display-ready PNG")
                break
            problem = _validate_visible_rgb(task_dir / images[0])
            if problem:
                errors.append(f"{name}:{index}: {problem}")
                break
            if name == "spatial_graph_construction" and not record.get("target", {}).get("graph_image"):
                errors.append(f"{name}:{index}: missing graph overlay image")
                break
            if name == "shortest_path_optimization" and record.get("target", {}).get("unit") != "metres":
                errors.append(f"{name}:{index}: route length unit must be metres")
                break

    elif name == "isochrone_service_area":
        cities = {record.get("input", {}).get("city") for record in records}
        speeds = {record.get("input", {}).get("speed_mps") for record in records}
        budgets = {record.get("input", {}).get("budget_minutes") for record in records}
        methods = {record.get("target", {}).get("construction_method") for record in records}
        if len(cities) < 15:
            errors.append(f"{name}: only {len(cities)} cities")
        if len(speeds) < 5 or len(budgets) < 4:
            errors.append(f"{name}: insufficient speed/budget diversity")
        if methods != {"buffered reachable street edges in a local projected CRS"}:
            errors.append(f"{name}: unexpected polygon construction methods: {methods}")

    elif name == "map_label_feature_anchoring":
        if any("label_anchor" not in record.get("target", {}) for record in records):
            errors.append(f"{name}: missing label-anchor coordinates")
        if any(record.get("input", {}).get("task_definition") != "text-to-geographic-feature grounding" for record in records):
            errors.append(f"{name}: unclear task definition")

    elif name == "map_text_detection_recognition_grouping":
        for index, record in enumerate(records, 1):
            groups = record.get("target", {}).get("groups", [])
            if len(groups) < 2:
                errors.append(f"{name}:{index}: fewer than two label groups")
                break
            if any(not group.get("group_id") or not group.get("words") for group in groups):
                errors.append(f"{name}:{index}: malformed group annotations")
                break

    elif name == "dense_land_cover_labeling":
        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:
            errors.append(f"{name}: cannot validate masks: {error}")
            return errors
        allowed = set(range(8)) | {255}
        for index, record in enumerate(records, 1):
            image_path = task_dir / record["input"]["images"][0]
            mask_path = task_dir / record["target"]["mask"]
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    errors.append(f"{name}:{index}: image/mask size mismatch")
                    break
                values = set(int(value) for value in np.unique(np.asarray(mask)))
                if not values.issubset(allowed):
                    errors.append(f"{name}:{index}: invalid mask IDs {sorted(values - allowed)}")
                    break
                if mask.mode != "L":
                    errors.append(f"{name}:{index}: mask must be single-channel L mode, found {mask.mode}")
                    break
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

