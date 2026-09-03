from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .scoring import extract_answer, is_artifact_target, score as strict_score


TASK_METRIC_REVISION = "2026-09-task-aware-v1"
ARTIFACT_PROTOCOL_REVISION = "2026-09-inline-artifact-v1"
MASK_SIZE = (64, 64)

# No leaf exposes more than three paper-facing metrics. ``task_aware_score`` is
# the normalized per-record primary score used for cross-task macro averaging;
# the remaining metrics are task-specific diagnostics.
TASK_METRIC_SCHEMA: dict[str, tuple[str, ...]] = {
    "cartographic_symbol_recognition": ("accuracy", "macro_f1"),
    "map_text_detection_recognition_grouping": ("task_aware_score", "text_similarity", "structure_f1"),
    "map_label_feature_anchoring": ("task_aware_score", "field_f1", "geometry_similarity"),
    "dense_land_cover_labeling": ("task_aware_score", "mask_miou", "mask_dice"),
    "remote_sensing_scene_classification": ("accuracy", "macro_f1"),
    "object_presence_counting": ("task_aware_score", "count_mae", "accuracy"),
    "change_localization": ("task_aware_score", "mask_iou", "object_f1"),
    "temporal_scene_matching": ("accuracy", "field_f1"),
    "visual_geolocation": ("task_aware_score", "location_field_accuracy", "haversine_km"),
    "coordinate_transformation": ("task_aware_score", "coordinate_error", "tolerance_accuracy"),
    "metric_distance_computation": ("task_aware_score", "relative_error", "tolerance_accuracy"),
    "topological_directional_reasoning": ("task_aware_score", "relation_f1", "accuracy"),
    "spatial_graph_construction": ("task_aware_score", "node_f1", "edge_f1"),
    "shortest_path_optimization": ("task_aware_score", "path_validity", "relative_cost_error"),
    "isochrone_service_area": ("task_aware_score", "polygon_iou", "relative_error"),
    "toponym_recognition": ("task_aware_score", "entity_f1", "exact_match"),
    "geo_entity_typing": ("accuracy", "macro_f1"),
    "textual_spatial_relation_extraction": ("task_aware_score", "relation_f1", "exact_match"),
    "cross_entity_comparison": ("task_aware_score", "field_f1", "relative_error"),
    "environmental_layer_identification": ("accuracy", "macro_f1"),
    "population_density_estimation": ("task_aware_score", "relative_error", "rank_correlation"),
    "geologic_geomorphic_interpretation": ("task_aware_score", "field_f1", "exact_match"),
    "geographic_fact_reasoning": ("task_aware_score", "exact_match", "token_f1"),
}

LOWER_IS_BETTER = {"count_mae", "haversine_km", "coordinate_error", "relative_error", "relative_cost_error"}


def _gold(record: dict[str, Any]) -> Any:
    target = record.get("target") or {}
    path = str((record.get("evaluation") or {}).get("target_field") or "")
    if path.startswith("target."):
        return target.get(path.split(".", 1)[1])
    for key in ("bloom_answer", "answer", "value", "layer_id"):
        if key in target:
            return target[key]
    return target


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", str(value).strip().casefold()).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _relative_error(prediction: Any, gold: Any) -> float | None:
    p, g = _number(prediction), _number(gold)
    if p is None or g is None:
        return None
    return abs(p - g) / max(abs(g), 1e-9)


def _soft_numeric(prediction: Any, gold: Any) -> float:
    error = _relative_error(prediction, gold)
    return 0.0 if error is None else max(0.0, 1.0 - min(error, 1.0))


def _tokens(value: Any) -> list[str]:
    return [token for token in _norm(value).split() if token]


def _f1(prediction: list[str], gold: list[str]) -> float:
    p, g = Counter(prediction), Counter(gold)
    denominator = sum(p.values()) + sum(g.values())
    return 1.0 if denominator == 0 else 2.0 * sum((p & g).values()) / denominator


def _token_f1(prediction: Any, gold: Any) -> float:
    return _f1(_tokens(prediction), _tokens(gold))


def _atomic(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for key in sorted(value):
            child = value[key]
            if isinstance(child, (dict, list)):
                values.extend(f"{key}:{item}" for item in _atomic(child))
            else:
                values.append(f"{key}:{_norm(child)}")
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_atomic(child))
        return values
    return [_norm(value)]


def _structured_similarity(prediction: Any, gold: Any) -> float:
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        return _soft_numeric(prediction, gold)
    if isinstance(gold, str) or isinstance(gold, bool) or gold is None:
        return float(_norm(prediction) == _norm(gold))
    if isinstance(gold, dict):
        if not isinstance(prediction, dict):
            return 0.0
        values = [
            _structured_similarity(prediction.get(key), value)
            for key, value in gold.items()
        ]
        return sum(values) / len(values) if values else 1.0
    if isinstance(gold, list):
        if not isinstance(prediction, list):
            return 0.0
        if all(not isinstance(item, (dict, list)) for item in gold + prediction):
            return _f1([_norm(item) for item in prediction], [_norm(item) for item in gold])
        return _f1(_atomic(prediction), _atomic(gold))
    return float(_norm(prediction) == _norm(gold))


def _edit_similarity(prediction: Any, gold: Any) -> float:
    first, second = _norm(prediction), _norm(gold)
    if not first and not second:
        return 1.0
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(first), len(second), 1))


def artifact_contract(record: dict[str, Any]) -> str | None:
    if not is_artifact_target(record):
        return None
    leaf = str(record.get("leaf") or "")
    variant = str((record.get("bloom") or {}).get("variant") or "")
    if leaf == "dense_land_cover_labeling":
        return (
            "Artifact answer contract: do not return a filename. Return a 64x64 class-index mask as "
            '{"answer":{"encoding":"rle-row-major","size":[64,64],"runs":[[class_id,count],...]}}. '
            "Runs must be positive, consecutive, row-major, and sum to 4096. Use only IDs from class_ontology."
        )
    if leaf == "change_localization":
        mask = (
            '{"encoding":"rle-row-major","size":[64,64],"runs":[[binary_value,count],...]}'
        )
        if variant == "create_change_annotation":
            return (
                "Artifact answer contract: do not return a filename. Return "
                f'{{"answer":{{"mask":{mask},"changed_object_ids":[...]}}}}. '
                "Binary values must be 0 or 1; positive row-major run counts must sum to 4096."
            )
        return (
            "Artifact answer contract: do not return a filename. Return "
            f'{{"answer":{mask}}}. Binary values must be 0 or 1 and counts must sum to 4096.'
        )
    if leaf == "spatial_graph_construction":
        return (
            "Artifact answer contract: do not return a filename. Return the graph inline as "
            '{"answer":{"directed":false,"nodes":[{"id":0,"x":0.0,"y":0.0},...],'
            '"edges":[{"source":0,"target":1,"length":0.0},...]}}. '
            "Node IDs must be unique and every edge endpoint must reference a listed node."
        )
    return None


def _rle_payload(prediction: Any) -> dict[str, Any] | None:
    if isinstance(prediction, dict) and isinstance(prediction.get("mask"), dict):
        prediction = prediction["mask"]
    if not isinstance(prediction, dict) or prediction.get("encoding") != "rle-row-major":
        return None
    return prediction


def _decode_rle(prediction: Any) -> np.ndarray | None:
    payload = _rle_payload(prediction)
    if payload is None or list(payload.get("size") or []) != list(MASK_SIZE):
        return None
    values: list[int] = []
    try:
        for run in payload.get("runs") or []:
            value, count = int(run[0]), int(run[1])
            if count <= 0 or len(values) + count > MASK_SIZE[0] * MASK_SIZE[1]:
                return None
            values.extend([value] * count)
    except (TypeError, ValueError, IndexError):
        return None
    if len(values) != MASK_SIZE[0] * MASK_SIZE[1]:
        return None
    return np.asarray(values, dtype=np.int32).reshape(MASK_SIZE)


def _gold_mask(record: dict[str, Any], task_dir: Path) -> tuple[np.ndarray, int] | None:
    gold = _gold(record)
    relative = gold.get("change_mask") if isinstance(gold, dict) else gold
    if not isinstance(relative, str):
        return None
    path = (task_dir / relative).resolve()
    if not path.is_file():
        return None
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize(MASK_SIZE[::-1], Image.Resampling.NEAREST), dtype=np.int32)
    ignore = int((record.get("target") or {}).get("ignore_index", 255))
    if str(record.get("leaf")) == "change_localization":
        array = (array > 0).astype(np.int32)
        ignore = -1
    return array, ignore


def _mask_metrics(record: dict[str, Any], prediction: Any, task_dir: Path) -> dict[str, float]:
    predicted = _decode_rle(prediction)
    target = _gold_mask(record, task_dir)
    if predicted is None or target is None:
        return {"mask_miou": 0.0, "mask_dice": 0.0}
    gold, ignore = target
    valid = gold != ignore
    labels = sorted(set(gold[valid].tolist()) | set(predicted[valid].tolist()))
    ious: list[float] = []
    dices: list[float] = []
    for label in labels:
        p, g = (predicted == label) & valid, (gold == label) & valid
        union, total = int((p | g).sum()), int(p.sum() + g.sum())
        if union:
            ious.append(float((p & g).sum()) / union)
        if total:
            dices.append(2.0 * float((p & g).sum()) / total)
    return {
        "mask_miou": sum(ious) / len(ious) if ious else 0.0,
        "mask_dice": sum(dices) / len(dices) if dices else 0.0,
    }


def _graph_metrics(record: dict[str, Any], prediction: Any, task_dir: Path) -> dict[str, float]:
    if not isinstance(prediction, dict):
        return {"node_f1": 0.0, "edge_f1": 0.0, "length_similarity": 0.0}
    relative = _gold(record)
    if not isinstance(relative, str) or not (task_dir / relative).is_file():
        return {"node_f1": 0.0, "edge_f1": 0.0, "length_similarity": 0.0}
    gold = json.loads((task_dir / relative).read_text(encoding="utf-8"))

    def nodes(payload: dict[str, Any]) -> dict[int, tuple[float, float]]:
        result: dict[int, tuple[float, float]] = {}
        for row in payload.get("nodes") or []:
            try:
                result[int(row["id"])] = (float(row["x"]), float(row["y"]))
            except (KeyError, TypeError, ValueError):
                continue
        return result

    p_nodes, g_nodes = nodes(prediction), nodes(gold)
    if not g_nodes:
        return {"node_f1": 0.0, "edge_f1": 0.0, "length_similarity": 0.0}
    xs = [point[0] for point in g_nodes.values()]
    ys = [point[1] for point in g_nodes.values()]
    geographic = all(-180 <= x <= 180 for x in xs) and all(-90 <= y <= 90 for y in ys)
    tolerance = 0.00015 if geographic else max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 0.01
    unmatched = set(g_nodes)
    mapping: dict[int, int] = {}
    for p_id, point in p_nodes.items():
        candidates = sorted(
            ((math.hypot(point[0] - g_nodes[g][0], point[1] - g_nodes[g][1]), g) for g in unmatched),
        )
        if candidates and candidates[0][0] <= tolerance:
            mapping[p_id] = candidates[0][1]
            unmatched.remove(candidates[0][1])
    node_f1 = 2 * len(mapping) / max(len(p_nodes) + len(g_nodes), 1)

    def edges(payload: dict[str, Any], node_map: dict[int, int] | None = None) -> tuple[set[tuple[int, int]], float]:
        result: set[tuple[int, int]] = set()
        total = 0.0
        for row in payload.get("edges") or []:
            try:
                a, b = int(row["source"]), int(row["target"])
                total += max(0.0, float(row.get("length", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if node_map is not None:
                if a not in node_map or b not in node_map:
                    continue
                a, b = node_map[a], node_map[b]
            result.add(tuple(sorted((a, b))))
        return result, total

    p_edges, p_length = edges(prediction, mapping)
    g_edges, g_length = edges(gold)
    edge_f1 = _f1([str(value) for value in p_edges], [str(value) for value in g_edges])
    length_error = abs(p_length - g_length) / max(g_length, 1e-9)
    return {
        "node_f1": node_f1,
        "edge_f1": edge_f1,
        "length_similarity": max(0.0, 1.0 - min(length_error, 1.0)),
    }


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    first, second = _number(value[0]), _number(value[1])
    return None if first is None or second is None else (first, second)


def _line_length(points: list[Any]) -> float:
    parsed = [_point(value) for value in points]
    if any(value is None for value in parsed):
        return 0.0
    result = 0.0
    for first, second in zip(parsed, parsed[1:]):
        assert first is not None and second is not None
        if all(abs(value) <= limit for value, limit in zip((*first, *second), (180, 90, 180, 90))):
            lon1, lat1, lon2, lat2 = map(math.radians, (*first, *second))
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            result += 6371008.8 * 2 * math.asin(min(1.0, math.sqrt(a)))
        else:
            result += math.hypot(first[0] - second[0], first[1] - second[1])
    return result


def _route_metrics(prediction: Any, gold: Any) -> dict[str, float]:
    if not isinstance(prediction, list) or not isinstance(gold, list) or len(prediction) < 2 or len(gold) < 2:
        return {"path_validity": 0.0, "relative_cost_error": 1.0, "route_similarity": 0.0}
    p_points, g_points = [_point(value) for value in prediction], [_point(value) for value in gold]
    if any(point is None for point in p_points + g_points):
        return {"path_validity": 0.0, "relative_cost_error": 1.0, "route_similarity": 0.0}
    assert all(point is not None for point in p_points + g_points)
    flat = [coordinate for point in g_points for coordinate in point]
    geographic = all(-180 <= flat[index] <= 180 and -90 <= flat[index + 1] <= 90 for index in range(0, len(flat), 2))
    tolerance = 0.00015 if geographic else max(max(flat) - min(flat), 1.0) * 0.01
    endpoints = float(
        math.dist(p_points[0], g_points[0]) <= tolerance
        and math.dist(p_points[-1], g_points[-1]) <= tolerance
    )
    matched = sum(any(math.dist(p, g) <= tolerance for g in g_points) for p in p_points)
    precision = matched / len(p_points)
    recall = sum(any(math.dist(g, p) <= tolerance for p in p_points) for g in g_points) / len(g_points)
    route_similarity = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    p_length, g_length = _line_length(prediction), _line_length(gold)
    relative = abs(p_length - g_length) / max(g_length, 1e-9)
    return {
        "path_validity": endpoints * route_similarity,
        "relative_cost_error": relative,
        "route_similarity": route_similarity,
    }


def _polygon_iou(prediction: Any, gold: Any) -> float | None:
    if not isinstance(prediction, dict) or not isinstance(gold, dict):
        return None
    try:
        from shapely.geometry import shape

        first, second = shape(prediction), shape(gold)
        union = first.union(second).area
        return 0.0 if union <= 0 else float(first.intersection(second).area / union)
    except Exception:
        return 0.0


def _haversine(prediction: Any, gold: Any) -> float | None:
    if not isinstance(prediction, dict) or not isinstance(gold, dict):
        return None
    def pair(value: dict[str, Any]) -> tuple[float, float] | None:
        lat = _number(value.get("latitude", value.get("lat")))
        lon = _number(value.get("longitude", value.get("lon")))
        return None if lat is None or lon is None else (lat, lon)
    p, g = pair(prediction), pair(gold)
    if p is None or g is None:
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, (*p, *g))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _kendall_score(prediction: Any, gold: Any) -> float | None:
    if not isinstance(prediction, list) or not isinstance(gold, list):
        return None
    p = [_norm(value) for value in prediction]
    g = [_norm(value) for value in gold]
    if len(p) != len(g) or set(p) != set(g) or len(g) < 2:
        return 0.0
    position = {value: index for index, value in enumerate(p)}
    concordant = discordant = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            concordant += position[g[i]] < position[g[j]]
            discordant += position[g[i]] > position[g[j]]
    tau = (concordant - discordant) / max(concordant + discordant, 1)
    return (tau + 1.0) / 2.0


def evaluate_task_aware(record: dict[str, Any], response_text: str, task_dir: Path) -> dict[str, Any]:
    prediction, parse_error = extract_answer(response_text)
    gold = _gold(record)
    strict = strict_score(record, response_text)
    leaf = str(record.get("leaf") or "unknown")
    variant = str((record.get("bloom") or {}).get("variant") or "")
    metrics: dict[str, float] = {}
    task_score = 0.0 if parse_error else _structured_similarity(prediction, gold)

    if parse_error:
        pass
    elif is_artifact_target(record) and leaf in {"dense_land_cover_labeling", "change_localization"}:
        mask = _mask_metrics(record, prediction, task_dir)
        metrics.update(mask)
        task_score = mask["mask_miou"]
        if leaf == "change_localization":
            metrics["mask_iou"] = metrics.pop("mask_miou")
            if isinstance(prediction, dict) and isinstance(gold, dict):
                metrics["object_f1"] = _f1(
                    [_norm(value) for value in prediction.get("changed_object_ids", [])],
                    [_norm(value) for value in gold.get("changed_object_ids", [])],
                )
                task_score = (metrics["mask_iou"] + metrics["object_f1"]) / 2
    elif is_artifact_target(record) and leaf == "spatial_graph_construction":
        graph = _graph_metrics(record, prediction, task_dir)
        metrics.update(graph)
        task_score = (graph["node_f1"] + graph["edge_f1"]) / 2
    elif leaf == "shortest_path_optimization" and isinstance(gold, list):
        route = _route_metrics(prediction, gold)
        metrics.update(route)
        task_score = (route["path_validity"] + max(0.0, 1.0 - min(route["relative_cost_error"], 1.0))) / 2
    elif leaf == "isochrone_service_area" and isinstance(gold, dict) and "type" in gold:
        iou = _polygon_iou(prediction, gold)
        metrics["polygon_iou"] = float(iou or 0.0)
        task_score = metrics["polygon_iou"]
    elif leaf == "population_density_estimation" and isinstance(gold, list):
        rank = _kendall_score(prediction, gold)
        if rank is not None:
            metrics["rank_correlation"] = rank
            task_score = rank

    kind = str((record.get("evaluation") or {}).get("type") or "")
    if not parse_error:
        if leaf in {"cartographic_symbol_recognition", "remote_sensing_scene_classification", "geo_entity_typing", "environmental_layer_identification"}:
            metrics["accuracy"] = float(strict["score"])
        if leaf in {"map_text_detection_recognition_grouping", "geographic_fact_reasoning"}:
            metrics["text_similarity"] = _edit_similarity(prediction, gold)
            metrics["token_f1"] = _token_f1(prediction, gold)
        if isinstance(gold, (dict, list)):
            metrics.setdefault("field_f1", _structured_similarity(prediction, gold))
            metrics.setdefault("structure_f1", _structured_similarity(prediction, gold))
        if isinstance(gold, list) and leaf in {"toponym_recognition", "textual_spatial_relation_extraction", "topological_directional_reasoning"}:
            name = "entity_f1" if leaf == "toponym_recognition" else "relation_f1"
            metrics[name] = _f1(_atomic(prediction), _atomic(gold)) if isinstance(prediction, list) else 0.0
            task_score = metrics[name]
        if leaf == "change_localization" and isinstance(gold, dict) and not is_artifact_target(record):
            p_ids = prediction.get("changed_object_ids", []) if isinstance(prediction, dict) else []
            g_ids = gold.get("changed_object_ids", [])
            metrics["object_f1"] = _f1([_norm(value) for value in p_ids], [_norm(value) for value in g_ids])
        if leaf == "visual_geolocation":
            if isinstance(gold, dict) and isinstance(prediction, dict):
                keys = [key for key in ("city", "country") if key in gold]
                if keys:
                    metrics["location_field_accuracy"] = sum(_norm(prediction.get(key)) == _norm(gold[key]) for key in keys) / len(keys)
                    task_score = metrics["location_field_accuracy"]
                distance = _haversine(prediction, gold)
                if distance is not None:
                    metrics["haversine_km"] = distance
            else:
                metrics["location_field_accuracy"] = float(strict["score"])
        if leaf == "coordinate_transformation" and isinstance(gold, list) and isinstance(prediction, list):
            p, g = [_number(value) for value in prediction], [_number(value) for value in gold]
            if len(p) == len(g) and p and all(value is not None for value in p + g):
                metrics["coordinate_error"] = math.dist(p, g)  # type: ignore[arg-type]
                tolerance = float((record.get("evaluation") or {}).get("absolute_tolerance", 1e-6))
                metrics["tolerance_accuracy"] = float(metrics["coordinate_error"] <= tolerance)
        if leaf in {"metric_distance_computation", "cross_entity_comparison", "population_density_estimation", "isochrone_service_area"}:
            error = _relative_error(prediction, gold)
            if error is not None:
                metrics["relative_error"] = error
                metrics["tolerance_accuracy"] = float(strict["score"])
                task_score = _soft_numeric(prediction, gold)
        if leaf == "object_presence_counting":
            p, g = _number(prediction), _number(gold)
            if p is not None and g is not None:
                metrics["count_mae"] = abs(p - g)
            else:
                metrics["accuracy"] = float(strict["score"])
        if leaf in {"toponym_recognition", "textual_spatial_relation_extraction", "geologic_geomorphic_interpretation", "geographic_fact_reasoning"}:
            metrics["exact_match"] = float(strict["score"])
        if kind in {"binary_classification", "classification", "exact_match"} and leaf not in {"geographic_fact_reasoning"}:
            metrics.setdefault("accuracy", float(strict["score"]))

    allowed = TASK_METRIC_SCHEMA.get(leaf, ("task_aware_score",))
    filtered = {key: float(value) for key, value in metrics.items() if key in allowed and math.isfinite(float(value))}
    filtered["task_aware_score"] = float(max(0.0, min(1.0, task_score)))
    return {
        "task_score": filtered["task_aware_score"],
        "task_metric": allowed[0],
        "task_metrics": filtered,
        "task_metric_revision": TASK_METRIC_REVISION,
        "artifact_protocol_revision": ARTIFACT_PROTOCOL_REVISION if is_artifact_target(record) else None,
        "strict_score": float(strict["score"]),
        "score": float(strict["score"]),
        "metric": strict["metric"],
        "parse_error": strict["parse_error"],
        "gold_hash": strict["gold_hash"],
    }
