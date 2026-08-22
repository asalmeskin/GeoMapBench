from __future__ import annotations

import copy
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import N_EXAMPLES, SEEDS, read_jsonl, sha256_file, utc_now, write_jsonl


BLOOM_REVISION = "2026-08-bloom-v1"
BLOOM_SEED_OFFSET = 860_000

BLOOM_LEVEL_NAMES: dict[str, str] = {
    "R": "Remember",
    "U": "Understand",
    "Ap": "Apply",
    "An": "Analyze",
    "E": "Evaluate",
    "C": "Create",
}

# Operational Bloom coverage agreed for the 23 GeoMapBench leaves.  The order
# also defines which levels receive the one extra example when 100 is not
# divisible by the number of levels (6-level leaves -> 17,17,17,17,16,16).
BLOOM_LEVELS: dict[str, tuple[str, ...]] = {
    "cartographic_symbol_recognition": ("R", "U", "Ap", "An", "E"),
    "map_text_detection_recognition_grouping": ("R", "U", "Ap", "An", "E", "C"),
    "map_label_feature_anchoring": ("R", "U", "Ap", "An", "E", "C"),
    "dense_land_cover_labeling": ("R", "U", "Ap", "An", "E", "C"),
    "remote_sensing_scene_classification": ("R", "U", "Ap", "An", "E"),
    "object_presence_counting": ("R", "U", "Ap", "An", "E"),
    "change_localization": ("U", "Ap", "An", "E", "C"),
    "temporal_scene_matching": ("U", "Ap", "An", "E", "C"),
    "visual_geolocation": ("R", "U", "Ap", "An", "E"),
    "coordinate_transformation": ("R", "U", "Ap", "An", "E", "C"),
    "metric_distance_computation": ("R", "U", "Ap", "An", "E"),
    "topological_directional_reasoning": ("R", "U", "Ap", "An", "E", "C"),
    "spatial_graph_construction": ("R", "U", "Ap", "An", "E", "C"),
    "shortest_path_optimization": ("R", "U", "Ap", "An", "E", "C"),
    "isochrone_service_area": ("R", "U", "Ap", "An", "E", "C"),
    "toponym_recognition": ("R", "U", "Ap", "An", "E"),
    "geo_entity_typing": ("R", "U", "Ap", "An", "E"),
    "textual_spatial_relation_extraction": ("R", "U", "Ap", "An", "E", "C"),
    "cross_entity_comparison": ("R", "U", "Ap", "An", "E", "C"),
    "environmental_layer_identification": ("R", "U", "Ap", "An", "E"),
    "population_density_estimation": ("R", "U", "Ap", "An", "E", "C"),
    "geologic_geomorphic_interpretation": ("R", "U", "Ap", "An", "E", "C"),
    "geographic_fact_reasoning": ("R", "U", "Ap", "An", "E"),
}

BLOOM_CURRENT_LEVELS: dict[str, tuple[str, ...]] = {
    "cartographic_symbol_recognition": ("U",),
    "map_text_detection_recognition_grouping": ("Ap", "An"),
    "map_label_feature_anchoring": ("An", "E"),
    "dense_land_cover_labeling": ("Ap", "An"),
    "remote_sensing_scene_classification": ("U",),
    "object_presence_counting": ("Ap", "An"),
    "change_localization": ("An", "C"),
    "temporal_scene_matching": ("An", "E"),
    "visual_geolocation": ("An", "E"),
    "coordinate_transformation": ("Ap",),
    "metric_distance_computation": ("Ap",),
    "topological_directional_reasoning": ("An",),
    "spatial_graph_construction": ("Ap", "An", "C"),
    "shortest_path_optimization": ("Ap", "An", "E"),
    "isochrone_service_area": ("Ap", "An", "C"),
    "toponym_recognition": ("U", "Ap"),
    "geo_entity_typing": ("U",),
    "textual_spatial_relation_extraction": ("An",),
    "cross_entity_comparison": ("Ap", "An"),
    "environmental_layer_identification": ("U", "An"),
    "population_density_estimation": ("R", "Ap"),
    "geologic_geomorphic_interpretation": ("U", "An"),
    "geographic_fact_reasoning": ("R", "U", "An"),
}


INVERSE_RELATION = {
    "north": "south",
    "northeast": "southwest",
    "east": "west",
    "southeast": "northwest",
    "south": "north",
    "southwest": "northeast",
    "west": "east",
    "northwest": "southeast",
    "within": "contains",
    "contains": "within",
    "touches": "touches",
    "intersects": "intersects",
    "overlaps": "overlaps",
    "near": "near",
}


def bloom_distribution(levels: Iterable[str], n: int = N_EXAMPLES) -> dict[str, int]:
    levels = tuple(levels)
    base, remainder = divmod(n, len(levels))
    return {level: base + (1 if i < remainder else 0) for i, level in enumerate(levels)}


def _assigned_levels(leaf: str, n: int) -> list[str]:
    levels = BLOOM_LEVELS[leaf]
    distribution = bloom_distribution(levels, n)
    assignments = [level for level in levels for _ in range(distribution[level])]
    random.Random(SEEDS[leaf] + BLOOM_SEED_OFFSET).shuffle(assignments)
    return assignments


def _wrong_choice(choices: Iterable[Any], correct: Any, seed: int) -> Any:
    alternatives = [item for item in choices if item != correct]
    if not alternatives:
        return correct
    alternatives = sorted(alternatives, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return alternatives[seed % len(alternatives)]


def _peer_value(
    records: list[dict[str, Any]],
    index: int,
    getter: Callable[[dict[str, Any]], Any],
) -> Any:
    correct = getter(records[index])
    for offset in range(1, len(records)):
        candidate = getter(records[(index + offset) % len(records)])
        if candidate is not None and candidate != correct:
            return candidate
    return correct


def _atomic_values(obj: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(obj, dict):
        for key in sorted(obj):
            out.extend(_atomic_values(obj[key]))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_atomic_values(value))
    elif obj is not None and not isinstance(obj, (dict, list)):
        out.append(obj)
    return out


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _response_eval(kind: str, *, tolerance: float | None = None, metrics: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": kind,
        "target_field": "target.bloom_answer",
    }
    if tolerance is not None:
        result["relative_tolerance"] = tolerance
    if metrics:
        result["metrics"] = metrics
    if kind in {"exact_match", "binary_classification", "classification"}:
        result["normalize"] = "lowercase_whitespace"
    return result


def _apply_variant(
    record: dict[str, Any],
    *,
    level: str,
    variant: str,
    question: str,
    answer: Any,
    evaluation: dict[str, Any],
    occurrence: int,
    extra_input: dict[str, Any] | None = None,
    choices: list[Any] | None = None,
    drop_choices: bool = False,
    source_record_ids: list[str] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(record)
    input_obj = out.setdefault("input", {})
    target_obj = out.setdefault("target", {})
    if "base_question" not in input_obj:
        input_obj["base_question"] = input_obj.get("question")
    input_obj["question"] = question
    if drop_choices:
        input_obj.pop("choices", None)
    if choices is not None:
        input_obj["choices"] = choices
    if extra_input:
        input_obj.update(copy.deepcopy(extra_input))

    target_obj["bloom_answer"] = copy.deepcopy(answer)
    target_obj["bloom_response_format"] = variant
    out["base_evaluation"] = copy.deepcopy(record.get("evaluation", {}))
    out["evaluation"] = {
        **copy.deepcopy(evaluation),
        "bloom_level": level,
        "bloom_level_name": BLOOM_LEVEL_NAMES[level],
    }
    source_ids = source_record_ids or [str(record.get("id", ""))]
    out["bloom"] = {
        "revision": BLOOM_REVISION,
        "level": level,
        "level_name": BLOOM_LEVEL_NAMES[level],
        "level_rank": list(BLOOM_LEVEL_NAMES).index(level) + 1,
        "variant": variant,
        "occurrence_within_level": occurrence,
        "source_record_ids": source_ids,
        "supported_levels": list(BLOOM_LEVELS[out["leaf"]]),
        "current_levels_before_expansion": list(BLOOM_CURRENT_LEVELS[out["leaf"]]),
    }
    return out


def _verification(correct: Any, wrong: Any, occurrence: int) -> tuple[Any, str]:
    if occurrence % 2 == 0 or wrong == correct:
        return correct, "yes"
    return wrong, "no"


def _class_name_from_ontology(ontology: dict[str, Any], class_id: Any) -> str:
    item = ontology.get(str(class_id), ontology.get(class_id))
    if isinstance(item, dict):
        return str(item.get("name", class_id))
    return str(item if item is not None else class_id)


def _load_graph_payload(task_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    rel = record.get("target", {}).get("graph")
    if not rel:
        return {}
    return json.loads((task_dir / rel).read_text(encoding="utf-8"))


def _mask_pixel(task_dir: Path, record: dict[str, Any], seed: int) -> tuple[int, int, int]:
    import numpy as np
    from PIL import Image

    target = record.get("target", {})
    mask_path = task_dir / target["mask"]
    ignore = int(target.get("ignore_index", 255))
    with Image.open(mask_path) as image:
        array = np.asarray(image)
    ys, xs = np.where(array != ignore)
    if len(xs) == 0:
        raise ValueError(f"{record.get('id')}: no labeled pixels in mask")
    k = seed % len(xs)
    x, y = int(xs[k]), int(ys[k])
    return x, y, int(array[y, x])


def _transform_maki(record, level, occurrence, records, index, task_dir):
    answer = record["target"].get("answer")
    choices = list(record.get("input", {}).get("choices", []))
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="symbol_label_recall",
            question="Identify the canonical Maki point-of-interest category represented by this symbol.",
            answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="symbol_semantic_interpretation",
            question="What real-world point-of-interest category does this cartographic symbol communicate to a map reader?",
            answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_symbol_legend",
            question="A cartographer uses this exact symbol to mark a location. Assign the location the correct POI category from the choices.",
            answer=answer, evaluation=_response_eval("classification"), choices=choices)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="discriminate_symbol_candidates",
            question="Analyze the symbol and discriminate among the candidate POI categories. Which choice is the best semantic match?",
            answer=answer, evaluation=_response_eval("classification"), choices=choices)
    wrong = _wrong_choice(choices, answer, occurrence)
    candidate, verdict = _verification(answer, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="verify_symbol_claim",
        question=f'A reviewer claims that this symbol represents the POI category "{candidate}". Is the claim correct? Answer yes or no.',
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_category": candidate})


def _transform_maptext(record, level, occurrence, records, index, task_dir):
    target = record["target"]
    groups = copy.deepcopy(target.get("groups", []))
    words = copy.deepcopy(target.get("words", []))
    if level == "R":
        word = words[occurrence % len(words)]
        vertices = word.get("vertices", [])
        return _apply_variant(record, level=level, occurrence=occurrence, variant="single_word_transcription",
            question=f"Transcribe only the map word enclosed by this polygon (image pixel coordinates): {vertices}.",
            answer=word.get("text"), evaluation=_response_eval("exact_match"), drop_choices=True,
            extra_input={"query_polygon": vertices})
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="label_group_count",
            question="How many complete geographic label groups are visible in this crop? Return one integer.",
            answer=int(target.get("group_count", len(groups))), evaluation=_response_eval("numeric", tolerance=0.0), drop_choices=True)
    if level == "Ap":
        stripped = [{k: v for k, v in word.items() if k != "group_id"} for word in words]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="word_detection_and_recognition",
            question="Detect every visible map-text word and return each word polygon with its transcription. Group IDs are not required.",
            answer=stripped, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "An":
        provided = [{k: v for k, v in word.items() if k != "group_id"} for word in words]
        answer = [
            {"group_id": group.get("group_id"), "text": group.get("text"), "word_texts": [w.get("text") for w in group.get("words", [])]}
            for group in groups
        ]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="phrase_grouping_from_detected_words",
            question="The word detections and transcriptions are provided. Analyze them and group words that form the same complete geographic map label.",
            answer=answer, evaluation=_response_eval("structured_exact"), drop_choices=True,
            extra_input={"detected_words": provided})
    if level == "E":
        assignments = [
            {"text": w.get("text"), "vertices": w.get("vertices"), "group_id": w.get("group_id")}
            for w in words
        ]
        candidate = copy.deepcopy(assignments)
        verdict = "yes"
        if occurrence % 2 == 1 and len(groups) >= 2 and candidate:
            current = candidate[0].get("group_id")
            other = next((g.get("group_id") for g in groups if g.get("group_id") != current), current)
            candidate[0]["group_id"] = other
            verdict = "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_grouping_annotation",
            question="Evaluate the proposed word-to-label grouping against the map crop. Is the proposed grouping fully correct? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], drop_choices=False,
            extra_input={"candidate_grouping": candidate})
    answer = {"groups": groups, "words": words, "group_count": int(target.get("group_count", len(groups)))}
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_complete_map_text_annotation",
        question="Create the complete map-text annotation: detect every visible word polygon, transcribe each word, and construct group IDs for words belonging to the same complete label.",
        answer=answer, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _transform_label_anchor(record, level, occurrence, records, index, task_dir):
    target = record["target"]
    correct = target.get("answer")
    choices = list(record.get("input", {}).get("choices", ["A", "B", "C", "D"]))
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="map_label_transcription",
            question="Transcribe the geographic label printed at the target anchor in the map image.",
            answer=target.get("feature_name"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="geometry_family_interpretation",
            question="What geometry family does the labeled geographic feature belong to: point, line, or polygon?",
            answer=target.get("geometry_family"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_label_grounding",
            question="Which candidate geometry (A, B, C, or D) is the geographic feature named by the visible map label?",
            answer=correct, evaluation=_response_eval("classification"), choices=choices)
    if level == "An":
        answer = {"candidate": correct, "geometry_family": target.get("geometry_family")}
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_label_geometry_match",
            question="Analyze both label placement and candidate geometry. Return the correct candidate letter and its geometry family.",
            answer=answer, evaluation=_response_eval("structured_exact"), choices=choices)
    if level == "E":
        wrong = _wrong_choice(choices, correct, occurrence)
        candidate, verdict = _verification(correct, wrong, occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_grounding_candidate",
            question=f"A proposed grounding assigns the visible label to candidate {candidate}. Is that grounding correct? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_geometry": candidate})
    answer = {
        "label": target.get("feature_name"),
        "candidate": correct,
        "feature_id": target.get("feature_id"),
        "geometry_family": target.get("geometry_family"),
        "label_anchor": target.get("label_anchor"),
        "geometry": target.get("geometry"),
    }
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_grounding_record",
        question="Create a structured label-to-feature grounding record containing the label text, selected candidate, feature ID, geometry family, label anchor, and target geometry.",
        answer=answer, evaluation=_response_eval("structured_generation"), choices=choices)


def _transform_landcover(record, level, occurrence, records, index, task_dir):
    input_obj, target = record["input"], record["target"]
    ontology = input_obj.get("class_ontology") or target.get("class_ontology") or {}
    class_ids = sorted((str(k) for k in ontology), key=lambda x: int(x) if x.isdigit() else x)
    counts = {str(k): int(v) for k, v in target.get("pixel_counts", {}).items() if str(k) != str(target.get("ignore_index", 255))}
    dominant_id = max(counts, key=counts.get) if counts else class_ids[0]
    dominant_name = _class_name_from_ontology(ontology, dominant_id)
    if level == "R":
        class_id = class_ids[occurrence % len(class_ids)]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="ontology_class_recall",
            question=f"In the OpenEarthMap ontology used by this benchmark, what land-cover class name corresponds to class ID {class_id}?",
            answer=_class_name_from_ontology(ontology, class_id), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="dominant_landcover_interpretation",
            question="Which land-cover class occupies the largest number of labeled pixels in this image?",
            answer=dominant_name, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        x, y, class_id = _mask_pixel(task_dir, record, SEEDS[record["leaf"]] + index + occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="pixelwise_class_application",
            question=f"Apply the OpenEarthMap ontology at image pixel (x={x}, y={y}), using an upper-left image origin. What class should this pixel receive?",
            answer=_class_name_from_ontology(ontology, class_id), evaluation=_response_eval("exact_match"), drop_choices=True,
            extra_input={"query_pixel": {"x": x, "y": y, "origin": "upper-left"}})
    if level == "An":
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], int(kv[0]) if kv[0].isdigit() else kv[0]))[:3]
        answer = [
            {"class_id": int(cid) if cid.isdigit() else cid, "class_name": _class_name_from_ontology(ontology, cid), "pixel_count": count}
            for cid, count in ranked
        ]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="landcover_composition_analysis",
            question="Analyze the segmentation target implied by the image and rank the three most prevalent labeled land-cover classes by pixel count.",
            answer=answer, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        wrong_id = _wrong_choice(class_ids, dominant_id, occurrence)
        candidate, verdict = _verification(dominant_id, wrong_id, occurrence)
        candidate_name = _class_name_from_ontology(ontology, candidate)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_dominant_class_claim",
            question=f'A reviewer claims that "{candidate_name}" is the dominant labeled land-cover class in this image. Is the claim correct? Answer yes or no.',
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_class": candidate_name})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_semantic_segmentation",
        question="Create the complete single-channel per-pixel OpenEarthMap semantic segmentation mask for this image using the declared class-index ontology.",
        answer=target.get("mask"), evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)


def _transform_eurosat(record, level, occurrence, records, index, task_dir):
    answer = record["target"].get("answer")
    choices = list(record.get("input", {}).get("choices", []))
    wrong = _wrong_choice(choices, answer, occurrence)
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="scene_label_recognition",
            question="Recognize the canonical EuroSAT scene label represented by this satellite patch.",
            answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="scene_semantic_interpretation",
            question="Which land-use/land-cover concept best describes this satellite scene?",
            answer=answer, evaluation=_response_eval("classification"), choices=choices)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_eurosat_ontology",
            question="Apply the EuroSAT class ontology to this unseen patch and return the correct scene class.",
            answer=answer, evaluation=_response_eval("classification"), choices=choices)
    if level == "An":
        pair = [answer, wrong]
        random.Random(SEEDS[record["leaf"]] + index).shuffle(pair)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="discriminate_scene_candidates",
            question="Analyze the spatial texture and land-use pattern. Which of these two candidate classes is better supported by the image?",
            answer=answer, evaluation=_response_eval("classification"), choices=pair)
    candidate, verdict = _verification(answer, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_scene_claim",
        question=f'A reviewer labels this patch as "{candidate}". Is that classification correct? Answer yes or no.',
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_class": candidate})


def _rsvqa_mode(record: dict[str, Any]) -> str:
    text = (str(record.get("target", {}).get("question_type", "")) + " " + str(record.get("input", {}).get("base_question") or record.get("input", {}).get("question", ""))).lower()
    return "counting" if ("count" in text or "how many" in text or "number of" in text) else "presence"


def _transform_rsvqa(record, level, occurrence, records, index, task_dir):
    original = record.get("input", {}).get("question", "")
    answer = record.get("target", {}).get("answer")
    mode = _rsvqa_mode(record)
    if level == "R":
        question = f"Recognize the requested object fact in the remote-sensing image and give only the answer. Query: {original}"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="object_fact_recognition",
            question=question, answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        question = f"Interpret the scene and answer the object {mode} question. Query: {original}"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="object_query_interpretation",
            question=question, answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        question = f"Apply visual {mode} to the image and answer this query: {original}"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_object_presence_counting",
            question=question, answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "An":
        question = f"Analyze all relevant image regions before answering the query. Return both the answer and whether the operation is presence detection or counting. Query: {original}"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_object_query",
            question=question, answer={"answer": answer, "operation": mode}, evaluation=_response_eval("structured_exact"), drop_choices=True)
    wrong = _peer_value(records, index, lambda r: r.get("target", {}).get("answer"))
    candidate, verdict = _verification(answer, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_object_answer",
        question=f"For the image and query below, a reviewer proposes the answer {json.dumps(candidate)}. Is it correct? Answer yes or no. Query: {original}",
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_answer": candidate})


def _transform_change(record, level, occurrence, records, index, task_dir):
    target = record["target"]
    ids = list(target.get("changed_object_ids", []))
    count = len(ids)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="change_count_interpretation",
            question="How many tracked building identities changed between the two timestamps? Return one integer.",
            answer=count, evaluation=_response_eval("numeric", tolerance=0.0), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_change_localization",
            question="Localize all building construction or demolition between the two dates as a binary change mask.",
            answer=target.get("change_mask"), evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_changed_objects",
            question="Analyze the temporal pair and return the complete set of changed tracked building IDs together with the total number of changed objects.",
            answer={"changed_object_ids": ids, "changed_object_count": count}, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        proposed = count if occurrence % 2 == 0 else count + 1
        verdict = "yes" if proposed == count else "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_change_count_claim",
            question=f"A reviewer claims that {proposed} tracked building objects changed between these dates. Is that claim correct? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_changed_object_count": proposed})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_change_annotation",
        question="Create the complete temporal change annotation: a binary change mask and the set of tracked building IDs that were constructed or demolished.",
        answer={"change_mask": target.get("change_mask"), "changed_object_ids": ids},
        evaluation=_response_eval("structured_generation"), drop_choices=True)


def _positive_temporal_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if bool(r.get("target", {}).get("same_area")) and len(r.get("input", {}).get("images", [])) == 2]


def _transform_temporal(record, level, occurrence, records, index, task_dir):
    same = bool(record.get("target", {}).get("same_area"))
    answer = "yes" if same else "no"
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="temporal_pair_interpretation",
            question="Interpret the two observations: do they depict the same geographic area at different times?",
            answer=answer, evaluation=_response_eval("binary_classification"), choices=["yes", "no"])
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_temporal_matching",
            question="Apply temporal scene matching to this pair and decide whether the geographic area is the same despite temporal appearance changes.",
            answer=answer, evaluation=_response_eval("binary_classification"), choices=["yes", "no"])
    if level == "An":
        relation = "same geographic area across time" if same else "different geographic areas"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_temporal_relation",
            question="Analyze the invariant spatial layout and changing visual content. Return the match decision and the temporal-pair relation.",
            answer={"same_area": same, "relation": relation}, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        proposed = same if occurrence % 2 == 0 else not same
        verdict = "yes" if proposed == same else "no"
        label = "the same geographic area" if proposed else "different geographic areas"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_temporal_match_claim",
            question=f"A reviewer claims that these images show {label}. Is the claim correct? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_same_area": proposed})

    positives = _positive_temporal_pairs(records)
    if len(positives) < 2:
        return _apply_variant(record, level=level, occurrence=occurrence, variant="create_temporal_match_record",
            question="Create a structured temporal match record for the two images, indicating whether they represent the same geographic area.",
            answer={"pair": [0, 1], "same_area": same}, evaluation=_response_eval("structured_generation"), drop_choices=True)
    first = positives[(2 * occurrence) % len(positives)]
    second = positives[(2 * occurrence + 1) % len(positives)]
    if second.get("group_id") == first.get("group_id"):
        for candidate in positives:
            if candidate.get("group_id") != first.get("group_id"):
                second = candidate
                break
    images = list(first["input"]["images"]) + list(second["input"]["images"])
    source_ids = [str(first.get("id")), str(second.get("id"))]
    out = _apply_variant(record, level=level, occurrence=occurrence, variant="create_temporal_pairing",
        question="Construct the pairing of these four images into the two pairs that depict the same geographic area at different times. Return 0-based index pairs.",
        answer={"pairs": [[0, 1], [2, 3]]}, evaluation=_response_eval("pair_construction"), drop_choices=True,
        extra_input={"images": images, "image_indices": [0, 1, 2, 3]}, source_record_ids=source_ids)
    out["group_id"] = f"{first.get('group_id')}|{second.get('group_id')}"
    return out


def _transform_geolocation(record, level, occurrence, records, index, task_dir):
    target = record["target"]
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="country_recognition",
            question="Which country was this geocoded photograph captured in?",
            answer=target.get("country"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="city_interpretation",
            question="Interpret the geographic visual cues and identify the city represented by this photograph.",
            answer=target.get("city"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_city_country_geolocation",
            question="Apply visual geolocation and return the city and country for this image.",
            answer={"city": target.get("city"), "country": target.get("country")}, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "An":
        answer = {k: target.get(k) for k in ("city", "country", "latitude", "longitude")}
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_full_geolocation",
            question="Analyze the image using multiple geographic cues and return city, country, and approximate latitude/longitude.",
            answer=answer, evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)
    correct_location = {"city": target.get("city"), "country": target.get("country")}
    peer = None
    for offset in range(1, len(records)):
        candidate_target = records[(index + offset) % len(records)].get("target", {})
        candidate_location = {"city": candidate_target.get("city"), "country": candidate_target.get("country")}
        if candidate_location != correct_location:
            peer = candidate_target
            break
    peer = peer or target
    candidate = correct_location
    if occurrence % 2 == 1:
        candidate = {"city": peer.get("city"), "country": peer.get("country")}
    verdict = "yes" if candidate == correct_location else "no"
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_geolocation_candidate",
        question=f"A reviewer proposes {candidate['city']}, {candidate['country']} as the location of this image. Is that location correct? Answer yes or no.",
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_location": candidate})


def _coordinate_pair(target: dict[str, Any]) -> dict[str, Any]:
    axes = list(target.get("axis_order", []))
    return {**{axis: target.get(axis) for axis in axes}, "axis_order": axes, "unit": target.get("unit"), "crs": target.get("crs")}


def _transform_coordinate(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    pair = _coordinate_pair(target)
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="crs_unit_recall",
            question=f"What unit is used for coordinates in the target CRS {inp.get('target_crs')} for this transformation?",
            answer=target.get("unit"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        kind = "geographic" if target.get("unit") == "degrees" else "projected"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="crs_type_interpretation",
            question=f"Is the target CRS {inp.get('target_crs')} geographic or projected in this record?",
            answer=kind, evaluation=_response_eval("classification"), choices=["geographic", "projected"])
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_coordinate_transform",
            question=inp.get("base_question") or inp.get("question") or "Transform the coordinate.",
            answer=pair, evaluation=_response_eval("numeric_coordinate_pair"), drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_coordinate_representation",
            question="Transform the coordinate and report the transformed coordinate together with target CRS, axis order, and unit.",
            answer=pair, evaluation=_response_eval("structured_numeric"), drop_choices=True)
    if level == "E":
        candidate = copy.deepcopy(pair)
        verdict = "yes"
        if occurrence % 2 == 1:
            axes = pair.get("axis_order", [])
            if axes:
                axis = axes[0]
                value = _numeric(candidate.get(axis))
                if value is not None:
                    tol = float(record.get("evaluation", {}).get("absolute_tolerance", 1.0))
                    candidate[axis] = value + max(tol * 10.0, abs(value) * 1e-3, 1e-4)
                    verdict = "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_coordinate_result",
            question=f"Evaluate this proposed transformed coordinate: {json.dumps(candidate, sort_keys=True)}. Is it correct within the task tolerance? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_coordinate": candidate})
    answer = {
        "source_crs": inp.get("source_crs"),
        "target_crs": inp.get("target_crs"),
        "source_coordinate": inp.get("coordinate"),
        "transformed_coordinate": pair,
        "transformation_mode": inp.get("transformation_mode"),
    }
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_transformation_record",
        question="Construct a complete machine-readable coordinate-transformation record containing source CRS, target CRS, source coordinate, transformation mode, transformed coordinate, axis order, and unit.",
        answer=answer, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _transform_distance(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="distance_unit_recall",
            question=f"What standard unit symbol should be used for {inp.get('requested_unit')} in this benchmark?",
            answer=target.get("unit"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="distance_method_interpretation",
            question="Which distance model is used for the A-to-B reference distance in this task?",
            answer=target.get("method"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_geodesic_distance",
            question=inp.get("base_question") or inp.get("question") or "Compute the geodesic distance.",
            answer=target.get("value"), evaluation=_response_eval("numeric", tolerance=float(record.get("evaluation", {}).get("relative_tolerance", 0.005))), drop_choices=True)
    if level == "An":
        answer = {k: target.get(k) for k in ("value", "unit", "distance_m", "distance_km", "method")}
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_distance_representations",
            question="Compute the requested distance and report it together with its canonical metre and kilometre representations and the distance method.",
            answer=answer, evaluation=_response_eval("structured_numeric"), drop_choices=True)
    correct = float(target.get("value"))
    candidate = correct if occurrence % 2 == 0 else correct * 1.1 + (1e-6 if correct == 0 else 0)
    verdict = "yes" if occurrence % 2 == 0 else "no"
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_distance_value",
        question=f"A reviewer reports the distance as {candidate:.8g} {target.get('unit')}. Is that value correct within the benchmark tolerance? Answer yes or no.",
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_value": candidate, "candidate_unit": target.get("unit")})


def _relation_family(relation: str) -> str:
    return "topological" if relation in {"within", "contains", "touches", "intersects", "overlaps"} else "directional"


def _transform_topology(record, level, occurrence, records, index, task_dir):
    target = record["target"]
    relation = str(target.get("relation"))
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="relation_label_recognition",
            question="Name the exact topological or directional relation represented by polygon A relative to polygon B.",
            answer=relation, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        family = _relation_family(relation)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="relation_family_interpretation",
            question="Does the A-to-B relation primarily express topology or cardinal direction?",
            answer=family, evaluation=_response_eval("classification"), choices=["topological", "directional"])
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_spatial_relation",
            question=record.get("input", {}).get("base_question") or record.get("input", {}).get("question") or "Determine the relation.",
            answer=target.get("answer"), evaluation=_response_eval("exact_match"), drop_choices=True)
    inverse = INVERSE_RELATION.get(relation, relation)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_bidirectional_relation",
            question="Analyze the configuration in both directions. Return the relation of A to B and the inverse relation of B to A.",
            answer={"A_to_B": relation, "B_to_A": inverse}, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        vocabulary = list(INVERSE_RELATION)
        wrong = _wrong_choice(vocabulary, relation, occurrence)
        candidate, verdict = _verification(relation, wrong, occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_relation_claim",
            question=f'A reviewer claims that the relation of A to B is "{candidate}". Is the claim correct? Answer yes or no.',
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_relation": candidate})
    answer = {
        "nodes": [target.get("region_a_name"), target.get("region_b_name")],
        "edges": [
            {"source": "A", "target": "B", "relation": relation},
            {"source": "B", "target": "A", "relation": inverse},
        ],
    }
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_relation_graph",
        question="Create a two-node spatial relation graph for polygons A and B, including the A-to-B relation and its valid inverse B-to-A relation.",
        answer=answer, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _graph_stats(payload: dict[str, Any]) -> dict[str, Any]:
    edges = list(payload.get("edges", []))
    lengths = [float(edge.get("length", 0.0)) for edge in edges]
    return {
        "directed": bool(payload.get("directed", False)),
        "node_count": len(payload.get("nodes", [])),
        "edge_count": len(edges),
        "total_edge_length_m": round(sum(lengths), 3),
        "mean_edge_length_m": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
    }


def _transform_graph(record, level, occurrence, records, index, task_dir):
    payload = _load_graph_payload(task_dir, record)
    stats = _graph_stats(payload)
    graph_path = record["target"].get("graph")
    ref = {"reference_graph": graph_path}
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="graph_node_count_lookup",
            question="Using the provided reference graph JSON, how many nodes are listed?",
            answer=stats["node_count"], evaluation=_response_eval("numeric", tolerance=0.0), drop_choices=True, extra_input=ref)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="graph_directionality_interpretation",
            question="Using the provided reference graph JSON, is this road graph directed or undirected?",
            answer="directed" if stats["directed"] else "undirected", evaluation=_response_eval("classification"),
            choices=["directed", "undirected"], extra_input=ref)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_edge_length_aggregation",
            question="Using the provided reference graph JSON, sum all metric edge lengths and return the total in metres.",
            answer=stats["total_edge_length_m"], evaluation=_response_eval("numeric", tolerance=0.001), drop_choices=True, extra_input=ref)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_graph_summary",
            question="Analyze the provided reference graph JSON and return node count, edge count, total edge length, and mean edge length.",
            answer=stats, evaluation=_response_eval("structured_numeric"), drop_choices=True, extra_input=ref)
    if level == "E":
        candidate = {"node_count": stats["node_count"], "edge_count": stats["edge_count"]}
        verdict = "yes"
        if occurrence % 2 == 1:
            candidate["edge_count"] += 1
            verdict = "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_graph_summary",
            question=f"Evaluate this proposed graph summary against the provided reference graph JSON: {json.dumps(candidate)}. Is it fully correct? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], extra_input={**ref, "candidate_graph_summary": candidate})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_road_graph",
        question="Extract road centerlines from the satellite image and create the complete routable graph with metric edge lengths.",
        answer=graph_path, evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)


def _straight_distance(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if all(abs(float(v)) <= 180 for v in [a[0], a[1], b[0], b[1]]):
        try:
            from pyproj import Geod
            _, _, metres = Geod(ellps="WGS84").inv(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            return abs(float(metres))
        except Exception:
            pass
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _transform_route(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    route = copy.deepcopy(target.get("route_coordinates", []))
    length = float(target.get("length", 0.0))
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="route_unit_recall",
            question="In what unit is the reference shortest-route length reported?",
            answer=target.get("unit"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="optimization_objective_interpretation",
            question="What optimization objective defines the reference route in this task?",
            answer="minimum total metric edge length", evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_route_length_calculation",
            question="The reference route coordinates are provided. Apply the path-length calculation and return the route length in metres.",
            answer=length, evaluation=_response_eval("numeric", tolerance=0.005), drop_choices=True,
            extra_input={"reference_route_coordinates": route})
    if level == "An":
        direct = _straight_distance(inp.get("start", []), inp.get("end", []))
        ratio = length / direct if direct > 0 else None
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_route_detour",
            question="Analyze the reference route relative to the direct start-to-end separation. Return route length, direct distance, and route/direct detour ratio.",
            answer={"route_length_m": round(length, 3), "direct_distance_m": round(direct, 3), "detour_ratio": round(ratio, 6) if ratio is not None else None},
            evaluation=_response_eval("structured_numeric"), drop_choices=True, extra_input={"reference_route_coordinates": route})
    if level == "E":
        candidate = length if occurrence % 2 == 0 else length * 1.12 + (0.1 if length == 0 else 0)
        verdict = "yes" if occurrence % 2 == 0 else "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_route_cost",
            question=f"A reviewer reports the least-distance route length as {candidate:.3f} metres. Is this cost correct within a 0.5% tolerance? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_route_length_m": round(candidate, 3)})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_shortest_route",
        question="Create the least-distance route between the specified start and end graph coordinates. Return the ordered route coordinates.",
        answer=route, evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)


def _transform_isochrone(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="service_budget_recall",
            question="What time budget, in minutes, is specified for this service-area problem?",
            answer=inp.get("budget_minutes"), evaluation=_response_eval("numeric", tolerance=0.0), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="service_area_concept_interpretation",
            question="Should reachability in this task follow the pedestrian street network or a straight-line circular radius?",
            answer="pedestrian street network", evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_speed_time_budget",
            question="Compute the maximum network travel distance in metres from the specified walking speed and time budget.",
            answer=inp.get("network_distance_budget_m"), evaluation=_response_eval("numeric", tolerance=0.001), drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_service_area_summary",
            question="Analyze the reachable network and return the reachable-node count and service-area size in square metres.",
            answer={"reachable_node_count": target.get("reachable_node_count"), "service_area_m2": target.get("service_area_m2")},
            evaluation=_response_eval("structured_numeric"), drop_choices=True)
    if level == "E":
        correct = float(target.get("service_area_m2", 0.0))
        candidate = correct if occurrence % 2 == 0 else correct * 1.2 + (1.0 if correct == 0 else 0)
        verdict = "yes" if occurrence % 2 == 0 else "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_service_area_claim",
            question=f"A reviewer reports a service area of {candidate:.3f} m^2 for this origin and budget. Is that value correct within the benchmark tolerance? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_service_area_m2": round(candidate, 3)})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_service_area_polygon",
        question="Create the complete pedestrian service-area polygon for the specified origin, walking speed, and time budget while respecting street-network connectivity.",
        answer=target.get("isochrone_geojson"), evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)


def _transform_toponym(record, level, occurrence, records, index, task_dir):
    text = record.get("input", {}).get("text", "")
    spans = copy.deepcopy(record.get("target", {}).get("spans", []))
    if level == "R":
        span = spans[occurrence % len(spans)]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="toponym_surface_recall",
            question=f"What exact place-name text occurs at character offsets {span.get('start')}:{span.get('end')} in the provided document?",
            answer=span.get("text"), evaluation=_response_eval("exact_match"), drop_choices=True,
            extra_input={"query_offsets": [span.get("start"), span.get("end")]})
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="toponym_count_interpretation",
            question="How many annotated toponym mentions occur in this document? Count mentions, not unique names.",
            answer=len(spans), evaluation=_response_eval("numeric", tolerance=0.0), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_toponym_extraction",
            question="Identify every toponym mention in the document and return its exact character offsets and surface text.",
            answer=spans, evaluation={**copy.deepcopy(record.get("evaluation", {})), "target_field": "target.bloom_answer"}, drop_choices=True)
    if level == "An":
        counts = Counter(str(span.get("text")) for span in spans)
        answer = [{"toponym": name, "mention_count": counts[name]} for name in sorted(counts)]
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_toponym_inventory",
            question="Analyze the document's toponym annotations and return each unique toponym surface form with its number of mentions.",
            answer=answer, evaluation=_response_eval("structured_exact"), drop_choices=True)
    candidate = copy.deepcopy(spans)
    verdict = "yes"
    if occurrence % 2 == 1 and candidate:
        candidate[0]["end"] = int(candidate[0].get("end", 0)) + 1
        verdict = "no"
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_toponym_annotation",
        question="Evaluate the proposed toponym span annotation against the document. Is the complete annotation correct? Answer yes or no.",
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_spans": candidate})


def _transform_geo_entity(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    ontology = inp.get("ontology", {})
    code = target.get("feature_class")
    name = target.get("feature_class_name")
    choices = list(inp.get("choices", []))
    correct_choice = target.get("answer")
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="geonames_class_recall",
            question=f"What is the GeoNames feature-class name for class code {code}?",
            answer=name, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="entity_type_interpretation",
            question=f"Interpret the geographic entity “{inp.get('mention')}” and return its broad GeoNames feature-class name.",
            answer=name, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_geonames_ontology",
            question=f"Apply the closed-set GeoNames ontology to “{inp.get('mention')}” using the provided location context. Select exactly one feature class.",
            answer=correct_choice, evaluation=_response_eval("classification"), choices=choices)
    if level == "An":
        wrong = _wrong_choice(choices, correct_choice, occurrence)
        pair = [correct_choice, wrong]
        random.Random(SEEDS[record["leaf"]] + index).shuffle(pair)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="discriminate_entity_classes",
            question=f"Analyze the entity name and coordinate context for “{inp.get('mention')}”. Which of these two GeoNames classes is better supported?",
            answer=correct_choice, evaluation=_response_eval("classification"), choices=pair)
    wrong = _wrong_choice(choices, correct_choice, occurrence)
    candidate, verdict = _verification(correct_choice, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_entity_type_claim",
        question=f'A reviewer assigns “{inp.get("mention")}” to GeoNames class "{candidate}". Is that classification correct? Answer yes or no.',
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
        extra_input={"candidate_feature_class": candidate})


def _transform_text_relation(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    relations = copy.deepcopy(target.get("relations"))
    atoms = [value for value in _atomic_values(relations) if isinstance(value, (str, int, float, bool))]
    first_relation = atoms[0] if atoms else target.get("answer")
    original_q = inp.get("question", "")
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="spatial_relation_recognition",
            question=f"Identify one spatial relation expressed in the passage that is relevant to this question: {original_q}",
            answer=first_relation, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="spatial_description_interpretation",
            question=f"Interpret the spatial description and answer the question: {original_q}",
            answer=target.get("answer"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_spatial_relations",
            question=f"Apply the spatial relation(s) stated or implied by the passage to answer: {original_q}",
            answer=target.get("answer"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_relation_set",
            question="Analyze the passage and return the complete annotated spatial-relation set relevant to the question.",
            answer=relations, evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        wrong = _peer_value(records, index, lambda r: r.get("target", {}).get("answer"))
        candidate, verdict = _verification(target.get("answer"), wrong, occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_spatial_answer",
            question=f"A reviewer proposes the answer {json.dumps(candidate)} to the spatial question. Is it correct given the passage? Answer yes or no. Original question: {original_q}",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_answer": candidate})
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_relation_record",
        question="Create a machine-readable spatial reasoning record containing the final answer and the complete annotated relation set supported by the passage.",
        answer={"answer": target.get("answer"), "relations": relations}, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _transform_cross_compare(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    entities = list(inp.get("entities", []))
    values = target.get("values", {})
    first = entities[0] if entities else next(iter(values), None)
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="indicator_value_recall",
            question=f"What was {first}'s {inp.get('indicator_name')} value in {inp.get('year')}? Return the numeric value in {target.get('unit')}.",
            answer=values.get(first), evaluation=_response_eval("numeric", tolerance=0.005), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="comparison_interpretation",
            question=inp.get("question", "Compare the entities."),
            answer=target.get("answer"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_difference_calculation",
            question=f"Compute the absolute difference in {inp.get('indicator_name')} between {entities[0]} and {entities[1]} in {inp.get('year')}. Return the value in {target.get('unit')}.",
            answer=target.get("absolute_difference"), evaluation=_response_eval("numeric", tolerance=0.005), drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_comparison_magnitude",
            question=f"Analyze the {inp.get('year')} {inp.get('indicator_name')} comparison. Return both entity values, the larger-to-smaller ratio, and which entity is larger.",
            answer={"values": values, "larger_to_smaller_ratio": target.get("larger_to_smaller_ratio"), "larger_entity": max(values, key=values.get) if values else None},
            evaluation=_response_eval("structured_numeric"), drop_choices=True)
    if level == "E":
        peer = entities[1] if target.get("answer") == entities[0] and len(entities) > 1 else (entities[0] if entities else target.get("answer"))
        candidate, verdict = _verification(target.get("answer"), peer, occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_comparison_claim",
            question=f'A reviewer answers "{candidate}" to the comparison question: {inp.get("question")}. Is that answer correct? Answer yes or no.',
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], extra_input={"candidate_answer": candidate})
    report = {
        "indicator": target.get("indicator"), "year": target.get("year"), "unit": target.get("unit"),
        "values": values, "requested_relation": target.get("relation"), "answer": target.get("answer"),
        "absolute_difference": target.get("absolute_difference"), "larger_to_smaller_ratio": target.get("larger_to_smaller_ratio"),
    }
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_comparison_report",
        question="Create a structured comparison report containing both entity values, indicator, year, unit, requested higher/lower relation, answer, absolute difference, and larger-to-smaller ratio.",
        answer=report, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _transform_environment(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    label = target.get("answer")
    choices = list(inp.get("choices", []))
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="environmental_unit_recall",
            question=f'What measurement unit is used for the WorldClim layer "{label}" in this benchmark?',
            answer=target.get("unit"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="environmental_layer_interpretation",
            question="Interpret the unlabeled raster values and spatial pattern. Which environmental layer is shown?",
            answer=label, evaluation=_response_eval("classification"), choices=choices)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_layer_identification",
            question="Apply the WorldClim layer ontology to this raster patch and select the correct environmental variable.",
            answer=label, evaluation=_response_eval("classification"), choices=choices)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_environmental_layer_profile",
            question="Analyze the raster patch and return the identified environmental layer, its unit, and the provided patch summary-statistic profile.",
            answer={"layer": label, "unit": target.get("unit"), "summary_statistics": target.get("summary_statistics")},
            evaluation=_response_eval("structured_numeric"), drop_choices=True)
    wrong = _wrong_choice(choices, label, occurrence)
    candidate, verdict = _verification(label, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_layer_claim",
        question=f'A reviewer claims this raster is "{candidate}". Is that layer identification correct? Answer yes or no.',
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], extra_input={"candidate_layer": candidate})


def _same_year_records(records: list[dict[str, Any]], year: Any) -> list[dict[str, Any]]:
    return [r for r in records if r.get("target", {}).get("year") == year]


def _transform_population(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    value = float(target.get("value"))
    country, year = inp.get("country"), inp.get("year")
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="population_density_recall",
            question=f"What was the population density of {country} in {year}, in people per square kilometre of land area?",
            answer=value, evaluation=_response_eval("numeric", tolerance=0.005), drop_choices=True)
    if level == "U":
        threshold = 100.0
        relation = "above" if value > threshold else ("below" if value < threshold else "equal to")
        return _apply_variant(record, level=level, occurrence=occurrence, variant="density_magnitude_interpretation",
            question=f"Was {country}'s population density in {year} above, below, or equal to {threshold:g} people per km^2?",
            answer=relation, evaluation=_response_eval("classification"), choices=["above", "below", "equal to"])
    if level == "Ap":
        area = 1000.0
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_density_to_area",
            question=f"Using {country}'s {year} population density, what population would correspond to a hypothetical land area of {area:.0f} km^2 at the same density?",
            answer=round(value * area, 3), evaluation=_response_eval("numeric", tolerance=0.005), drop_choices=True,
            extra_input={"hypothetical_area_km2": area})
    same_year = _same_year_records(records, year)
    if level == "An":
        peer = next((r for r in same_year if r.get("id") != record.get("id") and r.get("input", {}).get("country") != country), records[(index + 1) % len(records)])
        peer_country = peer.get("input", {}).get("country")
        peer_value = float(peer.get("target", {}).get("value"))
        higher = country if value > peer_value else peer_country
        answer = {"higher_density_country": higher, "absolute_difference": round(abs(value - peer_value), 6)}
        out = _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_density_pair",
            question=f"Compare population density in {year} for {country} and {peer_country}. Which is higher, and what is the absolute difference in people per km^2?",
            answer=answer, evaluation=_response_eval("structured_numeric"), drop_choices=True,
            extra_input={"comparison_countries": [country, peer_country]}, source_record_ids=[str(record.get("id")), str(peer.get("id"))])
        out["group_id"] = f"{record.get('group_id')}|{peer.get('group_id')}"
        return out
    if level == "E":
        candidate = value if occurrence % 2 == 0 else value * 1.15 + (0.1 if value == 0 else 0)
        verdict = "yes" if occurrence % 2 == 0 else "no"
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_density_value",
            question=f"A reviewer reports {country}'s population density in {year} as {candidate:.6f} people per km^2. Is that value correct within the benchmark tolerance? Answer yes or no.",
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"],
            extra_input={"candidate_density": round(candidate, 6)})
    pool = [r for r in same_year if r.get("input", {}).get("country")]
    if len(pool) < 3:
        pool = records
    chosen: list[dict[str, Any]] = []
    for offset in range(len(pool)):
        candidate = pool[(occurrence * 3 + offset) % len(pool)]
        if candidate.get("input", {}).get("country") not in {r.get("input", {}).get("country") for r in chosen}:
            chosen.append(candidate)
        if len(chosen) == 3:
            break
    ranking = sorted(chosen, key=lambda r: float(r.get("target", {}).get("value")), reverse=True)
    names = [r.get("input", {}).get("country") for r in chosen]
    answer = [r.get("input", {}).get("country") for r in ranking]
    source_ids = [str(r.get("id")) for r in chosen]
    out = _apply_variant(record, level=level, occurrence=occurrence, variant="create_density_ranking",
        question=f"Construct a descending population-density ranking for these countries in {chosen[0].get('target', {}).get('year')}: {', '.join(names)}.",
        answer=answer, evaluation=_response_eval("ranking_exact"), drop_choices=True,
        extra_input={"ranking_countries": names, "year": chosen[0].get("target", {}).get("year")}, source_record_ids=source_ids)
    out["group_id"] = "|".join(str(r.get("group_id")) for r in chosen)
    return out


def _transform_geology(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="geologic_unit_recall",
            question=f"What mapped geologic unit is reported at the coordinate {inp.get('coordinate')}?",
            answer=target.get("unit_name"), evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="lithology_interpretation",
            question=f"What lithology or geologic material description is associated with the mapped unit at {inp.get('coordinate')}?",
            answer=target.get("lithology") or "not specified", evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_geologic_lookup",
            question=inp.get("question", "Identify and interpret the mapped geologic unit."),
            answer={"unit_name": target.get("unit_name"), "lithology": target.get("lithology") or "not specified", "age": target.get("age") or "not specified"},
            evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "An":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_geologic_attributes",
            question="Analyze the mapped unit and organize its unit name, lithology, and age into separate semantic fields.",
            answer={"unit_name": target.get("unit_name"), "lithology": target.get("lithology") or "not specified", "age": target.get("age") or "not specified"},
            evaluation=_response_eval("structured_exact"), drop_choices=True)
    if level == "E":
        peer_unit = _peer_value(records, index, lambda r: r.get("target", {}).get("unit_name"))
        candidate, verdict = _verification(target.get("unit_name"), peer_unit, occurrence)
        return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_geologic_unit_claim",
            question=f'A reviewer identifies the mapped unit at this coordinate as "{candidate}". Is that unit identification correct? Answer yes or no.',
            answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], extra_input={"candidate_unit_name": candidate})
    profile = {"unit_name": target.get("unit_name"), "lithology": target.get("lithology") or "not specified", "age": target.get("age") or "not specified"}
    return _apply_variant(record, level=level, occurrence=occurrence, variant="create_geologic_profile",
        question="Create a concise machine-readable geologic profile for this coordinate containing unit name, lithology/material description, and age information.",
        answer=profile, evaluation=_response_eval("structured_generation"), drop_choices=True)


def _transform_geo_fact(record, level, occurrence, records, index, task_dir):
    inp, target = record["input"], record["target"]
    question = inp.get("question", "")
    answer = target.get("answer")
    query = target.get("reference_query")
    if level == "R":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="geographic_fact_recall",
            question=question, answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "U":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="geographic_question_interpretation",
            question=f"Interpret this geographic question and return a concise normalized answer: {question}",
            answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True)
    if level == "Ap":
        return _apply_variant(record, level=level, occurrence=occurrence, variant="apply_reference_query",
            question=f"Use the provided reference GeoSPARQL/SPARQL query as the formal operation for this question and return the geographic answer: {question}",
            answer=answer, evaluation=_response_eval("exact_match"), drop_choices=True,
            extra_input={"reference_query": query})
    if level == "An":
        if query is not None and str(query).strip():
            analysis_answer = query
            analysis_question = f"Analyze the geographic question and return the reference GeoSPARQL/SPARQL query that operationalizes it: {question}"
            analysis_eval = _response_eval("exact_match")
        else:
            analysis_answer = {"answer": answer, "reference_query_available": False}
            analysis_question = f"Analyze the geographic question and return the answer together with whether a reference GeoSPARQL/SPARQL query is available in this source record: {question}"
            analysis_eval = _response_eval("structured_exact")
        return _apply_variant(record, level=level, occurrence=occurrence, variant="analyze_question_to_query",
            question=analysis_question, answer=analysis_answer, evaluation=analysis_eval, drop_choices=True)
    wrong = _peer_value(records, index, lambda r: r.get("target", {}).get("answer"))
    candidate, verdict = _verification(answer, wrong, occurrence)
    return _apply_variant(record, level=level, occurrence=occurrence, variant="evaluate_geographic_answer",
        question=f"A reviewer proposes the answer {json.dumps(candidate)} to this geographic question: {question}. Is the proposed answer correct? Answer yes or no.",
        answer=verdict, evaluation=_response_eval("binary_classification"), choices=["yes", "no"], extra_input={"candidate_answer": candidate})


TRANSFORMERS: dict[str, Callable[..., dict[str, Any]]] = {
    "cartographic_symbol_recognition": _transform_maki,
    "map_text_detection_recognition_grouping": _transform_maptext,
    "map_label_feature_anchoring": _transform_label_anchor,
    "dense_land_cover_labeling": _transform_landcover,
    "remote_sensing_scene_classification": _transform_eurosat,
    "object_presence_counting": _transform_rsvqa,
    "change_localization": _transform_change,
    "temporal_scene_matching": _transform_temporal,
    "visual_geolocation": _transform_geolocation,
    "coordinate_transformation": _transform_coordinate,
    "metric_distance_computation": _transform_distance,
    "topological_directional_reasoning": _transform_topology,
    "spatial_graph_construction": _transform_graph,
    "shortest_path_optimization": _transform_route,
    "isochrone_service_area": _transform_isochrone,
    "toponym_recognition": _transform_toponym,
    "geo_entity_typing": _transform_geo_entity,
    "textual_spatial_relation_extraction": _transform_text_relation,
    "cross_entity_comparison": _transform_cross_compare,
    "environmental_layer_identification": _transform_environment,
    "population_density_estimation": _transform_population,
    "geologic_geomorphic_interpretation": _transform_geology,
    "geographic_fact_reasoning": _transform_geo_fact,
}


def _prepare_task(task_dir: Path, force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    leaf = task_dir.name
    if leaf not in BLOOM_LEVELS:
        raise ValueError(f"Unsupported GeoMapBench leaf: {leaf}")
    data_path = task_dir / "data.jsonl"
    manifest_path = task_dir / "manifest.json"
    records = read_jsonl(data_path)
    if len(records) != N_EXAMPLES:
        raise ValueError(f"{leaf}: expected {N_EXAMPLES} base records, found {len(records)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bloom_revision") == BLOOM_REVISION and not force:
        return records, manifest

    assignments = _assigned_levels(leaf, len(records))
    occurrence_by_level: dict[str, int] = defaultdict(int)
    transformed: list[dict[str, Any]] = []
    transformer = TRANSFORMERS[leaf]
    for index, (record, level) in enumerate(zip(records, assignments)):
        occurrence = occurrence_by_level[level]
        occurrence_by_level[level] += 1
        transformed.append(transformer(record, level, occurrence, records, index, task_dir))

    distribution = dict(Counter(r["bloom"]["level"] for r in transformed))
    expected = bloom_distribution(BLOOM_LEVELS[leaf], N_EXAMPLES)
    if distribution != expected:
        raise ValueError(f"{leaf}: Bloom distribution mismatch: observed={distribution}, expected={expected}")
    for record in transformed:
        if "bloom_answer" not in record.get("target", {}):
            raise ValueError(f"{leaf}:{record.get('id')}: missing target.bloom_answer")
        if record.get("bloom", {}).get("level") not in BLOOM_LEVELS[leaf]:
            raise ValueError(f"{leaf}:{record.get('id')}: invalid Bloom level")

    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(
        {
            "count": len(transformed),
            "created_at": utc_now(),
            "base_data_revision": manifest.get("data_revision"),
            "base_sha256_before_bloom": manifest.get("sha256"),
            "bloom_revision": BLOOM_REVISION,
            "bloom_balanced": True,
            "bloom_levels": list(BLOOM_LEVELS[leaf]),
            "bloom_level_names": {level: BLOOM_LEVEL_NAMES[level] for level in BLOOM_LEVELS[leaf]},
            "bloom_distribution": expected,
            "bloom_primary_target": "target.bloom_answer",
            "bloom_conversion": "metadata/prompt/target overlay only; original assets and original target fields retained",
        }
    )
    return transformed, new_manifest


def bloomify_root(root: Path, *, backup: bool = True, force: bool = False, require_all: bool = True) -> dict[str, Any]:
    root = Path(root)
    found = {p.name: p for p in root.iterdir() if p.is_dir() and (p / "data.jsonl").is_file()} if root.exists() else {}
    if require_all:
        missing = sorted(set(BLOOM_LEVELS) - set(found))
        if missing:
            raise FileNotFoundError("Missing GeoMapBench leaves: " + ", ".join(missing))

    prepared: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for leaf in sorted(set(found) & set(BLOOM_LEVELS)):
        prepared[leaf] = _prepare_task(found[leaf], force=force)

    # Only write after every leaf has successfully transformed in memory.
    for leaf, (records, manifest) in prepared.items():
        task_dir = found[leaf]
        current_manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        if current_manifest.get("bloom_revision") == BLOOM_REVISION and not force:
            continue
        backup_dir = task_dir / ".pre_bloom"
        if backup:
            backup_dir.mkdir(parents=True, exist_ok=True)
            data_backup = backup_dir / "data.jsonl"
            manifest_backup = backup_dir / "manifest.json"
            if not data_backup.exists():
                shutil.copy2(task_dir / "data.jsonl", data_backup)
            if not manifest_backup.exists():
                shutil.copy2(task_dir / "manifest.json", manifest_backup)

        temp_data = task_dir / ".data.bloom.tmp.jsonl"
        write_jsonl(temp_data, records)
        manifest["sha256"] = sha256_file(temp_data)
        temp_manifest = task_dir / ".manifest.bloom.tmp.json"
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_data.replace(task_dir / "data.jsonl")
        temp_manifest.replace(task_dir / "manifest.json")

    report = bloom_audit_root(root, require_all=require_all)
    (root / "bloom_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def restore_bloom_root(root: Path, *, require_all: bool = True) -> dict[str, Any]:
    root = Path(root)
    found = {p.name: p for p in root.iterdir() if p.is_dir() and (p / "data.jsonl").is_file()} if root.exists() else {}
    if require_all:
        missing = sorted(set(BLOOM_LEVELS) - set(found))
        if missing:
            raise FileNotFoundError("Missing GeoMapBench leaves: " + ", ".join(missing))
    restored = []
    for leaf in sorted(set(found) & set(BLOOM_LEVELS)):
        task_dir = found[leaf]
        backup_dir = task_dir / ".pre_bloom"
        if not (backup_dir / "data.jsonl").exists() or not (backup_dir / "manifest.json").exists():
            raise FileNotFoundError(f"{leaf}: missing .pre_bloom backup")
        shutil.copy2(backup_dir / "data.jsonl", task_dir / "data.jsonl")
        shutil.copy2(backup_dir / "manifest.json", task_dir / "manifest.json")
        restored.append(leaf)
    audit_path = root / "bloom_audit.json"
    if audit_path.exists():
        audit_path.unlink()
    return {"restored": restored, "count": len(restored)}


def bloom_audit_root(root: Path, *, require_all: bool = True) -> dict[str, Any]:
    root = Path(root)
    found = {p.name: p for p in root.iterdir() if p.is_dir() and (p / "data.jsonl").is_file()} if root.exists() else {}
    errors: list[str] = []
    if require_all:
        missing = sorted(set(BLOOM_LEVELS) - set(found))
        if missing:
            errors.append("Missing leaves: " + ", ".join(missing))
    leaves: dict[str, Any] = {}
    for leaf in sorted(set(found) & set(BLOOM_LEVELS)):
        task_dir = found[leaf]
        records = read_jsonl(task_dir / "data.jsonl")
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        counts = Counter(r.get("bloom", {}).get("level") for r in records)
        expected = bloom_distribution(BLOOM_LEVELS[leaf], len(records))
        leaf_errors = []
        if manifest.get("bloom_revision") != BLOOM_REVISION:
            leaf_errors.append(f"manifest bloom_revision={manifest.get('bloom_revision')!r}")
        if dict(counts) != expected:
            leaf_errors.append(f"distribution={dict(counts)} expected={expected}")
        for i, record in enumerate(records, 1):
            level = record.get("bloom", {}).get("level")
            if level not in BLOOM_LEVELS[leaf]:
                leaf_errors.append(f"record {i}: invalid level {level!r}")
                break
            if "bloom_answer" not in record.get("target", {}):
                leaf_errors.append(f"record {i}: missing target.bloom_answer")
                break
            if record.get("evaluation", {}).get("target_field") != "target.bloom_answer":
                leaf_errors.append(f"record {i}: evaluation does not target target.bloom_answer")
                break
        leaves[leaf] = {
            "supported_levels": list(BLOOM_LEVELS[leaf]),
            "distribution": dict(counts),
            "expected_distribution": expected,
            "errors": leaf_errors,
        }
        errors.extend(f"{leaf}: {message}" for message in leaf_errors)
    return {
        "bloom_revision": BLOOM_REVISION,
        "root": str(root),
        "leaf_count": len(leaves),
        "record_count": sum(sum(item["distribution"].values()) for item in leaves.values()),
        "valid": not errors,
        "errors": errors,
        "leaves": leaves,
    }
