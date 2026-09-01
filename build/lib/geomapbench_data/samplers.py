from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import (
    N_EXAMPLES,
    SEEDS,
    balanced_sample,
    base_record,
    copy_asset,
    finalize_task,
    iter_dicts,
    load_json,
    stable_sample,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def sample_maki(source: Path, output: Path) -> Path:
    leaf = "cartographic_symbol_recognition"
    icons = sorted(source.rglob("*.svg"), key=lambda p: p.as_posix())
    selected = stable_sample(icons, N_EXAMPLES, SEEDS[leaf], key=lambda p: p.as_posix())
    all_labels = sorted({p.stem.replace("-", " ") for p in icons})
    rng = random.Random(SEEDS[leaf])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, icon in enumerate(selected):
        answer = icon.stem.replace("-", " ")
        distractors = rng.sample([x for x in all_labels if x != answer], 3)
        choices = distractors + [answer]
        rng.shuffle(choices)
        record = base_record(
            leaf,
            i,
            "Maki map icons",
            "https://github.com/mapbox/maki",
            "CC0-1.0",
            icon.stem,
        )
        record.update(
            {
                "input": {
                    "images": [copy_asset(icon, task_dir / "assets", f"{i:03d}.svg")],
                    "question": "What point-of-interest category does this cartographic symbol represent?",
                    "choices": choices,
                },
                "target": {"answer": answer, "choice_index": choices.index(answer)},
                "evaluation": {"type": "exact_match", "normalize": "lowercase_whitespace"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _find_by_basename(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in _image_files(root):
        out.setdefault(path.name, path)
        out.setdefault(path.stem, path)
    return out


def _normalize_vertices(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for point in value:
        if isinstance(point, dict) and "x" in point and "y" in point:
            points.append([float(point["x"]), float(point["y"])])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append([float(point[0]), float(point[1])])
    return points if len(points) >= 3 else []


def _annotation_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _valid_maptext_word(word: Any) -> dict[str, Any] | None:
    if not isinstance(word, dict):
        return None
    text = str(word.get("text", "")).strip()
    vertices = _normalize_vertices(
        word.get("vertices") or word.get("polygon") or word.get("points")
    )
    if (
        not text
        or not vertices
        or _annotation_flag(word.get("illegible", False))
        or _annotation_flag(word.get("truncated", False))
    ):
        return None
    return {"text": text, "vertices": vertices}


def _sequence_from_words(words: list[Any], source_group_index: Any) -> dict[str, Any] | None:
    valid_words = [parsed for word in words if (parsed := _valid_maptext_word(word))]
    if not valid_words:
        return None
    xs = [point[0] for word in valid_words for point in word["vertices"]]
    ys = [point[1] for word in valid_words for point in word["vertices"]]
    return {
        "source_group_index": str(source_group_index),
        "text": " ".join(word["text"] for word in valid_words),
        "words": valid_words,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
    }


def _maptext_sequences(image_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse MapText sequence annotations without splitting multiword labels.

    The official release and related ICDAR exports occur in several equivalent
    layouts: a sequence can be a list of word dictionaries, a dictionary with a
    ``words`` list, or an image/sequence record whose entire ``groups`` list is
    the ordered word sequence. Explicit sequence/group IDs are also supported.
    """
    raw_groups = image_record.get("groups") or image_record.get("annotations") or []
    if not isinstance(raw_groups, list) or not raw_groups:
        return []

    # Flat word dictionaries may carry explicit sequence identifiers.
    if all(isinstance(item, dict) and not isinstance(item.get("words"), list) for item in raw_groups):
        id_keys = ("sequence_id", "group_id", "line_id", "sequence", "group")
        explicit_ids = []
        for item in raw_groups:
            explicit_ids.append(next((item.get(key) for key in id_keys if item.get(key) is not None), None))
        if any(value is not None and not isinstance(value, (dict, list)) for value in explicit_ids):
            ordered: dict[str, list[Any]] = {}
            for position, (item, identifier) in enumerate(zip(raw_groups, explicit_ids)):
                key = str(identifier if identifier is not None else f"ungrouped-{position}")
                ordered.setdefault(key, []).append(item)
            return [
                sequence
                for key, words in ordered.items()
                if (sequence := _sequence_from_words(words, key)) is not None
            ]

        # In the official flat layout, all dictionaries in one record are the
        # ordered words of one sequence (single-word records are valid too).
        sequence = _sequence_from_words(raw_groups, image_record.get("sequence_id", 0))
        return [sequence] if sequence else []

    sequences: list[dict[str, Any]] = []
    for group_index, group in enumerate(raw_groups):
        words: list[Any]
        if isinstance(group, list):
            words = group
        elif isinstance(group, dict) and isinstance(group.get("words"), list):
            words = group["words"]
        elif isinstance(group, dict) and isinstance(group.get("tokens"), list):
            words = group["tokens"]
        elif isinstance(group, dict):
            words = [group]
        else:
            continue
        sequence = _sequence_from_words(words, group_index)
        if sequence:
            sequences.append(sequence)
    return sequences


def _maptext_candidates(data: Any, image_lookup: dict[str, Path]) -> list[dict[str, Any]]:
    image_records = data if isinstance(data, list) else data.get("images", data.get("data", []))
    if not isinstance(image_records, list):
        raise ValueError("MapText JSON must contain a list of image or sequence records")

    # Aggregate every sequence belonging to the same source image. This handles
    # releases with one record per image as well as one record per sequence.
    by_image: dict[str, dict[str, Any]] = {}
    for image_index, image_record in enumerate(image_records):
        if not isinstance(image_record, dict):
            continue
        image_name = str(
            image_record.get("image")
            or image_record.get("file_name")
            or image_record.get("filename")
            or ""
        )
        image_path = image_lookup.get(image_name) or image_lookup.get(Path(image_name).stem)
        if image_path is None:
            continue
        bucket = by_image.setdefault(
            image_name or str(image_index),
            {"image": image_path, "image_name": image_name, "sequences": []},
        )
        for sequence in _maptext_sequences(image_record):
            sequence = dict(sequence)
            sequence["source_group_index"] = (
                f"{image_index}:{sequence['source_group_index']}"
            )
            bucket["sequences"].append(sequence)

    candidates: list[dict[str, Any]] = []
    for image_group, bucket in sorted(by_image.items()):
        sequences = bucket["sequences"]
        if len(sequences) < 2:
            continue
        for anchor_index, sequence in enumerate(sequences):
            candidates.append(
                {
                    "key": f"{image_group}:{sequence['source_group_index']}:{sequence['text']}",
                    "image": bucket["image"],
                    "image_name": bucket["image_name"],
                    "groups": sequences,
                    "anchor_index": anchor_index,
                    "image_group": image_group,
                }
            )
    return candidates

def _crop_maptext_example(item: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    from PIL import Image

    with Image.open(item["image"]) as source:
        image = source.convert("RGB")
        width, height = image.size
        anchor = item["groups"][item["anchor_index"]]
        x0, y0, x1, y1 = anchor["bbox"]
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        side = int(max(512, min(1400, max(x1 - x0, y1 - y0) * 6 + 220)))

        def crop_box(current_side: int) -> tuple[int, int, int, int]:
            left = max(0, min(width - current_side, int(round(center_x - current_side / 2))))
            top = max(0, min(height - current_side, int(round(center_y - current_side / 2))))
            right = min(width, left + current_side)
            bottom = min(height, top + current_side)
            return left, top, right, bottom

        box = crop_box(min(side, max(width, height)))
        included: list[dict[str, Any]] = []
        for _ in range(3):
            left, top, right, bottom = box
            included = [
                group
                for group in item["groups"]
                if group["bbox"][0] >= left
                and group["bbox"][1] >= top
                and group["bbox"][2] <= right
                and group["bbox"][3] <= bottom
            ]
            if len(included) >= 2 or (right - left == width and bottom - top == height):
                break
            side = min(max(width, height), int(side * 1.6))
            box = crop_box(side)
        if len(included) < 2:
            box = (0, 0, width, height)
            included = item["groups"]

        left, top, right, bottom = box
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.crop(box).save(destination, format="PNG", optimize=True)

    adjusted_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(included):
        adjusted_words = []
        for word in group["words"]:
            adjusted_words.append(
                {
                    "text": word["text"],
                    "vertices": [[round(x - left, 3), round(y - top, 3)] for x, y in word["vertices"]],
                }
            )
        adjusted_groups.append(
            {
                "group_id": f"g{group_index}",
                "text": group["text"],
                "words": adjusted_words,
            }
        )
    return adjusted_groups


def sample_maptext(source: Path, output: Path) -> Path:
    leaf = "map_text_detection_recognition_grouping"
    json_candidates = list(source.rglob("maptext_format.json")) or list(source.rglob("*.json"))
    if not json_candidates:
        raise FileNotFoundError("No MapText JSON annotation file found")
    image_lookup = _find_by_basename(source)
    candidates: list[dict[str, Any]] = []
    for annotation_path in sorted(json_candidates):
        try:
            candidates.extend(_maptext_candidates(load_json(annotation_path), image_lookup))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: x["key"])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        crop_path = task_dir / "assets" / f"{i:03d}_map_crop.png"
        groups = _crop_maptext_example(item, crop_path)
        flattened_words = [
            {**word, "group_id": group["group_id"]}
            for group in groups
            for word in group["words"]
        ]
        record = base_record(
            leaf,
            i,
            "Paris and Jerusalem Maps Text Dataset",
            "https://doi.org/10.5281/zenodo.14982663",
            "CC-BY-4.0",
            f"{item['image_group']}:{item['anchor_index']}",
        )
        record.update(
            {
                "input": {
                    "images": [crop_path.relative_to(task_dir).as_posix()],
                    "question": (
                        "Detect every visible map-text word in this crop, transcribe it, and assign a group ID. "
                        "Words that form the same multiword geographic label must share one group ID."
                    ),
                    "task_definition": {
                        "detection": "return one polygon for each word",
                        "recognition": "transcribe the word inside each polygon",
                        "grouping": "link words belonging to the same complete map label",
                    },
                },
                "target": {"groups": groups, "words": flattened_words, "group_count": len(groups)},
                "evaluation": {
                    "type": "map_text_spotting_and_grouping",
                    "metrics": ["word_polygon_iou", "normalized_edit_distance", "grouping_f1"],
                },
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records, {"minimum_groups_per_example": 2})


OPENEARTHMAP_CLASSES = [
    {"id": 0, "name": "bareland", "rgb": [128, 0, 0], "hex": "#800000"},
    {"id": 1, "name": "rangeland", "rgb": [0, 255, 36], "hex": "#00FF24"},
    {"id": 2, "name": "developed space", "rgb": [148, 148, 148], "hex": "#949494"},
    {"id": 3, "name": "road", "rgb": [255, 255, 255], "hex": "#FFFFFF"},
    {"id": 4, "name": "tree", "rgb": [34, 97, 38], "hex": "#226126"},
    {"id": 5, "name": "water", "rgb": [0, 69, 255], "hex": "#0045FF"},
    {"id": 6, "name": "agriculture land", "rgb": [75, 181, 73], "hex": "#4BB549"},
    {"id": 7, "name": "building", "rgb": [222, 31, 7], "hex": "#DE1F07"},
]
OPENEARTHMAP_IGNORE_INDEX = 255


def _paired_image_labels(source: Path) -> list[tuple[Path, Path, str]]:
    images: dict[tuple[str, str], Path] = {}
    labels: dict[tuple[str, str], Path] = {}
    for path in _image_files(source):
        parts = [p.lower() for p in path.parts]
        rel = path.relative_to(source)
        context_parts = [
            p
            for p in rel.parts[:-1]
            if p.lower() not in {"images", "image", "labels", "label", "masks", "mask"}
        ]
        key = ("/".join(context_parts), path.stem)
        is_label = any(token in {"labels", "label", "masks", "mask", "annotations"} for token in parts)
        if is_label:
            labels[key] = path
        else:
            images[key] = path
    pairs: list[tuple[Path, Path, str]] = []
    for key, image in images.items():
        label = labels.get(key)
        if label:
            pairs.append((image, label, f"{key[0]}/{key[1]}"))
    return pairs


def _copy_rgb_png(source: Path, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG", optimize=True)


def _normalize_openearthmap_mask(image_path: Path, label_path: Path, destination: Path) -> dict[int, int]:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        image_size = image.size
    with Image.open(label_path) as label:
        if label.size != image_size:
            raise ValueError(f"Image/mask size mismatch: {image_path} {image_size} vs {label_path} {label.size}")
        native = np.asarray(label.convert("RGB")) if label.mode == "P" else np.asarray(label)
        class_ids = None
        if native.ndim == 2:
            unique = set(int(value) for value in np.unique(native))
            if unique.issubset(set(range(8)) | {OPENEARTHMAP_IGNORE_INDEX}):
                class_ids = native.astype("uint8")
            elif unique.issubset(set(range(1, 9)) | {0, OPENEARTHMAP_IGNORE_INDEX}):
                class_ids = np.full(native.shape, OPENEARTHMAP_IGNORE_INDEX, dtype="uint8")
                for source_id in range(1, 9):
                    class_ids[native == source_id] = source_id - 1
        if class_ids is None:
            rgb = np.asarray(label.convert("RGB"), dtype="int32")
            class_ids = np.full(rgb.shape[:2], OPENEARTHMAP_IGNORE_INDEX, dtype="uint8")
            palette = np.asarray([item["rgb"] for item in OPENEARTHMAP_CLASSES], dtype="int32")
            distances = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)
            nearest = distances.argmin(axis=2)
            minimum = distances.min(axis=2)
            exact_or_near = minimum <= 9
            class_ids[exact_or_near] = nearest[exact_or_near].astype("uint8")
            black = np.all(rgb == 0, axis=2)
            class_ids[black] = OPENEARTHMAP_IGNORE_INDEX
            unexplained = (~exact_or_near) & (~black)
            if float(unexplained.mean()) > 0.001:
                colors = np.unique(rgb[unexplained].reshape(-1, 3), axis=0)[:10]
                raise ValueError(f"Unexpected OpenEarthMap label colors in {label_path}: {colors.tolist()}")

    valid = set(int(value) for value in np.unique(class_ids))
    if not valid.issubset(set(range(8)) | {OPENEARTHMAP_IGNORE_INDEX}):
        raise ValueError(f"Invalid normalized class IDs: {sorted(valid)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_ids, mode="L").save(destination, format="PNG", optimize=True)
    values, counts = np.unique(class_ids, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def sample_openearthmap(source: Path, output: Path) -> Path:
    leaf = "dense_land_cover_labeling"
    pairs = _paired_image_labels(source)
    by_region: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for pair in pairs:
        region = pair[2].split("/")[0] or "unknown_region"
        by_region[region].append(pair)
    selected = balanced_sample(by_region, N_EXAMPLES, SEEDS[leaf], key=lambda x: x[2])
    task_dir = output / leaf
    ontology = {str(item["id"]): {"name": item["name"], "rgb": item["rgb"], "hex": item["hex"]} for item in OPENEARTHMAP_CLASSES}
    legend = "; ".join(f"{item['id']}={item['name']}" for item in OPENEARTHMAP_CLASSES)
    records: list[dict[str, Any]] = []
    for i, (_, (image, label, group_id)) in enumerate(selected):
        image_destination = task_dir / "assets" / f"{i:03d}_image.png"
        mask_destination = task_dir / "assets" / f"{i:03d}_mask.png"
        _copy_rgb_png(image, image_destination)
        pixel_counts = _normalize_openearthmap_mask(image, label, mask_destination)
        record = base_record(
            leaf,
            i,
            "OpenEarthMap",
            "https://doi.org/10.5281/zenodo.7223446",
            "Source-dependent; label license follows the source image license",
            group_id.split("/")[0],
        )
        record.update(
            {
                "input": {
                    "images": [image_destination.relative_to(task_dir).as_posix()],
                    "question": (
                        "Produce a single-channel per-pixel semantic land-cover mask using this exact class-index "
                        f"ontology: {legend}. Use 255 only for ignored or unlabeled pixels."
                    ),
                    "classes": [item["name"] for item in OPENEARTHMAP_CLASSES],
                    "class_ontology": ontology,
                    "mask_encoding": "8-bit class-index PNG",
                },
                "target": {
                    "mask": mask_destination.relative_to(task_dir).as_posix(),
                    "classes": [item["name"] for item in OPENEARTHMAP_CLASSES],
                    "class_ontology": ontology,
                    "ignore_index": OPENEARTHMAP_IGNORE_INDEX,
                    "pixel_counts": {str(key): value for key, value in pixel_counts.items()},
                },
                "evaluation": {
                    "type": "semantic_segmentation",
                    "metrics": ["mean_iou", "macro_f1"],
                    "ignore_index": OPENEARTHMAP_IGNORE_INDEX,
                },
            }
        )
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {"mask_encoding": "uint8_class_index", "ignore_index": OPENEARTHMAP_IGNORE_INDEX, "ontology": ontology},
    )


def sample_eurosat(source: Path, output: Path) -> Path:
    leaf = "remote_sensing_scene_classification"
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in _image_files(source):
        class_name = path.parent.name
        if class_name.lower() not in {"images", "rgb", "eurosat_rgb"}:
            groups[class_name].append(path)
    selected = balanced_sample(groups, N_EXAMPLES, SEEDS[leaf], key=lambda p: p.as_posix())
    task_dir = output / leaf
    classes = sorted(groups)
    records: list[dict[str, Any]] = []
    for i, (answer, image) in enumerate(selected):
        record = base_record(
            leaf,
            i,
            "EuroSAT RGB",
            "https://doi.org/10.5281/zenodo.7711810",
            "MIT (plus Copernicus Sentinel data terms)",
            image.stem,
        )
        record.update(
            {
                "input": {
                    "images": [copy_asset(image, task_dir / "assets", f"{i:03d}{image.suffix.lower()}")],
                    "question": "Which land-use/land-cover scene class best describes this satellite patch?",
                    "choices": classes,
                },
                "target": {"answer": answer, "choice_index": classes.index(answer)},
                "evaluation": {"type": "classification", "metric": "accuracy"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _top_level_list(obj: Any, likely_key: str) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get(likely_key), list):
            return obj[likely_key]
        for value in obj.values():
            if isinstance(value, list):
                return value
        return list(obj.values())
    return []


def sample_rsvqa(source: Path, output: Path) -> Path:
    leaf = "object_presence_counting"
    qpath = source / "all_questions.json"
    apath = source / "all_answers.json"
    if not qpath.exists() or not apath.exists():
        raise FileNotFoundError("Expected all_questions.json and all_answers.json from RSVQA-LR")
    questions = _top_level_list(load_json(qpath), "questions")
    answers = _top_level_list(load_json(apath), "answers")
    answer_by_qid: dict[str, Any] = {}
    answer_by_id: dict[str, Any] = {}
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        value = answer.get("answer", answer.get("multiple_choice_answer", answer.get("label")))
        if value is None:
            continue
        if "question_id" in answer:
            answer_by_qid[str(answer["question_id"])] = value
        if "answer_id" in answer or "id" in answer:
            answer_by_id[str(answer.get("answer_id", answer.get("id")))] = value
    image_lookup = _find_by_basename(source)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        if not isinstance(q, dict):
            continue
        text = q.get("question") or q.get("text")
        qid = str(q.get("question_id", q.get("id", "")))
        image_id = q.get("img_id", q.get("image_id", q.get("image")))
        answer = q.get("answer") or answer_by_qid.get(qid)
        if answer is None and q.get("answer_id") is not None:
            answer = answer_by_id.get(str(q["answer_id"]))
        if not text or image_id is None or answer is None:
            continue
        image = image_lookup.get(str(image_id)) or image_lookup.get(Path(str(image_id)).stem)
        if image is None:
            for suffix in (".tif", ".png", ".jpg"):
                image = image_lookup.get(f"{image_id}{suffix}")
                if image:
                    break
        if image is None:
            continue
        qtype = str(q.get("type", q.get("question_type", "other")))
        if not any(token in qtype.lower() or token in str(text).lower() for token in ("count", "presence", "exist", "how many", "is there", "are there")):
            continue
        groups[qtype].append(
            {"key": qid or f"{image_id}:{text}", "question": text, "answer": answer, "image": image, "image_id": image_id}
        )
    selected = balanced_sample(groups, N_EXAMPLES, SEEDS[leaf], key=lambda x: x["key"])
    task_dir = output / leaf
    copied: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    for i, (qtype, item) in enumerate(selected):
        if item["image"] not in copied:
            copied[item["image"]] = copy_asset(
                item["image"], task_dir / "assets", f"image_{len(copied):03d}{item['image'].suffix.lower()}"
            )
        record = base_record(
            leaf,
            i,
            "RSVQA-LR",
            "https://doi.org/10.5281/zenodo.6344334",
            "CC-BY-4.0 (annotations; retain OSM attribution)",
            item["image_id"],
        )
        record.update(
            {
                "input": {"images": [copied[item["image"]]], "question": item["question"]},
                "target": {"answer": item["answer"], "question_type": qtype},
                "evaluation": {"type": "vqa", "metric": "normalized_accuracy"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _spacenet7_cubes(source: Path) -> dict[str, list[tuple[Path, Path, str]]]:
    cubes: dict[str, list[tuple[Path, Path, str]]] = {}
    for image_dir in source.rglob("images_masked"):
        aoi_dir = image_dir.parent
        label_dir = aoi_dir / "labels_match_pix"
        if not label_dir.exists():
            continue
        labels = {p.stem: p for p in label_dir.glob("*.geojson")}
        items: list[tuple[Path, Path, str]] = []
        for image in sorted(image_dir.glob("*.tif")):
            label = labels.get(image.stem)
            if label:
                items.append((image, label, image.stem))
        if len(items) >= 2:
            cubes[aoi_dir.name] = items
    return cubes


def _feature_identity(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    for key in ("Id", "id", "ID", "building_id", "track_id", "uid"):
        if props.get(key) not in (None, ""):
            return str(props[key])
    geometry = json.dumps(feature.get("geometry"), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha1(geometry.encode()).hexdigest()[:16]


def _load_features(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    return data.get("features", []) if isinstance(data, dict) else []


def _make_change_mask(features: Iterable[dict[str, Any]], reference_image: Path, destination: Path) -> None:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.features import rasterize

    with rasterio.open(reference_image) as src:
        shape = (src.height, src.width)
    geometries = [(f["geometry"], 255) for f in features if f.get("geometry")]
    mask = rasterize(geometries, out_shape=shape, fill=0, default_value=255, dtype="uint8") if geometries else np.zeros(shape, dtype="uint8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(destination)


def sample_spacenet7_change(source: Path, output: Path) -> Path:
    leaf = "change_localization"
    cubes = _spacenet7_cubes(source)
    candidates: list[tuple[str, tuple[Path, Path, str], tuple[Path, Path, str]]] = []
    for aoi, items in sorted(cubes.items()):
        for first, second in zip(items, items[1:]):
            candidates.append((aoi, first, second))
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: f"{x[0]}:{x[1][2]}:{x[2][2]}")
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (aoi, first, second) in enumerate(selected):
        f1 = _load_features(first[1])
        f2 = _load_features(second[1])
        ids1 = {_feature_identity(f): f for f in f1}
        ids2 = {_feature_identity(f): f for f in f2}
        changed_ids = sorted(set(ids1) ^ set(ids2))
        changed_features = [ids1.get(k) or ids2[k] for k in changed_ids]
        mask_path = task_dir / "assets" / f"{i:03d}_change.png"
        _make_change_mask(changed_features, second[0], mask_path)
        record = base_record(
            leaf,
            i,
            "SpaceNet 7 Multi-Temporal Urban Development",
            "https://registry.opendata.aws/spacenet/",
            "CC-BY-SA-4.0",
            aoi,
        )
        record.update(
            {
                "input": {
                    "images": [
                        copy_asset(first[0], task_dir / "assets", f"{i:03d}_t1.tif"),
                        copy_asset(second[0], task_dir / "assets", f"{i:03d}_t2.tif"),
                    ],
                    "question": "Localize building construction or demolition between the first and second date.",
                    "timestamps": [first[2], second[2]],
                },
                "target": {"change_mask": mask_path.relative_to(task_dir).as_posix(), "changed_object_ids": changed_ids},
                "evaluation": {"type": "change_detection", "metrics": ["change_iou", "object_f1"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def sample_spacenet7_matching(source: Path, output: Path) -> Path:
    leaf = "temporal_scene_matching"
    cubes = _spacenet7_cubes(source)
    positives: list[tuple[str, Path, Path, str]] = []
    for aoi, items in sorted(cubes.items()):
        for first, second in zip(items, items[1:]):
            positives.append((aoi, first[0], second[0], f"{aoi}:{first[2]}:{second[2]}"))
    rng = random.Random(SEEDS[leaf])
    pos = stable_sample(positives, 50, SEEDS[leaf], key=lambda x: x[3])
    aoi_names = sorted(cubes)
    negatives: list[tuple[str, Path, Path, str]] = []
    attempts = 0
    while len(negatives) < 50 and attempts < 10000:
        a, b = rng.sample(aoi_names, 2)
        ia = rng.choice(cubes[a])
        ib = rng.choice(cubes[b])
        key = f"{a}:{ia[2]}|{b}:{ib[2]}"
        if all(existing[3] != key for existing in negatives):
            negatives.append((f"{a}|{b}", ia[0], ib[0], key))
        attempts += 1
    if len(negatives) != 50:
        raise ValueError("Could not create 50 unique negative temporal matches")
    combined = [(True, x) for x in pos] + [(False, x) for x in negatives]
    rng.shuffle(combined)
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (answer, item) in enumerate(combined):
        aoi, first, second, key = item
        record = base_record(
            leaf,
            i,
            "SpaceNet 7 Multi-Temporal Urban Development",
            "https://registry.opendata.aws/spacenet/",
            "CC-BY-SA-4.0",
            aoi,
        )
        record.update(
            {
                "input": {
                    "images": [
                        copy_asset(first, task_dir / "assets", f"{i:03d}_a.tif"),
                        copy_asset(second, task_dir / "assets", f"{i:03d}_b.tif"),
                    ],
                    "question": "Do these images depict the same geographic area at different times?",
                    "choices": ["yes", "no"],
                },
                "target": {"answer": "yes" if answer else "no", "same_area": answer, "pair_key": key},
                "evaluation": {"type": "binary_classification", "metric": "accuracy"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def sample_geowebnews(source: Path, output: Path) -> Path:
    leaf = "toponym_recognition"
    candidates: list[dict[str, Any]] = []
    for text_path in sorted(source.rglob("*.txt")):
        ann_path = text_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        spans: list[dict[str, Any]] = []
        for line in ann_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("T"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            spec = fields[1].split()
            if len(spec) < 3 or ";" in fields[1]:
                continue
            try:
                start, end = int(spec[1]), int(spec[2])
            except ValueError:
                continue
            spans.append({"start": start, "end": end, "text": fields[2], "label": spec[0]})
        if spans:
            candidates.append({"key": text_path.as_posix(), "text": text, "spans": spans, "doc": text_path.stem})
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: x["key"])
    records: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        record = base_record(
            leaf,
            i,
            "GeoWebNews",
            "https://github.com/milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation",
            "Repository GPL-3.0; verify underlying corpus notices",
            item["doc"],
        )
        record.update(
            {
                "input": {"text": item["text"], "question": "Identify every toponym mention and its character offsets."},
                "target": {"spans": item["spans"]},
                "evaluation": {"type": "sequence_labeling", "metric": "strict_span_f1"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _spatial_qa_candidates(source: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        for obj in iter_dicts(data):
            context = obj.get("story") or obj.get("context") or obj.get("description")
            question = obj.get("question") or obj.get("query")
            answer = obj.get("answer") or obj.get("answers") or obj.get("label")
            relations = obj.get("relations") or obj.get("triplets") or obj.get("spatial_relations")
            if context and question and answer is not None:
                key = str(obj.get("id", f"{path.name}:{len(out)}"))
                out.append(
                    {
                        "key": key,
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "relations": relations,
                        "group": path.stem,
                    }
                )
    return out


def sample_spartun(source: Path, output: Path) -> Path:
    leaf = "textual_spatial_relation_extraction"
    candidates = _spatial_qa_candidates(source)
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: x["key"])
    records: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        record = base_record(
            leaf,
            i,
            "SpaRTUN / ReSQ",
            "https://github.com/HLR/SpaRTUN",
            "MIT",
            item["group"],
        )
        record.update(
            {
                "input": {
                    "text": item["context"],
                    "question": item["question"],
                    "instruction": "Extract the relevant spatial relation(s) and answer the question.",
                },
                "target": {"answer": item["answer"], "relations": item["relations"]},
                "evaluation": {"type": "relation_extraction_qa", "metrics": ["exact_match", "relation_f1"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def sample_geoquestions(source: Path, output: Path) -> Path:
    leaf = "geographic_fact_reasoning"
    candidates: list[dict[str, Any]] = []
    csv_paths = list(source.rglob("GeoQuestions1089.csv"))
    if csv_paths:
        with csv_paths[0].open(encoding="utf-8-sig", newline="") as f:
            for row_index, row in enumerate(csv.DictReader(f)):
                lowered = {str(k).lower(): v for k, v in row.items()}
                question = lowered.get("question") or lowered.get("natural language question")
                answer = lowered.get("answer") or lowered.get("answers")
                query = lowered.get("query") or lowered.get("sparql") or lowered.get("geosparql")
                if question and answer:
                    candidates.append({"key": str(row_index), "question": question, "answer": answer, "query": query})
    else:
        for path in source.rglob("GeoQuestions1089*.json"):
            for obj in iter_dicts(load_json(path)):
                question = obj.get("question") or obj.get("Question")
                answer = obj.get("answer") or obj.get("answers") or obj.get("Answer")
                query = obj.get("query") or obj.get("sparql") or obj.get("GeoSPARQL")
                if question and answer is not None:
                    candidates.append(
                        {"key": str(obj.get("id", len(candidates))), "question": question, "answer": answer, "query": query}
                    )
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: x["key"])
    records: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        record = base_record(
            leaf,
            i,
            "GeoQuestions1089",
            "https://github.com/AI-team-UoA/GeoQuestions1089",
            "CC-BY-4.0",
            item["key"],
        )
        record.update(
            {
                "input": {"question": item["question"]},
                "target": {"answer": item["answer"], "reference_query": item["query"]},
                "evaluation": {"type": "geospatial_qa", "metrics": ["answer_f1", "execution_accuracy"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)

