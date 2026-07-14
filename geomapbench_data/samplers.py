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


def _maptext_candidates(data: Any, image_lookup: dict[str, Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    image_records = data if isinstance(data, list) else data.get("images", data.get("data", []))
    if not isinstance(image_records, list):
        raise ValueError("MapText JSON must contain a list of image records")
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
        groups = image_record.get("groups") or image_record.get("annotations") or []
        for group_index, group in enumerate(groups):
            words: list[dict[str, Any]]
            if isinstance(group, list):
                words = [x for x in group if isinstance(x, dict)]
            elif isinstance(group, dict) and isinstance(group.get("words"), list):
                words = [x for x in group["words"] if isinstance(x, dict)]
            elif isinstance(group, dict):
                words = [group]
            else:
                continue
            valid = [
                w
                for w in words
                if str(w.get("text", "")).strip()
                and not bool(w.get("illegible", False))
                and not bool(w.get("truncated", False))
            ]
            if not valid:
                continue
            text = " ".join(str(w["text"]).strip() for w in valid)
            candidates.append(
                {
                    "key": f"{image_name}:{group_index}:{text}",
                    "image": image_path,
                    "image_name": image_name,
                    "text": text,
                    "words": [
                        {
                            "text": str(w["text"]).strip(),
                            "vertices": w.get("vertices") or w.get("polygon") or w.get("points"),
                        }
                        for w in valid
                    ],
                    "image_group": image_name or str(image_index),
                }
            )
    return candidates


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
    copied: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        if item["image"] not in copied:
            copied[item["image"]] = copy_asset(
                item["image"], task_dir / "assets", f"map_{len(copied):03d}{item['image'].suffix.lower()}"
            )
        record = base_record(
            leaf,
            i,
            "Paris and Jerusalem Maps Text Dataset",
            "https://doi.org/10.5281/zenodo.14982663",
            "CC-BY-4.0",
            item["image_group"],
        )
        record.update(
            {
                "input": {
                    "images": [copied[item["image"]]],
                    "question": "Transcribe the indicated map label and return its word polygons in reading order.",
                },
                "target": {"text": item["text"], "words": item["words"]},
                "evaluation": {
                    "type": "text_spotting",
                    "metrics": ["normalized_edit_distance", "polygon_iou", "grouping_f1"],
                },
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _paired_image_labels(source: Path) -> list[tuple[Path, Path, str]]:
    images: dict[tuple[str, str], Path] = {}
    labels: dict[tuple[str, str], Path] = {}
    for path in _image_files(source):
        parts = [p.lower() for p in path.parts]
        rel = path.relative_to(source)
        context_parts = [p for p in rel.parts[:-1] if p.lower() not in {"images", "image", "labels", "label", "masks", "mask"}]
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


def sample_openearthmap(source: Path, output: Path) -> Path:
    leaf = "dense_land_cover_labeling"
    pairs = _paired_image_labels(source)
    selected = stable_sample(pairs, N_EXAMPLES, SEEDS[leaf], key=lambda x: x[2])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    classes = ["bareland", "rangeland", "developed_space", "road", "tree", "water", "agriculture", "building"]
    for i, (image, label, group_id) in enumerate(selected):
        record = base_record(
            leaf,
            i,
            "OpenEarthMap",
            "https://doi.org/10.5281/zenodo.7223446",
            "Source-dependent; commonly CC-BY-NC-SA-4.0 (verify per region)",
            group_id.split("/")[0],
        )
        record.update(
            {
                "input": {
                    "images": [copy_asset(image, task_dir / "assets", f"{i:03d}_image{image.suffix.lower()}")],
                    "question": "Assign one land-cover class to every pixel.",
                    "classes": classes,
                },
                "target": {
                    "mask": copy_asset(label, task_dir / "assets", f"{i:03d}_mask{label.suffix.lower()}"),
                    "classes": classes,
                },
                "evaluation": {"type": "semantic_segmentation", "metrics": ["mean_iou", "macro_f1"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


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

