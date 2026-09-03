from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geomapbench_eval.prompts import build_messages
from geomapbench_eval.task_metrics import TASK_METRIC_SCHEMA, artifact_contract, evaluate_task_aware


def test_every_leaf_has_at_most_three_paper_metrics() -> None:
    assert len(TASK_METRIC_SCHEMA) == 23
    assert all(1 <= len(metrics) <= 3 for metrics in TASK_METRIC_SCHEMA.values())


def test_numeric_near_miss_gets_partial_task_score_but_not_strict_score(tmp_path: Path) -> None:
    record = {
        "id": "metric-000", "leaf": "metric_distance_computation",
        "input": {"question": "distance"},
        "target": {"bloom_answer": 100.0},
        "evaluation": {"type": "numeric", "target_field": "target.bloom_answer", "relative_tolerance": 0.005},
    }
    result = evaluate_task_aware(record, '{"answer":105}', tmp_path)
    assert result["strict_score"] == 0.0
    assert 0.9 <= result["task_score"] < 1.0
    assert result["task_metrics"]["relative_error"] == 0.05


def test_rle_artifact_contract_and_perfect_mask_score(tmp_path: Path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 32:] = 1
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.fromarray(mask).save(assets / "mask.png")
    record = {
        "id": "land-000", "leaf": "dense_land_cover_labeling",
        "bloom": {"variant": "create_semantic_segmentation"},
        "input": {"question": "segment", "class_ontology": {"0": "a", "1": "b"}},
        "target": {"bloom_answer": "assets/mask.png", "ignore_index": 255},
        "evaluation": {"type": "semantic_segmentation", "target_field": "target.bloom_answer"},
    }
    response = json.dumps({"answer": {
        "encoding": "rle-row-major", "size": [64, 64],
        "runs": [[value, 32] for _ in range(64) for value in (0, 1)],
    }})
    result = evaluate_task_aware(record, response, tmp_path)
    assert result["task_metrics"]["mask_miou"] == 1.0
    assert result["task_metrics"]["mask_dice"] == 1.0
    assert "do not return a filename" in artifact_contract(record)
    messages = build_messages(record, tmp_path, include_images=False)
    assert "rle-row-major" in messages[1]["content"][0]["text"]


def test_invalid_json_is_a_final_zero_not_a_rescore_retry(tmp_path: Path) -> None:
    record = {
        "id": "fact-000", "leaf": "geographic_fact_reasoning",
        "input": {"question": "q"}, "target": {"bloom_answer": "Paris"},
        "evaluation": {"type": "exact_match", "target_field": "target.bloom_answer"},
    }
    result = evaluate_task_aware(record, "Paris", tmp_path)
    assert result["parse_error"] == "invalid_json"
    assert result["task_score"] == 0.0
