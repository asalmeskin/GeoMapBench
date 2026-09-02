from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from geomapbench_data.bloom import BLOOM_LEVELS
from geomapbench_data.common import SEEDS
from geomapbench_eval.common import append_jsonl, digest
from geomapbench_eval.cumulative import (
    cumulative_cohort,
    migrate_legacy_base_outputs,
    write_cohort_manifest,
)
from geomapbench_eval.prompts import build_messages
from geomapbench_eval.runner import run


ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[tuple[Path, dict]]:
    rows = []
    for leaf in sorted(SEEDS):
        levels = BLOOM_LEVELS[leaf]
        directory = Path("benchmark") / leaf
        for index in range(100):
            rows.append((directory, {
                "id": f"{leaf}-{index:03d}", "leaf": leaf,
                "bloom": {"level": levels[index % len(levels)]},
                "input": {"question": "Answer yes."},
                "target": {"bloom_answer": "yes"},
                "evaluation": {"target_field": "target.bloom_answer", "type": "exact_match"},
            }))
    return rows


def _benchmark(root: Path) -> None:
    for directory, record in _records():
        destination = root / directory.name
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "data_clean.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _args(benchmark: Path, output: Path, manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_root=str(benchmark), output=str(output), model="fake/model",
        condition="base", top_k=5, temperature=0.0, max_tokens=16384,
        reasoning_effort=None, reasoning_enabled=False, timeout_seconds=1,
        retries=0, request_delay_seconds=0.0, retry_base_seconds=0.0,
        retry_max_seconds=0.0, max_consecutive_errors=2,
        max_image_bytes=8_000_000, max_cost_usd=10.0, limit=None,
        per_leaf_limit=None, record_ids_file=str(manifest), cumulative=True,
        progress_every=1000, no_images=False, no_clean=False, force=False,
        benchmark_content_hash="test-benchmark-hash",
    )


class _SuccessClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def complete(self, messages, config):
        type(self).calls += 1
        return {
            "choices": [{"message": {"content": '{"answer":"yes"}'}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001}, "_latency_seconds": 0.01, "_transport_attempts": 1,
        }


def test_cohorts_are_nested_deterministic_and_bloom_stratified() -> None:
    rows = _records()
    one = cumulative_cohort(rows, 1)
    six = cumulative_cohort(rows, 6)
    full = cumulative_cohort(rows, 100)
    one_ids = {record["id"] for _, record in one}
    six_ids = {record["id"] for _, record in six}
    full_ids = {record["id"] for _, record in full}
    assert len(one) == 23 and len(six) == 138 and len(full) == 2300
    assert one_ids < six_ids < full_ids
    assert [record["id"] for _, record in one] == [
        f"{leaf}-000" for leaf in sorted(SEEDS)
    ]
    for leaf in sorted(SEEDS):
        selected_levels = {
            record["bloom"]["level"] for _, record in six if record["leaf"] == leaf
        }
        assert selected_levels == set(BLOOM_LEVELS[leaf])
    assert [record["id"] for _, record in six] == [
        record["id"] for _, record in cumulative_cohort(rows, 6)
    ]


def test_manifest_can_grow_but_not_shrink(tmp_path: Path) -> None:
    rows = _records()
    _, first = write_cohort_manifest(
        rows, target_per_leaf=1, output_root=tmp_path, benchmark_content_hash="hash"
    )
    _, sixth = write_cohort_manifest(
        rows, target_per_leaf=6, output_root=tmp_path, benchmark_content_hash="hash"
    )
    assert set(first["selected_ids"]) < set(sixth["selected_ids"])
    with pytest.raises(ValueError, match="cannot shrink"):
        write_cohort_manifest(
            rows, target_per_leaf=1, output_root=tmp_path, benchmark_content_hash="hash"
        )


def test_legacy_migration_checks_prompt_model_condition_and_cohort(tmp_path: Path) -> None:
    directory = tmp_path / "benchmark" / "leaf"
    directory.mkdir(parents=True)
    record = {
        "id": "leaf-000", "leaf": "leaf", "input": {"question": "Answer yes."},
        "target": {"bloom_answer": "yes"},
        "evaluation": {"target_field": "target.bloom_answer", "type": "exact_match"},
    }
    prompt_hash = digest(build_messages(record, directory, contexts=None, include_images=True))
    source = tmp_path / "legacy"
    append_jsonl(source / "responses.jsonl", {
        "id": record["id"], "status": "ok", "condition": "base",
        "model": "fake/model", "prompt_hash": prompt_hash, "score": 1.0,
    })
    append_jsonl(source / "responses.jsonl", {
        "id": "leaf-001", "status": "ok", "condition": "base",
        "model": "fake/model", "prompt_hash": "wrong",
    })
    append_jsonl(source / "api_responses.jsonl", {
        "cache_key": "key", "id": record["id"], "condition": "base",
        "model": "fake/model", "prompt_hash": prompt_hash, "response": {"choices": []},
    })
    report = migrate_legacy_base_outputs(
        [source], destination=tmp_path / "new", model="fake/model",
        target_ids={record["id"]}, records_by_id={record["id"]: (directory, record)},
        prompt_hash_cache={},
    )
    assert report["results_imported"] == 1
    assert report["api_responses_imported"] == 1
    assert report["rejected_or_duplicate_rows"] == 1


def test_runner_grows_in_place_and_calls_only_new_ids(tmp_path: Path) -> None:
    benchmark, output = tmp_path / "benchmark", tmp_path / "output"
    _benchmark(benchmark)
    records = [(benchmark / directory.name, record) for directory, record in _records()]
    manifest1, _ = write_cohort_manifest(
        records, target_per_leaf=1, output_root=tmp_path / "cohort", benchmark_content_hash="hash"
    )
    args = _args(benchmark, output, manifest1)
    _SuccessClient.calls = 0
    with patch("geomapbench_eval.runner.OpenRouterClient", _SuccessClient):
        first = run(args)
    assert first["completed_total"] == 23
    assert _SuccessClient.calls == 23

    manifest6, _ = write_cohort_manifest(
        records, target_per_leaf=6, output_root=tmp_path / "cohort", benchmark_content_hash="hash"
    )
    args.record_ids_file = str(manifest6)
    _SuccessClient.calls = 0
    with patch("geomapbench_eval.runner.OpenRouterClient", _SuccessClient):
        grown = run(args)
        repeated = run(args)
    assert grown["completed_total"] == 138
    assert _SuccessClient.calls == 115
    assert repeated["run_stop_reason"] == "already_complete"


def test_v7_matrix_has_five_companies_no_muse_and_claude_rag() -> None:
    models = json.loads(
        (ROOT / "config/evaluation_models_2026-09-v7.json").read_text(encoding="utf-8")
    )
    assert len(models) == 5
    ids = {row["model"] for row in models}
    assert "meta/muse-spark-1.2" not in ids
    assert {model.split("/", 1)[0] for model in ids} == {
        "openai", "google", "mistralai", "qwen", "anthropic",
    }
    by_id = {row["model"]: row for row in models}
    assert by_id["anthropic/claude-sonnet-5"]["recommended_for_rag"] is True
