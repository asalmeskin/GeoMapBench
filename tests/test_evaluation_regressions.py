from __future__ import annotations

import base64
import io
import json
import urllib.error
import argparse
from pathlib import Path

from geomapbench_data.common import SEEDS
from geomapbench_eval.preflight import canonical_benchmark_records
from geomapbench_eval.prompts import _encode_image
from geomapbench_eval.openrouter import OpenRouterClient, OpenRouterConfig
from geomapbench_eval.rag import AgenticRAGRetriever
from geomapbench_eval.runner import select_records
from geomapbench_eval import runner


def test_canonical_loader_ignores_drive_copy_directories(tmp_path: Path) -> None:
    import json
    for leaf in SEEDS:
        directory = tmp_path / leaf
        directory.mkdir()
        with (directory / "data_clean.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(100):
                handle.write(json.dumps({"id": f"{leaf}-{index:03d}", "leaf": leaf, "input": {"question": "q"}}) + "\n")
    duplicate = tmp_path / "dense_land_cover_labeling (1)"
    duplicate.mkdir()
    (duplicate / "data.jsonl").write_text('{"id":"duplicate"}\n', encoding="utf-8")
    rows = canonical_benchmark_records(tmp_path)
    assert len(rows) == 2300
    assert all(" (1)" not in directory.name for directory, _ in rows)


def test_svg_is_rasterized_to_supported_png(tmp_path: Path) -> None:
    path = tmp_path / "icon.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"><path d="M0 0h15v15H0z"/></svg>', encoding="utf-8")
    part = _encode_image(path, 8_000_000)
    prefix, payload = part["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(payload).startswith(b"\x89PNG")


def test_subset_is_frozen_before_resume_filtering(tmp_path: Path) -> None:
    rows = []
    for leaf in ("a", "b"):
        for index in range(2):
            rows.append((tmp_path / leaf, {"id": f"{leaf}-{index}", "leaf": leaf}))
    remaining = select_records(rows, {"a-0"}, per_leaf_limit=1)
    assert [record["id"] for _, record in remaining] == ["b-0"]


def test_permanent_http_400_is_not_retried(monkeypatch) -> None:
    calls = 0

    def reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "https://openrouter.ai", 400, "Bad Request", {},
            io.BytesIO(b'{"error":"unsupported payload"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)
    client = OpenRouterClient(api_key="test")
    try:
        client.complete([{"role": "user", "content": "x"}], OpenRouterConfig("test/model", retries=4))
    except RuntimeError as error:
        assert "HTTP 400" in str(error)
        assert "unsupported payload" in str(error)
    else:
        raise AssertionError("Expected the permanent HTTP 400 to be surfaced")
    assert calls == 1


def test_agent_usage_is_counted_and_valid_response_is_cached(tmp_path: Path) -> None:
    class FakeClient:
        calls = 0

        def complete(self, messages, config):
            self.calls += 1
            return {
                "choices": [{"message": {"content": '{"queries":["EPSG 4326"]}'}}],
                "usage": {"cost": 0.002}, "_latency_seconds": 0.1,
            }

    retriever = AgenticRAGRetriever.__new__(AgenticRAGRetriever)
    retriever.agent_model = "test/agent"
    retriever.agent_cache = tmp_path
    retriever.agent_client = FakeClient()
    retriever.agent_config = OpenRouterConfig("test/agent")
    retriever._pending_usage = []
    first = retriever._agent("system", "user", "planner")
    second = retriever._agent("system", "user", "planner")
    events = retriever.pop_usage()
    assert first == second == {"queries": ["EPSG 4326"]}
    assert retriever.agent_client.calls == 1
    assert sum(float((event.get("usage") or {}).get("cost") or 0) for event in events) == 0.002
    assert [event["cached"] for event in events] == [False, True]


def test_agentic_merge_exposes_every_subquery_to_the_judge() -> None:
    merged = AgenticRAGRetriever._merge_rankings([
        [{"id": "a1"}, {"id": "shared"}, {"id": "a3"}],
        [{"id": "b1"}, {"id": "shared"}, {"id": "b3"}],
        [{"id": "c1"}, {"id": "c2"}],
    ])
    assert [row["id"] for row in merged[:4]] == ["a1", "b1", "c1", "shared"]
    assert [row["id"] for row in merged].count("shared") == 1


def test_agentic_runner_reports_answer_agent_and_total_cost(monkeypatch, tmp_path: Path) -> None:
    task = tmp_path / "leaf"
    task.mkdir()
    records = [
        (task, {"id": f"x-{index}", "leaf": "leaf", "input": {"question": "q"}, "target": {"answer": "ok"}})
        for index in range(2)
    ]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            self.events = []

        def search(self, query, leaf, top_k):
            self.events = [{"role": "planner", "usage": {"cost": 0.002}}]
            return []

        def pop_usage(self):
            events, self.events = self.events, []
            return events

        def supports_leaf(self, leaf):
            return True

    class FakeAnswerClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, messages, config):
            return {
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"cost": 0.001}, "_latency_seconds": 0.1,
            }

    monkeypatch.setattr(runner, "benchmark_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(runner, "AgenticRAGRetriever", FakeRetriever)
    monkeypatch.setattr(runner, "OpenRouterClient", FakeAnswerClient)
    args = argparse.Namespace(
        benchmark_root=str(tmp_path), output=str(tmp_path / "out"), model="test/answer",
        condition="agentic_rag", corpus_root=str(tmp_path), rag_backend="dense",
        top_k=5, candidate_k=50, reranker_model="none", max_passage_chars=1500,
        max_context_chars=6000, no_capability_gating=False, agent_model="test/agent",
        agent_max_hops=2, agent_subqueries=3, temperature=0.0, max_tokens=32,
        timeout_seconds=1, retries=0, max_image_bytes=1000, max_cost_usd=None,
        limit=None, per_leaf_limit=None, no_images=False, no_clean=False, force=False,
    )
    report = runner.run(args)
    result_rows = [json.loads(line) for line in (tmp_path / "out/responses.jsonl").read_text().splitlines()]
    assert report["answer_cost_usd_this_invocation"] == 0.002
    assert report["agent_cost_usd_this_invocation"] == 0.004
    assert report["reported_cost_usd_total"] == 0.006
    assert all(row["total_cost_usd"] == 0.003 for row in result_rows)
