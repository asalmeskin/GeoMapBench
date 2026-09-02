from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from geomapbench_data.common import SEEDS
from geomapbench_eval.common import append_jsonl, digest
from geomapbench_eval.openrouter import OpenRouterConfig, OpenRouterRetryExhausted
from geomapbench_eval.prompts import build_messages
from geomapbench_eval.rag import AgenticRAGRetriever
from geomapbench_eval.runner import experiment_identity, run


def _benchmark(root: Path) -> None:
    for leaf in sorted(SEEDS):
        directory = root / leaf
        directory.mkdir(parents=True)
        rows = [{
            "id": f"{leaf}-{index:03d}", "leaf": leaf,
            "input": {"question": "Answer yes."},
            "target": {"bloom_answer": "yes"},
            "evaluation": {"target_field": "target.bloom_answer", "type": "exact_match"},
        } for index in range(100)]
        (directory / "data_clean.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def _args(benchmark: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_root=str(benchmark), output=str(output), model="fake/model",
        condition="base", top_k=5, temperature=0.0, max_tokens=16384,
        reasoning_effort=None, reasoning_enabled=False, timeout_seconds=1,
        retries=0, request_delay_seconds=0.0, retry_base_seconds=0.0,
        retry_max_seconds=0.0, max_consecutive_errors=2,
        max_image_bytes=8_000_000, max_cost_usd=10.0, limit=None,
        per_leaf_limit=1, progress_every=100, no_images=False,
        no_clean=False, force=False,
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


def test_transport_circuit_breaker_pauses_and_exact_resume_finishes() -> None:
    class RateLimitedClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, messages, config):
            raise OpenRouterRetryExhausted("rate limited", status=429)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark, output = root / "benchmark", root / "output"
        _benchmark(benchmark)
        args = _args(benchmark, output)
        with patch("geomapbench_eval.runner.OpenRouterClient", RateLimitedClient):
            paused = run(args)
        assert paused["run_stop_reason"] == "transport_circuit_breaker"
        assert paused["attempted_this_invocation"] == 1
        assert paused["completed_total"] == 0
        assert paused["remaining_total"] == 23

        _SuccessClient.calls = 0
        with patch("geomapbench_eval.runner.OpenRouterClient", _SuccessClient):
            resumed = run(args)
        assert resumed["complete"] is True
        assert resumed["completed_total"] == 23
        assert _SuccessClient.calls == 23


def test_write_ahead_api_cache_avoids_duplicate_call() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark, output = root / "benchmark", root / "output"
        _benchmark(benchmark)
        output.mkdir()
        args = _args(benchmark, output)
        task_dir, record = experiment_identity(args)["records"][0]
        messages = build_messages(record, task_dir, contexts=None, include_images=True)
        prompt_hash = digest(messages)
        cache_key = digest({
            "id": record["id"], "model": args.model, "condition": args.condition,
            "prompt_hash": prompt_hash, "max_tokens": args.max_tokens,
            "reasoning_enabled": args.reasoning_enabled,
            "reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
        })
        append_jsonl(output / "api_responses.jsonl", {
            "cache_key": cache_key, "id": record["id"], "model": args.model,
            "condition": args.condition, "prompt_hash": prompt_hash,
            "retrieval_usage": {"cost": 0.0, "calls": 0},
            "response": {
                "choices": [{"message": {"content": '{"answer":"yes"}'}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001}, "_latency_seconds": 0.01,
            },
        })
        _SuccessClient.calls = 0
        with patch("geomapbench_eval.runner.OpenRouterClient", _SuccessClient):
            result = run(args)
        assert result["api_response_cache_hits"] == 1
        assert _SuccessClient.calls == 22
        assert result["complete"] is True


def test_invalid_agent_output_is_not_cached() -> None:
    class InvalidClient:
        calls = 0

        def complete(self, messages, config):
            self.calls += 1
            return {
                "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001},
            }

    with tempfile.TemporaryDirectory() as temporary:
        retriever = AgenticRAGRetriever.__new__(AgenticRAGRetriever)
        retriever.agent_model = "fake/agent"
        retriever.agent_cache = Path(temporary)
        retriever.agent_config = OpenRouterConfig("fake/agent")
        retriever.agent_client = InvalidClient()
        retriever._query_usage = {
            "cost": 0.0, "calls": 0, "cached_calls": 0,
            "agent_failures": 0, "failure_kinds": [],
        }
        assert retriever._agent("system", "user", "plan").get("_error") == "agent_invalid_json"
        assert retriever._agent("system", "user", "plan").get("_error") == "agent_invalid_json"
        assert retriever.agent_client.calls == 2
        assert list(Path(temporary).glob("*.json")) == []
