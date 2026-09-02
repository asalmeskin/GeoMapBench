from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geomapbench_eval.benchmark import stable_subset
from geomapbench_eval.analysis import compare
from geomapbench_eval.openrouter import generation_failure, response_text
from geomapbench_eval.preflight import benchmark_preflight
from geomapbench_eval.runner import run
from geomapbench_eval.scoring import is_artifact_target, score
from geomapbench_data.common import SEEDS


class EvalV18Tests(unittest.TestCase):
    def test_subset_is_selected_before_resume_filtering(self) -> None:
        records = []
        for leaf in ("a", "b"):
            directory = Path(leaf)
            for index in range(3):
                records.append((directory, {"id": f"{leaf}-{index}", "leaf": leaf}))
        selected = stable_subset(records, per_leaf_limit=1, limit=None)
        self.assertEqual([row[1]["id"] for row in selected], ["a-0", "b-0"])

    def test_structured_numeric_uses_tolerance_recursively(self) -> None:
        record = {
            "target": {"bloom_answer": {"value": 10.0, "items": [1.0, "A"]}},
            "evaluation": {
                "target_field": "target.bloom_answer",
                "type": "structured_numeric",
                "absolute_tolerance": 0.01,
            },
        }
        result = score(record, '{"answer":{"value":10.005,"items":[1.004,"a"]}}')
        self.assertEqual(result["score"], 1.0)

    def test_empty_and_length_responses_are_generation_failures(self) -> None:
        empty = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        length = {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}
        self.assertEqual(response_text(empty), "")
        self.assertEqual(generation_failure(empty, ""), "empty_response")
        self.assertEqual(generation_failure(length, "{}"), "token_limit")

    def test_file_gold_is_flagged_for_separate_reporting(self) -> None:
        record = {
            "target": {"bloom_answer": "assets/change.png"},
            "evaluation": {"target_field": "target.bloom_answer"},
        }
        self.assertTrue(is_artifact_target(record))

    def test_preflight_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            leaf = root / "leaf"
            cache = Path(temporary) / "cache"
            leaf.mkdir(parents=True)
            record = {
                "id": "leaf-000", "leaf": "leaf", "input": {"question": "q"},
                "target": {"bloom_answer": "a"},
                "evaluation": {"target_field": "target.bloom_answer"},
            }
            (leaf / "data_clean.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = [(leaf, record)]
            with patch("geomapbench_eval.preflight.canonical_benchmark_records", return_value=rows):
                first = benchmark_preflight(root, cache_root=cache)
                second = benchmark_preflight(root, cache_root=cache)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])

    def test_runner_pilot_is_resumable_end_to_end(self) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def complete(self, messages, config):
                return {
                    "choices": [{"message": {"content": '{"answer":"yes"}'}, "finish_reason": "stop"}],
                    "usage": {"cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 2}},
                    "_latency_seconds": 0.01,
                }

        with tempfile.TemporaryDirectory() as temporary:
            benchmark = Path(temporary) / "benchmark"
            output = Path(temporary) / "output"
            for leaf in sorted(SEEDS):
                directory = benchmark / leaf
                directory.mkdir(parents=True)
                rows = []
                for index in range(100):
                    rows.append({
                        "id": f"{leaf}-{index:03d}", "leaf": leaf,
                        "input": {"question": "Answer yes."},
                        "target": {"bloom_answer": "yes"},
                        "evaluation": {"target_field": "target.bloom_answer", "type": "exact_match"},
                    })
                (directory / "data_clean.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            args = argparse.Namespace(
                benchmark_root=str(benchmark), output=str(output), model="fake/model",
                condition="base", top_k=5, temperature=0.0, max_tokens=8192,
                reasoning_effort="minimal", timeout_seconds=1, retries=0,
                max_image_bytes=8_000_000, max_cost_usd=10.0, limit=None,
                per_leaf_limit=1, progress_every=100, no_images=False,
                no_clean=False, force=False,
            )
            with patch("geomapbench_eval.runner.OpenRouterClient", FakeClient):
                first = run(args)
                second = run(args)
            self.assertEqual(first["succeeded"], 23)
            self.assertEqual(first["generation_failures"], 0)
            self.assertEqual(second["run_stop_reason"], "already_complete")
            rag_output = Path(temporary) / "rag_output"
            shutil.copytree(output, rag_output)
            rag_config_path = rag_output / "run_config.json"
            rag_config = json.loads(rag_config_path.read_text())
            rag_config["condition"] = "base_rag"
            rag_config_path.write_text(json.dumps(rag_config), encoding="utf-8")
            comparison = compare(
                output / "responses.jsonl", rag_output / "responses.jsonl",
                Path(temporary) / "comparison",
            )
            self.assertEqual(comparison["protocol_validation"], "pass")
            rag_config["max_tokens"] = 4096
            rag_config_path.write_text(json.dumps(rag_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison refused"):
                compare(
                    output / "responses.jsonl", rag_output / "responses.jsonl",
                    Path(temporary) / "bad_comparison",
                )


if __name__ == "__main__":
    unittest.main()
