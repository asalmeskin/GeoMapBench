from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from .analysis import analyze, compare
from .common import atomic_json
from .preflight import benchmark_preflight
from .rag import stage_corpus
from .runner import run


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def _run_args(
    *, benchmark_root: Path, output: Path, model: str, condition: str,
    corpus_root: Path | None = None, per_leaf_limit: int | None = None,
    max_cost_usd: float | None = None,
    agent_model: str = "google/gemini-3.5-flash-lite",
) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_root=str(benchmark_root), output=str(output), model=model,
        condition=condition, corpus_root=str(corpus_root) if corpus_root else None,
        rag_backend="dense", top_k=5, candidate_k=50,
        reranker_model="BAAI/bge-reranker-base", max_passage_chars=1500,
        max_context_chars=6000, no_capability_gating=False,
        agent_model=agent_model, agent_max_hops=2, agent_subqueries=3,
        temperature=0.0, max_tokens=512, timeout_seconds=120, retries=4,
        max_image_bytes=8_000_000, max_cost_usd=max_cost_usd,
        limit=None, per_leaf_limit=per_leaf_limit, no_images=False,
        no_clean=False, force=False,
    )


def load_model_matrix(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not 7 <= len(models) <= 8:
        raise ValueError("Scientific model matrix must contain 7 or 8 models")
    ids = [str(row.get("id")) for row in models]
    if len(ids) != len(set(ids)) or any("/" not in model for model in ids):
        raise ValueError("Model matrix contains a missing or duplicate OpenRouter model ID")
    return models


def available_openrouter_models(timeout: int = 30) -> set[str]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "GeoMapBench/1.7"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(row.get("id")) for row in payload.get("data", [])}


def run_model_suite(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root, output = Path(args.benchmark_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    preflight = benchmark_preflight(benchmark_root, encode_assets=not args.skip_asset_preflight)
    atomic_json(output / "benchmark_preflight.json", preflight)
    models = load_model_matrix(Path(args.models))
    if not args.skip_model_preflight:
        available = available_openrouter_models()
        missing = [row["id"] for row in models if row["id"] not in available]
        if missing:
            raise ValueError(f"OpenRouter model IDs are no longer available: {missing}")
    rows, failures = [], []
    for item in models:
        model = str(item["id"])
        model_dir = output / _slug(model)
        try:
            report = run(_run_args(
                benchmark_root=benchmark_root, output=model_dir, model=model,
                condition="base", per_leaf_limit=args.per_leaf_limit,
                max_cost_usd=args.max_cost_usd_per_model,
            ))
            summary = analyze(model_dir / "responses.jsonl", model_dir / "analysis")
            rows.append({
                "model": model, "family": item.get("family"),
                "open_weights": item.get("open_weights"),
                "n": summary["record_count"],
                "macro_accuracy": summary["macro_by_condition"].get("base"),
                "micro_accuracy": summary["micro_by_condition"].get("base"),
                "invalid_json_rate": summary["invalid_json_rate"],
                "mean_latency_seconds": summary["mean_latency_seconds"],
                "reported_cost_usd": summary["reported_cost_usd"],
                "run_stop_reason": report.get("stop_reason"),
            })
        except Exception as error:
            failures.append({"model": model, "error": repr(error)})
            if args.fail_fast:
                raise
    fieldnames = [
        "model", "family", "open_weights", "n", "macro_accuracy",
        "micro_accuracy", "invalid_json_rate", "mean_latency_seconds",
        "reported_cost_usd", "run_stop_reason",
    ]
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    result = {"models_requested": len(models), "models_reported": len(rows), "failures": failures, "rows": rows}
    atomic_json(output / "model_comparison.json", result)
    return result


def run_rag_experiment(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root, output = Path(args.benchmark_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    preflight = benchmark_preflight(benchmark_root, encode_assets=not args.skip_asset_preflight)
    atomic_json(output / "benchmark_preflight.json", preflight)
    corpus = Path(args.corpus_root)
    if args.corpus_local_cache:
        corpus = stage_corpus(corpus, Path(args.corpus_local_cache))
    base_dir = output / "base"
    base_rag_dir = output / "base_rag"
    agentic_rag_dir = output / "agentic_rag"
    base_report = run(_run_args(
        benchmark_root=benchmark_root, output=base_dir, model=args.model,
        condition="base", per_leaf_limit=args.per_leaf_limit,
        max_cost_usd=args.max_cost_usd_per_condition,
    ))
    base_rag_report = run(_run_args(
        benchmark_root=benchmark_root, output=base_rag_dir, model=args.model,
        condition="base_rag", corpus_root=corpus, per_leaf_limit=args.per_leaf_limit,
        max_cost_usd=args.max_cost_usd_per_condition,
    ))
    agentic_rag_report = run(_run_args(
        benchmark_root=benchmark_root, output=agentic_rag_dir, model=args.model,
        condition="agentic_rag", corpus_root=corpus, per_leaf_limit=args.per_leaf_limit,
        max_cost_usd=args.max_cost_usd_per_condition, agent_model=args.agent_model,
    ))
    base_analysis = analyze(base_dir / "responses.jsonl", base_dir / "analysis")
    base_rag_analysis = analyze(base_rag_dir / "responses.jsonl", base_rag_dir / "analysis")
    agentic_rag_analysis = analyze(agentic_rag_dir / "responses.jsonl", agentic_rag_dir / "analysis")
    base_to_base_rag = compare(
        base_dir / "responses.jsonl", base_rag_dir / "responses.jsonl", output / "comparisons/base_to_base_rag",
    )
    base_to_agentic_rag = compare(
        base_dir / "responses.jsonl", agentic_rag_dir / "responses.jsonl", output / "comparisons/base_to_agentic_rag",
    )
    base_rag_to_agentic_rag = compare(
        base_rag_dir / "responses.jsonl", agentic_rag_dir / "responses.jsonl", output / "comparisons/base_rag_to_agentic_rag",
    )
    result = {
        "model": args.model,
        "agent_model": args.agent_model,
        "retrieval": {
            "base_rag": "BGE dense + cross-encoder reranking",
            "agentic_rag": "LLM planner/judge + BGE dense + cross-encoder reranking",
            "bm25": False,
        },
        "base_report": base_report,
        "base_rag_report": base_rag_report,
        "agentic_rag_report": agentic_rag_report,
        "base_macro": base_analysis["macro_by_condition"].get("base"),
        "base_rag_macro": base_rag_analysis["macro_by_condition"].get("base_rag"),
        "agentic_rag_macro": agentic_rag_analysis["macro_by_condition"].get("agentic_rag"),
        "comparisons": {
            "base_to_base_rag": base_to_base_rag,
            "base_to_agentic_rag": base_to_agentic_rag,
            "base_rag_to_agentic_rag": base_rag_to_agentic_rag,
        },
    }
    atomic_json(output / "experiment_summary.json", result)
    return result


def add_experiment_parsers(sub: argparse._SubParsersAction[Any]) -> None:
    suite = sub.add_parser("suite", help="Run and report the frozen 7-8 model base benchmark.")
    suite.add_argument("--benchmark-root", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--models", required=True)
    suite.add_argument("--per-leaf-limit", type=int)
    suite.add_argument("--max-cost-usd-per-model", type=float)
    suite.add_argument("--skip-model-preflight", action="store_true")
    suite.add_argument("--skip-asset-preflight", action="store_true")
    suite.add_argument("--fail-fast", action="store_true")

    rag = sub.add_parser("rag-experiment", help="Run paired base, base_rag, and agentic_rag on one answer model.")
    rag.add_argument("--benchmark-root", required=True)
    rag.add_argument("--corpus-root", required=True)
    rag.add_argument("--corpus-local-cache")
    rag.add_argument("--output", required=True)
    rag.add_argument("--model", default="qwen/qwen3.8-flash")
    rag.add_argument("--agent-model", default="google/gemini-3.5-flash-lite")
    rag.add_argument("--per-leaf-limit", type=int)
    rag.add_argument("--max-cost-usd-per-condition", type=float)
    rag.add_argument("--skip-asset-preflight", action="store_true")
