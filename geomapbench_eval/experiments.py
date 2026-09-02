from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .analysis import analyze, compare
from .common import atomic_json
from .openrouter import OpenRouterClient
from .preflight import benchmark_preflight
from .rag import AgenticRAGRetriever, DenseRAGRetriever, stage_corpus
from .runner import run


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _models(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Model config must be a non-empty JSON array: {path}")
    rows = [row for row in payload if isinstance(row, dict) and row.get("enabled", True)]
    for row in rows:
        if not row.get("model"):
            raise ValueError("Every model config row needs a model ID")
    return rows


def _model_row(path: Path, model: str) -> dict[str, Any]:
    matches = [row for row in _models(path) if str(row["model"]) == model]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one config row for {model!r} in {path}, found {len(matches)}")
    return matches[0]


def _run_namespace(
    args: argparse.Namespace,
    *,
    output: Path,
    model: str,
    condition: str,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_enabled: bool,
    temperature: float | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_root=args.benchmark_root,
        output=str(output),
        model=model,
        condition=condition,
        top_k=getattr(args, "top_k", 5),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_enabled=reasoning_enabled,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        request_delay_seconds=args.request_delay_seconds,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        max_consecutive_errors=args.max_consecutive_errors,
        max_image_bytes=8_000_000,
        max_cost_usd=getattr(args, "max_cost_usd_per_model", None),
        limit=None,
        per_leaf_limit=args.per_leaf_limit,
        progress_every=args.progress_every,
        no_images=False,
        no_clean=False,
        force=False,
        benchmark_content_hash=getattr(args, "benchmark_content_hash", None),
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    cache = Path(args.preflight_cache).expanduser() if args.preflight_cache else Path(args.output) / "_preflight_cache"
    return benchmark_preflight(
        Path(args.benchmark_root), cache_root=cache,
        force=args.force_preflight, max_image_bytes=8_000_000,
    )


def _catalog_check(model_rows: list[dict[str, Any]]) -> dict[str, str]:
    print("[suite] checking current OpenRouter model IDs", flush=True)
    try:
        catalog = OpenRouterClient().model_catalog()
    except Exception as error:
        print(
            f"[suite:warning] live catalog check unavailable ({error!r}); "
            "continuing with the frozen, previously validated model matrix",
            flush=True,
        )
        return {}
    problems: dict[str, str] = {}
    for row in model_rows:
        model = str(row["model"])
        card = catalog.get(model)
        if not card:
            problems[model] = "model_unavailable_in_catalog"
            continue
        modalities = set((card.get("architecture") or {}).get("input_modalities") or [])
        parameters = set(card.get("supported_parameters") or [])
        if "image" not in modalities:
            problems[model] = "model_catalog_does_not_advertise_image_input"
        elif "response_format" not in parameters:
            problems[model] = "model_catalog_does_not_advertise_response_format"
        reasoning = card.get("reasoning") or {}
        if reasoning.get("mandatory") and not bool(row.get("reasoning_enabled")):
            problems[model] = "catalog_requires_reasoning_but_config_disables_it"
        configured_effort = row.get("reasoning_effort")
        supported_efforts = set(reasoning.get("supported_efforts") or [])
        if configured_effort and supported_efforts and configured_effort not in supported_efforts:
            problems[model] = f"unsupported_reasoning_effort:{configured_effort}"
    if problems:
        print(f"[suite:warning] catalog validation skips: {problems}", flush=True)
    else:
        print("[suite] model preflight PASS", flush=True)
    return problems


def run_model_suite(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[suite] output: {output}", flush=True)
    preflight = _preflight(args)
    args.benchmark_content_hash = preflight["portable_benchmark_hash"]
    model_rows = _models(Path(args.models))
    print(f"[suite] loaded {len(model_rows)} configured models", flush=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if not args.skip_model_preflight:
        problems = _catalog_check(model_rows)
        failures.extend({"model": model, "error": error} for model, error in sorted(problems.items()))
        model_rows = [row for row in model_rows if str(row["model"]) not in problems]
    for index, model_row in enumerate(model_rows, 1):
        model = str(model_row["model"])
        print(f"[suite {index}/{len(model_rows)}] START {model}", flush=True)
        model_output = output / _slug(model)
        try:
            namespace = _run_namespace(
                args,
                output=model_output,
                model=model,
                condition="base",
                max_tokens=int(model_row.get("max_tokens", args.max_tokens)),
                reasoning_effort=model_row.get("reasoning_effort"),
                reasoning_enabled=bool(model_row.get("reasoning_enabled", False)),
                temperature=model_row.get("temperature", 0.0),
            )
            invocation = run(namespace)
            summary = analyze(model_output / "responses.jsonl", model_output / "analysis")
            condition = summary["condition_summary"].get("base", {})
            macro = summary["macro_by_condition"].get("base", 0.0)
            complete = bool(invocation.get("complete"))
            report = {
                "model": model,
                "family": model_row.get("family"),
                "tier": model_row.get("tier"),
                "open_weights": bool(model_row.get("open_weights")),
                "n": condition.get("n", 0),
                "target_n": invocation.get("target_records"),
                "complete": complete,
                "macro_accuracy": macro if complete else None,
                "provisional_macro_accuracy": macro,
                "text_answer_macro_accuracy": summary["text_answer_macro_by_condition"].get("base", 0.0),
                "micro_accuracy": condition.get("micro_accuracy", 0.0),
                "text_answer_micro_accuracy": condition.get("text_answer_micro_accuracy", 0.0),
                "artifact_target_rate": condition.get("artifact_target_rate", 0.0),
                "invalid_json_rate": condition.get("invalid_json_rate", 0.0),
                "generation_failure_rate": condition.get("generation_failure_rate", 0.0),
                "mean_latency_seconds": condition.get("mean_latency_seconds", 0.0),
                "reported_cost_usd": invocation.get(
                    "reported_cost_usd_total", condition.get("total_cost_usd", 0.0)
                ),
                "run_stop_reason": invocation.get("run_stop_reason"),
            }
            reports.append(report)
            print(
                f"[suite {index}/{len(model_rows)}] {'DONE' if complete else 'PAUSED'} {model}: "
                f"n={report['n']}/{report['target_n']} macro={report['provisional_macro_accuracy']} "
                f"invalid={report['invalid_json_rate']} "
                f"generation_fail={report['generation_failure_rate']} ${report['reported_cost_usd']:.4f}",
                flush=True,
            )
        except Exception as error:
            failures.append({"model": model, "error": repr(error)})
            print(f"[suite {index}/{len(model_rows)}] FAILED {model}: {error!r}", flush=True)
    reports.sort(key=lambda row: (
        not bool(row["complete"]),
        -float(row["provisional_macro_accuracy"]),
        float(row["reported_cost_usd"]),
    ))
    fieldnames = [
        "model", "family", "tier", "open_weights", "n", "target_n", "complete",
        "macro_accuracy", "provisional_macro_accuracy", "text_answer_macro_accuracy",
        "micro_accuracy", "text_answer_micro_accuracy", "artifact_target_rate",
        "invalid_json_rate", "generation_failure_rate", "mean_latency_seconds",
        "reported_cost_usd", "run_stop_reason",
    ]
    with (output / "model_suite_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(reports)
    result = {"preflight": preflight, "models": reports, "failures": failures}
    atomic_json(output / "model_suite_summary.json", result)
    print(f"[suite] finished: {len(reports)}/{len(model_rows)} model reports", flush=True)
    return result


def run_rag_suite(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[rag-suite] output: {output}", flush=True)
    requested = [value.strip() for value in args.conditions.split(",") if value.strip()]
    allowed = {"base_rag", "agentic_rag"}
    if not requested or set(requested) - allowed:
        raise ValueError(f"--conditions must be a comma-separated subset of {sorted(allowed)}")
    model_row = _model_row(Path(args.models), args.model)
    agent_row = _model_row(Path(args.models), args.agent_model)
    catalog_problems = _catalog_check([model_row, agent_row])
    if catalog_problems:
        raise ValueError(f"RAG model catalog validation failed: {catalog_problems}")
    max_tokens = int(model_row.get("max_tokens", 16384))
    reasoning_effort = model_row.get("reasoning_effort")
    reasoning_enabled = bool(model_row.get("reasoning_enabled", False))
    temperature = model_row.get("temperature", 0.0)
    preflight = _preflight(args)
    args.benchmark_content_hash = preflight["portable_benchmark_hash"]
    local_corpus = stage_corpus(Path(args.corpus_root), Path(args.work_root) / "corpus")
    reports: dict[str, Any] = {}
    for condition in requested:
        run_output = output / condition
        namespace = _run_namespace(
            args, output=run_output, model=args.model, condition=condition,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            reasoning_enabled=reasoning_enabled, temperature=temperature,
        )
        retriever = None
        if condition == "base_rag":
            retriever = DenseRAGRetriever(
                local_corpus, candidate_k=args.candidate_k, rerank=not args.no_rerank,
                max_passage_chars=args.max_passage_chars,
                max_context_chars=args.max_context_chars,
            )
        elif condition == "agentic_rag":
            retriever = AgenticRAGRetriever(
                local_corpus,
                candidate_k=args.candidate_k,
                rerank=not args.no_rerank,
                max_passage_chars=args.max_passage_chars,
                max_context_chars=args.max_context_chars,
                agent_model=args.agent_model,
                agent_cache=Path(args.agent_cache).expanduser(),
                agent_max_tokens=args.agent_max_tokens,
                agent_reasoning_effort=args.agent_reasoning_effort,
                max_subqueries=args.agent_subqueries,
                max_hops=args.agent_max_hops,
            )
        print(f"[rag-suite] START {condition}", flush=True)
        invocation = run(namespace, retriever=retriever)
        reports[condition] = {
            "run": invocation,
            "analysis": analyze(run_output / "responses.jsonl", run_output / "analysis"),
        }
        print(f"[rag-suite] DONE {condition}", flush=True)
        del retriever
    comparisons: dict[str, Any] = {}
    if {"base_rag", "agentic_rag"}.issubset(requested) and all(
        bool(reports[item]["run"].get("complete")) for item in ("base_rag", "agentic_rag")
    ):
        try:
            comparisons["base_rag_to_agentic_rag"] = compare(
                output / "base_rag" / "responses.jsonl",
                output / "agentic_rag" / "responses.jsonl",
                output / "comparisons" / "base_rag_to_agentic_rag",
            )
        except ValueError as error:
            comparisons["base_rag_to_agentic_rag"] = {
                "protocol_validation": "deferred",
                "reason": str(error),
            }
            print(f"[rag-suite] comparison deferred: {error}", flush=True)
    elif {"base_rag", "agentic_rag"}.issubset(requested):
        comparisons["base_rag_to_agentic_rag"] = {
            "protocol_validation": "deferred",
            "reason": "Both conditions must reach their exact target record count; rerun the same cell to resume.",
        }
    result = {
        "preflight": preflight,
        "model_config": model_row,
        "reports": reports,
        "comparisons": comparisons,
    }
    atomic_json(output / "rag_suite_summary.json", result)
    return result


def add_experiment_parsers(sub: argparse._SubParsersAction[Any]) -> None:
    suite = sub.add_parser("suite", help="Run the publication model suite with live logs.")
    suite.add_argument("--benchmark-root", required=True)
    suite.add_argument("--models", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--preflight-cache")
    suite.add_argument("--force-preflight", action="store_true")
    suite.add_argument("--skip-model-preflight", action="store_true")
    suite.add_argument("--per-leaf-limit", type=int)
    suite.add_argument("--max-tokens", type=int, default=16384)
    suite.add_argument("--max-cost-usd-per-model", type=float, default=50.0)
    suite.add_argument("--progress-every", type=int, default=5)
    suite.add_argument("--timeout-seconds", type=int, default=240)
    suite.add_argument("--retries", type=int, default=6)
    suite.add_argument("--request-delay-seconds", type=float, default=1.0)
    suite.add_argument("--retry-base-seconds", type=float, default=5.0)
    suite.add_argument("--retry-max-seconds", type=float, default=60.0)
    suite.add_argument("--max-consecutive-errors", type=int, default=2)

    rag = sub.add_parser("rag-suite", help="Run dense base_rag and agentic_rag independently; comparisons are protocol-locked later.")
    rag.add_argument("--benchmark-root", required=True)
    rag.add_argument("--corpus-root", required=True)
    rag.add_argument("--work-root", required=True)
    rag.add_argument("--output", required=True)
    rag.add_argument("--models", required=True, help="The exact model matrix used by the Evaluation notebook")
    rag.add_argument("--preflight-cache")
    rag.add_argument("--force-preflight", action="store_true")
    rag.add_argument("--model", required=True)
    rag.add_argument("--conditions", default="base_rag,agentic_rag")
    rag.add_argument("--per-leaf-limit", type=int)
    rag.add_argument("--max-cost-usd-per-model", type=float, default=25.0)
    rag.add_argument("--progress-every", type=int, default=5)
    rag.add_argument("--timeout-seconds", type=int, default=240)
    rag.add_argument("--retries", type=int, default=6)
    rag.add_argument("--request-delay-seconds", type=float, default=1.0)
    rag.add_argument("--retry-base-seconds", type=float, default=5.0)
    rag.add_argument("--retry-max-seconds", type=float, default=60.0)
    rag.add_argument("--max-consecutive-errors", type=int, default=2)
    rag.add_argument("--top-k", type=int, default=5)
    rag.add_argument("--candidate-k", type=int, default=40)
    rag.add_argument("--max-passage-chars", type=int, default=1500)
    rag.add_argument("--max-context-chars", type=int, default=6000)
    rag.add_argument("--no-rerank", action="store_true")
    rag.add_argument("--agent-model", default="google/gemini-3.5-flash-lite")
    rag.add_argument("--agent-cache", required=True)
    rag.add_argument("--agent-max-tokens", type=int, default=2048)
    rag.add_argument("--agent-reasoning-effort", default="minimal")
    rag.add_argument("--agent-subqueries", type=int, default=3)
    rag.add_argument("--agent-max-hops", type=int, default=2)
