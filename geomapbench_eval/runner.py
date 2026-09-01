from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from .benchmark import canonical_benchmark_records, stable_subset
from .common import append_jsonl, atomic_json, digest, read_jsonl, stable_json, utc_now
from .openrouter import (
    OpenRouterClient, OpenRouterConfig, finish_reason, generation_failure, response_text,
)
from .prompts import build_messages
from .protocol import protocol_descriptor
from .scoring import is_artifact_target, score


class Retriever(Protocol):
    last_trace: dict[str, Any]
    last_usage: dict[str, Any]

    def search(self, query: str, leaf: str, top_k: int) -> list[dict[str, Any]]: ...


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("status") == "ok"}


def _query(record: dict[str, Any]) -> str:
    inp = record.get("input") or {}
    preferred = inp.get("question") or inp.get("base_question") or inp.get("text")
    return str(preferred or stable_json(inp))


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)


def experiment_identity(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    records = canonical_benchmark_records(benchmark_root, prefer_clean=not args.no_clean)
    selected = stable_subset(records, per_leaf_limit=args.per_leaf_limit, limit=args.limit)
    ids = [str(record.get("id")) for _, record in selected]
    return {
        "benchmark_root": str(benchmark_root),
        "target_record_count": len(ids),
        "selected_ids_hash": digest(ids),
        "selected_records_hash": digest([record for _, record in selected]),
        "selected_ids": ids,
        "records": selected,
    }


def run(args: argparse.Namespace, *, retriever: Retriever | None = None) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "responses.jsonl"
    trace_path = output_root / "retrieval_trace.jsonl"
    condition = str(args.condition)
    if condition in {"base_rag", "agentic_rag"} and retriever is None:
        raise ValueError(f"condition={condition} requires a retriever")
    config = OpenRouterConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        reasoning_effort=args.reasoning_effort,
    )
    identity = experiment_identity(args)
    run_config = {
        "format": "GeoMapBench OpenRouter evaluation v4",
        "created_at": utc_now(),
        "model": args.model,
        "condition": condition,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "benchmark_root": str(benchmark_root),
        "top_k": args.top_k,
        "include_images": not args.no_images,
        "per_leaf_limit": args.per_leaf_limit,
        "limit": args.limit,
        "target_record_count": identity["target_record_count"],
        "selected_ids_hash": identity["selected_ids_hash"],
        "selected_records_hash": identity["selected_records_hash"],
        "protocol": protocol_descriptor(),
        "benchmark_content_hash": getattr(args, "benchmark_content_hash", None),
    }
    config_path = output_root / "run_config.json"
    if config_path.exists() and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = set(run_config) - {"created_at"}
        changed = [key for key in comparable if previous.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(
                f"Output directory has a different run configuration ({', '.join(changed)}); "
                "choose a new --output directory."
            )
    atomic_json(config_path, run_config)
    experiment = identity["records"]
    done = set() if args.force else completed_ids(result_path)
    selected = [item for item in experiment if str(item[1].get("id")) not in done]
    print(
        f"[run] model={args.model} condition={condition} | target={len(experiment)} "
        f"completed={len(experiment) - len(selected)} remaining={len(selected)} | output={output_root}",
        flush=True,
    )
    if not selected:
        summary = {
            "attempted_this_invocation": 0,
            "target_records": len(experiment),
            "succeeded": 0,
            "failed": 0,
            "reported_cost_usd": 0.0,
            "run_stop_reason": "already_complete",
            "results": str(result_path),
        }
        atomic_json(output_root / "run_summary.json", summary)
        return summary
    client = OpenRouterClient()
    spent = 0.0
    answer_cost = 0.0
    retrieval_cost = 0.0
    succeeded = failed = invalid = generation_failures = 0
    stop_reason = "completed"
    for index, (task_dir, record) in enumerate(selected, 1):
        if args.max_cost_usd is not None and spent >= args.max_cost_usd:
            stop_reason = "max_cost_reached"
            break
        record_id = str(record.get("id"))
        contexts: list[dict[str, Any]] | None = None
        retrieval_usage: dict[str, Any] = {"cost": 0.0, "calls": 0}
        try:
            if retriever is not None:
                contexts = retriever.search(_query(record), str(record.get("leaf", "")), args.top_k)
                retrieval_usage = dict(getattr(retriever, "last_usage", {}) or {})
                retrieval_cost += float(retrieval_usage.get("cost") or 0.0)
                if getattr(retriever, "last_trace", None):
                    trace = dict(retriever.last_trace)
                    trace.update({"id": record_id, "completed_at": utc_now()})
                    append_jsonl(trace_path, trace)
            messages = build_messages(
                record,
                task_dir,
                contexts=contexts,
                include_images=not args.no_images,
                max_image_bytes=args.max_image_bytes,
            )
            response = client.complete(messages, config)
            raw_text = response_text(response)
            failure_kind = generation_failure(response, raw_text)
            evaluation = score(record, raw_text)
            usage = response.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            answer_cost += cost
            spent = answer_cost + retrieval_cost
            if evaluation.get("parse_error"):
                invalid += 1
            if failure_kind:
                generation_failures += 1
            append_jsonl(result_path, {
                "id": record_id,
                "leaf": record.get("leaf"),
                "bloom": (record.get("bloom") or {}).get("level"),
                "bloom_variant": (record.get("bloom") or {}).get("variant"),
                "status": "ok",
                "condition": condition,
                "model": args.model,
                "completed_at": utc_now(),
                "prompt_hash": digest(messages),
                "retrieved_ids": [item.get("id") for item in contexts or []],
                "response": raw_text,
                "finish_reason": finish_reason(response),
                "generation_failure": failure_kind,
                "artifact_target": is_artifact_target(record),
                "reasoning_tokens": _reasoning_tokens(usage),
                "usage": usage,
                "retrieval_usage": retrieval_usage,
                "latency_seconds": response.get("_latency_seconds"),
                **evaluation,
            })
            succeeded += 1
        except Exception as error:
            append_jsonl(result_path, {
                "id": record_id,
                "leaf": record.get("leaf"),
                "status": "error",
                "condition": condition,
                "model": args.model,
                "completed_at": utc_now(),
                "error": repr(error),
            })
            failed += 1
            print(f"[run:error] {record_id}: {error!r}", flush=True)
        if index % args.progress_every == 0 or index == len(selected):
            print(
                f"[run] {index}/{len(selected)} attempted | {succeeded} ok | {failed} errors | "
                f"{invalid} invalid JSON | {generation_failures} generation failures | ${spent:.4f}",
                flush=True,
            )
    summary = {
        "attempted_this_invocation": succeeded + failed,
        "target_records": len(experiment),
        "succeeded": succeeded,
        "failed": failed,
        "invalid_json": invalid,
        "generation_failures": generation_failures,
        "answer_cost_usd": round(answer_cost, 8),
        "retrieval_cost_usd": round(retrieval_cost, 8),
        "reported_cost_usd": round(answer_cost + retrieval_cost, 8),
        "run_stop_reason": stop_reason,
        "results": str(result_path),
    }
    atomic_json(output_root / "run_summary.json", summary)
    return summary


def add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("run", help="Run one resumable OpenRouter benchmark condition.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=("base", "base_rag", "agentic_rag"), default="base")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="minimal")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-image-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-leaf-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--force", action="store_true")


def validate_run_args(args: argparse.Namespace) -> None:
    if args.condition != "base":
        raise ValueError("Use `rag-suite` for base_rag and agentic_rag so dense retrieval is initialized safely")
    if args.per_leaf_limit is not None and args.per_leaf_limit < 1:
        raise ValueError("--per-leaf-limit must be at least 1")
    if args.max_tokens < 1024:
        raise ValueError("--max-tokens below 1024 is unsafe for this structured-output benchmark")
