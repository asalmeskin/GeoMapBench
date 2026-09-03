from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from .benchmark import canonical_benchmark_records, stable_subset
from .common import append_jsonl, atomic_json, atomic_jsonl, digest, read_jsonl, stable_json, utc_now
from .openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterHTTPError,
    OpenRouterRetryExhausted,
    finish_reason,
    generation_failure,
    response_text,
)
from .prompts import build_messages
from .protocol import protocol_descriptor
from .scoring import is_artifact_target
from .task_metrics import evaluate_task_aware


class Retriever(Protocol):
    last_trace: dict[str, Any]
    last_usage: dict[str, Any]

    def search(
        self,
        query: str,
        leaf: str,
        top_k: int,
        *,
        record: dict[str, Any],
        task_dir: Path,
    ) -> list[dict[str, Any]]: ...


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id")): row
        for row in _rows(path)
        if row.get("status") == "ok"
    }


def completed_ids(path: Path) -> set[str]:
    return set(completed_rows(path))


def _cached_api_responses(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("cache_key")): row
        for row in _rows(path)
        if row.get("cache_key") and isinstance(row.get("response"), dict)
    }


def _query(record: dict[str, Any]) -> str:
    inp = record.get("input") or {}
    preferred = inp.get("question") or inp.get("base_question") or inp.get("text")
    return str(preferred or stable_json(inp))


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)


def _row_cost(row: dict[str, Any]) -> float:
    return float((row.get("usage") or {}).get("cost") or 0.0) + float(
        (row.get("retrieval_usage") or {}).get("cost") or 0.0
    )


def experiment_identity(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    records = canonical_benchmark_records(benchmark_root, prefer_clean=not args.no_clean)
    record_ids_file = getattr(args, "record_ids_file", None)
    if record_ids_file:
        payload = json.loads(Path(record_ids_file).expanduser().read_text(encoding="utf-8"))
        requested = payload.get("selected_ids") if isinstance(payload, dict) else payload
        if not isinstance(requested, list) or not requested:
            raise ValueError(f"record ID file has no selected_ids: {record_ids_file}")
        requested_ids = [str(value) for value in requested]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError(f"record ID file contains duplicates: {record_ids_file}")
        by_id = {str(record.get("id")): (directory, record) for directory, record in records}
        missing = [record_id for record_id in requested_ids if record_id not in by_id]
        if missing:
            raise ValueError(f"record ID file contains unknown IDs: {missing[:10]}")
        selected = [by_id[record_id] for record_id in requested_ids]
    else:
        selected = stable_subset(
            records,
            per_leaf_limit=getattr(args, "per_leaf_limit", None),
            limit=getattr(args, "limit", None),
        )
    ids = [str(record.get("id")) for _, record in selected]
    return {
        "benchmark_root": str(benchmark_root),
        "target_record_count": len(ids),
        "selected_ids_hash": digest(ids),
        "selected_records_hash": digest([record for _, record in selected]),
        "selected_ids": ids,
        "records": selected,
    }


def _state(
    path: Path,
    *,
    model: str,
    condition: str,
    current_id: str | None,
    stage: str,
    target: int,
    completed: int,
    attempted_this_invocation: int,
    succeeded_this_invocation: int,
    failed_this_invocation: int,
    cost_total: float,
    stop_reason: str | None = None,
) -> None:
    atomic_json(path, {
        "updated_at": utc_now(),
        "model": model,
        "condition": condition,
        "current_id": current_id,
        "stage": stage,
        "target_records": target,
        "completed_records": completed,
        "remaining_records": max(0, target - completed),
        "attempted_this_invocation": attempted_this_invocation,
        "succeeded_this_invocation": succeeded_this_invocation,
        "failed_this_invocation": failed_this_invocation,
        "reported_cost_usd_total": round(cost_total, 8),
        "stop_reason": stop_reason,
    })


def run(args: argparse.Namespace, *, retriever: Retriever | None = None) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "responses.jsonl"
    api_cache_path = output_root / "api_responses.jsonl"
    trace_path = output_root / "retrieval_trace.jsonl"
    state_path = output_root / "run_state.json"
    inflight_path = output_root / "inflight.json"
    condition = str(args.condition)
    if condition in {
        "base_rag", "agentic_rag", "multimodal_rag", "agentic_multimodal_rag",
    } and retriever is None:
        raise ValueError(f"condition={condition} requires a retriever")

    temperature = getattr(args, "temperature", 0.0)
    reasoning_effort = getattr(args, "reasoning_effort", None)
    reasoning_enabled = getattr(args, "reasoning_enabled", False)
    config = OpenRouterConfig(
        model=args.model,
        temperature=temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=getattr(args, "timeout_seconds", 240),
        retries=getattr(args, "retries", 6),
        reasoning_effort=reasoning_effort,
        reasoning_enabled=reasoning_enabled,
        request_delay_seconds=getattr(args, "request_delay_seconds", 0.0),
        retry_base_seconds=getattr(args, "retry_base_seconds", 5.0),
        retry_max_seconds=getattr(args, "retry_max_seconds", 60.0),
    )
    identity = experiment_identity(args)
    run_config = {
        "format": "GeoMapBench OpenRouter final task-aware evaluation v2.2",
        "created_at": utc_now(),
        "model": args.model,
        "condition": condition,
        "temperature": temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": reasoning_effort,
        "reasoning_enabled": reasoning_enabled,
        "benchmark_root": str(benchmark_root),
        "top_k": args.top_k,
        "include_images": not args.no_images,
        "per_leaf_limit": getattr(args, "per_leaf_limit", None),
        "limit": getattr(args, "limit", None),
        "record_ids_file": str(getattr(args, "record_ids_file", None) or ""),
        "cumulative": bool(getattr(args, "cumulative", False)),
        "target_record_count": identity["target_record_count"],
        "selected_ids_hash": identity["selected_ids_hash"],
        "selected_records_hash": identity["selected_records_hash"],
        "selected_ids": identity["selected_ids"],
        "protocol": protocol_descriptor(),
        "benchmark_content_hash": getattr(args, "benchmark_content_hash", None),
        "retrieval_config": getattr(args, "retrieval_config", None),
    }
    config_path = output_root / "run_config.json"
    if config_path.exists() and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if run_config["cumulative"] and previous.get("cumulative"):
            old_ids = set(previous.get("selected_ids") or [])
            if not old_ids.issubset(identity["selected_ids"]):
                raise ValueError("Cumulative output cannot shrink or replace previously selected IDs")
            comparable = {
                "model", "condition", "temperature", "max_tokens",
                "reasoning_effort", "reasoning_enabled", "top_k",
                "include_images", "protocol", "benchmark_content_hash", "cumulative",
                "retrieval_config",
            }
        else:
            comparable = set(run_config) - {"created_at"}
        changed = [key for key in comparable if previous.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(
                f"Output directory has a different run configuration ({', '.join(changed)}); "
                "choose a new --output directory."
            )
    atomic_json(config_path, run_config)

    experiment = identity["records"]
    prior_rows = _rows(result_path)
    target_ids = set(identity["selected_ids"])
    done_rows = {} if args.force else {
        record_id: row for record_id, row in completed_rows(result_path).items()
        if record_id in target_ids
    }
    selected = [item for item in experiment if str(item[1].get("id")) not in done_rows]
    api_cache = _cached_api_responses(api_cache_path)
    existing_answer_cost = sum(
        float(((row.get("response") or {}).get("usage") or {}).get("cost") or 0.0)
        for row in api_cache.values()
    )
    existing_retrieval_cost = sum(
        float((row.get("retrieval_usage") or {}).get("cost") or 0.0)
        for row in api_cache.values()
    ) + sum(
        float((row.get("retrieval_usage") or {}).get("cost") or 0.0)
        for row in prior_rows if row.get("status") == "error"
    )
    existing_cost = existing_answer_cost + existing_retrieval_cost
    print(
        f"[run] model={args.model} condition={condition} | target={len(experiment)} "
        f"completed={len(done_rows)} remaining={len(selected)} | cached_api={len(api_cache)} "
        f"| cost_total=${existing_cost:.4f} | output={output_root}",
        flush=True,
    )
    if not selected:
        summary = {
            "attempted_this_invocation": 0,
            "target_records": len(experiment),
            "completed_total": len(done_rows),
            "remaining_total": 0,
            "succeeded": 0,
            "failed": 0,
            "reported_cost_usd_this_invocation": 0.0,
            "answer_cost_usd_total": round(existing_answer_cost, 8),
            "agent_cost_usd_total": round(existing_retrieval_cost, 8),
            "reported_cost_usd_total": round(existing_cost, 8),
            "run_stop_reason": "already_complete",
            "complete": True,
            "results": str(result_path),
        }
        atomic_json(output_root / "run_summary.json", summary)
        _state(
            state_path, model=args.model, condition=condition, current_id=None,
            stage="complete", target=len(experiment), completed=len(done_rows),
            attempted_this_invocation=0, succeeded_this_invocation=0,
            failed_this_invocation=0, cost_total=existing_cost,
            stop_reason="already_complete",
        )
        return summary

    client = OpenRouterClient()
    answer_cost = 0.0
    retrieval_cost = 0.0
    succeeded = failed = invalid = generation_failures = 0
    api_cache_hits = 0
    stop_reason: str | None = None
    consecutive_transport_errors = 0
    max_consecutive = max(1, int(getattr(args, "max_consecutive_errors", 2)))
    attempted = 0

    for index, (task_dir, record) in enumerate(selected, 1):
        total_spent = existing_cost + answer_cost + retrieval_cost
        if args.max_cost_usd is not None and total_spent >= args.max_cost_usd:
            stop_reason = "max_cost_reached"
            print(
                f"[run:stop] total cost ${total_spent:.4f} reached cap ${args.max_cost_usd:.4f}",
                flush=True,
            )
            break

        record_id = str(record.get("id"))
        attempted += 1
        contexts: list[dict[str, Any]] | None = None
        retrieval_usage: dict[str, Any] = {"cost": 0.0, "calls": 0}
        messages: list[dict[str, Any]] | None = None
        prompt_hash: str | None = None
        try:
            _state(
                state_path, model=args.model, condition=condition, current_id=record_id,
                stage="retrieval" if retriever else "building_prompt", target=len(experiment),
                completed=len(done_rows) + succeeded, attempted_this_invocation=attempted,
                succeeded_this_invocation=succeeded, failed_this_invocation=failed,
                cost_total=total_spent,
            )
            if retriever is not None:
                contexts = retriever.search(
                    _query(record),
                    str(record.get("leaf", "")),
                    args.top_k,
                    record=record,
                    task_dir=task_dir,
                )
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
            prompt_hash = digest(messages)
            cache_key = digest({
                "id": record_id,
                "model": args.model,
                "condition": condition,
                "prompt_hash": prompt_hash,
                "max_tokens": args.max_tokens,
                "reasoning_enabled": reasoning_enabled,
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
            })
            atomic_json(inflight_path, {
                "updated_at": utc_now(), "stage": "requesting", "id": record_id,
                "index": index, "remaining_this_invocation": len(selected) - index + 1,
                "model": args.model, "condition": condition, "prompt_hash": prompt_hash,
                "cache_key": cache_key,
            })

            cached = None if args.force else api_cache.get(cache_key)
            if cached:
                response = dict(cached["response"])
                retrieval_usage = dict(cached.get("retrieval_usage") or retrieval_usage)
                api_cache_hits += 1
                print(f"[run:cache-hit] {record_id}: recovered saved API response", flush=True)
            else:
                print(
                    f"[run:item {index}/{len(selected)}] REQUEST {record_id}",
                    flush=True,
                )
                response = client.complete(messages, config)
                cache_row = {
                    "cache_key": cache_key,
                    "id": record_id,
                    "model": args.model,
                    "condition": condition,
                    "prompt_hash": prompt_hash,
                    "cached_at": utc_now(),
                    "retrieval_usage": retrieval_usage,
                    "response": response,
                }
                append_jsonl(api_cache_path, cache_row)
                api_cache[cache_key] = cache_row
                atomic_json(inflight_path, {
                    "updated_at": utc_now(), "stage": "response_cached", "id": record_id,
                    "model": args.model, "condition": condition, "cache_key": cache_key,
                })

            raw_text = response_text(response)
            failure_kind = generation_failure(response, raw_text)
            evaluation = evaluate_task_aware(record, raw_text, task_dir)
            usage = response.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            if not cached:
                answer_cost += cost
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
                "prompt_hash": prompt_hash,
                "retrieved_ids": [item.get("id") for item in contexts or []],
                "response": raw_text,
                "finish_reason": finish_reason(response),
                "generation_failure": failure_kind,
                "artifact_target": is_artifact_target(record),
                "reasoning_tokens": _reasoning_tokens(usage),
                "usage": usage,
                "retrieval_usage": retrieval_usage,
                "latency_seconds": response.get("_latency_seconds"),
                "transport_attempts": response.get("_transport_attempts", 1),
                "api_response_cache_hit": bool(cached),
                **evaluation,
            })
            succeeded += 1
            consecutive_transport_errors = 0
        except Exception as error:
            failed += 1
            status = getattr(error, "status", None)
            retryable_transport = (
                isinstance(error, OpenRouterRetryExhausted)
                and (status is None or status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500))
            ) or (
                isinstance(error, OpenRouterHTTPError)
                and (status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500))
            )
            append_jsonl(result_path, {
                "id": record_id,
                "leaf": record.get("leaf"),
                "status": "error",
                "condition": condition,
                "model": args.model,
                "completed_at": utc_now(),
                "prompt_hash": prompt_hash,
                "retrieval_usage": retrieval_usage,
                "retryable_transport": retryable_transport,
                "http_status": status,
                "error": repr(error),
            })
            print(f"[run:error] {record_id}: {error!r}", flush=True)
            consecutive_transport_errors = consecutive_transport_errors + 1 if retryable_transport else 0
            permanent_http = isinstance(error, OpenRouterHTTPError) and isinstance(status, int) and 400 <= status < 500 and not retryable_transport
            if permanent_http or status == 429 or consecutive_transport_errors >= max_consecutive:
                stop_reason = "permanent_http_error" if permanent_http else "transport_circuit_breaker"
                print(
                    "[run:stop] transport circuit breaker opened; rerun the same cell later "
                    "to resume without repeating completed records",
                    flush=True,
                )
        total_spent = existing_cost + answer_cost + retrieval_cost
        completed_now = len(done_rows) + succeeded
        _state(
            state_path, model=args.model, condition=condition, current_id=record_id,
            stage="paused" if stop_reason else "running", target=len(experiment),
            completed=completed_now, attempted_this_invocation=attempted,
            succeeded_this_invocation=succeeded, failed_this_invocation=failed,
            cost_total=total_spent, stop_reason=stop_reason,
        )
        if index % args.progress_every == 0 or index == len(selected) or stop_reason:
            print(
                f"[run] {attempted}/{len(selected)} attempted this invocation | {succeeded} ok | "
                f"{failed} errors | {invalid} invalid JSON | {generation_failures} generation "
                f"failures | cache_hits={api_cache_hits} | completed_total={completed_now}/{len(experiment)} "
                f"| ${total_spent:.4f}",
                flush=True,
            )
        if stop_reason:
            break

    final_completed = {
        record_id: row for record_id, row in completed_rows(result_path).items()
        if record_id in target_ids
    }
    # Keep append+fsync safety during inference, then leave one canonical final
    # row per record instead of accumulating generations of duplicate rows.
    compacted: dict[str, dict[str, Any]] = {}
    for row in _rows(result_path):
        compacted[str(row.get("id"))] = row
    atomic_jsonl(result_path, [compacted[key] for key in sorted(compacted)])
    cache_compacted = _cached_api_responses(api_cache_path)
    atomic_jsonl(api_cache_path, [cache_compacted[key] for key in sorted(cache_compacted)])
    complete = len(final_completed) == len(experiment)
    if stop_reason is None:
        stop_reason = "completed" if complete else "incomplete_errors"
    total_spent = existing_cost + answer_cost + retrieval_cost
    summary = {
        "attempted_this_invocation": attempted,
        "target_records": len(experiment),
        "completed_total": len(final_completed),
        "remaining_total": len(experiment) - len(final_completed),
        "succeeded": succeeded,
        "failed": failed,
        "invalid_json": invalid,
        "generation_failures": generation_failures,
        "invalid_json_this_invocation": invalid,
        "generation_failures_this_invocation": generation_failures,
        "api_response_cache_hits": api_cache_hits,
        "answer_cost_usd_this_invocation": round(answer_cost, 8),
        "retrieval_cost_usd_this_invocation": round(retrieval_cost, 8),
        "reported_cost_usd_this_invocation": round(answer_cost + retrieval_cost, 8),
        "answer_cost_usd_total": round(existing_answer_cost + answer_cost, 8),
        "agent_cost_usd_total": round(existing_retrieval_cost + retrieval_cost, 8),
        "reported_cost_usd_total": round(total_spent, 8),
        "run_stop_reason": stop_reason,
        "complete": complete,
        "results": str(result_path),
        "live_state": str(state_path),
        "inflight_state": str(inflight_path),
        "api_response_cache": str(api_cache_path),
    }
    atomic_json(output_root / "run_summary.json", summary)
    _state(
        state_path, model=args.model, condition=condition, current_id=None,
        stage="complete" if complete else "paused", target=len(experiment),
        completed=len(final_completed), attempted_this_invocation=attempted,
        succeeded_this_invocation=succeeded, failed_this_invocation=failed,
        cost_total=total_spent, stop_reason=stop_reason,
    )
    return summary


def add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("run", help="Run one resumable OpenRouter benchmark condition.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--condition",
        choices=(
            "base", "base_rag", "agentic_rag",
            "multimodal_rag", "agentic_multimodal_rag",
        ),
        default="base",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--reasoning-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=2)
    parser.add_argument("--max-image-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-leaf-limit", type=int)
    parser.add_argument("--record-ids-file")
    parser.add_argument("--cumulative", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--force", action="store_true")


def validate_run_args(args: argparse.Namespace) -> None:
    if args.condition != "base":
        raise ValueError("Use `rag-suite` for RAG conditions so multimodal retrieval is validated and initialized safely")
    if args.per_leaf_limit is not None and args.per_leaf_limit < 1:
        raise ValueError("--per-leaf-limit must be at least 1")
    if args.max_tokens < 1024:
        raise ValueError("--max-tokens below 1024 is unsafe for this structured-output benchmark")
