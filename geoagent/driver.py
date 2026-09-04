"""Resumable run loop for the GeoAgent condition.

Deliberately mirrors ``geomapbench_eval.runner.run``: the same output files, the
same cumulative cohort, the same two-phase API response cache, the same cost cap
and circuit breaker, and a ``run_config.json`` carrying the identical protocol
descriptor so ``geomapbench_eval.analysis.compare`` accepts a paired comparison
against the already-paid ``base`` results.

Differences: the answer stage is the GeoAgent pipeline rather than a single
call, and each row also stores ``raw_response`` (pre-repair) plus the agent
trace, which the notebook turns into free ablations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geomapbench_eval.benchmark import canonical_benchmark_records
from geomapbench_eval.common import (
    append_jsonl, atomic_json, atomic_jsonl, digest, read_jsonl, utc_now,
)
from geomapbench_eval.cumulative import write_cohort_manifest
from geomapbench_eval.openrouter import (
    OpenRouterClient, OpenRouterConfig, OpenRouterHTTPError, OpenRouterRetryExhausted,
    finish_reason, generation_failure, response_text,
)
from geomapbench_eval.protocol import protocol_descriptor

from . import AGENT_PROMPT_REVISION, AGENT_PROTOCOL_REVISION, REPAIR_REVISION, RETRIEVAL_REVISION
from .pipeline import AgenticPipeline, stage_summary


@dataclass
class RunConfig:
    benchmark_root: Path
    output: Path
    condition: str
    model: str
    max_tokens: int = 16384
    temperature: float | None = 0.0
    reasoning_effort: str | None = None
    reasoning_enabled: bool = False
    timeout_seconds: int = 240
    retries: int = 6
    request_delay_seconds: float = 1.0
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    max_consecutive_errors: int = 2
    max_image_bytes: int = 8_000_000
    max_cost_usd: float | None = None
    progress_every: int = 5
    top_k: int = 4
    record_ids_file: Path | None = None
    benchmark_content_hash: str | None = None
    retrieval_config: dict[str, Any] = field(default_factory=dict)
    max_revision_share: float = 0.25


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _completed(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in _rows(path) if row.get("status") == "ok"}


def _cached_api(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("cache_key")): row for row in _rows(path)
        if row.get("cache_key") and isinstance(row.get("response"), dict)
    }


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)


def build_cohort(
    benchmark_root: Path, output: Path, *, target_per_leaf: int, benchmark_content_hash: str,
) -> tuple[list[tuple[Path, dict[str, Any]]], Path, dict[str, Any]]:
    records = canonical_benchmark_records(benchmark_root, prefer_clean=True)
    cohort_path, cohort = write_cohort_manifest(
        records, target_per_leaf=target_per_leaf, output_root=output,
        benchmark_content_hash=benchmark_content_hash,
    )
    return records, cohort_path, cohort


def selected_records(
    records: list[tuple[Path, dict[str, Any]]], cohort_path: Path,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    payload = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
    wanted = [str(value) for value in payload["selected_ids"]]
    by_id = {str(record.get("id")): (directory, record) for directory, record in records}
    missing = [record_id for record_id in wanted if record_id not in by_id]
    if missing:
        raise ValueError(f"cohort references unknown IDs: {missing[:5]}")
    return [by_id[record_id] for record_id in wanted], wanted


def run_agentic(
    config: RunConfig,
    *,
    records: list[tuple[Path, dict[str, Any]]],
    cohort_path: Path,
    pipeline: AgenticPipeline,
    agent: Any = None,
) -> dict[str, Any]:
    output = Path(config.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "responses.jsonl"
    api_cache_path = output / "api_responses.jsonl"
    trace_path = output / "retrieval_trace.jsonl"
    agent_trace_path = output / "agent_trace.jsonl"
    state_path = output / "run_state.json"
    inflight_path = output / "inflight.json"

    experiment, wanted_ids = selected_records(records, cohort_path)
    identity_ids = [str(record.get("id")) for _, record in experiment]

    api_config = OpenRouterConfig(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
        reasoning_effort=config.reasoning_effort,
        reasoning_enabled=config.reasoning_enabled,
        request_delay_seconds=config.request_delay_seconds,
        retry_base_seconds=config.retry_base_seconds,
        retry_max_seconds=config.retry_max_seconds,
    )
    run_config = {
        "format": "GeoMapBench OpenRouter final task-aware evaluation v2.2",
        "created_at": utc_now(),
        "model": config.model,
        "condition": config.condition,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "reasoning_enabled": config.reasoning_enabled,
        "benchmark_root": str(Path(config.benchmark_root).expanduser().resolve()),
        "top_k": config.top_k,
        "include_images": True,
        "per_leaf_limit": None,
        "limit": None,
        "record_ids_file": str(cohort_path),
        "cumulative": True,
        "target_record_count": len(identity_ids),
        "selected_ids_hash": digest(identity_ids),
        "selected_records_hash": digest([record for _, record in experiment]),
        "selected_ids": identity_ids,
        "protocol": protocol_descriptor(),
        "benchmark_content_hash": config.benchmark_content_hash,
        "retrieval_config": dict(config.retrieval_config),
        "geoagent": {
            "prompt_revision": AGENT_PROMPT_REVISION,
            "agent_protocol_revision": AGENT_PROTOCOL_REVISION,
            "retrieval_revision": RETRIEVAL_REVISION,
            "repair_revision": REPAIR_REVISION,
        },
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        old_ids = set(previous.get("selected_ids") or [])
        if not old_ids.issubset(identity_ids):
            raise ValueError("Cumulative output cannot shrink or replace previously selected IDs")
        comparable = (
            "model", "condition", "temperature", "max_tokens", "reasoning_effort",
            "reasoning_enabled", "top_k", "include_images", "protocol",
            "benchmark_content_hash", "cumulative", "retrieval_config", "geoagent",
        )
        changed = [key for key in comparable if previous.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(
                f"Output directory has a different run configuration ({', '.join(changed)}); "
                "choose a new output directory or restore the previous settings."
            )
    atomic_json(config_path, run_config)

    done_rows = {
        record_id: row for record_id, row in _completed(result_path).items()
        if record_id in set(identity_ids)
    }
    pending = [item for item in experiment if str(item[1].get("id")) not in done_rows]
    api_cache = _cached_api(api_cache_path)
    existing_answer_cost = sum(
        float(((row.get("response") or {}).get("usage") or {}).get("cost") or 0.0)
        for row in api_cache.values()
    )
    existing_agent_cost = sum(
        float((row.get("retrieval_usage") or {}).get("cost") or 0.0)
        for row in _rows(result_path) if row.get("status") == "ok"
    )
    existing_cost = existing_answer_cost + existing_agent_cost
    print(
        f"[geoagent] model={config.model} condition={config.condition} | "
        f"target={len(experiment)} completed={len(done_rows)} remaining={len(pending)} | "
        f"cached_api={len(api_cache)} | prior_cost=${existing_cost:.4f} | output={output}",
        flush=True,
    )

    client = OpenRouterClient()
    answer_cost = agent_cost = 0.0
    succeeded = failed = invalid = generation_failures = cache_hits = revisions = 0
    stop_reason: str | None = None
    consecutive_transport_errors = 0
    attempted = 0

    def state(current_id: str | None, stage: str, spent: float) -> None:
        atomic_json(state_path, {
            "updated_at": utc_now(), "model": config.model, "condition": config.condition,
            "current_id": current_id, "stage": stage,
            "target_records": len(experiment),
            "completed_records": len(done_rows) + succeeded,
            "remaining_records": max(0, len(experiment) - len(done_rows) - succeeded),
            "attempted_this_invocation": attempted,
            "succeeded_this_invocation": succeeded,
            "failed_this_invocation": failed,
            "revisions_this_invocation": revisions,
            "reported_cost_usd_total": round(spent, 8),
            "stop_reason": stop_reason,
        })

    if not pending:
        summary = {
            "attempted_this_invocation": 0, "target_records": len(experiment),
            "completed_total": len(done_rows), "remaining_total": 0,
            "succeeded": 0, "failed": 0,
            "reported_cost_usd_this_invocation": 0.0,
            "answer_cost_usd_total": round(existing_answer_cost, 8),
            "agent_cost_usd_total": round(existing_agent_cost, 8),
            "reported_cost_usd_total": round(existing_cost, 8),
            "run_stop_reason": "already_complete", "complete": True,
            "results": str(result_path),
        }
        atomic_json(output / "run_summary.json", summary)
        state(None, "complete", existing_cost)
        return summary

    for index, (task_dir, record) in enumerate(pending, 1):
        spent = existing_cost + answer_cost + agent_cost
        if config.max_cost_usd is not None and spent >= config.max_cost_usd:
            stop_reason = "max_cost_reached"
            print(f"[geoagent:stop] ${spent:.4f} reached the cap ${config.max_cost_usd:.4f}", flush=True)
            break
        record_id = str(record.get("id"))
        attempted += 1
        call_log: list[dict[str, Any]] = []
        if agent is not None:
            agent.reset_usage()

        def answer_fn(messages: list[dict[str, Any]], tag: str) -> tuple[str, dict[str, Any]]:
            nonlocal answer_cost, cache_hits
            prompt_hash = digest(messages)
            cache_key = digest({
                "id": record_id, "model": config.model, "condition": config.condition,
                "prompt_hash": prompt_hash, "max_tokens": config.max_tokens,
                "reasoning_enabled": config.reasoning_enabled,
                "reasoning_effort": config.reasoning_effort,
                "temperature": config.temperature, "tag": tag,
            })
            budget_left = (
                config.max_cost_usd is None
                or existing_cost + answer_cost + agent_cost < config.max_cost_usd
            )
            cached = api_cache.get(cache_key)
            if cached is None and tag != "answer":
                if not budget_left:
                    return "", {"performed": False, "reason": "cost_cap"}
                # A reviewer that objects to everything must not be able to double
                # the bill; second attempts stay a minority of the run.
                if attempted >= 20 and revisions >= config.max_revision_share * attempted:
                    return "", {"performed": False, "reason": "revision_share_cap"}
            if cached is not None:
                response = dict(cached["response"])
                cache_hits += 1
                print(f"[geoagent:cache-hit] {record_id} ({tag})", flush=True)
            else:
                atomic_json(inflight_path, {
                    "updated_at": utc_now(), "stage": f"requesting:{tag}", "id": record_id,
                    "index": index, "model": config.model, "condition": config.condition,
                    "prompt_hash": prompt_hash, "cache_key": cache_key,
                })
                response = client.complete(messages, api_config)
                row = {
                    "cache_key": cache_key, "id": record_id, "model": config.model,
                    "condition": config.condition, "prompt_hash": prompt_hash,
                    "tag": tag, "cached_at": utc_now(), "response": response,
                }
                append_jsonl(api_cache_path, row)
                api_cache[cache_key] = row
                answer_cost += float((response.get("usage") or {}).get("cost") or 0.0)
            text = response_text(response)
            call_log.append({
                "tag": tag, "prompt_hash": prompt_hash, "cache_key": cache_key,
                "cached": cached is not None,
                "usage": response.get("usage") or {},
                "finish_reason": finish_reason(response),
                "latency_seconds": response.get("_latency_seconds"),
                "generation_failure": generation_failure(response, text),
                "response": response,
            })
            return text, {"performed": True}

        try:
            state(record_id, "agent", spent)
            print(f"[geoagent:item {index}/{len(pending)}] START {record_id} ({record.get('leaf')})", flush=True)
            started = time.monotonic()
            outcome = pipeline.solve(record, task_dir, answer_fn=answer_fn)
            wall_seconds = round(time.monotonic() - started, 3)
            primary = next((item for item in call_log if item["tag"] == "answer"), None)
            final = call_log[-1] if call_log else None
            if primary is None:
                raise RuntimeError("no answer call was made")
            agent_usage = dict(agent.usage) if agent is not None else {"cost": 0.0, "calls": 0}
            agent_cost += float(agent_usage.get("cost") or 0.0)
            revision_calls = sum(1 for item in call_log if item["tag"] != "answer")
            revisions += revision_calls
            retrieval_usage = {
                "cost": round(float(agent_usage.get("cost") or 0.0), 8),
                "calls": int(agent_usage.get("calls") or 0),
                "cached_calls": int(agent_usage.get("cached_calls") or 0),
                "agent_failures": int(agent_usage.get("failures") or 0),
                "revision_calls": revision_calls,
            }
            usage = {
                "cost": round(sum(float((item["usage"] or {}).get("cost") or 0.0) for item in call_log), 8),
                "prompt_tokens": sum(int((item["usage"] or {}).get("prompt_tokens") or 0) for item in call_log),
                "completion_tokens": sum(int((item["usage"] or {}).get("completion_tokens") or 0) for item in call_log),
                "answer_calls": len(call_log),
                "completion_tokens_details": (final["usage"] or {}).get("completion_tokens_details") if final else None,
            }
            failure_kind = final.get("generation_failure") if final else None
            if outcome.parse_error:
                invalid += 1
            if failure_kind:
                generation_failures += 1
            if outcome.retrieval_trace:
                append_jsonl(trace_path, {
                    **outcome.retrieval_trace, "id": record_id, "completed_at": utc_now(),
                })
            append_jsonl(agent_trace_path, {
                "id": record_id, "leaf": record.get("leaf"),
                "completed_at": utc_now(),
                "tools": outcome.tool_trace,
                "stages": stage_summary(outcome),
                "answer_calls": [
                    {key: item[key] for key in ("tag", "cached", "finish_reason", "latency_seconds")}
                    for item in call_log
                ],
            })
            append_jsonl(result_path, {
                "id": record_id,
                "leaf": record.get("leaf"),
                "bloom": (record.get("bloom") or {}).get("level"),
                "bloom_variant": (record.get("bloom") or {}).get("variant"),
                "status": "ok",
                "condition": config.condition,
                "model": config.model,
                "completed_at": utc_now(),
                "prompt_hash": primary["prompt_hash"],
                "retrieved_ids": [item.get("id") for item in outcome.contexts],
                "response": outcome.text,
                "raw_response": outcome.raw_text,
                "repairs": outcome.repairs,
                "revision_used": outcome.revision_used,
                "agent_stages": stage_summary(outcome),
                "finish_reason": final.get("finish_reason") if final else None,
                "generation_failure": failure_kind,
                "reasoning_tokens": _reasoning_tokens(usage),
                "usage": usage,
                "retrieval_usage": retrieval_usage,
                "latency_seconds": round(
                    sum(float(item.get("latency_seconds") or 0.0) for item in call_log), 3
                ) or (final.get("latency_seconds") if final else None),
                "wall_seconds": wall_seconds,
                "api_response_cache_hit": bool(primary["cached"]),
            })
            succeeded += 1
            consecutive_transport_errors = 0
        except Exception as error:
            failed += 1
            status = getattr(error, "status", None)
            retryable = (
                isinstance(error, OpenRouterRetryExhausted)
                and (status is None or status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500))
            ) or (
                isinstance(error, OpenRouterHTTPError)
                and (status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500))
            )
            append_jsonl(result_path, {
                "id": record_id, "leaf": record.get("leaf"), "status": "error",
                "condition": config.condition, "model": config.model,
                "completed_at": utc_now(), "retryable_transport": retryable,
                "http_status": status, "error": repr(error),
            })
            print(f"[geoagent:error] {record_id}: {error!r}", flush=True)
            consecutive_transport_errors = consecutive_transport_errors + 1 if retryable else 0
            permanent = (
                isinstance(error, OpenRouterHTTPError)
                and isinstance(status, int) and 400 <= status < 500 and not retryable
            )
            if permanent or status == 429 or consecutive_transport_errors >= max(1, config.max_consecutive_errors):
                stop_reason = "permanent_http_error" if permanent else "transport_circuit_breaker"
                print(
                    "[geoagent:stop] circuit breaker opened; rerun the same cell later to "
                    "resume without repeating completed records",
                    flush=True,
                )
        spent = existing_cost + answer_cost + agent_cost
        state(record_id, "paused" if stop_reason else "running", spent)
        if index % max(1, config.progress_every) == 0 or index == len(pending) or stop_reason:
            print(
                f"[geoagent] {attempted}/{len(pending)} attempted | {succeeded} ok | {failed} errors | "
                f"{invalid} unparsable | {generation_failures} generation failures | "
                f"cache_hits={cache_hits} | revisions={revisions} | "
                f"completed_total={len(done_rows) + succeeded}/{len(experiment)} | ${spent:.4f}",
                flush=True,
            )
        if stop_reason:
            break

    compacted: dict[str, dict[str, Any]] = {}
    for row in _rows(result_path):
        compacted[str(row.get("id"))] = row
    atomic_jsonl(result_path, [compacted[key] for key in sorted(compacted)])
    cache_rows = _cached_api(api_cache_path)
    atomic_jsonl(api_cache_path, [cache_rows[key] for key in sorted(cache_rows)])

    final_completed = {
        record_id: row for record_id, row in _completed(result_path).items()
        if record_id in set(identity_ids)
    }
    complete = len(final_completed) == len(experiment)
    if stop_reason is None:
        stop_reason = "completed" if complete else "incomplete_errors"
    spent = existing_cost + answer_cost + agent_cost
    summary = {
        "attempted_this_invocation": attempted,
        "target_records": len(experiment),
        "completed_total": len(final_completed),
        "remaining_total": len(experiment) - len(final_completed),
        "succeeded": succeeded, "failed": failed,
        "invalid_json_this_invocation": invalid,
        "generation_failures_this_invocation": generation_failures,
        "api_response_cache_hits": cache_hits,
        "revision_calls_this_invocation": revisions,
        "answer_cost_usd_this_invocation": round(answer_cost, 8),
        "agent_cost_usd_this_invocation": round(agent_cost, 8),
        "reported_cost_usd_this_invocation": round(answer_cost + agent_cost, 8),
        "answer_cost_usd_total": round(existing_answer_cost + answer_cost, 8),
        "agent_cost_usd_total": round(existing_agent_cost + agent_cost, 8),
        "reported_cost_usd_total": round(spent, 8),
        "run_stop_reason": stop_reason,
        "complete": complete,
        "results": str(result_path),
        "agent_trace": str(agent_trace_path),
    }
    atomic_json(output / "run_summary.json", summary)
    state(None, "complete" if complete else "paused", spent)
    return summary
