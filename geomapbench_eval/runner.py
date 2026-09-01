from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import append_jsonl, atomic_json, digest, read_jsonl, utc_now
from .openrouter import OpenRouterClient, OpenRouterConfig, response_text
from .prompts import build_messages
from .preflight import canonical_benchmark_records
from .rag import AgenticRAGRetriever, DenseRAGRetriever
from .scoring import score


def benchmark_records(root: Path, prefer_clean: bool = True) -> list[tuple[Path, dict[str, Any]]]:
    return canonical_benchmark_records(root, prefer_clean=prefer_clean)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("status") == "ok"}


def select_records(
    records: list[tuple[Path, dict[str, Any]]], done: set[str],
    *, per_leaf_limit: int | None = None, limit: int | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    selected = records
    if per_leaf_limit is not None:
        counts: Counter[str] = Counter()
        stratified: list[tuple[Path, dict[str, Any]]] = []
        for directory, record in selected:
            leaf = str(record.get("leaf", directory.name))
            if counts[leaf] < per_leaf_limit:
                stratified.append((directory, record))
                counts[leaf] += 1
        selected = stratified
    if limit:
        selected = selected[:limit]
    return [
        (directory, record) for directory, record in selected
        if str(record.get("id")) not in done
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "responses.jsonl"
    config = OpenRouterConfig(args.model, args.temperature, args.max_tokens, args.timeout_seconds, args.retries)
    run_config = {
        "format": "GeoMapBench OpenRouter evaluation v1",
        "created_at": utc_now(), "model": args.model, "condition": args.condition,
        "temperature": args.temperature, "max_tokens": args.max_tokens, "benchmark_root": str(benchmark_root),
        "corpus_root": args.corpus_root, "top_k": args.top_k, "include_images": not args.no_images,
        "per_leaf_limit": args.per_leaf_limit, "limit": args.limit,
        "rag_backend": args.rag_backend, "candidate_k": args.candidate_k,
        "reranker_model": args.reranker_model,
        "max_passage_chars": args.max_passage_chars,
        "max_context_chars": args.max_context_chars,
        "capability_gating": not args.no_capability_gating,
        "agent_model": args.agent_model,
        "agent_max_hops": args.agent_max_hops,
        "agent_subqueries": args.agent_subqueries,
    }
    existing_config = output_root / "run_config.json"
    if existing_config.exists() and not args.force:
        previous = __import__("json").loads(existing_config.read_text(encoding="utf-8"))
        for key in ("model", "condition", "temperature", "max_tokens", "benchmark_root", "corpus_root", "top_k", "include_images", "per_leaf_limit", "limit", "rag_backend", "candidate_k", "reranker_model", "max_passage_chars", "max_context_chars", "capability_gating", "agent_model", "agent_max_hops", "agent_subqueries"):
            if previous.get(key) != run_config.get(key):
                raise ValueError("Output directory has a different run configuration; choose a new --output or use --force.")
    atomic_json(existing_config, run_config)
    retriever = None
    if args.condition in {"base_rag", "agentic_rag"}:
        retriever_class = AgenticRAGRetriever if args.condition == "agentic_rag" else DenseRAGRetriever
        retriever_kwargs: dict[str, Any] = {}
        if args.condition == "agentic_rag":
            retriever_kwargs.update({
                "agent_model": args.agent_model,
                "agent_max_hops": args.agent_max_hops,
                "agent_subqueries": args.agent_subqueries,
                "agent_cache": output_root / "agent_cache",
            })
        retriever = retriever_class(
            Path(args.corpus_root), candidate_k=args.candidate_k,
            reranker_model=None if args.reranker_model.lower() == "none" else args.reranker_model,
            max_passage_chars=args.max_passage_chars,
            max_context_chars=args.max_context_chars,
            trace_path=output_root / "retrieval_trace.jsonl",
            capability_aware=not args.no_capability_gating,
            **retriever_kwargs,
        )
    client = OpenRouterClient()
    done = set() if args.force else completed_ids(result_path)
    records = benchmark_records(benchmark_root, prefer_clean=not args.no_clean)
    selected = select_records(
        records, done, per_leaf_limit=args.per_leaf_limit, limit=args.limit,
    )
    prior_rows = read_jsonl(result_path) if result_path.exists() else []
    prior_spent = sum(float(row.get("total_cost_usd") or (row.get("usage") or {}).get("cost") or 0.0) for row in prior_rows)
    spent = 0.0
    answer_spent = agent_spent = 0.0
    succeeded = failed = 0
    attempted = 0
    stop_reason = None
    print(
        f"[run] model={args.model} condition={args.condition} | "
        f"completed={len(done)} remaining={len(selected)} | output={output_root}",
        flush=True,
    )
    if not selected:
        print("[run] nothing to do; all selected record IDs are already complete", flush=True)
    for index, (task_dir, record) in enumerate(selected, 1):
        if args.max_cost_usd is not None and prior_spent + spent >= args.max_cost_usd:
            stop_reason = "max_cost_usd"
            break
        attempted += 1
        record_id = str(record.get("id"))
        contexts = None
        agent_events: list[dict[str, Any]] = []
        agent_cost = 0.0
        try:
            query = str((record.get("input") or {}).get("question") or (record.get("input") or {}).get("text") or "")
            if retriever:
                contexts = retriever.search(query, str(record.get("leaf", "")), args.top_k)
                agent_events = retriever.pop_usage()
                agent_cost = sum(float((event.get("usage") or {}).get("cost") or 0.0) for event in agent_events)
                spent += agent_cost
                agent_spent += agent_cost
            messages = build_messages(record, task_dir, contexts=contexts, include_images=not args.no_images, max_image_bytes=args.max_image_bytes)
            response = client.complete(messages, config)
            raw_text = response_text(response)
            evaluation = score(record, raw_text)
            usage = response.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            spent += cost
            answer_spent += cost
            leaf = str(record.get("leaf", task_dir.name))
            append_jsonl(result_path, {
                "id": record_id, "leaf": record.get("leaf"), "bloom": (record.get("bloom") or {}).get("level"),
                "status": "ok", "condition": args.condition, "model": args.model, "completed_at": utc_now(),
                "prompt_hash": digest(messages), "retrieved_ids": [item.get("id") for item in contexts or []],
                "rag_applicable": bool(retriever and retriever.supports_leaf(leaf)),
                "response": raw_text, "usage": usage, "agent_usage": agent_events,
                "answer_cost_usd": cost, "agent_cost_usd": agent_cost,
                "total_cost_usd": cost + agent_cost,
                "latency_seconds": response.get("_latency_seconds"), **evaluation,
            })
            succeeded += 1
        except Exception as error:
            if retriever:
                extra_events = retriever.pop_usage()
                if extra_events:
                    agent_events.extend(extra_events)
                    extra_cost = sum(float((event.get("usage") or {}).get("cost") or 0.0) for event in extra_events)
                    agent_cost += extra_cost; spent += extra_cost; agent_spent += extra_cost
            append_jsonl(result_path, {
                "id": record_id, "leaf": record.get("leaf"), "status": "error",
                "condition": args.condition, "model": args.model, "completed_at": utc_now(),
                "error": repr(error), "agent_usage": agent_events,
                "agent_cost_usd": agent_cost, "total_cost_usd": agent_cost,
            })
            failed += 1
            print(f"[run:error] {record_id}: {error!r}", flush=True)
        if index == 1 or index % 10 == 0 or index == len(selected):
            print(
                f"[run] {index}/{len(selected)} attempted | {succeeded} ok | "
                f"{failed} errors | ${spent:.4f}",
                flush=True,
            )
    report = {
        "selected_remaining": len(selected), "attempted": attempted,
        "succeeded": succeeded, "failed": failed,
        "reported_cost_usd_this_invocation": spent,
        "reported_cost_usd_total": prior_spent + spent,
        "answer_cost_usd_this_invocation": answer_spent,
        "agent_cost_usd_this_invocation": agent_spent,
        "stop_reason": stop_reason, "results": str(result_path),
    }
    atomic_json(output_root / "run_summary.json", report)
    print(
        f"[run] complete | attempted={attempted} ok={succeeded} errors={failed} "
        f"total_cost=${prior_spent + spent:.4f} stop_reason={stop_reason}",
        flush=True,
    )
    return report


def add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("run", help="Run a resumable OpenRouter benchmark condition.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=("base", "base_rag", "agentic_rag"), default="base")
    parser.add_argument("--corpus-root", help="Required for base_rag or agentic_rag")
    parser.add_argument("--rag-backend", choices=("dense",), default="dense")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--max-passage-chars", type=int, default=1500)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--no-capability-gating", action="store_true")
    parser.add_argument("--agent-model", default="google/gemini-3.5-flash-lite")
    parser.add_argument("--agent-max-hops", type=int, default=2)
    parser.add_argument("--agent-subqueries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-image-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-leaf-limit", type=int, help="Evaluate at most this many records per leaf; use 1 for a 23-leaf pilot.")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore completed records; preserves existing result rows.")


def validate_run_args(args: argparse.Namespace) -> None:
    if args.condition in {"base_rag", "agentic_rag"} and not args.corpus_root:
        raise ValueError("--corpus-root is required for RAG conditions")
    if args.per_leaf_limit is not None and args.per_leaf_limit < 1:
        raise ValueError("--per-leaf-limit must be at least 1")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k")
    if args.max_context_chars < 1 or args.max_passage_chars < 1:
        raise ValueError("Context character budgets must be positive")
    if args.agent_max_hops < 1 or args.agent_subqueries < 1:
        raise ValueError("Agent hop and subquery counts must be positive")
