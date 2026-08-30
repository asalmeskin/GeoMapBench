from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import append_jsonl, atomic_json, digest, read_jsonl, utc_now
from .openrouter import OpenRouterClient, OpenRouterConfig, response_text
from .prompts import build_messages
from .retrieval import LexicalRetriever
from .scoring import score


def benchmark_records(root: Path, prefer_clean: bool = True) -> list[tuple[Path, dict[str, Any]]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")
    rows: list[tuple[Path, dict[str, Any]]] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        filename = "data_clean.jsonl" if prefer_clean and (task_dir / "data_clean.jsonl").exists() else "data.jsonl"
        path = task_dir / filename
        if path.exists():
            rows.extend((task_dir, record) for record in read_jsonl(path))
    if not rows:
        raise ValueError(f"No benchmark JSONL files found under {root}")
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("status") == "ok"}


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
    }
    existing_config = output_root / "run_config.json"
    if existing_config.exists() and not args.force:
        previous = __import__("json").loads(existing_config.read_text(encoding="utf-8"))
        for key in ("model", "condition", "temperature", "max_tokens", "benchmark_root", "corpus_root", "top_k", "include_images"):
            if previous.get(key) != run_config.get(key):
                raise ValueError("Output directory has a different run configuration; choose a new --output or use --force.")
    atomic_json(existing_config, run_config)
    retriever = LexicalRetriever(Path(args.corpus_root)) if args.condition == "rag" else None
    client = OpenRouterClient()
    done = set() if args.force else completed_ids(result_path)
    records = benchmark_records(benchmark_root, prefer_clean=not args.no_clean)
    selected = [(directory, record) for directory, record in records if record.get("id") not in done]
    if args.limit:
        selected = selected[:args.limit]
    spent = 0.0
    succeeded = failed = 0
    for index, (task_dir, record) in enumerate(selected, 1):
        record_id = str(record.get("id"))
        contexts = None
        try:
            query = str((record.get("input") or {}).get("question") or (record.get("input") or {}).get("text") or "")
            if retriever:
                contexts = retriever.search(query, str(record.get("leaf", "")), args.top_k)
            messages = build_messages(record, task_dir, contexts=contexts, include_images=not args.no_images, max_image_bytes=args.max_image_bytes)
            response = client.complete(messages, config)
            raw_text = response_text(response)
            evaluation = score(record, raw_text)
            usage = response.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            if args.max_cost_usd is not None and spent + cost > args.max_cost_usd:
                break
            spent += cost
            append_jsonl(result_path, {
                "id": record_id, "leaf": record.get("leaf"), "bloom": (record.get("bloom") or {}).get("level"),
                "status": "ok", "condition": args.condition, "model": args.model, "completed_at": utc_now(),
                "prompt_hash": digest(messages), "retrieved_ids": [item.get("id") for item in contexts or []],
                "response": raw_text, "usage": usage, "latency_seconds": response.get("_latency_seconds"), **evaluation,
            })
            succeeded += 1
        except Exception as error:
            append_jsonl(result_path, {"id": record_id, "leaf": record.get("leaf"), "status": "error", "condition": args.condition, "model": args.model, "completed_at": utc_now(), "error": repr(error)})
            failed += 1
        if index % 10 == 0:
            print(f"{index}/{len(selected)} attempted | {succeeded} ok | {failed} errors | ${spent:.4f}")
    report = {"attempted": len(selected), "succeeded": succeeded, "failed": failed, "reported_cost_usd": spent, "results": str(result_path)}
    atomic_json(output_root / "run_summary.json", report)
    return report


def add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("run", help="Run a resumable OpenRouter benchmark condition.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=("base", "rag"), default="base")
    parser.add_argument("--corpus-root", help="Required for --condition rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-image-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore completed records; preserves existing result rows.")


def validate_run_args(args: argparse.Namespace) -> None:
    if args.condition == "rag" and not args.corpus_root:
        raise ValueError("--corpus-root is required for --condition rag")
