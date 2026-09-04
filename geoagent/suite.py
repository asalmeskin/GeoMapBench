"""One CLI command that runs the whole GeoAgent v3 condition end to end.

This is the codebase-side counterpart of what used to be five notebook cells:
cohort + corpus staging, runtime validation, a free offline toolbelt audit,
the live resumable run, and the offline analysis/ablation/plot pass. A Colab
notebook now only needs to mount Drive, clone this repository, install it, and
invoke ``geoagent-eval suite`` -- see ``GeoMapBench_AgenticRAG_v3.ipynb``.

Nothing here ever opens an earlier result directory for writing: previous
conditions are only ever read through byte-for-byte copies taken via
``report.snapshot_condition``, so `base` and the v2.2 RAG runs remain usable
as ablations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from geomapbench_eval.common import atomic_json
from geomapbench_eval.experiments import _catalog_check, _model_row, _modality_audit, _trusted_benchmark_report
from geomapbench_eval.rag import stage_corpus

from . import __version__
from .agents import CachedAgent
from .corpus_index import StructuredCorpusIndex
from .driver import RunConfig, build_cohort, run_agentic, selected_records
from .pipeline import AgenticPipeline
from .report import (
    condition_plots, paired_compare, pre_repair_ablation, retrieval_audit,
    snapshot_condition, toolbelt_audit,
)
from .retrieval import HybridMultimodalRetriever
from .taskview import TaskView
from .tools import propose_answer, run_toolbelt
from .validate import validate_runtime


def _offline_toolbelt_audit(
    cohort_records: list[tuple[Path, dict[str, Any]]], structured_index: StructuredCorpusIndex,
) -> dict[str, Any]:
    """Run the deterministic toolbelt over the whole cohort with zero API calls."""
    tool_counts: Counter[str] = Counter()
    proposal_counts: Counter[str] = Counter()
    exact_by_leaf: Counter[str] = Counter()
    records_by_leaf: Counter[str] = Counter()
    for directory, record in cohort_records:
        view = TaskView.from_record(record, directory)
        records_by_leaf[view.leaf] += 1
        results = run_toolbelt(view, structured_index)
        for result in results:
            if result.ok:
                tool_counts[result.name] += 1
        proposal = propose_answer(view, results)
        if proposal is not None:
            proposal_counts[f"{proposal.source}:{proposal.confidence}"] += 1
            if proposal.confidence == "exact":
                exact_by_leaf[view.leaf] += 1
    return {
        "records": len(cohort_records),
        "tool_counts": dict(tool_counts.most_common()),
        "proposal_counts": dict(proposal_counts.most_common()),
        "exact_by_leaf": dict(exact_by_leaf),
        "records_by_leaf": dict(records_by_leaf),
        "records_with_exact_proposal": sum(exact_by_leaf.values()),
    }


def run_geoagent_suite(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    condition_dir = output / args.condition
    print(f"[geoagent-suite] geoagent {__version__} | output: {output}", flush=True)

    benchmark_report = _trusted_benchmark_report(args)
    benchmark_hash = str(benchmark_report["portable_benchmark_hash"])

    models_path = Path(args.models)
    answer_row = _model_row(models_path, args.model)
    agent_row = _model_row(models_path, args.agent_model)
    problems = _catalog_check([answer_row, agent_row])
    if problems:
        raise ValueError(f"GeoAgent model catalog validation failed: {problems}")

    records, cohort_path, cohort = build_cohort(
        Path(args.benchmark_root), output,
        target_per_leaf=args.target_per_leaf, benchmark_content_hash=benchmark_hash,
    )
    cohort_records, cohort_ids = selected_records(records, cohort_path)
    print(
        f"[geoagent-suite] cohort target_per_leaf={cohort['target_per_leaf']} "
        f"records={cohort['target_record_count']} ids={cohort['selected_ids_hash'][:12]}",
        flush=True,
    )

    local_corpus = stage_corpus(Path(args.corpus_root), Path(args.work_root) / "corpus")
    retriever = HybridMultimodalRetriever(
        local_corpus,
        media_root=Path(args.corpus_root),
        candidate_k=args.candidate_k,
        image_candidate_k=args.image_candidate_k,
        max_reference_images=args.max_reference_images,
        rerank=not args.no_rerank,
        max_passage_chars=args.max_passage_chars,
        max_context_chars=args.max_context_chars,
    )
    structured_index = StructuredCorpusIndex(retriever.text_records)
    retriever.structured_index = structured_index
    print(f"[geoagent-suite] structured corpus views: {structured_index.stats}", flush=True)
    if structured_index.stats["indicator_observations"] < 1000:
        raise RuntimeError("Too few World Bank observations parsed; the indicator tool would be inert")
    if structured_index.stats["gazetteer_entries"] < 1000:
        raise RuntimeError("Too few geocoded corpus entries indexed; the gazetteer tool would be inert")

    validation = validate_runtime(retriever, structured_index, cohort_records, top_k=args.top_k)
    atomic_json(output / "geoagent_validation.json", validation)
    print("[geoagent-validation] PASS", json.dumps(validation, sort_keys=True)[:900], flush=True)

    offline_audit = _offline_toolbelt_audit(cohort_records, structured_index)
    atomic_json(output / "toolbelt_audit_offline.json", offline_audit)
    print(
        f"[geoagent-suite] offline toolbelt: {offline_audit['records_with_exact_proposal']}/"
        f"{offline_audit['records']} cohort records have a closed-form answer proposal",
        flush=True,
    )

    result: dict[str, Any] = {
        "geoagent_version": __version__,
        "benchmark_report": benchmark_report,
        "cohort": cohort,
        "validation": validation,
        "toolbelt_audit_offline": offline_audit,
    }
    if args.dry_run:
        print("[geoagent-suite] --dry-run: stopping before any paid API call", flush=True)
        atomic_json(output / "geoagent_suite_summary.json", result)
        return result

    agent = CachedAgent(
        model=args.agent_model,
        cache_root=Path(args.agent_cache).expanduser(),
        max_tokens=args.agent_max_tokens,
        reasoning_effort=args.agent_reasoning_effort,
        reasoning_enabled=True,
        request_delay_seconds=args.agent_request_delay_seconds,
    )
    pipeline = AgenticPipeline(
        retriever=retriever, agent=agent, structured_index=structured_index,
        top_k=args.top_k,
        use_analyst=not args.no_analyst, use_critic=not args.no_critic,
        use_verifier=not args.no_verifier, allow_revision=not args.no_revision,
        max_tool_chars=args.max_tool_chars,
    )
    run_config = RunConfig(
        benchmark_root=Path(args.benchmark_root),
        output=condition_dir,
        condition=args.condition,
        model=args.model,
        max_tokens=int(answer_row.get("max_tokens", 16384)),
        temperature=answer_row.get("temperature", 0.0),
        reasoning_effort=answer_row.get("reasoning_effort"),
        reasoning_enabled=bool(answer_row.get("reasoning_enabled", False)),
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        request_delay_seconds=args.request_delay_seconds,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
        max_consecutive_errors=args.max_consecutive_errors,
        max_cost_usd=args.max_cost_usd,
        progress_every=args.progress_every,
        top_k=args.top_k,
        benchmark_content_hash=benchmark_hash,
        retrieval_config={
            "system": "geoagent_v3",
            "candidate_k": args.candidate_k, "image_candidate_k": args.image_candidate_k,
            "top_k": args.top_k, "max_reference_images": args.max_reference_images,
            "max_passage_chars": args.max_passage_chars, "max_context_chars": args.max_context_chars,
            "max_tool_chars": args.max_tool_chars, "rerank": not args.no_rerank,
            "hybrid_lexical": True, "capability_boost": True, "mmr": True,
            "analyst": not args.no_analyst, "critic": not args.no_critic,
            "verifier": not args.no_verifier, "revision": not args.no_revision,
            "agent_model": args.agent_model,
        },
    )
    print(f"[geoagent-suite] START {args.condition}", flush=True)
    summary = run_agentic(run_config, records=records, cohort_path=cohort_path, pipeline=pipeline, agent=agent)
    print(f"[geoagent-suite] {'DONE' if summary['complete'] else 'PAUSED'} {args.condition}", flush=True)
    result["run"] = summary
    if not summary["complete"]:
        print(
            f"[geoagent-suite] incomplete ({summary['run_stop_reason']}); rerun the same command "
            "to resume without repeating completed records or paid calls",
            flush=True,
        )
        atomic_json(output / "geoagent_suite_summary.json", result)
        return result

    # -- offline analysis, free ablations, comparisons and plots ---------------
    from geomapbench_eval.analysis import analyze, compare

    benchmark_root = Path(args.benchmark_root)
    new_results = condition_dir / "responses.jsonl"
    analyses = {args.condition: analyze(new_results, condition_dir / "analysis", benchmark_root=benchmark_root)}
    snapshots: dict[str, Path] = {}
    ablation_root = output / "ablation_inputs"
    if args.base_results and Path(args.base_results).is_file():
        snapshots["base"] = snapshot_condition(Path(args.base_results), ablation_root / "base")
    for name in [item.strip() for item in (args.previous_rag_conditions or "").split(",") if item.strip()]:
        previous = Path(args.previous_rag_root or "") / name / "responses.jsonl"
        if previous.is_file():
            snapshots[name] = snapshot_condition(previous, ablation_root / name)
    for name, path in snapshots.items():
        analyses[name] = analyze(path, path.parent / "analysis", benchmark_root=benchmark_root)

    comparisons: dict[str, Any] = {}
    for name, path in snapshots.items():
        try:
            comparisons[f"{name}_to_{args.condition}"] = paired_compare(
                path, new_results, output / "comparisons" / f"{name}_to_{args.condition}",
                label=f"{name} -> {args.condition}",
            )
        except ValueError as error:
            comparisons[f"{name}_to_{args.condition}"] = {"protocol_validation": "deferred", "reason": str(error)}
            print(f"[geoagent-suite] comparison deferred for {name}: {error}", flush=True)
    if "base" in snapshots:
        try:
            comparisons["official_base_to_geoagent"] = compare(
                snapshots["base"], new_results, output / "comparisons" / "official_base_to_geoagent",
            )
        except ValueError as error:
            comparisons["official_base_to_geoagent"] = {"protocol_validation": "deferred", "reason": str(error)}
            print(f"[geoagent-suite] official comparison deferred: {error}", flush=True)

    repair_report = pre_repair_ablation(condition_dir, benchmark_root)
    tools_report = toolbelt_audit(condition_dir / "agent_trace.jsonl")
    retrieval_report = retrieval_audit(condition_dir / "retrieval_trace.jsonl")
    legacy_modality_audit = _modality_audit(condition_dir / "retrieval_trace.jsonl")
    plots = condition_plots(output, analyses, comparisons)

    result.update({
        "analyses": {key: {
            "condition_summary": value.get("condition_summary"),
            "task_aware_macro_by_condition": value.get("task_aware_macro_by_condition"),
            "strict_macro_by_condition": value.get("strict_macro_by_condition"),
        } for key, value in analyses.items()},
        "comparisons": comparisons,
        "repair_ablation": repair_report,
        "toolbelt_audit": tools_report,
        "retrieval_audit": retrieval_report,
        "modality_audit": legacy_modality_audit,
        "plots": plots,
    })
    atomic_json(output / "geoagent_suite_summary.json", result)
    print(f"[geoagent-suite] finished; summary written to {output / 'geoagent_suite_summary.json'}", flush=True)
    return result


def add_geoagent_parser(sub: argparse._SubParsersAction[Any]) -> None:
    suite = sub.add_parser(
        "suite",
        help="Run the tool-augmented, self-verifying GeoAgent v3 condition end to end.",
    )
    suite.add_argument("--benchmark-root", required=True)
    suite.add_argument("--corpus-root", required=True)
    suite.add_argument("--work-root", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--models", required=True, help="The exact model matrix used by the Evaluation notebook")
    suite.add_argument(
        "--benchmark-report", required=True,
        help="Existing passed benchmark_preflight.json; it is trusted without rescanning assets",
    )
    suite.add_argument("--model", required=True)
    suite.add_argument("--condition", default="geoagent_tool_rag")
    suite.add_argument(
        "--base-results", help="Optional completed base responses.jsonl, copied in for a paired comparison",
    )
    suite.add_argument(
        "--previous-rag-root", help="Optional root of an earlier rag-suite output, e.g. rag_suite_multimodal_claude_v221",
    )
    suite.add_argument(
        "--previous-rag-conditions", default="multimodal_rag,agentic_multimodal_rag",
        help="Comma-separated condition subfolders under --previous-rag-root to compare against",
    )
    suite.add_argument(
        "--target-per-leaf", type=int, default=1,
        help="Cumulative nested target per leaf, shared with base/rag-suite cohorts",
    )
    suite.add_argument("--max-cost-usd", type=float, default=16.0)
    suite.add_argument("--progress-every", type=int, default=5)
    suite.add_argument("--timeout-seconds", type=int, default=240)
    suite.add_argument("--retries", type=int, default=6)
    suite.add_argument("--request-delay-seconds", type=float, default=1.0)
    suite.add_argument("--retry-base-seconds", type=float, default=5.0)
    suite.add_argument("--retry-max-seconds", type=float, default=60.0)
    suite.add_argument("--max-consecutive-errors", type=int, default=2)
    suite.add_argument("--top-k", type=int, default=4)
    suite.add_argument("--candidate-k", type=int, default=60)
    suite.add_argument("--image-candidate-k", type=int, default=20)
    suite.add_argument("--max-reference-images", type=int, default=1)
    suite.add_argument("--max-passage-chars", type=int, default=900)
    suite.add_argument("--max-context-chars", type=int, default=3000)
    suite.add_argument("--max-tool-chars", type=int, default=2600)
    suite.add_argument("--no-rerank", action="store_true")
    suite.add_argument("--agent-model", default="google/gemini-3.5-flash-lite")
    suite.add_argument("--agent-cache", required=True)
    suite.add_argument("--agent-max-tokens", type=int, default=1024)
    suite.add_argument("--agent-reasoning-effort", default="minimal")
    suite.add_argument("--agent-request-delay-seconds", type=float, default=0.4)
    suite.add_argument("--no-analyst", action="store_true")
    suite.add_argument("--no-critic", action="store_true")
    suite.add_argument("--no-verifier", action="store_true")
    suite.add_argument("--no-revision", action="store_true")
    suite.add_argument(
        "--dry-run", action="store_true",
        help="Stop after cohort/corpus setup, runtime validation and the free toolbelt audit; no paid API calls",
    )
