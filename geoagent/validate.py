"""Runtime validation for the GeoAgent condition.

Mirrors ``geomapbench_eval.rag.MultimodalRAGRetriever.validate_runtime``: prove
that both indexes fire and that the final prompt really carries both
modalities, before any paid answer call is made. This runs the full stack one
record deep -- toolbelt, retrieval, and prompt assembly -- so it also proves
the "Verified computations" block reaches the prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geomapbench_eval.prompts import input_asset_paths
from geomapbench_eval.rag import RAG_APPLICABLE_LEAVES

from .prompting import build_agent_messages
from .taskview import TaskView
from .tools import run_toolbelt


def validate_runtime(
    retriever: Any,
    structured_index: Any,
    cohort_records: list[tuple[Path, dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    selected = next(
        (
            (directory, record) for directory, record in cohort_records
            if str(record.get("leaf")) in RAG_APPLICABLE_LEAVES
            and input_asset_paths(record, directory)
        ),
        None,
    )
    if selected is None:
        raise ValueError("No image-bearing RAG-applicable cohort record exists for validation")
    task_dir, record = selected
    view = TaskView.from_record(record, task_dir)
    tool_results = run_toolbelt(view, structured_index)
    image_paths = input_asset_paths(record, task_dir)
    contexts, trace = retriever.search_evidence(
        view, image_paths=image_paths,
        queries=[view.question] + view.entity_names()[:2], top_k=top_k,
    )
    blocks = [
        {"title": result.title, "authority": "authoritative", "text": result.text}
        for result in tool_results if result.ok
    ][:3]
    messages = build_agent_messages(
        record, task_dir, tool_blocks=blocks, contexts=contexts,
        answer_shape=None, include_images=True,
    )
    parts = messages[1]["content"]
    prompt_text = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
    image_parts = sum(part.get("type") == "image_url" for part in parts)
    report = {
        "status": "pass",
        "sample_id": view.record_id,
        "text_index_count": int(retriever.text_index.ntotal),
        "image_index_count": int(retriever.image_index.ntotal),
        "text_index_used": bool(trace.get("text_index_used")),
        "image_index_used": bool(trace.get("image_index_used")),
        "benchmark_images_encoded": int(trace.get("benchmark_image_count") or 0),
        "retrieved_passages": len(contexts),
        "capability_matched_passages": int(trace.get("capability_matches") or 0),
        "prompt_image_parts": image_parts,
        "fallback_sentence_present": (
            "answer from the original task images and your own knowledge" in prompt_text
        ),
        "verified_block_present": "Verified computations" in prompt_text,
        "structured_corpus": dict(getattr(structured_index, "stats", {})),
        "paid_api_calls": 0,
    }
    for key in ("text_index_used", "image_index_used", "fallback_sentence_present"):
        if not report[key]:
            raise RuntimeError(f"GeoAgent runtime validation failed on {key}: {report}")
    if report["benchmark_images_encoded"] < 1 or image_parts <= report["benchmark_images_encoded"]:
        raise RuntimeError(f"GeoAgent runtime validation failed: image transport mismatch: {report}")
    if not contexts:
        raise RuntimeError("GeoAgent runtime validation failed: retrieval returned no passages")
    return report
