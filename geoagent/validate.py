"""Runtime validation for the GeoAgent condition.

Mirrors ``geomapbench_eval.rag.MultimodalRAGRetriever.validate_runtime``: prove
that both indexes fire, that a retrieved corpus image is actually accessible on
disk, and that the final prompt really carries both modalities, before any
paid answer call is made.

The probe deliberately *forces* one accessible image hit into the validation
context rather than trusting the normal hybrid ranking to surface one within
``top_k`` -- MMR, capability boosting and BM25 can all legitimately push every
image-bearing candidate out of a real record's shortlist, and that is fine for
an ordinary run (retrieval is not required to attach a reference image on
every record). It is not fine for the one-time proof that image transport
works at all, so this mirrors ``MultimodalRAGRetriever.validate_runtime``
exactly: dense text hit [0] fused with one image hit known to resolve to a
real file, rendered, and checked end to end through prompt assembly.
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

    text_hits = retriever._rerank(
        view.question, retriever._dense(view.question, retriever.candidate_k), retriever.candidate_k,
    )
    image_hits, benchmark_image_count = retriever._image_hits(image_paths, view.question)
    if not text_hits or not image_hits:
        raise RuntimeError("GeoAgent runtime validation failed: text and image retrieval must both return hits")
    accessible_image_hit = next(
        (hit for hit in image_hits if retriever._corpus_image_path(hit["record"])), None,
    )
    if accessible_image_hit is None:
        raise RuntimeError("GeoAgent runtime validation failed: retrieved corpus image files are inaccessible")

    # Force both modalities into the rendered context; this is the one place
    # the probe does not use the ordinary top-k hybrid shortlist.
    validation_hits = retriever._fuse([text_hits[0]], [accessible_image_hit], 2)
    contexts = retriever._render(validation_hits)

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
    prompt_image_parts = sum(part.get("type") == "image_url" for part in parts)
    text_contexts = sum(bool(str(item["input"].get("text") or "").strip()) for item in contexts)
    reference_images = sum(len(item.get("image_paths") or []) for item in contexts)

    report = {
        "status": "pass",
        "sample_id": view.record_id,
        "text_index_count": int(retriever.text_index.ntotal),
        "image_index_count": int(retriever.image_index.ntotal),
        "benchmark_image_count": benchmark_image_count,
        "text_hits": len(text_hits),
        "image_hits": len(image_hits),
        "fused_hits": len(validation_hits),
        "rendered_text_contexts": text_contexts,
        "retrieved_reference_images": reference_images,
        "prompt_image_parts": prompt_image_parts,
        "fallback_sentence_present": (
            "answer from the original task images and your own knowledge" in prompt_text
        ),
        "verified_block_present": "Verified computations" in prompt_text,
        "structured_corpus": dict(getattr(structured_index, "stats", {})),
        "paid_api_calls": 0,
    }
    # The "Verified computations" block is legitimately optional per record --
    # not every probe record has an applicable tool -- so it is reported for
    # visibility but is not a pass/fail condition here.
    if (
        text_contexts < 1
        or reference_images < 1
        or prompt_image_parts <= benchmark_image_count
        or not report["fallback_sentence_present"]
    ):
        raise RuntimeError(f"GeoAgent runtime validation failed: {report}")
    return report
