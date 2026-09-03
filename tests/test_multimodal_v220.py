from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from geomapbench_eval.prompts import build_messages
from geomapbench_eval.rag import RAG_APPLICABLE_LEAVES, _clip_feature_tensor, stage_corpus


class _FakeTensor:
    ndim = 2

    def __init__(self, dimension: int):
        self.shape = (2, dimension)

    def norm(self, *args, **kwargs):
        raise AssertionError("Extraction should not call norm")


class _FakeModelOutput:
    def __init__(self, dimension: int):
        self.pooler_output = _FakeTensor(dimension)


def test_clip_feature_extraction_supports_tensor_and_model_output() -> None:
    legacy = _FakeTensor(512)
    modern = _FakeModelOutput(512)

    assert _clip_feature_tensor(
        legacy, expected_dimension=512, modality="image"
    ) is legacy
    assert _clip_feature_tensor(
        modern, expected_dimension=512, modality="image"
    ) is modern.pooler_output


def test_stage_corpus_includes_both_indexes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    indexes = source / "indexes"
    indexes.mkdir(parents=True)
    (source / "corpus_clean.jsonl").write_text("{}\n", encoding="utf-8")
    for name in (
        "text.faiss", "text_metadata.jsonl", "text_manifest.json",
        "image.faiss", "image_metadata.jsonl", "image_manifest.json",
    ):
        (indexes / name).write_bytes(name.encode())

    staged = stage_corpus(source, destination)

    for name in (
        "text.faiss", "text_metadata.jsonl", "text_manifest.json",
        "image.faiss", "image_metadata.jsonl", "image_manifest.json",
    ):
        assert (staged / "indexes" / name).is_file()


def test_rag_prompt_has_fallback_and_retrieved_image(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    original = task / "original.png"
    retrieved = tmp_path / "retrieved.png"
    Image.new("RGB", (8, 8), "blue").save(original)
    Image.new("RGB", (8, 8), "green").save(retrieved)
    record = {
        "id": "sample-001",
        "leaf": "visual_geolocation",
        "input": {"question": "Where is this?", "image": "original.png"},
        "target": {"bloom_answer": "x"},
        "evaluation": {"type": "exact_match"},
    }
    contexts = [{
        "id": "corpus-image-1",
        "input": {"title": "Reference", "text": "A useful geographic reference."},
        "image_paths": [str(retrieved)],
    }]

    messages = build_messages(record, task, contexts=contexts)
    parts = messages[1]["content"]
    text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
    image_parts = [part for part in parts if part.get("type") == "image_url"]

    assert "answer from the original task images and your own knowledge" in text
    assert "Retrieved reference image" in text
    assert len(image_parts) == 2


def test_coverage_gate_is_predeclared() -> None:
    assert len(RAG_APPLICABLE_LEAVES) == 14
    assert "visual_geolocation" in RAG_APPLICABLE_LEAVES
    assert "dense_land_cover_labeling" not in RAG_APPLICABLE_LEAVES
