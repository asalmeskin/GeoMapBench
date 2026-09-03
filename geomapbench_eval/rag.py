from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import canonical_benchmark_records
from .common import append_jsonl, atomic_json, read_jsonl, stable_json
from .openrouter import (
    OpenRouterClient, OpenRouterConfig, finish_reason, generation_failure, response_text,
)
from .prompts import build_messages, input_asset_paths, transport_image


TEXT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"
IMAGE_MODEL = "openai/clip-vit-base-patch32"
AGENT_PROTOCOL_REVISION = "2026-09-agent-multimodal-json-v5-abstain"
MULTIMODAL_RETRIEVAL_REVISION = "2026-09-bge-clip-rrf-v1"
RRF_K = 60
EXPECTED_TEXT_COUNT = 180_344
EXPECTED_IMAGE_COUNT = 1_794
EXPECTED_TEXT_DIMENSION = 384
EXPECTED_IMAGE_DIMENSION = 512

# Corpus coverage was declared before evaluation. Retrieval is disabled outside
# these leaves so irrelevant knowledge cannot damage image-only/perception tasks.
RAG_APPLICABLE_LEAVES = frozenset({
    "coordinate_transformation",
    "cross_entity_comparison",
    "geo_entity_typing",
    "geographic_fact_reasoning",
    "geologic_geomorphic_interpretation",
    "isochrone_service_area",
    "map_label_feature_anchoring",
    "metric_distance_computation",
    "population_density_estimation",
    "shortest_path_optimization",
    "spatial_graph_construction",
    "topological_directional_reasoning",
    "toponym_recognition",
    "visual_geolocation",
})


def stage_corpus(source: Path, destination: Path) -> Path:
    """Copy the corpus/index files from Drive to fast local storage, resumably."""
    source = source.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    names = ["corpus_clean.jsonl"] if (source / "corpus_clean.jsonl").exists() else ["corpus.jsonl"]
    for name in names:
        src = source / name
        dst = destination / name
        if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
            print(f"[rag] staging {name} ({src.stat().st_size / 1e6:.1f} MB)", flush=True)
            shutil.copy2(src, dst)
    src_indexes = source / "indexes"
    dst_indexes = destination / "indexes"
    dst_indexes.mkdir(exist_ok=True)
    for name in (
        "text.faiss", "text_metadata.jsonl", "text_manifest.json",
        "image.faiss", "image_metadata.jsonl", "image_manifest.json",
    ):
        src = src_indexes / name
        dst = dst_indexes / name
        if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
            print(f"[rag] staging indexes/{name}", flush=True)
            shutil.copy2(src, dst)
    return destination


def _record_text(record: dict[str, Any]) -> str:
    inp = record.get("input") or {}
    return (str(inp.get("title") or "") + "\n" + str(inp.get("text") or "")).strip()


def _clip_feature_tensor(output: Any, *, expected_dimension: int, modality: str) -> Any:
    """Accept both legacy Tensor and Transformers 5 ModelOutput CLIP APIs."""
    candidates = [output]
    for attribute in (f"{modality}_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            candidates.append(value)
    if isinstance(output, (tuple, list)):
        candidates.extend(output)
    for candidate in candidates:
        shape = getattr(candidate, "shape", None)
        ndim = getattr(candidate, "ndim", None)
        if (
            callable(getattr(candidate, "norm", None))
            and ndim == 2
            and shape is not None
            and int(shape[-1]) == expected_dimension
        ):
            return candidate
    raise TypeError(
        f"CLIP {modality} feature output {type(output).__name__} does not contain "
        f"a 2-D projected tensor with dimension {expected_dimension}"
    )


class MultimodalRAGRetriever:
    """BGE text retrieval plus CLIP image retrieval and rank-based fusion."""

    def __init__(
        self,
        corpus_root: Path,
        *,
        candidate_k: int = 40,
        rerank: bool = True,
        max_passage_chars: int = 1500,
        max_context_chars: int = 6000,
        media_root: Path | None = None,
        image_candidate_k: int = 20,
        max_reference_images: int = 1,
        trace_path: Path | None = None,
    ):
        try:
            import faiss
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install the project with the [rag-index] extra for RAG runs") from error
        self.faiss = faiss
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.candidate_k = candidate_k
        self.image_candidate_k = image_candidate_k
        self.max_reference_images = max_reference_images
        self.max_passage_chars = max_passage_chars
        self.max_context_chars = max_context_chars
        self.trace_path = trace_path
        self.corpus_root = corpus_root.expanduser().resolve()
        self.media_root = (media_root or corpus_root).expanduser().resolve()
        index_dir = corpus_root / "indexes"
        text_index_path = index_dir / "text.faiss"
        text_metadata_path = index_dir / "text_metadata.jsonl"
        text_manifest_path = index_dir / "text_manifest.json"
        image_index_path = index_dir / "image.faiss"
        image_metadata_path = index_dir / "image_metadata.jsonl"
        image_manifest_path = index_dir / "image_manifest.json"
        required = (
            text_index_path, text_metadata_path, text_manifest_path,
            image_index_path, image_metadata_path, image_manifest_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Multimodal indexes are incomplete; missing: " + ", ".join(missing)
            )
        self.text_records = read_jsonl(text_metadata_path)
        self.text_index = faiss.read_index(str(text_index_path))
        self.image_records = read_jsonl(image_metadata_path)
        self.image_index = faiss.read_index(str(image_index_path))
        if self.text_index.ntotal != len(self.text_records):
            raise ValueError(
                f"text.faiss has {self.text_index.ntotal} vectors but metadata has {len(self.text_records)} rows"
            )
        if self.image_index.ntotal != len(self.image_records):
            raise ValueError(
                f"image.faiss has {self.image_index.ntotal} vectors but metadata has {len(self.image_records)} rows"
            )
        text_manifest = json.loads(text_manifest_path.read_text())
        image_manifest = json.loads(image_manifest_path.read_text())
        text_model_name = str(text_manifest.get("model") or TEXT_MODEL)
        image_model_name = str(image_manifest.get("model") or IMAGE_MODEL)
        manifest_checks = {
            "text.model": (text_model_name, TEXT_MODEL),
            "text.count": (int(text_manifest.get("count") or -1), EXPECTED_TEXT_COUNT),
            "text.dimension": (int(text_manifest.get("dimension") or -1), EXPECTED_TEXT_DIMENSION),
            "image.model": (image_model_name, IMAGE_MODEL),
            "image.count": (int(image_manifest.get("count") or -1), EXPECTED_IMAGE_COUNT),
            "image.dimension": (int(image_manifest.get("dimension") or -1), EXPECTED_IMAGE_DIMENSION),
        }
        bad_manifest = {
            key: {"found": found, "expected": expected}
            for key, (found, expected) in manifest_checks.items() if found != expected
        }
        if bad_manifest:
            raise ValueError("Frozen multimodal corpus manifest mismatch: " + json.dumps(bad_manifest))
        if not text_manifest.get("normalized") or not image_manifest.get("normalized"):
            raise ValueError("Both frozen text and image indexes must contain normalized vectors")
        self.encoder = SentenceTransformer(text_model_name, device=self.device)
        probe = self.encoder.encode([BGE_QUERY_PREFIX + "dimension check"], normalize_embeddings=True)
        if int(probe.shape[1]) != int(self.text_index.d):
            raise ValueError(f"BGE query dimension {probe.shape[1]} != text FAISS dimension {self.text_index.d}")
        self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        projection = int(getattr(self.clip_model.config, "projection_dim", 0))
        if projection != int(self.image_index.d):
            raise ValueError(f"CLIP projection dimension {projection} != image FAISS dimension {self.image_index.d}")
        self.reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=512) if rerank else None
        self.last_trace: dict[str, Any] = {}
        self.last_usage: dict[str, Any] = {"cost": 0.0, "calls": 0}
        print(
            f"[rag] MULTIMODAL indexes ready: text={self.text_index.ntotal:,}x{self.text_index.d} "
            f"({text_model_name}); image={self.image_index.ntotal:,}x{self.image_index.d} "
            f"({image_model_name}); rerank={bool(self.reranker)}; device={self.device}",
            flush=True,
        )

    def _dense(self, query: str, k: int) -> list[dict[str, Any]]:
        vector = self.encoder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.text_index.search(
            np.ascontiguousarray(vector), min(k, self.text_index.ntotal)
        )
        hits: list[dict[str, Any]] = []
        for rank, (index, value) in enumerate(zip(indices[0], scores[0]), 1):
            if index < 0:
                continue
            record = self.text_records[int(index)]
            hits.append({"record": record, "dense_score": float(value), "dense_rank": rank})
        return hits

    def _rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not self.reranker or not hits:
            return hits[:top_k]
        pairs = [(query, _record_text(hit["record"])[:2400]) for hit in hits]
        batch_size = 64 if self.device == "cuda" else 16
        values = self.reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        for hit, value in zip(hits, values):
            hit["rerank_score"] = float(value)
        return sorted(hits, key=lambda item: (-item["rerank_score"], item["dense_rank"]))[:top_k]

    def _corpus_image_path(self, record: dict[str, Any]) -> Path | None:
        images = (record.get("input") or {}).get("images") or []
        if not isinstance(images, list) or not images:
            return None
        path = (self.media_root / str(images[0])).resolve()
        if self.media_root != path and self.media_root not in path.parents:
            return None
        return path if path.is_file() else None

    def _clip_query_vector(
        self, query: str, record: dict[str, Any], task_dir: Path,
    ) -> tuple[np.ndarray, int]:
        from PIL import Image

        images: list[Any] = []
        try:
            for source in input_asset_paths(record, task_dir):
                _, _, transported = transport_image(source)
                images.append(Image.open(transported).convert("RGB"))
            if images:
                inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    raw_features = self.clip_model.get_image_features(**inputs)
                features = _clip_feature_tensor(
                    raw_features,
                    expected_dimension=int(self.image_index.d),
                    modality="image",
                )
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                vector = features.mean(dim=0, keepdim=True)
                vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                return vector.cpu().numpy().astype("float32"), len(images)
            inputs = self.clip_processor(text=[query[:2000]], return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                raw_features = self.clip_model.get_text_features(**inputs)
            features = _clip_feature_tensor(
                raw_features,
                expected_dimension=int(self.image_index.d),
                modality="text",
            )
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return features.cpu().numpy().astype("float32"), 0
        finally:
            for image in images:
                image.close()

    def _image(
        self, query: str, record: dict[str, Any], task_dir: Path, k: int,
    ) -> tuple[list[dict[str, Any]], int]:
        vector, benchmark_image_count = self._clip_query_vector(query, record, task_dir)
        scores, indices = self.image_index.search(
            np.ascontiguousarray(vector), min(k, self.image_index.ntotal)
        )
        hits: list[dict[str, Any]] = []
        for rank, (index, value) in enumerate(zip(indices[0], scores[0]), 1):
            if index < 0 or not math.isfinite(float(value)):
                continue
            hits.append({
                "record": self.image_records[int(index)],
                "image_score": float(value),
                "image_rank": rank,
            })
        return hits, benchmark_image_count

    def _fuse(
        self,
        text_hits: list[dict[str, Any]],
        image_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        pool: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(text_hits, 1):
            record_id = str(hit["record"].get("id"))
            item = pool.setdefault(record_id, {"record": hit["record"], "fusion_score": 0.0})
            item.update(hit)
            item["fusion_score"] += 1.25 / (RRF_K + rank)
            item["text_used"] = True
        for rank, hit in enumerate(image_hits, 1):
            record_id = str(hit["record"].get("id"))
            item = pool.setdefault(record_id, {"record": hit["record"], "fusion_score": 0.0})
            item.update(hit)
            item["fusion_score"] += 1.25 / (RRF_K + rank)
            item["image_used"] = True
        return sorted(
            pool.values(),
            key=lambda item: (-float(item["fusion_score"]), str(item["record"].get("id"))),
        )[:top_k]

    def _render(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = self.max_context_chars
        attached_images = 0
        contexts: list[dict[str, Any]] = []
        for hit in hits:
            record = hit["record"]
            inp = record.get("input") or {}
            text = str(inp.get("text") or "")[: self.max_passage_chars]
            text = text[:remaining]
            image_path = self._corpus_image_path(record) if hit.get("image_used") else None
            if len(text.strip()) < 40 and image_path is None:
                continue
            context = {
                "id": str(record.get("id")),
                "input": {"title": str(inp.get("title") or "Reference"), "text": text},
                "retrieval": record.get("retrieval") or {},
                "fusion_score": float(hit.get("fusion_score") or 0.0),
                "modalities": [name for name in ("text", "image") if hit.get(f"{name}_used")],
                "image_paths": [],
            }
            if image_path is not None and attached_images < self.max_reference_images:
                context["image_paths"] = [str(image_path)]
                attached_images += 1
            contexts.append(context)
            remaining -= len(text)
            if remaining <= 0:
                break
        return contexts

    def _candidates(
        self, query: str, record: dict[str, Any], task_dir: Path, top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
        text_hits = self._rerank(query, self._dense(query, self.candidate_k), self.candidate_k)
        image_hits, benchmark_image_count = self._image(
            query, record, task_dir, self.image_candidate_k,
        )
        return self._fuse(text_hits, image_hits, top_k), text_hits, image_hits, benchmark_image_count

    def validate_runtime(self, benchmark_root: Path) -> dict[str, Any]:
        candidates = canonical_benchmark_records(benchmark_root)
        selected = next(
            (
                (directory, record) for directory, record in candidates
                if str(record.get("leaf")) in RAG_APPLICABLE_LEAVES
                and input_asset_paths(record, directory)
            ),
            None,
        )
        if selected is None:
            raise ValueError("No image-bearing RAG-applicable benchmark record exists for validation")
        task_dir, record = selected
        inp = record.get("input") or {}
        query = str(inp.get("question") or inp.get("base_question") or inp.get("text") or stable_json(inp))
        fused, text_hits, image_hits, benchmark_images = self._candidates(
            query, record, task_dir, min(5, self.candidate_k),
        )
        if not text_hits or not image_hits or not fused or benchmark_images < 1:
            raise RuntimeError("Multimodal validation failed: both text and image retrieval must return hits")
        accessible_image_hit = next(
            (hit for hit in image_hits if self._corpus_image_path(hit["record"])),
            None,
        )
        if accessible_image_hit is None:
            raise RuntimeError("Multimodal validation failed: retrieved corpus image files are inaccessible")
        # Validate the final transport path as well as the two indexes.  This is
        # deliberately performed before OpenRouterClient is constructed or any
        # paid answer request can be made.
        validation_hits = self._fuse([text_hits[0]], [accessible_image_hit], 2)
        contexts = self._render(validation_hits)
        text_contexts = sum(bool(str(item["input"].get("text") or "").strip()) for item in contexts)
        reference_images = sum(len(item.get("image_paths") or []) for item in contexts)
        messages = build_messages(record, task_dir, contexts=contexts, include_images=True)
        user_parts = messages[1]["content"]
        prompt_image_parts = sum(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in user_parts
        )
        prompt_text = "\n".join(
            str(part.get("text") or "")
            for part in user_parts
            if isinstance(part, dict) and part.get("type") == "text"
        )
        fallback_sentence_present = (
            "answer from the original task images and your own knowledge" in prompt_text
        )
        if (
            text_contexts < 1
            or reference_images < 1
            or prompt_image_parts <= benchmark_images
            or not fallback_sentence_present
        ):
            raise RuntimeError(
                "Multimodal validation failed: final prompt must contain retrieved text, "
                "an encoded retrieved image, the original benchmark image, and the knowledge fallback"
            )
        report = {
            "status": "pass",
            "revision": MULTIMODAL_RETRIEVAL_REVISION,
            "text_index_count": int(self.text_index.ntotal),
            "text_dimension": int(self.text_index.d),
            "image_index_count": int(self.image_index.ntotal),
            "image_dimension": int(self.image_index.d),
            "benchmark_image_count": benchmark_images,
            "text_hits": len(text_hits),
            "image_hits": len(image_hits),
            "fused_hits": len(fused),
            "rendered_text_contexts": text_contexts,
            "retrieved_reference_images": reference_images,
            "final_prompt_image_parts": prompt_image_parts,
            "fallback_sentence_present": fallback_sentence_present,
            "sample_id": str(record.get("id")),
            "both_modalities_verified": True,
            "paid_api_calls": 0,
        }
        print(
            "[rag-validation] PASS: BGE text retrieval + CLIP benchmark-image retrieval + "
            "corpus-image access + multimodal fusion are all active",
            flush=True,
        )
        print(f"[rag-validation] {json.dumps(report, sort_keys=True)}", flush=True)
        return report

    def search(
        self, query: str, leaf: str, top_k: int, *, record: dict[str, Any], task_dir: Path,
    ) -> list[dict[str, Any]]:
        if leaf not in RAG_APPLICABLE_LEAVES:
            self.last_usage = {"cost": 0.0, "calls": 0}
            self.last_trace = {
                "query": query[:300], "leaf": leaf, "mode": "multimodal_rag",
                "abstained": True, "abstain_reason": "outside_predeclared_corpus_coverage",
                "text_index_used": False, "image_index_used": False,
            }
            return []
        hits, text_hits, image_hits, benchmark_image_count = self._candidates(
            query, record, task_dir, top_k,
        )
        contexts = self._render(hits)
        self.last_usage = {"cost": 0.0, "calls": 0}
        self.last_trace = {
            "query": query[:300],
            "leaf": leaf,
            "mode": "multimodal_rag",
            "revision": MULTIMODAL_RETRIEVAL_REVISION,
            "text_index_used": True,
            "image_index_used": True,
            "benchmark_image_count": benchmark_image_count,
            "abstained": not bool(contexts),
            "record_ids": [str(item["record"].get("id")) for item in hits],
            "text_record_ids": [str(item["record"].get("id")) for item in text_hits[:top_k]],
            "image_record_ids": [str(item["record"].get("id")) for item in image_hits[:top_k]],
            "image_scores": [round(float(item["image_score"]), 6) for item in image_hits[:top_k]],
            "context_chars": sum(len(item["input"]["text"]) for item in contexts),
            "reference_images": sum(len(item.get("image_paths") or []) for item in contexts),
        }
        return contexts


class AgenticMultimodalRAGRetriever(MultimodalRAGRetriever):
    """Multimodal retrieval plus cached planning, judging, and real abstention."""

    def __init__(
        self,
        corpus_root: Path,
        *,
        agent_model: str,
        agent_cache: Path,
        agent_max_tokens: int = 2048,
        agent_reasoning_effort: str = "minimal",
        max_subqueries: int = 3,
        max_hops: int = 2,
        **kwargs: Any,
    ):
        super().__init__(corpus_root, **kwargs)
        self.agent_model = agent_model
        self.agent_cache = agent_cache
        self.agent_cache.mkdir(parents=True, exist_ok=True)
        self.agent_config = OpenRouterConfig(
            agent_model, temperature=0.0, max_tokens=agent_max_tokens,
            timeout_seconds=240, retries=6, reasoning_effort=agent_reasoning_effort,
            reasoning_enabled=True, request_delay_seconds=1.0,
        )
        self.agent_client = OpenRouterClient()
        self.max_subqueries = max_subqueries
        self.max_hops = max_hops
        self._query_usage = {
            "cost": 0.0, "calls": 0, "cached_calls": 0,
            "agent_failures": 0, "failure_kinds": [],
        }

    def _agent(self, system: str, user: str, tag: str) -> dict[str, Any]:
        key = hashlib.sha256(
            stable_json({
                "revision": AGENT_PROTOCOL_REVISION,
                "model": self.agent_model,
                "max_tokens": self.agent_config.max_tokens,
                "reasoning_enabled": self.agent_config.reasoning_enabled,
                "reasoning_effort": self.agent_config.reasoning_effort,
                "tag": tag,
                "system": system,
                "user": user,
            }).encode()
        ).hexdigest()
        path = self.agent_cache / f"{key}.json"
        response_path = self.agent_cache / f"{key}.response.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("status") == "ok" and isinstance(cached.get("value"), dict):
                self._query_usage["cached_calls"] += 1
                return dict(cached["value"])
        recovered_response = False
        if response_path.exists():
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            response = payload.get("response") if isinstance(payload, dict) else None
            if not isinstance(response, dict):
                response_path.unlink(missing_ok=True)
                response = None
            else:
                recovered_response = True
                self._query_usage["cached_calls"] += 1
                print(f"[rag:agent-cache-hit] recovered {tag} response", flush=True)
        else:
            response = None
        if response is None:
            response = self.agent_client.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.agent_config,
            )
            # Write the raw response before parsing so a runtime interruption cannot
            # turn a paid planner/judge response into a duplicate request.
            atomic_json(response_path, {"status": "received", "response": response})
            self._query_usage["calls"] += 1
        usage = response.get("usage") or {}
        if not recovered_response:
            self._query_usage["cost"] += float(usage.get("cost") or 0.0)
        raw = response_text(response)
        failure = generation_failure(response, raw)
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                failure = failure or "agent_non_object"
        except json.JSONDecodeError:
            value = {}
            failure = failure or "agent_invalid_json"
        if failure:
            response_path.unlink(missing_ok=True)
            self._query_usage["agent_failures"] += 1
            self._query_usage["failure_kinds"].append({
                "tag": tag,
                "kind": failure,
                "finish_reason": finish_reason(response),
            })
            print(
                f"[rag:agent-warning] {tag} failed ({failure}); using deterministic multimodal fallback",
                flush=True,
            )
            return {"_error": failure}
        atomic_json(path, {
            "status": "ok",
            "revision": AGENT_PROTOCOL_REVISION,
            "value": value,
        })
        response_path.unlink(missing_ok=True)
        return value

    def search(
        self, query: str, leaf: str, top_k: int, *, record: dict[str, Any], task_dir: Path,
    ) -> list[dict[str, Any]]:
        self._query_usage = {
            "cost": 0.0, "calls": 0, "cached_calls": 0,
            "agent_failures": 0, "failure_kinds": [],
        }
        if leaf not in RAG_APPLICABLE_LEAVES:
            self.last_usage = dict(self._query_usage)
            self.last_trace = {
                "query": query[:300], "leaf": leaf, "mode": "agentic_multimodal_rag",
                "abstained": True, "abstain_reason": "outside_predeclared_corpus_coverage",
                "text_index_used": False, "image_index_used": False,
                "agent_usage": self.last_usage,
            }
            return []
        plan_system = (
            "Plan text and image retrieval for a geospatial question. Return JSON only as "
            '{"queries":["..."]}. Produce at most '
            f"{self.max_subqueries} short, complementary search queries. Do not answer. "
            "The benchmark image is searched separately with CLIP."
        )
        plan = self._agent(plan_system, f"Leaf: {leaf}\nQuestion: {query[:2000]}", "plan")
        planned = [
            item.strip() for item in (plan.get("queries") or [])
            if isinstance(item, str) and item.strip()
        ][: self.max_subqueries]
        queries = [query] + [item for item in planned if item != query]
        text_pool: dict[str, dict[str, Any]] = {}
        for subquery in queries:
            for hit in self._dense(subquery, self.candidate_k):
                record_id = str(hit["record"].get("id"))
                previous = text_pool.get(record_id)
                if previous is None or hit["dense_score"] > previous["dense_score"]:
                    text_pool[record_id] = hit
        text_hits = self._rerank(query, list(text_pool.values()), self.candidate_k)
        image_hits, benchmark_image_count = self._image(
            query, record, task_dir, self.image_candidate_k,
        )
        candidates = self._fuse(text_hits, image_hits, max(top_k + 7, 12))
        kept = candidates[:top_k]
        judge_system = (
            "Audit optional geospatial text/image evidence. Return JSON only as "
            '{"keep":[1],"use_context":true,"sufficient":true,"followup":""}. '
            "keep uses 1-based evidence indices. Set use_context=false and keep=[] when the "
            "evidence is not clearly useful; abstention is preferred to distracting context. "
            "Do not answer the question."
        )
        for hop in range(self.max_hops):
            listing = "\n\n".join(
                f"[{index + 1}] modalities={','.join(name for name in ('text', 'image') if hit.get(name + '_used'))}; "
                f"image_similarity={hit.get('image_score')}; {_record_text(hit['record'])[:900]}"
                for index, hit in enumerate(candidates)
            )
            verdict = self._agent(
                judge_system, f"Question: {query[:1800]}\n\nPassages:\n{listing}", f"judge-{hop}"
            )
            indices = [
                item - 1 for item in (verdict.get("keep") or [])
                if isinstance(item, int) and 1 <= item <= len(candidates)
            ]
            if verdict.get("_error"):
                kept = candidates[: min(top_k, 3)]
            elif verdict.get("use_context") is False or not indices:
                kept = []
            else:
                kept = [candidates[index] for index in indices][:top_k]
            if verdict.get("sufficient") or hop == self.max_hops - 1:
                break
            followup = str(verdict.get("followup") or "").strip()
            if not followup:
                break
            for hit in self._dense(followup, self.candidate_k):
                record_id = str(hit["record"].get("id"))
                if record_id not in text_pool or hit["dense_score"] > text_pool[record_id]["dense_score"]:
                    text_pool[record_id] = hit
            text_hits = self._rerank(query, list(text_pool.values()), self.candidate_k)
            candidates = self._fuse(text_hits, image_hits, max(top_k + 7, 12))
        contexts = self._render(kept)
        self.last_usage = dict(self._query_usage)
        self.last_trace = {
            "query": query[:300],
            "leaf": leaf,
            "mode": "agentic_multimodal_rag",
            "revision": MULTIMODAL_RETRIEVAL_REVISION,
            "text_index_used": True,
            "image_index_used": True,
            "benchmark_image_count": benchmark_image_count,
            "abstained": not bool(contexts),
            "planned_queries": planned,
            "record_ids": [str(item["record"].get("id")) for item in kept],
            "context_chars": sum(len(item["input"]["text"]) for item in contexts),
            "reference_images": sum(len(item.get("image_paths") or []) for item in contexts),
            "agent_usage": self.last_usage,
        }
        return contexts


# Backward-compatible imports. The v2.2 suite uses the explicit multimodal
# names and new output conditions, so old text-only caches cannot be mixed in.
DenseRAGRetriever = MultimodalRAGRetriever
AgenticRAGRetriever = AgenticMultimodalRAGRetriever
