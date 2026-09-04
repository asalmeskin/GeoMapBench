"""Hybrid multimodal retrieval for GeoAgent v3.

Built on top of the validated v2.2 ``MultimodalRAGRetriever`` so the frozen
manifest checks, the CLIP image index and the cross-encoder are reused exactly.
What is new:

* multi-query dense retrieval (question + planner sub-queries + entity queries),
* in-pool BM25 lexical rescoring, so rare proper nouns are not washed out by a
  384-dimensional embedding,
* capability-aware boosting using the corpus' own ``retrieval.capabilities``
  declaration, which the v2.2 retriever ignored entirely,
* reciprocal-rank fusion over four ranked lists instead of two, and
* maximal-marginal-relevance selection so the final passages are not three
  near-duplicate chunks of the same article.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from geomapbench_eval.rag import (
    RAG_APPLICABLE_LEAVES, MultimodalRAGRetriever, _clip_feature_tensor, _record_text,
)
from geomapbench_eval.prompts import transport_image

from . import RETRIEVAL_REVISION

RRF_K = 60
WEIGHTS = {"rerank": 1.6, "dense": 1.0, "lexical": 0.9, "image": 1.1}
CAPABILITY_BONUS = 0.30
MMR_LAMBDA = 0.60
RERANK_POOL = 120

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).casefold())


class HybridMultimodalRetriever(MultimodalRAGRetriever):
    """Multi-query, capability-aware, diversity-selecting multimodal retrieval."""

    def __init__(self, *args: Any, structured_index: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.structured_index = structured_index
        self.revision = RETRIEVAL_REVISION

    # -- scoring components ----------------------------------------------------

    def _lexical_ranking(self, query: str, pool: list[dict[str, Any]]) -> list[str]:
        """BM25 over the candidate pool only; document frequencies come from the pool."""
        query_terms = [term for term in tokens(query) if len(term) > 1]
        if not query_terms or not pool:
            return []
        documents: list[tuple[str, list[str]]] = []
        for hit in pool:
            text = _record_text(hit["record"])[:1600]
            documents.append((str(hit["record"].get("id")), tokens(text)))
        lengths = [len(terms) for _, terms in documents]
        average = sum(lengths) / max(len(lengths), 1) or 1.0
        frequency: Counter[str] = Counter()
        for _, terms in documents:
            frequency.update(set(terms))
        total = len(documents)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, str]] = []
        for (record_id, terms), length in zip(documents, lengths):
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                if term not in counts:
                    continue
                df = frequency.get(term, 0)
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                tf = counts[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
            if score > 0:
                scored.append((score, record_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [record_id for _, record_id in scored]

    def _capability_match(self, record: dict[str, Any], leaf: str) -> bool:
        capabilities = (record.get("retrieval") or {}).get("capabilities") or []
        return leaf in {str(item) for item in capabilities}

    @staticmethod
    def _similarity(first: set[str], second: set[str]) -> float:
        if not first or not second:
            return 0.0
        return len(first & second) / len(first | second)

    def _mmr(self, ranked: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Greedy maximal marginal relevance on scale-free relevance.

        Fusion scores are normalised to [0, 1] first, otherwise the raw RRF
        magnitudes (around 0.05) make the redundancy penalty meaningless and
        three chunks of the same article can fill the whole context budget.
        """
        if len(ranked) <= top_k:
            return ranked
        signatures = {
            str(hit["record"].get("id")): set(tokens(_record_text(hit["record"])[:800]))
            for hit in ranked
        }
        top = max(float(hit.get("fusion_score") or 0.0) for hit in ranked) or 1.0
        selected: list[dict[str, Any]] = []
        remaining = list(ranked)
        while remaining and len(selected) < top_k:
            best, best_score = None, -1e9
            for hit in remaining:
                relevance = float(hit.get("fusion_score") or 0.0) / top
                penalty = max(
                    (
                        self._similarity(
                            signatures[str(hit["record"].get("id"))],
                            signatures[str(chosen["record"].get("id"))],
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * penalty
                if score > best_score:
                    best, best_score = hit, score
            assert best is not None
            selected.append(best)
            remaining.remove(best)
        return selected

    # -- image side ------------------------------------------------------------

    def _clip_vector(self, image_paths: list[Path], fallback_text: str) -> tuple[Any, int]:
        """Encode the benchmark images with CLIP, falling back to the question text."""
        from PIL import Image

        images: list[Any] = []
        try:
            for source in image_paths:
                _, _, transported = transport_image(Path(source))
                images.append(Image.open(transported).convert("RGB"))
            if images:
                inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    raw = self.clip_model.get_image_features(**inputs)
                features = _clip_feature_tensor(
                    raw, expected_dimension=int(self.image_index.d), modality="image",
                )
            else:
                inputs = self.clip_processor(
                    text=[fallback_text[:2000]], return_tensors="pt", padding=True, truncation=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    raw = self.clip_model.get_text_features(**inputs)
                features = _clip_feature_tensor(
                    raw, expected_dimension=int(self.image_index.d), modality="text",
                )
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            vector = features.mean(dim=0, keepdim=True)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return vector.cpu().numpy().astype("float32"), len(images)
        finally:
            for image in images:
                image.close()

    def _image_hits(self, image_paths: list[Path], fallback_text: str) -> tuple[list[dict[str, Any]], int]:
        vector, encoded = self._clip_vector(image_paths, fallback_text)
        scores, indices = self.image_index.search(
            np.ascontiguousarray(vector), min(self.image_candidate_k, self.image_index.ntotal)
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
        return hits, encoded

    # -- fusion ----------------------------------------------------------------

    def _fuse_ranked(
        self,
        pool: dict[str, dict[str, Any]],
        rankings: dict[str, list[str]],
        leaf: str,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {record_id: 0.0 for record_id in pool}
        for source, ordering in rankings.items():
            weight = WEIGHTS.get(source, 1.0)
            for rank, record_id in enumerate(ordering, 1):
                if record_id in scores:
                    scores[record_id] += weight / (RRF_K + rank)
        fused: list[dict[str, Any]] = []
        for record_id, hit in pool.items():
            matched = self._capability_match(hit["record"], leaf)
            hit = dict(hit)
            hit["capability_match"] = matched
            hit["fusion_score"] = scores[record_id] * (1.0 + CAPABILITY_BONUS * matched)
            fused.append(hit)
        fused.sort(key=lambda item: (-float(item["fusion_score"]), str(item["record"].get("id"))))
        return fused

    # -- public entry point ----------------------------------------------------

    def search_evidence(
        self,
        view: Any,
        *,
        image_paths: list[Path],
        queries: list[str],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return rendered contexts plus a full retrieval trace."""
        leaf = view.leaf
        if leaf not in RAG_APPLICABLE_LEAVES:
            return [], {
                "mode": "geoagent_hybrid",
                "revision": self.revision,
                "leaf": leaf,
                "abstained": True,
                "abstain_reason": "outside_predeclared_corpus_coverage",
                "text_index_used": False,
                "image_index_used": False,
            }
        ordered_queries = [item for item in dict.fromkeys(q.strip() for q in queries if q and q.strip())][:5]
        if not ordered_queries:
            ordered_queries = [view.question]

        pool: dict[str, dict[str, Any]] = {}
        dense_rankings: list[list[str]] = []
        for query in ordered_queries:
            ranking: list[str] = []
            for hit in self._dense(query, self.candidate_k):
                record_id = str(hit["record"].get("id"))
                ranking.append(record_id)
                previous = pool.get(record_id)
                if previous is None or hit["dense_score"] > previous.get("dense_score", -1e9):
                    merged = dict(previous or {})
                    merged.update(hit)
                    pool[record_id] = merged
            dense_rankings.append(ranking)

        # Rank fusion across sub-queries first, so one noisy sub-query cannot
        # dominate the pool ordering.
        dense_scores: dict[str, float] = {record_id: 0.0 for record_id in pool}
        for ranking in dense_rankings:
            for rank, record_id in enumerate(ranking, 1):
                dense_scores[record_id] = dense_scores.get(record_id, 0.0) + 1.0 / (RRF_K + rank)
        dense_order = sorted(dense_scores, key=lambda key: (-dense_scores[key], key))

        pool_hits = [pool[record_id] for record_id in dense_order]
        lexical_order = self._lexical_ranking(view.question, pool_hits)

        rerank_candidates = pool_hits[:RERANK_POOL]
        reranked = self._rerank(view.question, rerank_candidates, len(rerank_candidates))
        rerank_order = [str(hit["record"].get("id")) for hit in reranked]
        for hit in reranked:
            record_id = str(hit["record"].get("id"))
            if record_id in pool:
                pool[record_id]["rerank_score"] = hit.get("rerank_score")

        image_hits, benchmark_image_count = self._image_hits(image_paths, view.question)
        image_order: list[str] = []
        for hit in image_hits:
            record_id = str(hit["record"].get("id"))
            image_order.append(record_id)
            merged = dict(pool.get(record_id) or {})
            merged.update(hit)
            merged["image_used"] = True
            pool[record_id] = merged
        for record_id in dense_order:
            pool[record_id].setdefault("text_used", True)

        fused = self._fuse_ranked(
            pool,
            {
                "dense": dense_order,
                "lexical": lexical_order,
                "rerank": rerank_order,
                "image": image_order,
            },
            leaf,
        )
        shortlist = self._mmr(fused[: max(top_k * 4, 12)], top_k)
        contexts = self._render(shortlist)
        trace = {
            "mode": "geoagent_hybrid",
            "revision": self.revision,
            "leaf": leaf,
            "text_index_used": True,
            "image_index_used": bool(image_order),
            "benchmark_image_count": benchmark_image_count,
            "queries": ordered_queries,
            "pool_size": len(pool),
            "capability_matches": sum(bool(hit.get("capability_match")) for hit in shortlist),
            "record_ids": [str(hit["record"].get("id")) for hit in shortlist],
            "fusion_scores": [round(float(hit.get("fusion_score") or 0.0), 8) for hit in shortlist],
            "abstained": not bool(contexts),
            "context_chars": sum(len(item["input"]["text"]) for item in contexts),
            "reference_images": sum(len(item.get("image_paths") or []) for item in contexts),
        }
        return contexts, trace
