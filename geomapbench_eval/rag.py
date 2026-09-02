from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .common import append_jsonl, read_jsonl, stable_json
from .openrouter import (
    OpenRouterClient, OpenRouterConfig, finish_reason, generation_failure, response_text,
)


TEXT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"
AGENT_PROTOCOL_REVISION = "2026-09-agent-json-v3-no-invalid-cache"


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
    for name in ("text.faiss", "text_metadata.jsonl", "text_manifest.json"):
        src = src_indexes / name
        dst = dst_indexes / name
        if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
            print(f"[rag] staging indexes/{name}", flush=True)
            shutil.copy2(src, dst)
    return destination


def _record_text(record: dict[str, Any]) -> str:
    inp = record.get("input") or {}
    return (str(inp.get("title") or "") + "\n" + str(inp.get("text") or "")).strip()


class DenseRAGRetriever:
    """Dense-only retrieval with optional cross-encoder reranking; no BM25 path."""

    def __init__(
        self,
        corpus_root: Path,
        *,
        candidate_k: int = 40,
        rerank: bool = True,
        max_passage_chars: int = 1500,
        max_context_chars: int = 6000,
        trace_path: Path | None = None,
    ):
        try:
            import faiss
            import torch
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install the project with the [rag-index] extra for RAG runs") from error
        self.faiss = faiss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.candidate_k = candidate_k
        self.max_passage_chars = max_passage_chars
        self.max_context_chars = max_context_chars
        self.trace_path = trace_path
        index_dir = corpus_root / "indexes"
        index_path = index_dir / "text.faiss"
        metadata_path = index_dir / "text_metadata.jsonl"
        manifest_path = index_dir / "text_manifest.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Dense index missing under {index_dir}; run `geomaprag-data index-text --root ...` first"
            )
        self.records = read_jsonl(metadata_path)
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.records):
            raise ValueError(
                f"text.faiss has {self.index.ntotal} vectors but metadata has {len(self.records)} rows"
            )
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        model_name = str(manifest.get("model") or TEXT_MODEL)
        self.encoder = SentenceTransformer(model_name, device=self.device)
        probe = self.encoder.encode([BGE_QUERY_PREFIX + "dimension check"], normalize_embeddings=True)
        if int(probe.shape[1]) != int(self.index.d):
            raise ValueError(f"query encoder dimension {probe.shape[1]} != FAISS dimension {self.index.d}")
        self.reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=512) if rerank else None
        self.last_trace: dict[str, Any] = {}
        self.last_usage: dict[str, Any] = {"cost": 0.0, "calls": 0}
        print(
            f"[rag] dense index ready: {self.index.ntotal:,} vectors x {self.index.d}; "
            f"encoder={model_name}; rerank={bool(self.reranker)}; device={self.device}",
            flush=True,
        )

    def _dense(self, query: str, k: int) -> list[dict[str, Any]]:
        vector = self.encoder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(np.ascontiguousarray(vector), min(k, self.index.ntotal))
        hits: list[dict[str, Any]] = []
        for rank, (index, value) in enumerate(zip(indices[0], scores[0]), 1):
            if index < 0:
                continue
            record = self.records[int(index)]
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

    def _render(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = self.max_context_chars
        contexts: list[dict[str, Any]] = []
        for hit in hits:
            record = hit["record"]
            inp = record.get("input") or {}
            text = str(inp.get("text") or "")[: self.max_passage_chars]
            text = text[:remaining]
            if len(text.strip()) < 40:
                continue
            contexts.append({
                "id": str(record.get("id")),
                "input": {"title": str(inp.get("title") or "Reference"), "text": text},
                "retrieval": record.get("retrieval") or {},
            })
            remaining -= len(text)
            if remaining <= 0:
                break
        return contexts

    def search(self, query: str, leaf: str, top_k: int) -> list[dict[str, Any]]:
        hits = self._rerank(query, self._dense(query, self.candidate_k), top_k)
        contexts = self._render(hits)
        self.last_usage = {"cost": 0.0, "calls": 0}
        self.last_trace = {
            "query": query[:300],
            "leaf": leaf,
            "mode": "base_rag",
            "record_ids": [str(item["record"].get("id")) for item in hits],
            "dense_scores": [round(float(item["dense_score"]), 6) for item in hits],
            "context_chars": sum(len(item["input"]["text"]) for item in contexts),
        }
        return contexts


class AgenticRAGRetriever(DenseRAGRetriever):
    """Dense-only RAG plus cached planning and evidence judging calls."""

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
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("status") == "ok" and isinstance(cached.get("value"), dict):
                self._query_usage["cached_calls"] += 1
                return dict(cached["value"])
        response = self.agent_client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.agent_config,
        )
        usage = response.get("usage") or {}
        self._query_usage["calls"] += 1
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
            self._query_usage["agent_failures"] += 1
            self._query_usage["failure_kinds"].append({
                "tag": tag,
                "kind": failure,
                "finish_reason": finish_reason(response),
            })
            print(
                f"[rag:agent-warning] {tag} failed ({failure}); using deterministic dense fallback",
                flush=True,
            )
            return {"_error": failure}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "status": "ok",
            "revision": AGENT_PROTOCOL_REVISION,
            "value": value,
        }, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return value

    def search(self, query: str, leaf: str, top_k: int) -> list[dict[str, Any]]:
        self._query_usage = {
            "cost": 0.0, "calls": 0, "cached_calls": 0,
            "agent_failures": 0, "failure_kinds": [],
        }
        plan_system = (
            "Plan dense retrieval for a geospatial question. Return JSON only as "
            '{"queries":["..."]}. Produce at most '
            f"{self.max_subqueries} short, complementary search queries. Do not answer."
        )
        plan = self._agent(plan_system, f"Leaf: {leaf}\nQuestion: {query[:2000]}", "plan")
        planned = [
            item.strip() for item in (plan.get("queries") or [])
            if isinstance(item, str) and item.strip()
        ][: self.max_subqueries]
        queries = [query] + [item for item in planned if item != query]
        pool: dict[str, dict[str, Any]] = {}
        for subquery in queries:
            for hit in self._dense(subquery, self.candidate_k):
                record_id = str(hit["record"].get("id"))
                previous = pool.get(record_id)
                if previous is None or hit["dense_score"] > previous["dense_score"]:
                    pool[record_id] = hit
        candidates = self._rerank(query, list(pool.values()), max(top_k + 7, 12))
        kept = candidates[:top_k]
        judge_system = (
            "Audit geospatial evidence. Return JSON only as "
            '{"keep":[1],"sufficient":true,"followup":""}. '
            "keep uses 1-based passage indices. Do not answer the question."
        )
        for hop in range(self.max_hops):
            listing = "\n\n".join(
                f"[{index + 1}] {_record_text(hit['record'])[:900]}"
                for index, hit in enumerate(candidates)
            )
            verdict = self._agent(
                judge_system, f"Question: {query[:1800]}\n\nPassages:\n{listing}", f"judge-{hop}"
            )
            indices = [
                item - 1 for item in (verdict.get("keep") or [])
                if isinstance(item, int) and 1 <= item <= len(candidates)
            ]
            kept = [candidates[index] for index in indices][:top_k] or candidates[:top_k]
            if verdict.get("sufficient") or hop == self.max_hops - 1:
                break
            followup = str(verdict.get("followup") or "").strip()
            if not followup:
                break
            for hit in self._dense(followup, self.candidate_k):
                record_id = str(hit["record"].get("id"))
                if record_id not in pool or hit["dense_score"] > pool[record_id]["dense_score"]:
                    pool[record_id] = hit
            candidates = self._rerank(query, list(pool.values()), max(top_k + 7, 12))
        contexts = self._render(kept)
        self.last_usage = dict(self._query_usage)
        self.last_trace = {
            "query": query[:300],
            "leaf": leaf,
            "mode": "agentic_rag",
            "planned_queries": planned,
            "record_ids": [str(item["record"].get("id")) for item in kept],
            "context_chars": sum(len(item["input"]["text"]) for item in contexts),
            "agent_usage": self.last_usage,
        }
        return contexts
