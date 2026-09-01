from __future__ import annotations

import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .common import append_jsonl, read_jsonl, utc_now
from .openrouter import OpenRouterClient, OpenRouterConfig, response_text


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def stage_corpus(source: Path, destination: Path) -> Path:
    """Copy only retrieval-time corpus artifacts from slow mounted storage."""
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"GeoMapRAG corpus root not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    def copy_verified(src: Path, dst: Path) -> None:
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            print(f"[rag-stage] reuse {dst.name} ({dst.stat().st_size / 1_000_000:.1f} MB)", flush=True)
            return
        temporary = dst.with_name(f".{dst.name}.part")
        print(f"[rag-stage] copy {src} -> {dst} ({src.stat().st_size / 1_000_000:.1f} MB)", flush=True)
        shutil.copy2(src, temporary)
        if temporary.stat().st_size != src.stat().st_size:
            raise OSError(f"Incomplete staged copy: {src} -> {temporary}")
        os.replace(temporary, dst)

    # Dense retrieval reads text_metadata.jsonl directly; the full corpus JSONL
    # is intentionally not copied into ephemeral Colab storage.
    for name in ("quality_report.json", "release_manifest.json"):
        src = source / name
        if src.exists():
            copy_verified(src, destination / name)
    src_index, dst_index = source / "indexes", destination / "indexes"
    dst_index.mkdir(exist_ok=True)
    for name in ("text.faiss", "text_metadata.jsonl", "text_manifest.json"):
        src = src_index / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required dense retrieval artifact: {src}")
        dst = dst_index / name
        copy_verified(src, dst)
    print(f"[rag-stage] dense retrieval artifacts ready: {destination}", flush=True)
    return destination


class DenseRAGRetriever:
    """BGE dense retrieval plus cross-encoder reranking; no BM25 or agent calls."""

    def __init__(
        self, corpus_root: Path, *, candidate_k: int = 50,
        reranker_model: str | None = "BAAI/bge-reranker-base",
        max_passage_chars: int = 1500, max_context_chars: int = 6000,
        trace_path: Path | None = None, device: str | None = None,
        capability_aware: bool = True,
    ):
        import faiss
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.root = Path(corpus_root).expanduser().resolve()
        print(f"[retriever] loading dense corpus/index from {self.root}", flush=True)
        index_dir = self.root / "indexes"
        manifest_path = index_dir / "text_manifest.json"
        metadata_path = index_dir / "text_metadata.jsonl"
        index_path = index_dir / "text.faiss"
        for path in (manifest_path, metadata_path, index_path):
            if not path.exists():
                raise FileNotFoundError(path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records = read_jsonl(metadata_path)
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.records) or self.manifest.get("count") != len(self.records):
            raise ValueError("text.faiss, text_metadata.jsonl, and text_manifest.json disagree")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"[retriever] {len(self.records)} passages | embedding={self.manifest['model']} | "
            f"reranker={reranker_model or 'none'} | device={self.device}",
            flush=True,
        )
        self.encoder = SentenceTransformer(str(self.manifest["model"]), device=self.device)
        self.reranker = CrossEncoder(reranker_model, device=self.device, max_length=512) if reranker_model else None
        print("[retriever] models and FAISS index ready", flush=True)
        self.reranker_model = reranker_model
        self.candidate_k = candidate_k
        self.max_passage_chars = max_passage_chars
        self.max_context_chars = max_context_chars
        self.trace_path = Path(trace_path) if trace_path else None
        self.capability_aware = capability_aware
        self.covered_leaves = {
            str(capability)
            for row in self.records
            for capability in ((row.get("retrieval") or {}).get("capabilities") or [])
        }
        self._pending_usage: list[dict[str, Any]] = []

    def supports_leaf(self, leaf: str) -> bool:
        return not self.capability_aware or leaf in self.covered_leaves

    @staticmethod
    def _text(row: dict[str, Any]) -> str:
        inp = row.get("input") or {}
        return (str(inp.get("title") or "") + "\n" + str(inp.get("text") or "")).strip()

    def search(self, query: str, leaf: str, top_k: int, *, _write_trace: bool = True) -> list[dict[str, Any]]:
        applicable = self.supports_leaf(leaf)
        if not applicable:
            if _write_trace:
                self._trace(query, leaf, [], 0, applicable=False)
            return []
        encoded = self.encoder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")
        depth = min(self.candidate_k, self.index.ntotal)
        dense_scores, indices = self.index.search(np.ascontiguousarray(encoded), depth)
        candidates: list[dict[str, Any]] = []
        for dense_score, index in zip(dense_scores[0], indices[0]):
            if index < 0:
                continue
            row = self.records[int(index)]
            caps = set((row.get("retrieval") or {}).get("capabilities") or [])
            candidates.append({"row": row, "dense_score": float(dense_score), "capability_match": leaf in caps})
        if self.reranker and candidates:
            pairs = [(query, self._text(item["row"])[:2500]) for item in candidates]
            scores = self.reranker.predict(pairs, batch_size=64 if self.device == "cuda" else 16, show_progress_bar=False)
            for item, rerank_score in zip(candidates, scores):
                item["rerank_score"] = float(rerank_score) + (0.10 if item["capability_match"] else 0.0)
            candidates.sort(key=lambda item: (-item["rerank_score"], -item["dense_score"]))
        else:
            candidates.sort(key=lambda item: (not item["capability_match"], -item["dense_score"]))
        budget, output = self.max_context_chars, []
        for item in candidates[:top_k]:
            row = item["row"]
            inp = row.get("input") or {}
            text = str(inp.get("text") or "")[: self.max_passage_chars]
            text = text[:budget]
            if len(text.strip()) < 60:
                continue
            budget -= len(text)
            output.append({
                "id": str(row.get("id")),
                "input": {"title": str(inp.get("title") or "Reference"), "text": text},
                "retrieval": row.get("retrieval") or {},
                "_dense_score": item["dense_score"],
                "_rerank_score": item.get("rerank_score"),
            })
            if budget <= 0:
                break
        if _write_trace:
            self._trace(query, leaf, output, self.max_context_chars - budget, applicable=True)
        return output

    def pop_usage(self) -> list[dict[str, Any]]:
        usage, self._pending_usage = self._pending_usage, []
        return usage

    def _trace(self, query: str, leaf: str, rows: list[dict[str, Any]], context_chars: int, *, applicable: bool) -> None:
        if not self.trace_path:
            return
        append_jsonl(self.trace_path, {
            "created_at": utc_now(), "query": query[:300], "leaf": leaf,
            "backend": "dense_rerank", "applicable": applicable,
            "record_ids": [row.get("id") for row in rows],
            "context_chars": context_chars,
            "embedding_model": self.manifest.get("model"),
            "reranker_model": self.reranker_model,
        })


class AgenticRAGRetriever(DenseRAGRetriever):
    """LLM-planned multi-query dense RAG with evidence judging; still no BM25."""

    PLAN_SYSTEM = (
        "Plan dense retrieval for a geospatial benchmark. Return JSON only as "
        '{"queries":[str,...]}. Produce short search queries that cover all entities, '
        "places, coordinate systems, indicators, years, and relations. Do not answer."
    )
    JUDGE_SYSTEM = (
        "Audit retrieved evidence for a geospatial benchmark. Return JSON only as "
        '{"keep":[int,...],"sufficient":bool,"followup":str}. `keep` uses 1-based '
        "passage indices. `followup` is one short query for missing evidence. Do not answer."
    )

    def __init__(
        self, *args: Any, agent_model: str = "google/gemini-3.5-flash-lite",
        agent_max_hops: int = 2, agent_subqueries: int = 3,
        agent_cache: Path | None = None, **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.agent_model = agent_model
        self.agent_max_hops = agent_max_hops
        self.agent_subqueries = agent_subqueries
        self.agent_cache = Path(agent_cache or (self.root / ".agent_cache"))
        self.agent_cache.mkdir(parents=True, exist_ok=True)
        self.agent_client = OpenRouterClient()
        self.agent_config = OpenRouterConfig(
            agent_model, temperature=0.0, max_tokens=400,
            timeout_seconds=120, retries=4,
        )
        self._agent_errors: list[str] = []

    def _agent(self, system: str, user: str, tag: str) -> dict[str, Any]:
        key = hashlib.sha256(
            f"agentic-rag-v1|{self.agent_model}|{tag}|{system}|{user}".encode("utf-8")
        ).hexdigest()
        cache_path = self.agent_cache / f"{key}.json"
        if cache_path.exists():
            self._pending_usage.append({
                "role": tag, "model": self.agent_model, "usage": {},
                "latency_seconds": 0.0, "cached": True,
            })
            return json.loads(cache_path.read_text(encoding="utf-8"))["value"]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = self.agent_client.complete(messages, self.agent_config)
        usage = dict(response.get("usage") or {})
        self._pending_usage.append({
            "role": tag, "model": self.agent_model,
            "usage": usage, "latency_seconds": response.get("_latency_seconds"), "cached": False,
        })
        value = json.loads(response_text(response))
        temporary = cache_path.with_suffix(".json.part")
        temporary.write_text(json.dumps({"value": value}, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)
        return value

    def _safe_agent(self, system: str, user: str, tag: str) -> dict[str, Any]:
        try:
            return self._agent(system, user, tag)
        except Exception as error:
            self._agent_errors.append(f"{tag}:{error!r}")
            return {}

    def _budget(self, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        budget, output = self.max_context_chars, []
        for row in rows:
            if len(output) >= top_k:
                break
            copy = dict(row)
            inp = dict(copy.get("input") or {})
            text = str(inp.get("text") or "")[: self.max_passage_chars][:budget]
            if len(text.strip()) < 60:
                continue
            inp["text"] = text
            copy["input"] = inp
            output.append(copy)
            budget -= len(text)
            if budget <= 0:
                break
        return output

    @staticmethod
    def _merge_rankings(rankings: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Interleave per-query rankings so every planned query reaches the judge."""
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        depth = max((len(ranking) for ranking in rankings), default=0)
        for rank in range(depth):
            for ranking in rankings:
                if rank >= len(ranking):
                    continue
                row = ranking[rank]
                record_id = str(row.get("id"))
                if record_id not in seen:
                    seen.add(record_id)
                    merged.append(row)
        return merged

    def search(self, query: str, leaf: str, top_k: int, *, _write_trace: bool = True) -> list[dict[str, Any]]:
        self._agent_errors = []
        applicable = self.supports_leaf(leaf)
        if not applicable:
            if _write_trace:
                self._agentic_trace(query, leaf, [], 0, applicable=False)
            return []
        plan = self._safe_agent(
            self.PLAN_SYSTEM,
            f"Leaf: {leaf}\nQuestion: {query[:1500]}\nMaximum queries: {self.agent_subqueries}",
            "planner",
        )
        planned = [
            item.strip() for item in (plan.get("queries") or [])
            if isinstance(item, str) and item.strip()
        ][: self.agent_subqueries]
        queries = list(dict.fromkeys([query, *planned]))
        rankings: list[list[dict[str, Any]]] = []
        for subquery in queries:
            rankings.append(super().search(subquery, leaf, top_k + 5, _write_trace=False))
        ranked = self._merge_rankings(rankings)
        if not ranked:
            if _write_trace:
                self._agentic_trace(query, leaf, [], 0, applicable=True)
            return []

        final = ranked[:top_k]
        for hop in range(self.agent_max_hops):
            listing = "\n\n".join(
                f"[{index + 1}] {(row.get('input') or {}).get('title', 'Reference')}\n"
                f"{str((row.get('input') or {}).get('text') or '')[:700]}"
                for index, row in enumerate(ranked[: top_k + 5])
            )
            verdict = self._safe_agent(
                self.JUDGE_SYSTEM,
                f"Question: {query[:1500]}\n\nPassages:\n{listing}",
                "judge",
            )
            window = ranked[: top_k + 5]
            keep = [
                index - 1 for index in (verdict.get("keep") or [])
                if isinstance(index, int) and 1 <= index <= len(window)
            ]
            final = [window[index] for index in keep] or window[:top_k]
            if verdict.get("sufficient") or hop == self.agent_max_hops - 1:
                break
            followup = str(verdict.get("followup") or "").strip()
            if not followup:
                break
            rankings.append(super().search(followup, leaf, top_k + 5, _write_trace=False))
            ranked = self._merge_rankings(rankings)
        output = self._budget(final, top_k)
        if _write_trace:
            chars = sum(len(str((row.get("input") or {}).get("text") or "")) for row in output)
            self._agentic_trace(query, leaf, output, chars, applicable=True, subqueries=queries)
        return output

    def _agentic_trace(
        self, query: str, leaf: str, rows: list[dict[str, Any]], context_chars: int,
        *, applicable: bool, subqueries: list[str] | None = None,
    ) -> None:
        if not self.trace_path:
            return
        append_jsonl(self.trace_path, {
            "created_at": utc_now(), "query": query[:300], "leaf": leaf,
            "backend": "agentic_dense_rerank", "applicable": applicable,
            "subqueries": subqueries or [], "record_ids": [row.get("id") for row in rows],
            "context_chars": context_chars, "agent_model": self.agent_model,
            "agent_errors": self._agent_errors,
            "embedding_model": self.manifest.get("model"),
            "reranker_model": self.reranker_model,
        })
