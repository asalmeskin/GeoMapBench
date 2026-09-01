from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl

_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [item.lower() for item in _TOKEN.findall(text) if len(item) > 1]


class LexicalRetriever:
    """Deterministic BM25 retriever; safe fallback when embedding dependencies are unavailable."""
    def __init__(self, corpus_root: Path):
        path = corpus_root / "corpus_clean.jsonl"
        if not path.exists():
            path = corpus_root / "corpus.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus.jsonl or corpus_clean.jsonl under {corpus_root}")
        self.records = read_jsonl(path)
        self.docs = [tokens(str((row.get("input") or {}).get("title", "")) + " " + str((row.get("input") or {}).get("text", ""))) for row in self.records]
        self.lengths = [len(doc) for doc in self.docs]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df: Counter[str] = Counter(token for doc in self.docs for token in set(doc))

    def search(self, query: str, leaf: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = set(tokens(query))
        scored: list[tuple[float, str, dict[str, Any]]] = []
        n = len(self.records)
        for record, doc, length in zip(self.records, self.docs, self.lengths):
            term_counts = Counter(doc)
            score = 0.0
            for term in query_terms:
                if not term_counts[term]:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * term_counts[term] * 2.2 / (term_counts[term] + 1.2 * (1 - 0.75 + 0.75 * length / max(1, self.average_length)))
            capabilities = set((record.get("retrieval") or {}).get("capabilities") or [])
            if leaf in capabilities:
                score += 0.15
            if score:
                scored.append((score, str(record.get("id", "")), record))
        return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]
