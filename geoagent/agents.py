"""The three cheap LLM roles: analyst, critic and verifier.

Each role is a strict-JSON call to a small model, cached on Drive with the same
two-phase scheme the v2.2 agent used: the raw response is written *before* it is
parsed, so a runtime disconnect can never turn a paid call into a duplicate.

A failed role degrades to a deterministic fallback; the pipeline never depends
on an agent call succeeding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from geomapbench_eval.common import atomic_json, stable_json
from geomapbench_eval.openrouter import (
    OpenRouterClient, OpenRouterConfig, finish_reason, generation_failure, response_text,
)

from . import AGENT_PROTOCOL_REVISION


class CachedAgent:
    """One cheap model, three roles, one disk cache."""

    def __init__(
        self,
        *,
        model: str,
        cache_root: Path,
        max_tokens: int = 1024,
        reasoning_effort: str = "minimal",
        reasoning_enabled: bool = True,
        request_delay_seconds: float = 0.4,
        timeout_seconds: int = 180,
        retries: int = 5,
    ):
        self.model = model
        self.cache_root = Path(cache_root).expanduser()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.config = OpenRouterConfig(
            model, temperature=0.0, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds, retries=retries,
            reasoning_effort=reasoning_effort, reasoning_enabled=reasoning_enabled,
            request_delay_seconds=request_delay_seconds,
        )
        self.client = OpenRouterClient()
        self.usage = self._empty_usage()

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {"cost": 0.0, "calls": 0, "cached_calls": 0, "failures": 0, "failure_kinds": []}

    def reset_usage(self) -> None:
        self.usage = self._empty_usage()

    def call(self, system: str, user: str, tag: str) -> dict[str, Any]:
        key = hashlib.sha256(
            stable_json({
                "revision": AGENT_PROTOCOL_REVISION,
                "model": self.model,
                "max_tokens": self.config.max_tokens,
                "reasoning_enabled": self.config.reasoning_enabled,
                "reasoning_effort": self.config.reasoning_effort,
                "tag": tag, "system": system, "user": user,
            }).encode()
        ).hexdigest()
        parsed_path = self.cache_root / f"{key}.json"
        raw_path = self.cache_root / f"{key}.response.json"
        if parsed_path.exists():
            try:
                cached = json.loads(parsed_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = {}
            if cached.get("status") == "ok" and isinstance(cached.get("value"), dict):
                self.usage["cached_calls"] += 1
                return dict(cached["value"])
        response = None
        recovered = False
        if raw_path.exists():
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            candidate = payload.get("response") if isinstance(payload, dict) else None
            if isinstance(candidate, dict):
                response, recovered = candidate, True
                self.usage["cached_calls"] += 1
            else:
                raw_path.unlink(missing_ok=True)
        if response is None:
            response = self.client.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.config,
            )
            atomic_json(raw_path, {"status": "received", "response": response})
            self.usage["calls"] += 1
        if not recovered:
            self.usage["cost"] += float((response.get("usage") or {}).get("cost") or 0.0)
        raw_text = response_text(response)
        failure = generation_failure(response, raw_text)
        try:
            value = json.loads(raw_text)
            if not isinstance(value, dict):
                failure = failure or "agent_non_object"
        except json.JSONDecodeError:
            value, failure = {}, failure or "agent_invalid_json"
        if failure:
            raw_path.unlink(missing_ok=True)
            self.usage["failures"] += 1
            self.usage["failure_kinds"].append(
                {"tag": tag, "kind": failure, "finish_reason": finish_reason(response)}
            )
            return {"_error": failure}
        atomic_json(parsed_path, {"status": "ok", "revision": AGENT_PROTOCOL_REVISION, "value": value})
        raw_path.unlink(missing_ok=True)
        return value


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = (
    "You prepare a geospatial benchmark task for a stronger answering model. You never "
    "answer the task. Return one JSON object only, with exactly these keys:\n"
    '{"answer_shape": <a JSON value using the string "?" wherever the answering model '
    'must fill in a value>, "shape_note": "<one short sentence naming the required keys '
    'or value type>", "queries": ["<up to 3 short retrieval queries>"], '
    '"pitfalls": ["<up to 2 short warnings>"]}\n'
    "answer_shape must copy the exact key names, ordering and nesting that the question "
    "asks for. If the question wants a single scalar, use \"?\" alone. If it wants a list, "
    "use [\"?\"]. Never invent keys the question does not ask for, and never include the "
    "answer itself. Retrieval queries should name concrete entities, places or indicators; "
    "return an empty list when external documents cannot possibly help."
)

CRITIC_SYSTEM = (
    "You audit optional retrieved evidence for a geospatial question. Return one JSON "
    'object only: {"keep": [<1-based indices worth keeping>], "use_context": <bool>, '
    '"sufficient": <bool>, "followup": "<one short extra query or empty string>", '
    '"note": "<one short sentence>"}\n'
    "Keep evidence only when it is clearly about the same entity, place, indicator or year "
    "as the question. Prefer abstention: set use_context=false and keep=[] when the evidence "
    "is merely topically similar, because irrelevant context makes the answer worse. Never "
    "answer the question."
)

VERIFIER_SYSTEM = (
    "You are a strict format and consistency checker for a geospatial benchmark answer. "
    "You see the question, the verified computations, and a proposed answer. Return one "
    'JSON object only: {"verdict": "accept" | "revise", "reason": "<short>", '
    '"answer": <the corrected answer value, present only when verdict is "revise">}\n'
    "Revise only for a concrete defect: the answer contradicts a verified computation, uses "
    "keys or a value type the question did not ask for, is outside a stated closed set, "
    "carries units or prose inside a numeric field, or is missing a requested field. Never "
    "revise because you would have guessed differently; when the answer depends on reading "
    "an image you cannot see, accept it."
)


def analyse(agent: CachedAgent, view: Any) -> dict[str, Any]:
    user = (
        f"Task family: {view.leaf}\n"
        f"Answer value family: {view.evaluation_type}\n"
        f"Question: {view.question[:1800]}\n"
        f"Structured task input: {view.compact_json(1800)}\n"
        f"Images supplied to the answering model: {view.image_count}"
    )
    result = agent.call(ANALYST_SYSTEM, user, "analyst")
    if result.get("_error"):
        return {"answer_shape": None, "shape_note": "", "queries": [], "pitfalls": [], "_error": result["_error"]}
    queries = [
        str(item).strip() for item in (result.get("queries") or [])
        if isinstance(item, str) and item.strip()
    ][:3]
    pitfalls = [
        str(item).strip() for item in (result.get("pitfalls") or [])
        if isinstance(item, str) and item.strip()
    ][:2]
    return {
        "answer_shape": result.get("answer_shape"),
        "shape_note": str(result.get("shape_note") or "")[:300],
        "queries": queries,
        "pitfalls": pitfalls,
    }


def criticise(agent: CachedAgent, view: Any, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    listing = "\n\n".join(
        f"[{index}] {str(item.get('input', {}).get('title') or 'Reference')}: "
        f"{str(item.get('input', {}).get('text') or '')[:700]}"
        for index, item in enumerate(contexts, 1)
    )
    user = (
        f"Question: {view.question[:1500]}\n"
        f"Structured task input: {view.compact_json(900)}\n\n"
        f"Candidate evidence:\n{listing}"
    )
    result = agent.call(CRITIC_SYSTEM, user, "critic")
    if result.get("_error"):
        return {"keep": list(range(1, len(contexts) + 1))[:3], "use_context": True,
                "sufficient": True, "followup": "", "note": "", "_error": result["_error"]}
    keep = [
        int(item) for item in (result.get("keep") or [])
        if isinstance(item, int) and 1 <= item <= len(contexts)
    ]
    return {
        "keep": keep,
        "use_context": bool(result.get("use_context", bool(keep))),
        "sufficient": bool(result.get("sufficient", True)),
        "followup": str(result.get("followup") or "").strip()[:200],
        "note": str(result.get("note") or "")[:200],
    }


def verify(
    agent: CachedAgent, view: Any, tool_text: str, answer: Any, *, attempt: int = 0,
) -> dict[str, Any]:
    user = (
        f"Question: {view.question[:1500]}\n"
        f"Structured task input: {view.compact_json(900)}\n"
        f"Closed answer set: {json.dumps(view.choices[:40], ensure_ascii=False) if view.choices else 'none'}\n"
        f"Verified computations:\n{tool_text[:1800] or 'none'}\n\n"
        f"Proposed answer JSON: {json.dumps(answer, ensure_ascii=False)[:1800]}"
    )
    result = agent.call(VERIFIER_SYSTEM, user, f"verifier-{attempt}")
    if result.get("_error"):
        return {"verdict": "accept", "reason": "verifier_unavailable", "_error": result["_error"]}
    verdict = str(result.get("verdict") or "accept").strip().lower()
    return {
        "verdict": "revise" if verdict == "revise" else "accept",
        "reason": str(result.get("reason") or "")[:300],
        "answer": result.get("answer"),
    }
