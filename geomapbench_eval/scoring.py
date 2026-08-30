from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any


def extract_answer(text: str) -> tuple[Any, str | None]:
    try:
        parsed = json.loads(text)
        return (parsed.get("answer") if isinstance(parsed, dict) and "answer" in parsed else parsed), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return (parsed.get("answer") if isinstance(parsed, dict) and "answer" in parsed else parsed), None
            except json.JSONDecodeError:
                pass
        return None, "invalid_json"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _canonical(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps({str(k): json.loads(_canonical(v)) for k, v in sorted(value.items())}, sort_keys=True, separators=(",", ":"))
    if isinstance(value, list):
        return json.dumps([json.loads(_canonical(v)) for v in value], separators=(",", ":"))
    return json.dumps(_norm(value))


def _answer_target(record: dict[str, Any]) -> Any:
    target = record.get("target") or {}
    path = str((record.get("evaluation") or {}).get("target_field") or "")
    if path.startswith("target."):
        return target.get(path.split(".", 1)[1])
    for key in ("bloom_answer", "answer", "value", "layer_id"):
        if key in target:
            return target[key]
    return target


def score(record: dict[str, Any], response_text: str) -> dict[str, Any]:
    prediction, parse_error = extract_answer(response_text)
    evaluation = record.get("evaluation") or {}
    kind = str(evaluation.get("type") or evaluation.get("metric") or "exact_match")
    gold = _answer_target(record)
    if parse_error:
        return {"score": 0.0, "metric": kind, "parse_error": parse_error, "gold_hash": _canonical(gold)}
    if kind in {"numeric_tolerance", "numeric", "distance"} or isinstance(gold, (int, float)):
        try:
            tolerance = float(evaluation.get("tolerance", evaluation.get("absolute_tolerance", 1e-6)))
            correct = math.isclose(float(prediction), float(gold), abs_tol=tolerance, rel_tol=float(evaluation.get("relative_tolerance", 0.0)))
        except (TypeError, ValueError):
            correct = False
    elif kind in {"set_f1", "relation_f1"} and isinstance(gold, list) and isinstance(prediction, list):
        g, p = Counter(map(_norm, gold)), Counter(map(_norm, prediction))
        overlap = sum((g & p).values())
        correct = 0.0 if not (sum(g.values()) + sum(p.values())) else 2 * overlap / (sum(g.values()) + sum(p.values()))
        return {"score": correct, "metric": kind, "parse_error": None, "gold_hash": _canonical(gold)}
    elif isinstance(gold, (dict, list)):
        correct = _canonical(prediction) == _canonical(gold)
    else:
        correct = _norm(prediction) == _norm(gold)
    return {"score": float(bool(correct)), "metric": kind, "parse_error": None, "gold_hash": _canonical(gold)}
