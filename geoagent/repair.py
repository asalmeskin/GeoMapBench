"""Deterministic contract validation and repair of the agent's own answer.

The benchmark's strict metric compares an exact JSON structure, and the
task-aware metric collapses to zero whenever a run-length mask does not decode
or a numeric answer arrives as a string. A large share of the base model's lost
points are *format* losses, not reasoning losses.

Everything here operates on the agent's own output plus the task input. No gold
answer, tolerance or Bloom label is consulted. Every change is recorded in a
``repairs`` list so the notebook can report the pre-repair scores as an
ablation, using the untouched ``raw_response`` that the runner also stores.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from . import REPAIR_REVISION

MASK_CELLS = 4096
MASK_SIZE = [64, 64]

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_TRUE_FALSE = {
    "true": "yes", "false": "no", "correct": "yes", "incorrect": "no",
    "yes": "yes", "no": "no", "y": "yes", "n": "no", "valid": "yes", "invalid": "no",
}


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.strip().casefold())


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("−", "-")
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned)
        if match:
            try:
                result = float(match.group())
            except ValueError:
                return None
            return result if math.isfinite(result) else None
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_answer(text: str) -> tuple[Any, str | None]:
    """A more forgiving reader than the scorer's, used only to rebuild valid JSON."""
    candidate = _FENCE.sub("", str(text or "")).strip()
    for attempt in (candidate, _largest_object(candidate)):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        while isinstance(parsed, dict) and "answer" in parsed and len(parsed) == 1:
            parsed = parsed["answer"]
        return parsed, None
    return None, "invalid_json"


def _largest_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return ""
    return text[start : end + 1]


def render(answer: Any) -> str:
    return json.dumps({"answer": answer}, ensure_ascii=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Individual repairs
# ---------------------------------------------------------------------------


def repair_choices(answer: Any, choices: list[Any]) -> tuple[Any, str | None]:
    if not choices or isinstance(answer, (list, dict)):
        return answer, None
    normalized = {_norm(choice): choice for choice in choices}
    key = _norm(answer)
    if key in normalized:
        return (normalized[key] if normalized[key] != answer else answer), (
            None if normalized[key] == answer else "choice_exact_case"
        )
    # Closed yes/no sets accept common synonyms.
    if set(normalized) <= {"yes", "no"} and key in _TRUE_FALSE:
        return normalized[_TRUE_FALSE[key]], "choice_boolean_synonym"
    # A choice rendered as "P — populated place" may come back as either half.
    for choice_key, choice in normalized.items():
        parts = [part.strip() for part in re.split(r"[-–—:]", choice_key) if part.strip()]
        if key and (key in parts or key == choice_key.replace(" ", "")):
            return choice, "choice_partial_match"
    contained = [choice for choice_key, choice in normalized.items() if key and key in choice_key]
    if len(contained) == 1:
        return contained[0], "choice_substring_match"
    return answer, None


def repair_yes_no(answer: Any, question: str) -> tuple[Any, str | None]:
    if not re.search(r"answer yes or no", question or "", re.IGNORECASE):
        return answer, None
    if isinstance(answer, bool):
        return ("yes" if answer else "no"), "yes_no_from_boolean"
    if isinstance(answer, dict):
        for key in ("answer", "verdict", "value", "correct"):
            if key in answer:
                inner, _ = repair_yes_no(answer[key], question)
                if inner in {"yes", "no"}:
                    return inner, "yes_no_unwrapped"
        return answer, None
    key = _norm(answer)
    if key in _TRUE_FALSE and key not in {"yes", "no"}:
        return _TRUE_FALSE[key], "yes_no_synonym"
    if key not in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", key)
        if match:
            return match.group(1), "yes_no_extracted"
    return answer, None


def repair_numeric(answer: Any, question: str, evaluation_type: str) -> tuple[Any, str | None]:
    numeric_task = evaluation_type in {"numeric", "numeric_tolerance", "distance"}
    if not numeric_task or isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return answer, None
    if isinstance(answer, str):
        value = _num(answer)
        if value is not None:
            return value, "numeric_from_string"
    if isinstance(answer, dict) and len(answer) <= 3:
        for key in ("value", "answer", "result", "distance", "length"):
            if key in answer:
                value = _num(answer[key])
                if value is not None:
                    return value, "numeric_unwrapped"
    return answer, None


def repair_rle(answer: Any, valid_ids: list[Any] | None) -> tuple[Any, str | None]:
    """Make a run-length mask decodable: the scorer returns 0 for anything else."""
    payload = answer
    wrapper_key = None
    if isinstance(answer, dict) and isinstance(answer.get("mask"), dict):
        payload, wrapper_key = answer["mask"], "mask"
    if not isinstance(payload, dict) or "runs" not in payload:
        return answer, None
    runs_in = payload.get("runs")
    if not isinstance(runs_in, list) or not runs_in:
        return answer, None
    allowed = {int(item) for item in (valid_ids or []) if str(item).lstrip("-").isdigit()}
    runs: list[list[int]] = []
    changed = False
    for run in runs_in:
        if isinstance(run, dict):
            value, count = run.get("value", run.get("class_id")), run.get("count", run.get("length"))
            changed = True
        elif isinstance(run, (list, tuple)) and len(run) >= 2:
            value, count = run[0], run[1]
        else:
            changed = True
            continue
        value_n, count_n = _num(value), _num(count)
        if value_n is None or count_n is None or count_n <= 0:
            changed = True
            continue
        value_i, count_i = int(round(value_n)), int(round(count_n))
        if allowed and value_i not in allowed and value_i != 255:
            value_i = min(allowed, key=lambda item: abs(item - value_i))
            changed = True
        if runs and runs[-1][0] == value_i:
            runs[-1][1] += count_i
            changed = True
        else:
            runs.append([value_i, count_i])
    if not runs:
        return answer, None
    total = sum(count for _, count in runs)
    if total != MASK_CELLS:
        changed = True
        if total > MASK_CELLS:
            trimmed: list[list[int]] = []
            remaining = MASK_CELLS
            for value, count in runs:
                if remaining <= 0:
                    break
                take = min(count, remaining)
                trimmed.append([value, take])
                remaining -= take
            runs = trimmed
        else:
            runs[-1][1] += MASK_CELLS - total
    repaired = {
        "encoding": "rle-row-major",
        "size": list(MASK_SIZE),
        "runs": [[value, count] for value, count in runs],
    }
    if payload.get("size") != MASK_SIZE or payload.get("encoding") != "rle-row-major":
        changed = True
    if not changed:
        return answer, None
    if wrapper_key:
        rebuilt = dict(answer)
        rebuilt["mask"] = repaired
        return rebuilt, "rle_normalised"
    return repaired, "rle_normalised"


def repair_graph(answer: Any) -> tuple[Any, str | None]:
    if not isinstance(answer, dict) or "nodes" not in answer or "edges" not in answer:
        return answer, None
    nodes_in = answer.get("nodes")
    edges_in = answer.get("edges")
    if not isinstance(nodes_in, list) or not isinstance(edges_in, list):
        return answer, None
    nodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    changed = False
    for node in nodes_in:
        if not isinstance(node, dict):
            changed = True
            continue
        identifier, x, y = _num(node.get("id")), _num(node.get("x")), _num(node.get("y"))
        if identifier is None or x is None or y is None:
            changed = True
            continue
        key = int(identifier)
        if key in seen:
            changed = True
            continue
        seen.add(key)
        nodes.append({"id": key, "x": float(x), "y": float(y)})
    positions = {node["id"]: (node["x"], node["y"]) for node in nodes}
    edges: list[dict[str, Any]] = []
    for edge in edges_in:
        if not isinstance(edge, dict):
            changed = True
            continue
        source, target = _num(edge.get("source")), _num(edge.get("target"))
        if source is None or target is None:
            changed = True
            continue
        a, b = int(source), int(target)
        if a not in positions or b not in positions or a == b:
            changed = True
            continue
        length = _num(edge.get("length"))
        if length is None or length <= 0:
            (x1, y1), (x2, y2) = positions[a], positions[b]
            length = math.hypot(x1 - x2, y1 - y2)
            changed = True
        edges.append({"source": a, "target": b, "length": round(float(length), 3)})
    if not nodes or not changed:
        return answer, None
    rebuilt = dict(answer)
    rebuilt["nodes"] = nodes
    rebuilt["edges"] = edges
    rebuilt.setdefault("directed", False)
    return rebuilt, "graph_normalised"


def repair_spans(answer: Any, document: str) -> tuple[Any, str | None]:
    """Re-derive character offsets by locating each surface form in the document.

    Language models cannot count characters, and the span metric is strict, so
    the agent proposes the strings and the tool computes the offsets.
    """
    if not document or not isinstance(answer, list) or not answer:
        return answer, None
    if not all(isinstance(item, dict) for item in answer):
        return answer, None
    if not any("text" in item for item in answer):
        return answer, None
    cursor = 0
    used: list[tuple[int, int]] = []
    repaired: list[dict[str, Any]] = []
    changed = False
    for item in answer:
        surface = item.get("text")
        if not isinstance(surface, str) or not surface.strip():
            repaired.append(item)
            continue
        proposed = _num(item.get("start"))
        found = _locate(document, surface, cursor, used, int(proposed) if proposed is not None else None)
        entry = dict(item)
        if found is None:
            repaired.append(entry)
            continue
        start, end = found
        if entry.get("start") != start or entry.get("end") != end:
            changed = True
        entry["start"], entry["end"] = start, end
        entry["text"] = document[start:end]
        used.append((start, end))
        cursor = max(cursor, start + 1)
        repaired.append(entry)
    return (repaired, "spans_offset_aligned") if changed else (answer, None)


def _locate(
    document: str, surface: str, cursor: int, used: list[tuple[int, int]], proposed: int | None,
) -> tuple[int, int] | None:
    occurrences: list[int] = []
    start = document.find(surface)
    while start != -1 and len(occurrences) < 200:
        occurrences.append(start)
        start = document.find(surface, start + 1)
    if not occurrences:
        return None
    free = [
        position for position in occurrences
        if not any(position < end and position + len(surface) > begin for begin, end in used)
    ] or occurrences
    if proposed is not None:
        best = min(free, key=lambda position: (abs(position - proposed), position))
    else:
        forward = [position for position in free if position >= cursor]
        best = forward[0] if forward else free[0]
    return best, best + len(surface)


def repair_structure_keys(answer: Any, proposal: Any) -> tuple[Any, str | None]:
    """Adopt the tool's key names when the model produced the right values under
    different names and the same arity."""
    if not isinstance(answer, dict) or not isinstance(proposal, dict):
        return answer, None
    if set(answer) == set(proposal) or len(answer) != len(proposal):
        return answer, None
    unmatched = dict(answer)
    rebuilt: dict[str, Any] = {}
    for key, expected in proposal.items():
        match = next(
            (
                candidate for candidate, value in unmatched.items()
                if _close(value, expected) or _norm(value) == _norm(expected)
            ),
            None,
        )
        if match is None:
            return answer, None
        rebuilt[key] = unmatched.pop(match)
    if unmatched:
        return answer, None
    return rebuilt, "structure_keys_aligned"


def repair_tool_disagreement(
    answer: Any, proposal: Any, *, relative_tolerance: float = 1e-4,
) -> tuple[Any, str | None]:
    """Substitute a value the toolbelt derived in closed form from the task input."""
    if proposal is None:
        return answer, None
    if isinstance(proposal, (int, float)) and not isinstance(proposal, bool):
        current = _num(answer)
        if current is None:
            return proposal, "tool_value_substituted_unparsable"
        if abs(current - float(proposal)) > max(abs(float(proposal)) * relative_tolerance, 1e-9):
            return proposal, "tool_value_substituted_mismatch"
        return answer, None
    if isinstance(proposal, str):
        if _norm(answer) != _norm(proposal):
            return proposal, "tool_text_substituted"
        return answer, None
    if isinstance(proposal, dict):
        if not isinstance(answer, dict):
            return proposal, "tool_object_substituted"
        merged = dict(proposal)
        differing = [
            key for key in proposal
            if _norm(answer.get(key)) != _norm(proposal[key])
            and not _close(answer.get(key), proposal[key])
        ]
        extra = [key for key in answer if key not in proposal]
        if not differing and not extra:
            return answer, None
        return merged, "tool_object_substituted"
    if isinstance(proposal, list):
        if not isinstance(answer, list) or [
            _norm(item) for item in answer
        ] != [_norm(item) for item in proposal]:
            return proposal, "tool_list_substituted"
    return answer, None


def _close(first: Any, second: Any) -> bool:
    a, b = _num(first), _num(second)
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs(b) * 1e-6, 1e-9)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def repair_answer(
    raw_text: str,
    view: Any,
    *,
    exact_proposal: Any = None,
    ontology_ids: list[Any] | None = None,
) -> dict[str, Any]:
    """Return the repaired answer, the JSON to store, and the audit trail."""
    answer, parse_error = parse_answer(raw_text)
    actions: list[str] = []
    if parse_error:
        if exact_proposal is not None:
            # The toolbelt derived this value in closed form from the task input,
            # so an unreadable model response does not have to cost the record.
            return {
                "answer": exact_proposal,
                "text": render(exact_proposal),
                "repairs": ["recovered_unparsable_response_with_tool_value"],
                "parse_error": None,
                "original_parse_error": parse_error,
                "recovered": True,
                "revision": REPAIR_REVISION,
            }
        return {
            "answer": None, "text": str(raw_text), "repairs": [],
            "parse_error": parse_error, "recovered": False, "revision": REPAIR_REVISION,
        }

    question = view.question
    for repair in (
        lambda value: repair_yes_no(value, question),
        lambda value: repair_choices(value, view.choices),
        lambda value: repair_numeric(value, question, view.evaluation_type),
        lambda value: repair_rle(value, ontology_ids),
        repair_graph,
        lambda value: repair_spans(value, view.get("text") or ""),
        lambda value: repair_structure_keys(value, exact_proposal),
        lambda value: repair_tool_disagreement(value, exact_proposal),
    ):
        answer, action = repair(answer)
        if action:
            actions.append(action)

    # A closed set is authoritative; re-apply it after any substitution.
    if view.choices:
        answer, action = repair_choices(answer, view.choices)
        if action and action not in actions:
            actions.append(action)

    return {
        "answer": answer,
        "text": render(answer),
        "repairs": actions,
        "parse_error": None,
        "recovered": False,
        "revision": REPAIR_REVISION,
    }
