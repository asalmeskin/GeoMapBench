"""The single, audited channel between a benchmark record and the agent.

Nothing in ``geoagent`` may accept a raw benchmark record. Every stage takes a
:class:`TaskView`, which carries only the fields that the *base* condition
already puts in front of the model:

* ``record["input"]`` minus the binary asset references (the images are passed
  separately, exactly as ``geomapbench_eval.prompts.build_messages`` does),
* ``record["leaf"]`` -- the public task folder name, already used for retrieval
  routing by the v2.2 protocol,
* ``record["evaluation"]["type"]`` -- already consumed by the shared
  ``_answer_contract`` helper for both ``base`` and every RAG condition.

Gold answers (``target``), Bloom variant labels (``bloom``) and scoring
tolerances are never exposed. ``TaskView.from_record`` raises if they leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Mirrors geomapbench_eval.prompts. Duplicated deliberately so this module can
# be imported and unit-tested without the heavy image/embedding dependencies;
# ``assert_prompt_contract_matches`` checks the two definitions agree at runtime.
IMAGE_ASSET_KEYS = frozenset({
    "images", "image", "map_image", "graph_image", "route_image", "isochrone_image",
})
DOCUMENT_ASSET_KEYS = frozenset({"reference_graph"})

FORBIDDEN_RECORD_KEYS = ("target", "bloom", "base_evaluation", "evaluation")

MAX_PAYLOAD_FIELD_CHARS = 60_000


def assert_prompt_contract_matches() -> None:
    """Fail loudly if the upstream prompt builder changes its asset keys."""
    from geomapbench_eval import prompts

    if set(prompts.IMAGE_ASSET_KEYS) != set(IMAGE_ASSET_KEYS):
        raise RuntimeError(
            "geoagent.taskview.IMAGE_ASSET_KEYS drifted from geomapbench_eval.prompts"
        )
    if set(prompts.DOCUMENT_ASSET_KEYS) != set(DOCUMENT_ASSET_KEYS):
        raise RuntimeError(
            "geoagent.taskview.DOCUMENT_ASSET_KEYS drifted from geomapbench_eval.prompts"
        )


def is_asset_key(key: str) -> bool:
    return key in IMAGE_ASSET_KEYS or key in DOCUMENT_ASSET_KEYS or key.endswith("_image")


@dataclass(frozen=True)
class TaskView:
    """Everything the agent is allowed to know about one benchmark record."""

    record_id: str
    leaf: str
    question: str
    payload: dict[str, Any]
    choices: list[Any]
    evaluation_type: str
    image_count: int
    document_count: int
    asset_names: tuple[str, ...] = field(default=())

    @classmethod
    def from_record(cls, record: dict[str, Any], task_dir: Path) -> "TaskView":
        inp = record.get("input")
        if not isinstance(inp, dict):
            raise ValueError(f"{record.get('id')}: input must be an object")
        payload = {
            key: value for key, value in inp.items()
            if not is_asset_key(key)
        }
        for key in FORBIDDEN_RECORD_KEYS:
            if key in payload:
                raise RuntimeError(
                    f"{record.get('id')}: refusing to build a TaskView that exposes {key!r}"
                )
        assets: list[str] = []
        for key, value in inp.items():
            if not is_asset_key(key):
                continue
            if isinstance(value, str):
                assets.append(value)
            elif isinstance(value, list):
                assets.extend(str(item) for item in value)
        question = str(
            payload.get("question")
            or payload.get("base_question")
            or payload.get("text")
            or json.dumps(payload, sort_keys=True)[:2000]
        )
        choices = payload.get("choices")
        evaluation = record.get("evaluation") or {}
        return cls(
            record_id=str(record.get("id")),
            leaf=str(record.get("leaf") or "unknown"),
            question=question,
            payload=payload,
            choices=list(choices) if isinstance(choices, list) else [],
            evaluation_type=str(evaluation.get("type") or evaluation.get("metric") or "exact_match"),
            image_count=sum(1 for name in assets if not name.lower().endswith(".json")),
            document_count=sum(1 for name in assets if name.lower().endswith(".json")),
            asset_names=tuple(assets),
        )

    # -- convenience accessors -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def has(self, *keys: str) -> bool:
        return all(key in self.payload for key in keys)

    def text_fields(self) -> dict[str, Any]:
        """Payload with over-long free text trimmed, for planner/critic prompts."""
        trimmed: dict[str, Any] = {}
        for key, value in self.payload.items():
            if isinstance(value, str) and len(value) > 1200:
                trimmed[key] = value[:1200] + f"... [{len(value)} chars total]"
            else:
                trimmed[key] = value
        return trimmed

    def compact_json(self, limit: int = 2400) -> str:
        text = json.dumps(self.text_fields(), sort_keys=True, ensure_ascii=False)
        return text if len(text) <= limit else text[:limit] + "...}"

    def entity_names(self) -> list[str]:
        """Deterministic named entities taken straight from the structured input."""
        names: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                cleaned = value.strip()
                if 2 <= len(cleaned) <= 90 and cleaned not in names:
                    names.append(cleaned)
            elif isinstance(value, list):
                for item in value[:8]:
                    add(item)
            elif isinstance(value, dict):
                for key in ("name", "city", "country", "label", "mention"):
                    if key in value:
                        add(value[key])

        for key in (
            "country", "city", "entities", "mention", "reference_place",
            "labels", "points", "comparison_countries", "ranking_countries",
            "place", "region", "toponym", "indicator_name", "seed_city",
        ):
            if key in self.payload:
                add(self.payload[key])
        return names[:8]
