"""The per-record agent loop.

    TaskView -> toolbelt -> analyst -> hybrid retrieval -> critic -> answer
             -> deterministic repair -> verifier -> optional single revision

Every stage is optional and degrades safely: a failed agent call, an empty
retrieval or an unavailable tool leaves the remaining stages intact, so the
worst case is the v2.2 behaviour plus deterministic repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from geomapbench_eval.prompts import input_asset_paths

from . import repair as repair_module
from .agents import CachedAgent, analyse, criticise, verify
from .prompting import build_agent_messages, build_revision_messages
from .taskview import TaskView
from .tools import propose_answer, run_toolbelt

# Leaves where a cheap verifier cannot see what matters (the image) and would
# only add noise; the deterministic repairs still run everywhere.
VISION_ONLY_LEAVES = frozenset({
    "cartographic_symbol_recognition",
    "remote_sensing_scene_classification",
    "temporal_scene_matching",
    "environmental_layer_identification",
})


@dataclass
class AgentOutcome:
    answer: Any = None
    text: str = ""
    raw_text: str = ""
    repairs: list[str] = field(default_factory=list)
    stages: dict[str, Any] = field(default_factory=dict)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    revision_used: bool = False
    parse_error: str | None = None


class AgenticPipeline:
    def __init__(
        self,
        *,
        retriever: Any,
        agent: CachedAgent | None,
        structured_index: Any = None,
        top_k: int = 4,
        max_image_bytes: int = 8_000_000,
        use_analyst: bool = True,
        use_critic: bool = True,
        use_verifier: bool = True,
        allow_revision: bool = True,
        max_tool_chars: int = 2600,
    ):
        self.retriever = retriever
        self.agent = agent
        self.structured_index = structured_index
        self.top_k = top_k
        self.max_image_bytes = max_image_bytes
        self.use_analyst = use_analyst and agent is not None
        self.use_critic = use_critic and agent is not None
        self.use_verifier = use_verifier and agent is not None
        self.allow_revision = allow_revision
        self.max_tool_chars = max_tool_chars

    # -- helpers ---------------------------------------------------------------

    def _tool_blocks(self, results: list[Any]) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        budget = self.max_tool_chars
        for result in results:
            if not result.ok or not result.text:
                continue
            text = result.text[: max(0, budget)]
            if not text:
                break
            blocks.append({
                "title": result.title or result.name,
                "authority": (
                    "authoritative" if result.authority == "exact"
                    else "corpus evidence" if result.authority == "strong" else "approximate"
                ),
                "text": text,
            })
            budget -= len(text)
        return blocks

    @staticmethod
    def _ontology_ids(view: TaskView) -> list[Any]:
        ontology = view.get("class_ontology") or view.get("ontology")
        if isinstance(ontology, dict):
            return list(ontology.keys())
        return []

    # -- main entry point ------------------------------------------------------

    def solve(
        self,
        record: dict[str, Any],
        task_dir: Path,
        *,
        answer_fn: Callable[[list[dict[str, Any]], str], tuple[str, dict[str, Any]]],
    ) -> AgentOutcome:
        view = TaskView.from_record(record, task_dir)
        outcome = AgentOutcome()

        # 1. Deterministic toolbelt.
        tool_results = run_toolbelt(view, self.structured_index)
        outcome.tool_trace = [result.trace() for result in tool_results]
        proposal = propose_answer(view, tool_results)
        exact_proposal = proposal.value if proposal and proposal.confidence == "exact" else None
        outcome.stages["proposal"] = proposal.trace() if proposal else None

        # 2. Analyst: answer shape and retrieval queries.
        analysis: dict[str, Any] = {"answer_shape": None, "shape_note": "", "queries": [], "pitfalls": []}
        if self.use_analyst:
            analysis = analyse(self.agent, view)
        outcome.stages["analyst"] = {
            "shape_note": analysis.get("shape_note"),
            "queries": analysis.get("queries"),
            "error": analysis.get("_error"),
        }

        # 3. Hybrid retrieval.
        contexts: list[dict[str, Any]] = []
        retrieval_trace: dict[str, Any] = {}
        if self.retriever is not None:
            image_paths = input_asset_paths(record, task_dir) if view.image_count else []
            queries = [view.question] + list(analysis.get("queries") or []) + view.entity_names()[:2]
            contexts, retrieval_trace = self.retriever.search_evidence(
                view, image_paths=image_paths, queries=queries, top_k=self.top_k,
            )

        # 4. Critic: keep, abstain or ask one follow-up.
        critique: dict[str, Any] = {}
        if contexts and self.use_critic:
            critique = criticise(self.agent, view, contexts)
            if critique.get("use_context") is False:
                contexts = []
            elif critique.get("keep"):
                contexts = [contexts[index - 1] for index in critique["keep"]][: self.top_k]
            followup = critique.get("followup") or ""
            if contexts and followup and not critique.get("sufficient", True) and self.retriever is not None:
                extra, extra_trace = self.retriever.search_evidence(
                    view,
                    image_paths=input_asset_paths(record, task_dir) if view.image_count else [],
                    queries=[followup, view.question],
                    top_k=max(2, self.top_k - len(contexts)),
                )
                known = {item["id"] for item in contexts}
                contexts += [item for item in extra if item["id"] not in known]
                retrieval_trace["followup"] = {
                    "query": followup, "added": len(contexts) - len(known),
                    "record_ids": extra_trace.get("record_ids"),
                }
        retrieval_trace["critic"] = {
            key: critique.get(key) for key in ("use_context", "sufficient", "note", "_error")
        } if critique else None
        retrieval_trace["kept_context_count"] = len(contexts)
        outcome.contexts = contexts
        outcome.retrieval_trace = retrieval_trace

        # 5. Answer.
        blocks = self._tool_blocks(tool_results)
        messages = build_agent_messages(
            record, task_dir,
            tool_blocks=blocks, contexts=contexts,
            answer_shape=analysis.get("answer_shape"),
            shape_note=str(analysis.get("shape_note") or ""),
            pitfalls=list(analysis.get("pitfalls") or []),
            include_images=True, max_image_bytes=self.max_image_bytes,
        )
        raw_text, _ = answer_fn(messages, "answer")
        outcome.raw_text = raw_text

        # 6. Deterministic repair.
        repaired = repair_module.repair_answer(
            raw_text, view,
            exact_proposal=exact_proposal,
            ontology_ids=self._ontology_ids(view),
        )
        outcome.answer = repaired["answer"]
        outcome.text = repaired["text"]
        outcome.repairs = list(repaired["repairs"])
        outcome.parse_error = repaired["parse_error"]

        # 7. Verify, and revise at most once.
        if self.use_verifier and view.leaf not in VISION_ONLY_LEAVES:
            # A closed-form answer needs no reviewer: a cheap model can only make
            # a verified computation worse, and skipping it saves a call.
            needs_review = exact_proposal is None
            if needs_review:
                tool_text = "\n".join(f"{block['title']}: {block['text']}" for block in blocks)
                review = verify(self.agent, view, tool_text, outcome.answer)
                outcome.stages["verifier"] = {
                    "verdict": review.get("verdict"), "reason": review.get("reason"),
                    "error": review.get("_error"),
                }
                if review.get("verdict") == "revise":
                    candidate = review.get("answer")
                    if self.allow_revision:
                        revision_messages = build_revision_messages(
                            record, task_dir,
                            previous_answer=outcome.answer,
                            objection=str(review.get("reason") or "format defect"),
                            tool_blocks=blocks, contexts=contexts,
                            answer_shape=analysis.get("answer_shape"),
                            shape_note=str(analysis.get("shape_note") or ""),
                            include_images=True, max_image_bytes=self.max_image_bytes,
                        )
                        revised_raw, meta = answer_fn(revision_messages, "revision")
                        if meta.get("performed"):
                            outcome.revision_used = True
                            revised = repair_module.repair_answer(
                                revised_raw, view,
                                exact_proposal=exact_proposal,
                                ontology_ids=self._ontology_ids(view),
                            )
                            if revised["parse_error"] is None:
                                outcome.answer = revised["answer"]
                                outcome.text = revised["text"]
                                outcome.repairs += [f"revision:{item}" for item in revised["repairs"]]
                                outcome.repairs.append("answer_revised_after_review")
                                outcome.parse_error = None
                        elif candidate is not None and repaired["parse_error"] is not None:
                            outcome.answer = candidate
                            outcome.text = repair_module.render(candidate)
                            outcome.repairs.append("verifier_answer_used_after_parse_failure")
                            outcome.parse_error = None
                    elif candidate is not None and repaired["parse_error"] is not None:
                        outcome.answer = candidate
                        outcome.text = repair_module.render(candidate)
                        outcome.repairs.append("verifier_answer_used_after_parse_failure")
                        outcome.parse_error = None

        # A last guarantee: never store an unparsable response when a tool knew
        # the answer outright.
        if outcome.parse_error is not None and exact_proposal is not None:
            outcome.answer = exact_proposal
            outcome.text = repair_module.render(exact_proposal)
            outcome.repairs.append("tool_value_used_after_parse_failure")
            outcome.parse_error = None

        outcome.stages["tool_names"] = [
            result.name for result in tool_results if result.ok
        ]
        outcome.stages["repairs"] = list(outcome.repairs)
        return outcome


def stage_summary(outcome: AgentOutcome) -> dict[str, Any]:
    return {
        "tools": outcome.stages.get("tool_names", []),
        "proposal": outcome.stages.get("proposal"),
        "analyst": outcome.stages.get("analyst"),
        "verifier": outcome.stages.get("verifier"),
        "repairs": outcome.repairs,
        "revision_used": outcome.revision_used,
        "context_count": len(outcome.contexts),
    }
