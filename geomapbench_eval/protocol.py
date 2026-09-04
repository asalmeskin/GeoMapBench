from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .prompts import IMAGE_CONVERTER_REVISION, PROMPT_REVISION, RAG_PROMPT_REVISION
from .rag import MULTIMODAL_RETRIEVAL_REVISION
from .scoring import SCORING_REVISION
from .task_metrics import ARTIFACT_PROTOCOL_REVISION, TASK_METRIC_REVISION


EVALUATION_PROTOCOL_REVISION = "2026-09-final-task-aware-v1"

# Every protocol_descriptor() field except git_commit/git_dirty: those two
# record which commit produced a run for provenance, but they are not a
# semantic protocol change. A resumable cumulative output (base, rag-suite,
# geoagent-eval suite) must not refuse to resume just because an unrelated
# commit -- a docs fix, a performance fix, anything that doesn't touch a
# revision constant below -- was pulled between two invocations of the same
# output directory.
RESUME_COMPARABLE_PROTOCOL_FIELDS = (
    "evaluation_protocol_revision", "package_version", "prompt_revision",
    "rag_prompt_revision", "multimodal_retrieval_revision", "scoring_revision",
    "task_metric_revision", "artifact_protocol_revision", "image_converter_revision",
    "canonical_loader",
)


def protocol_matches_for_resume(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """True if two protocol descriptors are compatible for a cumulative resume.

    Ignores git_commit/git_dirty; compares every actual revision constant.
    """
    return all(
        previous.get(field) == current.get(field) for field in RESUME_COMPARABLE_PROTOCOL_FIELDS
    )


def protocol_descriptor() -> dict[str, Any]:
    """Fields that must match before Base and RAG results can be compared."""
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {
        "evaluation_protocol_revision": EVALUATION_PROTOCOL_REVISION,
        "package_version": __version__,
        "prompt_revision": PROMPT_REVISION,
        "rag_prompt_revision": RAG_PROMPT_REVISION,
        "multimodal_retrieval_revision": MULTIMODAL_RETRIEVAL_REVISION,
        "scoring_revision": SCORING_REVISION,
        "task_metric_revision": TASK_METRIC_REVISION,
        "artifact_protocol_revision": ARTIFACT_PROTOCOL_REVISION,
        "image_converter_revision": IMAGE_CONVERTER_REVISION,
        "canonical_loader": "23x100-v1",
        "git_commit": commit,
        "git_dirty": dirty,
    }
