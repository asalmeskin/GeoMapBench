from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .prompts import IMAGE_CONVERTER_REVISION, PROMPT_REVISION
from .scoring import SCORING_REVISION
from .task_metrics import ARTIFACT_PROTOCOL_REVISION, TASK_METRIC_REVISION


EVALUATION_PROTOCOL_REVISION = "2026-09-final-task-aware-v1"


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
        "scoring_revision": SCORING_REVISION,
        "task_metric_revision": TASK_METRIC_REVISION,
        "artifact_protocol_revision": ARTIFACT_PROTOCOL_REVISION,
        "image_converter_revision": IMAGE_CONVERTER_REVISION,
        "canonical_loader": "23x100-v1",
        "git_commit": commit,
        "git_dirty": dirty,
    }
