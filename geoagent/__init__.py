"""GeoAgent v3: tool-augmented, self-verifying multimodal agentic RAG for GeoMapBench.

The package never reads ``record["target"]``, ``record["bloom"]`` or
``record["evaluation"]`` beyond the public ``type`` field that the shared
prompt builder already exposes to every condition, including ``base``.
``geoagent.taskview.TaskView`` is the only channel through which task
information reaches the agent, and it enforces that contract at runtime.
"""

__version__ = "3.0.0"

AGENT_SYSTEM_REVISION = "2026-09-geoagent-tool-augmented-v3"
AGENT_PROMPT_REVISION = "2026-09-geoagent-authoritative-evidence-v3"
AGENT_PROTOCOL_REVISION = "2026-09-geoagent-planner-critic-verifier-v3"
RETRIEVAL_REVISION = "2026-09-geoagent-hybrid-capability-rrf-mmr-v3"
REPAIR_REVISION = "2026-09-geoagent-contract-repair-v3"
