from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


LIVE_RUNNER = '''def run_live(command, *, cwd=None):
    import os, subprocess
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        list(map(str, command)), cwd=str(cwd) if cwd else None, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Command failed with exit code {return_code}")
'''


EVALUATION = notebook([
    markdown("""# GeoMapBench v2.0.0 — cumulative five-model evaluation

This notebook uses one permanent output directory. `TARGET_PER_LEAF` is the **cumulative total**, not the number of fresh calls. After importing a valid one-per-leaf run, target `6` makes at most five new calls per leaf. The deterministic nested cohort preserves the original first example, then balances Bloom levels before adding repeated levels.

Compatible rows from the old v6 one-per-leaf and partial six-per-leaf folders are imported only after checking the model, condition, selected ID, and exact prompt hash. Muse is removed; the benchmark now contains five model families."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v2.0.0"             # push this exact tag before running

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

TARGET_PER_LEAF = 6             # cumulative: 1 -> 6 -> 10 -> 100 (full)
MAX_COST_USD_PER_MODEL = 50.0
FORCE_PREFLIGHT = False
REQUEST_DELAY_SECONDS = 1.0
PROGRESS_EVERY = 5
"""),
    code(LIVE_RUNNER + """
from google.colab import drive
drive.mount("/content/drive")

import os, shutil, subprocess, sys
from pathlib import Path

repo = Path("/content/GeoMapBench")
if repo.exists():
    shutil.rmtree(repo)
run_live(["git", "clone", "--depth", "1", "--branch", GIT_REF, GITHUB_REPO, str(repo)])
run_live([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images_v191")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed release:", subprocess.check_output(
    [sys.executable, "-c", "import geomapbench_eval; print(geomapbench_eval.__version__)"], text=True,
).strip())
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs are merged from stdout/stderr. Rerun this same cell to resume.
from pathlib import Path
import sys

models = repo / "config/evaluation_models_2026-09-v7.json"
output = Path(RESULTS_ROOT) / "model_suite_cumulative_v7"
legacy_candidates = [
    Path(RESULTS_ROOT) / "model_suite_1_per_leaf_v6",
    Path(RESULTS_ROOT) / "model_suite_6_per_leaf_v6",
]

command = [
    sys.executable, "-u", "-m", "geomapbench_eval", "suite",
    "--benchmark-root", BENCHMARK_ROOT, "--models", str(models),
    "--output", str(output), "--target-per-leaf", str(TARGET_PER_LEAF),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight_v191"),
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_MODEL),
    "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
    "--progress-every", str(PROGRESS_EVERY), "--timeout-seconds", "240",
    "--retries", "6", "--retry-base-seconds", "5", "--retry-max-seconds", "60",
    "--max-consecutive-errors", "2",
]
for legacy in legacy_candidates:
    if legacy.is_dir() and legacy.resolve() != output.resolve():
        command += ["--legacy-output", str(legacy)]
        print("Will safely migrate compatible rows from:", legacy)
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]
run_live(command, cwd=repo)
print("Cumulative run directory:", output)
"""),
    code("""# Compact monitoring/report table and migration audit.
import json, pandas as pd

summary = json.loads((output / "model_suite_summary.json").read_text(encoding="utf-8"))
display(pd.DataFrame(summary["models"]))
if summary["failures"]:
    display(pd.DataFrame(summary["failures"]))
print("Cohort:", {k: summary["cohort"][k] for k in (
    "target_per_leaf", "target_record_count", "selected_ids_hash"
)})
print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
if summary.get("migration"):
    display(pd.DataFrame(summary["migration"]))
for path in sorted(output.glob("*/run_state.json")):
    print(path.parent.name, "->", json.loads(path.read_text(encoding="utf-8")))
"""),
    markdown("""## Operating rules

- Keep `model_suite_cumulative_v7`. Do not rename, delete, or manually merge its JSONL files.
- Increase only `TARGET_PER_LEAF`: `6`, optionally `10`, then `100` for all 2,300 records. Decreasing is intentionally rejected.
- Rerunning is safe: completed results and write-ahead API responses are reused; interrupted/error records retry.
- Paid partial-v6 responses outside today's Bloom cohort remain available as migration sources and import automatically when a later target includes them.
- Cohort manifests, migration report, run state, and inflight state make selection and progress auditable."""),
])


RAG = notebook([
    markdown("""# GeoMapBench v2.0.0 — cumulative Claude RAG evaluation

This independent notebook runs only `base_rag` and `agentic_rag`; BM25 is absent. Claude Sonnet 5 is the answer model for both conditions. Gemini Flash-Lite is used only as the planner/judge in `agentic_rag`.

It may run simultaneously with Evaluation on another computer. Both notebooks independently compute the same deterministic Bloom-stratified cohort. Old Mistral answer results are intentionally not imported because changing the answer model changes the experiment."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v2.0.0"

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
CORPUS_ROOT = "/content/drive/MyDrive/GeoMapRAG_Corpus"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

ANSWER_MODEL = "anthropic/claude-sonnet-5"
AGENT_MODEL = "google/gemini-3.5-flash-lite"
CONDITIONS = "base_rag,agentic_rag"
TARGET_PER_LEAF = 6             # cumulative: 1 -> 6 -> 10 -> 100 (full)
MAX_COST_USD_PER_CONDITION = 50.0
FORCE_PREFLIGHT = False
REQUEST_DELAY_SECONDS = 1.0
PROGRESS_EVERY = 5
"""),
    code(LIVE_RUNNER + """
from google.colab import drive
drive.mount("/content/drive")

import os, shutil, subprocess, sys
from pathlib import Path

repo = Path("/content/GeoMapBench")
if repo.exists():
    shutil.rmtree(repo)
run_live(["git", "clone", "--depth", "1", "--branch", GIT_REF, GITHUB_REPO, str(repo)])
run_live([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[rag-index]"])

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
assert Path(CORPUS_ROOT).is_dir(), f"RAG corpus not found: {CORPUS_ROOT}"
for name in ("indexes/text.faiss", "indexes/text_metadata.jsonl"):
    assert (Path(CORPUS_ROOT) / name).is_file(), f"Missing dense artifact: {name}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images_v191")
os.environ["HF_HOME"] = str(Path(CACHE_ROOT) / "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(Path(CACHE_ROOT) / "huggingface" / "sentence_transformers")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed release:", subprocess.check_output(
    [sys.executable, "-c", "import geomapbench_eval; print(geomapbench_eval.__version__)"], text=True,
).strip())
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs; rerun this cell after disconnect or circuit-breaker pause.
from pathlib import Path
import sys

output = Path(RESULTS_ROOT) / "rag_suite_cumulative_v7"
work_root = Path("/content/geomaprag_work_v200")
models = repo / "config/evaluation_models_2026-09-v7.json"
command = [
    sys.executable, "-u", "-m", "geomapbench_eval", "rag-suite",
    "--benchmark-root", BENCHMARK_ROOT, "--corpus-root", CORPUS_ROOT,
    "--work-root", str(work_root), "--output", str(output),
    "--target-per-leaf", str(TARGET_PER_LEAF), "--models", str(models),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight_v191"),
    "--agent-cache", str(Path(CACHE_ROOT) / "agent_cache_v191"),
    "--model", ANSWER_MODEL, "--agent-model", AGENT_MODEL,
    "--agent-reasoning-effort", "minimal", "--conditions", CONDITIONS,
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_CONDITION),
    "--top-k", "5", "--candidate-k", "40",
    "--max-passage-chars", "1500", "--max-context-chars", "6000",
    "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
    "--progress-every", str(PROGRESS_EVERY), "--timeout-seconds", "240",
    "--retries", "6", "--retry-base-seconds", "5", "--retry-max-seconds", "60",
    "--max-consecutive-errors", "2",
]
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]
run_live(command, cwd=repo)
print("Cumulative RAG directory:", output)
"""),
    code("""# Paired result, reliability, cost, and live-state summary.
import json, pandas as pd

summary = json.loads((output / "rag_suite_summary.json").read_text(encoding="utf-8"))
rows = []
for condition, report in summary["reports"].items():
    analysis = report["analysis"]
    stats = analysis["condition_summary"].get(condition, {})
    rows.append({
        "condition": condition,
        "complete": report["run"].get("complete"),
        "completed_total": report["run"].get("completed_total"),
        "target_records": report["run"].get("target_records"),
        "macro_accuracy": analysis["macro_by_condition"].get(condition),
        "text_answer_macro_accuracy": analysis["text_answer_macro_by_condition"].get(condition),
        "answer_cost_usd": report["run"].get("answer_cost_usd_total"),
        "agent_cost_usd": report["run"].get("agent_cost_usd_total"),
        **stats,
    })
display(pd.DataFrame(rows))
if summary["comparisons"]:
    display(pd.DataFrame([
        {"comparison": name, **value} for name, value in summary["comparisons"].items()
    ]))
print("Cohort:", {k: summary["cohort"][k] for k in (
    "target_per_leaf", "target_record_count", "selected_ids_hash"
)})
print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
for path in sorted(output.glob("*/run_state.json")):
    print(path.parent.name, "->", json.loads(path.read_text(encoding="utf-8")))
"""),
    markdown("""## Resume and fairness

- Keep `rag_suite_cumulative_v7` and only increase `TARGET_PER_LEAF`.
- Both RAG conditions use identical Claude parameters, selected IDs, benchmark hash, prompt/scoring protocol, and target. Only retrieval orchestration differs.
- Evaluation is not a prerequisite. Its Claude base condition is later comparable because both notebooks independently create the same cohort.
- Answer and agent write-ahead caches, model caches, preflight cache, responses, traces, and live-state files support restart/resume.
- Old Mistral answer rows are incompatible with Claude and are not migrated. Compatible downloads and planner/judge cache entries remain reusable.
- Target `100` is the complete 2,300-record experiment."""),
])


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "GeoMapBench_Evaluation.ipynb": EVALUATION,
        "GeoMapBench_RAG_Final.ipynb": RAG,
    }.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
