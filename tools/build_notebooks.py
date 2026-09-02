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
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
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
    markdown("""# GeoMapBench v1.9.1 — resilient six-company evaluation

This orchestration-only notebook clones a frozen release and runs six multimodal models from six independent companies. All image conversion, scoring, long retry/backoff, transport circuit breaking, write-ahead API response caching, exact resume logic, completeness checks, and reporting live in the codebase.

Use `PER_LEAF_LIMIT = 1` for a smoke test, then `5` for the 115-record stability pilot. Only models with zero generation failures and near-zero invalid JSON should be run with `None` for the full 2,300 records."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v1.9.1"              # create/push this exact tag after uploading this release

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

PER_LEAF_LIMIT = 1              # 1 smoke -> 5 stability pilot -> None full
MAX_COST_USD_PER_MODEL = 50.0
FORCE_PREFLIGHT = False          # True only if benchmark assets changed

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
run_live([
    "git", "clone", "--depth", "1", "--branch", GIT_REF,
    GITHUB_REPO, str(repo),
])
run_live([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images_v191")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed release:", subprocess.check_output(
    [sys.executable, "-c", "import geomapbench_eval; print(geomapbench_eval.__version__)"],
    text=True,
).strip())
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs are merged from stdout/stderr. Rerun this same cell after any pause/disconnect.
from pathlib import Path
import sys

models = repo / "config/evaluation_models_2026-09-v6.json"
scope = "full" if PER_LEAF_LIMIT is None else f"{PER_LEAF_LIMIT}_per_leaf"
output = Path(RESULTS_ROOT) / f"model_suite_{scope}_v6"

command = [
    sys.executable, "-u", "-m", "geomapbench_eval", "suite",
    "--benchmark-root", BENCHMARK_ROOT,
    "--models", str(models),
    "--output", str(output),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight_v191"),
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_MODEL),
    "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
    "--progress-every", str(PROGRESS_EVERY),
    "--timeout-seconds", "240",
    "--retries", "6",
    "--retry-base-seconds", "5",
    "--retry-max-seconds", "60",
    "--max-consecutive-errors", "2",
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]

run_live(command, cwd=repo)
print("Run directory:", output)
"""),
    code("""# Compact monitoring/report table. Incomplete runs show only provisional accuracy.
import json, pandas as pd

summary_path = output / "model_suite_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
display(pd.DataFrame(summary["models"]))
if summary["failures"]:
    print("Catalog/config/model-level skips:")
    display(pd.DataFrame(summary["failures"]))

print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
print("\\nPer-model live state files:")
for path in sorted(output.glob("*/run_state.json")):
    state = json.loads(path.read_text(encoding="utf-8"))
    print(path.parent.name, "->", state)
"""),
    markdown("""## Resume and full-run rules

- A reconnect requires rerunning setup/API-key cells and then the run cell. Completed record IDs are skipped.
- `api_responses.jsonl` is a write-ahead response cache; a response received just before interruption is recovered without another model call.
- `run_state.json` and `inflight.json` show the current record and stage in Drive.
- A persistent 429 opens the circuit breaker and pauses that model instead of producing hundreds of errors. Wait for provider recovery and rerun the same cell.
- A partial model has `complete=false`; its provisional accuracy is never promoted to final `macro_accuracy`.
- After the 115-record stability pilot passes, set `PER_LEAF_LIMIT=None`. The output directory changes automatically."""),
])


RAG = notebook([
    markdown("""# GeoMapBench v1.9.1 — dense Base RAG and Agentic RAG

This independent notebook runs only `base_rag` and `agentic_rag`. BM25 is absent. The answer model is Mistral Small 4 because the observed pilot showed zero generation/JSON failures at the lowest stable cost. Agent planning/judging uses Gemini 3.5 Flash-Lite with minimal mandatory reasoning.

The notebook does not require the Evaluation notebook and can run simultaneously on another computer. Exact manifests and selected-ID hashes enforce fairness later."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v1.9.1"

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
CORPUS_ROOT = "/content/drive/MyDrive/GeoMapRAG_Corpus"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

ANSWER_MODEL = "mistralai/mistral-small-2603"
AGENT_MODEL = "google/gemini-3.5-flash-lite"
CONDITIONS = "base_rag,agentic_rag"

PER_LEAF_LIMIT = 1              # 1 smoke -> 5 stability pilot -> None full
MAX_COST_USD_PER_CONDITION = 25.0
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
run_live([
    "git", "clone", "--depth", "1", "--branch", GIT_REF,
    GITHUB_REPO, str(repo),
])
run_live([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[rag-index]"])

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
assert Path(CORPUS_ROOT).is_dir(), f"RAG corpus not found: {CORPUS_ROOT}"
for name in ("indexes/text.faiss", "indexes/text_metadata.jsonl"):
    assert (Path(CORPUS_ROOT) / name).is_file(), f"Missing dense retrieval artifact: {name}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images_v191")
os.environ["HF_HOME"] = str(Path(CACHE_ROOT) / "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(Path(CACHE_ROOT) / "huggingface" / "sentence_transformers")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed release:", subprocess.check_output(
    [sys.executable, "-c", "import geomapbench_eval; print(geomapbench_eval.__version__)"],
    text=True,
).strip())
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs; rerun this cell after a disconnect or circuit-breaker pause.
from pathlib import Path
import sys

scope = "full" if PER_LEAF_LIMIT is None else f"{PER_LEAF_LIMIT}_per_leaf"
output = Path(RESULTS_ROOT) / f"rag_suite_{scope}_v6"
work_root = Path("/content/geomaprag_work_v191")
models = repo / "config/evaluation_models_2026-09-v6.json"

command = [
    sys.executable, "-u", "-m", "geomapbench_eval", "rag-suite",
    "--benchmark-root", BENCHMARK_ROOT,
    "--corpus-root", CORPUS_ROOT,
    "--work-root", str(work_root),
    "--output", str(output),
    "--models", str(models),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight_v191"),
    "--agent-cache", str(Path(CACHE_ROOT) / "agent_cache_v191"),
    "--model", ANSWER_MODEL,
    "--agent-model", AGENT_MODEL,
    "--agent-reasoning-effort", "minimal",
    "--conditions", CONDITIONS,
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_CONDITION),
    "--top-k", "5",
    "--candidate-k", "40",
    "--max-passage-chars", "1500",
    "--max-context-chars", "6000",
    "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
    "--progress-every", str(PROGRESS_EVERY),
    "--timeout-seconds", "240",
    "--retries", "6",
    "--retry-base-seconds", "5",
    "--retry-max-seconds", "60",
    "--max-consecutive-errors", "2",
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]

run_live(command, cwd=repo)
print("Run directory:", output)
"""),
    code("""# Paired result and reliability summary
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
        **stats,
    })
display(pd.DataFrame(rows))

if summary["comparisons"]:
    display(pd.DataFrame([
        {"comparison": name, **comparison}
        for name, comparison in summary["comparisons"].items()
    ]))
print("Preflight cache hit:", summary["preflight"].get("cache_hit"))

for path in sorted(output.glob("*/run_state.json")):
    print(path.parent.name, "->", json.loads(path.read_text(encoding="utf-8")))
"""),
    markdown("""## Resume and fairness

- The answer-response write-ahead cache, agent cache, Hugging Face cache, preflight cache, responses, traces, and live state all survive runtime restarts where appropriate.
- Invalid agent outputs are not cached; agent failures are logged and fall back deterministically to dense retrieval.
- Comparison is deferred until both RAG conditions contain the exact same target IDs.
- Run the smoke test, then the 115-record stability pilot, then set `PER_LEAF_LIMIT=None`.
- The separate Evaluation run for Mistral can later be compared with these RAG conditions because both use the same tag, benchmark hash, answer-model configuration, prompt, scoring, and selected subset."""),
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
