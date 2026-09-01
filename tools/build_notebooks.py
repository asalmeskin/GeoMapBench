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


EVALUATION = notebook([
    markdown("""# GeoMapBench — publication model evaluation

This notebook contains orchestration only. The evaluator, image conversion, scoring, resume logic, preflight cache, and reporting all live in the cloned codebase.

Start with `PER_LEAF_LIMIT = 1`. Only after the pilot has zero generation failures and a near-zero invalid-JSON rate should you set it to `None` for all 2,300 examples per model."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v1.8.2"               # push this exact release tag before running

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

PER_LEAF_LIMIT = 1               # pilot: 1; full publication run: None
MAX_COST_USD_PER_MODEL = 25.0
MAX_TOKENS = 8192
FORCE_PREFLIGHT = False           # True only when benchmark assets changed
"""),
    code("""from google.colab import drive
drive.mount("/content/drive")

import os, shutil, subprocess, sys
from pathlib import Path

repo = Path("/content/GeoMapBench")
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", GIT_REF, GITHUB_REPO, str(repo)],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)], check=True)

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed:", repo)
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs are streamed directly into the cell.
from pathlib import Path
import subprocess

models = repo / "config/evaluation_models_2026-09-v4.json"
run_name = "model_suite_full_v4" if PER_LEAF_LIMIT is None else f"model_suite_{PER_LEAF_LIMIT}_per_leaf_v4"
output = Path(RESULTS_ROOT) / run_name

command = [
    "geomapbench-eval", "suite",
    "--benchmark-root", BENCHMARK_ROOT,
    "--models", str(models),
    "--output", str(output),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight"),
    "--max-tokens", str(MAX_TOKENS),
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_MODEL),
    "--progress-every", "5",
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]

print("Running:", " ".join(command), flush=True)
completed = subprocess.run(command, cwd=repo)
if completed.returncode:
    raise RuntimeError(f"GeoMapBench suite failed with exit code {completed.returncode}")
print("Saved to:", output)
"""),
    code("""# Compact result table
import json, pandas as pd
summary_path = output / "model_suite_summary.json"
summary = json.loads(summary_path.read_text())
display(pd.DataFrame(summary["models"]))
if summary["failures"]:
    print("Model-level skips/failures:")
    display(pd.DataFrame(summary["failures"]))
print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
"""),
    markdown("""## Moving from pilot to full run

1. Confirm `generation_failure_rate == 0` for every retained model.
2. Confirm invalid JSON is acceptably close to zero.
3. Change `PER_LEAF_LIMIT` to `None` in the first cell and rerun from the command cell.

The full run uses a different output directory. Completed full-run records are append-safe and skipped after a Colab reconnect."""),
])


RAG = notebook([
    markdown("""# GeoMapBench — dense Base RAG and Agentic RAG

This notebook runs only two retrieval conditions: dense retrieval plus reranking (`base_rag`) and agent-planned dense retrieval (`agentic_rag`). BM25 is not used anywhere in this pipeline.

It is fully independent of the Evaluation notebook and can run concurrently on another computer. Fairness is enforced later by protocol-locked comparison of the saved run manifests and exact record IDs."""),
    code("""# EDIT ONLY THIS CELL
GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "v1.8.2"

BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
CORPUS_ROOT = "/content/drive/MyDrive/GeoMapRAG_Corpus"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

ANSWER_MODEL = "qwen/qwen3.8-flash"
AGENT_MODEL = "google/gemini-3.5-flash-lite"
CONDITIONS = "base_rag,agentic_rag"

PER_LEAF_LIMIT = 1               # pilot first; None for the final paired run
MAX_COST_USD_PER_CONDITION = 25.0
FORCE_PREFLIGHT = False
"""),
    code("""from google.colab import drive
drive.mount("/content/drive")

import os, shutil, subprocess, sys
from pathlib import Path

repo = Path("/content/GeoMapBench")
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", GIT_REF, GITHUB_REPO, str(repo)],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[rag-index]"],
    check=True,
)

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
assert Path(CORPUS_ROOT).is_dir(), f"RAG corpus not found: {CORPUS_ROOT}"
for name in ("indexes/text.faiss", "indexes/text_metadata.jsonl"):
    assert (Path(CORPUS_ROOT) / name).is_file(), f"Missing dense retrieval artifact: {name}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images")
os.environ["PYTHONUNBUFFERED"] = "1"
print("Installed:", repo)
"""),
    code("""import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
print("API key loaded in this runtime only.")
"""),
    code("""# Live logs; corpus/index staging and all retrieval happen inside the codebase.
from pathlib import Path
import subprocess

run_name = "rag_suite_full_v4" if PER_LEAF_LIMIT is None else f"rag_suite_{PER_LEAF_LIMIT}_per_leaf_v4"
output = Path(RESULTS_ROOT) / run_name
work_root = Path("/content/geomaprag_work")
models = repo / "config/evaluation_models_2026-09-v4.json"

command = [
    "geomapbench-eval", "rag-suite",
    "--benchmark-root", BENCHMARK_ROOT,
    "--corpus-root", CORPUS_ROOT,
    "--work-root", str(work_root),
    "--output", str(output),
    "--models", str(models),
    "--preflight-cache", str(Path(CACHE_ROOT) / "preflight"),
    "--agent-cache", str(Path(CACHE_ROOT) / "agent_cache"),
    "--model", ANSWER_MODEL,
    "--agent-model", AGENT_MODEL,
    "--conditions", CONDITIONS,
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_CONDITION),
    "--top-k", "5",
    "--candidate-k", "40",
    "--progress-every", "5",
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
if FORCE_PREFLIGHT:
    command += ["--force-preflight"]

print("Running:", " ".join(command), flush=True)
completed = subprocess.run(command, cwd=repo)
if completed.returncode:
    raise RuntimeError(f"GeoMapBench RAG suite failed with exit code {completed.returncode}")
print("Saved to:", output)
"""),
    code("""# Paired result summary
import json, pandas as pd
summary = json.loads((output / "rag_suite_summary.json").read_text())
rows = []
for condition, report in summary["reports"].items():
    stats = report["analysis"]["condition_summary"][condition]
    rows.append({"condition": condition, "macro_accuracy": report["analysis"]["macro_by_condition"][condition], **stats})
display(pd.DataFrame(rows))

if summary["comparisons"]:
    display(pd.DataFrame([
        {"condition": condition, **comparison}
        for condition, comparison in summary["comparisons"].items()
    ]))
print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
"""),
    markdown("""## Full paired run

This notebook does not need any Evaluation output. After its own pilot has no generation failures, set `PER_LEAF_LIMIT = None` and run it independently. Rerunning after a disconnect skips completed RAG IDs.

When both people finish, copy the desired Evaluation `responses.jsonl` together with its sibling `run_config.json` next to the RAG results and use `geomapbench-eval compare`. The comparison command refuses mismatched model settings, protocol revisions, benchmark hashes, subset definitions, or completed ID sets."""),
])


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "GeoMapBench_Evaluation.ipynb": EVALUATION,
        "GeoMapBench_RAG_Final.ipynb": RAG,
    }
    for name, value in outputs.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
