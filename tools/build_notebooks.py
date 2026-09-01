from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_CONFIG = '''# Edit only this cell, then use Runtime -> Run all.
REPO_URL = "https://github.com/asalmeskin/GeoMapBench.git"
GIT_REF = "main"  # For a paper, replace with the final v1.7.0 tag or commit SHA.

BENCHMARK_ROOT = "/content/drive/MyDrive/geomapbench_100"
RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"

# None = full 100/leaf. Use 1 for the mandatory 23-call pilot, then switch to None.
PER_LEAF_LIMIT = None
'''


BOOTSTRAP = '''from google.colab import drive
drive.mount("/content/drive")

import getpass, os, shutil, subprocess
from pathlib import Path

if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")

repo = Path("/content/GeoMapBench")
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", GIT_REF], check=True)
print("checked out:", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
'''


def build() -> None:
    simple = notebook([
        markdown("# GeoMapBench — simple eight-model evaluation\n\nThis notebook contains no evaluator logic. It mounts Drive, clones a frozen code release, validates the canonical 23 × 100 benchmark, runs the model suite, and displays the report. Rerun after a disconnect to resume completed IDs."),
        code(COMMON_CONFIG + '\nMAX_COST_USD_PER_MODEL = 25.0\n'),
        code(BOOTSTRAP + '\nsubprocess.run(["python", "-m", "pip", "install", "-q", "-e", str(repo)], check=True)\n'),
        code('''models = repo / "config/evaluation_models_2026-09.json"
output = Path(RESULTS_ROOT) / ("model_suite_full" if PER_LEAF_LIMIT is None else f"model_suite_{PER_LEAF_LIMIT}_per_leaf")

command = [
    "geomapbench-eval", "suite",
    "--benchmark-root", BENCHMARK_ROOT,
    "--models", str(models),
    "--output", str(output),
    "--max-cost-usd-per-model", str(MAX_COST_USD_PER_MODEL),
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
subprocess.run(command, check=True)
'''),
        code('''import pandas as pd
report = pd.read_csv(output / "model_comparison.csv")
display(report.sort_values("macro_accuracy", ascending=False))
print("Saved to:", output)
'''),
    ])

    rag = notebook([
        markdown("# GeoMapBench — Base, base_rag, and agentic_rag\n\nThis notebook contains no retrieval or evaluator implementation. It mounts Drive, clones a frozen code release, stages the existing corpus/index locally, and runs three paired conditions on one answer model. Both RAG conditions use dense BGE retrieval and reranking; neither uses BM25."),
        code(COMMON_CONFIG + '''
CORPUS_ROOT = "/content/drive/MyDrive/GeoMapRAG_Corpus"
ANSWER_MODEL = "qwen/qwen3.8-flash"
AGENT_MODEL = "google/gemini-3.5-flash-lite"
MAX_COST_USD_PER_CONDITION = 25.0
'''),
        code(BOOTSTRAP + '\nsubprocess.run(["python", "-m", "pip", "install", "-q", "-e", f"{repo}[rag-index]"], check=True)\n'),
        code('''output = Path(RESULTS_ROOT) / ("qwen38_rag_modes_full" if PER_LEAF_LIMIT is None else f"qwen38_rag_modes_{PER_LEAF_LIMIT}_per_leaf")
command = [
    "geomapbench-eval", "rag-experiment",
    "--benchmark-root", BENCHMARK_ROOT,
    "--corpus-root", CORPUS_ROOT,
    "--corpus-local-cache", "/content/geomaprag_corpus",
    "--output", str(output),
    "--model", ANSWER_MODEL,
    "--agent-model", AGENT_MODEL,
    "--max-cost-usd-per-condition", str(MAX_COST_USD_PER_CONDITION),
]
if PER_LEAF_LIMIT is not None:
    command += ["--per-leaf-limit", str(PER_LEAF_LIMIT)]
subprocess.run(command, check=True)
'''),
        code('''import json, pandas as pd
summary = json.loads((output / "experiment_summary.json").read_text())
comparisons = summary["comparisons"]
display(pd.DataFrame([
    {"condition": "base", "macro": summary["base_macro"], "delta_vs_base": 0.0},
    {"condition": "base_rag", "macro": summary["base_rag_macro"], "delta_vs_base": comparisons["base_to_base_rag"]["mean_delta"]},
    {"condition": "agentic_rag", "macro": summary["agentic_rag_macro"], "delta_vs_base": comparisons["base_to_agentic_rag"]["mean_delta"]},
]))
display(pd.DataFrame([
    {"comparison": name, "n": value["paired_record_count"], "delta": value["mean_delta"], "ci_low": value["delta_ci_low"], "ci_high": value["delta_ci_high"]}
    for name, value in comparisons.items()
]))
per_leaf = pd.read_csv(output / "comparisons/base_to_agentic_rag/rag_comparison.csv")
display(per_leaf.groupby("leaf", as_index=False)["delta"].mean().sort_values("delta", ascending=False))
print("Saved to:", output)
'''),
    ])

    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    (out / "GeoMapBench_Simple_Evaluation.ipynb").write_text(json.dumps(simple, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "GeoMapBench_Final_RAG.ipynb").write_text(json.dumps(rag, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
