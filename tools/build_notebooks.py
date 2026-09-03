from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip().splitlines(True)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(value).strip().splitlines(True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "GeoMapBench final", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LIVE_RUNNER = r'''
def run_live(command, *, cwd=None, env_extra=None, display_command=None):
    import os, subprocess
    environment = os.environ.copy()
    environment.update(env_extra or {})
    environment["PYTHONUNBUFFERED"] = "1"
    shown = display_command or list(map(str, command))
    print("Running:", " ".join(map(str, shown)), flush=True)
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


def clone_repository(destination, repository, ref):
    import shutil
    from pathlib import Path

    destination = Path(destination)
    staging = destination.with_name(destination.name + "_staging")
    if staging.exists():
        shutil.rmtree(staging)
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    clone_url = repository
    shown_url = repository
    if token and repository.startswith("https://github.com/"):
        clone_url = repository.replace("https://", f"https://x-access-token:{token}@", 1)
        shown_url = repository + " (using Colab secret GITHUB_TOKEN)"
    command = ["git", "clone", "--depth", "1", "--branch", ref, clone_url, str(staging)]
    shown = ["git", "clone", "--depth", "1", "--branch", ref, shown_url, str(staging)]
    try:
        run_live(command, env_extra={"GIT_TERMINAL_PROMPT": "0"}, display_command=shown)
    except Exception as error:
        raise RuntimeError(
            "Git clone failed. Confirm GIT_REF exists on GitHub. If the repository is private, "
            "add a Colab secret named GITHUB_TOKEN with read access and rerun this cell."
        ) from error
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
'''


SETUP = LIVE_RUNNER + r'''
from google.colab import drive
drive.mount("/content/drive")

import os, subprocess, sys
from pathlib import Path

repo = Path("/content/GeoMapBench")
clone_repository(repo, GITHUB_REPO, GIT_REF)
INSTALL_TARGET = f"{repo}[rag-index]" if INSTALL_RAG else str(repo)
run_live([sys.executable, "-m", "pip", "install", "-q", "-e", INSTALL_TARGET])

assert Path(BENCHMARK_ROOT).is_dir(), f"Benchmark not found: {BENCHMARK_ROOT}"
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
os.environ["GEOMAPBENCH_IMAGE_CACHE"] = str(Path(CACHE_ROOT) / "converted_images_final")
os.environ["HF_HOME"] = str(Path(CACHE_ROOT) / "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(Path(CACHE_ROOT) / "huggingface" / "sentence_transformers")
os.environ["PYTHONUNBUFFERED"] = "1"
version = subprocess.check_output(
    [sys.executable, "-c", "import geomapbench_eval; print(geomapbench_eval.__version__)"], text=True,
).strip()
assert version == "2.2.0", f"Expected GeoMapBench 2.2.0, installed {version}"
print("Installed final release:", version)
'''


API_KEY = r'''
import getpass, os
if not os.environ.get("OPENROUTER_API_KEY"):
    try:
        from google.colab import userdata
        os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY") or ""
    except Exception:
        pass
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")
assert os.environ["OPENROUTER_API_KEY"], "OPENROUTER_API_KEY is empty"
print("API key loaded in this runtime only.")
'''


EVALUATION = notebook([
    markdown("""
    # GeoMapBench — final cumulative benchmark

    This notebook upgrades existing benchmark outputs in place into one canonical final suite. Saved raw answers are rescored locally with task-aware metrics. It makes new paid calls only for:

    1. the newly added Llama 4 Scout baseline; and
    2. legacy file-artifact rows whose old prompt did not produce a measurable prediction.

    Invalid JSON and genuine generation failures are preserved as failures and are not selectively retried. Logs are streamed live. Re-running the execution cell resumes from the write-ahead API cache and completed records.
    """),
    code(r'''
    # EDIT ONLY THIS CELL
    GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
    GIT_REF = "main"  # use a real tag here only after you create/push that tag

    BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
    RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
    CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

    TARGET_PER_LEAF = 50  # keep 50 to adopt the completed run; later increase to 100 for all 2,300
    MAX_COST_USD_PER_MODEL = 50.0
    REQUEST_DELAY_SECONDS = 1.0
    PROGRESS_EVERY = 5
    FORCE_PREFLIGHT = False

    FINAL_OUTPUT_NAME = "model_suite_final"
    # Newest compatible suite first. Older names are adopted only when they
    # exist, then removed after every final model is verified complete.
    LEGACY_OUTPUT_NAMES = [
        "model_suite_cumulative_v7",
        "model_suite_6_per_leaf_v6",
        "model_suite_1_per_leaf_v6",
        "model_suite_1_per_leaf_v4",
        "model_suite_1_per_leaf",
        "model_suite_full",
    ]
    DELETE_LEGACY_AFTER_SUCCESS = True  # only after all 6 final models are complete
    INSTALL_RAG = False
    '''),
    code(SETUP),
    code(API_KEY),
    code(r'''
    # Live migration + selective execution. Rerun this exact cell after any disconnect/pause.
    from pathlib import Path
    import sys

    models = repo / "config/evaluation_models_final.json"
    output = Path(RESULTS_ROOT) / FINAL_OUTPUT_NAME
    command = [
        sys.executable, "-u", "-m", "geomapbench_eval", "suite",
        "--benchmark-root", BENCHMARK_ROOT,
        "--models", str(models),
        "--output", str(output),
        "--target-per-leaf", str(TARGET_PER_LEAF),
        "--preflight-cache", str(Path(CACHE_ROOT) / "preflight_final"),
        "--max-cost-usd-per-model", str(MAX_COST_USD_PER_MODEL),
        "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
        "--progress-every", str(PROGRESS_EVERY),
        "--timeout-seconds", "240", "--retries", "6",
        "--retry-base-seconds", "5", "--retry-max-seconds", "60",
        "--max-consecutive-errors", "2",
    ]
    found_legacy = []
    for name in LEGACY_OUTPUT_NAMES:
        candidate = Path(RESULTS_ROOT) / name
        if candidate.is_dir() and candidate.resolve() != output.resolve():
            command += ["--legacy-output", str(candidate)]
            found_legacy.append(candidate)
            print("Verified migration source:", candidate)
    if DELETE_LEGACY_AFTER_SUCCESS and found_legacy:
        command += ["--delete-legacy-after-success"]
    if FORCE_PREFLIGHT:
        command += ["--force-preflight"]
    run_live(command, cwd=repo)
    print("Canonical final suite:", output)
    '''),
    code(r'''
    # Compact paper tables and saved Seaborn figures.
    import json, pandas as pd
    from IPython.display import Image, display

    summary = json.loads((output / "model_suite_summary.json").read_text(encoding="utf-8"))
    columns = [
        "family", "model", "n", "target_n", "complete", "task_aware_macro",
        "strict_macro_accuracy", "format_reliability", "invalid_json_rate",
        "generation_failure_rate", "median_latency_seconds", "p95_latency_seconds",
        "reported_cost_usd", "run_stop_reason",
    ]
    display(pd.DataFrame(summary["models"])[columns])
    if summary.get("migration"):
        display(pd.DataFrame(summary["migration"]))
    if summary.get("failures"):
        display(pd.DataFrame(summary["failures"]))
    print("Cohort:", {key: summary["cohort"][key] for key in ("target_per_leaf", "target_record_count", "selected_ids_hash")})
    print("Preflight cache hit:", summary["preflight"].get("cache_hit"))
    print("Legacy cleanup:", summary.get("legacy_cleanup", []))
    for name in summary.get("plots", []):
        display(Image(filename=str(output / "plots" / name)))
    '''),
    markdown("""
    Operating rule: keep only `model_suite_final`. To extend the study, increase `TARGET_PER_LEAF` in the same notebook and run the same cell. Never rename or manually combine JSONL files. The cohort is nested and Bloom-stratified, so completed IDs are reused and only new IDs are requested.
    """),
])


RAG = notebook([
    markdown("""
    # GeoMapBench — validated multimodal RAG evaluation

    This notebook runs two new conditions: `multimodal_rag` and `agentic_multimodal_rag`. Both use BGE text retrieval, the existing CLIP image index, benchmark-image-to-corpus-image retrieval, rank fusion, optional retrieved reference images, and a predeclared 14-task coverage gate. BM25 is absent.

    The paid answer evaluation cannot start until a runtime validation proves that both the 180,344-record text index and the 1,794-image CLIP index are loaded, a benchmark image is encoded, both searches return results, fusion succeeds, and retrieved corpus image files are accessible. The old 1,662-asset benchmark preflight is not rerun; its already-passed report is trusted for the portable benchmark hash.

    Old text-only RAG results are preserved as an ablation and are never imported into this new output. Re-running the execution cell resumes only the v2.2 multimodal conditions.
    """),
    code(r'''
    # EDIT ONLY THIS CELL
    GITHUB_REPO = "https://github.com/asalmeskin/GeoMapBench.git"
    GIT_REF = "main"

    BENCHMARK_ROOT = "/content/drive/MyDrive/GeoMapBench_Data/geomapbench_100"
    CORPUS_ROOT = "/content/drive/MyDrive/GeoMapRAG_Corpus"
    RESULTS_ROOT = "/content/drive/MyDrive/geomapbench_results_final"
    CACHE_ROOT = "/content/drive/MyDrive/geomapbench_runtime_cache"

    ANSWER_MODEL = "anthropic/claude-sonnet-5"
    AGENT_MODEL = "google/gemini-3.5-flash-lite"
    TARGET_PER_LEAF = 1  # smoke: 1; then increase in-place to 6, 50, or 100
    MAX_COST_USD_PER_CONDITION = 30.0
    REQUEST_DELAY_SECONDS = 1.0
    PROGRESS_EVERY = 5

    FINAL_OUTPUT_NAME = "rag_suite_multimodal_claude_v220"
    TRUSTED_BENCHMARK_REPORT = f"{CACHE_ROOT}/preflight_final/benchmark_preflight.json"
    BASE_RESULTS = f"{RESULTS_ROOT}/model_suite_final/anthropic_claude-sonnet-5/responses.jsonl"
    INSTALL_RAG = True
    '''),
    code(SETUP + r'''
assert Path(CORPUS_ROOT).is_dir(), f"RAG corpus not found: {CORPUS_ROOT}"
for relative in (
    "indexes/text.faiss", "indexes/text_metadata.jsonl", "indexes/text_manifest.json",
    "indexes/image.faiss", "indexes/image_metadata.jsonl", "indexes/image_manifest.json",
):
    assert (Path(CORPUS_ROOT) / relative).is_file(), f"Missing multimodal artifact: {relative}"
assert Path(TRUSTED_BENCHMARK_REPORT).is_file(), (
    "The already-passed benchmark report is missing: " + TRUSTED_BENCHMARK_REPORT
)
old_text_only_outputs = [
    Path(RESULTS_ROOT) / "rag_suite_final_gpt",
    Path(RESULTS_ROOT) / "rag_suite_final",
]
for old_text_only in old_text_only_outputs:
    if old_text_only.exists():
        print("Preserving text-only ablation (not imported):", old_text_only)
print("New isolated multimodal output:", Path(RESULTS_ROOT) / FINAL_OUTPUT_NAME)
'''),
    code(API_KEY),
    code(r'''
    # Live RAG execution. Rerun this exact cell after any disconnect/pause.
    from pathlib import Path
    import sys

    output = Path(RESULTS_ROOT) / FINAL_OUTPUT_NAME
    command = [
        sys.executable, "-u", "-m", "geomapbench_eval", "rag-suite",
        "--benchmark-root", BENCHMARK_ROOT, "--corpus-root", CORPUS_ROOT,
        "--work-root", "/content/geomaprag_multimodal_work_v220",
        "--output", str(output), "--target-per-leaf", str(TARGET_PER_LEAF),
        "--models", str(repo / "config/evaluation_models_final.json"),
        "--benchmark-report", TRUSTED_BENCHMARK_REPORT,
        "--agent-cache", str(Path(CACHE_ROOT) / "agent_cache_multimodal_v220"),
        "--model", ANSWER_MODEL, "--agent-model", AGENT_MODEL,
        "--agent-reasoning-effort", "minimal",
        "--conditions", "multimodal_rag,agentic_multimodal_rag",
        "--max-cost-usd-per-model", str(MAX_COST_USD_PER_CONDITION),
        "--top-k", "3", "--candidate-k", "40", "--image-candidate-k", "20",
        "--max-reference-images", "1",
        "--max-passage-chars", "1200", "--max-context-chars", "3000",
        "--request-delay-seconds", str(REQUEST_DELAY_SECONDS),
        "--progress-every", str(PROGRESS_EVERY),
        "--timeout-seconds", "240", "--retries", "6",
        "--retry-base-seconds", "5", "--retry-max-seconds", "60",
        "--max-consecutive-errors", "2",
    ]
    if Path(BASE_RESULTS).is_file():
        command += ["--base-results", BASE_RESULTS]
    run_live(command, cwd=repo)
    print("Canonical multimodal RAG suite:", output)
    '''),
    code(r'''
    # Matched conditions, paired comparison and Seaborn figures.
    import json, pandas as pd
    from IPython.display import Image, display

    summary = json.loads((output / "rag_suite_summary.json").read_text(encoding="utf-8"))
    rows = []
    for condition, report in summary["reports"].items():
        stats = report["analysis"]["condition_summary"].get(condition, {})
        rows.append({
            "condition": condition,
            "complete": report["run"].get("complete"),
            "completed_total": report["run"].get("completed_total"),
            "target_records": report["run"].get("target_records"),
            **stats,
        })
    display(pd.DataFrame(rows))
    comparisons = summary.get("comparisons", {})
    if comparisons:
        display(pd.DataFrame([{"comparison": key, **value} for key, value in comparisons.items()]))
    print("Cohort:", {key: summary["cohort"][key] for key in ("target_per_leaf", "target_record_count", "selected_ids_hash")})
    print("Trusted benchmark report reused:", summary["benchmark_report"].get("reused_without_rescan"))
    print("Multimodal runtime validation:", summary["multimodal_validation"])
    audits = {
        condition: report.get("modality_audit", {})
        for condition, report in summary["reports"].items()
    }
    display(pd.DataFrame([{"condition": key, **value} for key, value in audits.items()]))
    for condition, audit in audits.items():
        if summary["reports"][condition]["run"].get("completed_total", 0):
            assert audit.get("both_modalities_observed"), f"Modality audit failed: {condition}"
    for name in summary.get("plots", []):
        display(Image(filename=str(output / "plots" / name)))
    '''),
    markdown("""
    Start with `TARGET_PER_LEAF=1`. Confirm the explicit `[rag-validation] PASS` line and the modality-audit table, then increase the same value to 50 in the same output directory. Each larger cohort contains all earlier IDs, so prior calls are reused. Do not rename the output and do not copy files from any earlier `rag_suite_final*` directory; those are text-only ablations.
    """),
])


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "GeoMapBench_Evaluation_Final.ipynb": EVALUATION,
        "GeoMapBench_RAG_Final.ipynb": RAG,
    }.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
