from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import atomic_json, read_jsonl


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci(values: list[float], seed: int = 41023, rounds: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return (_mean(values), _mean(values))
    rng = random.Random(seed)
    estimates = sorted(_mean([rng.choice(values) for _ in values]) for _ in range(rounds))
    return estimates[int(.025 * rounds)], estimates[int(.975 * rounds) - 1]


def _plots(table: list[dict[str, Any]], bloom: dict[tuple[str, str], list[float]], output: Path) -> list[str]:
    """Create compact paper-draft diagnostics; callers still receive tables if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created: list[str] = []
    if table:
        labels = [row["leaf"].replace("_", "\n") for row in table]
        scores = [row["score"] for row in table]
        fig, axis = plt.subplots(figsize=(max(10, len(labels) * .42), 5))
        axis.bar(range(len(labels)), scores, color="#377eb8")
        axis.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1), ylabel="Score", title="GeoMapBench score by leaf")
        axis.tick_params(axis="x", labelrotation=60, labelsize=7)
        fig.tight_layout(); path = output / "per_leaf.png"; fig.savefig(path, dpi=220); plt.close(fig)
        created.append(path.name)
    if bloom:
        order = ["R", "U", "Ap", "An", "E", "C"]
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for (condition, level), values in bloom.items():
            grouped[condition][level] = _mean(values)
        fig, axis = plt.subplots(figsize=(6.5, 4))
        for condition, values in sorted(grouped.items()):
            levels = [level for level in order if level in values]
            axis.plot(levels, [values[level] for level in levels], marker="o", label=condition)
        axis.set(ylim=(0, 1), xlabel="Bloom level", ylabel="Score", title="Score by Bloom level")
        axis.legend(); fig.tight_layout(); path = output / "bloom.png"; fig.savefig(path, dpi=220); plt.close(fig)
        created.append(path.name)
    return created


def analyze(results_path: Path, output: Path, *, make_plots: bool = True) -> dict[str, Any]:
    rows = [row for row in read_jsonl(results_path) if row.get("status") == "ok" and isinstance(row.get("score"), (int, float))]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("condition", "base")), str(row.get("leaf", "unknown")))].append(row)
    table: list[dict[str, Any]] = []
    for (condition, leaf), group in sorted(groups.items()):
        values = [float(item["score"]) for item in group]
        text_rows = [item for item in group if not item.get("artifact_target")]
        text_values = [float(item["score"]) for item in text_rows]
        low, high = _ci(values)
        table.append({
            "condition": condition, "leaf": leaf, "n": len(group),
            "score": round(_mean(values), 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
            "text_answer_n": len(text_rows),
            "text_answer_score": round(_mean(text_values), 4) if text_values else None,
            "invalid_rate": round(sum(bool(item.get("parse_error")) for item in group) / len(group), 4),
            "generation_failure_rate": round(sum(bool(item.get("generation_failure")) for item in group) / len(group), 4),
            "cost_usd": round(sum(
                float((item.get("usage") or {}).get("cost") or 0)
                + float((item.get("retrieval_usage") or {}).get("cost") or 0)
                for item in group
            ), 6),
            "latency_seconds": round(_mean([float(item.get("latency_seconds") or 0) for item in group]), 3),
        })
    macro: dict[str, list[float]] = defaultdict(list)
    text_macro: dict[str, list[float]] = defaultdict(list)
    bloom: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in table:
        macro[row["condition"]].append(row["score"])
        if row["text_answer_score"] is not None:
            text_macro[row["condition"]].append(float(row["text_answer_score"]))
    for row in rows:
        if row.get("bloom"):
            bloom[(str(row.get("condition")), str(row["bloom"]))].append(float(row["score"]))
    output.mkdir(parents=True, exist_ok=True)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", "base"))].append(row)
    condition_summary = {}
    for condition, group in sorted(by_condition.items()):
        condition_summary[condition] = {
            "n": len(group),
            "micro_accuracy": round(_mean([float(item["score"]) for item in group]), 4),
            "invalid_json_rate": round(sum(bool(item.get("parse_error")) for item in group) / len(group), 4),
            "generation_failure_rate": round(sum(bool(item.get("generation_failure")) for item in group) / len(group), 4),
            "artifact_target_rate": round(sum(bool(item.get("artifact_target")) for item in group) / len(group), 4),
            "text_answer_micro_accuracy": round(_mean([
                float(item["score"]) for item in group if not item.get("artifact_target")
            ]), 4),
            "mean_latency_seconds": round(_mean([float(item.get("latency_seconds") or 0) for item in group]), 3),
            "total_cost_usd": round(sum(
                float((item.get("usage") or {}).get("cost") or 0)
                + float((item.get("retrieval_usage") or {}).get("cost") or 0)
                for item in group
            ), 6),
        }
    summary = {
        "record_count": len(rows),
        "macro_by_condition": {key: round(_mean(value), 4) for key, value in sorted(macro.items())},
        "text_answer_macro_by_condition": {key: round(_mean(value), 4) for key, value in sorted(text_macro.items())},
        "condition_summary": condition_summary,
        "bloom_by_condition": {f"{condition}:{level}": round(_mean(value), 4) for (condition, level), value in sorted(bloom.items())},
        "per_leaf": table,
        "plots": _plots(table, bloom, output) if make_plots else [],
    }
    atomic_json(output / "summary.json", summary)
    with (output / "per_leaf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["condition", "leaf", "n", "score"])
        writer.writeheader(); writer.writerows(table)
    return summary


def compare(base_path: Path, rag_path: Path, output: Path) -> dict[str, Any]:
    """Compare two completed conditions by the same record IDs."""
    fairness_fields = (
        "model", "temperature", "max_tokens", "reasoning_effort", "top_k",
        "include_images", "per_leaf_limit", "limit", "target_record_count",
        "selected_ids_hash", "selected_records_hash", "protocol",
        "benchmark_content_hash",
    )

    def run_config(path: Path) -> dict[str, Any]:
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing run_config.json beside {path}; protocol-locked comparison is impossible"
            )
        return __import__("json").loads(config_path.read_text(encoding="utf-8"))

    first_config, second_config = run_config(base_path), run_config(rag_path)
    mismatches = [
        f"{field}: {first_config.get(field)!r} != {second_config.get(field)!r}"
        for field in fairness_fields
        if first_config.get(field) != second_config.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Runs are not evaluation-protocol compatible; comparison refused.\n - "
            + "\n - ".join(mismatches)
        )

    def scored(path: Path) -> dict[str, dict[str, Any]]:
        return {
            str(row["id"]): row for row in read_jsonl(path)
            if row.get("status") == "ok" and isinstance(row.get("score"), (int, float))
        }
    base, rag = scored(base_path), scored(rag_path)
    common = sorted(set(base) & set(rag))
    if not common:
        raise ValueError("No common successful record IDs between base and RAG results")
    if set(base) != set(rag):
        raise ValueError(
            "Runs do not contain the exact same completed record IDs; comparison refused "
            f"(first_only={len(set(base) - set(rag))}, second_only={len(set(rag) - set(base))})"
        )
    rows = [{"id": record_id, "leaf": str(base[record_id].get("leaf", "unknown")), "base_score": float(base[record_id]["score"]), "rag_score": float(rag[record_id]["score"])} for record_id in common]
    for row in rows:
        row["delta"] = row["rag_score"] - row["base_score"]
    by_leaf: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_leaf[row["leaf"]].append(row["delta"])
    deltas = [row["delta"] for row in rows]
    low, high = _ci(deltas)
    summary = {
        "protocol_validation": "pass",
        "first_condition": first_config.get("condition"),
        "second_condition": second_config.get("condition"),
        "paired_record_count": len(rows), "base_only_records": len(set(base) - set(rag)),
        "rag_only_records": len(set(rag) - set(base)), "mean_delta": round(_mean(deltas), 4),
        "delta_ci_low": round(low, 4), "delta_ci_high": round(high, 4),
        "per_leaf_delta": {leaf: round(_mean(values), 4) for leaf, values in sorted(by_leaf.items())},
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "rag_comparison.json", summary)
    with (output / "rag_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "leaf", "base_score", "rag_score", "delta"])
        writer.writeheader(); writer.writerows(rows)
    return summary


def add_analyze_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("analyze", help="Aggregate scored results into publication-ready tables with bootstrap CIs.")
    parser.add_argument("--results", required=True, help="responses.jsonl from one run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")

    compare_parser = sub.add_parser("compare", help="Produce paired base-versus-RAG improvements by record ID.")
    compare_parser.add_argument("--base-results", required=True)
    compare_parser.add_argument("--rag-results", required=True)
    compare_parser.add_argument("--output", required=True)
