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
    all_rows = read_jsonl(results_path)
    rows = [row for row in all_rows if row.get("status") == "ok" and isinstance(row.get("score"), (int, float))]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("condition", "base")), str(row.get("leaf", "unknown")))].append(row)
    table: list[dict[str, Any]] = []
    for (condition, leaf), group in sorted(groups.items()):
        values = [float(item["score"]) for item in group]
        low, high = _ci(values)
        table.append({"condition": condition, "leaf": leaf, "n": len(group), "score": round(_mean(values), 4), "ci_low": round(low, 4), "ci_high": round(high, 4), "invalid_rate": round(sum(bool(item.get("parse_error")) for item in group) / len(group), 4), "cost_usd": round(sum(float((item.get("usage") or {}).get("cost") or 0) for item in group), 6), "latency_seconds": round(_mean([float(item.get("latency_seconds") or 0) for item in group]), 3)})
    macro: dict[str, list[float]] = defaultdict(list)
    bloom: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in table:
        macro[row["condition"]].append(row["score"])
    for row in rows:
        if row.get("bloom"):
            bloom[(str(row.get("condition")), str(row["bloom"]))].append(float(row["score"]))
    output.mkdir(parents=True, exist_ok=True)
    micro: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        micro[str(row.get("condition", "base"))].append(float(row["score"]))
    summary = {
        "record_count": len(rows),
        "models": sorted({str(row.get("model")) for row in rows}),
        "macro_by_condition": {key: round(_mean(value), 4) for key, value in sorted(macro.items())},
        "micro_by_condition": {key: round(_mean(value), 4) for key, value in sorted(micro.items())},
        "micro_ci_by_condition": {key: [round(x, 4) for x in _ci(value)] for key, value in sorted(micro.items())},
        "bloom_by_condition": {f"{condition}:{level}": round(_mean(value), 4) for (condition, level), value in sorted(bloom.items())},
        "invalid_json_rate": round(sum(bool(row.get("parse_error")) for row in rows) / max(1, len(rows)), 4),
        "reported_cost_usd": round(sum(float(row.get("total_cost_usd") or (row.get("usage") or {}).get("cost") or 0) for row in all_rows), 6),
        "answer_cost_usd": round(sum(float(row.get("answer_cost_usd") or (row.get("usage") or {}).get("cost") or 0) for row in all_rows), 6),
        "agent_cost_usd": round(sum(float(row.get("agent_cost_usd") or 0) for row in all_rows), 6),
        "error_row_count": sum(row.get("status") == "error" for row in all_rows),
        "mean_latency_seconds": round(_mean([float(row.get("latency_seconds") or 0) for row in rows]), 3),
        "per_leaf": table, "plots": _plots(table, bloom, output) if make_plots else [],
    }
    atomic_json(output / "summary.json", summary)
    with (output / "per_leaf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["condition", "leaf", "n", "score"])
        writer.writeheader(); writer.writerows(table)
    return summary


def compare(base_path: Path, rag_path: Path, output: Path) -> dict[str, Any]:
    """Compare two completed conditions by the same record IDs."""
    def scored(path: Path) -> dict[str, dict[str, Any]]:
        return {
            str(row["id"]): row for row in read_jsonl(path)
            if row.get("status") == "ok" and isinstance(row.get("score"), (int, float))
        }
    base, rag = scored(base_path), scored(rag_path)
    common = sorted(set(base) & set(rag))
    if not common:
        raise ValueError("No common successful record IDs between base and RAG results")
    rows = [{"id": record_id, "leaf": str(base[record_id].get("leaf", "unknown")), "base_score": float(base[record_id]["score"]), "rag_score": float(rag[record_id]["score"])} for record_id in common]
    for row in rows:
        row["delta"] = row["rag_score"] - row["base_score"]
    by_leaf: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_leaf[row["leaf"]].append(row["delta"])
    deltas = [row["delta"] for row in rows]
    low, high = _ci(deltas)
    summary = {
        "paired_record_count": len(rows), "base_only_records": len(set(base) - set(rag)),
        "rag_only_records": len(set(rag) - set(base)), "mean_delta": round(_mean(deltas), 4),
        "delta_ci_low": round(low, 4), "delta_ci_high": round(high, 4),
        "per_leaf_delta": {leaf: round(_mean(values), 4) for leaf, values in sorted(by_leaf.items())},
    }
    for label, predicate in {
        "rag_applicable": lambda row: bool(rag[row["id"]].get("rag_applicable")),
        "rag_not_applicable": lambda row: not bool(rag[row["id"]].get("rag_applicable")),
    }.items():
        subset = [row["delta"] for row in rows if predicate(row)]
        ci = _ci(subset) if subset else (0.0, 0.0)
        summary[label] = {"n": len(subset), "mean_delta": round(_mean(subset), 4), "ci": [round(x, 4) for x in ci]}
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
