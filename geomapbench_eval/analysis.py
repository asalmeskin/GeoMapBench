from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark import canonical_benchmark_records
from .common import atomic_json, atomic_jsonl, read_jsonl, utc_now
from .scoring import extract_answer
from .task_metrics import LOWER_IS_BETTER, TASK_METRIC_REVISION, TASK_METRIC_SCHEMA, evaluate_task_aware


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _ci(values: list[float], seed: int = 41023, rounds: int = 4000) -> tuple[float, float]:
    if len(values) < 2:
        return (_mean(values), _mean(values))
    rng = random.Random(seed)
    estimates = sorted(_mean([rng.choice(values) for _ in values]) for _ in range(rounds))
    return estimates[int(.025 * rounds)], estimates[int(.975 * rounds) - 1]


def _answer_target(record: dict[str, Any]) -> Any:
    target = record.get("target") or {}
    path = str((record.get("evaluation") or {}).get("target_field") or "")
    if path.startswith("target."):
        return target.get(path.split(".", 1)[1])
    return target.get("bloom_answer", target.get("answer", target))


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _macro_f1(pairs: list[tuple[Any, Any]]) -> float | None:
    clean = [(_norm(prediction), _norm(gold)) for prediction, gold in pairs if prediction is not None]
    if not clean:
        return None
    labels = sorted({value for pair in clean for value in pair})
    values = []
    for label in labels:
        tp = sum(prediction == label and gold == label for prediction, gold in clean)
        fp = sum(prediction == label and gold != label for prediction, gold in clean)
        fn = sum(prediction != label and gold == label for prediction, gold in clean)
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return _mean(values)


def canonical_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path) if path.exists() else []:
        key = (str(row.get("id")), str(row.get("model")), str(row.get("condition")))
        latest[key] = row
    return [latest[key] for key in sorted(latest)]


def rescore_in_place(results_path: Path, benchmark_root: Path) -> dict[str, Any]:
    """Upgrade saved raw responses without making any model/API calls."""
    records = canonical_benchmark_records(benchmark_root, prefer_clean=True)
    by_id = {str(record.get("id")): (directory, record) for directory, record in records}
    rows = canonical_rows(results_path)
    rescored = missing_benchmark = 0
    for row in rows:
        record_id = str(row.get("id"))
        if row.get("status") != "ok" or record_id not in by_id:
            missing_benchmark += int(record_id not in by_id)
            continue
        task_dir, record = by_id[record_id]
        row.update(evaluate_task_aware(record, str(row.get("response") or ""), task_dir))
        row["rescored_at"] = utc_now()
        rescored += 1
    atomic_jsonl(results_path, rows)
    return {
        "revision": TASK_METRIC_REVISION,
        "canonical_rows": len(rows),
        "rescored_rows": rescored,
        "missing_benchmark_rows": missing_benchmark,
        "api_calls": 0,
    }


def _resolve_benchmark(results_path: Path, benchmark_root: Path | None) -> Path:
    if benchmark_root is not None:
        return benchmark_root.expanduser().resolve()
    config = results_path.parent / "run_config.json"
    if not config.is_file():
        raise ValueError("Task-aware analysis needs --benchmark-root or a sibling run_config.json")
    value = json.loads(config.read_text(encoding="utf-8")).get("benchmark_root")
    if not value:
        raise ValueError(f"benchmark_root is missing from {config}")
    return Path(value).expanduser().resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else fallback
    # Later rows may have different leaf-specific metric columns.
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    results_path: Path,
    output: Path,
    *,
    benchmark_root: Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    benchmark = _resolve_benchmark(results_path, benchmark_root)
    rescore = rescore_in_place(results_path, benchmark)
    benchmark_rows = canonical_benchmark_records(benchmark, prefer_clean=True)
    records = {str(record.get("id")): record for _, record in benchmark_rows}
    rows = [
        row for row in canonical_rows(results_path)
        if row.get("status") == "ok" and isinstance(row.get("task_score"), (int, float))
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("condition", "base")), str(row.get("leaf", "unknown")))].append(row)

    per_leaf: list[dict[str, Any]] = []
    metrics_long: list[dict[str, Any]] = []
    for (condition, leaf), group in sorted(groups.items()):
        task_values = [float(item["task_score"]) for item in group]
        strict_values = [float(item.get("strict_score", item.get("score", 0.0))) for item in group]
        low, high = _ci(task_values)
        metric_values: dict[str, list[float]] = defaultdict(list)
        class_pairs: list[tuple[Any, Any]] = []
        for item in group:
            for name, value in (item.get("task_metrics") or {}).items():
                if name != "task_aware_score" and isinstance(value, (int, float)):
                    metric_values[str(name)].append(float(value))
            record = records.get(str(item.get("id")))
            if record and "macro_f1" in TASK_METRIC_SCHEMA.get(leaf, ()):
                prediction, error = extract_answer(str(item.get("response") or ""))
                class_pairs.append(("__invalid_json__" if error else prediction, _answer_target(record)))
        macro_f1 = _macro_f1(class_pairs)
        if macro_f1 is not None:
            metric_values["macro_f1"].append(macro_f1)

        schema = TASK_METRIC_SCHEMA.get(leaf, ("task_aware_score",))
        leaf_row: dict[str, Any] = {
            "condition": condition,
            "leaf": leaf,
            "n": len(group),
            "task_aware_score": round(_mean(task_values), 4),
            "task_ci_low": round(low, 4),
            "task_ci_high": round(high, 4),
            "strict_accuracy": round(_mean(strict_values), 4),
            "invalid_json_rate": round(sum(bool(item.get("parse_error")) for item in group) / len(group), 4),
            "generation_failure_rate": round(sum(bool(item.get("generation_failure")) for item in group) / len(group), 4),
            "cost_usd": round(sum(
                float((item.get("usage") or {}).get("cost") or 0.0)
                + float((item.get("retrieval_usage") or {}).get("cost") or 0.0)
                for item in group
            ), 6),
            "median_latency_seconds": round(_median([float(item.get("latency_seconds") or 0.0) for item in group]), 3),
        }
        for name in schema:
            if name == "task_aware_score":
                continue
            values = metric_values.get(name, [])
            if values:
                aggregate = _median(values) if name in LOWER_IS_BETTER else _mean(values)
                leaf_row[name] = round(aggregate, 6)
                metrics_long.append({
                    "condition": condition, "leaf": leaf, "metric": name,
                    "value": round(aggregate, 6), "n": len(values),
                    "direction": "lower" if name in LOWER_IS_BETTER else "higher",
                })
        metrics_long.append({
            "condition": condition, "leaf": leaf, "metric": "task_aware_score",
            "value": leaf_row["task_aware_score"], "n": len(task_values), "direction": "higher",
        })
        per_leaf.append(leaf_row)

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", "base"))].append(row)
    condition_summary: dict[str, dict[str, Any]] = {}
    for condition, group in sorted(by_condition.items()):
        leaves = [row for row in per_leaf if row["condition"] == condition]
        latencies = [float(item.get("latency_seconds") or 0.0) for item in group]
        condition_summary[condition] = {
            "n": len(group),
            "task_aware_macro": round(_mean([float(row["task_aware_score"]) for row in leaves]), 4),
            "strict_macro_accuracy": round(_mean([float(row["strict_accuracy"]) for row in leaves]), 4),
            "task_aware_micro": round(_mean([float(item["task_score"]) for item in group]), 4),
            "strict_micro_accuracy": round(_mean([float(item.get("strict_score", 0.0)) for item in group]), 4),
            "invalid_json_rate": round(sum(bool(item.get("parse_error")) for item in group) / len(group), 4),
            "generation_failure_rate": round(sum(bool(item.get("generation_failure")) for item in group) / len(group), 4),
            "format_reliability": round(sum(not item.get("parse_error") and not item.get("generation_failure") for item in group) / len(group), 4),
            "median_latency_seconds": round(_median(latencies), 3),
            "p95_latency_seconds": round(sorted(latencies)[max(0, int(.95 * len(latencies)) - 1)], 3),
            "total_cost_usd": round(sum(
                float((item.get("usage") or {}).get("cost") or 0.0)
                + float((item.get("retrieval_usage") or {}).get("cost") or 0.0)
                for item in group
            ), 6),
            "agent_call_count": sum(int((item.get("retrieval_usage") or {}).get("calls") or 0) for item in group),
            "agent_cached_call_count": sum(int((item.get("retrieval_usage") or {}).get("cached_calls") or 0) for item in group),
        }

    bloom: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("bloom"):
            bloom[(str(row.get("condition")), str(row.get("bloom")))].append(float(row["task_score"]))
    summary = {
        "task_metric_revision": TASK_METRIC_REVISION,
        "record_count": len(rows),
        "rescore": rescore,
        "condition_summary": condition_summary,
        "task_aware_macro_by_condition": {key: value["task_aware_macro"] for key, value in condition_summary.items()},
        "strict_macro_by_condition": {key: value["strict_macro_accuracy"] for key, value in condition_summary.items()},
        "macro_by_condition": {key: value["strict_macro_accuracy"] for key, value in condition_summary.items()},
        "text_answer_macro_by_condition": {key: value["strict_macro_accuracy"] for key, value in condition_summary.items()},
        "bloom_task_score_by_condition": {
            f"{condition}:{level}": round(_mean(values), 4)
            for (condition, level), values in sorted(bloom.items())
        },
        "per_leaf": per_leaf,
        "task_metrics": metrics_long,
    }
    atomic_json(output / "summary.json", summary)
    _write_csv(output / "per_leaf.csv", per_leaf, ["condition", "leaf", "n", "task_aware_score"])
    _write_csv(output / "task_metrics_long.csv", metrics_long, ["condition", "leaf", "metric", "value", "n", "direction"])
    return summary


def compare(base_path: Path, rag_path: Path, output: Path) -> dict[str, Any]:
    """Paired, protocol-locked comparison using task-aware and strict scores."""
    fairness_fields = (
        "model", "temperature", "max_tokens", "reasoning_effort", "reasoning_enabled",
        "include_images", "per_leaf_limit", "limit", "target_record_count",
        "selected_ids_hash", "selected_records_hash", "benchmark_content_hash",
    )

    semantic_protocol_fields = (
        "evaluation_protocol_revision", "prompt_revision", "scoring_revision",
        "task_metric_revision", "artifact_protocol_revision", "image_converter_revision",
        "canonical_loader",
    )

    def run_config(path: Path) -> dict[str, Any]:
        config_path = path.parent / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing run_config.json beside {path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    first_config, second_config = run_config(base_path), run_config(rag_path)
    mismatches = [
        f"{field}: {first_config.get(field)!r} != {second_config.get(field)!r}"
        for field in fairness_fields if first_config.get(field) != second_config.get(field)
    ]
    first_protocol = first_config.get("protocol") or {}
    second_protocol = second_config.get("protocol") or {}
    mismatches.extend(
        f"protocol.{field}: {first_protocol.get(field)!r} != {second_protocol.get(field)!r}"
        for field in semantic_protocol_fields
        if first_protocol.get(field) != second_protocol.get(field)
    )
    first_condition = str(first_config.get("condition") or "")
    second_condition = str(second_config.get("condition") or "")
    if "rag" in first_condition and "rag" in second_condition:
        if first_config.get("retrieval_config") != second_config.get("retrieval_config"):
            mismatches.append(
                f"retrieval_config: {first_config.get('retrieval_config')!r} != "
                f"{second_config.get('retrieval_config')!r}"
            )
    if mismatches:
        raise ValueError("Runs are not evaluation-protocol compatible; comparison refused.\n - " + "\n - ".join(mismatches))

    # Standalone comparisons must be just as safe as suite-driven comparisons.
    # Upgrade historical rows from their stored raw responses before reading
    # task-aware scores; this performs no API/model calls.
    first_benchmark = Path(str(first_config["benchmark_root"])).expanduser().resolve()
    second_benchmark = Path(str(second_config["benchmark_root"])).expanduser().resolve()
    if not first_benchmark.is_dir() and second_benchmark.is_dir():
        first_benchmark = second_benchmark
    if not second_benchmark.is_dir() and first_benchmark.is_dir():
        second_benchmark = first_benchmark
    rescore_in_place(base_path, first_benchmark)
    rescore_in_place(rag_path, second_benchmark)

    def scored(path: Path) -> dict[str, dict[str, Any]]:
        return {
            str(row["id"]): row for row in canonical_rows(path)
            if row.get("status") == "ok" and isinstance(row.get("task_score"), (int, float))
        }

    first, second = scored(base_path), scored(rag_path)
    if set(first) != set(second) or not first:
        raise ValueError(
            "Runs do not contain the exact same completed record IDs; comparison refused "
            f"(first_only={len(set(first) - set(second))}, second_only={len(set(second) - set(first))})"
        )
    rows: list[dict[str, Any]] = []
    for record_id in sorted(first):
        first_task, second_task = float(first[record_id]["task_score"]), float(second[record_id]["task_score"])
        first_strict = float(first[record_id].get("strict_score", 0.0))
        second_strict = float(second[record_id].get("strict_score", 0.0))
        rows.append({
            "id": record_id, "leaf": str(first[record_id].get("leaf", "unknown")),
            "first_task_score": first_task, "second_task_score": second_task,
            "task_delta": second_task - first_task,
            "first_strict_score": first_strict, "second_strict_score": second_strict,
            "strict_delta": second_strict - first_strict,
        })
    task_deltas = [row["task_delta"] for row in rows]
    strict_deltas = [row["strict_delta"] for row in rows]
    task_low, task_high = _ci(task_deltas)
    strict_low, strict_high = _ci(strict_deltas)
    by_leaf: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_leaf[row["leaf"]].append(row["task_delta"])
    summary = {
        "protocol_validation": "pass",
        "first_condition": first_config.get("condition"),
        "second_condition": second_config.get("condition"),
        "paired_record_count": len(rows),
        "task_aware_mean_delta": round(_mean(task_deltas), 4),
        "task_aware_delta_ci_low": round(task_low, 4),
        "task_aware_delta_ci_high": round(task_high, 4),
        "strict_mean_delta": round(_mean(strict_deltas), 4),
        "strict_delta_ci_low": round(strict_low, 4),
        "strict_delta_ci_high": round(strict_high, 4),
        "win_tie_loss": {
            "wins": sum(value > 0 for value in task_deltas),
            "ties": sum(value == 0 for value in task_deltas),
            "losses": sum(value < 0 for value in task_deltas),
        },
        "per_leaf_task_delta": {leaf: round(_mean(values), 4) for leaf, values in sorted(by_leaf.items())},
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "rag_comparison.json", summary)
    _write_csv(output / "rag_comparison.csv", rows, ["id", "leaf", "task_delta"])
    return summary


def add_analyze_parser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser("analyze", help="Offline task-aware rescoring and publication tables; no API calls.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-root")
    parser.add_argument("--no-plots", action="store_true")

    compare_parser = sub.add_parser("compare", help="Produce a paired protocol-locked condition comparison.")
    compare_parser.add_argument("--base-results", required=True)
    compare_parser.add_argument("--rag-results", required=True)
    compare_parser.add_argument("--output", required=True)
