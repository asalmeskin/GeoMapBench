"""Offline analysis: paired comparisons, free ablations and figures.

Nothing here makes an API call. The pre-repair ablation reuses the untouched
``raw_response`` column that the driver stores next to the repaired answer, so
the contribution of the deterministic repair layer is measurable without paying
for a second run.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geomapbench_eval.analysis import _ci, _mean, _write_csv, analyze, canonical_rows, rescore_in_place
from geomapbench_eval.common import atomic_json, atomic_jsonl, read_jsonl

FAIRNESS_FIELDS = (
    "model", "temperature", "max_tokens", "reasoning_effort", "reasoning_enabled",
    "include_images", "per_leaf_limit", "limit", "target_record_count",
    "selected_ids_hash", "selected_records_hash", "benchmark_content_hash",
)
SEMANTIC_PROTOCOL_FIELDS = (
    "evaluation_protocol_revision", "prompt_revision", "scoring_revision",
    "task_metric_revision", "artifact_protocol_revision", "image_converter_revision",
    "canonical_loader",
)


def _run_config(results_path: Path) -> dict[str, Any]:
    config_path = Path(results_path).parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json beside {results_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def snapshot_condition(results_path: Path, destination: Path) -> Path:
    """Copy a previous condition next to the new run before comparing it.

    ``compare`` and ``rescore_in_place`` rewrite the file they are handed. The
    earlier ablations are never touched: every comparison runs on a copy that
    lives inside the new output directory.
    """
    results_path = Path(results_path).expanduser().resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = results_path.parent / "run_config.json"
    if not results_path.is_file() or not config.is_file():
        raise FileNotFoundError(f"Cannot snapshot {results_path}: responses.jsonl or run_config.json missing")
    target = destination / "responses.jsonl"
    if not target.exists() or target.stat().st_size != results_path.stat().st_size:
        shutil.copy2(results_path, target)
    shutil.copy2(config, destination / "run_config.json")
    return target


def paired_compare(
    first_results: Path,
    second_results: Path,
    output: Path,
    *,
    label: str = "comparison",
    skip_rescore: bool = False,
) -> dict[str, Any]:
    """Protocol-locked paired comparison between any two conditions.

    Identical to ``geomapbench_eval.analysis.compare`` except that it does not
    require two retrieval-bearing conditions to share a ``retrieval_config``:
    comparing two *different* retrieval systems is the entire point here, and
    every fairness field that could bias the result is still enforced.

    ``skip_rescore`` skips the (potentially Drive-I/O-heavy) rescore pass when
    the caller already rescored both files -- or knows they are already
    correctly scored -- moments earlier in the same process.
    """
    first_config, second_config = _run_config(first_results), _run_config(second_results)
    mismatches = [
        f"{field}: {first_config.get(field)!r} != {second_config.get(field)!r}"
        for field in FAIRNESS_FIELDS if first_config.get(field) != second_config.get(field)
    ]
    first_protocol = first_config.get("protocol") or {}
    second_protocol = second_config.get("protocol") or {}
    mismatches.extend(
        f"protocol.{field}: {first_protocol.get(field)!r} != {second_protocol.get(field)!r}"
        for field in SEMANTIC_PROTOCOL_FIELDS
        if first_protocol.get(field) != second_protocol.get(field)
    )
    if mismatches:
        raise ValueError(
            "Runs are not evaluation-protocol compatible; comparison refused.\n - "
            + "\n - ".join(mismatches)
        )
    if not skip_rescore:
        first_root = Path(str(first_config["benchmark_root"])).expanduser().resolve()
        second_root = Path(str(second_config["benchmark_root"])).expanduser().resolve()
        if not first_root.is_dir() and second_root.is_dir():
            first_root = second_root
        if not second_root.is_dir() and first_root.is_dir():
            second_root = first_root
        rescore_in_place(Path(first_results), first_root)
        rescore_in_place(Path(second_results), second_root)

    def scored(path: Path) -> dict[str, dict[str, Any]]:
        return {
            str(row["id"]): row for row in canonical_rows(Path(path))
            if row.get("status") == "ok" and isinstance(row.get("task_score"), (int, float))
        }

    first, second = scored(first_results), scored(second_results)
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("The two runs share no completed record IDs")
    rows: list[dict[str, Any]] = []
    for record_id in shared:
        a, b = first[record_id], second[record_id]
        rows.append({
            "id": record_id,
            "leaf": str(a.get("leaf", "unknown")),
            "bloom": a.get("bloom"),
            "first_task_score": float(a["task_score"]),
            "second_task_score": float(b["task_score"]),
            "task_delta": float(b["task_score"]) - float(a["task_score"]),
            "first_strict_score": float(a.get("strict_score", 0.0)),
            "second_strict_score": float(b.get("strict_score", 0.0)),
            "strict_delta": float(b.get("strict_score", 0.0)) - float(a.get("strict_score", 0.0)),
        })
    task_deltas = [row["task_delta"] for row in rows]
    strict_deltas = [row["strict_delta"] for row in rows]
    task_low, task_high = _ci(task_deltas)
    strict_low, strict_high = _ci(strict_deltas)
    by_leaf: dict[str, list[float]] = defaultdict(list)
    by_leaf_strict: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_leaf[row["leaf"]].append(row["task_delta"])
        by_leaf_strict[row["leaf"]].append(row["strict_delta"])
    wins = sum(value > 0 for value in task_deltas)
    losses = sum(value < 0 for value in task_deltas)
    summary = {
        "protocol_validation": "pass",
        "label": label,
        "first_condition": first_config.get("condition"),
        "second_condition": second_config.get("condition"),
        "paired_record_count": len(rows),
        "first_task_aware_macro": round(
            _mean([_mean(values) for values in _grouped(first, "task_score").values()]), 4),
        "second_task_aware_macro": round(
            _mean([_mean(values) for values in _grouped(second, "task_score").values()]), 4),
        "first_strict_macro": round(
            _mean([_mean(values) for values in _grouped(first, "strict_score").values()]), 4),
        "second_strict_macro": round(
            _mean([_mean(values) for values in _grouped(second, "strict_score").values()]), 4),
        "task_aware_mean_delta": round(_mean(task_deltas), 4),
        "task_aware_delta_ci_low": round(task_low, 4),
        "task_aware_delta_ci_high": round(task_high, 4),
        "strict_mean_delta": round(_mean(strict_deltas), 4),
        "strict_delta_ci_low": round(strict_low, 4),
        "strict_delta_ci_high": round(strict_high, 4),
        "win_tie_loss": {
            "wins": wins,
            "ties": sum(value == 0 for value in task_deltas),
            "losses": losses,
        },
        "sign_test_p_value": _sign_test(wins, losses),
        "per_leaf_task_delta": {leaf: round(_mean(values), 4) for leaf, values in sorted(by_leaf.items())},
        "per_leaf_strict_delta": {leaf: round(_mean(values), 4) for leaf, values in sorted(by_leaf_strict.items())},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "comparison.json", summary)
    _write_csv(output / "comparison.csv", rows, ["id", "leaf", "task_delta"])
    return summary


def _grouped(rows: dict[str, dict[str, Any]], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        grouped[str(row.get("leaf", "unknown"))].append(float(row.get(key, 0.0) or 0.0))
    return grouped


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test on the discordant pairs."""
    from math import comb

    total = wins + losses
    if total == 0:
        return 1.0
    observed = min(wins, losses)
    tail = sum(comb(total, k) for k in range(observed + 1)) / (2.0 ** total)
    return round(min(1.0, 2.0 * tail), 6)


# ---------------------------------------------------------------------------
# Free ablations
# ---------------------------------------------------------------------------


def pre_repair_ablation(condition_dir: Path, benchmark_root: Path) -> dict[str, Any]:
    """Score the model's untouched output to isolate the repair layer's effect."""
    condition_dir = Path(condition_dir)
    source = condition_dir / "responses.jsonl"
    rows = [row for row in read_jsonl(source) if row.get("status") == "ok"]
    if not rows or not any("raw_response" in row for row in rows):
        return {"available": False, "reason": "no raw_response column"}
    shadow = condition_dir / "ablations" / "pre_repair"
    shadow.mkdir(parents=True, exist_ok=True)
    shutil.copy2(condition_dir / "run_config.json", shadow / "run_config.json")
    atomic_jsonl(shadow / "responses.jsonl", [
        {**row, "response": row.get("raw_response") or row.get("response")}
        for row in rows
    ])
    before = analyze(
        shadow / "responses.jsonl", shadow / "analysis", benchmark_root=Path(benchmark_root),
    )
    after = analyze(
        source, condition_dir / "analysis", benchmark_root=Path(benchmark_root),
    )
    condition = next(iter(after["condition_summary"]), "")
    report = {
        "available": True,
        "condition": condition,
        "pre_repair_task_aware_macro": before["task_aware_macro_by_condition"].get(condition),
        "post_repair_task_aware_macro": after["task_aware_macro_by_condition"].get(condition),
        "pre_repair_strict_macro": before["strict_macro_by_condition"].get(condition),
        "post_repair_strict_macro": after["strict_macro_by_condition"].get(condition),
        "pre_repair_invalid_json_rate": before["condition_summary"].get(condition, {}).get("invalid_json_rate"),
        "post_repair_invalid_json_rate": after["condition_summary"].get(condition, {}).get("invalid_json_rate"),
        "records_touched": sum(1 for row in rows if row.get("repairs")),
        "repair_counts": dict(Counter(
            action for row in rows for action in (row.get("repairs") or [])
        ).most_common()),
    }
    atomic_json(condition_dir / "ablations" / "pre_repair_summary.json", report)
    return report


def toolbelt_audit(agent_trace_path: Path) -> dict[str, Any]:
    """How often each tool fired, and how often it answered the question outright."""
    rows = read_jsonl(Path(agent_trace_path)) if Path(agent_trace_path).is_file() else []
    latest = {str(row.get("id")): row for row in rows}
    by_tool: Counter[str] = Counter()
    by_leaf_tool: dict[str, Counter[str]] = defaultdict(Counter)
    proposals: Counter[str] = Counter()
    exact_proposals: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for row in latest.values():
        leaf = str(row.get("leaf"))
        for tool in row.get("tools") or []:
            name = str(tool.get("tool"))
            if tool.get("status") == "ok":
                by_tool[name] += 1
                by_leaf_tool[leaf][name] += 1
            elif tool.get("status") == "error":
                errors[name] += 1
        proposal = (row.get("stages") or {}).get("proposal")
        if proposal:
            proposals[str(proposal.get("source"))] += 1
            if proposal.get("confidence") == "exact":
                exact_proposals[str(proposal.get("source"))] += 1
    return {
        "records": len(latest),
        "tool_fire_counts": dict(by_tool.most_common()),
        "tool_errors": dict(errors.most_common()),
        "proposal_counts": dict(proposals.most_common()),
        "exact_proposal_counts": dict(exact_proposals.most_common()),
        "records_with_exact_proposal": sum(exact_proposals.values()),
        "per_leaf_tools": {leaf: dict(counter.most_common()) for leaf, counter in sorted(by_leaf_tool.items())},
    }


def retrieval_audit(trace_path: Path) -> dict[str, Any]:
    rows = read_jsonl(Path(trace_path)) if Path(trace_path).is_file() else []
    latest = {str(row.get("id")): row for row in rows}
    if not latest:
        return {"records": 0}
    values = list(latest.values())
    return {
        "records": len(values),
        "text_index_used": sum(bool(row.get("text_index_used")) for row in values),
        "image_index_used": sum(bool(row.get("image_index_used")) for row in values),
        "benchmark_image_queries": sum(int(row.get("benchmark_image_count") or 0) > 0 for row in values),
        "retrieved_images_attached": sum(int(row.get("reference_images") or 0) for row in values),
        "capability_matched_passages": sum(int(row.get("capability_matches") or 0) for row in values),
        "abstentions": sum(bool(row.get("abstained")) for row in values),
        "critic_rejections": sum(
            1 for row in values if (row.get("critic") or {}).get("use_context") is False
        ),
        "followups": sum(1 for row in values if row.get("followup")),
        "mean_kept_contexts": round(
            _mean([float(row.get("kept_context_count") or 0) for row in values]), 3
        ),
        "both_modalities_observed": bool(
            sum(bool(row.get("text_index_used")) for row in values)
            and sum(bool(row.get("image_index_used")) for row in values)
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def condition_plots(output: Path, analyses: dict[str, dict[str, Any]], comparisons: dict[str, dict[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from geomapbench_eval.plots import BLOOM_ORDER, PALETTE, _bold, _legend_bottom, _save, _short_leaf, _theme

    _theme()
    plot_root = Path(output) / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    long_rows: list[dict[str, Any]] = []
    bloom_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    for condition, report in analyses.items():
        label = _condition_label(condition)
        for leaf in report.get("per_leaf", []):
            reliability = 1.0 - max(
                float(leaf.get("invalid_json_rate", 0)), float(leaf.get("generation_failure_rate", 0))
            )
            for metric, value in (
                ("Task-aware", leaf.get("task_aware_score")),
                ("Strict", leaf.get("strict_accuracy")),
                ("Reliability", reliability),
            ):
                if value is not None:
                    long_rows.append({"Condition": label, "Metric": metric, "Score": 100 * float(value)})
            heat_rows.append({
                "Condition": label, "Task": _short_leaf(str(leaf["leaf"])),
                "Score": 100 * float(leaf["task_aware_score"]),
            })
        for key, value in report.get("bloom_task_score_by_condition", {}).items():
            _, level = key.split(":", 1)
            bloom_rows.append({"Condition": label, "Bloom level": level, "Score": 100 * float(value)})

    if long_rows:
        frame = pd.DataFrame(long_rows)
        width = max(7.5, frame["Condition"].nunique() * 2.4)
        fig, axis = plt.subplots(figsize=(width, 5.1))
        sns.barplot(data=frame, x="Condition", y="Score", hue="Metric",
                    palette=PALETTE[:3], errorbar=("ci", 95), capsize=.08, ax=axis)
        axis.set(xlabel="Condition", ylabel="Score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=3)
        created.append(_save(fig, plot_root / "geoagent_overview.png"))

    if heat_rows:
        pivot = pd.DataFrame(heat_rows).pivot_table(
            index="Task", columns="Condition", values="Score", aggfunc="mean",
        )
        fig, axis = plt.subplots(figsize=(max(7.5, pivot.shape[1] * 2.0), 9.2))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100,
                    linewidths=.45, cbar_kws={"label": "Task-aware score (%)"}, ax=axis)
        axis.set(xlabel="Condition", ylabel="Task")
        _bold(axis)
        for text in axis.texts:
            text.set_fontweight("bold")
        created.append(_save(fig, plot_root / "geoagent_task_heatmap.png", bottom=.04))

    for key, comparison in comparisons.items():
        deltas = (comparison or {}).get("per_leaf_task_delta") or {}
        if not deltas:
            continue
        frame = pd.DataFrame([
            {"Task": _short_leaf(leaf), "Delta": 100 * float(value)}
            for leaf, value in deltas.items()
        ]).sort_values("Delta")
        fig, axis = plt.subplots(figsize=(8.3, 8.8))
        colors = ["#D64B4B" if value < 0 else "#2EAD4A" for value in frame["Delta"]]
        sns.barplot(data=frame, y="Task", x="Delta", palette=colors, hue="Task", legend=False, ax=axis)
        axis.axvline(0, color="black", linewidth=1)
        axis.set(
            xlabel=(
                f"{_condition_label(str(comparison.get('second_condition')))} - "
                f"{_condition_label(str(comparison.get('first_condition')))} (percentage points)"
            ),
            ylabel="Task",
        )
        _bold(axis)
        created.append(_save(fig, plot_root / f"delta_{key}.png", bottom=.04))

    if bloom_rows:
        frame = pd.DataFrame(bloom_rows)
        frame["Bloom level"] = pd.Categorical(frame["Bloom level"], BLOOM_ORDER, ordered=True)
        fig, axis = plt.subplots(figsize=(8.2, 5.1))
        sns.lineplot(data=frame.sort_values("Bloom level"), x="Bloom level", y="Score",
                     hue="Condition", marker="o", linewidth=2.4,
                     palette=PALETTE[:frame["Condition"].nunique()], ax=axis)
        axis.set(xlabel="Bloom level", ylabel="Task-aware score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=min(3, frame["Condition"].nunique()))
        created.append(_save(fig, plot_root / "geoagent_bloom_profile.png"))
    return created


def _condition_label(condition: str) -> str:
    return {
        "base": "Base (no retrieval)",
        "multimodal_rag": "v2.2 multimodal RAG",
        "agentic_multimodal_rag": "v2.2 agentic RAG",
        "geoagent_tool_rag": "GeoAgent v3",
        "geoagent_no_tools": "GeoAgent v3 (no tools)",
    }.get(condition, condition.replace("_", " ").title())
