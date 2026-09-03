from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = ["#2878B5", "#F28522", "#2EAD4A", "#D64B4B", "#7B61A8", "#18A999"]
BLOOM_ORDER = ["R", "U", "Ap", "An", "E", "C"]


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": True,
        "figure.facecolor": "white",
    })


def _bold(axis: plt.Axes) -> None:
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontweight("bold")
    axis.grid(axis="y", linestyle="--", linewidth=.7, alpha=.55)


def _legend_bottom(fig: plt.Figure, axis: plt.Axes, *, columns: int) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if axis.legend_ is not None:
        axis.legend_.remove()
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(.5, -.03),
        ncol=max(1, columns), title=None, frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _save(fig: plt.Figure, path: Path, *, bottom: float = .2) -> str:
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path.name


def _summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_plots(output: Path, model_rows: list[dict[str, Any]]) -> list[str]:
    """Create the three publication figures for the model benchmark."""
    _theme()
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    display = {str(row["model"]): str(row.get("family") or row["model"]) for row in model_rows}
    long_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    bloom_rows: list[dict[str, Any]] = []
    for row in model_rows:
        model = str(row["model"])
        analysis_path = output / _slug(model) / "analysis" / "summary.json"
        if not analysis_path.is_file():
            continue
        report = _summary(analysis_path)
        for leaf in report.get("per_leaf", []):
            reliability = 1.0 - max(float(leaf.get("invalid_json_rate", 0)), float(leaf.get("generation_failure_rate", 0)))
            for metric, value in (
                ("Task-aware", leaf.get("task_aware_score")),
                ("Strict", leaf.get("strict_accuracy")),
                ("Reliability", reliability),
            ):
                if value is not None:
                    long_rows.append({"Model": display[model], "Metric": metric, "Score": 100 * float(value)})
            heat_rows.append({"Model": display[model], "Task": _short_leaf(str(leaf["leaf"])), "Score": 100 * float(leaf["task_aware_score"])})
        for key, value in report.get("bloom_task_score_by_condition", {}).items():
            _, level = key.split(":", 1)
            bloom_rows.append({"Model": display[model], "Bloom level": level, "Score": 100 * float(value)})

    created: list[str] = []
    if long_rows:
        frame = pd.DataFrame(long_rows)
        fig, axis = plt.subplots(figsize=(max(8.5, len(display) * 1.35), 5.2))
        sns.barplot(data=frame, x="Model", y="Score", hue="Metric", palette=PALETTE[:3], errorbar=("ci", 95), capsize=.08, ax=axis)
        axis.set(xlabel="Model", ylabel="Score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=3)
        created.append(_save(fig, plot_root / "benchmark_overview.png"))
    if heat_rows:
        pivot = pd.DataFrame(heat_rows).pivot(index="Task", columns="Model", values="Score")
        fig, axis = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.3), 9.2))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, linewidths=.45, cbar_kws={"label": "Score (%)"}, ax=axis)
        axis.set(xlabel="Model", ylabel="Task")
        _bold(axis)
        colorbar = axis.collections[0].colorbar
        colorbar.ax.yaxis.label.set_fontweight("bold")
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        for text in axis.texts:
            text.set_fontweight("bold")
        created.append(_save(fig, plot_root / "benchmark_task_heatmap.png", bottom=.04))
    if bloom_rows:
        frame = pd.DataFrame(bloom_rows)
        frame["Bloom level"] = pd.Categorical(frame["Bloom level"], BLOOM_ORDER, ordered=True)
        fig, axis = plt.subplots(figsize=(9.4, 5.2))
        sns.lineplot(data=frame.sort_values("Bloom level"), x="Bloom level", y="Score", hue="Model", marker="o", linewidth=2.2, palette=PALETTE[:len(display)], ax=axis)
        axis.set(xlabel="Bloom level", ylabel="Task-aware score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=min(3, len(display)))
        created.append(_save(fig, plot_root / "benchmark_bloom_profile.png", bottom=.25))
    return created


def rag_plots(output: Path, analyses: dict[str, dict[str, Any]], comparison: dict[str, Any] | None) -> list[str]:
    """Create matched multimodal/agentic-multimodal paper figures."""
    _theme()
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, Any]] = []
    bloom_rows: list[dict[str, Any]] = []
    for condition, report in analyses.items():
        label = condition.replace("_", " ").title()
        for leaf in report.get("per_leaf", []):
            reliability = 1.0 - max(float(leaf.get("invalid_json_rate", 0)), float(leaf.get("generation_failure_rate", 0)))
            for metric, value in (
                ("Task-aware", leaf.get("task_aware_score")),
                ("Strict", leaf.get("strict_accuracy")),
                ("Reliability", reliability),
            ):
                if value is not None:
                    long_rows.append({"Condition": label, "Metric": metric, "Score": 100 * float(value)})
        for key, value in report.get("bloom_task_score_by_condition", {}).items():
            _, level = key.split(":", 1)
            bloom_rows.append({"Condition": label, "Bloom level": level, "Score": 100 * float(value)})

    created: list[str] = []
    if long_rows:
        fig, axis = plt.subplots(figsize=(7.8, 5.1))
        sns.barplot(data=pd.DataFrame(long_rows), x="Condition", y="Score", hue="Metric", palette=PALETTE[:3], errorbar=("ci", 95), capsize=.08, ax=axis)
        axis.set(xlabel="RAG condition", ylabel="Score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=3)
        created.append(_save(fig, plot_root / "rag_overview.png"))
    if comparison and comparison.get("per_leaf_task_delta"):
        frame = pd.DataFrame([
            {"Task": _short_leaf(leaf), "Delta": 100 * float(value)}
            for leaf, value in comparison["per_leaf_task_delta"].items()
        ]).sort_values("Delta")
        fig, axis = plt.subplots(figsize=(8.3, 8.8))
        colors = ["#D64B4B" if value < 0 else "#2EAD4A" for value in frame["Delta"]]
        sns.barplot(data=frame, y="Task", x="Delta", palette=colors, hue="Task", legend=False, ax=axis)
        axis.axvline(0, color="black", linewidth=1)
        axis.set(xlabel="Agentic multimodal − multimodal (percentage points)", ylabel="Task")
        _bold(axis)
        created.append(_save(fig, plot_root / "rag_task_delta.png", bottom=.04))
    if bloom_rows:
        frame = pd.DataFrame(bloom_rows)
        frame["Bloom level"] = pd.Categorical(frame["Bloom level"], BLOOM_ORDER, ordered=True)
        fig, axis = plt.subplots(figsize=(8.2, 5.1))
        sns.lineplot(data=frame.sort_values("Bloom level"), x="Bloom level", y="Score", hue="Condition", marker="o", linewidth=2.4, palette=PALETTE[:2], ax=axis)
        axis.set(xlabel="Bloom level", ylabel="Task-aware score (%)", ylim=(0, 100))
        _bold(axis); _legend_bottom(fig, axis, columns=2)
        created.append(_save(fig, plot_root / "rag_bloom_profile.png"))
    return created


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _short_leaf(value: str) -> str:
    replacements = {
        "cartographic_symbol_recognition": "Cartographic symbols",
        "map_text_detection_recognition_grouping": "Map text",
        "map_label_feature_anchoring": "Label anchoring",
        "dense_land_cover_labeling": "Land cover",
        "remote_sensing_scene_classification": "Remote-sensing scene",
        "object_presence_counting": "Object counting",
        "change_localization": "Change localization",
        "temporal_scene_matching": "Temporal matching",
        "visual_geolocation": "Visual geolocation",
        "coordinate_transformation": "Coordinate transform",
        "metric_distance_computation": "Metric distance",
        "topological_directional_reasoning": "Topology/direction",
        "spatial_graph_construction": "Spatial graph",
        "shortest_path_optimization": "Shortest path",
        "isochrone_service_area": "Isochrone",
        "toponym_recognition": "Toponym recognition",
        "geo_entity_typing": "Entity typing",
        "textual_spatial_relation_extraction": "Spatial relations",
        "cross_entity_comparison": "Entity comparison",
        "environmental_layer_identification": "Environmental layer",
        "population_density_estimation": "Population density",
        "geologic_geomorphic_interpretation": "Geology/geomorphology",
        "geographic_fact_reasoning": "Geographic facts",
    }
    return replacements.get(value, value.replace("_", " ").title())
