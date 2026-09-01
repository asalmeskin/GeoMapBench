from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .benchmark_guard import build_guard
from .common import CorpusWorkspace, atomic_write_json, utc_now
from .config import PROFILES, BuildProfile
from .epsg import build_epsg
from .geonames import build_geonames
from .osm import build_osm
from .wikipedia import build_wikipedia
from .wikidata import build_wikidata
from .worldbank import build_worldbank
from .wikimedia import build_wikimedia


StageFn = Callable[[CorpusWorkspace, BuildProfile, Any], dict[str, Any]]

STAGES: tuple[tuple[str, StageFn], ...] = (
    ("epsg", build_epsg),
    ("geonames", build_geonames),
    ("worldbank", build_worldbank),
    ("wikipedia", build_wikipedia),
    ("wikidata", build_wikidata),
    ("wikimedia", build_wikimedia),
    ("osm", build_osm),
)


def build_all(
    output: Path,
    *,
    benchmark_root: Path | None,
    profile_name: str = "publication",
    stages: set[str] | None = None,
    spatial_exclusion_km: float = 2.0,
) -> dict[str, Any]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choices={sorted(PROFILES)}")
    profile = PROFILES[profile_name]
    workspace = CorpusWorkspace(output)
    guard = build_guard(
        benchmark_root,
        workspace.cache_dir / "benchmark_guard.json",
        spatial_exclusion_km=spatial_exclusion_km,
    )

    selected = set(stages or [name for name, _ in STAGES])
    unknown = selected - {name for name, _ in STAGES}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    reports: list[dict[str, Any]] = []
    for name, fn in STAGES:
        if name not in selected:
            continue
        print("\n" + "=" * 80)
        print(f"GeoMapRAG stage: {name}")
        print("=" * 80)
        report = fn(workspace, profile, guard)
        reports.append(report)
        manifest = workspace.materialize()
        print(f"Materialized corpus: {manifest['count']} records")

    manifest = workspace.materialize()
    failures = [item for report in reports for item in report.get("failed_units", [])]
    run_report = {
        "profile": profile_name,
        "profile_config": asdict(profile),
        "selected_stages": [name for name, _ in STAGES if name in selected],
        "output": str(workspace.root),
        "benchmark_root": None if benchmark_root is None else str(Path(benchmark_root).expanduser().resolve()),
        "spatial_exclusion_km": spatial_exclusion_km,
        "completed_at": utc_now(),
        "stages": reports,
        "materialized_manifest": manifest,
        "failed_unit_count": len(failures),
        "resume_instruction": "Rerun the identical build command. Completed atomic shards and cached responses will be reused.",
    }
    atomic_write_json(workspace.state_dir / "last_build.json", run_report)
    return run_report
