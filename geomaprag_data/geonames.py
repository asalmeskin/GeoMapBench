from __future__ import annotations

import csv
import hashlib
import heapq
import io
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, atomic_write_jsonl, make_record, read_jsonl
from .config import (
    BuildProfile,
    CAPABILITY_HINTS,
    GEONAMES_COUNTRIES,
    GEONAMES_FEATURE_CLASSES,
)
from .http import CachedHTTP


GEONAMES_DUMP_BASE = "https://download.geonames.org/export/dump"
GEONAMES_FEATURE_CODES_URL = f"{GEONAMES_DUMP_BASE}/featureCodes_en.txt"
GEONAMES_COUNTRY_INFO_URL = f"{GEONAMES_DUMP_BASE}/countryInfo.txt"

_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]


def _feature_codes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) >= 2:
                description = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
                out[parts[0]] = f"{parts[1]} — {description}"
    return out


def _country_names(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) >= 5:
                out[parts[0]] = parts[4]
    return out


def _stable_score(geoname_id: str) -> int:
    return int(hashlib.sha256(f"geomaprag-geonames:{geoname_id}".encode("utf-8")).hexdigest(), 16)


def _push_smallest(
    heap: list[tuple[int, str, dict[str, Any]]],
    row: dict[str, Any],
    limit: int,
) -> None:
    # Python heapq is a min-heap. Store -score so heap[0] is the worst (largest)
    # score among the retained deterministic candidates.
    score = _stable_score(str(row["geonameid"]))
    item = (-score, str(row["geonameid"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _iter_archive_rows(archive: Path) -> Any:
    with zipfile.ZipFile(archive) as zf:
        txt_names = [name for name in zf.namelist() if name.endswith(".txt") and not name.lower().endswith("readme.txt")]
        if len(txt_names) != 1:
            raise ValueError(f"Could not identify a single GeoNames data file in {archive}: {txt_names}")
        with zf.open(txt_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            for parts in csv.reader(text, delimiter="\t"):
                if len(parts) >= len(_COLUMNS):
                    yield dict(zip(_COLUMNS, parts))


def _balanced_selection(
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    total: int,
) -> list[dict[str, Any]]:
    """Deterministically balance across feature classes and countries.

    The first pass allocates an equal target to all nine GeoNames feature
    classes. If a rare class cannot fill its quota, a second round-robin pass
    redistributes the unused capacity instead of returning a needlessly small
    corpus.
    """
    from collections import deque

    classes = sorted(GEONAMES_FEATURE_CLASSES)
    base, remainder = divmod(total, len(classes))
    targets = {feature_class: base + (1 if index < remainder else 0) for index, feature_class in enumerate(classes)}

    queues: dict[tuple[str, str], deque[dict[str, Any]]] = {}
    countries_by_class: dict[str, list[str]] = defaultdict(list)
    for (class_code, country), rows in pools.items():
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: (_stable_score(str(row["geonameid"])), str(row["geonameid"])))
        queues[(class_code, country)] = deque(ordered)
        countries_by_class[class_code].append(country)
    for class_code in countries_by_class:
        countries_by_class[class_code] = sorted(set(countries_by_class[class_code]))

    selected: list[dict[str, Any]] = []
    class_counts: dict[str, int] = defaultdict(int)

    # Pass 1: equal class targets, round-robin over countries within each class.
    for feature_class in classes:
        countries = countries_by_class.get(feature_class, [])
        if not countries:
            continue
        cursor = 0
        empty_rounds = 0
        while class_counts[feature_class] < targets[feature_class] and empty_rounds < len(countries):
            country = countries[cursor % len(countries)]
            queue = queues[(feature_class, country)]
            if queue:
                selected.append(queue.popleft())
                class_counts[feature_class] += 1
                empty_rounds = 0
            else:
                empty_rounds += 1
            cursor += 1

    # Pass 2: redistribute any quota left by rare feature classes. Rotate over
    # classes and countries so abundant P/A/S entries cannot dominate.
    class_cursor = 0
    country_cursor: dict[str, int] = defaultdict(int)
    inactive_rounds = 0
    while len(selected) < total and inactive_rounds < len(classes):
        feature_class = classes[class_cursor % len(classes)]
        countries = countries_by_class.get(feature_class, [])
        added = False
        if countries:
            for _ in range(len(countries)):
                idx = country_cursor[feature_class] % len(countries)
                country_cursor[feature_class] += 1
                queue = queues[(feature_class, countries[idx])]
                if queue:
                    selected.append(queue.popleft())
                    class_counts[feature_class] += 1
                    added = True
                    break
        inactive_rounds = 0 if added else inactive_rounds + 1
        class_cursor += 1

    selected.sort(key=lambda row: (row["feature_class"], row["country_code"], _stable_score(str(row["geonameid"]))))
    return selected[:total]



def _guard_fingerprint(guard: BenchmarkGuard) -> str:
    payload = "|".join(
        [
            f"km={guard.spatial_exclusion_km}",
            *sorted(f"g:{x}" for x in guard.geoname_ids),
            *sorted(f"c:{lat:.6f},{lon:.6f}" for lat, lon in guard.coords),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _country_candidate_cache(
    workspace: CorpusWorkspace,
    country: str,
    pool_limit: int,
    guard: BenchmarkGuard,
) -> Path:
    return (
        workspace.cache_dir
        / "geonames"
        / "candidate_shards"
        / f"{country}_pool{pool_limit}_{_guard_fingerprint(guard)}.jsonl"
    )

def build_geonames(workspace: CorpusWorkspace, profile: BuildProfile, guard: BenchmarkGuard) -> dict[str, Any]:
    unit = f"country_dumps_9class_balanced_{profile.geonames_max_records}"
    if workspace.shard_done("geonames", unit):
        return {"stage": "geonames", "written": 0, "cached_units": 1, "failed_units": []}

    http = CachedHTTP(workspace.cache_dir)
    download_dir = workspace.cache_dir / "geonames" / "downloads"
    feature_codes_path = http.download(GEONAMES_FEATURE_CODES_URL, download_dir / "featureCodes_en.txt", timeout=120)
    country_info_path = http.download(GEONAMES_COUNTRY_INFO_URL, download_dir / "countryInfo.txt", timeout=120)
    feature_codes = _feature_codes(feature_codes_path)
    country_names = _country_names(country_info_path)

    # Keep a bounded deterministic candidate pool for each (feature class,
    # country), so even large country dumps do not consume unbounded RAM.
    target_per_class = math.ceil(profile.geonames_max_records / len(GEONAMES_FEATURE_CLASSES))
    per_country_class_pool = max(80, math.ceil(target_per_class / max(1, len(GEONAMES_COUNTRIES))) * 4)
    heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)

    country_bar = tqdm(GEONAMES_COUNTRIES, desc="GeoNames country dumps", unit="country", dynamic_ncols=True)
    for country in country_bar:
        candidate_cache = _country_candidate_cache(workspace, country, per_country_class_pool, guard)
        if candidate_cache.exists():
            cached_rows = read_jsonl(candidate_cache, tolerate_trailing_partial=True)
            for row in cached_rows:
                feature_class = str(row.get("feature_class") or "")
                if feature_class in GEONAMES_FEATURE_CLASSES:
                    _push_smallest(heaps[(feature_class, country)], row, per_country_class_pool)
            country_bar.set_postfix(country=country, status="cached", candidates=len(cached_rows))
            continue

        archive = http.download(
            f"{GEONAMES_DUMP_BASE}/{country}.zip",
            download_dir / f"{country}.zip",
            timeout=300,
            max_attempts=6,
        )
        local_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
        accepted = 0
        for row in _iter_archive_rows(archive):
            feature_class = row.get("feature_class")
            if feature_class not in GEONAMES_FEATURE_CLASSES or not row.get("name"):
                continue
            geoname_id = str(row.get("geonameid") or "")
            if not geoname_id or geoname_id in guard.geoname_ids:
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except Exception:
                continue
            if guard.near(lat, lon):
                continue
            _push_smallest(local_heaps[feature_class], row, per_country_class_pool)
            accepted += 1

        cached_rows: list[dict[str, Any]] = []
        for feature_class, heap in sorted(local_heaps.items()):
            rows = [row for _, _, row in heap]
            rows.sort(key=lambda row: (_stable_score(str(row["geonameid"])), str(row["geonameid"])))
            cached_rows.extend(rows)
            for row in rows:
                _push_smallest(heaps[(feature_class, country)], row, per_country_class_pool)
        atomic_write_jsonl(candidate_cache, cached_rows)
        country_bar.set_postfix(country=country, status="processed", candidates=len(cached_rows), scanned=accepted)

    pools: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [row for _, _, row in heap]
        for key, heap in heaps.items()
    }
    selected = _balanced_selection(pools, profile.geonames_max_records)
    existing = workspace.existing_ids()
    records: list[dict[str, Any]] = []

    bar = tqdm(selected, desc="GeoNames balanced entries", unit="place", dynamic_ncols=True)
    for row in bar:
        geoname_id = str(row["geonameid"])
        record_id = f"geonames:{geoname_id}"
        if record_id in existing:
            continue
        lat, lon = float(row["latitude"]), float(row["longitude"])
        name = row["name"]
        country_name = country_names.get(row["country_code"], row["country_code"])
        feature_key = f"{row['feature_class']}.{row['feature_code']}"
        feature_description = feature_codes.get(feature_key, feature_key)
        class_description = GEONAMES_FEATURE_CLASSES[row["feature_class"]]
        population = int(row.get("population") or 0)
        alternates = [x for x in (row.get("alternatenames") or "").split(",") if x][:12]
        text = (
            f"{name} is a GeoNames {class_description} in {country_name}. "
            f"Feature class {row['feature_class']}; feature code {feature_key}: {feature_description}. "
            f"Coordinates: latitude {lat:.6f}, longitude {lon:.6f}. "
            f"Population field: {population}. Time zone: {row.get('timezone') or 'unknown'}."
        )
        if alternates:
            text += " Alternate names include: " + ", ".join(alternates) + "."
        if guard.reject_text(text):
            continue
        records.append(
            make_record(
                record_id=record_id,
                source_name="GeoNames",
                source_url=f"https://www.geonames.org/{geoname_id}/",
                license_name="CC BY 4.0",
                attribution="GeoNames geographical database",
                group_id=geoname_id,
                modality="structured",
                title=name,
                text=text,
                source_id=geoname_id,
                geo={"lat": lat, "lon": lon, "country_code": row["country_code"]},
                capabilities=CAPABILITY_HINTS["GeoNames"],
                document_type="gazetteer_entry",
                generator="geomaprag_data.geonames",
                extra={
                    "feature_class": row["feature_class"],
                    "feature_class_name": class_description,
                    "feature_code": row["feature_code"],
                    "feature_description": feature_description,
                    "country": country_name,
                    "admin1_code": row.get("admin1_code"),
                    "admin2_code": row.get("admin2_code"),
                    "population": population,
                    "elevation": row.get("elevation") or None,
                    "dem": row.get("dem") or None,
                    "timezone": row.get("timezone") or None,
                    "modification_date": row.get("modification_date") or None,
                    "alternate_names": alternates,
                },
            )
        )

    class_counts: dict[str, int] = defaultdict(int)
    for record in records:
        class_counts[str(record["extra"]["feature_class"])] += 1
    workspace.write_shard(
        "geonames",
        unit,
        records,
        meta={
            "status": "complete",
            "selected": len(records),
            "feature_class_counts": dict(sorted(class_counts.items())),
            "countries": list(GEONAMES_COUNTRIES),
        },
    )
    return {"stage": "geonames", "written": len(records), "cached_units": 0, "failed_units": []}
