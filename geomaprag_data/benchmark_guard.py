from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, text_hash


_QID_RE = re.compile(r"\bQ\d+\b")
_OSM_PATH_RE = re.compile(r"\b(?:node|way|relation)[/:](\d+)\b", re.IGNORECASE)
GUARD_CACHE_VERSION = 2


@dataclass
class BenchmarkGuard:
    text_hashes: set[str] = field(default_factory=set)
    coords: list[tuple[float, float]] = field(default_factory=list)
    osm_ids: set[str] = field(default_factory=set)
    qids: set[str] = field(default_factory=set)
    geoname_ids: set[str] = field(default_factory=set)
    commons_page_ids: set[str] = field(default_factory=set)
    worldbank_observations: set[str] = field(default_factory=set)
    source_files: list[str] = field(default_factory=list)
    spatial_exclusion_km: float = 2.0

    def near(self, lat: Any, lon: Any) -> bool:
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            return False
        for other_lat, other_lon in self.coords:
            if haversine_km(lat_f, lon_f, other_lat, other_lon) <= self.spatial_exclusion_km:
                return True
        return False

    def reject_text(self, text: str) -> bool:
        return bool(text) and text_hash(text) in self.text_hashes

    def to_json(self) -> dict[str, Any]:
        return {
            "text_hashes": sorted(self.text_hashes),
            "coords": [[lat, lon] for lat, lon in self.coords],
            "osm_ids": sorted(self.osm_ids),
            "qids": sorted(self.qids),
            "geoname_ids": sorted(self.geoname_ids),
            "commons_page_ids": sorted(self.commons_page_ids),
            "worldbank_observations": sorted(self.worldbank_observations),
            "source_files": self.source_files,
            "spatial_exclusion_km": self.spatial_exclusion_km,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BenchmarkGuard":
        return cls(
            text_hashes=set(payload.get("text_hashes", [])),
            coords=[(float(a), float(b)) for a, b in payload.get("coords", [])],
            osm_ids=set(str(x) for x in payload.get("osm_ids", [])),
            qids=set(str(x) for x in payload.get("qids", [])),
            geoname_ids=set(str(x) for x in payload.get("geoname_ids", [])),
            commons_page_ids=set(str(x) for x in payload.get("commons_page_ids", [])),
            worldbank_observations=set(str(x) for x in payload.get("worldbank_observations", [])),
            source_files=list(payload.get("source_files", [])),
            spatial_exclusion_km=float(payload.get("spatial_exclusion_km", 2.0)),
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _maybe_coord(obj: dict[str, Any], guard: BenchmarkGuard) -> None:
    lower = {str(k).lower(): v for k, v in obj.items()}
    lat = next((lower[k] for k in ("lat", "latitude") if k in lower), None)
    lon = next((lower[k] for k in ("lon", "lng", "longitude") if k in lower), None)
    try:
        if lat is not None and lon is not None:
            lat_f, lon_f = float(lat), float(lon)
            if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
                guard.coords.append((round(lat_f, 6), round(lon_f, 6)))
    except Exception:
        pass


def _scan(obj: Any, guard: BenchmarkGuard, key_hint: str = "") -> None:
    if isinstance(obj, dict):
        _maybe_coord(obj, guard)
        source_obj = obj.get("source")
        group_id = obj.get("group_id")
        if isinstance(source_obj, dict) and group_id is not None:
            source_name = str(source_obj.get("name") or "").lower()
            if "geonames" in source_name:
                guard.geoname_ids.add(str(group_id))
            if "openstreetmap" in source_name or "overpass" in source_name:
                for match in _OSM_PATH_RE.finditer(str(group_id)):
                    guard.osm_ids.add(match.group(1))
            if "wikimedia commons" in source_name:
                tail = str(group_id).rsplit(":", 1)[-1]
                if tail.isdigit():
                    guard.commons_page_ids.add(tail)
            if "world development indicators" in source_name or "world bank" in source_name:
                # GeoMapBench WDI group ids are indicator:year:ISO3 or
                # indicator:year:ISO3_A:ISO3_B. Exclude every exact
                # country-indicator-year observation referenced by an
                # evaluation record while retaining the broader public WDI
                # corpus. This is a stricter anti-contamination policy than
                # exact question-text matching alone.
                parts = str(group_id).split(":")
                if len(parts) >= 3 and parts[1].isdigit():
                    indicator, year = parts[0], parts[1]
                    for country_code in parts[2:]:
                        code = country_code.strip().upper()
                        if re.fullmatch(r"[A-Z]{3}", code):
                            guard.worldbank_observations.add(f"{indicator}:{year}:{code}")
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in {"osm_id", "osm_way_id", "osm_node_id", "osm_relation_id", "way_id", "node_id", "relation_id"}:
                guard.osm_ids.add(str(value))
            if key_lower in {"geonameid", "geonames_id", "geoname_id"}:
                guard.geoname_ids.add(str(value))
            if key_lower in {"qid", "wikidata_id", "wikidata_qid"} and isinstance(value, (str, int)):
                guard.qids.add(str(value))
            _scan(value, guard, key_lower)
    elif isinstance(obj, list):
        # Some records store coordinates as [lon, lat]. Only accept when the key
        # explicitly signals coordinate content to avoid treating arbitrary pairs
        # (pixel positions, values, etc.) as geographic locations.
        if key_hint in {"coordinates", "coordinate", "center", "centroid"} and len(obj) == 2:
            try:
                lon, lat = float(obj[0]), float(obj[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    guard.coords.append((round(lat, 6), round(lon, 6)))
            except Exception:
                pass
        for value in obj:
            _scan(value, guard, key_hint)
    elif isinstance(obj, str):
        if len(obj) >= 80:
            guard.text_hashes.add(text_hash(obj))
        guard.qids.update(_QID_RE.findall(obj))
        for match in _OSM_PATH_RE.finditer(obj):
            guard.osm_ids.add(match.group(1))


def build_guard(
    benchmark_root: Path | None,
    cache_path: Path,
    *,
    spatial_exclusion_km: float = 2.0,
    refresh: bool = False,
) -> BenchmarkGuard:
    if benchmark_root is None:
        return BenchmarkGuard(spatial_exclusion_km=spatial_exclusion_km)
    root = Path(benchmark_root).expanduser().resolve()
    if not root.exists():
        print(f"Benchmark guard: root not found, continuing without exclusions: {root}")
        return BenchmarkGuard(spatial_exclusion_km=spatial_exclusion_km)

    files = sorted(root.glob("*/data.jsonl"))
    if not files:
        files = sorted(root.rglob("data.jsonl"))
    signature = {
        str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }
    if cache_path.exists() and not refresh:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_version") == GUARD_CACHE_VERSION and payload.get("benchmark_signature") == signature:
                return BenchmarkGuard.from_json(payload["guard"])
        except Exception:
            pass

    guard = BenchmarkGuard(spatial_exclusion_km=spatial_exclusion_km)
    for path in files:
        guard.source_files.append(str(path.relative_to(root)))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    _scan(json.loads(line), guard)

    guard.coords = sorted(set(guard.coords))
    payload = {
        "cache_version": GUARD_CACHE_VERSION,
        "benchmark_root": str(root),
        "benchmark_signature": signature,
        "guard": guard.to_json(),
    }
    atomic_write_json(cache_path, payload)
    print(
        "Benchmark guard:",
        f"{len(guard.text_hashes)} text hashes,",
        f"{len(guard.coords)} coordinates,",
        f"{len(guard.osm_ids)} OSM ids,",
        f"{len(guard.qids)} Wikidata QIDs,",
        f"{len(guard.geoname_ids)} GeoNames ids,",
        f"{len(guard.commons_page_ids)} Commons page ids,",
        f"{len(guard.worldbank_observations)} World Bank observations",
    )
    return guard
