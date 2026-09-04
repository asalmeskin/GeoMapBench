"""Structured views over the frozen GeoMapRAG corpus.

The 180,344-record text index is already loaded in memory for dense retrieval.
Two extra passes over that same list give the agent structured tools that
embedding search cannot provide:

* an exact ``(indicator, country, year)`` table parsed from the World Bank
  observation sentences, and
* a gazetteer keyed on entry titles and coordinates.

Only record *indices* are stored, so the extra memory is a few tens of MB.
The frozen corpus already withholds every observation, OSM/GeoNames/Wikidata id
and 2 km neighbourhood used by the benchmark, so nothing here can leak an
answer; missing observations are reported as missing and estimated openly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

WDI_SENTENCE = re.compile(
    r"^In (?P<year>\d{4}), (?P<country>.+?) had (?P<label>.+?) of "
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) (?P<unit>.+?)\. "
    r"World Bank indicator: (?P<code>[A-Z0-9.]+)\."
)

GAZETTEER_DOCUMENT_TYPES = {
    "gazetteer_entry", "wikidata_settlement", "wikidata_mountain",
    "wikidata_river", "wikidata_lake", "wikidata_island", "wikidata_protected_area",
}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return text


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    first, second = math.radians(lat1), math.radians(lat2)
    dlat = second - first
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(first) * math.cos(second) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


class StructuredCorpusIndex:
    """Indicator table plus gazetteer built from the frozen corpus metadata."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.indicator: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.by_name: dict[str, list[int]] = {}
        self.points: list[tuple[float, float, int]] = []
        indicator_rows = gazetteer_rows = 0
        for index, record in enumerate(records):
            inp = record.get("input") or {}
            text = str(inp.get("text") or "")
            document_type = str((record.get("retrieval") or {}).get("document_type") or "")
            if document_type == "country_indicator_observation" or "World Bank indicator:" in text:
                match = WDI_SENTENCE.match(text.strip())
                if match:
                    key = (match.group("code"), normalize_name(match.group("country")))
                    try:
                        value = float(match.group("value"))
                    except ValueError:
                        value = None
                    if value is not None:
                        self.indicator.setdefault(key, []).append({
                            "year": int(match.group("year")),
                            "value": value,
                            "unit": match.group("unit"),
                            "country": match.group("country"),
                            "indicator": match.group("code"),
                            "record_id": str(record.get("id")),
                        })
                        indicator_rows += 1
                    continue
            geo = inp.get("geo") if isinstance(inp.get("geo"), dict) else None
            title = str(inp.get("title") or "").strip()
            if geo and title:
                lat, lon = geo.get("lat"), geo.get("lon")
                try:
                    lat_f, lon_f = float(lat), float(lon)
                except (TypeError, ValueError):
                    continue
                if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
                    continue
                self.by_name.setdefault(normalize_name(title), []).append(index)
                self.points.append((lat_f, lon_f, index))
                gazetteer_rows += 1
        for rows in self.indicator.values():
            rows.sort(key=lambda row: row["year"])
        self.stats = {
            "indicator_series": len(self.indicator),
            "indicator_observations": indicator_rows,
            "gazetteer_entries": gazetteer_rows,
            "distinct_gazetteer_names": len(self.by_name),
        }

    # -- indicators ------------------------------------------------------------

    def indicator_series(self, code: str, country: str) -> list[dict[str, Any]]:
        return list(self.indicator.get((str(code), normalize_name(country)), []))

    @staticmethod
    def interpolate(series: list[dict[str, Any]], year: int) -> dict[str, Any] | None:
        """Linear interpolation between the bracketing years, else nearest carry."""
        if not series:
            return None
        exact = next((row for row in series if row["year"] == year), None)
        if exact is not None:
            return {"value": exact["value"], "method": "exact corpus observation", "year": year}
        before = [row for row in series if row["year"] < year]
        after = [row for row in series if row["year"] > year]
        if before and after:
            low, high = before[-1], after[0]
            span = high["year"] - low["year"]
            weight = (year - low["year"]) / span if span else 0.0
            return {
                "value": low["value"] + weight * (high["value"] - low["value"]),
                "method": f"linear interpolation between {low['year']} and {high['year']}",
                "year": year,
            }
        nearest = (before[-1] if before else after[0])
        return {
            "value": nearest["value"],
            "method": f"nearest available year {nearest['year']} carried forward",
            "year": year,
        }

    # -- gazetteer -------------------------------------------------------------

    def _entry(self, index: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        record = self.records[index]
        inp = record.get("input") or {}
        geo = inp.get("geo") or {}
        entry = {
            "id": str(record.get("id")),
            "title": str(inp.get("title") or ""),
            "source": str(record.get("source") or ""),
            "document_type": str((record.get("retrieval") or {}).get("document_type") or ""),
            "lat": float(geo.get("lat")),
            "lon": float(geo.get("lon")),
            "text": str(inp.get("text") or "")[:600],
        }
        entry.update(extra or {})
        return entry

    def locate(self, name: str) -> dict[str, Any] | None:
        """Best gazetteer match for a name, preferring the most prominent entry."""
        key = normalize_name(name)
        indices = self.by_name.get(key)
        if not indices:
            stripped = re.sub(r"^(area around|the)\s+", "", key).strip()
            indices = self.by_name.get(stripped)
        if not indices:
            return None
        ranked = sorted(
            indices,
            key=lambda index: (
                0 if "wikidata" in str((self.records[index].get("retrieval") or {}).get("document_type") or "") else 1,
                -len(str((self.records[index].get("input") or {}).get("text") or "")),
            ),
        )
        return self._entry(ranked[0], {"candidate_count": len(indices)})

    def near(self, lat: float, lon: float, *, limit: int = 5, max_km: float = 300.0) -> list[dict[str, Any]]:
        window = [
            (haversine_km(lat, lon, point_lat, point_lon), index)
            for point_lat, point_lon, index in self.points
            if abs(point_lat - lat) <= 3.0 and abs(point_lon - lon) <= 3.0
        ]
        window.sort(key=lambda pair: pair[0])
        return [
            self._entry(index, {"distance_km": distance})
            for distance, index in window[:limit] if distance <= max_km
        ]
