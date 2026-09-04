"""Deterministic geospatial toolbelt.

Every tool is a pure function of a :class:`~geoagent.taskview.TaskView` (plus,
for the two corpus tools, the frozen GeoMapRAG corpus). No tool can see a gold
answer, so a tool result is a *derivation from the same information the base
model receives* -- it is the agent computing instead of guessing.

Tools declare an ``authority``:

``exact``     closed-form from the structured task input; the answer model is
              told to use the value verbatim, and the repair stage will
              substitute it into a scalar answer that disagrees.
``strong``    derived from the frozen corpus or from geometry; used as evidence.
``advisory``  a hint, a constraint or a consistency check.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    name: str
    status: str  # ok | skipped | error
    authority: str = "advisory"  # exact | strong | advisory
    title: str = ""
    text: str = ""
    value: Any = None
    primary_key: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def trace(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "status": self.status,
            "authority": self.authority,
            "primary_key": self.primary_key,
            "value": _jsonable(self.value),
            "detail": _jsonable(self.detail),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _skip(name: str, reason: str = "not_applicable") -> ToolResult:
    return ToolResult(name=name, status="skipped", detail={"reason": reason})


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value.replace(",", ""))
        if match:
            try:
                result = float(match.group())
            except ValueError:
                return None
            return result if math.isfinite(result) else None
    return None


def _fmt(value: float, decimals: int = 6) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{round(value, decimals):.{decimals}f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Unit tables (mirrors geomapbench_data.static_generators.DISTANCE_UNITS)
# ---------------------------------------------------------------------------

UNIT_SYMBOL = {"metres": "m", "kilometres": "km", "miles": "mi", "nautical_miles": "nmi"}
UNIT_METRES = {"metres": 1.0, "kilometres": 1000.0, "miles": 1609.344, "nautical_miles": 1852.0}
UNIT_DECIMALS = {"metres": 1, "kilometres": 3, "miles": 3, "nautical_miles": 3}

CARDINALS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]

WDI_LABEL_TO_CODE = {
    "population density": "EN.POP.DNST",
    "total population": "SP.POP.TOTL",
    "gdp per capita": "NY.GDP.PCAP.CD",
    "forest area share": "AG.LND.FRST.ZS",
}


def _geod():
    from pyproj import Geod

    return Geod(ellps="WGS84")


def _segment_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Exactly the benchmark's own convention (network_generators._segment_length)."""
    if all(abs(v) <= 180 for v in (a[0], a[1], b[0], b[1])):
        try:
            _, _, metres = _geod().inv(a[0], a[1], b[0], b[1])
            return float(abs(metres))
        except Exception:
            pass
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        lon = _num(value.get("longitude", value.get("lon", value.get("x"))))
        lat = _num(value.get("latitude", value.get("lat", value.get("y"))))
        return None if lon is None or lat is None else (lon, lat)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = _num(value[0]), _num(value[1])
        return None if first is None or second is None else (first, second)
    return None


def cardinal_from_bearing(dx: float, dy: float) -> str:
    """Matches geomapbench_data.static_generators._cardinal."""
    angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return CARDINALS[int((angle + 22.5) // 45) % 8]


# ---------------------------------------------------------------------------
# 1. Coordinate reference system transformation
# ---------------------------------------------------------------------------

_AXIS_QUESTION = re.compile(
    r"return\s+([a-z_]+)\s+and\s+([a-z_]+)\s+in\s+([a-z]+)", re.IGNORECASE
)
_CLASS_ID = re.compile(r"class\s*(?:id|index)\s*([0-9]+)", re.IGNORECASE)


def _target_axis_names(target_crs: str, question: str) -> tuple[str, str]:
    match = _AXIS_QUESTION.search(question or "")
    if match:
        first, second = match.group(1).lower(), match.group(2).lower()
        if {first, second} in (
            {"longitude", "latitude"}, {"easting", "northing"}, {"x", "y"},
        ):
            return first, second
    text = str(target_crs).upper()
    if "4326" in text:
        return "longitude", "latitude"
    if "3857" in text:
        return "x", "y"
    if re.search(r"32[67]\d\d", text):
        return "easting", "northing"
    return "x", "y"


def tool_crs_transform(view) -> ToolResult:
    name = "crs_transform"
    source_crs = view.get("source_crs")
    target_crs = view.get("target_crs")
    if not source_crs or not target_crs:
        return _skip(name)
    try:
        from pyproj import CRS, Transformer
    except Exception as error:  # pragma: no cover - environment guard
        return ToolResult(name=name, status="error", detail={"error": repr(error)})
    try:
        target = CRS.from_user_input(str(target_crs))
        geographic = bool(target.is_geographic)
        unit = "degrees" if geographic else "metres"
        precision = 7 if geographic else 3
        axis_x, axis_y = _target_axis_names(str(target_crs), view.question)
        detail: dict[str, Any] = {
            "target_crs": str(target_crs),
            "target_crs_kind": "geographic" if geographic else "projected",
            "target_unit": unit,
            "axis_order": [axis_x, axis_y],
        }
        lines = [
            f"Target CRS {target_crs} is {detail['target_crs_kind']}; "
            f"its coordinate unit is {unit}; axis order is {axis_x} then {axis_y}."
        ]
        value: dict[str, Any] = {
            "target_crs_kind": detail["target_crs_kind"],
            "target_unit": unit,
        }
        coordinate = view.get("coordinate")
        if isinstance(coordinate, dict):
            axes = coordinate.get("axis_order")
            if not isinstance(axes, list) or len(axes) < 2:
                axes = [key for key in coordinate if key not in ("axis_order", "unit", "crs")][:2]
            source_x, source_y = _num(coordinate.get(axes[0])), _num(coordinate.get(axes[1]))
            if source_x is not None and source_y is not None:
                forward = Transformer.from_crs(str(source_crs), str(target_crs), always_xy=True)
                out_x, out_y = forward.transform(source_x, source_y)
                inverse = Transformer.from_crs(str(target_crs), str(source_crs), always_xy=True)
                back_x, back_y = inverse.transform(out_x, out_y)
                roundtrip = max(abs(back_x - source_x), abs(back_y - source_y))
                transformed = {
                    axis_x: round(float(out_x), precision),
                    axis_y: round(float(out_y), precision),
                    "axis_order": [axis_x, axis_y],
                    "unit": unit,
                    "crs": str(target_crs),
                }
                value["transformed_coordinate"] = transformed
                detail["roundtrip_error"] = roundtrip
                lines.append(
                    f"PROJ transformation {source_crs} -> {target_crs} of "
                    f"({_fmt(source_x)}, {_fmt(source_y)}) gives "
                    f"{axis_x}={transformed[axis_x]}, {axis_y}={transformed[axis_y]} "
                    f"(round-trip residual {roundtrip:.3e})."
                )
                candidate = view.get("candidate_coordinate")
                if isinstance(candidate, dict):
                    deltas = {
                        axis: abs((_num(candidate.get(axis)) or 0.0) - transformed[axis])
                        for axis in (axis_x, axis_y)
                        if _num(candidate.get(axis)) is not None
                    }
                    if deltas:
                        worst = max(deltas.values())
                        tolerance = 1e-5 if geographic else 1.0
                        detail["candidate_max_abs_error"] = worst
                        detail["candidate_within_task_tolerance"] = bool(worst <= tolerance)
                        value["candidate_verdict"] = "yes" if worst <= tolerance else "no"
                        lines.append(
                            f"The proposed candidate coordinate differs from the computed "
                            f"transformation by at most {worst:.6g} {unit}; the benchmark "
                            f"tolerance for a {detail['target_crs_kind']} target CRS is "
                            f"{tolerance:g}, so the candidate is "
                            f"{'correct' if worst <= tolerance else 'incorrect'} (answer "
                            f"{value['candidate_verdict']})."
                        )
        return ToolResult(
            name=name, status="ok", authority="exact",
            title="Verified CRS transformation (PROJ/EPSG)",
            text=" ".join(lines), value=value,
            primary_key="transformed_coordinate" if "transformed_coordinate" in value else None,
            detail=detail,
        )
    except Exception as error:
        return ToolResult(name=name, status="error", detail={"error": repr(error)})


# ---------------------------------------------------------------------------
# 2. WGS84 geodesic distance
# ---------------------------------------------------------------------------


def tool_geodesic_distance(view) -> ToolResult:
    name = "geodesic_distance"
    points = view.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return _skip(name)
    first, second = _point(points[0]), _point(points[1])
    if first is None or second is None:
        return _skip(name, "unparsable_points")
    try:
        _, _, metres = _geod().inv(first[0], first[1], second[0], second[1])
        metres = float(abs(metres))
        requested = str(view.get("requested_unit") or "kilometres")
        factor = UNIT_METRES.get(requested, 1000.0)
        decimals = UNIT_DECIMALS.get(requested, 3)
        converted = round(metres / factor, decimals)
        value = {
            "value": converted,
            "unit": UNIT_SYMBOL.get(requested, requested),
            "unit_id": requested,
            "distance_m": round(metres, 3),
            "distance_km": round(metres / 1000.0, 6),
            "method": "WGS84 inverse geodesic",
            "all_units": {
                key: round(metres / UNIT_METRES[key], UNIT_DECIMALS[key])
                for key in UNIT_METRES
            },
        }
        lines = [
            f"WGS84 inverse geodesic distance between the two given coordinates is "
            f"{value['distance_m']} m = {value['distance_km']} km = "
            f"{converted} {value['unit']} (requested unit: {requested}). "
            f"The reference distance method for this benchmark family is "
            f"'WGS84 inverse geodesic'."
        ]
        candidate = _num(view.get("candidate_value"))
        if candidate is not None:
            error = abs(candidate - converted) / max(abs(converted), 1e-9)
            value["candidate_relative_error"] = error
            value["candidate_verdict"] = "yes" if error <= 0.005 else "no"
            lines.append(
                f"The reviewer's candidate value {candidate:g} differs from the computed "
                f"value by {error * 100:.4f}%; the benchmark tolerance is 0.5%, so the "
                f"candidate is {'correct' if error <= 0.005 else 'incorrect'} "
                f"(answer {value['candidate_verdict']})."
            )
        return ToolResult(
            name=name, status="ok", authority="exact",
            title="Verified geodesic distance (pyproj Geod, WGS84)",
            text=" ".join(lines), value=value, primary_key="value",
            detail={"requested_unit": requested},
        )
    except Exception as error:
        return ToolResult(name=name, status="error", detail={"error": repr(error)})


# ---------------------------------------------------------------------------
# 3. Route / polyline metrics
# ---------------------------------------------------------------------------


def tool_route_metrics(view) -> ToolResult:
    name = "route_metrics"
    route = view.get("reference_route_coordinates")
    start, end = view.get("start"), view.get("end")
    have_route = isinstance(route, list) and len(route) >= 2
    have_ends = _point(start) is not None and _point(end) is not None
    if not have_route and not have_ends:
        return _skip(name)
    try:
        value: dict[str, Any] = {}
        lines: list[str] = []
        if have_ends:
            a, b = _point(start), _point(end)
            assert a is not None and b is not None
            direct = _segment_length(a, b)
            value["direct_distance_m"] = round(direct, 3)
            lines.append(
                f"Straight-line separation between the given start and end graph "
                f"coordinates is {value['direct_distance_m']} metres."
            )
        if have_route:
            points = [_point(item) for item in route]
            if any(item is None for item in points):
                return _skip(name, "unparsable_route")
            total = 0.0
            for previous, current in zip(points, points[1:]):
                assert previous is not None and current is not None
                total += _segment_length(previous, current)
            value["route_length_m"] = round(total, 3)
            value["route_vertex_count"] = len(points)
            lines.append(
                f"Summing the {len(points) - 1} consecutive segments of the supplied "
                f"reference route with the benchmark's own segment-length rule gives a "
                f"total route length of {value['route_length_m']} metres."
            )
            if "direct_distance_m" in value and value["direct_distance_m"] > 0:
                ratio = total / value["direct_distance_m"]
                value["detour_ratio"] = round(ratio, 6)
                lines.append(
                    f"Route/direct detour ratio is {value['detour_ratio']}."
                )
        candidate = _num(view.get("candidate_route_length_m"))
        if candidate is not None and "route_length_m" in value:
            error = abs(candidate - value["route_length_m"]) / max(value["route_length_m"], 1e-9)
            value["candidate_relative_error"] = error
            value["candidate_verdict"] = "yes" if error <= 0.005 else "no"
            lines.append(
                f"The reviewer's candidate route length differs by {error * 100:.4f}% "
                f"(0.5% tolerance), so the claim is "
                f"{'correct' if error <= 0.005 else 'incorrect'} "
                f"(answer {value['candidate_verdict']})."
            )
        if not value:
            return _skip(name)
        return ToolResult(
            name=name, status="ok",
            authority="exact" if have_route else "strong",
            title="Verified route geometry",
            text=" ".join(lines), value=value,
            primary_key="route_length_m" if "route_length_m" in value else None,
        )
    except Exception as error:
        return ToolResult(name=name, status="error", detail={"error": repr(error)})


# ---------------------------------------------------------------------------
# 4. Speed / time budget
# ---------------------------------------------------------------------------


def tool_service_budget(view) -> ToolResult:
    name = "service_budget"
    minutes = _num(view.get("budget_minutes"))
    speed = _num(view.get("speed_mps"))
    declared = _num(view.get("network_distance_budget_m"))
    if minutes is None or speed is None:
        return _skip(name)
    computed = minutes * 60.0 * speed
    value = {
        "budget_minutes": minutes,
        "speed_mps": speed,
        "network_distance_budget_m": round(computed, 6),
        "reachability_model": "pedestrian street network",
    }
    lines = [
        f"Maximum network travel distance = {minutes:g} min x 60 s x {speed:g} m/s = "
        f"{value['network_distance_budget_m']:g} metres."
    ]
    if declared is not None:
        value["declared_network_distance_budget_m"] = declared
        lines.append(
            f"The task input declares {declared:g} m, which "
            f"{'agrees' if abs(declared - computed) <= 1e-6 * max(1.0, abs(declared)) else 'disagrees'} "
            f"with the computed budget."
        )
    origin = _point(view.get("origin"))
    if origin is not None:
        value["origin"] = {"longitude": origin[0], "latitude": origin[1]}
    return ToolResult(
        name=name, status="ok", authority="exact",
        title="Verified speed/time budget",
        text=" ".join(lines), value=value, primary_key="network_distance_budget_m",
    )


# ---------------------------------------------------------------------------
# 5. Unit symbol recall
# ---------------------------------------------------------------------------


def tool_unit_symbol(view) -> ToolResult:
    name = "unit_symbol"
    requested = view.get("requested_unit")
    if not isinstance(requested, str) or requested not in UNIT_SYMBOL:
        return _skip(name)
    symbol = UNIT_SYMBOL[requested]
    return ToolResult(
        name=name, status="ok", authority="exact",
        title="Benchmark unit table",
        text=(
            f"In this benchmark the unit id '{requested}' is reported with the standard "
            f"symbol '{symbol}' ({_fmt(UNIT_METRES[requested])} metres per unit, rounded to "
            f"{UNIT_DECIMALS[requested]} decimals)."
        ),
        value={"unit": symbol, "unit_id": requested},
        primary_key="unit",
    )


# ---------------------------------------------------------------------------
# 6. Class-ontology lookup
# ---------------------------------------------------------------------------


def tool_class_ontology(view) -> ToolResult:
    name = "class_ontology"
    ontology = view.get("class_ontology") or view.get("ontology")
    if not isinstance(ontology, dict) or not ontology:
        return _skip(name)

    def label(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("name") or entry.get("label") or entry)
        return str(entry)

    value: dict[str, Any] = {"class_count": len(ontology)}
    lines: list[str] = []
    match = _CLASS_ID.search(view.question or "")
    if match:
        key = match.group(1)
        entry = ontology.get(key, ontology.get(str(int(key)) if key.isdigit() else key))
        if entry is not None:
            value["queried_class_id"] = key
            value["queried_class_name"] = label(entry)
            lines.append(
                f"In the declared ontology, class ID {key} is exactly "
                f"\"{value['queried_class_name']}\"."
            )
    ids = sorted(ontology, key=lambda item: int(item) if str(item).isdigit() else 10**9)
    value["valid_class_ids"] = ids
    value["id_to_name"] = {str(key): label(ontology[key]) for key in ids}
    lines.append(
        "Valid class identifiers for any produced mask or class answer: "
        + ", ".join(f"{key}={label(ontology[key])}" for key in ids)
        + "."
    )
    return ToolResult(
        name=name, status="ok",
        authority="exact" if "queried_class_name" in value else "advisory",
        title="Declared class ontology",
        text=" ".join(lines), value=value,
        primary_key="queried_class_name" if "queried_class_name" in value else None,
    )


# ---------------------------------------------------------------------------
# 7. Character-offset text span
# ---------------------------------------------------------------------------

_OFFSET_QUESTION = re.compile(r"offsets\s+(\d+)\s*:\s*(\d+)")


def tool_text_span(view) -> ToolResult:
    name = "text_span"
    text = view.get("text")
    if not isinstance(text, str) or not text:
        return _skip(name)
    offsets = view.get("query_offsets")
    start = end = None
    if isinstance(offsets, list) and len(offsets) >= 2:
        start, end = _num(offsets[0]), _num(offsets[1])
    if start is None or end is None:
        match = _OFFSET_QUESTION.search(view.question or "")
        if match:
            start, end = float(match.group(1)), float(match.group(2))
    value: dict[str, Any] = {"document_chars": len(text)}
    lines = [f"The supplied document contains exactly {len(text)} characters."]
    if start is not None and end is not None:
        lo, hi = int(start), int(end)
        if 0 <= lo < hi <= len(text):
            span = text[lo:hi]
            value["span_text"] = span
            value["span_offsets"] = [lo, hi]
            lines.append(
                f"Characters [{lo}:{hi}) of the document are exactly \"{span}\"."
            )
            return ToolResult(
                name=name, status="ok", authority="exact",
                title="Verified character-offset extraction",
                text=" ".join(lines), value=value, primary_key="span_text",
            )
    return ToolResult(
        name=name, status="ok", authority="advisory",
        title="Document statistics",
        text=(
            " ".join(lines)
            + " Character offsets must be counted from 0 over this exact string, including"
            " every space and newline; end offsets are exclusive."
        ),
        value=value,
    )


# ---------------------------------------------------------------------------
# 8. World Bank indicator lookup over the frozen corpus
# ---------------------------------------------------------------------------


def _indicator_code(view) -> str | None:
    code = view.get("indicator")
    if isinstance(code, str) and re.fullmatch(r"[A-Z]{2}\.[A-Z0-9.]+", code):
        return code
    label = str(view.get("indicator_name") or "").strip().lower()
    return WDI_LABEL_TO_CODE.get(label)


def tool_indicator_lookup(view, corpus) -> ToolResult:
    name = "indicator_lookup"
    if corpus is None:
        return _skip(name, "no_corpus_index")
    code = _indicator_code(view)
    year = _num(view.get("year"))
    countries: list[str] = []
    for key in ("country", "entities", "comparison_countries", "ranking_countries"):
        item = view.get(key)
        if isinstance(item, str):
            countries.append(item)
        elif isinstance(item, list):
            countries.extend(str(entry) for entry in item if isinstance(entry, str))
    countries = list(dict.fromkeys(countries))
    if not code or year is None or not countries:
        return _skip(name)
    series: dict[str, Any] = {}
    lines: list[str] = []
    for country in countries[:5]:
        observations = corpus.indicator_series(code, country)
        if not observations:
            continue
        exact = next((row for row in observations if row["year"] == int(year)), None)
        nearest = sorted(observations, key=lambda row: (abs(row["year"] - int(year)), row["year"]))[:4]
        estimate = corpus.interpolate(observations, int(year))
        series[country] = {
            "exact": exact,
            "nearest_years": nearest,
            "estimate_for_requested_year": estimate,
        }
        if exact is not None:
            lines.append(
                f"Corpus observation: {country} {code} in {int(year)} = "
                f"{exact['value']} {exact['unit']}."
            )
        else:
            rendered = "; ".join(
                f"{row['year']}: {row['value']}" for row in sorted(nearest, key=lambda r: r["year"])
            )
            note = (
                f"estimated {estimate['value']:.6g} for {int(year)} by "
                f"{estimate['method']}" if estimate else "no estimate possible"
            )
            lines.append(
                f"The corpus does not contain {country} {code} for {int(year)} "
                f"(the frozen corpus withholds every benchmark observation). "
                f"Neighbouring observations -- {rendered} -- give {note}."
            )
    if not series:
        return _skip(name, "no_matching_corpus_observations")
    value: dict[str, Any] = {"indicator": code, "year": int(year), "series": series}
    if len(series) >= 2:
        estimates = {
            country: (data["exact"] or data["estimate_for_requested_year"] or {}).get("value")
            for country, data in series.items()
        }
        usable = {key: item for key, item in estimates.items() if isinstance(item, (int, float))}
        if len(usable) >= 2:
            ordered = sorted(usable.items(), key=lambda pair: -pair[1])
            value["values_for_requested_year"] = usable
            value["descending_ranking"] = [country for country, _ in ordered]
            value["larger_entity"] = ordered[0][0]
            if len(ordered) == 2:
                value["absolute_difference"] = round(abs(ordered[0][1] - ordered[1][1]), 6)
                value["larger_to_smaller_ratio"] = round(
                    max(abs(ordered[0][1]), abs(ordered[1][1]))
                    / max(min(abs(ordered[0][1]), abs(ordered[1][1])), 1e-12), 6
                )
            lines.append(
                "Descending order implied by these corpus-derived values: "
                + " > ".join(f"{country} ({usable[country]:.6g})" for country, _ in ordered)
                + "."
            )
    area = _num(view.get("hypothetical_area_km2"))
    if area is not None and len(series) == 1:
        only = next(iter(series.values()))
        base = (only["exact"] or only["estimate_for_requested_year"] or {}).get("value")
        if isinstance(base, (int, float)):
            value["scaled_to_area"] = round(base * area, 3)
            lines.append(
                f"Scaling that density by the hypothetical land area of {area:g} km^2 "
                f"gives {value['scaled_to_area']}."
            )
    return ToolResult(
        name=name, status="ok", authority="strong",
        title="World Bank indicator evidence from the frozen corpus",
        text=" ".join(lines), value=value,
        detail={"countries": countries[:5], "indicator": code},
    )


# ---------------------------------------------------------------------------
# 9. Gazetteer lookup / bearing
# ---------------------------------------------------------------------------


def tool_gazetteer(view, corpus) -> ToolResult:
    name = "gazetteer"
    if corpus is None:
        return _skip(name, "no_corpus_index")
    labels = view.get("labels")
    pair: list[str] = []
    if isinstance(labels, dict):
        for key in ("A", "B"):
            if isinstance(labels.get(key), str):
                pair.append(labels[key])
    names = pair or view.entity_names()[:2]
    if not names:
        return _skip(name)
    resolved: dict[str, Any] = {}
    lines: list[str] = []
    for label in names[:2]:
        hit = corpus.locate(label)
        if hit:
            resolved[label] = hit
            lines.append(
                f"Gazetteer match for \"{label}\": {hit['title']} at "
                f"latitude {hit['lat']:.4f}, longitude {hit['lon']:.4f} "
                f"(source {hit['source']})."
            )
    value: dict[str, Any] = {"resolved": resolved}
    if len(resolved) == 2 and pair:
        (name_a, a), (name_b, b) = list(resolved.items())[:2]
        direction = cardinal_from_bearing(a["lon"] - b["lon"], a["lat"] - b["lat"])
        separation = _segment_length((a["lon"], a["lat"]), (b["lon"], b["lat"]))
        value["approximate_direction_a_to_b"] = direction
        value["separation_m"] = round(separation, 1)
        lines.append(
            f"Using those gazetteer centroids, A lies approximately to the {direction} "
            f"of B ({separation / 1000.0:.1f} km apart). This is an approximation from "
            f"point centroids, not from the plotted polygons: trust the map image when "
            f"the two disagree."
        )
    if not resolved:
        return _skip(name, "no_gazetteer_match")
    return ToolResult(
        name=name, status="ok", authority="advisory",
        title="Gazetteer evidence from the frozen corpus",
        text=" ".join(lines), value=value,
    )


# ---------------------------------------------------------------------------
# 10. Nearby gazetteer context for a coordinate
# ---------------------------------------------------------------------------


def tool_nearby_places(view, corpus) -> ToolResult:
    name = "nearby_places"
    if corpus is None:
        return _skip(name, "no_corpus_index")
    context = view.get("context") if isinstance(view.get("context"), dict) else view.payload
    lat = _num(context.get("latitude", context.get("lat")))
    lon = _num(context.get("longitude", context.get("lon")))
    coordinate = view.get("coordinate")
    if (lat is None or lon is None) and isinstance(coordinate, (list, dict)):
        point = _point(coordinate)
        if point is not None:
            lon, lat = point
    if lat is None or lon is None:
        return _skip(name)
    neighbours = corpus.near(lat, lon, limit=4)
    if not neighbours:
        return _skip(name, "no_neighbours")
    lines = [
        f"Nearest frozen-corpus entries to latitude {lat:.4f}, longitude {lon:.4f} "
        "(the corpus deliberately excludes anything within 2 km of a benchmark "
        "coordinate, so these are context, not the answer):"
    ]
    for item in neighbours:
        lines.append(
            f"- {item['title']} ({item['source']}, {item['distance_km']:.1f} km): "
            f"{item['text'][:220]}"
        )
    return ToolResult(
        name=name, status="ok", authority="advisory",
        title="Nearby gazetteer context",
        text="\n".join(lines), value={"neighbours": neighbours},
    )


# ---------------------------------------------------------------------------
# 11. Closed-set constraint
# ---------------------------------------------------------------------------


def tool_closed_set(view) -> ToolResult:
    name = "closed_set"
    if not view.choices:
        return _skip(name)
    rendered = "; ".join(str(choice) for choice in view.choices[:40])
    return ToolResult(
        name=name, status="ok", authority="exact",
        title="Closed answer set",
        text=(
            "The answer must be exactly one of the following strings, copied "
            f"character for character: {rendered}."
        ),
        value={"choices": list(view.choices)},
    )


# ---------------------------------------------------------------------------
# 12. Artifact structure contract
# ---------------------------------------------------------------------------


def tool_artifact_contract(view) -> ToolResult:
    name = "artifact_contract"
    leaf = view.leaf
    question = (view.question or "").lower()
    mask_leaf = leaf in {"dense_land_cover_labeling", "change_localization"}
    if mask_leaf and ("mask" in question or "segmentation" in question):
        return ToolResult(
            name=name, status="ok", authority="exact",
            title="Mask encoding contract",
            text=(
                "A 64x64 row-major run-length mask has exactly 4096 cells. Emit runs as "
                "[value, count] pairs with strictly positive integer counts whose sum is "
                "exactly 4096, reading left to right then top to bottom. Prefer a small "
                "number of large runs that reflect the real spatial layout; never pad with "
                "an out-of-ontology value."
            ),
            value={"cells": 4096, "size": [64, 64], "encoding": "rle-row-major"},
        )
    if leaf == "spatial_graph_construction" and "graph" in question:
        return ToolResult(
            name=name, status="ok", authority="exact",
            title="Graph encoding contract",
            text=(
                "Emit nodes with unique integer ids and float x/y in the same coordinate "
                "space as the task input, and emit edges only between listed node ids. "
                "Every edge needs a positive length consistent with its endpoints."
            ),
            value={"encoding": "inline-node-edge"},
        )
    return _skip(name)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

CORPUS_TOOLS: dict[str, Callable[[Any, Any], ToolResult]] = {
    "indicator_lookup": tool_indicator_lookup,
    "gazetteer": tool_gazetteer,
    "nearby_places": tool_nearby_places,
}

PLAIN_TOOLS: dict[str, Callable[[Any], ToolResult]] = {
    "crs_transform": tool_crs_transform,
    "geodesic_distance": tool_geodesic_distance,
    "route_metrics": tool_route_metrics,
    "service_budget": tool_service_budget,
    "unit_symbol": tool_unit_symbol,
    "class_ontology": tool_class_ontology,
    "text_span": tool_text_span,
    "closed_set": tool_closed_set,
    "artifact_contract": tool_artifact_contract,
}

# Which corpus tools are worth running per leaf. Keeps the toolbelt cheap and
# stops an irrelevant gazetteer hit from polluting a perception task.
CORPUS_TOOL_ROUTES: dict[str, tuple[str, ...]] = {
    "population_density_estimation": ("indicator_lookup",),
    "cross_entity_comparison": ("indicator_lookup",),
    "topological_directional_reasoning": ("gazetteer",),
    "geo_entity_typing": ("gazetteer", "nearby_places"),
    "geologic_geomorphic_interpretation": ("nearby_places",),
    "visual_geolocation": ("gazetteer",),
    "geographic_fact_reasoning": ("gazetteer",),
    "toponym_recognition": (),
}


def run_toolbelt(view, corpus=None) -> list[ToolResult]:
    """Run every applicable deterministic tool for one task view."""
    results: list[ToolResult] = []
    for tool in PLAIN_TOOLS.values():
        try:
            result = tool(view)
        except Exception as error:  # a tool must never break a run
            result = ToolResult(name=getattr(tool, "__name__", "tool"), status="error",
                                detail={"error": repr(error)})
        if result.status != "skipped":
            results.append(result)
    for key in CORPUS_TOOL_ROUTES.get(view.leaf, ()):
        try:
            result = CORPUS_TOOLS[key](view, corpus)
        except Exception as error:
            result = ToolResult(name=key, status="error", detail={"error": repr(error)})
        if result.status != "skipped":
            results.append(result)
    order = {"exact": 0, "strong": 1, "advisory": 2}
    results.sort(key=lambda item: (order.get(item.authority, 3), item.name))
    return results


# ---------------------------------------------------------------------------
# Answer proposals
# ---------------------------------------------------------------------------
#
# A proposal maps one *explicitly recognised* request to a value the toolbelt
# derived from the task input. Nothing is proposed unless the question wording
# and the required input fields both match, so an unrecognised task simply
# falls through to ordinary model reasoning.
#
# ``confidence="exact"``  closed-form arithmetic or a literal slice of the
#                         supplied input; the repair stage may substitute it.
# ``confidence="strong"`` corpus-derived or convention-derived; evidence only.


@dataclass
class AnswerProposal:
    value: Any
    source: str
    confidence: str
    rationale: str

    def trace(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "value": _jsonable(self.value),
        }


def _by_name(results: list[ToolResult]) -> dict[str, ToolResult]:
    return {result.name: result for result in results if result.ok}


def _asks(question: str, *fragments: str) -> bool:
    lowered = (question or "").lower()
    return any(fragment.lower() in lowered for fragment in fragments)


def propose_answer(view, results: list[ToolResult]) -> AnswerProposal | None:
    """Recognise a request the toolbelt can answer outright."""
    tools = _by_name(results)
    question = view.question or ""
    yes_no = _asks(question, "answer yes or no")

    # 1. Verified yes/no verification questions.
    if yes_no:
        for key in ("crs_transform", "geodesic_distance", "route_metrics"):
            verdict = (tools.get(key).value or {}).get("candidate_verdict") if key in tools else None
            if verdict in {"yes", "no"}:
                return AnswerProposal(
                    verdict, key, "exact",
                    "the candidate was compared against a value computed from the task input",
                )
        # A verification question never wants the underlying quantity as its answer.
        return None

    # 2. Coordinate transformation family.
    crs = tools.get("crs_transform")
    if crs is not None:
        value = crs.value or {}
        pair = value.get("transformed_coordinate")
        if _asks(question, "what unit is used for coordinates in the target crs"):
            return AnswerProposal(value.get("target_unit"), "crs_transform", "exact",
                                  "the unit is read from the target CRS definition")
        if _asks(question, "geographic or projected"):
            return AnswerProposal(value.get("target_crs_kind"), "crs_transform", "exact",
                                  "the CRS type is read from the target CRS definition")
        if pair and _asks(question, "construct a complete machine-readable coordinate-transformation record"):
            return AnswerProposal(
                {
                    "source_crs": view.get("source_crs"),
                    "target_crs": view.get("target_crs"),
                    "source_coordinate": view.get("coordinate"),
                    "transformed_coordinate": pair,
                    "transformation_mode": view.get("transformation_mode"),
                },
                "crs_transform", "exact",
                "every field of the requested record is present in the task input or computed by PROJ",
            )
        if pair and _asks(question, "transform the coordinate", "transformed coordinate"):
            return AnswerProposal(pair, "crs_transform", "exact",
                                  "PROJ transformed the supplied source coordinate")

    # 3. Geodesic distance family.
    distance = tools.get("geodesic_distance")
    if distance is not None:
        value = distance.value or {}
        if _asks(question, "canonical metre and kilometre representations"):
            return AnswerProposal(
                {
                    "value": value.get("value"), "unit": value.get("unit"),
                    "distance_m": value.get("distance_m"), "distance_km": value.get("distance_km"),
                    "method": value.get("method"),
                },
                "geodesic_distance", "exact",
                "all four quantities follow from the WGS84 inverse geodesic solution",
            )
        if _asks(question, "geodesic distance", "distance from a", "distance between"):
            return AnswerProposal(value.get("value"), "geodesic_distance", "exact",
                                  "pyproj solved the inverse geodesic for the supplied points")

    unit_tool = tools.get("unit_symbol")
    if unit_tool is not None and _asks(question, "standard unit symbol"):
        return AnswerProposal((unit_tool.value or {}).get("unit"), "unit_symbol", "exact",
                              "the SI/standard symbol for the declared unit id")

    # 4. Route family.
    route = tools.get("route_metrics")
    if route is not None:
        value = route.value or {}
        if "route_length_m" in value and _asks(question, "return route length, direct distance"):
            return AnswerProposal(
                {
                    "route_length_m": value.get("route_length_m"),
                    "direct_distance_m": value.get("direct_distance_m"),
                    "detour_ratio": value.get("detour_ratio"),
                },
                "route_metrics", "exact",
                "the supplied reference route was measured with the benchmark segment rule",
            )
        if "route_length_m" in value and _asks(question, "apply the path-length calculation"):
            return AnswerProposal(value.get("route_length_m"), "route_metrics", "exact",
                                  "the supplied reference route was measured segment by segment")

    # 5. Service-area budget family.
    budget = tools.get("service_budget")
    if budget is not None:
        value = budget.value or {}
        if _asks(question, "maximum network travel distance"):
            return AnswerProposal(value.get("network_distance_budget_m"), "service_budget", "exact",
                                  "speed x time from the task input")
        if _asks(question, "what time budget"):
            return AnswerProposal(value.get("budget_minutes"), "service_budget", "exact",
                                  "the time budget is stated in the task input")

    # 6. Literal slices of the supplied input.
    span = tools.get("text_span")
    if span is not None and (span.value or {}).get("span_text") is not None:
        if _asks(question, "what exact place-name text occurs at character offsets", "characters at offsets"):
            return AnswerProposal((span.value or {})["span_text"], "text_span", "exact",
                                  "the substring was sliced from the supplied document")

    ontology = tools.get("class_ontology")
    if ontology is not None and (ontology.value or {}).get("queried_class_name"):
        if _asks(question, "class name corresponds to class id", "what land-cover class name"):
            return AnswerProposal((ontology.value or {})["queried_class_name"], "class_ontology", "exact",
                                  "the name was read from the ontology supplied in the task input")

    # 7. Corpus-derived indicator answers (evidence-grade, never substituted).
    indicator = tools.get("indicator_lookup")
    if indicator is not None:
        value = indicator.value or {}
        if _asks(question, "descending population-density ranking", "ranking for these countries"):
            ranking = value.get("descending_ranking")
            if ranking:
                return AnswerProposal(ranking, "indicator_lookup", "strong",
                                      "ordering implied by corpus observations for the requested year")
        if _asks(question, "which had the higher", "which country recorded a larger"):
            if value.get("larger_entity"):
                return AnswerProposal(value["larger_entity"], "indicator_lookup", "strong",
                                      "the larger corpus-derived value for the requested year")
        if _asks(question, "which had the lower", "which country had the smaller"):
            ranking = value.get("descending_ranking") or []
            if len(ranking) == 2:
                return AnswerProposal(ranking[-1], "indicator_lookup", "strong",
                                      "the smaller corpus-derived value for the requested year")
        if _asks(question, "absolute difference in") and value.get("absolute_difference") is not None:
            return AnswerProposal(value["absolute_difference"], "indicator_lookup", "strong",
                                  "difference of the corpus-derived values")
        if _asks(question, "hypothetical land area") and value.get("scaled_to_area") is not None:
            return AnswerProposal(value["scaled_to_area"], "indicator_lookup", "strong",
                                  "corpus-derived density scaled by the stated area")
    return None
