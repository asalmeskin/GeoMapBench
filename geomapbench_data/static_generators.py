from __future__ import annotations

import csv
import math
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import N_EXAMPLES, SEEDS, balanced_sample, base_record, finalize_task, stable_sample
from .download import download, extract_zip


NE_COUNTRIES_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
NE_PLACES_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_populated_places.zip"


def natural_earth(cache: Path):
    import geopandas as gpd

    root = cache / "natural_earth"
    countries_dir = extract_zip(download(NE_COUNTRIES_URL, root / "countries.zip"), root / "countries")
    places_dir = extract_zip(download(NE_PLACES_URL, root / "places.zip"), root / "places")
    country_shp = next(countries_dir.glob("*.shp"))
    places_shp = next(places_dir.glob("*.shp"))
    countries = gpd.read_file(country_shp).to_crs(4326)
    places = gpd.read_file(places_shp).to_crs(4326)
    return countries, places


def _place_name(row) -> str:
    for key in ("NAME", "NAMEPAR", "NAMEASCII", "name"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown place"


def _country_name(row) -> str:
    for key in ("ADMIN", "NAME_EN", "SOVEREIGNT", "NAME", "name"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown country"


def _utm_epsg(longitude: float, latitude: float) -> int:
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def _coordinate_case(longitude: float, latitude: float, mode: str) -> dict[str, Any]:
    """Build one reversible coordinate-transformation case."""
    from pyproj import Transformer

    utm = f"EPSG:{_utm_epsg(longitude, latitude)}"
    definitions = {
        "geographic_to_utm": ("EPSG:4326", utm, "longitude", "latitude", "easting", "northing", "degrees", "metres"),
        "utm_to_geographic": (utm, "EPSG:4326", "easting", "northing", "longitude", "latitude", "metres", "degrees"),
        "geographic_to_web_mercator": ("EPSG:4326", "EPSG:3857", "longitude", "latitude", "x", "y", "degrees", "metres"),
        "web_mercator_to_geographic": ("EPSG:3857", "EPSG:4326", "x", "y", "longitude", "latitude", "metres", "degrees"),
        "utm_to_web_mercator": (utm, "EPSG:3857", "easting", "northing", "x", "y", "metres", "metres"),
        "web_mercator_to_utm": ("EPSG:3857", utm, "x", "y", "easting", "northing", "metres", "metres"),
    }
    if mode not in definitions:
        raise ValueError(f"Unknown coordinate-transformation mode: {mode}")

    source_crs, target_crs, source_x_name, source_y_name, target_x_name, target_y_name, source_unit, target_unit = definitions[mode]
    wgs84_to_source = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
    source_x, source_y = wgs84_to_source.transform(longitude, latitude)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    target_x, target_y = transformer.transform(source_x, source_y)
    inverse = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    roundtrip_x, roundtrip_y = inverse.transform(target_x, target_y)

    projected = target_unit == "metres"
    precision = 3 if projected else 7
    tolerance = 1.0 if projected else 1e-5
    return {
        "mode": mode,
        "source_crs": source_crs,
        "target_crs": target_crs,
        "source": {
            source_x_name: round(float(source_x), 7 if source_unit == "degrees" else 3),
            source_y_name: round(float(source_y), 7 if source_unit == "degrees" else 3),
            "axis_order": [source_x_name, source_y_name],
            "unit": source_unit,
        },
        "target": {
            target_x_name: round(float(target_x), precision),
            target_y_name: round(float(target_y), precision),
            "axis_order": [target_x_name, target_y_name],
            "unit": target_unit,
        },
        "roundtrip_error": max(abs(roundtrip_x - source_x), abs(roundtrip_y - source_y)),
        "absolute_tolerance": tolerance,
    }


def generate_coordinate_transform(cache: Path, output: Path) -> Path:
    leaf = "coordinate_transformation"
    _, places = natural_earth(cache)
    candidates = [
        row
        for _, row in places.iterrows()
        if row.geometry and not row.geometry.is_empty and abs(row.geometry.y) < 80
    ]
    selected = stable_sample(
        candidates,
        N_EXAMPLES,
        SEEDS[leaf],
        key=lambda row: f"{_place_name(row)}:{row.geometry.x:.6f}:{row.geometry.y:.6f}",
    )
    modes = (
        "geographic_to_utm",
        "utm_to_geographic",
        "geographic_to_web_mercator",
        "web_mercator_to_geographic",
        "utm_to_web_mercator",
        "web_mercator_to_utm",
    )
    records: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        case = _coordinate_case(lon, lat, modes[i % len(modes)])
        source = case["source"]
        source_values = [source[name] for name in source["axis_order"]]
        target_axes = case["target"]["axis_order"]
        question = (
            f"Transform the coordinate ({source_values[0]}, {source_values[1]}) from "
            f"{case['source_crs']} to {case['target_crs']}. Interpret the input in "
            f"{source['axis_order'][0]}-then-{source['axis_order'][1]} order and return "
            f"{target_axes[0]} and {target_axes[1]} in {case['target']['unit']}."
        )
        record = base_record(
            leaf,
            i,
            "Natural Earth populated places + PROJ/EPSG",
            "https://www.naturalearthdata.com/",
            "Natural Earth public domain; PROJ data terms apply",
            f"{_place_name(row)}:{case['mode']}",
        )
        record.update(
            {
                "input": {
                    "question": question,
                    "transformation_mode": case["mode"],
                    "source_crs": case["source_crs"],
                    "target_crs": case["target_crs"],
                    "coordinate": source,
                    "reference_place": _place_name(row),
                },
                "target": {
                    **case["target"],
                    "crs": case["target_crs"],
                },
                "evaluation": {
                    "type": "numeric_coordinate_pair",
                    "absolute_tolerance": case["absolute_tolerance"],
                    "unit": case["target"]["unit"],
                },
            }
        )
        if case["roundtrip_error"] > 1e-5:
            raise ValueError(f"Unstable CRS round trip for {_place_name(row)}: {case}")
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {"transformation_modes": list(modes), "crs_pair_count": len({(r['input']['source_crs'], r['input']['target_crs']) for r in records})},
    )


def _render_world_pair(countries, first, second, destination: Path, first_label: str, second_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    countries.boundary.plot(ax=ax, color="#a8a8a8", linewidth=0.35)
    xs = [first.x, second.x]
    ys = [first.y, second.y]
    ax.plot(xs, ys, color="#6a3d9a", linewidth=1.2, linestyle="--")
    ax.scatter(xs, ys, c=["#e31a1c", "#1f78b4"], s=35, zorder=3)
    ax.annotate("A", (xs[0], ys[0]), xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.annotate("B", (xs[1], ys[1]), xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_axis_off()
    ax.set_title(f"A: {first_label}    B: {second_label}", fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


DISTANCE_UNITS: dict[str, dict[str, Any]] = {
    "metres": {
        "question_label": "metres",
        "target_unit": "m",
        "metres_per_unit": 1.0,
        "decimals": 1,
    },
    "kilometres": {
        "question_label": "kilometres",
        "target_unit": "km",
        "metres_per_unit": 1000.0,
        "decimals": 3,
    },
    "miles": {
        "question_label": "statute miles",
        "target_unit": "mi",
        "metres_per_unit": 1609.344,
        "decimals": 3,
    },
    "nautical_miles": {
        "question_label": "nautical miles",
        "target_unit": "nmi",
        "metres_per_unit": 1852.0,
        "decimals": 3,
    },
}


def _convert_distance_from_metres(metres: float, unit_id: str) -> float:
    """Convert a WGS84 geodesic distance to one declared output unit."""
    if unit_id not in DISTANCE_UNITS:
        raise ValueError(f"Unsupported distance unit: {unit_id}")
    spec = DISTANCE_UNITS[unit_id]
    return round(float(metres) / float(spec["metres_per_unit"]), int(spec["decimals"]))


def generate_metric_distance(cache: Path, output: Path) -> Path:
    from pyproj import Geod

    leaf = "metric_distance_computation"
    countries, places = natural_earth(cache)
    if "POP_MAX" in places:
        places = places.sort_values("POP_MAX", ascending=False).head(1000)
    rows = [row for _, row in places.iterrows() if row.geometry and not row.geometry.is_empty]
    rng = random.Random(SEEDS[leaf])
    pairs: list[tuple[Any, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < N_EXAMPLES:
        a, b = rng.sample(rows, 2)
        key = tuple(sorted((_place_name(a), _place_name(b))))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))

    # Exactly 25 examples per unit. The pairs are already deterministically
    # shuffled, so this does not create geographic blocks by unit.
    unit_ids = tuple(DISTANCE_UNITS)
    geod = Geod(ellps="WGS84")
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(pairs):
        _, _, metres = geod.inv(a.geometry.x, a.geometry.y, b.geometry.x, b.geometry.y)
        metres = float(abs(metres))
        unit_id = unit_ids[i % len(unit_ids)]
        unit_spec = DISTANCE_UNITS[unit_id]
        value = _convert_distance_from_metres(metres, unit_id)
        name_a, name_b = _place_name(a), _place_name(b)
        image_path = task_dir / "assets" / f"{i:03d}.png"
        _render_world_pair(countries, a.geometry, b.geometry, image_path, name_a, name_b)
        record = base_record(
            leaf,
            i,
            "Natural Earth populated places",
            "https://www.naturalearthdata.com/",
            "Public domain",
            f"{name_a}|{name_b}|{unit_id}",
        )
        record.update(
            {
                "input": {
                    "images": [image_path.relative_to(task_dir).as_posix()],
                    "question": (
                        f"What is the WGS 84 geodesic distance from A ({name_a}) "
                        f"to B ({name_b}) in {unit_spec['question_label']}?"
                    ),
                    "points": [
                        {"name": name_a, "longitude": a.geometry.x, "latitude": a.geometry.y},
                        {"name": name_b, "longitude": b.geometry.x, "latitude": b.geometry.y},
                    ],
                    "requested_unit": unit_id,
                },
                "target": {
                    "value": value,
                    "unit": unit_spec["target_unit"],
                    "unit_id": unit_id,
                    # Canonical values make cross-unit auditing straightforward
                    # while the requested answer remains in target.value.
                    "distance_m": round(metres, 3),
                    "distance_km": round(metres / 1000.0, 6),
                    "method": "WGS84 inverse geodesic",
                },
                "evaluation": {
                    "type": "numeric",
                    "relative_tolerance": 0.005,
                    "unit": unit_spec["target_unit"],
                },
            }
        )
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {
            "distance_units": list(unit_ids),
            "examples_per_unit": N_EXAMPLES // len(unit_ids),
        },
    )


def _cardinal(dx: float, dy: float) -> str:
    angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    labels = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return labels[int((angle + 22.5) // 45) % 8]


def _plot_polygon(ax, geometry, facecolor: str, edgecolor: str, alpha: float = 0.35, linewidth: float = 1.8) -> None:
    import geopandas as gpd

    gpd.GeoSeries([geometry], crs="EPSG:4326").plot(
        ax=ax,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )


def _set_geometry_bounds(ax, geometries: list[Any], padding_fraction: float = 0.15) -> None:
    bounds = [geometry.bounds for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not bounds:
        return
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    width = max(max_x - min_x, 0.5)
    height = max(max_y - min_y, 0.5)
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction
    ax.set_xlim(max(-180, min_x - pad_x), min(180, max_x + pad_x))
    ax.set_ylim(max(-90, min_y - pad_y), min(90, max_y + pad_y))


def _place_area_polygon(point, country_geometry):
    """Create a visible polygon around a place that remains inside its country."""
    import geopandas as gpd

    local_crs = gpd.GeoSeries([point], crs="EPSG:4326").estimate_utm_crs()
    if local_crs is None:
        return None
    local = gpd.GeoSeries([point, country_geometry], crs="EPSG:4326").to_crs(local_crs)
    local_point, local_country = local.iloc[0], local.iloc[1]
    if local_country.is_empty or not local_country.is_valid:
        local_country = local_country.buffer(0)
    clearance = float(local_point.distance(local_country.boundary))
    if not math.isfinite(clearance) or clearance <= 100:
        return None

    # Use a 12-sided polygon rather than an ambiguous point marker. The radius
    # adapts to border clearance and is capped to keep the highlighted area local.
    radius = min(20_000.0, max(750.0, clearance * 0.35))
    candidate = local_point.buffer(radius, quad_segs=3)
    for _ in range(6):
        if candidate.within(local_country):
            return gpd.GeoSeries([candidate], crs=local_crs).to_crs(4326).iloc[0]
        radius *= 0.5
        candidate = local_point.buffer(radius, quad_segs=3)
    return None


def _render_within_polygons(countries, area_a, country_b, destination: Path, place_name: str, country_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), dpi=150)

    # Country-level view.
    ax = axes[0]
    countries.boundary.plot(ax=ax, color="#c7c7c7", linewidth=0.3)
    _plot_polygon(ax, country_b, "#80b1d3", "#1f78b4", alpha=0.28, linewidth=1.6)
    _plot_polygon(ax, area_a, "#fb8072", "#d7301f", alpha=0.8, linewidth=2.0)
    _set_geometry_bounds(ax, [country_b], 0.12)
    pa = area_a.representative_point()
    pb = country_b.representative_point()
    ax.annotate("A", (pa.x, pa.y), ha="center", va="center", fontsize=11, weight="bold", color="#7f0000",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d7301f", "pad": 1.5})
    ax.annotate("B", (pb.x, pb.y), ha="center", va="center", fontsize=11, weight="bold", color="#084594",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#1f78b4", "pad": 1.5})
    ax.set_title(f"Country context: B = {country_name}", fontsize=9)
    ax.set_axis_off()

    # Local view makes polygon A clearly distinguishable from a point.
    ax = axes[1]
    _plot_polygon(ax, country_b, "#deebf7", "#1f78b4", alpha=0.45, linewidth=1.3)
    _plot_polygon(ax, area_a, "#fb8072", "#d7301f", alpha=0.9, linewidth=2.4)
    min_x, min_y, max_x, max_y = area_a.bounds
    width = max(max_x - min_x, 0.03)
    height = max(max_y - min_y, 0.03)
    ax.set_xlim(min_x - width * 4, max_x + width * 4)
    ax.set_ylim(min_y - height * 4, max_y + height * 4)
    ax.annotate("A", (pa.x, pa.y), ha="center", va="center", fontsize=12, weight="bold", color="#7f0000",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d7301f", "pad": 1.5})
    ax.text(0.02, 0.98, "Blue area = polygon B", transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#1f78b4", "pad": 2})
    ax.set_title(f"Local view: A = area around {place_name}", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.suptitle("Determine the relation between polygon A and polygon B", fontsize=11, weight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_direction_polygons(countries, geometry_a, geometry_b, destination: Path, name_a: str, name_b: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    countries.boundary.plot(ax=ax, color="#d0d0d0", linewidth=0.3)
    _plot_polygon(ax, geometry_a, "#fb8072", "#d7301f", alpha=0.75, linewidth=2.0)
    _plot_polygon(ax, geometry_b, "#80b1d3", "#1f78b4", alpha=0.65, linewidth=2.0)
    _set_geometry_bounds(ax, [geometry_a, geometry_b], 0.18)

    pa = geometry_a.representative_point()
    pb = geometry_b.representative_point()
    ax.annotate("A", (pa.x, pa.y), ha="center", va="center", fontsize=12, weight="bold", color="#7f0000",
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d7301f", "pad": 1.8})
    ax.annotate("B", (pb.x, pb.y), ha="center", va="center", fontsize=12, weight="bold", color="#084594",
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#1f78b4", "pad": 1.8})
    ax.set_title(f"A: {name_a}     B: {name_b}", fontsize=10, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_topology_direction(cache: Path, output: Path) -> Path:
    import geopandas as gpd

    leaf = "topological_directional_reasoning"
    countries, places = natural_earth(cache)
    countries = countries[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    places = places[places.geometry.notna() & ~places.geometry.is_empty].copy()

    # Exclude very small regions that remain difficult to distinguish even when
    # plotted as polygons. Area is measured in an equal-area CRS.
    country_areas = countries.to_crs(6933).geometry.area
    countries = countries[country_areas >= 10_000_000_000].copy()  # 10,000 km²
    countries = countries.sort_values("ADMIN").drop_duplicates("ADMIN")

    joined = gpd.sjoin(places, countries[["geometry", "ADMIN"]], predicate="within", how="inner")
    within_rows = [row for _, row in joined.iterrows()]
    candidate_count = min(len(within_rows), 500)
    ordered_candidates = stable_sample(
        within_rows,
        candidate_count,
        SEEDS[leaf],
        key=lambda row: f"{_place_name(row)}:{row.get('ADMIN', '')}",
    )
    within_candidates: list[tuple[Any, Any, Any]] = []
    country_lookup = {str(row["ADMIN"]): row for _, row in countries.iterrows()}
    for row in ordered_candidates:
        country_name = str(row["ADMIN"])
        country = country_lookup.get(country_name)
        if country is None:
            continue
        area_polygon = _place_area_polygon(row.geometry, country.geometry)
        if area_polygon is not None and not area_polygon.is_empty:
            within_candidates.append((row, country, area_polygon))
        if len(within_candidates) >= 120:
            break
    within_selected = stable_sample(
        within_candidates,
        50,
        SEEDS[leaf] + 7,
        key=lambda item: f"{_place_name(item[0])}:{item[1]['ADMIN']}",
    )

    country_rows = [row for _, row in countries.iterrows()]
    rng = random.Random(SEEDS[leaf] + 1)
    pair_rows: list[tuple[Any, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(pair_rows) < 50:
        a, b = rng.sample(country_rows, 2)
        key = tuple(sorted((_country_name(a), _country_name(b))))
        if key in seen:
            continue
        if a.geometry.intersects(b.geometry) and not a.geometry.touches(b.geometry):
            continue
        seen.add(key)
        pair_rows.append((a, b))

    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    mixed: list[tuple[str, Any]] = [("within", item) for item in within_selected] + [("direction", item) for item in pair_rows]
    rng.shuffle(mixed)
    for i, (kind, item) in enumerate(mixed):
        image_path = task_dir / "assets" / f"{i:03d}.png"
        if kind == "within":
            row, country, area_a = item
            place_name, country_name = _place_name(row), str(country["ADMIN"])
            _render_within_polygons(countries, area_a, country.geometry, image_path, place_name, country_name)
            question = (
                f"Is polygon A (the highlighted area around {place_name}) completely within "
                f"polygon B ({country_name})?"
            )
            target = {
                "relation": "within",
                "answer": "yes",
                "region_a_name": f"area around {place_name}",
                "region_b_name": country_name,
                "geometry_representation": "polygon",
            }
            group_id = f"within:{country_name}:{place_name}"
        else:
            a, b = item
            pa, pb = a.geometry.representative_point(), b.geometry.representative_point()
            relation = "touches" if a.geometry.touches(b.geometry) else _cardinal(pa.x - pb.x, pa.y - pb.y)
            name_a, name_b = _country_name(a), _country_name(b)
            _render_direction_polygons(countries, a.geometry, b.geometry, image_path, name_a, name_b)
            if relation == "touches":
                question = f"What topological relation holds between polygon A ({name_a}) and polygon B ({name_b})?"
            else:
                question = f"What is the approximate cardinal direction of polygon A ({name_a}) relative to polygon B ({name_b})?"
            target = {
                "relation": relation,
                "answer": relation,
                "region_a_name": name_a,
                "region_b_name": name_b,
                "geometry_representation": "polygon",
            }
            group_id = f"direction:{name_a}|{name_b}"
        record = base_record(
            leaf,
            i,
            "Natural Earth",
            "https://www.naturalearthdata.com/",
            "Public domain",
            group_id,
        )
        record.update(
            {
                "input": {
                    "images": [image_path.relative_to(task_dir).as_posix()],
                    "question": question,
                    "visual_geometry": "polygon",
                    "labels": {"A": target["region_a_name"], "B": target["region_b_name"]},
                },
                "target": target,
                "evaluation": {"type": "relation_classification", "metric": "accuracy"},
            }
        )
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {
            "visual_geometry": "polygon",
            "within_examples": 50,
            "directional_examples": 50,
        },
    )


GEONAMES_COUNTRIES = ["CH", "IS", "NZ", "KE", "PE", "NP", "MA", "JP", "ZA", "ID"]
GEONAMES_CLASS_NAMES = {
    "A": "administrative feature",
    "H": "hydrographic feature",
    "L": "area feature",
    "P": "populated place",
    "R": "road or railroad feature",
    "S": "spot or facility feature",
    "T": "terrain feature",
    "U": "undersea feature",
    "V": "vegetation feature",
}


def _parse_geonames(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for fields in csv.reader(f, delimiter="\t"):
            if len(fields) < 15 or not fields[1].strip() or fields[6] not in GEONAMES_CLASS_NAMES:
                continue
            try:
                lat, lon = float(fields[4]), float(fields[5])
                population = int(fields[14] or 0)
            except ValueError:
                continue
            rows.append(
                {
                    "id": fields[0],
                    "name": fields[1],
                    "latitude": lat,
                    "longitude": lon,
                    "feature_class": fields[6],
                    "feature_code": fields[7],
                    "country_code": fields[8],
                    "population": population,
                }
            )
    return rows


def generate_geo_entity_typing(cache: Path, output: Path) -> Path:
    leaf = "geo_entity_typing"
    root = cache / "geonames"
    all_rows: list[dict[str, Any]] = []
    for code in GEONAMES_COUNTRIES:
        archive = download(f"https://download.geonames.org/export/dump/{code}.zip", root / f"{code}.zip")
        target = root / code
        marker = target / ".extracted"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            marker.write_text("ok\n", encoding="utf-8")
        txt = target / f"{code}.txt"
        if not txt.is_file():
            candidates = [path for path in target.glob("*.txt") if path.name.lower() != "readme.txt"]
            if len(candidates) != 1:
                raise FileNotFoundError(f"Could not identify GeoNames data file for {code}: {candidates}")
            txt = candidates[0]
        all_rows.extend(_parse_geonames(txt))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[row["feature_class"]].append(row)
    selected = balanced_sample(groups, N_EXAMPLES, SEEDS[leaf], key=lambda x: f"{x['id']}:{x['name']}")
    ontology = {code: GEONAMES_CLASS_NAMES[code] for code in sorted(GEONAMES_CLASS_NAMES)}
    choices = [f"{code} — {ontology[code]}" for code in ontology]
    legend = "; ".join(choices)
    records: list[dict[str, Any]] = []
    for i, (_, item) in enumerate(selected):
        code = item["feature_class"]
        answer = f"{code} — {ontology[code]}"
        record = base_record(
            leaf,
            i,
            "GeoNames geographical database",
            "https://download.geonames.org/export/dump/",
            "CC-BY-4.0",
            item["id"],
        )
        record.update(
            {
                "input": {
                    "question": (
                        f"Classify the geographic entity “{item['name']}” using exactly one GeoNames "
                        f"feature class. Closed-set legend: {legend}."
                    ),
                    "mention": item["name"],
                    "context": {
                        "country_code": item["country_code"],
                        "latitude": item["latitude"],
                        "longitude": item["longitude"],
                    },
                    "ontology": ontology,
                    "choices": choices,
                },
                "target": {
                    "answer": answer,
                    "feature_class": code,
                    "feature_class_name": ontology[code],
                    "feature_code": item["feature_code"],
                    "choice_index": choices.index(answer),
                },
                "evaluation": {"type": "classification", "metrics": ["accuracy", "macro_f1"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records, {"ontology": ontology})


WORLDCLIM_BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
ENVIRONMENTAL_LAYERS = {
    "elevation": {
        "label": "elevation",
        "archive": "wc2.1_10m_elev.zip",
        "file": "wc2.1_10m_elev.tif",
        "unit": "metres above sea level",
        "scale": 1.0,
    },
    "annual_mean_temperature": {
        "label": "annual mean temperature",
        "archive": "wc2.1_10m_bio.zip",
        "file": "wc2.1_10m_bio_1.tif",
        "unit": "degrees Celsius",
        "scale": 0.1,
    },
    "annual_temperature_range": {
        "label": "annual temperature range",
        "archive": "wc2.1_10m_bio.zip",
        "file": "wc2.1_10m_bio_7.tif",
        "unit": "degrees Celsius",
        "scale": 0.1,
    },
    "temperature_seasonality": {
        "label": "temperature seasonality",
        "archive": "wc2.1_10m_bio.zip",
        "file": "wc2.1_10m_bio_4.tif",
        "unit": "standard-deviation index × 100",
        "scale": 1.0,
    },
    "annual_precipitation": {
        "label": "annual precipitation",
        "archive": "wc2.1_10m_bio.zip",
        "file": "wc2.1_10m_bio_12.tif",
        "unit": "millimetres",
        "scale": 1.0,
    },
    "precipitation_seasonality": {
        "label": "precipitation seasonality",
        "archive": "wc2.1_10m_bio.zip",
        "file": "wc2.1_10m_bio_15.tif",
        "unit": "coefficient of variation",
        "scale": 1.0,
    },
}


def _worldclim_layer_paths(cache: Path) -> dict[str, Path]:
    root = cache / "worldclim_2_1_10m"
    extracted: dict[str, Path] = {}
    for archive_name in sorted({spec["archive"] for spec in ENVIRONMENTAL_LAYERS.values()}):
        archive = download(f"{WORLDCLIM_BASE_URL}/{archive_name}", root / archive_name)
        extracted[archive_name] = extract_zip(archive, root / archive_name.removesuffix(".zip"))
    paths: dict[str, Path] = {}
    for layer_id, spec in ENVIRONMENTAL_LAYERS.items():
        matches = list(extracted[spec["archive"]].rglob(spec["file"]))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {spec['file']}, found {matches}")
        paths[layer_id] = matches[0]
    return paths


def _read_environmental_patch(path: Path, longitude: float, latitude: float, scale: float):
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    half_width = 4.0
    with rasterio.open(path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        left, bottom = transformer.transform(longitude - half_width, latitude - half_width)
        right, top = transformer.transform(longitude + half_width, latitude + half_width)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        patch = src.read(
            1,
            window=window,
            out_shape=(128, 128),
            boundless=True,
            masked=True,
            resampling=Resampling.bilinear,
        ).astype("float64")
    patch *= float(scale)
    compressed = patch.compressed()
    if compressed.size < patch.size * 0.65:
        raise ValueError(f"Too little valid raster coverage around {latitude}, {longitude}")
    if not np.isfinite(compressed).all():
        raise ValueError("Non-finite environmental raster values")
    return patch


def _render_environmental_patch(patch, destination: Path) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    values = patch.compressed()
    low, high = np.percentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(values.min()), float(values.max() + 1e-6)
    fig, ax = plt.subplots(figsize=(5.0, 4.3), dpi=130)
    image = ax.imshow(patch, cmap="viridis", vmin=low, vmax=high, interpolation="bilinear")
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.ax.set_ylabel("Raster value", rotation=270, labelpad=13)
    fig.tight_layout(pad=0.15)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return {
        "minimum": round(float(values.min()), 4),
        "maximum": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "standard_deviation": round(float(values.std()), 4),
    }


def generate_environmental_layer(cache: Path, koppen_raster: Path | None, output: Path) -> Path:
    """Build a balanced multi-layer identification task.

    ``koppen_raster`` is retained as a deprecated positional argument for CLI/API
    compatibility. The revised task uses six official WorldClim layers so the
    target is no longer constant.
    """
    leaf = "environmental_layer_identification"
    _, places = natural_earth(cache)
    rows = [
        row
        for _, row in places.iterrows()
        if row.geometry and not row.geometry.is_empty and abs(float(row.geometry.y)) <= 70
    ]
    layer_paths = _worldclim_layer_paths(cache)
    groups = {layer_id: rows for layer_id in ENVIRONMENTAL_LAYERS}
    selected = balanced_sample(
        groups,
        N_EXAMPLES,
        SEEDS[leaf],
        key=lambda row: f"{_place_name(row)}:{row.geometry.x:.5f}:{row.geometry.y:.5f}",
    )
    choices = [ENVIRONMENTAL_LAYERS[layer_id]["label"] for layer_id in sorted(ENVIRONMENTAL_LAYERS)]
    templates = (
        "Which environmental variable is represented by this raster patch around {place}?",
        "Identify the environmental layer visualized for the region centred on {place}.",
        "What type of environmental raster is shown near {place}?",
        "Select the variable mapped in this unlabeled raster around {place}.",
    )
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (layer_id, row) in enumerate(selected):
        spec = ENVIRONMENTAL_LAYERS[layer_id]
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        patch = _read_environmental_patch(layer_paths[layer_id], lon, lat, spec["scale"])
        image_path = task_dir / "assets" / f"{i:03d}_{layer_id}.png"
        statistics = _render_environmental_patch(patch, image_path)
        place = _place_name(row)
        answer = spec["label"]
        record = base_record(
            leaf,
            i,
            "WorldClim 2.1 global climate and elevation rasters",
            "https://www.worldclim.org/data/worldclim21.html",
            "WorldClim data license; free for research and related activities",
            f"{layer_id}:{place}",
        )
        record.update(
            {
                "input": {
                    "images": [image_path.relative_to(task_dir).as_posix()],
                    "question": templates[i % len(templates)].format(place=place),
                    "choices": choices,
                    "coordinate": {"longitude": lon, "latitude": lat, "crs": "EPSG:4326"},
                    "note": "The colorbar is intentionally unlabeled; infer the layer from its values and spatial pattern.",
                },
                "target": {
                    "answer": answer,
                    "choice_index": choices.index(answer),
                    "layer_id": layer_id,
                    "unit": spec["unit"],
                    "summary_statistics": statistics,
                },
                "evaluation": {"type": "classification", "metrics": ["accuracy", "macro_f1"]},
            }
        )
        records.append(record)
    distribution = {layer_id: sum(r["target"]["layer_id"] == layer_id for r in records) for layer_id in ENVIRONMENTAL_LAYERS}
    return finalize_task(
        output,
        leaf,
        records,
        {"layer_distribution": distribution, "deprecated_koppen_argument_used": koppen_raster is not None},
    )

