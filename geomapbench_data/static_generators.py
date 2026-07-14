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


def generate_coordinate_transform(cache: Path, output: Path) -> Path:
    from pyproj import Transformer

    leaf = "coordinate_transformation"
    _, places = natural_earth(cache)
    candidates = [row for _, row in places.iterrows() if row.geometry and not row.geometry.is_empty and abs(row.geometry.y) < 80]
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda row: f"{_place_name(row)}:{row.geometry.x:.6f}")
    records: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        zone = int((lon + 180) // 6) + 1
        target_epsg = (32600 if lat >= 0 else 32700) + zone
        transformer = Transformer.from_crs(4326, target_epsg, always_xy=True)
        easting, northing = transformer.transform(lon, lat)
        record = base_record(
            leaf,
            i,
            "Natural Earth populated places + PROJ/EPSG",
            "https://www.naturalearthdata.com/",
            "Natural Earth public domain; PROJ data terms apply",
            _place_name(row),
        )
        record.update(
            {
                "input": {
                    "question": f"Transform WGS 84 longitude {lon:.6f}, latitude {lat:.6f} to EPSG:{target_epsg}.",
                    "coordinate": {"crs": "EPSG:4326", "longitude": lon, "latitude": lat},
                    "target_crs": f"EPSG:{target_epsg}",
                },
                "target": {
                    "easting_m": round(easting, 3),
                    "northing_m": round(northing, 3),
                    "crs": f"EPSG:{target_epsg}",
                },
                "evaluation": {"type": "numeric", "absolute_tolerance_m": 1.0},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


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
    geod = Geod(ellps="WGS84")
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(pairs):
        _, _, meters = geod.inv(a.geometry.x, a.geometry.y, b.geometry.x, b.geometry.y)
        name_a, name_b = _place_name(a), _place_name(b)
        image_path = task_dir / "assets" / f"{i:03d}.png"
        _render_world_pair(countries, a.geometry, b.geometry, image_path, name_a, name_b)
        record = base_record(
            leaf,
            i,
            "Natural Earth populated places",
            "https://www.naturalearthdata.com/",
            "Public domain",
            f"{name_a}|{name_b}",
        )
        record.update(
            {
                "input": {
                    "images": [image_path.relative_to(task_dir).as_posix()],
                    "question": f"What is the WGS 84 geodesic distance from A ({name_a}) to B ({name_b}) in kilometres?",
                    "points": [
                        {"name": name_a, "longitude": a.geometry.x, "latitude": a.geometry.y},
                        {"name": name_b, "longitude": b.geometry.x, "latitude": b.geometry.y},
                    ],
                },
                "target": {"distance_km": round(meters / 1000, 3), "method": "WGS84 inverse geodesic"},
                "evaluation": {"type": "numeric", "relative_tolerance": 0.005},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _cardinal(dx: float, dy: float) -> str:
    angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    labels = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return labels[int((angle + 22.5) // 45) % 8]


def generate_topology_direction(cache: Path, output: Path) -> Path:
    import geopandas as gpd

    leaf = "topological_directional_reasoning"
    countries, places = natural_earth(cache)
    countries = countries[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    places = places[places.geometry.notna() & ~places.geometry.is_empty].copy()
    joined = gpd.sjoin(places, countries[["geometry", "ADMIN"]], predicate="within", how="inner")
    within_rows = [row for _, row in joined.iterrows()]
    within_selected = stable_sample(within_rows, 50, SEEDS[leaf], key=lambda row: f"{_place_name(row)}:{row.get('ADMIN','')}")

    country_rows = [row for _, row in countries.iterrows()]
    rng = random.Random(SEEDS[leaf] + 1)
    pair_rows: list[tuple[Any, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(pair_rows) < 50:
        a, b = rng.sample(country_rows, 2)
        key = tuple(sorted((_country_name(a), _country_name(b))))
        if key not in seen:
            seen.add(key)
            pair_rows.append((a, b))

    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    mixed: list[tuple[str, Any]] = [("within", x) for x in within_selected] + [("direction", x) for x in pair_rows]
    rng.shuffle(mixed)
    for i, (kind, item) in enumerate(mixed):
        if kind == "within":
            row = item
            place_name, country_name = _place_name(row), str(row["ADMIN"])
            country = countries[countries["ADMIN"] == country_name].iloc[0]
            image_path = task_dir / "assets" / f"{i:03d}.png"
            _render_world_pair(countries, row.geometry, country.geometry.representative_point(), image_path, place_name, country_name)
            question = f"Is point A ({place_name}) within region B ({country_name})?"
            target = {"relation": "within", "answer": "yes"}
            group_id = country_name
        else:
            a, b = item
            pa, pb = a.geometry.representative_point(), b.geometry.representative_point()
            relation = "touches" if a.geometry.touches(b.geometry) else _cardinal(pa.x - pb.x, pa.y - pb.y)
            name_a, name_b = _country_name(a), _country_name(b)
            image_path = task_dir / "assets" / f"{i:03d}.png"
            _render_world_pair(countries, pa, pb, image_path, name_a, name_b)
            if relation == "touches":
                question = f"What topological relation holds between A ({name_a}) and B ({name_b})?"
            else:
                question = f"What is the approximate cardinal direction of A ({name_a}) relative to B ({name_b})?"
            target = {"relation": relation, "answer": relation}
            group_id = f"{name_a}|{name_b}"
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
                "input": {"images": [image_path.relative_to(task_dir).as_posix()], "question": question},
                "target": target,
                "evaluation": {"type": "relation_classification", "metric": "accuracy"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


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
        txt = next(target.glob("*.txt"))
        all_rows.extend(_parse_geonames(txt))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[row["feature_class"]].append(row)
    selected = balanced_sample(groups, N_EXAMPLES, SEEDS[leaf], key=lambda x: f"{x['id']}:{x['name']}")
    choices = [GEONAMES_CLASS_NAMES[k] for k in sorted(GEONAMES_CLASS_NAMES)]
    records: list[dict[str, Any]] = []
    for i, (_, item) in enumerate(selected):
        answer = GEONAMES_CLASS_NAMES[item["feature_class"]]
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
                    "question": f"What geographic entity type is the place name “{item['name']}”?",
                    "mention": item["name"],
                    "context": {"country_code": item["country_code"], "latitude": item["latitude"], "longitude": item["longitude"]},
                    "choices": choices,
                },
                "target": {
                    "answer": answer,
                    "feature_class": item["feature_class"],
                    "feature_code": item["feature_code"],
                },
                "evaluation": {"type": "classification", "metric": "macro_f1"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


KOPPEN_CLASSES = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc",
    20: "Dsd", 21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF",
}


def generate_environmental_layer(cache: Path, koppen_raster: Path, output: Path) -> Path:
    import rasterio
    from pyproj import Transformer

    leaf = "environmental_layer_identification"
    _, places = natural_earth(cache)
    rows = [row for _, row in places.iterrows() if row.geometry and not row.geometry.is_empty]
    candidates: list[tuple[Any, int]] = []
    with rasterio.open(koppen_raster) as src:
        transformer = Transformer.from_crs(4326, src.crs, always_xy=True)
        for row in rows:
            x, y = transformer.transform(row.geometry.x, row.geometry.y)
            value = int(next(src.sample([(x, y)]))[0])
            if value in KOPPEN_CLASSES:
                candidates.append((row, value))
    selected = stable_sample(candidates, N_EXAMPLES, SEEDS[leaf], key=lambda x: f"{_place_name(x[0])}:{x[1]}")
    records: list[dict[str, Any]] = []
    for i, (row, value) in enumerate(selected):
        code = KOPPEN_CLASSES[value]
        name = _place_name(row)
        record = base_record(
            leaf,
            i,
            "1-km Köppen–Geiger climate classification",
            "https://doi.org/10.5281/zenodo.5347837",
            "CC-BY-4.0",
            name,
        )
        record.update(
            {
                "input": {
                    "question": f"What Köppen–Geiger climate class occurs at {name} ({row.geometry.y:.5f}, {row.geometry.x:.5f})?",
                    "coordinate": {"longitude": row.geometry.x, "latitude": row.geometry.y, "crs": "EPSG:4326"},
                },
                "target": {"class_code": code, "raster_value": value},
                "evaluation": {"type": "classification", "metric": "accuracy"},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records, {"source_raster_sha256_note": "Recorded by the release pipeline"})

