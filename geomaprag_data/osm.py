from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, box
from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, make_record, slugify
from .config import BuildProfile, CAPABILITY_HINTS, select_places
from .http import CachedHTTP


OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def _bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111_320.0
    cos_lat = max(0.1, abs(math.cos(math.radians(lat))))
    lon_delta = radius_m / (111_320.0 * cos_lat)
    return lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta


def _query(lat: float, lon: float, radius_m: int) -> str:
    south, west, north, east = _bbox(lat, lon, radius_m)
    bbox = f"{south},{west},{north},{east}"
    # One medium-size regional query can generate several local map crops. POIs
    # are node-only to avoid the expensive broad nwr(... out geom) pattern that
    # caused the v1 notebook to time out repeatedly.
    return f"""
[out:json][timeout:90][maxsize:268435456];
(
  way["highway"]({bbox});
  way["waterway"]({bbox});
  way["railway"]({bbox});
  way["natural"]({bbox});
  way["landuse"]({bbox});
  way["leisure"]({bbox});
  way["building"]["name"]({bbox});
);
out tags center geom qt;
(
  node["name"]["amenity"]({bbox});
  node["name"]["tourism"]({bbox});
  node["name"]["place"]({bbox});
  node["name"]["railway"]({bbox});
  node["name"]["natural"]({bbox});
);
out tags qt;
""".strip()


def _center(element: dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        if "lat" in element and "lon" in element:
            return float(element["lat"]), float(element["lon"])
    except Exception:
        pass
    center = element.get("center") or {}
    try:
        return float(center["lat"]), float(center["lon"])
    except Exception:
        pass
    geometry = element.get("geometry") or []
    points: list[tuple[float, float]] = []
    for point in geometry:
        try:
            points.append((float(point["lat"]), float(point["lon"])))
        except Exception:
            pass
    if points:
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
    return None, None


def _category(tags: dict[str, Any]) -> str:
    for key in ("highway", "waterway", "railway", "natural", "landuse", "leisure", "building", "amenity", "tourism", "place"):
        if key in tags:
            return key
    return "other"


def _utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _tile_offsets(count: int, offset: float) -> list[tuple[str, float, float]]:
    candidates = [
        ("center", 0.0, 0.0),
        ("north", 0.0, offset),
        ("south", 0.0, -offset),
        ("east", offset, 0.0),
        ("west", -offset, 0.0),
        ("northeast", offset * 0.7, offset * 0.7),
        ("northwest", -offset * 0.7, offset * 0.7),
        ("southeast", offset * 0.7, -offset * 0.7),
        ("southwest", -offset * 0.7, -offset * 0.7),
    ]
    return candidates[: max(1, min(count, len(candidates)))]


def _projected_features(elements: list[dict[str, Any]], transformer: Transformer) -> list[tuple[str, dict[str, Any], Any]]:
    features: list[tuple[str, dict[str, Any], Any]] = []
    for element in elements:
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        projected: list[tuple[float, float]] = []
        for point in geometry:
            try:
                x, y = transformer.transform(float(point["lon"]), float(point["lat"]))
                projected.append((x, y))
            except Exception:
                pass
        shape = None
        if len(projected) >= 2:
            closed = len(projected) >= 4 and projected[0] == projected[-1]
            polygon_feature = closed and any(k in tags for k in ("landuse", "natural", "leisure", "building"))
            try:
                shape = Polygon(projected) if polygon_feature else LineString(projected)
                if polygon_feature and not shape.is_valid:
                    shape = shape.buffer(0)
            except Exception:
                shape = None
        else:
            lat, lon = _center(element)
            if lat is not None and lon is not None:
                try:
                    x, y = transformer.transform(lon, lat)
                    shape = Point(x, y)
                except Exception:
                    pass
        if shape is not None and not shape.is_empty:
            features.append((_category(tags), tags, shape))
    return features


def render_map_tile(
    elements: list[dict[str, Any]],
    region_lat: float,
    region_lon: float,
    offset_x: float,
    offset_y: float,
    radius_m: float,
    destination: Path,
) -> tuple[bool, dict[str, float]]:
    epsg = _utm_epsg(region_lat, region_lon)
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    center_x, center_y = forward.transform(region_lon, region_lat)
    tile_x, tile_y = center_x + offset_x, center_y + offset_y
    tile_lon, tile_lat = inverse.transform(tile_x, tile_y)
    clip = box(tile_x - radius_m, tile_y - radius_m, tile_x + radius_m, tile_y + radius_m)
    features = _projected_features(elements, forward)
    clipped_features: list[tuple[str, dict[str, Any], Any]] = []
    for category, tags, shape in features:
        try:
            clipped = shape.intersection(clip)
        except Exception:
            continue
        if not clipped.is_empty:
            clipped_features.append((category, tags, clipped))
    if len(clipped_features) < 4:
        return False, {"lat": tile_lat, "lon": tile_lon}

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f7f7f5")

    # Area fills first.
    for category, tags, shape in clipped_features:
        parts = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            x, y = part.exterior.xy
            if category == "natural":
                face = "#e2eee0"
            elif category == "leisure":
                face = "#e8efe0"
            elif category == "building":
                face = "#e2ded7"
            else:
                face = "#eee9df"
            ax.fill(x, y, facecolor=face, edgecolor="#d1d1cc", linewidth=0.25, zorder=1)

    road_width = {
        "motorway": 2.4,
        "trunk": 2.1,
        "primary": 1.8,
        "secondary": 1.4,
        "tertiary": 1.1,
        "residential": 0.7,
        "service": 0.45,
        "pedestrian": 0.45,
    }
    for category, tags, shape in clipped_features:
        parts = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
        for part in parts:
            if part.geom_type != "LineString":
                continue
            x, y = part.xy
            if category == "waterway":
                ax.plot(x, y, color="#6fa8cf", linewidth=1.15, zorder=4)
            elif category == "railway":
                ax.plot(x, y, color="#5d5d5d", linewidth=0.85, linestyle="--", zorder=5)
            elif category == "highway":
                highway = str(tags.get("highway", ""))
                width = road_width.get(highway, 0.55)
                major = highway in {"motorway", "trunk", "primary"}
                ax.plot(
                    x,
                    y,
                    color="#555555" if major else "#969696",
                    linewidth=width,
                    solid_capstyle="round",
                    zorder=6,
                )

    poi_count = 0
    for category, tags, shape in clipped_features:
        if category not in {"amenity", "tourism", "place", "railway", "natural"} or poi_count >= 35:
            continue
        try:
            point = shape if shape.geom_type == "Point" else shape.representative_point()
            ax.scatter([point.x], [point.y], s=6, color="#454545", linewidths=0, zorder=8)
            poi_count += 1
        except Exception:
            pass

    ax.set_xlim(tile_x - radius_m, tile_x + radius_m)
    ax.set_ylim(tile_y - radius_m, tile_y + radius_m)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(destination, dpi=160, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return True, {"lat": float(tile_lat), "lon": float(tile_lon), "projection_epsg": epsg}


def _feature_records(
    elements: list[dict[str, Any]],
    city: str,
    country: str,
    limit: int,
    guard: BenchmarkGuard,
    existing: set[str],
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    priority_map = {
        "place": 0,
        "waterway": 1,
        "railway": 2,
        "natural": 3,
        "amenity": 4,
        "tourism": 5,
        "leisure": 6,
        "highway": 7,
        "landuse": 8,
        "building": 9,
    }
    for element in elements:
        osm_id = str(element.get("id") or "")
        if not osm_id or osm_id in guard.osm_ids:
            continue
        tags = element.get("tags") or {}
        category = _category(tags)
        name = tags.get("name")
        if category == "highway" and not name:
            continue
        if not name and category in {"amenity", "tourism", "place", "building", "leisure"}:
            continue
        candidates.append((0 if name else 1, priority_map.get(category, 99), element))
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("type")), int(item[2].get("id") or 0)))

    records: list[dict[str, Any]] = []
    for _, _, element in candidates:
        if len(records) >= limit:
            break
        element_type = str(element.get("type") or "element")
        osm_id = str(element.get("id") or "")
        record_id = f"osm:{element_type}:{osm_id}"
        if record_id in existing:
            continue
        tags = element.get("tags") or {}
        category = _category(tags)
        name = str(tags.get("name") or f"OSM {element_type} {osm_id}")
        lat, lon = _center(element)
        if lat is not None and lon is not None and guard.near(lat, lon):
            continue
        useful_keys = (
            "highway",
            "place",
            "amenity",
            "tourism",
            "railway",
            "waterway",
            "natural",
            "landuse",
            "leisure",
            "building",
            "surface",
            "oneway",
            "maxspeed",
            "bridge",
            "tunnel",
        )
        useful_tags = {key: tags[key] for key in useful_keys if key in tags}
        description = (
            f"OpenStreetMap {element_type} {osm_id}. Name: {name}. "
            + "; ".join(f"{key}={value}" for key, value in useful_tags.items())
        ).strip()
        if guard.reject_text(description):
            continue
        geo = None if lat is None or lon is None else {"lat": lat, "lon": lon, "region": city, "country": country}
        records.append(
            make_record(
                record_id=record_id,
                source_name="OpenStreetMap",
                source_url=f"https://www.openstreetmap.org/{element_type}/{osm_id}",
                license_name="ODbL 1.0",
                attribution="© OpenStreetMap contributors",
                group_id=f"{element_type}/{osm_id}",
                modality="structured",
                title=name,
                text=description,
                source_id=f"{element_type}/{osm_id}",
                geo=geo,
                capabilities=CAPABILITY_HINTS["OpenStreetMap"],
                document_type=f"osm_{category}_feature",
                generator="geomaprag_data.osm",
                extra={"osm_type": element_type, "osm_id": osm_id, "tags": useful_tags},
            )
        )
    return records


def build_osm(
    workspace: CorpusWorkspace,
    profile: BuildProfile,
    guard: BenchmarkGuard,
    *,
    checkpoint_every: int = 10,
    courtesy_sleep_seconds: float = 1.25,
) -> dict[str, Any]:
    http = CachedHTTP(workspace.cache_dir)
    seeds = select_places(profile.osm_seed_count)
    existing = workspace.existing_ids()
    failures: list[dict[str, str]] = []
    written = 0
    cached = 0
    maps_written = 0

    bar = tqdm(total=len(seeds), desc="OSM regions", unit="region", dynamic_ncols=True)
    for index, (city, country, lat, lon) in enumerate(seeds, 1):
        unit = f"{index:03d}_{country}_{city}"
        if workspace.shard_done("osm", unit):
            cached += 1
            bar.set_postfix(city=city, status="cached")
            bar.update(1)
            continue
        if guard.near(lat, lon):
            workspace.write_shard(
                "osm",
                unit,
                [],
                meta={"status": "benchmark_spatial_exclusion", "city": city, "country": country},
            )
            bar.set_postfix(city=city, status="excluded")
            bar.update(1)
            continue

        try:
            query = _query(lat, lon, profile.osm_radius_m)
            payload = http.post_json_rotating(
                OVERPASS_ENDPOINTS,
                {"data": query},
                f"overpass/v2/{country}/{city}/{profile.osm_radius_m}",
                timeout=140,
                attempts_per_endpoint=2,
            )
            elements = payload.get("elements") or []
            records = _feature_records(
                elements,
                city,
                country,
                profile.osm_docs_per_region,
                guard,
                existing,
            )

            tile_records: list[dict[str, Any]] = []
            region_dir = workspace.maps_dir / "osm_v2" / f"{index:03d}_{country}_{slugify(city)}"
            for tile_name, dx, dy in _tile_offsets(profile.osm_maps_per_region, profile.osm_tile_offset_m):
                destination = region_dir / f"{tile_name}.png"
                rendered, tile_geo = render_map_tile(
                    elements,
                    lat,
                    lon,
                    dx,
                    dy,
                    profile.osm_map_radius_m,
                    destination,
                )
                if not rendered:
                    continue
                if guard.near(tile_geo["lat"], tile_geo["lon"]):
                    destination.unlink(missing_ok=True)
                    continue
                record_id = f"osm-map-v2:{country}:{slugify(city)}:{tile_name}"
                if record_id in existing:
                    continue
                relative = destination.relative_to(workspace.root).as_posix()
                tile_records.append(
                    make_record(
                        record_id=record_id,
                        source_name="OpenStreetMap",
                        source_url="https://www.openstreetmap.org/",
                        license_name="ODbL 1.0",
                        attribution="© OpenStreetMap contributors",
                        group_id=f"{country}:{city}:{tile_name}",
                        modality="map_image",
                        title=f"Unlabeled map context: {city}, {country}, {tile_name}",
                        text=(
                            f"Clean projected OpenStreetMap context near {city}, {country}; "
                            f"tile {tile_name}, approximately {profile.osm_map_radius_m} metres from tile center. "
                            "The rendered image contains no place labels, title, axes, or coordinate text."
                        ),
                        source_id=f"{country}:{city}:{tile_name}",
                        geo={
                            "lat": tile_geo["lat"],
                            "lon": tile_geo["lon"],
                            "seed_city": city,
                            "seed_country": country,
                            "radius_m": profile.osm_map_radius_m,
                            "projection_epsg": tile_geo["projection_epsg"],
                        },
                        media_paths=[relative],
                        capabilities=CAPABILITY_HINTS["OpenStreetMap"],
                        document_type="unlabeled_projected_map_crop",
                        generator="geomaprag_data.osm",
                        extra={
                            "tile": tile_name,
                            "labels_in_pixels": False,
                            "coordinate_axes_in_pixels": False,
                            "metric_clipping": True,
                            "semantic_styling": True,
                        },
                    )
                )

            all_records = records + tile_records
            workspace.write_shard(
                "osm",
                unit,
                all_records,
                meta={
                    "status": "complete",
                    "city": city,
                    "country": country,
                    "element_count": len(elements),
                    "structured_count": len(records),
                    "map_count": len(tile_records),
                    "overpass_endpoints": OVERPASS_ENDPOINTS,
                },
            )
            for record in all_records:
                existing.add(record["id"])
            written += len(records)
            maps_written += len(tile_records)
            bar.set_postfix(city=city, docs=len(records), maps=len(tile_records), status="ok")
            if courtesy_sleep_seconds:
                time.sleep(courtesy_sleep_seconds)
        except Exception as error:
            failures.append({"unit": unit, "city": city, "error": repr(error)})
            print(f"\nOSM warning: {city}: {error!r}")
            bar.set_postfix(city=city, status="failed")
        finally:
            bar.update(1)

        if checkpoint_every and index % checkpoint_every == 0:
            workspace.materialize()
    bar.close()
    return {
        "stage": "osm",
        "written": written,
        "maps_written": maps_written,
        "cached_units": cached,
        "failed_units": failures,
    }
