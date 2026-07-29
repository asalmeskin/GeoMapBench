from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .api_generators import WIKIMEDIA_CITIES
from .common import N_EXAMPLES, SEEDS, base_record, copy_asset, finalize_task, stable_sample


OSM_LICENSE = "ODbL-1.0; maps must attribute © OpenStreetMap contributors"

GLOBAL_OSM_CITIES = [
    ("Zurich", "Switzerland", 47.3769, 8.5417),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Warsaw", "Poland", 52.2297, 21.0122),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Nairobi", "Kenya", -1.2864, 36.8172),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Istanbul", "Türkiye", 41.0082, 28.9784),
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Amman", "Jordan", 31.9539, 35.9106),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("New York City", "United States", 40.7128, -74.0060),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("São Paulo", "Brazil", -23.5505, -46.6333),
    ("Bogotá", "Colombia", 4.7110, -74.0721),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Auckland", "New Zealand", -36.8509, 174.7645),
]


def _configure_osmnx(cache: Path):
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache / "osmnx_http"
    ox.settings.requests_timeout = 300
    ox.settings.log_console = False
    if hasattr(ox.settings, "overpass_rate_limit"):
        ox.settings.overpass_rate_limit = True
    return ox


def _feature_id(index: Any) -> str:
    if isinstance(index, tuple):
        return ":".join(str(x) for x in index)
    return str(index)


def _geometry_family(geometry) -> str:
    kind = geometry.geom_type.lower()
    if "point" in kind:
        return "point"
    if "line" in kind:
        return "line"
    return "polygon"


def _iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    elif geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)


def _plot_anchor_candidates(gdf, target_position: int, target_name: str, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    letters = ["A", "B", "C", "D"]
    for position, ((_, row), color, letter) in enumerate(zip(gdf.iterrows(), colors, letters)):
        geometry = row.geometry
        if geometry.geom_type in {"Point", "MultiPoint"}:
            point = geometry.representative_point()
            ax.scatter([point.x], [point.y], s=95, facecolors="none", edgecolors=color, linewidths=2.2)
        else:
            try:
                import geopandas as gpd

                gpd.GeoSeries([geometry], crs=gdf.crs).plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=2.2,
                )
            except Exception:
                continue
            point = geometry.representative_point()
        ax.annotate(
            letter,
            (point.x, point.y),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=12,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": color, "pad": 1.5},
        )
        if position == target_position:
            ax.annotate(
                target_name,
                (point.x, point.y),
                xytext=(14, -16),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
                bbox={"facecolor": "#fff7bc", "alpha": 0.9, "edgecolor": "#666", "pad": 2},
            )
    ax.set_title("Match the displayed map label to its geographic feature", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.text(0.01, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color="#555")
    fig.tight_layout(pad=0.3)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _features_from_point_with_retry(ox, point, tags, distance: int, city: str):
    import time

    last_error = None
    for attempt in range(5):
        try:
            return ox.features_from_point(point, tags=tags, dist=distance)
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Could not download named features for {city}: {last_error}")


def generate_osm_label_anchoring(cache: Path, output: Path) -> Path:
    import geopandas as gpd

    leaf = "map_label_feature_anchoring"
    ox = _configure_osmnx(cache)
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    tags = {
        "amenity": [
            "restaurant", "cafe", "school", "university", "hospital", "clinic",
            "library", "theatre", "cinema", "marketplace", "place_of_worship",
        ],
        "tourism": ["attraction", "museum", "gallery", "hotel", "viewpoint"],
        "leisure": ["park", "garden", "stadium", "sports_centre", "playground"],
        "historic": True,
    }
    choices = ["A", "B", "C", "D"]
    for city_index, (city, country, lat, lon) in enumerate(GLOBAL_OSM_CITIES):
        gdf = _features_from_point_with_retry(ox, (lat, lon), tags, 4000, city)
        if gdf.empty or "name" not in gdf:
            raise ValueError(f"No named OSM features found for {city}")
        gdf = gdf[gdf["name"].notna() & gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["name"] = gdf["name"].astype(str)
        gdf["_fid"] = [_feature_id(index) for index in gdf.index]
        gdf["_family"] = [_geometry_family(geometry) for geometry in gdf.geometry]
        gdf = gdf.sort_values("_fid").drop_duplicates("_fid")
        if len(gdf) < 12:
            raise ValueError(f"Need at least 12 named OSM features for {city}, found {len(gdf)}")
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        projected["_anchor"] = projected.geometry.representative_point()
        targets = stable_sample(list(gdf.index), 5, SEEDS[leaf] + city_index, key=_feature_id)
        for local_index, target_index in enumerate(targets):
            target_projected = projected.loc[[target_index]].iloc[0]
            family = target_projected["_family"]
            target_point = target_projected["_anchor"]
            candidate_pool = projected[projected["_family"] == family].copy()
            if len(candidate_pool) < 4:
                candidate_pool = projected.copy()
            candidate_pool["_distance"] = candidate_pool["_anchor"].distance(target_point)
            negative_indices = [index for index in candidate_pool.sort_values(["_distance", "_fid"]).index if index != target_index][:3]
            if len(negative_indices) < 3:
                raise ValueError(f"Could not construct four candidates for {city} target {target_index}")
            subset_indices = [target_index, *negative_indices]
            rng = random.Random(SEEDS[leaf] + city_index * 100 + local_index)
            rng.shuffle(subset_indices)
            target_position = subset_indices.index(target_index)
            subset = projected.reindex(subset_indices).copy()
            target = gdf.loc[[target_index]].iloc[0]
            global_index = len(records)
            image_path = task_dir / "assets" / f"{global_index:03d}.png"
            _plot_anchor_candidates(subset, target_position, str(target["name"]), image_path)
            anchor_wgs84 = gpd.GeoSeries([target.geometry.representative_point()], crs=gdf.crs).to_crs(4326).iloc[0]
            record = base_record(
                leaf,
                global_index,
                "OpenStreetMap named features via Overpass",
                "https://wiki.openstreetmap.org/wiki/Overpass_API",
                OSM_LICENSE,
                f"{city}:{target['_fid']}",
            )
            record.update(
                {
                    "input": {
                        "images": [image_path.relative_to(task_dir).as_posix()],
                        "question": (
                            f"The displayed map label “{target['name']}” is anchored at its real map position. "
                            "Which candidate geometry (A, B, C, or D) is the geographic feature named by that label?"
                        ),
                        "choices": choices,
                        "city": city,
                        "country": country,
                        "task_definition": "text-to-geographic-feature grounding",
                    },
                    "target": {
                        "answer": choices[target_position],
                        "choice_index": target_position,
                        "feature_id": target["_fid"],
                        "feature_name": str(target["name"]),
                        "geometry_family": target["_family"],
                        "label_anchor": {"longitude": anchor_wgs84.x, "latitude": anchor_wgs84.y},
                        "geometry": target.geometry.__geo_interface__,
                    },
                    "evaluation": {"type": "multimodal_grounding", "metrics": ["choice_accuracy", "geometry_iou"]},
                }
            )
            records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {"cities": [city for city, _, _, _ in GLOBAL_OSM_CITIES], "examples_per_city": 5},
    )


TILE_RE = re.compile(r"(AOI_\d+_[A-Za-z]+_img\d+|img\d+)", re.IGNORECASE)


def _tile_key(path: Path) -> str | None:
    match = TILE_RE.search(path.stem)
    return match.group(1).lower() if match else None


def _spacenet3_pairs(source: Path) -> list[tuple[Path, Path, str]]:
    images: dict[str, list[Path]] = defaultdict(list)
    labels: dict[str, Path] = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        key = _tile_key(path)
        if not key:
            continue
        if path.suffix.lower() in {".tif", ".tiff"}:
            images[key].append(path)
        elif path.suffix.lower() in {".geojson", ".json"} and "road" in path.as_posix().lower():
            labels[key] = path
    pairs: list[tuple[Path, Path, str]] = []
    for key in sorted(set(images) & set(labels)):
        # Prefer the three-band pan-sharpened product when several rasters exist.
        candidates = sorted(images[key], key=lambda p: ("rgb-pansharpen" not in p.as_posix().lower(), p.as_posix()))
        pairs.append((candidates[0], labels[key], key))
    return pairs


def _iter_lines(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    if not geometry:
        return
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "LineString":
        yield coords
    elif kind == "MultiLineString":
        yield from coords


def _segment_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    if all(abs(v) <= 180 for v in (*a, *b)):
        from pyproj import Geod

        _, _, meters = Geod(ellps="WGS84").inv(a[0], a[1], b[0], b[1])
        return float(abs(meters))
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _road_graph(label_path: Path):
    import networkx as nx

    payload = json.loads(label_path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    for feature in payload.get("features", []):
        for line in _iter_lines(feature.get("geometry") or {}):
            for raw_a, raw_b in zip(line, line[1:]):
                a = (round(float(raw_a[0]), 7), round(float(raw_a[1]), 7))
                b = (round(float(raw_b[0]), 7), round(float(raw_b[1]), 7))
                if a == b:
                    continue
                length = _segment_length(a, b)
                if graph.has_edge(a, b):
                    graph[a][b]["length"] = min(graph[a][b]["length"], length)
                else:
                    graph.add_edge(a, b, length=length)
    return graph


def _graph_payload(graph) -> dict[str, Any]:
    nodes = sorted(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    edges = [
        {"source": index[a], "target": index[b], "length": round(float(data["length"]), 3)}
        for a, b, data in sorted(graph.edges(data=True), key=lambda e: (index[e[0]], index[e[1]]))
    ]
    return {"directed": False, "nodes": [{"id": i, "x": p[0], "y": p[1]} for i, p in enumerate(nodes)], "edges": edges}


def sample_spacenet3_graph(source: Path, output: Path) -> Path:
    leaf = "spatial_graph_construction"
    pairs = _spacenet3_pairs(source)
    selected = stable_sample(pairs, N_EXAMPLES, SEEDS[leaf], key=lambda x: x[2])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (image, label, tile) in enumerate(selected):
        graph = _road_graph(label)
        if graph.number_of_edges() == 0:
            raise ValueError(f"Empty road graph for {label}")
        target_path = task_dir / "assets" / f"{i:03d}_graph.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(_graph_payload(graph), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record = base_record(
            leaf,
            i,
            "SpaceNet 3 Roads",
            "https://registry.opendata.aws/spacenet/",
            "CC-BY-SA-4.0",
            tile,
        )
        record.update(
            {
                "input": {
                    "images": [copy_asset(image, task_dir / "assets", f"{i:03d}_image{image.suffix.lower()}")],
                    "question": "Extract the road centerlines and construct a routable graph with metric edge lengths.",
                },
                "target": {"graph": target_path.relative_to(task_dir).as_posix(), "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges()},
                "evaluation": {"type": "road_graph", "metrics": ["apls", "topology_f1"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _route_endpoints(graph):
    import networkx as nx

    component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(component)
    start = sorted(component)[0]
    lengths = nx.single_source_dijkstra_path_length(subgraph, start, weight="length")
    a = max(lengths, key=lambda node: (lengths[node], node))
    lengths = nx.single_source_dijkstra_path_length(subgraph, a, weight="length")
    b = max(lengths, key=lambda node: (lengths[node], node))
    return subgraph, a, b


def _plot_route(graph, route: list[tuple[float, float]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for a, b in graph.edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", linewidth=0.5)
    ax.plot([p[0] for p in route], [p[1] for p in route], color="#e31a1c", linewidth=2.5)
    ax.scatter([route[0][0], route[-1][0]], [route[0][1], route[-1][1]], c=["#33a02c", "#1f78b4"], s=30)
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def sample_spacenet3_shortest_path(source: Path, output: Path) -> Path:
    import networkx as nx

    leaf = "shortest_path_optimization"
    pairs = _spacenet3_pairs(source)
    selected = stable_sample(pairs, N_EXAMPLES, SEEDS[leaf], key=lambda x: x[2])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for i, (image, label, tile) in enumerate(selected):
        graph = _road_graph(label)
        subgraph, start, end = _route_endpoints(graph)
        route = nx.shortest_path(subgraph, start, end, weight="length")
        length = nx.shortest_path_length(subgraph, start, end, weight="length")
        route_path = task_dir / "assets" / f"{i:03d}_route.png"
        _plot_route(subgraph, route, route_path)
        record = base_record(
            leaf,
            i,
            "SpaceNet 3 Roads",
            "https://registry.opendata.aws/spacenet/",
            "CC-BY-SA-4.0",
            tile,
        )
        record.update(
            {
                "input": {
                    "images": [copy_asset(image, task_dir / "assets", f"{i:03d}_image{image.suffix.lower()}")],
                    "question": "Compute the least-distance route between the specified graph coordinates.",
                    "start": list(start),
                    "end": list(end),
                },
                "target": {
                    "route_image": route_path.relative_to(task_dir).as_posix(),
                    "route_coordinates": [list(p) for p in route],
                    "length": round(float(length), 3),
                },
                "evaluation": {"type": "routing", "metrics": ["path_validity", "relative_cost_error", "apls"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


def _plot_osm_isochrone(graph, center, polygon, destination: Path, show_polygon: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    segments = []
    for u, v in graph.edges():
        try:
            segments.append(
                [
                    (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                    (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
    if segments:
        ax.add_collection(LineCollection(segments, colors="#c9c9c9", linewidths=0.35))
    if show_polygon and polygon is not None and not polygon.is_empty:
        for part in _iter_polygon_parts(polygon):
            x, y = part.exterior.xy
            ax.fill(x, y, color="#6a3d9a", alpha=0.35)
            ax.plot(x, y, color="#6a3d9a", linewidth=1.2)
    ax.scatter([graph.nodes[center]["x"]], [graph.nodes[center]["y"]], c="#e31a1c", s=25, zorder=4)
    xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
    ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
    if xs and ys:
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
    ax.set_axis_off()
    ax.text(0.01, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color="#555")
    fig.tight_layout(pad=0.1)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _service_area_polygon(graph, reachable_nodes, ox, buffer_metres: float = 35.0):
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    projected = ox.project_graph(graph)
    reachable = set(reachable_nodes)
    lines = []
    for u, v, data in projected.edges(data=True):
        if u not in reachable or v not in reachable:
            continue
        geometry = data.get("geometry")
        if geometry is None:
            geometry = LineString(
                [
                    (float(projected.nodes[u]["x"]), float(projected.nodes[u]["y"])),
                    (float(projected.nodes[v]["x"]), float(projected.nodes[v]["y"])),
                ]
            )
        if geometry is not None and not geometry.is_empty:
            lines.append(geometry)
    if not lines:
        center = next(iter(reachable))
        geometry = gpd.GeoSeries.from_xy(
            [projected.nodes[center]["x"]],
            [projected.nodes[center]["y"]],
            crs=projected.graph["crs"],
        ).iloc[0].buffer(buffer_metres)
    else:
        geometry = unary_union(lines).buffer(buffer_metres, cap_style=1, join_style=1).buffer(0)
    if geometry.is_empty:
        raise ValueError("Empty buffered service-area geometry")
    area_m2 = float(geometry.area)
    geometry_wgs84 = gpd.GeoSeries([geometry], crs=projected.graph["crs"]).to_crs(4326).iloc[0]
    return geometry_wgs84, area_m2


def _graph_from_point_with_retry(ox, point, distance: int, city: str):
    import time

    last_error = None
    for attempt in range(5):
        try:
            return ox.graph_from_point(
                point,
                dist=distance,
                network_type="walk",
                simplify=True,
                retain_all=False,
            )
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Could not download walking network for {city}: {last_error}")


def generate_osm_isochrones(cache: Path, output: Path) -> Path:
    import gc
    import networkx as nx
    from shapely.geometry import mapping

    leaf = "isochrone_service_area"
    ox = _configure_osmnx(cache)
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    walking_speeds_mps = (0.9, 1.1, 1.3, 1.5, 1.7)
    budgets_min = (5, 10, 15, 20)
    query_distance = 3200

    for city_index, (city, country, lat, lon) in enumerate(GLOBAL_OSM_CITIES):
        graph = _graph_from_point_with_retry(ox, (lat, lon), query_distance, city)
        if graph.number_of_nodes() < 10 or graph.number_of_edges() == 0:
            raise ValueError(f"Walking graph for {city} is too small")
        if graph.is_directed():
            largest_component = max(nx.weakly_connected_components(graph), key=len)
        else:
            largest_component = max(nx.connected_components(graph), key=len)
        ranked_nodes = sorted(
            largest_component,
            key=lambda node: (
                (float(graph.nodes[node]["x"]) - lon) ** 2 + (float(graph.nodes[node]["y"]) - lat) ** 2,
                str(node),
            ),
        )
        candidate_nodes = ranked_nodes[: min(300, len(ranked_nodes))]
        chosen = stable_sample(candidate_nodes, 5, SEEDS[leaf] + city_index, key=str)
        for local_index, center in enumerate(chosen):
            global_index = len(records)
            budget_min = budgets_min[global_index % len(budgets_min)]
            speed_mps = walking_speeds_mps[(global_index * 3) % len(walking_speeds_mps)]
            distance_budget_m = budget_min * 60.0 * speed_mps
            reachable_lengths = nx.single_source_dijkstra_path_length(
                graph,
                center,
                cutoff=distance_budget_m,
                weight="length",
            )
            reachable_nodes = sorted(reachable_lengths, key=str)
            if not reachable_nodes:
                reachable_nodes = [center]
            polygon, area_m2 = _service_area_polygon(graph, reachable_nodes, ox)
            input_path = task_dir / "assets" / f"{global_index:03d}_input.png"
            target_path = task_dir / "assets" / f"{global_index:03d}_isochrone.png"
            _plot_osm_isochrone(graph, center, polygon, input_path, False)
            _plot_osm_isochrone(graph, center, polygon, target_path, True)
            record = base_record(
                leaf,
                global_index,
                "OpenStreetMap pedestrian network via Overpass",
                "https://wiki.openstreetmap.org/wiki/Overpass_API",
                OSM_LICENSE,
                f"{city}:{center}:{budget_min}:{speed_mps:.1f}",
            )
            record.update(
                {
                    "input": {
                        "images": [input_path.relative_to(task_dir).as_posix()],
                        "question": (
                            f"Compute the pedestrian service area reachable within {budget_min} minutes from the red "
                            f"point at a constant walking speed of {speed_mps:.1f} m/s. Respect the street network; "
                            "do not fill disconnected barriers or gaps."
                        ),
                        "budget_minutes": budget_min,
                        "speed_mps": speed_mps,
                        "network_distance_budget_m": distance_budget_m,
                        "city": city,
                        "country": country,
                        "origin": {
                            "longitude": float(graph.nodes[center]["x"]),
                            "latitude": float(graph.nodes[center]["y"]),
                        },
                    },
                    "target": {
                        "isochrone_image": target_path.relative_to(task_dir).as_posix(),
                        "isochrone_geojson": mapping(polygon),
                        "reachable_node_count": len(reachable_nodes),
                        "reachable_node_ids": [str(node) for node in reachable_nodes],
                        "service_area_m2": round(area_m2, 3),
                        "construction_method": "buffered reachable street edges in a local projected CRS",
                    },
                    "evaluation": {
                        "type": "service_area",
                        "metrics": ["polygon_iou", "reachable_node_f1", "relative_area_error"],
                    },
                }
            )
            records.append(record)
            del reachable_lengths, reachable_nodes, polygon
            gc.collect()
        del graph
        gc.collect()
    return finalize_task(
        output,
        leaf,
        records,
        {
            "cities": [city for city, _, _, _ in GLOBAL_OSM_CITIES],
            "examples_per_city": 5,
            "walking_speeds_mps": list(walking_speeds_mps),
            "budgets_minutes": list(budgets_min),
            "polygon_method": "buffered_reachable_edges",
        },
    )

