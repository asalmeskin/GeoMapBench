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


def _configure_osmnx(cache: Path):
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache / "osmnx_http"
    ox.settings.requests_timeout = 180
    ox.settings.log_console = False
    return ox


def _feature_id(index: Any) -> str:
    if isinstance(index, tuple):
        return ":".join(str(x) for x in index)
    return str(index)


def _plot_anchor_candidates(gdf, target_name: str, destination: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    letters = ["A", "B", "C", "D"]
    for (_, row), color, letter in zip(gdf.iterrows(), colors, letters):
        geometry = row.geometry
        try:
            import geopandas as gpd

            gpd.GeoSeries([geometry], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=color, linewidth=2)
        except Exception:
            continue
        point = geometry.representative_point()
        ax.annotate(letter, (point.x, point.y), color=color, fontsize=12, weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": color, "pad": 1.5})
        ax.annotate(str(row["name"]), (point.x, point.y), xytext=(9, 0), textcoords="offset points", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1})
    ax.set_title(f"Anchor the label: {target_name}", fontsize=10)
    ax.set_axis_off()
    ax.text(0.01, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color="#555")
    fig.tight_layout(pad=0.3)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination.as_posix()


def generate_osm_label_anchoring(cache: Path, output: Path) -> Path:
    import geopandas as gpd
    import pandas as pd

    leaf = "map_label_feature_anchoring"
    ox = _configure_osmnx(cache)
    rng = random.Random(SEEDS[leaf])
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    cities = WIKIMEDIA_CITIES[:10]
    tags = {"amenity": True, "tourism": True, "leisure": True, "natural": True, "place": True}
    for city_index, (city, country, lat, lon) in enumerate(cities):
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=5000)
        if gdf.empty or "name" not in gdf:
            raise ValueError(f"No named OSM features found for {city}")
        gdf = gdf[gdf["name"].notna() & gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["_fid"] = [_feature_id(idx) for idx in gdf.index]
        # One feature per OSM identity; deterministic order before city-specific shuffle.
        gdf = gdf.sort_values("_fid").drop_duplicates("_fid")
        indices = list(gdf.index)
        random.Random(SEEDS[leaf] + city_index).shuffle(indices)
        if len(indices) < 40:
            raise ValueError(f"Need 40 named OSM features for {city}, found {len(indices)}")
        for local_index in range(10):
            subset_indices = indices[local_index * 4 : local_index * 4 + 4]
            subset = gdf.loc[subset_indices].copy()
            target_position = rng.randrange(4)
            target = subset.iloc[target_position]
            global_index = len(records)
            image_path = task_dir / "assets" / f"{global_index:03d}.png"
            _plot_anchor_candidates(subset, str(target["name"]), image_path)
            choices = ["A", "B", "C", "D"]
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
                        "question": f"Which candidate geometry is anchored to the label “{target['name']}”?",
                        "choices": choices,
                    },
                    "target": {
                        "answer": choices[target_position],
                        "choice_index": target_position,
                        "feature_id": target["_fid"],
                        "geometry": target.geometry.__geo_interface__,
                    },
                    "evaluation": {"type": "multimodal_grounding", "metrics": ["choice_accuracy", "geometry_iou"]},
                }
            )
            records.append(record)
    rng.shuffle(records)
    # Re-assign public IDs after shuffling without changing group/provenance.
    for index, record in enumerate(records):
        record["id"] = f"{leaf}-{index:03d}"
    return finalize_task(output, leaf, records)


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

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    for u, v in graph.edges:
        ax.plot([graph.nodes[u]["x"], graph.nodes[v]["x"]], [graph.nodes[u]["y"], graph.nodes[v]["y"]], color="#c9c9c9", linewidth=0.35)
    if show_polygon and not polygon.is_empty:
        x, y = polygon.exterior.xy
        ax.fill(x, y, color="#6a3d9a", alpha=0.35)
        ax.plot(x, y, color="#6a3d9a", linewidth=1.2)
    ax.scatter([graph.nodes[center]["x"]], [graph.nodes[center]["y"]], c="#e31a1c", s=25, zorder=4)
    ax.set_axis_off()
    ax.text(0.01, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color="#555")
    fig.tight_layout(pad=0.1)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def generate_osm_isochrones(cache: Path, output: Path) -> Path:
    import networkx as nx
    from shapely.geometry import MultiPoint, mapping

    leaf = "isochrone_service_area"
    ox = _configure_osmnx(cache)
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    for city_index, (city, country, lat, lon) in enumerate(WIKIMEDIA_CITIES[:10]):
        graph = ox.graph_from_point((lat, lon), dist=3000, network_type="walk", simplify=True)
        # 1.4 m/s is a declared benchmark assumption, not an OSM attribute.
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["travel_time"] = float(data.get("length", 0.0)) / 1.4
        nodes = sorted(graph.nodes)
        chosen = stable_sample(nodes, 10, SEEDS[leaf] + city_index, key=str)
        for local_index, center in enumerate(chosen):
            global_index = len(records)
            budget_min = 5 + (local_index % 3) * 5
            subgraph = nx.ego_graph(graph, center, radius=budget_min * 60, distance="travel_time")
            points = MultiPoint([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in subgraph.nodes])
            polygon = points.convex_hull
            if polygon.geom_type != "Polygon":
                polygon = points.buffer(0.0001).convex_hull
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
                f"{city}:{center}:{budget_min}",
            )
            record.update(
                {
                    "input": {
                        "images": [input_path.relative_to(task_dir).as_posix()],
                        "question": f"Compute the area reachable within {budget_min} minutes on foot from the red point, assuming 1.4 m/s.",
                        "budget_minutes": budget_min,
                        "origin": {"longitude": graph.nodes[center]["x"], "latitude": graph.nodes[center]["y"]},
                    },
                    "target": {
                        "isochrone_image": target_path.relative_to(task_dir).as_posix(),
                        "isochrone_geojson": mapping(polygon),
                        "reachable_node_count": subgraph.number_of_nodes(),
                    },
                    "evaluation": {"type": "service_area", "metrics": ["polygon_iou", "reachable_node_f1"]},
                }
            )
            records.append(record)
    return finalize_task(output, leaf, records)

