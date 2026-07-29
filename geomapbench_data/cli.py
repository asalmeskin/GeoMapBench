from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .api_generators import (
    generate_cross_entity_comparison,
    generate_geology,
    generate_population_density,
    generate_visual_geolocation,
)
from .common import SEEDS
from .download import download_zenodo, extract_zip
from .network_generators import (
    generate_osm_isochrones,
    generate_osm_label_anchoring,
    sample_spacenet3_graph,
    sample_spacenet3_shortest_path,
)
from .samplers import (
    sample_eurosat,
    sample_geoquestions,
    sample_geowebnews,
    sample_maki,
    sample_maptext,
    sample_openearthmap,
    sample_rsvqa,
    sample_spacenet7_change,
    sample_spacenet7_matching,
    sample_spartun,
)
from .static_generators import (
    generate_coordinate_transform,
    generate_environmental_layer,
    generate_geo_entity_typing,
    generate_metric_distance,
    generate_topology_direction,
)
from .validate import validate_root


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


KNOWN_ZENODO = {
    "maptext": ("14982663", ["images.zip", "maptext_format.json"]),
    "eurosat": ("7711810", ["EuroSAT_RGB.zip"]),
    "rsvqa": (
        "6344334",
        ["all_answers.json", "all_questions.json", "Images_LR.zip"],
    ),
}


def fetch_known(name: str, raw_root: Path) -> None:
    record_id, names = KNOWN_ZENODO[name]
    destination = raw_root / name
    paths = download_zenodo(record_id, destination, names)
    for path in paths:
        if path.suffix.lower() == ".zip":
            extract_zip(path, destination / path.stem)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exactly 100 examples for each GeoMapBench taxonomy leaf.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download a small/medium official Zenodo source.")
    fetch.add_argument("dataset", choices=sorted(KNOWN_ZENODO))
    fetch.add_argument("--raw-root", type=_path, required=True)

    for name in ("maki", "maptext", "openearthmap", "eurosat", "rsvqa", "spacenet7-change", "spacenet7-match", "geowebnews", "spartun", "geoquestions", "spacenet3-graph", "spacenet3-route"):
        p = sub.add_parser(name)
        p.add_argument("--source", type=_path, required=True)
        p.add_argument("--output", type=_path, required=True)

    for name in ("coordinate-transform", "metric-distance", "topology-direction", "geo-entity-typing", "population-density", "cross-entity-comparison", "geology", "visual-geolocation", "osm-label-anchoring", "osm-isochrone"):
        p = sub.add_parser(name)
        p.add_argument("--cache", type=_path, required=True)
        p.add_argument("--output", type=_path, required=True)

    env = sub.add_parser("environmental-layer")
    env.add_argument("--cache", type=_path, required=True)
    env.add_argument(
        "--koppen-raster",
        type=_path,
        required=False,
        help="Deprecated compatibility argument; the revised task uses balanced WorldClim layers.",
    )
    env.add_argument("--output", type=_path, required=True)

    val = sub.add_parser("validate")
    val.add_argument("--root", type=_path, required=True)
    val.add_argument("--require-all", action="store_true")
    val.add_argument("--skip-assets", action="store_true")

    sub.add_parser("seeds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    command = args.command
    if command == "fetch":
        fetch_known(args.dataset, args.raw_root)
        return 0
    if command == "seeds":
        print(json.dumps(SEEDS, indent=2, sort_keys=True))
        return 0
    if command == "validate":
        errors = validate_root(args.root, require_all=args.require_all, require_assets=not args.skip_assets)
        if errors:
            print("\n".join(errors))
            return 1
        print("Validation passed")
        return 0

    source_dispatch = {
        "maki": sample_maki,
        "maptext": sample_maptext,
        "openearthmap": sample_openearthmap,
        "eurosat": sample_eurosat,
        "rsvqa": sample_rsvqa,
        "spacenet7-change": sample_spacenet7_change,
        "spacenet7-match": sample_spacenet7_matching,
        "geowebnews": sample_geowebnews,
        "spartun": sample_spartun,
        "geoquestions": sample_geoquestions,
        "spacenet3-graph": sample_spacenet3_graph,
        "spacenet3-route": sample_spacenet3_shortest_path,
    }
    if command in source_dispatch:
        path = source_dispatch[command](args.source, args.output)
        print(path)
        return 0

    cache_dispatch = {
        "coordinate-transform": generate_coordinate_transform,
        "metric-distance": generate_metric_distance,
        "topology-direction": generate_topology_direction,
        "geo-entity-typing": generate_geo_entity_typing,
        "population-density": generate_population_density,
        "cross-entity-comparison": generate_cross_entity_comparison,
        "geology": generate_geology,
        "visual-geolocation": generate_visual_geolocation,
        "osm-label-anchoring": generate_osm_label_anchoring,
        "osm-isochrone": generate_osm_isochrones,
    }
    if command in cache_dispatch:
        path = cache_dispatch[command](args.cache, args.output)
        print(path)
        return 0
    if command == "environmental-layer":
        print(generate_environmental_layer(args.cache, args.koppen_raster, args.output))
        return 0
    raise AssertionError(command)


if __name__ == "__main__":
    raise SystemExit(main())

