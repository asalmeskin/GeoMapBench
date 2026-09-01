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
from .clean_data import main as clean_data_main
from .bloom import BLOOM_LEVELS, bloom_audit_root, bloomify_root, restore_bloom_root
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

    clean = sub.add_parser("clean", help="Create model-facing data_clean.jsonl sidecars without modifying benchmark originals.")
    clean.add_argument("--root", type=_path, required=True)
    clean.add_argument("--overwrite", action="store_true")
    clean.add_argument("--no-provenance", action="store_true")
    clean.add_argument("--dry-run", action="store_true")
    clean.add_argument("--allow-non100", action="store_true")

    bloomify = sub.add_parser("bloomify", help="Convert an existing 23-leaf GeoMapBench release into a balanced Bloom variant in place without changing assets.")
    bloomify.add_argument("--root", type=_path, required=True)
    bloomify.add_argument("--no-backup", action="store_true", help="Do not save the original data.jsonl/manifest.json under each leaf's .pre_bloom directory.")
    bloomify.add_argument("--force", action="store_true", help="Rebuild Bloom metadata/questions even if the same Bloom revision is already present.")
    bloomify.add_argument("--allow-partial", action="store_true", help="Allow conversion of a root that contains fewer than all 23 leaves.")

    bloom_audit = sub.add_parser("bloom-audit", help="Audit Bloom level coverage and per-leaf balance.")
    bloom_audit.add_argument("--root", type=_path, required=True)
    bloom_audit.add_argument("--allow-partial", action="store_true")

    restore_bloom = sub.add_parser("bloom-restore", help="Restore pre-Bloom data.jsonl and manifest.json backups.")
    restore_bloom.add_argument("--root", type=_path, required=True)
    restore_bloom.add_argument("--allow-partial", action="store_true")
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
    if command == "clean":
        clean_argv = ["--root", str(args.root)]
        if args.overwrite:
            clean_argv.append("--overwrite")
        if args.no_provenance:
            clean_argv.append("--no-provenance")
        if args.dry_run:
            clean_argv.append("--dry-run")
        if args.allow_non100:
            clean_argv.append("--allow-non100")
        return clean_data_main(clean_argv)
    if command == "validate":
        errors = validate_root(args.root, require_all=args.require_all, require_assets=not args.skip_assets)
        if errors:
            print("\n".join(errors))
            return 1
        print("Validation passed")
        return 0


    if command == "bloomify":
        report = bloomify_root(
            args.root,
            backup=not args.no_backup,
            force=args.force,
            require_all=not args.allow_partial,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("valid") else 1
    if command == "bloom-audit":
        report = bloom_audit_root(args.root, require_all=not args.allow_partial)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("valid") else 1
    if command == "bloom-restore":
        print(json.dumps(restore_bloom_root(args.root, require_all=not args.allow_partial), indent=2, sort_keys=True))
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

