from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_all
from .clean_data import clean_corpus
from .common import CorpusWorkspace
from .config import PROFILES
from .index import build_image_index, build_text_index
from .freeze import freeze_release
from .migrate import migrate_legacy_root
from .validate import validate_corpus


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and maintain the large, resume-safe GeoMapRAG retrieval corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build/extend the corpus. Safe to rerun after interruption.")
    build.add_argument("--output", type=_path, required=True)
    build.add_argument("--benchmark-root", type=_path, required=False)
    build.add_argument("--profile", choices=sorted(PROFILES), default="iclr")
    build.add_argument(
        "--stages",
        default="",
        help="Optional comma-separated subset: epsg,geonames,worldbank,wikipedia,wikidata,wikimedia,osm",
    )
    build.add_argument("--spatial-exclusion-km", type=float, default=2.0)

    migrate = sub.add_parser("migrate", help="Migrate GeoMapRAG_Corpus_v1 into GeoMapRAG_Corpus non-destructively.")
    migrate.add_argument("--old-root", type=_path, required=True)
    migrate.add_argument("--new-root", type=_path, required=True)

    materialize = sub.add_parser("materialize", help="Rebuild corpus.jsonl from legacy snapshots and completed shards.")
    materialize.add_argument("--root", type=_path, required=True)

    clean = sub.add_parser("clean", help="Create corpus_clean.jsonl and provenance sidecar.")
    clean.add_argument("--root", type=_path, required=True)
    clean.add_argument("--overwrite", action="store_true")
    clean.add_argument("--no-provenance", action="store_true")

    validate = sub.add_parser("validate", help="Validate schema, duplicates, assets, and optional scale targets.")
    validate.add_argument("--root", type=_path, required=True)
    validate.add_argument("--profile", choices=sorted(PROFILES), required=False)
    validate.add_argument("--strict-scale", action="store_true")

    status = sub.add_parser("status", help="Show incremental shard/corpus status.")
    status.add_argument("--root", type=_path, required=True)

    freeze = sub.add_parser("freeze", help="Hash data and optionally the exact code snapshot into a paper-release manifest.")
    freeze.add_argument("--root", type=_path, required=True)
    freeze.add_argument("--code-root", type=_path, required=False, help="Optional checked-out repository root; records Git commit and code/config hashes.")

    text = sub.add_parser("index-text", help="Build a FAISS text index with resumable embedding-batch cache.")
    text.add_argument("--root", type=_path, required=True)
    text.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    text.add_argument("--batch-size", type=int, default=64)

    image = sub.add_parser("index-image", help="Build a CLIP/FAISS image index (maps + geocoded photos) with embedding cache.")
    image.add_argument("--root", type=_path, required=True)
    image.add_argument("--model", default="openai/clip-vit-base-patch32")
    image.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "migrate":
        print(json.dumps(migrate_legacy_root(args.old_root, args.new_root), indent=2, sort_keys=True))
        return 0
    if args.command == "build":
        stages = {x.strip() for x in args.stages.split(",") if x.strip()} or None
        report = build_all(
            args.output,
            benchmark_root=args.benchmark_root,
            profile_name=args.profile,
            stages=stages,
            spatial_exclusion_km=args.spatial_exclusion_km,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if report.get("failed_unit_count", 0):
            print("\nPARTIAL BUILD: rerun the identical command; completed shards/cache will be reused.")
            return 2
        return 0
    if args.command == "materialize":
        print(json.dumps(CorpusWorkspace(args.root).materialize(), indent=2, sort_keys=True))
        return 0
    if args.command == "clean":
        summary = clean_corpus(
            args.root,
            overwrite=args.overwrite,
            write_provenance=not args.no_provenance,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        report = validate_corpus(args.root, profile=args.profile, strict_scale=args.strict_scale)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("valid") else 1
    if args.command == "status":
        print(json.dumps(CorpusWorkspace(args.root).status(), indent=2, sort_keys=True))
        return 0
    if args.command == "freeze":
        report = freeze_release(args.root, code_root=args.code_root)
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2, sort_keys=True))
        print(f"Release inventory written: {args.root / 'release_manifest.json'}")
        return 0
    if args.command == "index-text":
        print(json.dumps(build_text_index(args.root, model_name=args.model, batch_size=args.batch_size), indent=2, sort_keys=True))
        return 0
    if args.command == "index-image":
        print(json.dumps(build_image_index(args.root, model_name=args.model, batch_size=args.batch_size), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
