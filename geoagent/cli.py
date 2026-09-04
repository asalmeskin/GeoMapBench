from __future__ import annotations

import argparse
import json

from .suite import add_geoagent_parser, run_geoagent_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tool-augmented, self-verifying agentic RAG for GeoMapBench.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    add_geoagent_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "suite":
        print(json.dumps(run_geoagent_suite(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
