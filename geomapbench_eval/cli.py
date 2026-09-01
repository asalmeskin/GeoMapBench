from __future__ import annotations

import argparse
import json

from .analysis import add_analyze_parser, analyze, compare
from .experiments import add_experiment_parsers, run_model_suite, run_rag_suite
from .runner import add_run_parser, run, validate_run_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible OpenRouter evaluation for GeoMapBench.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_run_parser(sub); add_analyze_parser(sub); add_experiment_parsers(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        validate_run_args(args)
        print(json.dumps(run(args), indent=2, sort_keys=True))
    elif args.command == "analyze":
        print(json.dumps(analyze(__import__("pathlib").Path(args.results), __import__("pathlib").Path(args.output), make_plots=not args.no_plots), indent=2, sort_keys=True))
    elif args.command == "compare":
        print(json.dumps(compare(__import__("pathlib").Path(args.base_results), __import__("pathlib").Path(args.rag_results), __import__("pathlib").Path(args.output)), indent=2, sort_keys=True))
    elif args.command == "suite":
        print(json.dumps(run_model_suite(args), indent=2, sort_keys=True))
    elif args.command == "rag-suite":
        print(json.dumps(run_rag_suite(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
