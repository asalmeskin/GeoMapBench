from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file


CLEAN_FILENAME = "corpus_clean.jsonl"
METADATA_DIRNAME = "_clean_metadata"


def clean_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = record.get("source") or {}
    input_obj = copy.deepcopy(record.get("input") or {})
    retrieval = copy.deepcopy(record.get("retrieval") or {})

    clean = {
        "id": record["id"],
        "source": source.get("name", "Unknown") if isinstance(source, dict) else str(source),
        "input": input_obj,
        "retrieval": {
            "document_type": retrieval.get("document_type", "reference"),
            "capabilities": retrieval.get("capabilities", []),
        },
    }

    provenance = {
        "id": record["id"],
        "dataset": record.get("dataset"),
        "data_revision": record.get("data_revision"),
        "group_id": record.get("group_id"),
        "source": copy.deepcopy(source),
        "provenance": copy.deepcopy(record.get("provenance")),
    }
    if "extra" in record:
        provenance["extra"] = copy.deepcopy(record["extra"])
    return clean, provenance


def clean_corpus(root: Path, *, overwrite: bool = False, write_provenance: bool = True) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    source_path = root / "corpus.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing full corpus: {source_path}")
    clean_path = root / CLEAN_FILENAME
    metadata_root = root / METADATA_DIRNAME
    provenance_path = metadata_root / "provenance.jsonl"
    summary_path = metadata_root / "summary.json"

    if clean_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {clean_path}; pass --overwrite")
    if write_provenance and provenance_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {provenance_path}; pass --overwrite")

    records = read_jsonl(source_path)
    clean_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"corpus.jsonl record {index}: missing id")
        if record_id in seen:
            raise ValueError(f"corpus.jsonl record {index}: duplicate id {record_id}")
        seen.add(record_id)
        input_obj = record.get("input")
        if not isinstance(input_obj, dict) or not str(input_obj.get("text") or "").strip():
            raise ValueError(f"{record_id}: missing input.text")
        clean, provenance = clean_record(record)
        clean_rows.append(clean)
        provenance_rows.append(provenance)

    atomic_write_jsonl(clean_path, clean_rows)
    if write_provenance:
        atomic_write_jsonl(provenance_path, provenance_rows)

    # Verify an exact ID/order correspondence with the full corpus.
    reread = read_jsonl(clean_path)
    if [r["id"] for r in reread] != [r["id"] for r in records]:
        raise RuntimeError("Clean corpus changed record identity/order")

    original_bytes = source_path.stat().st_size
    clean_bytes = clean_path.stat().st_size
    summary = {
        "format": "GeoMapRAG model-facing clean view",
        "record_count": len(clean_rows),
        "source_file": source_path.name,
        "clean_file": clean_path.name,
        "original_data_modified": False,
        "clean_sha256": sha256_file(clean_path),
        "original_bytes": original_bytes,
        "clean_bytes": clean_bytes,
        "reduction_percent": round(100.0 * (1.0 - clean_bytes / original_bytes), 2) if original_bytes else 0.0,
        "policy": {
            "kept": ["id", "source.name", "input", "retrieval.document_type", "retrieval.capabilities"],
            "moved_to_provenance_sidecar": [
                "dataset",
                "data_revision",
                "group_id",
                "source.url/license/attribution",
                "provenance",
                "extra",
            ],
        },
        "provenance_file": str(provenance_path.relative_to(root)) if write_provenance else None,
    }
    atomic_write_json(summary_path, summary)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a compact model-facing GeoMapRAG corpus without touching corpus.jsonl.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-provenance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        summary = clean_corpus(args.root, overwrite=args.overwrite, write_provenance=not args.no_provenance)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
