from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


CORPUS_REVISION = "2026-08-geomaprag-iclr-v2"
DATASET_NAME = "GeoMapRAG"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    _atomic_write(path, _write)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    materialized = list(records)

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in materialized:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    _atomic_write(path, _write)


def read_jsonl(path: Path, tolerate_trailing_partial: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            if tolerate_trailing_partial and line_no == len(lines):
                continue
            raise
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(obj)
    return rows


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "item"


def chunk_text(text: str, target_words: int = 420, overlap_words: int = 55) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for sentence in sentences:
        count = len(sentence.split())
        if current and word_count + count > target_words:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_count = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_count += len(previous.split())
                if overlap_count >= overlap_words:
                    break
            current = overlap
            word_count = sum(len(s.split()) for s in current)
        current.append(sentence)
        word_count += count
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk) >= 160]


def make_record(
    *,
    record_id: str,
    source_name: str,
    source_url: str,
    license_name: str,
    group_id: str,
    modality: str,
    title: str,
    text: str,
    source_id: str | None = None,
    geo: dict[str, Any] | None = None,
    media_paths: list[str] | None = None,
    capabilities: Iterable[str] = (),
    document_type: str = "reference",
    attribution: str | None = None,
    generator: str | None = None,
    retrieved_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_obj: dict[str, Any] = {
        "modality": modality,
        "title": str(title),
        "text": str(text),
    }
    if geo:
        input_obj["geo"] = copy.deepcopy(geo)
    if media_paths:
        input_obj["images"] = list(media_paths)

    source: dict[str, Any] = {
        "name": source_name,
        "url": source_url,
        "license": license_name,
    }
    if attribution:
        source["attribution"] = attribution

    record: dict[str, Any] = {
        "id": record_id,
        "dataset": DATASET_NAME,
        "data_revision": CORPUS_REVISION,
        "group_id": str(group_id),
        "source": source,
        "input": input_obj,
        "retrieval": {
            "document_type": document_type,
            "capabilities": sorted(set(str(x) for x in capabilities)),
        },
        "provenance": {
            "source_id": None if source_id is None else str(source_id),
            "retrieved_at": retrieved_at or utc_now(),
            "generator": generator or "geomaprag_data",
        },
    }
    if extra:
        record["extra"] = copy.deepcopy(extra)
    return record


def normalize_legacy_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert the v1 notebook schema into the richer v2 corpus schema."""
    if "id" in row and isinstance(row.get("source"), dict) and isinstance(row.get("input"), dict):
        return row

    doc_id = str(row.get("doc_id") or row.get("id") or "")
    if not doc_id:
        raise ValueError("Legacy GeoMapRAG record lacks doc_id/id")
    source_name = row.get("source")
    if isinstance(source_name, dict):
        source_name = source_name.get("name") or "Unknown"
    source_name = str(source_name or "Unknown")
    media = row.get("media_path")
    media_paths = [str(media)] if media else None
    capabilities = row.get("capabilities") or []
    extra = copy.deepcopy(row.get("extra") or {})
    extra["legacy_record"] = {k: copy.deepcopy(v) for k, v in row.items() if k not in {"text", "title"}}

    return make_record(
        record_id=doc_id,
        source_name=source_name,
        source_url=str(row.get("uri") or ""),
        license_name=str(row.get("license") or "unspecified; see upstream source"),
        group_id=str(row.get("source_id") or doc_id),
        modality=str(row.get("modality") or "text"),
        title=str(row.get("title") or doc_id),
        text=str(row.get("text") or ""),
        source_id=None if row.get("source_id") is None else str(row.get("source_id")),
        geo=copy.deepcopy(row.get("geo")) if isinstance(row.get("geo"), dict) else None,
        media_paths=media_paths,
        capabilities=capabilities,
        document_type="legacy_v1_reference",
        generator="legacy_notebook_v1",
        extra=extra,
    )


class CorpusWorkspace:
    """Resume-safe GeoMapRAG workspace.

    The canonical incremental state is a set of atomic source/unit shards plus
    immutable legacy bootstrap files. ``corpus.jsonl`` is a materialized view.
    If a run dies after a shard is written, rerunning skips that shard and
    continues at the next unfinished unit.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.root / "_cache"
        self.shard_dir = self.root / "_shards"
        self.legacy_dir = self.root / "_legacy"
        self.state_dir = self.root / "_state"
        self.maps_dir = self.root / "maps"
        for path in (self.cache_dir, self.shard_dir, self.legacy_dir, self.state_dir, self.maps_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._bootstrap_current_corpus_once()

    @property
    def corpus_path(self) -> Path:
        return self.root / "corpus.jsonl"

    def _bootstrap_current_corpus_once(self) -> None:
        marker = self.state_dir / "bootstrap_complete.json"
        if marker.exists():
            return
        imported: list[str] = []
        if self.corpus_path.exists() and self.corpus_path.stat().st_size:
            digest = sha256_file(self.corpus_path)
            destination = self.legacy_dir / f"bootstrap_{digest[:12]}.jsonl"
            if not destination.exists():
                shutil.copy2(self.corpus_path, destination)
            imported.append(destination.name)
        atomic_write_json(
            marker,
            {
                "created_at": utc_now(),
                "imported": imported,
                "note": "Existing corpus snapshot captured before incremental v2 shards were added.",
            },
        )

    def shard_path(self, source: str, unit: str) -> Path:
        return self.shard_dir / slugify(source) / f"{slugify(unit)}.jsonl"

    def shard_meta_path(self, source: str, unit: str) -> Path:
        return self.shard_dir / slugify(source) / f"{slugify(unit)}.meta.json"

    def shard_done(self, source: str, unit: str) -> bool:
        return self.shard_path(source, unit).exists() and self.shard_meta_path(source, unit).exists()

    def write_shard(
        self,
        source: str,
        unit: str,
        records: Iterable[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> Path:
        rows = list(records)
        ids = [str(r.get("id") or "") for r in rows]
        if any(not value for value in ids):
            raise ValueError(f"{source}/{unit}: shard record without id")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{source}/{unit}: duplicate ids inside shard")
        path = self.shard_path(source, unit)
        atomic_write_jsonl(path, rows)
        payload = {
            "source": source,
            "unit": unit,
            "count": len(rows),
            "sha256": sha256_file(path),
            "completed_at": utc_now(),
            "data_revision": CORPUS_REVISION,
        }
        if meta:
            payload.update(meta)
        atomic_write_json(self.shard_meta_path(source, unit), payload)
        return path

    def iter_input_files(self) -> Iterator[Path]:
        for path in sorted(self.legacy_dir.glob("*.jsonl")):
            yield path
        for path in sorted(self.shard_dir.glob("*/*.jsonl")):
            yield path

    def materialize(self) -> dict[str, Any]:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        records: list[dict[str, Any]] = []
        duplicate_ids = 0
        duplicate_text = 0
        for path in self.iter_input_files():
            for raw in read_jsonl(path, tolerate_trailing_partial=True):
                record = normalize_legacy_record(raw)
                record_id = str(record["id"])
                text = str(record.get("input", {}).get("text", ""))
                digest = text_hash(text)
                if record_id in seen_ids:
                    duplicate_ids += 1
                    continue
                if text and digest in seen_text:
                    duplicate_text += 1
                    continue
                seen_ids.add(record_id)
                if text:
                    seen_text.add(digest)
                records.append(record)

        atomic_write_jsonl(self.corpus_path, records)
        sources = Counter(str(r.get("source", {}).get("name", "Unknown")) for r in records)
        modalities = Counter(str(r.get("input", {}).get("modality", "unknown")) for r in records)
        manifest = {
            "dataset": DATASET_NAME,
            "data_revision": CORPUS_REVISION,
            "created_at": utc_now(),
            "count": len(records),
            "corpus_file": self.corpus_path.name,
            "sha256": sha256_file(self.corpus_path),
            "sources": dict(sorted(sources.items())),
            "modalities": dict(sorted(modalities.items())),
            "deduplication": {
                "duplicate_ids_skipped": duplicate_ids,
                "duplicate_texts_skipped": duplicate_text,
            },
            "incremental_state": {
                "legacy_files": len(list(self.legacy_dir.glob("*.jsonl"))),
                "completed_shards": len(list(self.shard_dir.glob("*/*.jsonl"))),
            },
        }
        atomic_write_json(self.root / "manifest.json", manifest)
        return manifest

    def existing_ids(self) -> set[str]:
        ids: set[str] = set()
        for path in self.iter_input_files():
            for row in read_jsonl(path, tolerate_trailing_partial=True):
                try:
                    ids.add(str(normalize_legacy_record(row)["id"]))
                except Exception:
                    continue
        return ids

    def status(self) -> dict[str, Any]:
        source_shards = Counter(path.parent.name for path in self.shard_dir.glob("*/*.jsonl"))
        return {
            "root": str(self.root),
            "corpus_exists": self.corpus_path.exists(),
            "corpus_sha256": sha256_file(self.corpus_path) if self.corpus_path.exists() else None,
            "legacy_files": [p.name for p in sorted(self.legacy_dir.glob("*.jsonl"))],
            "completed_shards": dict(sorted(source_shards.items())),
        }
