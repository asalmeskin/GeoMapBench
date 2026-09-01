from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, utc_now


def _code_snapshot(code_root: Path | None) -> dict[str, Any] | None:
    if code_root is None:
        return None
    root = Path(code_root).expanduser().resolve()
    if not root.exists():
        return {"root": str(root), "exists": False}

    tracked_files: list[Path] = []
    for relative in (
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("config/geomaprag_sources.csv"),
        Path("schema/geomaprag_entry.schema.json"),
        Path("notebooks/GeoMapRAG_Corpus_NUMBERED.ipynb"),
    ):
        path = root / relative
        if path.is_file():
            tracked_files.append(path)
    package_dir = root / "geomaprag_data"
    if package_dir.is_dir():
        tracked_files.extend(path for path in package_dir.glob("*.py") if path.is_file())

    git_commit = None
    git_dirty = None
    if (root / ".git").exists():
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
                ).strip()
            )
        except Exception:
            pass

    return {
        "root": str(root),
        "exists": True,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "files": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(set(tracked_files))
        },
    }


def freeze_release(root: Path, *, code_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not (root / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Missing corpus.jsonl under {root}")

    include_roots = ["_cache", "_shards", "maps", "images", "indexes", "_clean_metadata", "_state"]
    top_files = ["corpus.jsonl", "corpus_clean.jsonl", "manifest.json", "quality_report.json"]
    files: list[Path] = []
    for name in top_files:
        path = root / name
        if path.is_file():
            files.append(path)
    for name in include_roots:
        directory = root / name
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())

    entries = []
    for path in sorted(set(files)):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    report = {
        "format": "GeoMapRAG frozen release inventory",
        "created_at": utc_now(),
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "code_snapshot": _code_snapshot(code_root),
        "note": (
            "Archive this manifest with the corpus and raw cache used for the paper. "
            "Live upstream services can change after the release date; when --code-root is provided, "
            "the manifest also records the Git commit and hashes of the GeoMapRAG implementation/configuration."
        ),
    }
    atomic_write_json(root / "release_manifest.json", report)
    return report
