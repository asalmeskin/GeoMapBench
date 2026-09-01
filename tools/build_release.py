from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent.parent / "GeoMapBench-v1.7.1-final.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> list[Path]:
    allowed_top_level = {
        ".gitignore", "FINAL_RUN_GUIDE.md", "README.md", "pyproject.toml", "requirements.txt",
        "config", "geomapbench_data", "geomapbench_eval", "geomaprag_data",
        "notebooks", "schema", "tests", "tools",
    }
    blocked_parts = {
        "__pycache__", ".pytest_cache", "geomapbench_data_kit.egg-info",
        "build", "dist", "dist_check", "dist_check_v17", ".git",
    }
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).parts[0] in allowed_top_level
        and path.name != "RELEASE_MANIFEST.json"
        and path.suffix not in {".pyc", ".pyo"}
        and not (set(path.parts) & blocked_parts)
    )


def build() -> Path:
    files = included_files()
    manifest = {
        "release": "GeoMapBench 1.7.1 final",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "file_count_excluding_manifest": len(files),
        "notebooks": [
            "notebooks/GeoMapBench_Simple_Evaluation.ipynb",
            "notebooks/GeoMapBench_Final_RAG.ipynb",
        ],
        "files": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files},
    }
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = included_files() + [manifest_path]
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, (Path("GeoMapBench-final") / path.relative_to(ROOT)).as_posix())
    return OUTPUT


if __name__ == "__main__":
    print(build())
