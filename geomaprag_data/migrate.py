from __future__ import annotations

import shutil
from pathlib import Path

from .common import sha256_file


def _copy_tree_missing(source: Path, destination: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def migrate_legacy_root(old_root: Path, new_root: Path) -> dict[str, object]:
    """Non-destructively migrate ``GeoMapRAG_Corpus_v1`` to the new root.

    If the new root does not exist, the Drive directory is moved as a unit. If
    both roots exist, old cache/maps are copied only when missing and the old
    corpus is imported as an immutable legacy JSONL sidecar so materialization
    can deduplicate it safely.
    """

    old_root = Path(old_root).expanduser().resolve()
    new_root = Path(new_root).expanduser().resolve()
    report: dict[str, object] = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "action": "none",
        "copied_files": 0,
    }
    if old_root == new_root or not old_root.exists():
        return report

    if not new_root.exists():
        new_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_root), str(new_root))
        report["action"] = "moved_old_root_to_new_root"
        return report

    new_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied += _copy_tree_missing(old_root / "_cache", new_root / "_cache")
    copied += _copy_tree_missing(old_root / "maps", new_root / "maps")
    legacy_dir = new_root / "_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    old_corpus = old_root / "corpus.jsonl"
    if old_corpus.exists() and old_corpus.stat().st_size:
        digest = sha256_file(old_corpus)
        destination = legacy_dir / f"imported_v1_{digest[:12]}.jsonl"
        if not destination.exists():
            shutil.copy2(old_corpus, destination)
            copied += 1
    report["action"] = "merged_non_destructively"
    report["copied_files"] = copied
    return report
