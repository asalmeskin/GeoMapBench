from __future__ import annotations

from pathlib import Path

import geomapbench_data.validate as validate_module
from geomapbench_data.common import SEEDS
from geomapbench_data.validate import REVISED_TASKS, validate_root


def test_legacy_isochrone_not_in_revised_tasks() -> None:
    assert "isochrone_service_area" not in REVISED_TASKS


def test_validate_root_ignores_noncanonical_task_like_directories(tmp_path: Path, monkeypatch) -> None:
    canonical = next(iter(SEEDS))
    canonical_dir = tmp_path / canonical
    canonical_dir.mkdir()
    (canonical_dir / "data.jsonl").write_text("{}\n", encoding="utf-8")

    duplicate = tmp_path / f"{canonical} (1)"
    duplicate.mkdir()
    (duplicate / "data.jsonl").write_text("{}\n", encoding="utf-8")

    visited: list[str] = []

    def fake_validate_task(task_dir: Path, require_assets: bool = True) -> list[str]:
        visited.append(task_dir.name)
        return []

    monkeypatch.setattr(validate_module, "validate_task", fake_validate_task)
    errors = validate_root(tmp_path, require_all=False)

    assert errors == []
    assert visited == [canonical]


def test_validate_root_reports_missing_canonical_leaves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "validate_task", lambda *args, **kwargs: [])
    errors = validate_root(tmp_path, require_all=True)
    assert len(errors) == 1
    assert errors[0].startswith("Missing tasks: ")
    for leaf in SEEDS:
        assert leaf in errors[0]


def test_validate_root_reports_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert validate_root(missing) == [f"Dataset root does not exist: {missing}"]


def test_validate_root_reports_file_instead_of_directory(tmp_path: Path) -> None:
    path = tmp_path / "dataset"
    path.write_text("not a directory", encoding="utf-8")
    assert validate_root(path) == [f"Dataset root is not a directory: {path}"]
