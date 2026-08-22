import json
from pathlib import Path

from PIL import Image

from geomapbench_data.bloom import (
    BLOOM_LEVELS,
    BLOOM_REVISION,
    bloom_distribution,
    bloomify_root,
    restore_bloom_root,
)
from geomapbench_data.common import SEEDS, sha256_file, write_jsonl
from geomapbench_data.validate import validate_task


def _manifest(task: Path, leaf: str) -> None:
    payload = {
        "leaf": leaf,
        "count": 100,
        "seed": SEEDS[leaf],
        "data_revision": "2026-07-comments-v2",
        "created_at": "2026-08-22T00:00:00+00:00",
        "data_file": "data.jsonl",
        "sha256": sha256_file(task / "data.jsonl"),
    }
    (task / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_bloom_distributions_are_exact() -> None:
    for leaf, levels in BLOOM_LEVELS.items():
        distribution = bloom_distribution(levels, 100)
        assert sum(distribution.values()) == 100
        if len(levels) == 5:
            assert set(distribution.values()) == {20}
        else:
            assert list(distribution.values()) == [17, 17, 17, 17, 16, 16]


def test_bloomify_and_restore_maki_without_touching_assets(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    task = root / "cartographic_symbol_recognition"
    assets = task / "assets"
    assets.mkdir(parents=True)
    icon = assets / "icon.svg"
    icon.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"></svg>', encoding="utf-8")

    choices = ["cafe", "hospital", "park", "school"]
    records = []
    for i in range(100):
        answer = choices[i % len(choices)]
        records.append(
            {
                "id": f"cartographic_symbol_recognition-{i:03d}",
                "leaf": "cartographic_symbol_recognition",
                "seed": SEEDS["cartographic_symbol_recognition"],
                "group_id": f"g{i}",
                "source": {"name": "mock", "url": "https://example.com", "license": "CC0"},
                "input": {"images": ["assets/icon.svg"], "question": "base", "choices": choices},
                "target": {"answer": answer, "choice_index": choices.index(answer)},
                "evaluation": {"type": "classification", "metric": "accuracy"},
            }
        )
    write_jsonl(task / "data.jsonl", records)
    _manifest(task, "cartographic_symbol_recognition")
    original = (task / "data.jsonl").read_bytes()
    original_asset = icon.read_bytes()

    report = bloomify_root(root, require_all=False)
    assert report["valid"] is True
    assert report["leaves"]["cartographic_symbol_recognition"]["distribution"] == {
        "R": 20,
        "U": 20,
        "Ap": 20,
        "An": 20,
        "E": 20,
    }
    manifest = json.loads((task / "manifest.json").read_text())
    assert manifest["bloom_revision"] == BLOOM_REVISION
    assert icon.read_bytes() == original_asset
    assert validate_task(task) == []

    restored = restore_bloom_root(root, require_all=False)
    assert restored["count"] == 1
    assert (task / "data.jsonl").read_bytes() == original
    assert icon.read_bytes() == original_asset


def test_dense_landcover_bloom_uses_existing_mask(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    task = root / "dense_land_cover_labeling"
    assets = task / "assets"
    assets.mkdir(parents=True)
    ontology = {
        str(i): {"name": name, "rgb": [i, i, i], "hex": "#000000"}
        for i, name in enumerate(
            ["bareland", "rangeland", "developed space", "road", "tree", "water", "agriculture land", "building"]
        )
    }
    records = []
    for i in range(100):
        image_path = assets / f"{i:03d}_image.png"
        mask_path = assets / f"{i:03d}_mask.png"
        Image.new("RGB", (8, 8), (30, 60, 90)).save(image_path)
        Image.new("L", (8, 8), i % 8).save(mask_path)
        records.append(
            {
                "id": f"dense_land_cover_labeling-{i:03d}",
                "leaf": "dense_land_cover_labeling",
                "seed": SEEDS["dense_land_cover_labeling"],
                "group_id": f"g{i}",
                "source": {"name": "mock", "url": "https://example.com", "license": "test"},
                "input": {
                    "images": [f"assets/{i:03d}_image.png"],
                    "question": "base",
                    "class_ontology": ontology,
                    "classes": [v["name"] for v in ontology.values()],
                },
                "target": {
                    "mask": f"assets/{i:03d}_mask.png",
                    "class_ontology": ontology,
                    "ignore_index": 255,
                    "pixel_counts": {str(i % 8): 64},
                },
                "evaluation": {"type": "semantic_segmentation", "metrics": ["mean_iou"]},
            }
        )
    write_jsonl(task / "data.jsonl", records)
    _manifest(task, "dense_land_cover_labeling")

    report = bloomify_root(root, require_all=False)
    assert report["valid"] is True
    assert validate_task(task) == []
    converted = [json.loads(line) for line in (task / "data.jsonl").read_text().splitlines()]
    assert {r["bloom"]["level"] for r in converted} == {"R", "U", "Ap", "An", "E", "C"}
    assert all("bloom_answer" in r["target"] for r in converted)
