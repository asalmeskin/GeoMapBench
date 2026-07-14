import csv
import json
from pathlib import Path

from PIL import Image

from geomapbench_data.samplers import sample_eurosat, sample_geoquestions, sample_maki, sample_maptext
from geomapbench_data.validate import validate_root


def _image(path: Path, color=(20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_offline_samplers_are_exact_and_valid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    maki = raw / "maki" / "icons"
    maki.mkdir(parents=True)
    for i in range(120):
        (maki / f"icon-{i:03d}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"><path d="M0 0h15v15H0z"/></svg>',
            encoding="utf-8",
        )
    sample_maki(raw / "maki", out)

    eurosat = raw / "eurosat"
    for class_index in range(10):
        for image_index in range(11):
            _image(eurosat / f"class_{class_index}" / f"{image_index:03d}.jpg", (class_index * 10, image_index * 10, 80))
    sample_eurosat(eurosat, out)

    maptext = raw / "maptext"
    _image(maptext / "images" / "map.jpg")
    annotation = [
        {
            "image": "map.jpg",
            "groups": [
                {"text": f"Place {i}", "vertices": [[0, 0], [10, 0], [10, 5], [0, 5]], "illegible": False, "truncated": False}
                for i in range(120)
            ],
        }
    ]
    (maptext / "maptext_format.json").write_text(json.dumps(annotation), encoding="utf-8")
    sample_maptext(maptext, out)

    geoquestions = raw / "geoquestions"
    geoquestions.mkdir(parents=True)
    with (geoquestions / "GeoQuestions1089.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "query"])
        writer.writeheader()
        for i in range(120):
            writer.writerow({"question": f"Question {i}?", "answer": f"Answer {i}", "query": f"SELECT {i}"})
    sample_geoquestions(geoquestions, out)

    assert validate_root(out) == []
    for task in out.iterdir():
        if task.is_dir():
            assert sum(1 for _ in (task / "data.jsonl").open(encoding="utf-8")) == 100

