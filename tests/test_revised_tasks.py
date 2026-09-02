import json
from pathlib import Path

import numpy as np
from PIL import Image

from geomapbench_data.samplers import OPENEARTHMAP_CLASSES, _annotation_flag, sample_openearthmap
from geomapbench_data.static_generators import _coordinate_case
from geomapbench_data.validate import validate_task


def test_coordinate_cases_cover_six_reversible_modes() -> None:
    modes = [
        "geographic_to_utm",
        "utm_to_geographic",
        "geographic_to_web_mercator",
        "web_mercator_to_geographic",
        "utm_to_web_mercator",
        "web_mercator_to_utm",
    ]
    pairs = set()
    for mode in modes:
        case = _coordinate_case(8.5417, 47.3769, mode)
        pairs.add((case["source_crs"], case["target_crs"]))
        assert case["roundtrip_error"] < 1e-5
        assert case["source_crs"] != case["target_crs"]
    assert len(pairs) == 6


def test_string_annotation_flags_are_parsed_correctly() -> None:
    assert _annotation_flag("false") is False
    assert _annotation_flag("true") is True


def test_openearthmap_rgb_masks_are_normalized(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "out"
    palette = np.asarray([item["rgb"] for item in OPENEARTHMAP_CLASSES], dtype=np.uint8)
    for index in range(100):
        region = source / f"region_{index % 10}"
        images = region / "images"
        labels = region / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (30, 60, 90)).save(images / f"tile_{index:03d}.tif")
        ids = np.fromfunction(lambda y, x: (x + y + index) % 8, (16, 16), dtype=int).astype(np.uint8)
        Image.fromarray(palette[ids], mode="RGB").save(labels / f"tile_{index:03d}.tif")

    sample_openearthmap(source, output)
    task = output / "dense_land_cover_labeling"
    assert validate_task(task) == []
    records = [json.loads(line) for line in (task / "data.jsonl").read_text().splitlines()]
    with Image.open(task / records[0]["target"]["mask"]) as mask:
        assert mask.mode == "L"
        assert set(np.unique(np.asarray(mask))).issubset(set(range(8)) | {255})


def test_maptext_flat_sequence_words_stay_in_one_group(tmp_path: Path) -> None:
    from geomapbench_data.samplers import _maptext_candidates, _maptext_sequences

    first_record = {
        "image": "map.png",
        "groups": [
            {
                "text": "Champs",
                "vertices": [[10, 10], [60, 10], [60, 30], [10, 30]],
                "illegible": "false",
                "truncated": "false",
            },
            {
                "text": "Elysées",
                "vertices": [[65, 10], [125, 10], [125, 30], [65, 30]],
                "illegible": "false",
                "truncated": "false",
            },
        ],
    }
    sequences = _maptext_sequences(first_record)
    assert len(sequences) == 1
    assert sequences[0]["text"] == "Champs Elysées"
    assert len(sequences[0]["words"]) == 2

    image_path = tmp_path / "map.png"
    Image.new("RGB", (200, 200), "white").save(image_path)
    lookup = {"map.png": image_path, "map": image_path}
    data = [
        first_record,
        {
            "image": "map.png",
            "groups": [
                {
                    "text": "Paris",
                    "vertices": [[20, 80], [80, 80], [80, 100], [20, 100]],
                    "illegible": "false",
                    "truncated": "false",
                }
            ],
        },
    ]
    candidates = _maptext_candidates(data, lookup)
    assert len(candidates) == 2
    aggregated = candidates[0]["groups"]
    assert len(aggregated) == 2
    assert aggregated[0]["text"] == "Champs Elysées"
    assert [word["text"] for word in aggregated[0]["words"]] == ["Champs", "Elysées"]


def test_distance_units_are_distinct_and_reversible() -> None:
    from geomapbench_data.static_generators import DISTANCE_UNITS, _convert_distance_from_metres

    metres = 1852.0
    assert set(DISTANCE_UNITS) == {"metres", "kilometres", "miles", "nautical_miles"}
    assert _convert_distance_from_metres(metres, "metres") == 1852.0
    assert _convert_distance_from_metres(metres, "kilometres") == 1.852
    assert _convert_distance_from_metres(metres, "nautical_miles") == 1.0
    assert _convert_distance_from_metres(1609.344, "miles") == 1.0


def test_population_density_uses_multiple_years(tmp_path: Path, monkeypatch) -> None:
    import geomapbench_data.api_generators as module

    countries = {
        f"C{i:03d}": {"name": f"Country {i}"}
        for i in range(120)
    }

    def fake_countries(cache: Path):
        return countries

    def fake_indicator(cache: Path, indicator: str, year: int = module.WORLD_BANK_YEAR):
        return {
            code: {
                "value": float(index + year / 1000),
                "countryiso3code": code,
            }
            for index, code in enumerate(countries)
        }

    monkeypatch.setattr(module, "_world_bank_countries", fake_countries)
    monkeypatch.setattr(module, "_world_bank_indicator", fake_indicator)
    module.generate_population_density(tmp_path / "cache", tmp_path / "out")
    task = tmp_path / "out" / "population_density_estimation"
    assert validate_task(task) == []
    records = [json.loads(line) for line in (task / "data.jsonl").read_text().splitlines()]
    assert len({record["target"]["year"] for record in records}) == len(module.POPULATION_DENSITY_YEARS)


def test_place_area_polygon_is_visible_and_within_country() -> None:
    from shapely.geometry import Point, box

    from geomapbench_data.static_generators import _place_area_polygon

    country = box(7.0, 46.0, 10.0, 48.0)
    area = _place_area_polygon(Point(8.5, 47.0), country)
    assert area is not None
    assert area.geom_type == "Polygon"
    assert area.within(country)
    assert area.area > 0


def test_spacenet_high_dynamic_range_tiff_becomes_visible_rgb(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    from geomapbench_data.network_generators import render_spacenet_display

    source = tmp_path / "source.tif"
    destination = tmp_path / "display.png"
    y, x = np.mgrid[0:64, 0:64]
    data = np.stack(
        [
            1000 + x * 80 + y * 10,
            500 + x * 30 + y * 70,
            1500 + x * 15 + y * 45,
        ]
    ).astype(np.uint16)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(8.0, 48.0, 0.001, 0.001),
    ) as dataset:
        dataset.write(data)

    render_spacenet_display(source, destination)
    with Image.open(destination) as image:
        array = np.asarray(image)
        assert image.mode == "RGB"
        assert array.dtype == np.uint8
        assert float(array.mean()) > 20
        assert float(np.percentile(array, 99) - np.percentile(array, 1)) > 50
