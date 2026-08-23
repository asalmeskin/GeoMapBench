from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from geomaprag_data.clean_data import clean_corpus
from geomaprag_data.common import CorpusWorkspace, make_record, read_jsonl
from geomaprag_data.migrate import migrate_legacy_root
from geomaprag_data.osm import _tile_offsets, render_map_tile


def _record(record_id: str, text: str) -> dict:
    return make_record(
        record_id=record_id,
        source_name="UnitTest",
        source_url="https://example.test/",
        license_name="CC0",
        group_id=record_id,
        modality="text",
        title=record_id,
        text=text,
        source_id=record_id,
        capabilities=["geographic_fact_reasoning"],
        generator="test",
    )


def test_workspace_preserves_legacy_and_resumes_shards(tmp_path: Path) -> None:
    root = tmp_path / "GeoMapRAG_Corpus"
    root.mkdir()
    legacy = {
        "doc_id": "legacy:1",
        "modality": "text",
        "source": "Wikipedia",
        "title": "Legacy",
        "text": "A sufficiently long legacy geographic reference passage for testing purposes.",
        "source_id": "1",
    }
    (root / "corpus.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    workspace = CorpusWorkspace(root)
    workspace.write_shard(
        "wikipedia",
        "unit_a",
        [_record("new:1", "A new geographic reference passage that is long enough for the corpus test.")],
    )
    assert workspace.shard_done("wikipedia", "unit_a")
    manifest = workspace.materialize()
    assert manifest["count"] == 2
    rows = read_jsonl(root / "corpus.jsonl")
    assert {row["id"] for row in rows} == {"legacy:1", "new:1"}

    # A second process sees the same shard and does not need to recreate it.
    workspace2 = CorpusWorkspace(root)
    assert workspace2.shard_done("wikipedia", "unit_a")
    assert workspace2.materialize()["count"] == 2


def test_clean_corpus_keeps_model_fields_and_moves_provenance(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    workspace = CorpusWorkspace(root)
    workspace.write_shard(
        "unit",
        "one",
        [_record("r:1", "This is a compact but sufficiently long retrieval passage for cleaner validation.")],
    )
    workspace.materialize()
    summary = clean_corpus(root, overwrite=True)
    assert summary["record_count"] == 1
    clean = read_jsonl(root / "corpus_clean.jsonl")[0]
    assert set(clean) == {"id", "source", "input", "retrieval"}
    assert clean["id"] == "r:1"
    provenance = read_jsonl(root / "_clean_metadata" / "provenance.jsonl")[0]
    assert provenance["id"] == "r:1"
    assert "provenance" in provenance


def test_migrate_old_root_without_overwrite(tmp_path: Path) -> None:
    old = tmp_path / "GeoMapRAG_Corpus_v1"
    new = tmp_path / "GeoMapRAG_Corpus"
    old.mkdir()
    (old / "corpus.jsonl").write_text('{"doc_id":"x","text":"legacy text"}\n', encoding="utf-8")
    report = migrate_legacy_root(old, new)
    assert report["action"] == "moved_old_root_to_new_root"
    assert not old.exists()
    assert (new / "corpus.jsonl").exists()


def test_tile_offsets_are_deterministic() -> None:
    assert _tile_offsets(6, 700) == _tile_offsets(6, 700)
    assert _tile_offsets(6, 700)[0] == ("center", 0.0, 0.0)
    assert len(_tile_offsets(6, 700)) == 6


def test_osm_renderer_produces_unlabeled_rgb_map(tmp_path: Path) -> None:
    lat, lon = 47.3769, 8.5417
    elements = [
        {
            "type": "way",
            "id": 1,
            "tags": {"highway": "primary", "name": "Synthetic Road"},
            "geometry": [
                {"lat": lat - 0.005, "lon": lon - 0.004},
                {"lat": lat, "lon": lon},
                {"lat": lat + 0.005, "lon": lon + 0.004},
            ],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"highway": "residential", "name": "Synthetic Street"},
            "geometry": [
                {"lat": lat - 0.004, "lon": lon + 0.003},
                {"lat": lat + 0.004, "lon": lon - 0.003},
            ],
        },
        {
            "type": "way",
            "id": 3,
            "tags": {"waterway": "stream", "name": "Synthetic Stream"},
            "geometry": [
                {"lat": lat - 0.004, "lon": lon - 0.002},
                {"lat": lat + 0.004, "lon": lon + 0.002},
            ],
        },
        {
            "type": "node",
            "id": 4,
            "lat": lat,
            "lon": lon + 0.001,
            "tags": {"amenity": "school", "name": "Synthetic POI"},
        },
    ]
    path = tmp_path / "map.png"
    rendered, geo = render_map_tile(elements, lat, lon, 0, 0, 700, path)
    assert rendered
    assert path.exists()
    with Image.open(path) as image:
        assert image.mode in {"RGB", "RGBA"}
        assert image.width > 100 and image.height > 100
    assert abs(geo["lat"] - lat) < 0.01


def test_benchmark_guard_extracts_exact_worldbank_observations(tmp_path: Path) -> None:
    from geomaprag_data.benchmark_guard import build_guard

    benchmark = tmp_path / "benchmark" / "cross_entity_comparison"
    benchmark.mkdir(parents=True)
    row = {
        "id": "x",
        "group_id": "EN.POP.DNST:2023:CHE:DEU",
        "source": {"name": "World Development Indicators: EN.POP.DNST"},
        "input": {"question": "Which country had the larger population density?"},
    }
    (benchmark / "data.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    guard = build_guard(tmp_path / "benchmark", tmp_path / "guard.json")
    assert "EN.POP.DNST:2023:CHE" in guard.worldbank_observations
    assert "EN.POP.DNST:2023:DEU" in guard.worldbank_observations


def test_geonames_balancer_redistributes_sparse_class_quota() -> None:
    from geomaprag_data.geonames import _balanced_selection

    def row(geonameid: str, feature_class: str, country: str) -> dict:
        return {"geonameid": geonameid, "feature_class": feature_class, "country_code": country}

    pools = {
        ("A", "CH"): [row("1", "A", "CH")],
        ("P", "CH"): [row(str(i), "P", "CH") for i in range(2, 15)],
    }
    selected = _balanced_selection(pools, 6)
    assert len(selected) == 6
    assert any(item["feature_class"] == "A" for item in selected)
    assert sum(item["feature_class"] == "P" for item in selected) == 5


def test_embedding_batch_key_is_content_sensitive() -> None:
    from geomaprag_data.index import _batch_key

    first = _batch_key("model", ["id:content_hash_a"])
    second = _batch_key("model", ["id:content_hash_b"])
    assert first != second


def test_epsg_info_deduplication_is_deterministic():
    from types import SimpleNamespace
    from geomaprag_data.epsg import _dedupe_epsg_infos

    infos = [
        SimpleNamespace(auth_name="EPSG", code="3021"),
        SimpleNamespace(auth_name="EPSG", code="2393"),
        SimpleNamespace(auth_name="EPSG", code="3021"),
        SimpleNamespace(auth_name="EPSG", code="3106"),
        SimpleNamespace(auth_name="epsg", code="2393"),
    ]

    result = _dedupe_epsg_infos(infos)
    assert [(str(x.auth_name).upper(), str(x.code)) for x in result] == [
        ("EPSG", "2393"),
        ("EPSG", "3021"),
        ("EPSG", "3106"),
    ]
