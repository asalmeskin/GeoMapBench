from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from geoagent import repair as R
from geoagent import tools as T
from geoagent.agents import CachedAgent, analyse
from geoagent.corpus_index import StructuredCorpusIndex, normalize_name
from geoagent.driver import RunConfig, run_agentic
from geoagent.pipeline import AgenticPipeline
from geoagent.prompting import build_agent_messages
from geoagent.retrieval import HybridMultimodalRetriever
from geoagent.taskview import TaskView, assert_prompt_contract_matches
from geoagent.suite import _already_scored_by_current_code
from geoagent.validate import validate_runtime
from geomapbench_eval.common import atomic_json, read_jsonl
from geomapbench_eval.cumulative import write_cohort_manifest
from geomapbench_eval.protocol import protocol_descriptor


def _view(leaf, question, payload, choices=None, evaluation_type="exact_match"):
    record = {
        "id": "probe", "leaf": leaf,
        "input": {"question": question, **payload, **({"choices": choices} if choices else {})},
        "evaluation": {"type": evaluation_type},
        "target": {"answer": "SECRET"}, "bloom": {"level": "Ap", "variant": "secret"},
    }
    return TaskView.from_record(record, Path("."))


# ---------------------------------------------------------------------------
# Fairness contract
# ---------------------------------------------------------------------------


def test_prompt_contract_matches_geomapbench_eval() -> None:
    assert_prompt_contract_matches()


def test_taskview_never_exposes_gold_or_bloom() -> None:
    view = _view("metric_distance_computation", "Compute the distance.", {"requested_unit": "kilometres"})
    assert "target" not in view.payload
    assert "bloom" not in view.payload
    assert "SECRET" not in view.compact_json(9999)


def test_taskview_rejects_a_record_that_leaks_target() -> None:
    record = {"id": "x", "leaf": "l", "input": {"target": 1}}
    with pytest.raises(RuntimeError):
        TaskView.from_record(record, Path("."))


def test_taskview_strips_binary_assets_and_counts_images() -> None:
    view = _view(
        "metric_distance_computation", "Compute the distance.",
        {"points": [{"name": "Paris", "longitude": 2.3522, "latitude": 48.8566},
                   {"name": "Lyon", "longitude": 4.8357, "latitude": 45.7640}],
         "requested_unit": "kilometres", "images": ["assets/000.png"]},
    )
    assert "images" not in view.payload
    assert view.image_count == 1
    assert view.entity_names() == ["Paris", "Lyon"]


# ---------------------------------------------------------------------------
# Deterministic toolbelt: CRS, geodesic distance, route, budget, offsets
# ---------------------------------------------------------------------------


def test_crs_transform_matches_closed_form_web_mercator() -> None:
    lon, lat = 2.3522, 48.8566
    view = _view(
        "coordinate_transformation",
        "Transform the coordinate (2.3522, 48.8566) from EPSG:4326 to EPSG:3857. Interpret the "
        "input in longitude-then-latitude order and return x and y in metres.",
        {"source_crs": "EPSG:4326", "target_crs": "EPSG:3857",
         "coordinate": {"longitude": lon, "latitude": lat,
                        "axis_order": ["longitude", "latitude"], "unit": "degrees"},
         "transformation_mode": "geographic_to_web_mercator"},
    )
    result = T.tool_crs_transform(view)
    assert result.status == "ok"
    pair = result.value["transformed_coordinate"]
    expected_x = 6378137.0 * math.radians(lon)
    expected_y = 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    assert abs(pair["x"] - expected_x) < 0.01
    assert abs(pair["y"] - expected_y) < 0.01
    assert pair["unit"] == "metres"
    assert pair["axis_order"] == ["x", "y"]
    assert result.value["target_crs_kind"] == "projected"
    proposal = T.propose_answer(view, [result])
    assert proposal is not None and proposal.confidence == "exact" and proposal.value == pair


def test_crs_transform_utm_round_trip() -> None:
    from pyproj import Transformer

    lon, lat = 2.3522, 48.8566
    easting, northing = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform(lon, lat)
    view = _view(
        "coordinate_transformation",
        "Transform the coordinate from EPSG:32631 to EPSG:4326. Interpret the input in "
        "easting-then-northing order and return longitude and latitude in degrees.",
        {"source_crs": "EPSG:32631", "target_crs": "EPSG:4326",
         "coordinate": {"easting": round(easting, 3), "northing": round(northing, 3),
                        "axis_order": ["easting", "northing"], "unit": "metres"}},
    )
    pair = T.tool_crs_transform(view).value["transformed_coordinate"]
    assert pair["axis_order"] == ["longitude", "latitude"]
    assert abs(pair["longitude"] - lon) < 1e-5
    assert abs(pair["latitude"] - lat) < 1e-5


def test_crs_transform_verification_yes_no() -> None:
    view = _view(
        "coordinate_transformation",
        "Evaluate this proposed transformed coordinate: {...}. Is it correct within the task "
        "tolerance? Answer yes or no.",
        {"source_crs": "EPSG:4326", "target_crs": "EPSG:3857",
         "coordinate": {"longitude": 2.3522, "latitude": 48.8566,
                        "axis_order": ["longitude", "latitude"], "unit": "degrees"},
         "candidate_coordinate": {"x": 0.0, "y": 0.0}},
        choices=["yes", "no"],
    )
    result = T.tool_crs_transform(view)
    assert result.value["candidate_verdict"] == "no"
    assert T.propose_answer(view, [result]).value == "no"


def test_geodesic_distance_matches_pyproj_and_unit_table() -> None:
    from pyproj import Geod

    view = _view(
        "metric_distance_computation",
        "What is the WGS 84 geodesic distance from A (Paris) to B (Lyon) in kilometres?",
        {"points": [{"name": "Paris", "longitude": 2.3522, "latitude": 48.8566},
                   {"name": "Lyon", "longitude": 4.8357, "latitude": 45.7640}],
         "requested_unit": "kilometres"},
    )
    result = T.tool_geodesic_distance(view)
    _, _, reference_m = Geod(ellps="WGS84").inv(2.3522, 48.8566, 4.8357, 45.7640)
    assert result.value["value"] == round(abs(reference_m) / 1000.0, 3)
    assert 380.0 < result.value["value"] < 400.0
    assert result.value["unit"] == "km"
    assert result.value["method"] == "WGS84 inverse geodesic"
    assert result.value["all_units"]["miles"] == round(abs(reference_m) / 1609.344, 3)
    assert T.propose_answer(view, [result]).value == result.value["value"]


def test_geodesic_distance_verification() -> None:
    view = _view(
        "metric_distance_computation",
        "A reviewer reports the distance as 500 km. Is that value correct within the benchmark "
        "tolerance? Answer yes or no.",
        {"points": [{"longitude": 2.3522, "latitude": 48.8566},
                   {"longitude": 4.8357, "latitude": 45.7640}],
         "requested_unit": "kilometres", "candidate_value": 500.0},
        choices=["yes", "no"],
    )
    result = T.tool_geodesic_distance(view)
    assert T.propose_answer(view, [result]).value == "no"


def test_route_metrics_and_service_budget() -> None:
    view = _view(
        "shortest_path_optimization",
        "Analyze the reference route relative to the direct start-to-end separation. Return "
        "route length, direct distance, and route/direct detour ratio.",
        {"reference_route_coordinates": [[0, 0], [0, 1]], "start": [0, 0], "end": [0, 1]},
        evaluation_type="structured_numeric",
    )
    result = T.tool_route_metrics(view)
    assert result.status == "ok"
    proposal = T.propose_answer(view, [result])
    assert sorted(proposal.value) == ["detour_ratio", "direct_distance_m", "route_length_m"]

    view = _view(
        "isochrone_service_area",
        "Compute the maximum network travel distance in metres from the specified walking "
        "speed and time budget.",
        {"budget_minutes": 15, "speed_mps": 1.3, "network_distance_budget_m": 1170.0},
    )
    result = T.tool_service_budget(view)
    assert T.propose_answer(view, [result]).value == 1170.0


def test_ontology_and_offset_tools() -> None:
    view = _view(
        "dense_land_cover_labeling",
        "In the OpenEarthMap ontology used by this benchmark, what land-cover class name "
        "corresponds to class ID 5?",
        {"class_ontology": {"4": {"name": "tree"}, "5": {"name": "water"}}},
    )
    result = T.tool_class_ontology(view)
    assert T.propose_answer(view, [result]).value == "water"

    document = "Flooding was reported in Lyon and later in Grenoble."
    view = _view(
        "toponym_recognition",
        "What exact place-name text occurs at character offsets 25:29 in the provided document?",
        {"text": document, "query_offsets": [25, 29]},
    )
    result = T.tool_text_span(view)
    assert T.propose_answer(view, [result]).value == "Lyon"


def test_vision_only_leaf_has_no_proposal() -> None:
    view = _view("cartographic_symbol_recognition", "Identify the symbol.", {})
    assert T.propose_answer(view, []) is None


# ---------------------------------------------------------------------------
# Deterministic repair
# ---------------------------------------------------------------------------


def test_parse_answer_variants() -> None:
    assert R.parse_answer('```json\n{"answer": 12}\n```')[0] == 12
    assert R.parse_answer('{"answer": {"answer": "yes"}}')[0] == "yes"
    assert R.parse_answer("Sure! {\"answer\": [1,2]} done")[0] == [1, 2]
    assert R.parse_answer("not json")[1] == "invalid_json"


def test_repair_choices_and_yes_no() -> None:
    question = "Is that value correct within the benchmark tolerance? Answer yes or no."
    assert R.repair_yes_no(True, question)[0] == "yes"
    assert R.repair_yes_no("True", question)[0] == "yes"
    assert R.repair_yes_no("No, it is not correct", question)[0] == "no"
    assert R.repair_choices(
        "populated place", ["P — populated place", "T — terrain feature"]
    )[0] == "P — populated place"
    assert R.repair_choices("Forest", ["forest", "river"])[0] == "forest"


def test_repair_numeric_coercion() -> None:
    assert R.repair_numeric("431.27 km", "distance", "numeric")[0] == 431.27
    assert R.repair_numeric({"value": 5.5}, "d", "numeric")[0] == 5.5
    assert R.repair_numeric(5.5, "d", "numeric") == (5.5, None)


def test_repair_rle_normalises_to_4096_cells() -> None:
    short = {"encoding": "rle-row-major", "size": [64, 64], "runs": [[1, 2000], [2, 1000]]}
    fixed, action = R.repair_rle(short, ["1", "2", "3"])
    assert sum(count for _, count in fixed["runs"]) == 4096
    assert action == "rle_normalised"

    long_run = {"encoding": "rle-row-major", "size": [64, 64], "runs": [[1, 5000]]}
    fixed, _ = R.repair_rle(long_run, ["1"])
    assert sum(count for _, count in fixed["runs"]) == 4096

    bad_ids = {"encoding": "rle-row-major", "size": [64, 64], "runs": [[99, 4096]]}
    fixed, _ = R.repair_rle(bad_ids, ["1", "2", "3"])
    assert fixed["runs"][0][0] == 3

    wrapped = {"mask": dict(short), "changed_object_ids": ["a"]}
    fixed, _ = R.repair_rle(wrapped, ["0", "1", "2"])
    assert sum(c for _, c in fixed["mask"]["runs"]) == 4096
    assert sorted(fixed) == ["changed_object_ids", "mask"]

    valid = {"encoding": "rle-row-major", "size": [64, 64], "runs": [[1, 4096]]}
    assert R.repair_rle(valid, ["1"])[1] is None


def test_repair_graph_drops_dangling_edges_and_fills_length() -> None:
    graph = {
        "nodes": [{"id": 0, "x": 0, "y": 0}, {"id": 1, "x": 3, "y": 4}],
        "edges": [{"source": 0, "target": 1}, {"source": 0, "target": 9, "length": 2}],
    }
    fixed, action = R.repair_graph(graph)
    assert action == "graph_normalised"
    assert len(fixed["edges"]) == 1
    assert fixed["edges"][0]["length"] == 5.0


def test_repair_spans_realigns_offsets() -> None:
    document = "Paris is north of Lyon. Paris is large."
    spans = [{"text": "Paris", "start": 0, "end": 5}, {"text": "Lyon", "start": 99, "end": 103}]
    fixed, action = R.repair_spans(spans, document)
    assert action == "spans_offset_aligned"
    assert (fixed[1]["start"], fixed[1]["end"]) == (18, 22)
    assert document[fixed[1]["start"]:fixed[1]["end"]] == "Lyon"


def test_repair_structure_keys_aligns_to_the_tool_proposal() -> None:
    proposal = {"route_length_m": 100.0, "direct_distance_m": 50.0, "detour_ratio": 2.0}
    model = {"length": 100.0, "direct": 50.0, "ratio": 2.0}
    fixed, action = R.repair_structure_keys(model, proposal)
    assert action == "structure_keys_aligned"
    assert sorted(fixed) == sorted(proposal)


def test_repair_tool_disagreement_substitutes_scalars_and_text() -> None:
    assert R.repair_tool_disagreement(9.0, 10.0)[0] == 10.0
    assert R.repair_tool_disagreement(10.0, 10.0)[1] is None
    assert R.repair_tool_disagreement("kilometre", "km")[0] == "km"


def test_repair_answer_end_to_end() -> None:
    view = _view(
        "dense_land_cover_labeling", "Create the complete mask.",
        {"class_ontology": {"1": {"name": "bareland"}, "2": {"name": "tree"}}},
    )
    out = R.repair_answer(
        '{"answer": {"encoding": "rle-row-major", "size": [64,64], "runs": [[1, 100]]}}',
        view, ontology_ids=["1", "2"],
    )
    assert sum(c for _, c in out["answer"]["runs"]) == 4096
    assert "rle_normalised" in out["repairs"]

    view = _view(
        "metric_distance_computation", "A reviewer reports the distance as 5. Answer yes or no.",
        {"candidate_value": 5}, choices=["yes", "no"],
    )
    assert R.repair_answer('{"answer": true}', view)["answer"] == "yes"

    view = _view("coordinate_transformation", "Transform the coordinate.", {})
    out = R.repair_answer("garbage", view, exact_proposal={"x": 1})
    assert out["answer"] == {"x": 1}
    assert out["parse_error"] is None


# ---------------------------------------------------------------------------
# Hybrid retrieval fusion (no FAISS/CLIP instantiated)
# ---------------------------------------------------------------------------


def _hit(record_id, title, text, capabilities=()):
    return {"record": {"id": record_id, "input": {"title": title, "text": text},
                       "retrieval": {"capabilities": list(capabilities)}}}


def test_lexical_ranking_prefers_the_rare_proper_noun() -> None:
    from geoagent.retrieval import HybridMultimodalRetriever

    retriever = HybridMultimodalRetriever.__new__(HybridMultimodalRetriever)
    pool = [
        _hit("a", "Reykjavik", "Reykjavik is the capital of Iceland and its largest city."),
        _hit("b", "Paris", "Paris is the capital of France and a major European city."),
        _hit("c", "Generic", "A city is a large human settlement with administrative status."),
    ]
    ranking = retriever._lexical_ranking("population of Reykjavik Iceland", pool)
    assert ranking[0] == "a"
    assert retriever._lexical_ranking("", pool) == []


def test_fusion_applies_capability_bonus_without_disturbing_others() -> None:
    from geoagent.retrieval import HybridMultimodalRetriever

    retriever = HybridMultimodalRetriever.__new__(HybridMultimodalRetriever)
    pool = {item["record"]["id"]: item for item in [
        _hit("a", "Reykjavik", "text"), _hit("b", "Reykjavik", "text"), _hit("c", "City", "text"),
    ]}
    rankings = {"dense": ["a", "b", "c"], "lexical": ["a", "c"], "rerank": ["b", "a"], "image": []}
    plain = {item["record"]["id"]: item["fusion_score"]
             for item in retriever._fuse_ranked(pool, rankings, "no_such_leaf")}
    pool["b"]["record"]["retrieval"]["capabilities"] = ["toponym_recognition"]
    boosted = {item["record"]["id"]: item for item in
               retriever._fuse_ranked(pool, rankings, "toponym_recognition")}
    assert boosted["b"]["capability_match"] is True
    assert boosted["a"]["capability_match"] is False
    assert round(boosted["b"]["fusion_score"] / plain["b"], 4) == 1.30
    assert boosted["a"]["fusion_score"] == plain["a"]


def _bare_retriever(media_root: Path) -> HybridMultimodalRetriever:
    """A HybridMultimodalRetriever with __init__ skipped, for probing internals
    without FAISS/CLIP/a cross-encoder actually loaded."""
    import types

    retriever = HybridMultimodalRetriever.__new__(HybridMultimodalRetriever)
    retriever.reranker = None
    retriever.candidate_k = 10
    retriever.image_candidate_k = 10
    retriever.max_reference_images = 1
    retriever.max_passage_chars = 900
    retriever.max_context_chars = 3000
    retriever.media_root = media_root
    retriever.corpus_root = media_root
    retriever.text_index = types.SimpleNamespace(ntotal=1)
    retriever.image_index = types.SimpleNamespace(ntotal=1)
    return retriever


def test_validate_runtime_forces_an_accessible_image_hit_into_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The natural top-k hybrid shortlist is not guaranteed to keep any
    # image-bearing candidate (MMR/capability boosting can legitimately push
    # them all out), so the probe must force one in rather than trust ranking.
    media_root = tmp_path / "corpus"
    (media_root / "assets").mkdir(parents=True)
    (media_root / "assets" / "pic.png").write_bytes(b"png")
    text_record = {"id": "text:1", "input": {"title": "T",
                                             "text": "A retrievable passage long enough to survive rendering."}}
    image_record = {"id": "img:1", "input": {"title": "Pic", "text": "", "images": ["assets/pic.png"]}}

    task_dir = tmp_path / "visual_geolocation"
    (task_dir / "assets").mkdir(parents=True)
    (task_dir / "assets" / "000.png").write_bytes(b"png")
    record = {
        "id": "visual_geolocation:000", "leaf": "visual_geolocation",
        "input": {"images": ["assets/000.png"], "question": "Where is this?"},
        "evaluation": {"type": "exact_match"},
    }

    retriever = _bare_retriever(media_root)
    monkeypatch.setattr(retriever, "_dense", lambda query, k: [
        {"record": text_record, "dense_score": 1.0, "dense_rank": 1},
    ])
    monkeypatch.setattr(retriever, "_image_hits", lambda image_paths, fallback_text: (
        [{"record": image_record, "image_score": 0.9, "image_rank": 1}], len(image_paths),
    ))

    report = validate_runtime(retriever, None, [(task_dir, record)], top_k=2)
    assert report["status"] == "pass"
    assert report["retrieved_reference_images"] >= 1
    assert report["prompt_image_parts"] > report["benchmark_image_count"]
    assert report["fallback_sentence_present"] is True
    # No applicable deterministic tool fires for this synthetic record (no
    # structured_index, no matching payload fields), so the "Verified
    # computations" block is correctly absent -- it is informational, not a
    # pass/fail condition.
    assert report["verified_block_present"] is False


def test_validate_runtime_raises_when_no_image_hit_is_accessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "corpus"
    media_root.mkdir(parents=True)
    text_record = {"id": "text:1", "input": {"title": "T",
                                             "text": "A retrievable passage long enough to survive rendering."}}
    # References a file that does not exist under media_root.
    missing_image_record = {"id": "img:1", "input": {"title": "Pic", "text": "", "images": ["assets/missing.png"]}}

    task_dir = tmp_path / "visual_geolocation"
    (task_dir / "assets").mkdir(parents=True)
    (task_dir / "assets" / "000.png").write_bytes(b"png")
    record = {
        "id": "visual_geolocation:000", "leaf": "visual_geolocation",
        "input": {"images": ["assets/000.png"], "question": "Where is this?"},
        "evaluation": {"type": "exact_match"},
    }

    retriever = _bare_retriever(media_root)
    monkeypatch.setattr(retriever, "_dense", lambda query, k: [
        {"record": text_record, "dense_score": 1.0, "dense_rank": 1},
    ])
    monkeypatch.setattr(retriever, "_image_hits", lambda image_paths, fallback_text: (
        [{"record": missing_image_record, "image_score": 0.9, "image_rank": 1}], len(image_paths),
    ))

    with pytest.raises(RuntimeError, match="inaccessible"):
        validate_runtime(retriever, None, [(task_dir, record)], top_k=2)


def test_mmr_breaks_up_near_duplicate_passages() -> None:
    from geoagent.retrieval import HybridMultimodalRetriever

    retriever = HybridMultimodalRetriever.__new__(HybridMultimodalRetriever)
    duplicates = [
        {**_hit("d1", "Nile", "The Nile is a major river in northeastern Africa."), "fusion_score": 1.0},
        {**_hit("d2", "Nile", "The Nile is a major river in northeastern Africa."), "fusion_score": 0.99},
        {**_hit("d3", "Andes", "The Andes are a mountain range along South America."), "fusion_score": 0.50},
    ]
    selected = [item["record"]["id"] for item in retriever._mmr(duplicates, 2)]
    assert selected == ["d1", "d3"]
    assert len(retriever._mmr(duplicates, 5)) == 3


# ---------------------------------------------------------------------------
# Structured corpus index and corpus-backed tools
# ---------------------------------------------------------------------------


def _wdi(country, year, value, label="population density",
         unit="people per sq. km of land area", code="EN.POP.DNST"):
    return {
        "id": f"worldbank:{code}:{year}:{country}", "source": "World Bank",
        "input": {"modality": "structured", "title": f"{country} — {label}, {year}",
                  "text": f"In {year}, {country} had {label} of {value} {unit}. "
                          f"World Bank indicator: {code}. World Bank region: Europe."},
        "retrieval": {"document_type": "country_indicator_observation", "capabilities": []},
    }


def _place(title, lat, lon, source="GeoNames", document_type="gazetteer_entry"):
    return {
        "id": f"{source}:{title}", "source": source,
        "input": {"modality": "structured", "title": title,
                  "text": f"{title} is a place. Coordinates: latitude {lat}, longitude {lon}.",
                  "geo": {"lat": lat, "lon": lon}},
        "retrieval": {"document_type": document_type, "capabilities": []},
    }


@pytest.fixture()
def structured_index() -> StructuredCorpusIndex:
    records = [
        _wdi("France", 2015, 118.0), _wdi("France", 2019, 122.0), _wdi("France", 2022, 124.0),
        _wdi("Portugal", 2015, 112.0), _wdi("Portugal", 2019, 111.0),
        _wdi("Portugal", 2010, 800.0, label="GDP per capita", unit="current US dollars",
             code="NY.GDP.PCAP.CD"),
        _place("Paris", 48.8566, 2.3522), _place("Lyon", 45.7640, 4.8357),
        _place("Berlin", 52.5200, 13.4050, source="Wikidata", document_type="wikidata_settlement"),
        {"id": "noise", "source": "Wikipedia",
         "input": {"modality": "text", "title": "Rivers", "text": "A river is a watercourse."},
         "retrieval": {"document_type": "reference", "capabilities": []}},
    ]
    return StructuredCorpusIndex(records)


def test_indicator_series_parsed_and_interpolated(structured_index: StructuredCorpusIndex) -> None:
    series = structured_index.indicator_series("EN.POP.DNST", "France")
    assert [row["year"] for row in series] == [2015, 2019, 2022]
    assert series[0]["value"] == 118.0
    assert len(structured_index.indicator_series("EN.POP.DNST", "  france ")) == 3
    assert len(structured_index.indicator_series("NY.GDP.PCAP.CD", "Portugal")) == 1

    exact = StructuredCorpusIndex.interpolate(series, 2019)
    assert (exact["value"], exact["method"]) == (122.0, "exact corpus observation")
    mid = StructuredCorpusIndex.interpolate(series, 2017)
    assert round(mid["value"], 3) == 120.0
    carried = StructuredCorpusIndex.interpolate(series, 2030)
    assert carried["value"] == 124.0


def test_gazetteer_lookup_and_nearby(structured_index: StructuredCorpusIndex) -> None:
    hit = structured_index.locate("Paris")
    assert (round(hit["lat"], 4), round(hit["lon"], 4)) == (48.8566, 2.3522)
    assert structured_index.locate("Berlin")["source"] == "Wikidata"
    assert structured_index.locate("area around Paris")["title"] == "Paris"
    assert structured_index.locate("Atlantis") is None
    nearest = structured_index.near(48.9, 2.4, limit=2)
    assert nearest[0]["title"] == "Paris"
    assert nearest[0]["distance_km"] < 20


def test_indicator_tool_discloses_withheld_years_and_estimates(structured_index: StructuredCorpusIndex) -> None:
    view = _view(
        "population_density_estimation", "What was the population density of France in 2017?",
        {"country": "France", "year": 2017, "indicator": "EN.POP.DNST"}, evaluation_type="numeric",
    )
    result = T.tool_indicator_lookup(view, structured_index)
    assert result.status == "ok"
    assert result.authority == "strong"
    assert "does not contain" in result.text
    estimate = result.value["series"]["France"]["estimate_for_requested_year"]
    assert round(estimate["value"], 3) == 120.0


def test_indicator_tool_ranks_two_countries(structured_index: StructuredCorpusIndex) -> None:
    view = _view(
        "cross_entity_comparison", "Which had the higher population density in 2019: France or Portugal?",
        {"entities": ["France", "Portugal"], "indicator": "EN.POP.DNST",
         "indicator_name": "population density", "year": 2019},
    )
    result = T.tool_indicator_lookup(view, structured_index)
    assert result.value["descending_ranking"] == ["France", "Portugal"]
    assert T.propose_answer(view, [result]).value == "France"

    view = _view(
        "cross_entity_comparison", "For 2019, which country had the smaller population density: "
        "France or Portugal?",
        {"entities": ["France", "Portugal"], "indicator_name": "population density", "year": 2019},
    )
    proposal = T.propose_answer(view, [T.tool_indicator_lookup(view, structured_index)])
    assert proposal.value == "Portugal"


def test_gazetteer_tool_stays_advisory(structured_index: StructuredCorpusIndex) -> None:
    view = _view(
        "topological_directional_reasoning",
        "What is the approximate cardinal direction of polygon A (Paris) relative to polygon B (Lyon)?",
        {"labels": {"A": "Paris", "B": "Lyon"}},
    )
    result = T.tool_gazetteer(view, structured_index)
    assert result.value["approximate_direction_a_to_b"] == "northwest"
    assert result.authority == "advisory"
    assert T.propose_answer(view, [result]) is None


def test_normalize_name_strips_accents_and_punctuation() -> None:
    assert normalize_name("Côte d'Ivoire") == "cote d ivoire"


# ---------------------------------------------------------------------------
# End-to-end pipeline, resumable driver, and prompt assembly
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search_evidence(self, view, *, image_paths, queries, top_k):
        self.calls += 1
        contexts = [
            {"id": "corpus:1", "input": {"title": "Paris", "text": "Paris is a city in France."},
             "image_paths": [], "modalities": ["text"]},
            {"id": "corpus:2", "input": {"title": "Lyon", "text": "Lyon is a city in France."},
             "image_paths": [], "modalities": ["text"]},
        ]
        return contexts, {"mode": "fake", "text_index_used": True, "image_index_used": True,
                          "benchmark_image_count": len(image_paths), "queries": queries,
                          "capability_matches": 1, "record_ids": ["corpus:1", "corpus:2"]}


def _stub_openrouter(monkeypatch: pytest.MonkeyPatch, script: list[str]) -> None:
    import geomapbench_eval.openrouter as openrouter_module

    remaining = list(script)

    def fake_complete(self, messages, config):
        text = remaining.pop(0) if remaining else '{"answer": "stub"}'
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5},
            "_latency_seconds": 0.1,
        }

    monkeypatch.setattr(openrouter_module.OpenRouterClient, "complete", fake_complete, raising=True)
    monkeypatch.setattr(openrouter_module.OpenRouterClient, "__init__", lambda self, api_key=None: None)


def test_pipeline_corrects_a_wrong_verification_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = tmp_path / "metric_distance_computation"
    (task_dir / "assets").mkdir(parents=True)
    (task_dir / "assets" / "000.png").write_bytes(b"png")
    record = {
        "id": "metric_distance_computation:000", "leaf": "metric_distance_computation",
        "input": {
            "images": ["assets/000.png"],
            "question": "A reviewer reports the distance as 500 km. Is that value correct within "
                        "the benchmark tolerance? Answer yes or no.",
            "points": [{"name": "Paris", "longitude": 2.3522, "latitude": 48.8566},
                      {"name": "Lyon", "longitude": 4.8357, "latitude": 45.7640}],
            "requested_unit": "kilometres", "candidate_value": 500.0, "choices": ["yes", "no"],
        },
        "target": {"bloom_answer": "no"}, "bloom": {"level": "E", "variant": "evaluate_distance_value"},
        "evaluation": {"type": "binary_classification"},
    }
    _stub_openrouter(monkeypatch, [
        json.dumps({"answer_shape": "?", "shape_note": "a yes/no string",
                    "queries": ["Paris Lyon distance"], "pitfalls": []}),
        json.dumps({"keep": [1], "use_context": True, "sufficient": True, "followup": "", "note": ""}),
        '{"answer": "Yes, that is correct"}',
    ])
    agent = CachedAgent(model="fake/agent", cache_root=tmp_path / "agent_cache")
    retriever = _FakeRetriever()
    pipeline = AgenticPipeline(retriever=retriever, agent=agent, structured_index=None, top_k=2)

    calls: list[str] = []

    def answer_fn(messages, tag):
        calls.append(tag)
        from geomapbench_eval.openrouter import OpenRouterClient, OpenRouterConfig

        response = OpenRouterClient().complete(messages, OpenRouterConfig("fake"))
        return response["choices"][0]["message"]["content"], {"performed": True}

    outcome = pipeline.solve(record, task_dir, answer_fn=answer_fn)

    assert retriever.calls == 1
    assert calls[0] == "answer"
    assert "geodesic_distance" in outcome.stages["tool_names"]
    assert outcome.stages["proposal"]["confidence"] == "exact"
    assert outcome.stages["proposal"]["value"] == "no"
    assert outcome.answer == "no"
    assert outcome.raw_text == '{"answer": "Yes, that is correct"}'
    assert json.loads(outcome.text)["answer"] == "no"


def test_prompt_carries_verified_and_retrieved_blocks(tmp_path: Path) -> None:
    task_dir = tmp_path / "metric_distance_computation"
    (task_dir / "assets").mkdir(parents=True)
    (task_dir / "assets" / "000.png").write_bytes(b"png")
    record = {
        "id": "x", "leaf": "metric_distance_computation",
        "input": {"images": ["assets/000.png"], "question": "Answer yes or no."},
        "target": {}, "evaluation": {"type": "binary_classification"},
    }
    messages = build_agent_messages(
        record, task_dir,
        tool_blocks=[{"title": "t", "authority": "authoritative", "text": "verified text"}],
        contexts=[{"id": "c", "input": {"title": "T", "text": "body"}, "image_paths": []}],
        answer_shape="?", shape_note="a yes/no string", pitfalls=["care"],
    )
    text = "\n".join(p.get("text", "") for p in messages[1]["content"] if p["type"] == "text")
    assert "Verified computations" in text
    assert "Retrieved text evidence" in text
    assert "Suggested answer shape" in text
    assert "answer from the original task images and your own knowledge" in text
    assert sum(p["type"] == "image_url" for p in messages[1]["content"]) == 1


def test_cached_agent_serves_a_repeat_call_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_openrouter(monkeypatch, [
        json.dumps({"answer_shape": "?", "shape_note": "note", "queries": ["q"], "pitfalls": []}),
    ])
    cache = tmp_path / "cache"
    first = CachedAgent(model="fake/agent", cache_root=cache)
    view = _view("metric_distance_computation", "Compute the distance.", {"requested_unit": "kilometres"})
    analyse(first, view)

    _stub_openrouter(monkeypatch, [])  # nothing left to serve; a second call must not need one
    second = CachedAgent(model="fake/agent", cache_root=cache)
    replay = analyse(second, view)
    assert replay["queries"] == ["q"]
    assert second.usage["cached_calls"] == 1
    assert second.usage["calls"] == 0


def test_driver_resumes_for_free_and_recovers_unparsable_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = tmp_path / "bench" / "coordinate_transformation"
    (task_dir / "assets").mkdir(parents=True)
    (task_dir / "assets" / "000.png").write_bytes(b"png")
    base_record = {
        "leaf": "coordinate_transformation",
        "input": {
            "images": ["assets/000.png"],
            "question": "Transform the coordinate from EPSG:4326 to EPSG:3857. Interpret the "
                        "input in longitude-then-latitude order and return x and y in metres.",
            "source_crs": "EPSG:4326", "target_crs": "EPSG:3857",
            "coordinate": {"longitude": 2.3522, "latitude": 48.8566,
                           "axis_order": ["longitude", "latitude"], "unit": "degrees"},
            "transformation_mode": "geographic_to_web_mercator",
        },
        "target": {"bloom_answer": {"x": 1.0}},
        "bloom": {"level": "Ap", "variant": "apply_coordinate_transform"},
        "evaluation": {"type": "numeric_coordinate_pair"},
    }
    # Distinct coordinates per record: the cheap analyst is cached by prompt
    # content, not by record id, so two identical records would collapse into
    # one cached analyst call and desynchronise the scripted answer sequence
    # below.
    records = []
    for index, (lon, lat) in enumerate([(2.3522, 48.8566), (13.4050, 52.5200)]):
        record = copy.deepcopy(base_record)
        record["id"] = f"coordinate_transformation:{index:03d}"
        record["input"]["coordinate"]["longitude"] = lon
        record["input"]["coordinate"]["latitude"] = lat
        records.append((task_dir, record))

    output = tmp_path / "out"
    cohort_path, cohort = write_cohort_manifest(
        records, target_per_leaf=2, output_root=output, benchmark_content_hash="hash",
    )
    assert cohort["target_record_count"] == 2

    class _AbstainingRetriever:
        def search_evidence(self, view, *, image_paths, queries, top_k):
            return [], {"mode": "fake", "text_index_used": False, "image_index_used": False,
                       "abstained": True, "abstain_reason": "outside_predeclared_corpus_coverage"}

    _stub_openrouter(monkeypatch, [
        json.dumps({"answer_shape": {"x": "?", "y": "?"}, "shape_note": "x and y in metres",
                    "queries": [], "pitfalls": []}),
        '{"answer": {"x": 1.0, "y": 2.0, "axis_order": ["x","y"], "unit": "metres", "crs": "EPSG:3857"}}',
        json.dumps({"answer_shape": {"x": "?", "y": "?"}, "shape_note": "x and y in metres",
                    "queries": [], "pitfalls": []}),
        "this is not json",
    ])
    agent = CachedAgent(model="fake/agent", cache_root=tmp_path / "cache")
    pipeline = AgenticPipeline(retriever=_AbstainingRetriever(), agent=agent, structured_index=None, top_k=3)
    config = RunConfig(
        benchmark_root=tmp_path / "bench", output=output / "geoagent_tool_rag",
        condition="geoagent_tool_rag", model="fake/answer", max_tokens=16384,
        temperature=None, max_cost_usd=5.0, progress_every=1, top_k=3,
        benchmark_content_hash="hash", retrieval_config={"system": "geoagent_v3"},
    )
    summary = run_agentic(config, records=records, cohort_path=cohort_path, pipeline=pipeline, agent=agent)
    assert summary["complete"] is True
    assert summary["completed_total"] == 2
    assert summary["revision_calls_this_invocation"] == 0

    rows = {row["id"]: row for row in read_jsonl(output / "geoagent_tool_rag" / "responses.jsonl")}
    first = rows["coordinate_transformation:000"]
    answer = json.loads(first["response"])["answer"]
    assert round(answer["x"]) == 261846  # PROJ-corrected, not the model's wrong x=1.0
    assert sorted(answer) == ["axis_order", "crs", "unit", "x", "y"]
    assert json.loads(first["raw_response"])["answer"]["x"] == 1.0
    assert any("tool_object_substituted" in item for item in first["repairs"])

    second = rows["coordinate_transformation:001"]
    assert second["repairs"] == ["recovered_unparsable_response_with_tool_value"]
    assert second["raw_response"] == "this is not json"
    assert json.loads(second["response"])["answer"]["unit"] == "metres"

    run_config = json.loads((output / "geoagent_tool_rag" / "run_config.json").read_text())
    assert run_config["condition"] == "geoagent_tool_rag"
    assert run_config["cumulative"] is True
    assert len(run_config["selected_ids"]) == 2

    # Resume: everything is already complete, so a second invocation costs nothing.
    _stub_openrouter(monkeypatch, [])
    again = run_agentic(config, records=records, cohort_path=cohort_path, pipeline=pipeline, agent=agent)
    assert again["run_stop_reason"] == "already_complete"
    assert again["reported_cost_usd_this_invocation"] == 0.0

    from geoagent.report import toolbelt_audit

    audit = toolbelt_audit(output / "geoagent_tool_rag" / "agent_trace.jsonl")
    assert audit["records"] == 2
    assert audit["records_with_exact_proposal"] == 2


# ---------------------------------------------------------------------------
# Rescore avoidance: canonical_benchmark_records caching, skip_rescore plumbing
# ---------------------------------------------------------------------------
#
# rescore_in_place (via analyze/compare) re-walks every row and, for mask/graph
# leaves, reopens a gold asset per row -- real cost against a Drive-mounted
# benchmark. These prove the caching and skip-rescore paths that avoid paying
# that cost more than once per file within a single suite invocation.


def _make_run_config(results_path: Path, *, protocol: dict, benchmark_root: Path) -> None:
    atomic_json(results_path.parent / "run_config.json", {
        "condition": "base", "benchmark_root": str(benchmark_root), "protocol": protocol,
    })


def test_already_scored_detects_a_matching_protocol(tmp_path: Path) -> None:
    results = tmp_path / "responses.jsonl"
    results.write_text(json.dumps({"id": "a", "status": "ok", "task_score": 0.8}) + "\n", encoding="utf-8")
    _make_run_config(results, protocol=protocol_descriptor(), benchmark_root=tmp_path)
    assert _already_scored_by_current_code(results) is True


def test_already_scored_is_false_on_a_stale_protocol(tmp_path: Path) -> None:
    results = tmp_path / "responses.jsonl"
    results.write_text(json.dumps({"id": "a", "status": "ok", "task_score": 0.8}) + "\n", encoding="utf-8")
    stale = {**protocol_descriptor(), "task_metric_revision": "some-old-revision"}
    _make_run_config(results, protocol=stale, benchmark_root=tmp_path)
    assert _already_scored_by_current_code(results) is False


def test_already_scored_is_false_when_task_score_is_missing(tmp_path: Path) -> None:
    # Exactly the shape of a freshly written geoagent row, which the driver
    # never populates with task_score at write time -- rescoring is required.
    results = tmp_path / "responses.jsonl"
    results.write_text(json.dumps({"id": "a", "status": "ok", "response": "{}"}) + "\n", encoding="utf-8")
    _make_run_config(results, protocol=protocol_descriptor(), benchmark_root=tmp_path)
    assert _already_scored_by_current_code(results) is False


def test_already_scored_is_false_without_a_run_config(tmp_path: Path) -> None:
    results = tmp_path / "responses.jsonl"
    results.write_text(json.dumps({"id": "a", "status": "ok", "task_score": 0.8}) + "\n", encoding="utf-8")
    assert _already_scored_by_current_code(results) is False


def _synthetic_benchmark(root: Path) -> None:
    """A minimal 23-leaf, 100-record-per-leaf benchmark satisfying
    canonical_benchmark_records' shape checks, for exercising its cache."""
    from geomapbench_data.common import SEEDS

    for leaf in SEEDS:
        leaf_dir = root / leaf
        leaf_dir.mkdir(parents=True)
        rows = [{"id": f"{leaf}-{index:03d}", "leaf": leaf, "input": {"text": "x" * 40}} for index in range(100)]
        with (leaf_dir / "data_clean.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def test_canonical_benchmark_records_reads_disk_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geomapbench_eval import benchmark as benchmark_module

    benchmark_module._canonical_benchmark_records_cached.cache_clear()
    root = tmp_path / "bench"
    _synthetic_benchmark(root)

    calls = {"count": 0}
    original_read_jsonl = benchmark_module.read_jsonl

    def counting_read_jsonl(path):
        calls["count"] += 1
        return original_read_jsonl(path)

    monkeypatch.setattr(benchmark_module, "read_jsonl", counting_read_jsonl)

    first = benchmark_module.canonical_benchmark_records(root)
    assert len(first) == 2300
    assert calls["count"] == 23  # one read per leaf file

    second = benchmark_module.canonical_benchmark_records(root)
    assert len(second) == 2300
    assert calls["count"] == 23  # unchanged: served from the process-wide cache

    benchmark_module._canonical_benchmark_records_cached.cache_clear()


def test_analyze_and_compare_accept_skip_rescore_without_raising() -> None:
    import inspect

    from geomapbench_eval.analysis import analyze, compare

    assert "skip_rescore" in inspect.signature(analyze).parameters
    assert "skip_rescore" in inspect.signature(compare).parameters


def test_analyze_skip_rescore_trusts_pre_populated_scores_and_touches_no_gold_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geomapbench_eval import analysis as analysis_module

    benchmark_root = tmp_path / "bench"
    _synthetic_benchmark(benchmark_root)
    results = tmp_path / "condition" / "responses.jsonl"
    results.parent.mkdir(parents=True)
    row = {
        "id": "coordinate_transformation-000", "leaf": "coordinate_transformation",
        "condition": "base", "model": "m", "status": "ok",
        "response": '{"answer": "precomputed"}', "task_score": 0.75, "strict_score": 1.0,
        "parse_error": None, "generation_failure": None, "usage": {}, "retrieval_usage": {},
        "latency_seconds": 1.0,
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def forbidden_rescore(*args, **kwargs):
        raise AssertionError("skip_rescore=True must not touch rescore_in_place")

    monkeypatch.setattr(analysis_module, "rescore_in_place", forbidden_rescore)

    summary = analysis_module.analyze(
        results, tmp_path / "analysis", benchmark_root=benchmark_root, skip_rescore=True,
    )
    assert summary["condition_summary"]["base"]["n"] == 1
    assert summary["condition_summary"]["base"]["task_aware_macro"] == pytest.approx(0.75)
    assert summary["rescore"]["skipped"] is True
