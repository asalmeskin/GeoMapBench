# GeoMapBench Bloom-balanced expansion

This release adds a **deterministic Bloom-taxonomy overlay** to an already-built 23-leaf GeoMapBench dataset. It does **not** download new upstream data and does **not** regenerate imagery. It rewrites only `data.jsonl` and `manifest.json` while retaining all original task assets and all original target fields.

The converter is intended for the user's existing frozen `geomapbench_100` release. Every leaf remains exactly 100 examples.

## Operational interpretation of Bloom levels

GeoMapBench uses the revised Bloom taxonomy as an **operational benchmark stratification**, not as a claim about a model's internal cognitive process:

- `R` — **Remember**: recognize or retrieve a label, unit, entity, fact, or directly indexed annotation.
- `U` — **Understand**: interpret or classify a geographic representation or relationship.
- `Ap` — **Apply**: execute a declared geospatial procedure or apply an ontology/rule.
- `An` — **Analyze**: decompose, compare, aggregate, invert, or derive relationships among components.
- `E` — **Evaluate**: verify or reject a candidate answer, annotation, route cost, grouping, or claim.
- `C` — **Create**: construct a structured artifact such as a mask, graph, relation graph, route, service-area polygon, or multi-field record.

Some leaves do not have a scientifically defensible `Create` or `Remember` variant from the current source annotations. Those leaves use five levels rather than forcing an artificial sixth level.

## Exact balance

For five-level leaves, the converter creates exactly:

- 20 examples per level.

For six-level leaves, the converter creates exactly:

- 17 `R`
- 17 `U`
- 17 `Ap`
- 17 `An`
- 16 `E`
- 16 `C`

For the two five-level leaves without `Remember` (`change_localization` and `temporal_scene_matching`), the 20/level balance is over `U, Ap, An, E, C`.

The level assignment is shuffled deterministically using each leaf's existing immutable seed plus a fixed Bloom offset. Therefore repeated conversion of the same base release produces the same level assignment and prompts.

## Per-leaf coverage and variants

| Leaf | Levels | Main Bloom variants generated from the current record |
|---|---|---|
| `cartographic_symbol_recognition` | R,U,Ap,An,E | symbol-label recognition; semantic interpretation; apply legend; discriminate candidates; verify category claim |
| `map_text_detection_recognition_grouping` | R,U,Ap,An,E,C | transcribe one polygon; count label groups; word spotting; group supplied words; evaluate grouping; create complete spotting/grouping annotation |
| `map_label_feature_anchoring` | R,U,Ap,An,E,C | transcribe visible label; identify geometry family; select candidate; return candidate+geometry type; verify grounding; create structured grounding record |
| `dense_land_cover_labeling` | R,U,Ap,An,E,C | recall ontology mapping; dominant class; classify a queried pixel; rank class prevalence; verify class claim; create complete mask |
| `remote_sensing_scene_classification` | R,U,Ap,An,E | recognize class; interpret scene; apply EuroSAT ontology; discriminate two candidates; verify classification |
| `object_presence_counting` | R,U,Ap,An,E | recognize requested fact; interpret presence/count query; apply visual counting; return answer+operation; verify proposed answer |
| `change_localization` | U,Ap,An,E,C | count changed tracked objects; create change mask; return IDs+count; verify count; create complete change annotation |
| `temporal_scene_matching` | U,Ap,An,E,C | interpret pair; apply temporal matching; return relation; verify match claim; construct same-AOI image pairings from existing positive pairs |
| `visual_geolocation` | R,U,Ap,An,E | country; city; city+country; full city/country/coordinates; verify candidate location |
| `coordinate_transformation` | R,U,Ap,An,E,C | target unit; CRS type; transform coordinates; report axes/unit; verify candidate transform; create complete transformation record |
| `metric_distance_computation` | R,U,Ap,An,E | unit symbol; distance-method interpretation; compute requested distance; report canonical forms; verify numeric claim |
| `topological_directional_reasoning` | R,U,Ap,An,E,C | relation label; topology-vs-direction family; apply relation; derive inverse relation; verify claim; create bidirectional relation graph |
| `spatial_graph_construction` | R,U,Ap,An,E,C | node-count lookup from reference graph; graph directionality; aggregate edge lengths; graph summary; verify summary; construct graph from imagery |
| `shortest_path_optimization` | R,U,Ap,An,E,C | route unit; optimization objective; compute length from supplied route; detour analysis; verify route cost; construct shortest route |
| `isochrone_service_area` | R,U,Ap,An,E,C | time budget; network-reachability concept; speed×time distance; node/area summary; verify area; create service-area polygon |
| `toponym_recognition` | R,U,Ap,An,E | recover text at offsets; count mentions; extract spans; unique-toponym inventory; verify span annotation |
| `geo_entity_typing` | R,U,Ap,An,E | class-code recall; interpret broad class; apply ontology; discriminate two classes; verify classification |
| `textual_spatial_relation_extraction` | R,U,Ap,An,E,C | recognize relation; interpret description; apply relation; return relation set; verify answer; create structured relation record |
| `cross_entity_comparison` | R,U,Ap,An,E,C | retrieve one indicator value; interpret higher/lower query; compute difference; ratio/value analysis; verify answer; create comparison report |
| `environmental_layer_identification` | R,U,Ap,An,E | recall layer unit; interpret raster; apply layer ontology; return layer+unit+summary profile; verify layer claim |
| `population_density_estimation` | R,U,Ap,An,E,C | retrieve density; interpret against a threshold; apply density to hypothetical area; compare same-year countries; verify value; create three-country ranking |
| `geologic_geomorphic_interpretation` | R,U,Ap,An,E,C | unit name; lithology/material; apply geologic lookup; organize geologic attributes; verify unit; create structured geologic profile |
| `geographic_fact_reasoning` | R,U,Ap,An,E | direct fact; interpret question; apply supplied reference query; map question to formal query; verify proposed answer |

The machine-readable version is `config/bloom_taxonomy.csv`.

## Record format

The original target fields are retained. Bloom conversion adds:

```json
{
  "input": {
    "base_question": "...",
    "question": "Bloom-specific question"
  },
  "target": {
    "...original target fields...": "...",
    "bloom_answer": "the primary answer for the Bloom variant",
    "bloom_response_format": "variant identifier"
  },
  "base_evaluation": {
    "...original evaluation...": "..."
  },
  "evaluation": {
    "type": "...",
    "target_field": "target.bloom_answer",
    "bloom_level": "An",
    "bloom_level_name": "Analyze"
  },
  "bloom": {
    "revision": "2026-08-bloom-v1",
    "level": "An",
    "level_name": "Analyze",
    "variant": "...",
    "source_record_ids": ["..."],
    "supported_levels": ["R", "U", "Ap", "An", "E", "C"]
  }
}
```

The manifest keeps the underlying `data_revision` and adds a separate `bloom_revision`, `bloom_distribution`, and `base_sha256_before_bloom`.

## Convert an existing release

```bash
geomapbench-data validate --root /path/to/geomapbench_100 --require-all
geomapbench-data bloomify --root /path/to/geomapbench_100
geomapbench-data validate --root /path/to/geomapbench_100 --require-all
geomapbench-data bloom-audit --root /path/to/geomapbench_100
```

By default, before conversion each leaf saves only the two small metadata files under:

```text
<leaf>/.pre_bloom/data.jsonl
<leaf>/.pre_bloom/manifest.json
```

No imagery, masks, graph files, or other large assets are duplicated.

To restore the original pre-Bloom metadata:

```bash
geomapbench-data bloom-restore --root /path/to/geomapbench_100
```

## Important benchmark-design caveat

Bloom balancing changes the **question/response operation** attached to each existing source example. It does not create independent geographic evidence. Therefore all Bloom variants derived from one source group must continue to share leakage controls. The converter records `source_record_ids`, and for the few multi-record variants it combines their `group_id`s.

For future train/dev data, do not generate Bloom siblings from benchmark source groups. Keep GeoMapBench frozen as evaluation-only and construct training corpora from disjoint data sources or source groups.
