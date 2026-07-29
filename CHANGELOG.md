# Changelog

## 1.2.0 — July 2026 additional data-quality fixes

Five additional leaves must be refreshed when upgrading from 1.1.0:

- `metric_distance_computation`
- `population_density_estimation`
- `shortest_path_optimization`
- `spatial_graph_construction`
- `topological_directional_reasoning`

### Corrections

- Balanced metric-distance answers across metres, kilometres, statute miles, and nautical miles while retaining canonical metre/kilometre targets for auditability.
- Balanced population-density examples across nine World Bank years: 2000, 2005, 2010, 2015, 2018, 2020, 2021, 2022, and 2023.
- Added percentile-stretched 8-bit RGB SpaceNet inputs so notebook and browser viewers do not display high-bit-depth TIFFs as black.
- Added road-graph and shortest-route overlays over the satellite imagery.
- Replaced ambiguous point markers in topological/directional reasoning with filled polygon A/B regions and a two-scale within-region visualization.
- Added an in-place SpaceNet visual migration that reuses existing task assets and does not require redownloading SpaceNet 3.

## 1.1.0 — July 2026 benchmark-quality revision

The following eight leaves must be regenerated when upgrading from 1.0.0:

- `coordinate_transformation`
- `cross_entity_comparison`
- `dense_land_cover_labeling`
- `environmental_layer_identification`
- `geo_entity_typing`
- `isochrone_service_area`
- `map_label_feature_anchoring`
- `map_text_detection_recognition_grouping`

All other leaves are unchanged and should not be rebuilt solely for this upgrade.

### Corrections

- Added six reversible CRS transformation modes with explicit axis order, units, and round-trip checks.
- Added nine reference years, four indicators, and higher/lower variants to cross-entity comparison.
- Decoded official OpenEarthMap RGB label colors into single-channel class-index masks and added strict image/mask validation.
- Replaced the one-layer environmental task with balanced WorldClim elevation and bioclimatic layer identification.
- Added the complete GeoNames feature-class ontology to every geo-entity-typing example.
- Expanded OSM tasks to 20 globally distributed cities.
- Replaced convex-hull isochrones with buffered reachable street edges and varied speed/time parameters.
- Removed candidate feature names from anchoring images and placed only the target label at its real anchor position.
- Rebuilt map-text records as crop-level text detection, recognition, and grouping examples with at least two groups.
- Added task-specific release validation and `data_revision = 2026-07-comments-v2` to manifests.
