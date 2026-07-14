# GeoMapBench taxonomy data kit

This package builds **exactly 100 deterministic examples for each of 23 leaves** in the revised GeoMapBench taxonomy. It does not bundle multi-gigabyte upstream imagery or silently relicense it. Instead, it supplies official source links, download helpers, fixed per-leaf seeds, source-specific samplers, public-API generators, cached raw responses, provenance, and validation.

## Taxonomy decisions

No original capability was removed. One over-broad leaf was corrected:

- `Map Text Extraction and Label Anchoring` became two measurable leaves:
  - **Map Text Detection, Recognition, and Grouping**, using real word polygons and groups.
  - **Map Label-to-Feature Anchoring**, generated from named OpenStreetMap features with exact geometries.

Three high-value, well-supported gaps were added:

- **Remote-Sensing Scene Classification** (EuroSAT).
- **Object Presence and Counting** (RSVQA-LR).
- **Visual Geolocation** (license-filtered, geocoded Wikimedia Commons images).

The complete source-to-leaf mapping is in [`config/sources.csv`](config/sources.csv). The revised tree is in [`taxonomy_updated.tex`](taxonomy_updated.tex).

## Reproducibility contract

- Every command produces exactly 100 JSONL records or fails loudly.
- Every leaf has a unique immutable seed in `geomapbench_data/common.py`.
- Candidate collections are sorted before seeded sampling.
- API responses and OSMnx queries are cached. Do not delete the cache after freezing a benchmark release.
- Every task directory contains `data.jsonl`, `manifest.json`, a SHA-256 checksum, source/license metadata, and any local assets.
- A fixed seed controls selection, **not changes in a live upstream database**. For a paper release, archive the raw cache, record retrieval dates, and publish its hashes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The examples below assume:

```bash
RAW=$PWD/data/raw
CACHE=$PWD/data/cache
OUT=$PWD/data/geomapbench_100
mkdir -p "$RAW" "$CACHE" "$OUT"
```

## Official public sources and build commands

### Small and medium downloads

The built-in fetcher resolves current file URLs through the official Zenodo API and extracts archives without re-downloading cached files.

```bash
geomapbench-data fetch maptext --raw-root "$RAW"
geomapbench-data fetch eurosat --raw-root "$RAW"
geomapbench-data fetch rsvqa --raw-root "$RAW"

git clone https://github.com/mapbox/maki "$RAW/maki"
git clone https://github.com/milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation "$RAW/geowebnews"
git clone https://github.com/HLR/SpaRTUN "$RAW/spartun"
git clone https://github.com/AI-team-UoA/GeoQuestions1089 "$RAW/geoquestions"
```

Build the corresponding leaves:

```bash
geomapbench-data maki --source "$RAW/maki" --output "$OUT"
geomapbench-data maptext --source "$RAW/maptext" --output "$OUT"
geomapbench-data eurosat --source "$RAW/eurosat" --output "$OUT"
geomapbench-data rsvqa --source "$RAW/rsvqa" --output "$OUT"
geomapbench-data geowebnews --source "$RAW/geowebnews" --output "$OUT"
geomapbench-data spartun --source "$RAW/spartun" --output "$OUT"
geomapbench-data geoquestions --source "$RAW/geoquestions" --output "$OUT"
```

SpaRTUN’s repository links to the released SpaRTUN/ReSQ files; place those JSON files anywhere below `$RAW/spartun` before sampling.

### Large imagery sources

Download [OpenEarthMap](https://doi.org/10.5281/zenodo.7223446) from its official record, preserve its per-region attribution file, then run:

```bash
geomapbench-data openearthmap --source "$RAW/openearthmap" --output "$OUT"
```

SpaceNet is available as anonymous AWS Open Data. The following prefixes contain the required releases:

```bash
aws s3 sync --no-sign-request s3://spacenet-dataset/spacenet/SN3_roads/ "$RAW/spacenet3/"
aws s3 cp --no-sign-request \
  s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz \
  "$RAW/spacenet7/SN7_buildings_train.tar.gz"
tar -xzf "$RAW/spacenet7/SN7_buildings_train.tar.gz" -C "$RAW/spacenet7"
```

Then build four leaves:

```bash
geomapbench-data spacenet3-graph --source "$RAW/spacenet3" --output "$OUT"
geomapbench-data spacenet3-route --source "$RAW/spacenet3" --output "$OUT"
geomapbench-data spacenet7-change --source "$RAW/spacenet7" --output "$OUT"
geomapbench-data spacenet7-match --source "$RAW/spacenet7" --output "$OUT"
```

### Deterministic public-data/API generators

```bash
geomapbench-data coordinate-transform --cache "$CACHE" --output "$OUT"
geomapbench-data metric-distance --cache "$CACHE" --output "$OUT"
geomapbench-data topology-direction --cache "$CACHE" --output "$OUT"
geomapbench-data geo-entity-typing --cache "$CACHE" --output "$OUT"
geomapbench-data population-density --cache "$CACHE" --output "$OUT"
geomapbench-data cross-entity-comparison --cache "$CACHE" --output "$OUT"
geomapbench-data geology --cache "$CACHE" --output "$OUT"
geomapbench-data visual-geolocation --cache "$CACHE" --output "$OUT"
geomapbench-data osm-label-anchoring --cache "$CACHE" --output "$OUT"
geomapbench-data osm-isochrone --cache "$CACHE" --output "$OUT"
```

For **Environmental Layer Identification**, download a historical GeoTIFF from the official [1-km Köppen–Geiger record](https://doi.org/10.5281/zenodo.5347837), unzip it, and pass the selected raster explicitly:

```bash
geomapbench-data environmental-layer \
  --cache "$CACHE" \
  --koppen-raster "$RAW/koppen/KGClim_1984_2013.tif" \
  --output "$OUT"
```

## Validate

During incremental work, validate the leaves already built:

```bash
geomapbench-data validate --root "$OUT"
```

For a release, require all 23 leaves and all referenced assets:

```bash
geomapbench-data validate --root "$OUT" --require-all
```

## Output record

Each line follows `schema/entry.schema.json` and includes:

- stable `id`, `leaf`, `seed`, and leakage-control `group_id`;
- source name, official URL, and conservative license statement;
- multimodal input fields and an exact/structured target;
- task-appropriate evaluation metadata;
- attribution metadata for per-image Wikimedia licenses.

## Benchmark/SFT separation

Do not randomly split individual questions after generating Bloom variants. Split by `group_id` (AOI, map sheet, document, country pair, or source object) first, and keep every derivative of that group in one split. For your proposed system:

1. Freeze the 100 examples per leaf as **GeoMapBench evaluation only**.
2. Create SFT data from different source groups, spatial areas, dates, and preferably different upstream datasets.
3. Deduplicate text and images perceptually before training.
4. Keep RAG corpora separate from benchmark answers and reference GeoSPARQL queries.
5. Apply the Bloom template only after the source split; otherwise one geographic fact can leak across Bloom levels.

## License cautions

- OpenEarthMap inherits different imagery licenses by region. The generator records a conservative notice, but you must retain its attribution table and verify each chosen region before redistribution.
- OpenStreetMap-derived databases require attribution and may trigger ODbL share-alike duties. Generated map images include attribution.
- Macrostrat integrates many map providers; retain each returned source identifier and its original terms.
- GeoWebNews is openly released in a GPL repository, but underlying corpus notices still need review before republishing text.
- This package is an engineering aid, not legal advice.

