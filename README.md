# GeoMapBench taxonomy data kit

## Final cumulative evaluation workflow (v2.1.0)

The final release includes two orchestration-only Colab notebooks:

- `notebooks/GeoMapBench_Evaluation_Final.ipynb`: six multimodal models from six independent companies, including the economical Llama 4 Scout baseline; Muse is removed.
- `notebooks/GeoMapBench_RAG_Final.ipynb`: independent Claude dense `base_rag` and `agentic_rag` runs.

Version 2.1 adds offline task-aware rescoring and publication figures. Existing
raw responses are migrated into `model_suite_final` and rescored without API
calls. Only legacy file-artifact rows are incompatible: their old prompt asked
for a filename rather than a measurable prediction. Those rows alone receive a
new inline RLE-mask or graph-JSON prompt. Invalid JSON and genuine token-limit
failures remain final failures and are never selectively retried.

The RAG pipeline is dense-only. It uses the corpus `text.faiss` index built with
`BAAI/bge-small-en-v1.5`, optionally reranks candidates with
`BAAI/bge-reranker-base`, and does not use BM25.

Both suites default to a 16,384-token output budget. Reasoning is disabled for
models where the current catalog permits it and set to the minimum/low supported
effort for mandatory-reasoning Gemini models. Response rows record `finish_reason`,
reasoning-token usage, empty responses, and token-limit generation failures. The evaluator loads only the 23
canonical benchmark leaves, chooses pilot subsets before resume filtering, and
converts SVG/TIFF assets to supported PNG payloads inside the codebase.

Preflight is persistent and content-aware. Pass the same Drive-backed directory
with `--preflight-cache`: a successful preflight is skipped when the canonical
JSONL files, referenced assets, prompt revision, and image converter revision are
unchanged. Use `--force-preflight` after changing benchmark data.

Both suites use a deterministic nested Bloom-stratified cohort. Keep one output
directory and increase `--target-per-leaf`: target 1 contains 23 records, target
6 contains 138 total records, and target 100 is the complete 2,300-record run.
Increasing 1 to 6 runs only the 115 missing records. The first canonical record
is retained for compatibility with the old one-per-leaf pilot, then the selector
covers all supported Bloom levels before repeating levels.

Every condition writes `run_state.json`, `inflight.json`, and a write-ahead
`api_responses.jsonl` cache. A response received immediately before a runtime
disconnect is recovered without another API call. Long exponential retry with
jitter handles transient failures; persistent rate limits open a circuit breaker
and pause the run so it can be resumed safely later. Partial runs are marked
`complete=false` and are never promoted to final accuracy.

After every invocation, append-safe stores are compacted to one canonical row
per ID. The final analysis keeps strict accuracy, adds one normalized
task-aware score plus at most two task-specific diagnostics per leaf, and writes
Seaborn PNG/PDF figures with bold labels, no figure title, and one shared legend
below each multi-series plot. See `METRICS_PROTOCOL.md`.

The RAG notebook never launches a no-retrieval Base and has no dependency on an
Evaluation output. Both notebooks load the same model matrix and save exact model
parameters, protocol revisions, selected-record content hashes, and completed IDs.
They can therefore run concurrently on different computers. The comparison
command later refuses to compare outputs unless all fairness fields and IDs match.

This package builds **exactly 100 deterministic examples for each of 23 leaves** in the revised GeoMapBench taxonomy. It does not bundle multi-gigabyte upstream imagery or silently relicense it. Instead, it supplies official source links, download helpers, fixed per-leaf seeds, source-specific samplers, public-API generators, cached raw responses, provenance, and validation.


## August 2026 Bloom-balanced benchmark overlay

Release `1.4.0` retains the Bloom-balanced conversion and robust canonical-leaf validation from 1.3.x, and adds the large resume-safe GeoMapRAG corpus pipeline described below. It adds an optional deterministic Bloom-taxonomy conversion for an already-built 23-leaf benchmark. It keeps every leaf at exactly 100 examples, reuses the existing source records/assets, and balances each leaf across all defensible Bloom levels from the current annotations. Five-level leaves contain 20 examples per level; six-level leaves contain 17/17/17/17/16/16 examples.

The conversion is intentionally lightweight: it rewrites only `data.jsonl` and `manifest.json`, preserves all original target fields, adds `target.bloom_answer`, and stores the original two metadata files under `.pre_bloom/`. It does not redownload OpenEarthMap, SpaceNet, MapText, or any other upstream source.

```bash
geomapbench-data validate --root "$OUT" --require-all
geomapbench-data bloomify --root "$OUT"
geomapbench-data validate --root "$OUT" --require-all
geomapbench-data bloom-audit --root "$OUT"
```

Restore the pre-Bloom metadata if needed:

```bash
geomapbench-data bloom-restore --root "$OUT"
```

See [`BLOOM_EXPANSION.md`](BLOOM_EXPANSION.md) and [`config/bloom_taxonomy.csv`](config/bloom_taxonomy.csv) for the exact per-leaf levels, prompts, targets, and balancing policy. The ready-to-run Colab is [`notebooks/GeoMapBench_Bloom_Update_Colab.ipynb`](notebooks/GeoMapBench_Bloom_Update_Colab.ipynb).

## July 2026 quality revisions

Release `1.2.0` includes the original eight `1.1.0` corrections plus five additional benchmark-quality fixes:

- Coordinate transformation now balances six reversible transformation modes instead of always starting in WGS 84 and ending in local UTM.
- Cross-entity comparison spans nine years, four World Bank indicators, and both higher/lower comparisons.
- OpenEarthMap RGB annotations are decoded into validated single-channel class-index masks with an explicit eight-class ontology.
- Environmental layer identification is a balanced six-way WorldClim raster-identification task rather than a one-layer Köppen-only task.
- GeoNames feature-class codes and definitions are included in every geo-entity-typing prompt.
- OSM isochrones cover 20 globally distributed cities, multiple time budgets and walking speeds, and use buffered reachable street edges instead of a convex hull.
- Map-label anchoring hides candidate names and displays the target label at its actual anchor position over four nearby candidate geometries.
- Map-text examples require detection, transcription, and grouping of every visible word in a crop, with at least two label groups per example.
- Metric-distance questions are balanced across metres, kilometres, statute miles, and nautical miles instead of always requesting kilometres.
- Population-density questions span nine World Bank reference years from 2000 through 2023.
- SpaceNet 3 graph and route inputs are converted from high-bit-depth TIFFs to contrast-stretched RGB PNGs; graph and route overlays are rendered over the satellite image.
- Topological/directional reasoning uses clearly filled polygon regions rather than ambiguous point markers.

The validator enforces regeneration-specific revision checks for the twelve currently strict revised leaves. The accepted legacy `isochrone_service_area` leaf still receives normal record/count/checksum/asset/Bloom validation, but is exempt from the later isochrone-regeneration diversity assertions. Revised manifests use `data_revision = "2026-07-comments-v2"` where applicable.

The original eight-task Colab remains at [`notebooks/GeoMapBench_fixed_tasks_colab.ipynb`](notebooks/GeoMapBench_fixed_tasks_colab.ipynb). A second notebook, [`notebooks/GeoMapBench_additional_fixes_colab.ipynb`](notebooks/GeoMapBench_additional_fixes_colab.ipynb), rebuilds the three lightweight public-data tasks and upgrades the two existing SpaceNet folders in place without redownloading SpaceNet.

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

For **Environmental Layer Identification**, the generator automatically downloads the official 10-arc-minute WorldClim 2.1 bioclimatic and elevation archives and builds a balanced six-layer task:

```bash
geomapbench-data environmental-layer \
  --cache "$CACHE" \
  --output "$OUT"
```

`--koppen-raster` remains accepted only as a deprecated compatibility argument and is not needed by the revised task.

## Rebuild the original eight revised leaves

Existing unaffected task directories do not need regeneration:

```bash
geomapbench-data maptext --source "$RAW/maptext" --output "$OUT"
geomapbench-data openearthmap --source "$RAW/openearthmap" --output "$OUT"
geomapbench-data coordinate-transform --cache "$CACHE" --output "$OUT"
geomapbench-data geo-entity-typing --cache "$CACHE" --output "$OUT"
geomapbench-data cross-entity-comparison --cache "$CACHE" --output "$OUT"
geomapbench-data environmental-layer --cache "$CACHE" --output "$OUT"
geomapbench-data osm-label-anchoring --cache "$CACHE" --output "$OUT"
geomapbench-data osm-isochrone --cache "$CACHE" --output "$OUT"
```


## Apply only the five additional 1.2.0 fixes

The metric-distance, population-density, and topology/direction tasks can be regenerated from lightweight cached public sources:

```bash
geomapbench-data metric-distance --cache "$CACHE" --output "$OUT"
geomapbench-data population-density --cache "$CACHE" --output "$OUT"
geomapbench-data topology-direction --cache "$CACHE" --output "$OUT"
```

For existing SpaceNet 3 task folders, `upgrade_existing_spacenet_visuals` reuses the copied TIFFs already stored in each task and converts them to display-ready PNGs while producing satellite graph/route overlays. The additional-fixes Colab performs this operation through a validated staging directory, so the multi-gigabyte source release is not downloaded again.

## Validate

During incremental work, validate the leaves already built:

```bash
geomapbench-data validate --root "$OUT"
```

For a release, require all 23 leaves and all referenced assets:

```bash
geomapbench-data validate --root "$OUT" --require-all
```

Validation is keyed to the canonical 23 leaf names in `geomapbench_data.common.SEEDS`. Noncanonical sibling directories (for example Google Drive copies ending in ` (1)`) are ignored rather than treated as benchmark leaves. The accepted legacy `isochrone_service_area` release is validated normally but is exempt from the newer regeneration-specific diversity checks.

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


---

## Large GeoMapRAG retrieval corpus

This repository also contains `geomaprag_data/`, a large, resume-safe retrieval-corpus builder designed to complement GeoMapBench evaluation. The full design is documented in [`GEOMAPRAG_DATA.md`](GEOMAPRAG_DATA.md), and the numbered Colab runner is [`notebooks/GeoMapRAG_Corpus_NUMBERED.ipynb`](notebooks/GeoMapRAG_Corpus_NUMBERED.ipynb).

The `publication` profile builds a fresh corpus from Wikipedia, Wikidata/QLever, GeoNames, World Bank, EPSG/PROJ, Wikimedia Commons, and OpenStreetMap. Existing materialized corpus files are never imported as source records. Every completed network/source unit is committed as an atomic shard, so interrupted Colab runs resume without discarding completed work. A benchmark-overlap guard filters exact text/source identifiers and nearby coordinates before RAG records are admitted.

```bash
geomaprag-data build \
  --output /content/drive/MyDrive/GeoMapRAG_Corpus \
  --benchmark-root /content/drive/MyDrive/GeoMapBench_Data/geomapbench_100 \
  --profile publication

geomaprag-data validate \
  --root /content/drive/MyDrive/GeoMapRAG_Corpus \
  --profile publication \
  --strict-scale

geomaprag-data clean \
  --root /content/drive/MyDrive/GeoMapRAG_Corpus \
  --overwrite
```

The benchmark cleaner lives in `geomapbench_data/clean_data.py` and is available through:

```bash
geomapbench-data clean --root /path/to/geomapbench_100 --overwrite
```

## Run GeoMapBench with OpenRouter

`geomapbench_eval` is a resumable inference/evaluation harness. It never sends
gold targets, source metadata, seeds, group IDs, or provenance to a model. It
uses only the task input and the input assets, records raw model responses plus
usage/latency, and scores locally. Keep your API key out of notebooks and Git:

```powershell
$env:OPENROUTER_API_KEY = "..."
geomapbench-data validate --root C:\data\geomapbench_100 --require-all
geomapbench-data bloom-audit --root C:\data\geomapbench_100
geomapbench-data clean --root C:\data\geomapbench_100

# Cumulative model suite. Reuse the same output and increase the target.
geomapbench-eval suite `
  --benchmark-root C:\data\geomapbench_100 `
  --models config\evaluation_models_final.json `
  --output results\model_suite_final `
  --preflight-cache cache\preflight `
  --target-per-leaf 50 `
  --legacy-output results\model_suite_cumulative_v7 `
  --delete-legacy-after-success

# Independent dense-only RAG pilot. It needs no Base/Evaluation output.
geomapbench-eval rag-suite `
  --benchmark-root C:\data\geomapbench_100 `
  --corpus-root C:\data\GeoMapRAG_Corpus `
  --work-root C:\temp\geomaprag `
  --output results\rag_suite_final `
  --models config\evaluation_models_final.json `
  --model anthropic/claude-sonnet-5 `
  --agent-cache cache\agent_v191 `
  --preflight-cache cache\preflight `
  --conditions base_rag,agentic_rag `
  --target-per-leaf 6

geomapbench-eval analyze `
  --results results\model_suite_final\anthropic_claude-sonnet-5\responses.jsonl `
  --output results\model_suite_final\anthropic_claude-sonnet-5\analysis `
  --benchmark-root C:\data\geomapbench_100

# Later, after transferring the independently completed folders to one machine:
geomapbench-eval compare `
  --base-results results\model_suite_final\anthropic_claude-sonnet-5\responses.jsonl `
  --rag-results results\rag_suite_final\agentic_rag\responses.jsonl `
  --output results\claude_agentic_comparison
```

The analysis command writes `per_leaf.csv`, `task_metrics_long.csv`, and
`summary.json`. Suite commands additionally write publication-ready Seaborn
plots as both 300-DPI PNG and vector PDF. The final RAG conditions use BGE dense retrieval and
cross-encoder reranking; `agentic_rag` adds a cached planner/judge loop. BM25 is
not used. Inspect retrieved IDs and retrieval metrics before attributing answer
changes to RAG. The command
writes a per-leaf CSV, macro averages, Bloom-level summary, format-failure
rate, bootstrap confidence intervals, latency, and OpenRouter-reported cost.

For a paper release, keep the cumulative output directory fixed, pin the exact
model ID, store the API response metadata, and do not use
`--force` on a completed output directory: it intentionally reruns records and
appends another immutable response row. Use `--no-images` only for a declared
text-only ablation, never as the default comparison for visual leaves.
