from __future__ import annotations

from collections import Counter
from typing import Any

from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, chunk_text, make_record, slugify
from .config import BuildProfile, CAPABILITY_HINTS, select_places
from .http import CachedHTTP


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _geosearch(http: CachedHTTP, city: str, lat: float, lon: float, limit: int) -> list[dict[str, Any]]:
    payload = http.get_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "format": "json",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": 10000,
            "gslimit": limit,
            "gsnamespace": 0,
            "maxlag": 5,
        },
        f"wikipedia/geosearch/{slugify(city)}",
        timeout=120,
        max_attempts=12,
    )
    return payload.get("query", {}).get("geosearch", [])


def _pages(http: CachedHTTP, city: str, page_ids: list[int], batch_size: int = 50) -> list[dict[str, Any]]:
    """Fetch page extracts in deterministic API-safe batches.

    MediaWiki limits the number of page IDs accepted by a normal client in one
    request. Batching lets the ICLR profile retrieve substantially broader
    geographic context without relying on privileged API limits.
    """
    unique_ids = sorted(set(int(page_id) for page_id in page_ids))
    pages: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(unique_ids), batch_size), 1):
        batch = unique_ids[start : start + batch_size]
        payload = http.get_json(
            WIKIPEDIA_API,
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "extracts|coordinates|info|pageprops",
                "explaintext": 1,
                "inprop": "url",
                "pageids": "|".join(str(x) for x in batch),
                "maxlag": 5,
            },
            f"wikipedia/pages/{slugify(city)}/batch-{batch_index:02d}",
            timeout=150,
            max_attempts=12,
        )
        pages.extend(payload.get("query", {}).get("pages", []))
    return pages


def build_wikipedia(
    workspace: CorpusWorkspace,
    profile: BuildProfile,
    guard: BenchmarkGuard,
    *,
    checkpoint_every: int = 20,
) -> dict[str, Any]:
    http = CachedHTTP(workspace.cache_dir)
    seeds = select_places(profile.wikipedia_seed_count)
    failures: list[dict[str, str]] = []
    written = 0
    skipped = 0
    global_ids = workspace.existing_ids()
    estimated_chunks = 0

    bar = tqdm(total=len(seeds), desc="Wikipedia regions", unit="region", dynamic_ncols=True)
    for index, (city, country, lat, lon) in enumerate(seeds, 1):
        unit = f"{index:03d}_{country}_{city}"
        if workspace.shard_done("wikipedia", unit):
            skipped += 1
            meta = workspace.shard_meta_path("wikipedia", unit)
            try:
                import json

                count = int(json.loads(meta.read_text(encoding="utf-8")).get("count", 0))
            except Exception:
                count = 0
            estimated_chunks += count
            bar.set_postfix(city=city, status="cached", corpus_chunks=estimated_chunks)
            bar.update(1)
            continue

        if guard.near(lat, lon):
            workspace.write_shard(
                "wikipedia",
                unit,
                [],
                meta={"status": "benchmark_spatial_exclusion", "city": city, "country": country},
            )
            bar.set_postfix(city=city, status="excluded")
            bar.update(1)
            continue

        try:
            hits = _geosearch(http, city, lat, lon, profile.wikipedia_pages_per_seed)
            page_ids = [int(hit["pageid"]) for hit in hits if hit.get("pageid") is not None]
            pages = _pages(http, city, page_ids)
            pages = sorted(pages, key=lambda page: int(page.get("pageid") or 0))
            records: list[dict[str, Any]] = []
            for page in pages:
                if page.get("missing") or "disambiguation" in (page.get("pageprops") or {}):
                    continue
                page_id = str(page.get("pageid"))
                extract = str(page.get("extract") or "")
                if len(extract) < 250:
                    continue
                coordinates = (page.get("coordinates") or [{}])[0]
                page_lat = coordinates.get("lat", lat)
                page_lon = coordinates.get("lon", lon)
                if guard.near(page_lat, page_lon):
                    continue
                for chunk_index, chunk in enumerate(chunk_text(extract, target_words=300, overlap_words=45)):
                    if guard.reject_text(chunk):
                        continue
                    record_id = f"wikipedia:{page_id}:{chunk_index}"
                    if record_id in global_ids:
                        continue
                    records.append(
                        make_record(
                            record_id=record_id,
                            source_name="Wikipedia",
                            source_url=str(page.get("fullurl") or "https://en.wikipedia.org/"),
                            license_name="CC BY-SA 4.0 / GFDL; retain upstream attribution",
                            attribution="Wikipedia contributors",
                            group_id=page_id,
                            modality="text",
                            title=str(page.get("title") or page_id),
                            text=chunk,
                            source_id=page_id,
                            geo={
                                "lat": page_lat,
                                "lon": page_lon,
                                "seed_city": city,
                                "seed_country": country,
                            },
                            capabilities=CAPABILITY_HINTS["Wikipedia"],
                            document_type="encyclopedic_geographic_context",
                            generator="geomaprag_data.wikipedia",
                            extra={"page_id": page_id, "chunk_index": chunk_index},
                        )
                    )
            workspace.write_shard(
                "wikipedia",
                unit,
                records,
                meta={"status": "complete", "city": city, "country": country, "page_count": len(pages)},
            )
            for record in records:
                global_ids.add(record["id"])
            written += len(records)
            estimated_chunks += len(records)
            bar.set_postfix(city=city, new_docs=len(records), corpus_chunks=estimated_chunks)
        except Exception as error:
            failures.append({"unit": unit, "city": city, "error": repr(error)})
            bar.set_postfix(city=city, status="failed")
            print(f"\nWikipedia warning: {city}: {error!r}")
        finally:
            bar.update(1)

        if checkpoint_every and index % checkpoint_every == 0:
            workspace.materialize()
    bar.close()
    return {
        "stage": "wikipedia",
        "written": written,
        "cached_units": skipped,
        "failed_units": failures,
        "target_chunks": profile.wikipedia_target_chunks,
    }
