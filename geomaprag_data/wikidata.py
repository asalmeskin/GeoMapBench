from __future__ import annotations

import re
from typing import Any

from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, make_record
from .config import BuildProfile, CAPABILITY_HINTS, WIKIDATA_FAMILIES
from .http import CachedHTTP


QLEVER_WIKIDATA_API = "https://qlever.dev/api/wikidata"
_POINT_RE = re.compile(r"Point\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)")


def _query_for(qid: str, limit: int) -> str:
    # Exact instance-of queries are intentionally used instead of P31/P279*
    # transitive expansion. They are much cheaper and more stable on public
    # endpoints while still yielding a broad entity corpus.
    return f"""
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item ?itemLabel ?countryLabel ?coord ?elevation WHERE {{
  ?item wdt:P31 wd:{qid};
        wdt:P625 ?coord;
        rdfs:label ?itemLabel.
  FILTER(LANG(?itemLabel) = "en")
  OPTIONAL {{
    ?item wdt:P17 ?country.
    ?country rdfs:label ?countryLabel.
    FILTER(LANG(?countryLabel) = "en")
  }}
  OPTIONAL {{ ?item wdt:P2044 ?elevation. }}
}}
ORDER BY ?item
LIMIT {int(limit)}
""".strip()


def _coord(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    match = _POINT_RE.search(value)
    if not match:
        return None, None
    try:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    except Exception:
        return None, None


def build_wikidata(
    workspace: CorpusWorkspace,
    profile: BuildProfile,
    guard: BenchmarkGuard,
    *,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    http = CachedHTTP(workspace.cache_dir)
    families = sorted(WIKIDATA_FAMILIES.items())
    existing = workspace.existing_ids()
    failures: list[dict[str, str]] = []
    written = 0
    cached = 0

    bar = tqdm(total=len(families), desc="Wikidata families", unit="family", dynamic_ncols=True)
    for family_index, (family, (qid, type_label)) in enumerate(families, 1):
        unit = f"{family}_{profile.wikidata_per_family}"
        if workspace.shard_done("wikidata", unit):
            cached += 1
            bar.set_postfix(family=family, status="cached")
            bar.update(1)
            continue
        try:
            query = _query_for(qid, profile.wikidata_per_family)
            payload = http.get_json(
                QLEVER_WIKIDATA_API,
                {"query": query},
                f"wikidata/qlever/{family}/{profile.wikidata_per_family}",
                timeout=300,
                max_attempts=5,
                extra_headers={"Accept": "application/sparql-results+json"},
            )
            bindings = payload.get("results", {}).get("bindings", [])
            records: list[dict[str, Any]] = []
            for binding in bindings:
                uri = (binding.get("item") or {}).get("value")
                if not uri:
                    continue
                item_qid = str(uri).rsplit("/", 1)[-1]
                record_id = f"wikidata:{item_qid}"
                if record_id in existing or item_qid in guard.qids:
                    continue
                title = str((binding.get("itemLabel") or {}).get("value") or item_qid)
                country = (binding.get("countryLabel") or {}).get("value")
                coord_text = (binding.get("coord") or {}).get("value")
                lat, lon = _coord(coord_text)
                if lat is not None and lon is not None and guard.near(lat, lon):
                    continue
                elevation = (binding.get("elevation") or {}).get("value")
                text = f"{title} is a Wikidata geographic entity classified as {type_label} ({qid})."
                if country:
                    text += f" Country or sovereign territory: {country}."
                if lat is not None and lon is not None:
                    text += f" Coordinates: latitude {lat:.6f}, longitude {lon:.6f}."
                if elevation is not None:
                    text += f" Elevation: {elevation} metres above sea level where stated by Wikidata."
                if guard.reject_text(text):
                    continue
                geo = None if lat is None or lon is None else {"lat": lat, "lon": lon}
                records.append(
                    make_record(
                        record_id=record_id,
                        source_name="Wikidata",
                        source_url=f"https://www.wikidata.org/wiki/{item_qid}",
                        license_name="CC0 1.0",
                        attribution="Wikidata contributors",
                        group_id=item_qid,
                        modality="structured",
                        title=title,
                        text=text,
                        source_id=item_qid,
                        geo=geo,
                        capabilities=CAPABILITY_HINTS["Wikidata"],
                        document_type=f"wikidata_{family}",
                        generator="geomaprag_data.wikidata",
                        extra={
                            "qid": item_qid,
                            "instance_of_qid": qid,
                            "instance_of_label": type_label,
                            "country": country,
                            "coordinate_literal": coord_text,
                            "elevation": elevation,
                        },
                    )
                )
            workspace.write_shard(
                "wikidata",
                unit,
                records,
                meta={"status": "complete", "family": family, "type_qid": qid, "bindings": len(bindings)},
            )
            for record in records:
                existing.add(record["id"])
            written += len(records)
            bar.set_postfix(family=family, new_docs=len(records))
        except Exception as error:
            failures.append({"unit": unit, "error": repr(error)})
            print(f"\nWikidata warning {family}: {error!r}")
            bar.set_postfix(family=family, status="failed")
        finally:
            bar.update(1)
        if checkpoint_every and family_index % checkpoint_every == 0:
            workspace.materialize()
    bar.close()
    return {"stage": "wikidata", "written": written, "cached_units": cached, "failed_units": failures}
