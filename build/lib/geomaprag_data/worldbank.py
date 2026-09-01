from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, make_record
from .config import BuildProfile, CAPABILITY_HINTS, WORLD_BANK_INDICATORS
from .http import CachedHTTP


WORLD_BANK_BASE = "https://api.worldbank.org/v2"


def _country_metadata(http: CachedHTTP) -> dict[str, dict[str, Any]]:
    payload = http.get_json(
        f"{WORLD_BANK_BASE}/country",
        {"format": "json", "per_page": 400},
        "worldbank/countries",
        timeout=120,
        max_attempts=8,
    )
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    countries: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        iso3 = str(row.get("id") or "").upper()
        region = str((row.get("region") or {}).get("value") or "")
        # WDI includes aggregates (World, income groups, regions). They are
        # useful analytically but are not geographic entities comparable to
        # countries, so the RAG corpus intentionally keeps real economies only.
        if not iso3 or region.lower() == "aggregates":
            continue
        countries[iso3] = row
    return countries


def build_worldbank(
    workspace: CorpusWorkspace,
    profile: BuildProfile,
    guard: BenchmarkGuard,
    *,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    http = CachedHTTP(workspace.cache_dir)
    countries = _country_metadata(http)
    failures: list[dict[str, str]] = []
    written = 0
    cached = 0
    excluded_benchmark_observations = 0
    units = [(indicator, year) for indicator in sorted(WORLD_BANK_INDICATORS) for year in profile.worldbank_years]
    existing = workspace.existing_ids()

    bar = tqdm(total=len(units), desc="World Bank indicators", unit="query", dynamic_ncols=True)
    for unit_index, (indicator, year) in enumerate(units, 1):
        unit = f"{indicator}_{year}"
        if workspace.shard_done("worldbank", unit):
            cached += 1
            bar.set_postfix(indicator=indicator, year=year, status="cached")
            bar.update(1)
            continue
        try:
            url = f"{WORLD_BANK_BASE}/country/all/indicator/{indicator}"
            payload = http.get_json(
                url,
                {"format": "json", "per_page": 400, "date": year},
                f"worldbank/{indicator}/{year}",
                timeout=120,
                max_attempts=8,
            )
            rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            label, unit_name = WORLD_BANK_INDICATORS[indicator]
            records: list[dict[str, Any]] = []
            excluded_here = 0
            for row in rows or []:
                iso3 = str(row.get("countryiso3code") or "").upper()
                value = row.get("value")
                if not iso3 or iso3 not in countries or value is None:
                    continue
                observation_key = f"{indicator}:{year}:{iso3}"
                if observation_key in guard.worldbank_observations:
                    excluded_here += 1
                    continue
                record_id = f"worldbank:{observation_key}"
                if record_id in existing:
                    continue
                meta = countries[iso3]
                country = str((row.get("country") or {}).get("value") or meta.get("name") or iso3)
                region = str((meta.get("region") or {}).get("value") or "")
                income_level = str((meta.get("incomeLevel") or {}).get("value") or "")
                lending_type = str((meta.get("lendingType") or {}).get("value") or "")
                text = f"In {year}, {country} had {label} of {value} {unit_name}. World Bank indicator: {indicator}."
                if region:
                    text += f" World Bank region: {region}."
                if income_level:
                    text += f" Income classification: {income_level}."
                if guard.reject_text(text):
                    continue
                records.append(
                    make_record(
                        record_id=record_id,
                        source_name="World Bank",
                        source_url=f"https://data.worldbank.org/indicator/{indicator}",
                        license_name="CC BY 4.0",
                        attribution="World Bank World Development Indicators",
                        group_id=observation_key,
                        modality="structured",
                        title=f"{country} — {label}, {year}",
                        text=text,
                        source_id=observation_key,
                        capabilities=CAPABILITY_HINTS["World Bank"],
                        document_type="country_indicator_observation",
                        generator="geomaprag_data.worldbank",
                        extra={
                            "indicator": indicator,
                            "indicator_label": label,
                            "year": year,
                            "country": country,
                            "country_iso3": iso3,
                            "region": region or None,
                            "income_level": income_level or None,
                            "lending_type": lending_type or None,
                            "value": value,
                            "unit": unit_name,
                            "provider_longitude": meta.get("longitude") or None,
                            "provider_latitude": meta.get("latitude") or None,
                        },
                    )
                )
            records.sort(key=lambda record: str(record["id"]))
            workspace.write_shard(
                "worldbank",
                unit,
                records,
                meta={
                    "status": "complete",
                    "indicator": indicator,
                    "year": year,
                    "benchmark_observations_excluded": excluded_here,
                },
            )
            for record in records:
                existing.add(record["id"])
            written += len(records)
            excluded_benchmark_observations += excluded_here
            bar.set_postfix(indicator=indicator, year=year, new_docs=len(records), excluded=excluded_here)
        except Exception as error:
            failures.append({"unit": unit, "error": repr(error)})
            print(f"\nWorld Bank warning {unit}: {error!r}")
            bar.set_postfix(indicator=indicator, year=year, status="failed")
        finally:
            bar.update(1)

        if checkpoint_every and unit_index % checkpoint_every == 0:
            workspace.materialize()
    bar.close()
    return {
        "stage": "worldbank",
        "written": written,
        "cached_units": cached,
        "benchmark_observations_excluded": excluded_benchmark_observations,
        "failed_units": failures,
    }
