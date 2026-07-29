from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .common import N_EXAMPLES, SEEDS, balanced_sample, base_record, finalize_task, stable_sample
from .download import USER_AGENT, download
from .static_generators import _place_name, natural_earth


def _cached_json(url: str, path: Path, refresh: bool = False) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_YEAR = 2023
POPULATION_DENSITY_YEARS = (2000, 2005, 2010, 2015, 2018, 2020, 2021, 2022, 2023)


def _world_bank_countries(cache: Path) -> dict[str, dict[str, Any]]:
    url = f"{WORLD_BANK_BASE}/country?format=json&per_page=400"
    payload = _cached_json(url, cache / "world_bank" / "countries.json")
    rows = payload[1]
    return {
        row["id"]: row
        for row in rows
        if row.get("region", {}).get("id") and row.get("capitalCity") is not None
    }


def _world_bank_indicator(cache: Path, indicator: str, year: int = WORLD_BANK_YEAR) -> dict[str, dict[str, Any]]:
    url = f"{WORLD_BANK_BASE}/country/all/indicator/{indicator}?format=json&per_page=400&date={year}"
    payload = _cached_json(url, cache / "world_bank" / f"{indicator}_{year}.json")
    rows = payload[1]
    return {row["countryiso3code"]: row for row in rows if row.get("countryiso3code") and row.get("value") is not None}


def generate_population_density(cache: Path, output: Path) -> Path:
    leaf = "population_density_estimation"
    countries = _world_bank_countries(cache)
    groups: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for year in POPULATION_DENSITY_YEARS:
        table = _world_bank_indicator(cache, "EN.POP.DNST", year)
        groups[str(year)] = [
            (code, countries[code], table[code])
            for code in sorted(set(countries) & set(table))
        ]

    selected = balanced_sample(
        groups,
        N_EXAMPLES,
        SEEDS[leaf],
        key=lambda item: item[0],
    )
    records: list[dict[str, Any]] = []
    year_counts: dict[int, int] = {year: 0 for year in POPULATION_DENSITY_YEARS}
    for i, (year_text, item) in enumerate(selected):
        code, country, row = item
        year = int(year_text)
        name = country["name"]
        value = float(row["value"])
        year_counts[year] += 1
        record = base_record(
            leaf,
            i,
            "World Development Indicators: EN.POP.DNST",
            "https://data.worldbank.org/indicator/EN.POP.DNST",
            "CC-BY-4.0",
            f"{code}:{year}",
        )
        record.update(
            {
                "input": {
                    "question": (
                        f"What was the population density of {name} in {year}, "
                        "in people per square kilometre of land area?"
                    ),
                    "country": name,
                    "country_code": code,
                    "year": year,
                    "indicator": "EN.POP.DNST",
                },
                "target": {
                    "value": round(value, 6),
                    "unit": "people per sq. km of land area",
                    "year": year,
                    "indicator": "EN.POP.DNST",
                },
                "evaluation": {
                    "type": "numeric",
                    "relative_tolerance": 0.005,
                    "unit": "people per sq. km of land area",
                },
            }
        )
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {
            "years": list(POPULATION_DENSITY_YEARS),
            "year_distribution": {str(year): year_counts[year] for year in POPULATION_DENSITY_YEARS},
            "indicator": "EN.POP.DNST",
        },
    )


COMPARISON_INDICATORS = {
    "EN.POP.DNST": ("population density", "people per sq. km"),
    "SP.POP.TOTL": ("total population", "people"),
    "NY.GDP.PCAP.CD": ("GDP per capita", "current US dollars"),
    "AG.LND.FRST.ZS": ("forest area share", "percent of land area"),
}
COMPARISON_YEARS = (2000, 2005, 2010, 2015, 2018, 2020, 2021, 2022, 2023)


def generate_cross_entity_comparison(cache: Path, output: Path) -> Path:
    leaf = "cross_entity_comparison"
    countries = _world_bank_countries(cache)
    tables = {
        (indicator, year): _world_bank_indicator(cache, indicator, year)
        for indicator in COMPARISON_INDICATORS
        for year in COMPARISON_YEARS
    }
    grouped_candidates: dict[str, list[tuple[str, int, str, str]]] = {}
    for (indicator, year), table in tables.items():
        codes = sorted(set(countries) & set(table))
        pairs: list[tuple[str, int, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for offset, code in enumerate(codes):
            if len(codes) < 2:
                continue
            other = codes[(offset * 37 + year + 17) % len(codes)]
            if code == other:
                other = codes[(codes.index(code) + 1) % len(codes)]
            pair = tuple(sorted((code, other)))
            if pair in seen:
                continue
            value_a = float(table[pair[0]]["value"])
            value_b = float(table[pair[1]]["value"])
            if value_a == value_b:
                continue
            seen.add(pair)
            pairs.append((indicator, year, pair[0], pair[1]))
        grouped_candidates[f"{indicator}:{year}"] = pairs

    from .common import balanced_sample

    selected = balanced_sample(
        grouped_candidates,
        N_EXAMPLES,
        SEEDS[leaf],
        key=lambda x: f"{x[0]}:{x[1]}:{x[2]}:{x[3]}",
    )
    templates = (
        ("higher", lambda metric, year, a, b: f"Which had the higher {metric} in {year}: {a} or {b}?"),
        ("lower", lambda metric, year, a, b: f"Which had the lower {metric} in {year}: {a} or {b}?"),
        ("higher", lambda metric, year, a, b: f"Compare {a} and {b}. Which country recorded a larger {metric} value in {year}?"),
        ("lower", lambda metric, year, a, b: f"For {year}, which country had the smaller {metric}: {a} or {b}?"),
    )
    records: list[dict[str, Any]] = []
    for i, (_, (indicator, year, code_a, code_b)) in enumerate(selected):
        table = tables[(indicator, year)]
        name_a, name_b = countries[code_a]["name"], countries[code_b]["name"]
        value_a = float(table[code_a]["value"])
        value_b = float(table[code_b]["value"])
        metric_name, unit = COMPARISON_INDICATORS[indicator]
        relation, template = templates[i % len(templates)]
        if relation == "higher":
            answer = name_a if value_a > value_b else name_b
        else:
            answer = name_a if value_a < value_b else name_b
        difference = abs(value_a - value_b)
        ratio = max(abs(value_a), abs(value_b)) / max(min(abs(value_a), abs(value_b)), 1e-12)
        record = base_record(
            leaf,
            i,
            f"World Development Indicators: {indicator}",
            f"https://api.worldbank.org/v2/indicator/{indicator}",
            "CC-BY-4.0",
            f"{indicator}:{year}:{code_a}:{code_b}",
        )
        record.update(
            {
                "input": {
                    "question": template(metric_name, year, name_a, name_b),
                    "indicator": indicator,
                    "indicator_name": metric_name,
                    "year": year,
                    "entities": [name_a, name_b],
                    "comparison_direction": relation,
                },
                "target": {
                    "answer": answer,
                    "relation": relation,
                    "values": {name_a: value_a, name_b: value_b},
                    "absolute_difference": difference,
                    "larger_to_smaller_ratio": ratio,
                    "unit": unit,
                    "year": year,
                    "indicator": indicator,
                },
                "evaluation": {"type": "comparison", "metrics": ["accuracy", "value_consistency"]},
            }
        )
        records.append(record)
    return finalize_task(
        output,
        leaf,
        records,
        {
            "years": sorted({r["target"]["year"] for r in records}),
            "indicators": sorted({r["target"]["indicator"] for r in records}),
        },
    )


def _macrostrat_data(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        success = payload.get("success")
        if isinstance(success, dict) and isinstance(success.get("data"), list):
            return success["data"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    return []


def generate_geology(cache: Path, output: Path) -> Path:
    leaf = "geologic_geomorphic_interpretation"
    _, places = natural_earth(cache)
    rows = [row for _, row in places.iterrows() if row.geometry and not row.geometry.is_empty]
    rng = random.Random(SEEDS[leaf])
    rows = sorted(rows, key=lambda row: f"{_place_name(row)}:{row.geometry.x:.6f}")
    rng.shuffle(rows)
    selected: list[tuple[Any, dict[str, Any]]] = []
    api_cache = cache / "macrostrat"
    for row in rows:
        if len(selected) == N_EXAMPLES:
            break
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        params = urllib.parse.urlencode({"lat": f"{lat:.6f}", "lng": f"{lon:.6f}"})
        url = f"https://macrostrat.org/api/v2/geologic_units/map?{params}"
        key = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
        try:
            payload = _cached_json(url, api_cache / f"{key}.json")
        except Exception:
            continue
        data = _macrostrat_data(payload)
        if data:
            selected.append((row, data[0]))
            time.sleep(0.05)
    if len(selected) != N_EXAMPLES:
        raise ValueError(f"Macrostrat returned only {len(selected)} usable locations")
    records: list[dict[str, Any]] = []
    for i, (row, unit) in enumerate(selected):
        lon, lat = float(row.geometry.x), float(row.geometry.y)
        name = unit.get("name") or unit.get("strat_name") or "unnamed geologic unit"
        lithology = unit.get("lith") or unit.get("lithology") or unit.get("descrip")
        age = unit.get("age") or unit.get("b_int_name") or unit.get("t_int_name")
        record = base_record(
            leaf,
            i,
            "Macrostrat geologic map API v2",
            "https://macrostrat.org/api/v2/geologic_units/map",
            "Macrostrat terms; underlying map-source licenses vary and must be retained",
            str(unit.get("map_id", f"{lat:.4f}:{lon:.4f}")),
        )
        record.update(
            {
                "input": {
                    "question": f"Identify and interpret the mapped geologic unit at {lat:.5f}, {lon:.5f} near {_place_name(row)}.",
                    "coordinate": {"longitude": lon, "latitude": lat, "crs": "EPSG:4326"},
                },
                "target": {"unit_name": name, "lithology": lithology, "age": age, "raw_unit": unit},
                "evaluation": {"type": "structured_generation", "required_fields": ["unit_name", "lithology", "age"]},
            }
        )
        records.append(record)
    return finalize_task(output, leaf, records)


# Two images are selected near each fixed city, yielding exactly 100 entries.
WIKIMEDIA_CITIES = [
    ("Zurich", "Switzerland", 47.3769, 8.5417), ("Paris", "France", 48.8566, 2.3522),
    ("Rome", "Italy", 41.9028, 12.4964), ("Oslo", "Norway", 59.9139, 10.7522),
    ("Reykjavik", "Iceland", 64.1466, -21.9426), ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Athens", "Greece", 37.9838, 23.7275), ("Istanbul", "Türkiye", 41.0082, 28.9784),
    ("Marrakesh", "Morocco", 31.6295, -7.9811), ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Nairobi", "Kenya", -1.2921, 36.8219), ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Accra", "Ghana", 5.6037, -0.1870), ("Dakar", "Senegal", 14.7167, -17.4677),
    ("Tunis", "Tunisia", 36.8065, 10.1815), ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Tehran", "Iran", 35.6892, 51.3890), ("Mumbai", "India", 19.0760, 72.8777),
    ("Kathmandu", "Nepal", 27.7172, 85.3240), ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Singapore", "Singapore", 1.3521, 103.8198), ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780), ("Taipei", "Taiwan", 25.0330, 121.5654),
    ("Jakarta", "Indonesia", -6.2088, 106.8456), ("Sydney", "Australia", -33.8688, 151.2093),
    ("Auckland", "New Zealand", -36.8509, 174.7645), ("Honolulu", "United States", 21.3069, -157.8583),
    ("Vancouver", "Canada", 49.2827, -123.1207), ("San Francisco", "United States", 37.7749, -122.4194),
    ("Mexico City", "Mexico", 19.4326, -99.1332), ("Havana", "Cuba", 23.1136, -82.3666),
    ("New York", "United States", 40.7128, -74.0060), ("Quebec City", "Canada", 46.8139, -71.2080),
    ("Lima", "Peru", -12.0464, -77.0428), ("Quito", "Ecuador", -0.1807, -78.4678),
    ("Bogota", "Colombia", 4.7110, -74.0721), ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816), ("Santiago", "Chile", -33.4489, -70.6693),
    ("La Paz", "Bolivia", -16.4897, -68.1193), ("Cusco", "Peru", -13.5319, -71.9675),
    ("Jerusalem", "Israel", 31.7683, 35.2137), ("Tbilisi", "Georgia", 41.7151, 44.8271),
    ("Almaty", "Kazakhstan", 43.2220, 76.8512), ("Ulaanbaatar", "Mongolia", 47.8864, 106.9057),
    ("Hanoi", "Vietnam", 21.0278, 105.8342), ("Manila", "Philippines", 14.5995, 120.9842),
    ("Kuala Lumpur", "Malaysia", 3.1390, 101.6869), ("Edinburgh", "United Kingdom", 55.9533, -3.1883),
]


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return str(value.get("value")) if isinstance(value, dict) and value.get("value") is not None else None


def _strip_metadata(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        if im.mode not in {"RGB", "L"}:
            im = im.convert("RGB")
        im.save(destination, format="JPEG", quality=92, optimize=True)


def generate_visual_geolocation(cache: Path, output: Path) -> Path:
    leaf = "visual_geolocation"
    task_dir = output / leaf
    records: list[dict[str, Any]] = []
    api = "https://commons.wikimedia.org/w/api.php"
    for city_index, (city, country, lat, lon) in enumerate(WIKIMEDIA_CITIES):
        params = {
            "action": "query", "format": "json", "generator": "geosearch",
            "ggsprimary": "all", "ggsnamespace": 6, "ggsradius": 10000,
            "ggscoord": f"{lat}|{lon}", "ggslimit": 50,
            "prop": "coordinates|imageinfo", "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1280,
        }
        url = api + "?" + urllib.parse.urlencode(params)
        payload = _cached_json(url, cache / "wikimedia" / f"{city_index:02d}_{city.replace(' ', '_')}.json")
        candidates: list[dict[str, Any]] = []
        for page in payload.get("query", {}).get("pages", {}).values():
            infos = page.get("imageinfo") or []
            coords = page.get("coordinates") or []
            if not infos or not coords:
                continue
            info = infos[0]
            if info.get("mime") not in {"image/jpeg", "image/png"} or int(info.get("width", 0)) < 800:
                continue
            metadata = info.get("extmetadata") or {}
            license_name = _metadata_value(metadata, "LicenseShortName") or ""
            if not any(token in license_name.lower() for token in ("cc0", "public domain", "cc by")):
                continue
            candidates.append({"page": page, "info": info, "coord": coords[0], "license": license_name, "metadata": metadata})
        selected = stable_sample(candidates, 2, SEEDS[leaf] + city_index, key=lambda x: str(x["page"].get("pageid")))
        for item in selected:
            index = len(records)
            info = item["info"]
            source_url = info.get("thumburl") or info["url"]
            downloaded = download(source_url, cache / "wikimedia" / "originals" / f"{index:03d}{Path(urllib.parse.urlparse(source_url).path).suffix or '.jpg'}")
            clean_path = task_dir / "assets" / f"{index:03d}.jpg"
            _strip_metadata(downloaded, clean_path)
            image_lat = float(item["coord"]["lat"])
            image_lon = float(item["coord"]["lon"])
            record = base_record(
                leaf,
                index,
                "Wikimedia Commons geocoded media",
                "https://commons.wikimedia.org/wiki/Commons:API/MediaWiki",
                item["license"],
                f"{city}:{item['page'].get('pageid')}",
            )
            record.update(
                {
                    "input": {
                        "images": [clean_path.relative_to(task_dir).as_posix()],
                        "question": "Where was this image captured? Return city, country, and approximate coordinates.",
                    },
                    "target": {"city": city, "country": country, "latitude": image_lat, "longitude": image_lon},
                    "attribution": {
                        "commons_title": item["page"].get("title"),
                        "description_url": info.get("descriptionurl"),
                        "artist": _metadata_value(item["metadata"], "Artist"),
                        "credit": _metadata_value(item["metadata"], "Credit"),
                        "license_url": _metadata_value(item["metadata"], "LicenseUrl"),
                    },
                    "evaluation": {"type": "geolocation", "metrics": ["country_accuracy", "city_accuracy", "haversine_km"]},
                }
            )
            records.append(record)
    return finalize_task(output, leaf, records)
