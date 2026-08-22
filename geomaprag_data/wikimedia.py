from __future__ import annotations

import html
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, make_record, slugify
from .config import BuildProfile, CAPABILITY_HINTS, select_places
from .http import CachedHTTP


COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return None
    text = html.unescape(str(value))
    return text.strip() or None


def _allowed_license(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("cc0", "public domain", "cc by", "cc-by", "cc by-sa", "cc-by-sa"))


def _save_clean_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        max_side = 1280
        if max(rgb.size) > max_side:
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb.save(temporary, format="JPEG", quality=92, optimize=True)
    temporary.replace(destination)


def build_wikimedia(
    workspace: CorpusWorkspace,
    profile: BuildProfile,
    guard: BenchmarkGuard,
    *,
    checkpoint_every: int = 15,
) -> dict[str, Any]:
    http = CachedHTTP(workspace.cache_dir)
    seeds = select_places(profile.wikimedia_seed_count, seed=7331)
    existing = workspace.existing_ids()
    failures: list[dict[str, str]] = []
    written = 0
    cached = 0
    bar = tqdm(total=len(seeds), desc="Wikimedia geocoded images", unit="region", dynamic_ncols=True)

    for index, (city, country, lat, lon) in enumerate(seeds, 1):
        unit = f"{index:03d}_{country}_{city}"
        if workspace.shard_done("wikimedia", unit):
            cached += 1
            bar.set_postfix(city=city, status="cached")
            bar.update(1)
            continue
        try:
            payload = http.get_json(
                COMMONS_API,
                {
                    "action": "query",
                    "format": "json",
                    "generator": "geosearch",
                    "ggsprimary": "all",
                    "ggsnamespace": 6,
                    "ggsradius": 10000,
                    "ggscoord": f"{lat}|{lon}",
                    "ggslimit": 60,
                    "prop": "coordinates|imageinfo",
                    "iiprop": "url|mime|size|extmetadata",
                    "iiurlwidth": 1280,
                    "maxlag": 5,
                },
                f"wikimedia/geosearch/{country}/{city}",
                timeout=150,
                max_attempts=10,
            )
            pages = list((payload.get("query", {}).get("pages") or {}).values())
            candidates: list[dict[str, Any]] = []
            for page in pages:
                page_id = str(page.get("pageid") or "")
                if not page_id or page_id in guard.commons_page_ids:
                    continue
                infos = page.get("imageinfo") or []
                coords = page.get("coordinates") or []
                if not infos or not coords:
                    continue
                info = infos[0]
                if info.get("mime") not in {"image/jpeg", "image/png"} or int(info.get("width", 0)) < 800:
                    continue
                metadata = info.get("extmetadata") or {}
                license_name = _metadata_value(metadata, "LicenseShortName") or ""
                if not _allowed_license(license_name):
                    continue
                try:
                    image_lat = float(coords[0]["lat"])
                    image_lon = float(coords[0]["lon"])
                except Exception:
                    continue
                if guard.near(image_lat, image_lon):
                    continue
                candidates.append(
                    {
                        "page": page,
                        "info": info,
                        "coord": {"lat": image_lat, "lon": image_lon},
                        "license": license_name,
                        "metadata": metadata,
                    }
                )

            candidates.sort(key=lambda item: int(item["page"].get("pageid") or 0))
            rng = random.Random(880_000 + index)
            rng.shuffle(candidates)
            selected = candidates[: profile.wikimedia_images_per_seed]
            records: list[dict[str, Any]] = []
            for item in selected:
                page_id = str(item["page"]["pageid"])
                record_id = f"wikimedia:{page_id}"
                if record_id in existing:
                    continue
                info = item["info"]
                source_url = str(info.get("thumburl") or info.get("url") or "")
                if not source_url:
                    continue
                suffix = Path(source_url.split("?", 1)[0]).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png"}:
                    suffix = ".jpg"
                downloaded = http.download(
                    source_url,
                    workspace.cache_dir / "wikimedia" / "originals" / f"{page_id}{suffix}",
                    timeout=240,
                    max_attempts=6,
                )
                destination = workspace.root / "images" / "wikimedia" / f"{page_id}.jpg"
                if not destination.exists():
                    _save_clean_image(downloaded, destination)
                relative = destination.relative_to(workspace.root).as_posix()
                metadata = item["metadata"]
                title = str(item["page"].get("title") or f"Wikimedia Commons {page_id}")
                description = _metadata_value(metadata, "ImageDescription") or title
                # Strip simple HTML tags from Commons metadata while retaining a useful caption.
                import re

                description = re.sub(r"<[^>]+>", " ", description)
                description = " ".join(description.split())[:1200]
                image_lat = item["coord"]["lat"]
                image_lon = item["coord"]["lon"]
                text = (
                    f"Geocoded Wikimedia Commons image near {city}, {country}. "
                    f"Coordinates: latitude {image_lat:.6f}, longitude {image_lon:.6f}. "
                    f"Description: {description}"
                )
                records.append(
                    make_record(
                        record_id=record_id,
                        source_name="Wikimedia Commons",
                        source_url=str(info.get("descriptionurl") or source_url),
                        license_name=item["license"],
                        attribution=_metadata_value(metadata, "Artist") or "Wikimedia Commons contributor",
                        group_id=f"{city}:{page_id}",
                        modality="geo_image",
                        title=title,
                        text=text,
                        source_id=page_id,
                        geo={"lat": image_lat, "lon": image_lon, "seed_city": city, "seed_country": country},
                        media_paths=[relative],
                        capabilities=CAPABILITY_HINTS["Wikimedia Commons"],
                        document_type="geocoded_ground_photo",
                        generator="geomaprag_data.wikimedia",
                        extra={
                            "commons_page_id": page_id,
                            "artist": _metadata_value(metadata, "Artist"),
                            "credit": _metadata_value(metadata, "Credit"),
                            "license_url": _metadata_value(metadata, "LicenseUrl"),
                            "mime": info.get("mime"),
                            "original_width": info.get("width"),
                            "original_height": info.get("height"),
                            "exif_stripped": True,
                        },
                    )
                )
            workspace.write_shard(
                "wikimedia",
                unit,
                records,
                meta={"status": "complete", "city": city, "country": country, "candidate_count": len(candidates)},
            )
            for record in records:
                existing.add(record["id"])
            written += len(records)
            bar.set_postfix(city=city, images=len(records), status="ok")
        except Exception as error:
            failures.append({"unit": unit, "city": city, "error": repr(error)})
            print(f"\nWikimedia warning {city}: {error!r}")
            bar.set_postfix(city=city, status="failed")
        finally:
            bar.update(1)
        if checkpoint_every and index % checkpoint_every == 0:
            workspace.materialize()
    bar.close()
    return {"stage": "wikimedia", "written": written, "cached_units": cached, "failed_units": failures}
