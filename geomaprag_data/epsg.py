from __future__ import annotations

from typing import Any

from pyproj import CRS
from pyproj.database import query_crs_info
from tqdm.auto import tqdm

from .benchmark_guard import BenchmarkGuard
from .common import CorpusWorkspace, make_record
from .config import BuildProfile, CAPABILITY_HINTS


def build_epsg(workspace: CorpusWorkspace, profile: BuildProfile, guard: BenchmarkGuard) -> dict[str, Any]:
    unit = f"epsg_{profile.epsg_max_records}"
    if workspace.shard_done("epsg", unit):
        return {"stage": "epsg", "written": 0, "cached_units": 1, "failed_units": []}

    infos = sorted(query_crs_info(auth_name="EPSG"), key=lambda info: int(info.code))[: profile.epsg_max_records]
    existing = workspace.existing_ids()
    records: list[dict[str, Any]] = []
    bar = tqdm(infos, desc="EPSG / PROJ", unit="CRS", dynamic_ncols=True)
    for info in bar:
        record_id = f"epsg:{info.code}"
        if record_id in existing:
            continue
        try:
            crs = CRS.from_authority(info.auth_name, info.code)
            axes = [
                {
                    "name": axis.name,
                    "abbrev": axis.abbrev,
                    "direction": axis.direction,
                    "unit": axis.unit_name,
                }
                for axis in crs.axis_info
            ]
            area = crs.area_of_use
            datum_name = getattr(getattr(crs, "datum", None), "name", None)
            coordinate_system_name = getattr(getattr(crs, "coordinate_system", None), "name", None)
            axis_text = "; ".join(
                f"{axis['name']} ({axis['direction']}, unit={axis['unit']})" for axis in axes
            )
            text = (
                f"EPSG:{info.code} is the coordinate reference system {crs.name}. "
                f"CRS type: {crs.type_name}. Axes: {axis_text or 'unspecified'}. "
                f"Datum: {datum_name or 'unspecified'}. Coordinate system: {coordinate_system_name or 'unspecified'}."
            )
            if area:
                text += f" Area of use: {area.name}."
            records.append(
                make_record(
                    record_id=record_id,
                    source_name="EPSG / PROJ",
                    source_url=f"https://epsg.io/{info.code}",
                    license_name="EPSG dataset terms / PROJ database redistribution terms",
                    attribution="EPSG Geodetic Parameter Dataset and PROJ",
                    group_id=f"EPSG:{info.code}",
                    modality="structured",
                    title=f"EPSG:{info.code} {crs.name}",
                    text=text,
                    source_id=f"EPSG:{info.code}",
                    capabilities=CAPABILITY_HINTS["EPSG / PROJ"],
                    document_type="coordinate_reference_system",
                    generator="geomaprag_data.epsg",
                    extra={
                        "authority": info.auth_name,
                        "code": str(info.code),
                        "crs_type": crs.type_name,
                        "axes": axes,
                        "area_of_use": None
                        if area is None
                        else {
                            "name": area.name,
                            "west": area.west,
                            "south": area.south,
                            "east": area.east,
                            "north": area.north,
                        },
                        "datum": datum_name,
                    },
                )
            )
        except Exception as error:
            print(f"EPSG warning {info.code}: {error!r}")

    workspace.write_shard("epsg", unit, records, meta={"status": "complete", "requested": len(infos)})
    return {"stage": "epsg", "written": len(records), "cached_units": 0, "failed_units": []}
