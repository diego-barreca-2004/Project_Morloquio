"""Tracce GPS pubbliche di OpenStreetMap (API 0.6 ``trackpoints``), paginate da 5000 punti.

L'API accetta bbox di al massimo 0,25 gradi quadrati: i bbox più grandi vengono divisi in celle.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from morloquio.common.http import FileRecord, PoliteClient
from morloquio.common.manifest import update_manifest
from morloquio.common.paths import raw_dir
from morloquio.common.runlog import get_logger
from morloquio.sources.catalog import SourceSpec

SOURCE_ID = "osm_gps_traces"
_DONE = "_complete"

log = get_logger("sources.osm_gps")


def split_bbox(bbox, max_area_deg2: float = 0.25) -> list[tuple[float, float, float, float]]:
    west, south, east, north = bbox
    if (east - west) * (north - south) <= max_area_deg2:
        return [tuple(bbox)]
    side = math.sqrt(max_area_deg2)
    nx, ny = math.ceil((east - west) / side), math.ceil((north - south) / side)
    dx, dy = (east - west) / nx, (north - south) / ny
    return [
        (west + i * dx, south + j * dy, west + (i + 1) * dx, south + (j + 1) * dy) for j in range(ny) for i in range(nx)
    ]


def _bbox_key(bbox) -> str:
    return "_".join(f"{v:.4f}" for v in bbox)


def fetch_trackpoints(client: PoliteClient, spec: SourceSpec, bbox) -> list[Path]:
    page_size = int(spec.params["page_size"])
    paths: list[Path] = []
    for part in split_bbox(bbox, float(spec.params["max_area_deg2"])):
        directory = raw_dir(SOURCE_ID) / _bbox_key(part)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / _DONE).exists():
            paths.extend(sorted(directory.glob("page_*.gpx"), key=lambda p: int(p.stem.split("_")[1])))
            continue
        page = 0
        while True:
            params = {"bbox": ",".join(f"{v:.6f}" for v in part), "page": str(page)}
            response = client.request("GET", spec.params["api_url"], params=params)
            text = response.text
            path = directory / f"page_{page}.gpx"
            path.write_text(text, encoding="utf-8")
            data = text.encode("utf-8")
            record = FileRecord(
                path=path,
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
                url=str(response.url),
                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            update_manifest(directory, path.name, record, license=spec.license)
            paths.append(path)
            if text.count("<trkpt") < page_size:
                break
            page += 1
        (directory / _DONE).write_text("ok\n")
        log.info("tracce GPS scaricate", extra={"bbox": list(part), "pages": page + 1})
    return paths


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def load_points(paths) -> gpd.GeoDataFrame:
    """Punti GPS da file GPX: colonne ``track`` (file:indice traccia), ``seq`` e geometria EPSG:4326."""
    rows = []
    for path in paths:
        root = ET.parse(path).getroot()
        tracks = [el for el in root.iter() if _local(el.tag) == "trk"]
        for t_idx, trk in enumerate(tracks):
            seq = 0
            for pt in trk.iter():
                if _local(pt.tag) != "trkpt":
                    continue
                rows.append(
                    {
                        "track": f"{Path(path).stem}:{t_idx}",
                        "seq": seq,
                        "geometry": Point(float(pt.get("lon")), float(pt.get("lat"))),
                    }
                )
                seq += 1
    return gpd.GeoDataFrame(rows, columns=["track", "seq", "geometry"], geometry="geometry", crs="EPSG:4326")
