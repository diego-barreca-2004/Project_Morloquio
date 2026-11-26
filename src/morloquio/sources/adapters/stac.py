"""Cataloghi STAC aperti (Earth Search): Sentinel-2 L2A e Copernicus DEM, letti a finestra dai COG.

Si scarica solo la finestra dell'AOI, nel CRS nativo dell'asset: niente scene intere.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from morloquio.common.http import USER_AGENT
from morloquio.common.paths import raw_dir
from morloquio.common.runlog import get_logger
from morloquio.sources.catalog import SourceSpec

log = get_logger("sources.stac")

_GDAL_ENV = {
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def select_items(items: list[dict], max_cloud: float | None, max_items: int) -> list[dict]:
    """Meno nuvole prima, a parità di nuvole il più recente; scarta oltre ``max_cloud``."""

    def cloud(item):
        return item.get("properties", {}).get("eo:cloud_cover")

    kept = [i for i in items if max_cloud is None or cloud(i) is None or cloud(i) <= max_cloud]
    by_date = sorted(kept, key=lambda i: i.get("properties", {}).get("datetime") or "", reverse=True)
    by_cloud = sorted(by_date, key=lambda i: cloud(i) if cloud(i) is not None else 0.0)
    return by_cloud[:max_items]


def search_items(
    spec: SourceSpec,
    collection: str,
    bbox,
    datetime: str | None,
    max_cloud: float | None,
    max_items: int,
) -> list[dict]:
    client = Client.open(spec.params["stac_url"], headers={"User-Agent": USER_AGENT})
    query = {"eo:cloud_cover": {"lte": max_cloud}} if max_cloud is not None else None
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=datetime,
        query=query,
        max_items=max(50, max_items * 5),
    )
    return select_items(list(search.items_as_dicts()), max_cloud, max_items)


def fetch_window(href: str, bbox, dest: Path) -> Path:
    """Legge la finestra del bbox (EPSG:4326) dal raster ``href`` e la salva come GeoTIFF nel CRS nativo."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with rasterio.Env(**_GDAL_ENV), rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=int(window.width),
            height=int(window.height),
            transform=src.window_transform(window),
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(part, "w", **profile) as out:
            out.write(data)
    part.replace(dest)
    return dest


def fetch_collection(spec: SourceSpec, bbox) -> list[Path]:
    """Cerca gli item della collezione del catalogo e scarica la finestra AOI di ogni asset richiesto."""
    p = spec.params
    items = search_items(
        spec, p["collection"], bbox, p.get("datetime"), p.get("max_cloud"), int(p.get("max_items", 10))
    )
    paths = []
    for item in items:
        for asset in p["assets"]:
            href = item["assets"][asset]["href"]
            dest = raw_dir(spec.id) / item["id"] / f"{asset}.tif"
            if not dest.exists():
                fetch_window(href, bbox, dest)
            paths.append(dest)
        log.info("item STAC scaricato", extra={"source": spec.id, "item": item["id"]})
    return paths
