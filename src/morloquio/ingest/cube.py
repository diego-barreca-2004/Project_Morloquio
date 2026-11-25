"""Stadio 0 — cubo dati per tile: DTM, DSM first, DSM last (1 m) e ortofoto (0,5 m) nel CRS di lavoro.

Ogni tile ha un alone di ``HALO_M`` metri riempito dalle tavole vicine: i filtri degli stadi successivi
non vedono bordi artificiali. Combinazione delle sorgenti: "primo valido" in ordine di nome file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from morloquio.common.aoi import AOI
from morloquio.common.grid import neighbors, pst_tavola_for, tile_bounds
from morloquio.common.paths import processed_dir, raw_dir
from morloquio.common.runlog import get_logger

HALO_M = 64
LIDAR_LAYERS = {"dtm_1x1": "dtm", "dsm_first_1x1": "dsm_first", "dsm_last_1x1": "dsm_last"}
MIN_VALID_FOR_LIDAR = 0.01  # sotto l'1% di pixel validi nel nucleo la tile non ha LiDAR utile

_FLOAT_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "nodata": float("nan"),
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}

log = get_logger("ingest.cube")


def tile_grid(tile_id: str, work_crs: str, res_m: float, halo_m: float = HALO_M) -> tuple[Affine, int, int]:
    """Griglia della tile nel CRS di lavoro: limiti agganciati a multipli di ``res_m`` più l'alone."""
    w, s, e, n = transform_bounds("EPSG:4326", work_crs, *tile_bounds(tile_id), densify_pts=21)
    left = math.floor((w - halo_m) / res_m) * res_m
    bottom = math.floor((s - halo_m) / res_m) * res_m
    right = math.ceil((e + halo_m) / res_m) * res_m
    top = math.ceil((n + halo_m) / res_m) * res_m
    width = int(round((right - left) / res_m))
    height = int(round((top - bottom) / res_m))
    return from_origin(left, top, res_m, res_m), width, height


def tier_for(has_lidar: bool, has_ortho: bool) -> str:
    if has_lidar:
        return "T1"
    return "T2" if has_ortho else "T3"


def pst_sources_for(tile_id: str) -> dict[str, list[Path]]:
    """Tavole PST già scaricate per la tile e le 8 vicine, per prodotto."""
    names = {f"{pst_tavola_for(t)}.tif" for t in [tile_id, *neighbors(tile_id)]}
    out = {}
    for product in LIDAR_LAYERS:
        root = raw_dir("pst_lidar") / product
        out[product] = sorted(p for p in root.glob("*.tif") if p.name in names) if root.exists() else []
    return out


def _core_window(width: int, height: int, res_m: float, halo_m: float) -> tuple[slice, slice]:
    h = int(round(halo_m / res_m))
    return slice(h, height - h), slice(h, width - h)


def _mosaic(
    sources: list[Path],
    transform: Affine,
    width: int,
    height: int,
    crs: str,
    bands: int,
    resampling: Resampling,
) -> np.ndarray:
    dest = np.full((bands, height, width), np.nan, dtype="float32")
    for src_path in sorted(sources):
        with rasterio.open(src_path) as src:
            tmp = np.full((bands, height, width), np.nan, dtype="float32")
            for b in range(bands):
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=tmp[b],
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
            fill = np.isnan(dest) & ~np.isnan(tmp)
            dest[fill] = tmp[fill]
    return dest


def build_cube(tile_id: str, aoi: AOI, raw_lidar: dict[str, list[Path]], raw_ortho: list[Path]) -> Path:
    out = processed_dir("cube") / tile_id
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "tile_id": tile_id,
        "work_crs": aoi.work_crs,
        "halo_m": HALO_M,
        "valid_fraction": {},
        "sources": {},
    }

    transform, width, height = tile_grid(tile_id, aoi.work_crs, 1.0)
    rows, cols = _core_window(width, height, 1.0, HALO_M)
    for product, layer in LIDAR_LAYERS.items():
        sources = raw_lidar.get(product) or []
        path = out / f"{layer}.tif"
        if not sources:
            path.unlink(missing_ok=True)
            continue
        arr = _mosaic(sources, transform, width, height, aoi.work_crs, 1, Resampling.bilinear)[0]
        with rasterio.open(
            path, "w", width=width, height=height, crs=aoi.work_crs, transform=transform, **_FLOAT_PROFILE
        ) as ds:
            ds.write(arr, 1)
        meta["valid_fraction"][layer] = float(np.mean(~np.isnan(arr[rows, cols])))
        meta["sources"][layer] = [str(p) for p in sorted(sources)]

    has_ortho = False
    ortho_path = out / "ortho.tif"
    if raw_ortho:
        o_transform, o_width, o_height = tile_grid(tile_id, aoi.work_crs, 0.5)
        rgb = _mosaic(raw_ortho, o_transform, o_width, o_height, aoi.work_crs, 3, Resampling.cubic)
        valid = ~np.isnan(rgb).any(axis=0)
        data = np.where(np.isnan(rgb), 0, np.clip(rgb, 0, 255)).astype("uint8")
        with rasterio.open(
            ortho_path,
            "w",
            driver="GTiff",
            width=o_width,
            height=o_height,
            count=3,
            dtype="uint8",
            crs=aoi.work_crs,
            transform=o_transform,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as ds:
            ds.write(data)
            ds.write_mask(valid.astype("uint8") * 255)
        o_rows, o_cols = _core_window(o_width, o_height, 0.5, HALO_M)
        meta["valid_fraction"]["ortho"] = float(np.mean(valid[o_rows, o_cols]))
        meta["sources"]["ortho"] = [str(p) for p in sorted(raw_ortho)]
        has_ortho = meta["valid_fraction"]["ortho"] > MIN_VALID_FOR_LIDAR
    else:
        ortho_path.unlink(missing_ok=True)

    has_lidar = meta["valid_fraction"].get("dtm", 0.0) > MIN_VALID_FOR_LIDAR
    meta["tier"] = tier_for(has_lidar, has_ortho)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("cubo costruito", extra={"tile_id": tile_id, "tier": meta["tier"], **meta["valid_fraction"]})
    return out
