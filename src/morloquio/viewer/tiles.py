"""Tile XYZ (PNG 256 px, EPSG:3857) da uno o più raster del run, per il visualizzatore locale.

Più raster (le tile della griglia, con alone) si mosaicano "primo valido"; se si passano i nuclei delle tile
(``cores``, poligoni nel CRS del raster) ogni raster contribuisce solo dentro il proprio nucleo, così gli aloni
non coprono i dati della tile vicina.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import mercantile
import numpy as np
import rasterio
import shapely
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

TILE_PX = 256
_EPS_DEG = 1e-7  # restringe il bbox: una tile che tocca solo il bordo non va generata

Mode = Literal["gray", "heat", "rgb"]


def _stretch(datasets, mode: Mode) -> tuple[float, float]:
    """Intervallo di valori per il grigio: percentili 2–98 su una versione ridotta dei raster."""
    if mode != "gray":
        return 0.0, 1.0
    samples = []
    for ds in datasets:
        scale = max(1, max(ds.width, ds.height) // 512)
        arr = ds.read(1, out_shape=(max(1, ds.height // scale), max(1, ds.width // scale)), masked=True)
        vals = arr.astype("float64").filled(np.nan)  # anche raster interi (hillshade uint8)
        samples.append(vals[~np.isnan(vals)])
    vals = np.concatenate(samples) if samples else np.empty(0)
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [2, 98])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _colorize(acc: np.ndarray, valid: np.ndarray, mode: Mode, stretch: tuple[float, float]) -> np.ndarray:
    rgba = np.zeros((TILE_PX, TILE_PX, 4), dtype="uint8")
    if mode == "rgb":
        rgba[..., :3] = np.clip(np.nan_to_num(acc), 0, 255).transpose(1, 2, 0).astype("uint8")
        rgba[..., 3] = np.where(valid, 255, 0)
        return rgba
    v = np.nan_to_num(acc[0])
    if mode == "gray":
        lo, hi = stretch
        g = (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
        rgba[..., 3] = np.where(valid, 255, 0)
        return rgba
    # heat: punteggio 0–1, trasparente sotto 0,15, da giallo (basso) a rosso (alto)
    t = np.clip(v, 0, 1)
    rgba[..., 0] = 255
    rgba[..., 1] = (230 * (1 - t) ** 0.7).astype("uint8")
    rgba[..., 2] = (40 * (1 - t)).astype("uint8")
    alpha = np.clip((t - 0.15) / 0.35, 0, 1) * 235
    rgba[..., 3] = np.where(valid, alpha, 0).astype("uint8")
    return rgba


def render_xyz(
    raster: Path | Sequence[Path],
    out_dir: Path,
    zooms: range = range(12, 19),
    mode: Mode = "gray",
    cores: Sequence | None = None,
) -> int:
    """Scrive ``out_dir/{z}/{x}/{y}.png`` per le tile con almeno un pixel valido; ritorna quante ne ha scritte."""
    paths = [Path(raster)] if isinstance(raster, (str, Path)) else [Path(p) for p in raster]
    if not paths:
        return 0
    if cores is not None and len(cores) != len(paths):
        raise ValueError("cores deve avere un poligono per raster")
    out_dir = Path(out_dir)
    datasets = [rasterio.open(p) for p in paths]
    try:
        bands = 3 if mode == "rgb" else 1
        merc = [transform_bounds(ds.crs, "EPSG:3857", *ds.bounds, densify_pts=21) for ds in datasets]
        lonlat = np.array([transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21) for ds in datasets])
        w, s = lonlat[:, 0].min() + _EPS_DEG, lonlat[:, 1].min() + _EPS_DEG
        e, n = lonlat[:, 2].max() - _EPS_DEG, lonlat[:, 3].max() - _EPS_DEG
        cores_3857 = None
        if cores is not None:
            cores_3857 = [
                shapely.transform(
                    c, Transformer.from_crs(ds.crs, "EPSG:3857", always_xy=True).transform, interleaved=False
                )
                for c, ds in zip(cores, datasets, strict=True)
            ]
        masks = [ds.read_masks(1) if mode == "rgb" else None for ds in datasets]
        stretch = _stretch(datasets, mode)
        count = 0
        for tile in mercantile.tiles(w, s, e, n, list(zooms)):
            xb = mercantile.xy_bounds(tile)
            dst_transform = from_bounds(xb.left, xb.bottom, xb.right, xb.top, TILE_PX, TILE_PX)
            acc = np.full((bands, TILE_PX, TILE_PX), np.nan, dtype="float32")
            for k, ds in enumerate(datasets):
                left, bottom, right, top = merc[k]
                if right <= xb.left or left >= xb.right or top <= xb.bottom or bottom >= xb.top:
                    continue
                tmp = np.full((bands, TILE_PX, TILE_PX), np.nan, dtype="float32")
                for b in range(bands):
                    reproject(
                        source=rasterio.band(ds, b + 1),
                        destination=tmp[b],
                        src_nodata=ds.nodata,
                        dst_transform=dst_transform,
                        dst_crs="EPSG:3857",
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                if masks[k] is not None:
                    m = np.zeros((TILE_PX, TILE_PX), dtype="uint8")
                    reproject(
                        source=masks[k],
                        destination=m,
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        dst_transform=dst_transform,
                        dst_crs="EPSG:3857",
                        resampling=Resampling.nearest,
                    )
                    tmp[:, m < 128] = np.nan
                if cores_3857 is not None:
                    inside = geometry_mask(
                        [cores_3857[k]], out_shape=(TILE_PX, TILE_PX), transform=dst_transform, invert=True
                    )
                    tmp[:, ~inside] = np.nan
                fill = np.isnan(acc) & ~np.isnan(tmp)
                acc[fill] = tmp[fill]
            valid = ~np.isnan(acc).any(axis=0)
            if not valid.any():
                continue
            path = out_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_colorize(acc, valid, mode, stretch), "RGBA").save(path, optimize=True)
            count += 1
        return count
    finally:
        for ds in datasets:
            ds.close()
