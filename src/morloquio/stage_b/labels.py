"""Etichette per pixel dalle vie OSM, trattate come positive-unlabeled.

OSM è incompleto (molti sentieri e piste mancano) e spostato di qualche metro. Quindi: positivo solo il corridoio
attorno alla centerline (semilarghezza per classe), peso 0 nella fascia di incertezza subito fuori, e i pixel
lontani da OSM sono "non etichettati" con peso basso, non negativi certi (Remote Sensing 18(10):1589; ISPRS
XLIX-B1-2026). Peso 0 anche fuori dal nucleo della tile (l'alone può essere il nucleo di una tile di test) e dove
manca il DTM.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def _half_width(highway, table: dict) -> float:
    return float(table.get(str(highway), table["default"]))


def _burn(geoms: list[BaseGeometry], transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)


def rasterize_labels(
    ways: gpd.GeoDataFrame,
    transform: Affine,
    shape: tuple[int, int],
    core: BaseGeometry,
    valid: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """``(target, weight)``: target uint8 0/1, weight float32 (1 positivi, 0 incerti o esclusi, basso altrove)."""
    table = params["half_width_m"]
    band = float(params["ignore_band_m"])
    positive, near = [], []
    for highway, geom in zip(ways["highway"], ways.geometry, strict=True):
        if geom is None or geom.is_empty:
            continue
        w = _half_width(highway, table)
        positive.append(geom.buffer(w))
        near.append(geom.buffer(w + band))
    pos = _burn(positive, transform, shape)
    close = _burn(near, transform, shape)
    inside = _burn([core], transform, shape) & np.asarray(valid, bool)
    weight = np.full(shape, float(params["unlabeled_weight"]), dtype="float32")
    weight[close & ~pos] = 0.0
    weight[pos] = 1.0
    weight[~inside] = 0.0
    return pos.astype("uint8"), weight


def build_labels(tile_dir: Path, ways: gpd.GeoDataFrame, params: dict, core: BaseGeometry | None = None) -> Path:
    """Scrive ``target.npy``, ``weight.npy`` e ``labels.json`` accanto ai canali della tile (stessa griglia)."""
    from morloquio.candidates.vectorize import core_polygon

    tile_dir = Path(tile_dir)
    meta = json.loads((tile_dir / "meta.json").read_text())
    transform = Affine(*meta["transform"])
    shape = tuple(meta["shape"])
    x = np.load(tile_dir / "features.npy", mmap_mode="r")
    valid = np.asarray(x[meta["channels"].index("valid")]) > 0.5
    if core is None:
        core = core_polygon(meta["tile_id"], meta["crs"])
    west, north = transform.c, transform.f
    bounds = box(west, north + transform.e * shape[0], west + transform.a * shape[1], north).buffer(20.0)
    local = ways.to_crs(meta["crs"])
    local = local[local.intersects(bounds)]
    target, weight = rasterize_labels(local, transform, shape, core, valid, params)
    np.save(tile_dir / "target.npy", target)
    np.save(tile_dir / "weight.npy", weight)
    info = {"params": params, "ways": int(len(local))}
    (tile_dir / "labels.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return tile_dir
