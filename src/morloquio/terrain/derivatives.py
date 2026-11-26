"""Derivati del terreno dal DTM LiDAR (array float32, NaN = nodata, ``res`` in metri).

Tutti i filtri sono NaN-aware: i buchi del DTM restano buchi e non sporcano i pixel vicini oltre il raggio del
kernel. Accumulo di deflusso: riempimento delle depressioni (priority-flood) + direzioni D8, implementati qui
(numba) per non dipendere da binari esterni.
"""

from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from morloquio.common.cache import params_hash
from morloquio.common.paths import processed_dir
from morloquio.common.runlog import get_logger

LAYERS = ("slope", "hillshade", "lrm", "curv_profile", "curv_plan", "chm", "flowacc")

log = get_logger("terrain.derivatives")


def _nan_gaussian(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussiana con convoluzione normalizzata: ignora i NaN invece di propagarli."""
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0).astype("float64")
    num = gaussian_filter(filled, sigma_px, mode="nearest", truncate=3.0)
    den = gaussian_filter(valid.astype("float64"), sigma_px, mode="nearest", truncate=3.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out.astype("float32")


def _gradients(dtm: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """dz/dx (verso est) e dz/dy (verso nord); le righe del raster crescono verso sud."""
    gy_rows, gx = np.gradient(dtm.astype("float64"), res, res)
    return gx, -gy_rows


def slope_deg(dtm: np.ndarray, res: float) -> np.ndarray:
    gx, gy = _gradients(dtm, res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def hillshade_multi(dtm: np.ndarray, res: float, azimuths=(315, 270, 225, 360), altitude: float = 45.0) -> np.ndarray:
    gx, gy = _gradients(dtm, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)  # direzione di massima discesa, 0 = nord, senso orario
    alt = np.radians(altitude)
    shades = []
    for az in azimuths:
        a = np.radians(az)
        shades.append(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(a - aspect))
    hs = np.clip(np.mean(shades, axis=0), 0, 1) * 254 + 1
    return np.where(np.isnan(hs), 0, hs).astype("uint8")


def local_relief(dtm: np.ndarray, res: float, sigma_m: float = 10.0) -> np.ndarray:
    """Local Relief Model: DTM meno il DTM lisciato; evidenzia solchi (negativi) e dossi (positivi)."""
    smooth = _nan_gaussian(dtm, sigma_m / res)
    return (dtm - smooth).astype("float32")


def curvatures(dtm: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Curvatura di profilo e planare (Zevenbergen-Thorne), 1/m."""
    p, q = _gradients(dtm, res)
    r = np.gradient(p, res, axis=1)
    t = -np.gradient(q, res, axis=0)
    s = np.gradient(p, res, axis=0) * -1
    g2 = p**2 + q**2
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = -(p**2 * r + 2 * p * q * s + q**2 * t) / (g2 * (1 + g2) ** 1.5)
        plan = -(q**2 * r - 2 * p * q * s + p**2 * t) / (g2**1.5)
    flat = g2 < 1e-10
    prof[flat & ~np.isnan(dtm)] = 0.0
    plan[flat & ~np.isnan(dtm)] = 0.0
    return prof.astype("float32"), plan.astype("float32")


def _shift(z: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``out[r, c] = z[r + dr, c + dc]``, NaN fuori dal raster."""
    h, w = z.shape
    out = np.full_like(z, np.nan)
    rs, re_ = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    if rs < re_ and cs < ce:
        out[rs:re_, cs:ce] = z[rs + dr : re_ + dr, cs + dc : ce + dc]
    return out


def openness(dtm: np.ndarray, res: float, radius_m: float = 10.0, n_dir: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Openness positiva e negativa (gradi) entro ``radius_m`` su ``n_dir`` direzioni (Yokoyama 2002, parametri RVT).

    90° su qualunque superficie piana, anche inclinata: non dipende dalla pendenza, a differenza dell'ombreggiatura.
    Positiva < 90° in solchi e trincee, negativa < 90° su creste e cigli. Le direzioni senza dati contano come
    orizzonte piatto.
    """
    z = dtm.astype("float64")
    r = max(1, int(round(radius_m / res)))
    pos = np.zeros_like(z)
    neg = np.zeros_like(z)
    for a in np.arange(n_dir) * 2 * np.pi / n_dir:
        hi = np.full_like(z, -np.inf)
        lo = np.full_like(z, np.inf)
        seen = set()
        for k in range(1, r + 1):
            dc, dr = int(round(k * np.cos(a))), int(round(-k * np.sin(a)))
            if (dr, dc) in seen or (dr, dc) == (0, 0):
                continue
            seen.add((dr, dc))
            with np.errstate(invalid="ignore"):
                ang = np.degrees(np.arctan((_shift(z, dr, dc) - z) / (np.hypot(dr, dc) * res)))
            hi = np.fmax(hi, ang)
            lo = np.fmin(lo, ang)
        pos += 90.0 - np.where(np.isfinite(hi), hi, 0.0)
        neg += 90.0 + np.where(np.isfinite(lo), lo, 0.0)
    pos /= n_dir
    neg /= n_dir
    missing = np.isnan(z)
    pos[missing] = np.nan
    neg[missing] = np.nan
    return pos.astype("float32"), neg.astype("float32")


def canopy_height(dsm_first: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    chm = dsm_first.astype("float32") - dtm.astype("float32")
    return np.where(np.isnan(chm), np.nan, np.maximum(chm, 0.0)).astype("float32")


_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_DIST = np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)])


@numba.njit(cache=True)
def _priority_flood(z: np.ndarray, valid: np.ndarray, eps: float) -> np.ndarray:
    """Riempie le depressioni (Barnes et al. 2014, variante epsilon): ogni cella drena verso un bordo o un NaN."""
    rows, cols = z.shape
    filled = z.copy()
    closed = ~valid
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            edge = r == 0 or c == 0 or r == rows - 1 or c == cols - 1
            if not edge:
                for k in range(8):
                    if not valid[r + _DR[k], c + _DC[k]]:
                        edge = True
                        break
            if edge:
                closed[r, c] = True
                _heap_push(heap, (filled[r, c], np.int64(r * cols + c)))
    while len(heap) > 0:
        elev, idx = _heap_pop(heap)
        r, c = idx // cols, idx % cols
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= rows or cc >= cols or closed[rr, cc]:
                continue
            closed[rr, cc] = True
            if filled[rr, cc] <= elev:
                filled[rr, cc] = elev + eps
            _heap_push(heap, (filled[rr, cc], np.int64(rr * cols + cc)))
    return filled


@numba.njit(cache=True)
def _heap_push(heap, item):
    heap.append(item)
    i = len(heap) - 1
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent][0] <= heap[i][0]:
            break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent


@numba.njit(cache=True)
def _heap_pop(heap):
    last = heap.pop()
    if len(heap) == 0:
        return last
    top = heap[0]
    heap[0] = last
    i, n = 0, len(heap)
    while True:
        left, right = 2 * i + 1, 2 * i + 2
        smallest = i
        if left < n and heap[left][0] < heap[smallest][0]:
            smallest = left
        if right < n and heap[right][0] < heap[smallest][0]:
            smallest = right
        if smallest == i:
            break
        heap[smallest], heap[i] = heap[i], heap[smallest]
        i = smallest
    return top


@numba.njit(cache=True)
def _d8_accumulate(filled: np.ndarray, valid: np.ndarray, order: np.ndarray, res: float) -> np.ndarray:
    rows, cols = filled.shape
    acc = np.ones(filled.shape, dtype=np.float64)
    for i in range(order.size):
        idx = order[i]
        r, c = idx // cols, idx % cols
        if not valid[r, c]:
            continue
        best, best_k = 0.0, -1
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= rows or cc >= cols or not valid[rr, cc]:
                continue
            drop = (filled[r, c] - filled[rr, cc]) / (_DIST[k] * res)
            if drop > best:
                best, best_k = drop, k
        if best_k >= 0:
            acc[r + _DR[best_k], c + _DC[best_k]] += acc[r, c]
    return acc


def flow_accumulation(dtm: np.ndarray, res: float) -> np.ndarray:
    """Numero di celle a monte (inclusa la cella), D8 sul DTM riempito; NaN dove il DTM è NaN."""
    z = dtm.astype("float64")
    valid = ~np.isnan(z)
    filled = _priority_flood(np.where(valid, z, 0.0), valid, 1e-4)
    order = np.argsort(-np.where(valid, filled, -np.inf), axis=None, kind="stable").astype(np.int64)
    acc = _d8_accumulate(filled, valid, order, res)
    return np.where(valid, acc, np.nan).astype("float32")


def _write_like(path: Path, arr: np.ndarray, profile: dict) -> None:
    prof = dict(profile)
    if arr.dtype == np.uint8:
        prof.update(dtype="uint8", nodata=0, predictor=2)
    else:
        prof.update(dtype="float32", nodata=float("nan"), predictor=3)
    prof.update(driver="GTiff", count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr, 1)


def compute_tile(tile_id: str, params: dict) -> Path:
    """Calcola i derivati della tile dal cubo e li scrive in ``processed/terrain/<params_hash>/<tile_id>/``."""
    cube = processed_dir("cube") / tile_id
    out = processed_dir("terrain") / params_hash(params) / tile_id
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cube / "dtm.tif") as ds:
        dtm = ds.read(1)
        profile = ds.profile
        res = float(ds.res[0])
    layers: dict[str, np.ndarray] = {
        "slope": slope_deg(dtm, res),
        "hillshade": hillshade_multi(dtm, res, tuple(params["hillshade_azimuths"]), params["hillshade_altitude"]),
        "lrm": local_relief(dtm, res, params["lrm_sigma_m"]),
        "flowacc": flow_accumulation(dtm, res),
    }
    layers["curv_profile"], layers["curv_plan"] = curvatures(dtm, res)
    if (cube / "dsm_first.tif").exists():
        with rasterio.open(cube / "dsm_first.tif") as ds:
            layers["chm"] = canopy_height(ds.read(1), dtm)
    else:
        layers["chm"] = np.full_like(dtm, np.nan)
    for name, arr in layers.items():
        _write_like(out / f"{name}.tif", arr, profile)
    (out / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    log.info("derivati calcolati", extra={"tile_id": tile_id, "params_hash": out.parent.name})
    return out
