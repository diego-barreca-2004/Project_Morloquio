"""Restauro dei raster LiDAR PST ricampionati con nearest-neighbour dal servizio di download.

I file PST arrivano in EPSG:4326 a 1e-5° (0,876 × 1,110 m a 38° N), ottenuti dalla griglia UTM 1 m originale con
nearest-neighbour: in x una colonna ogni ~8 è duplicata, in y una riga ogni ~10 manca. Riproiettarli di nuovo
(bilineare) lascia righe regolari N-S ed E-O ogni 7–10 m, che i filtri di linearità scambiano per sentieri.

Qui si torna alla griglia originale invece di ricampionare: ogni pixel sorgente è esattamente il valore di un nodo
UTM, quindi basta riportarlo al nodo più vicino. L'allineamento dei nodi (offset rispetto ai metri interi, diverso
per contratto) si stima scegliendo quello che rende il DTM più liscio; i nodi mai campionati (le righe saltate)
si interpolano dai vicini. Un raster che non mostra duplicati (non nearest-neighbour) resta com'è.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from morloquio.common.paths import interim_dir
from morloquio.common.runlog import get_logger

NN_MIN_FRACTION = 0.03  # coppie di pixel orizzontali identiche: ~12% per nearest-neighbour, ~0 per dati continui
WINDOW_PX = 256
COARSE_STEP = 0.05
FINE_STEP = 0.01

log = get_logger("ingest.restore")


def nn_fraction(arr: np.ma.MaskedArray) -> float:
    """Frazione di coppie di pixel orizzontali valide con valore identico."""
    z = np.ma.asarray(arr).astype("float64").filled(np.nan)
    left, right = z[:, :-1], z[:, 1:]
    pairs = ~np.isnan(left) & ~np.isnan(right)
    if not pairs.any():
        return 0.0
    return float(np.mean(left[pairs] == right[pairs]))


def _scatter(x: np.ndarray, y: np.ndarray, z: np.ndarray, ox: float, oy: float):
    """Riporta ogni campione al nodo più vicino della griglia 1 m con nodi in (k + ox, k + oy)."""
    ok = ~np.isnan(z)
    x, y, z = x[ok], y[ok], z[ok]
    i = np.round(x - ox).astype(np.int64)
    j = np.round(y - oy).astype(np.int64)
    i0, j1 = int(i.min()), int(j.max())
    c, r = i - i0, j1 - j
    h, w = int(r.max()) + 1, int(c.max()) + 1
    flat = r * w + c
    cnt = np.bincount(flat, minlength=h * w)
    total = np.bincount(flat, weights=z, minlength=h * w)
    total2 = np.bincount(flat, weights=z * z, minlength=h * w)
    has = cnt > 0
    grid = np.full(h * w, np.nan)
    grid[has] = total[has] / cnt[has]
    multi = cnt > 1
    var = total2[multi] / cnt[multi] - grid[multi] ** 2
    conflicts = float(np.mean(var > 1e-6)) if multi.any() else 0.0
    return grid.reshape(h, w), conflicts, (i0 + ox, j1 + oy)


def _fill_gaps(grid: np.ndarray) -> np.ndarray:
    """Nodi isolati senza campione (righe saltate): media dei vicini sopra/sotto, altrimenti sinistra/destra."""
    p = np.pad(grid, 1, constant_values=np.nan)
    vert = (p[:-2, 1:-1] + p[2:, 1:-1]) / 2
    hor = (p[1:-1, :-2] + p[1:-1, 2:]) / 2
    out = grid.copy()
    miss = np.isnan(out)
    out[miss] = np.where(np.isnan(vert), hor, vert)[miss]
    return out


def _roughness(grid: np.ndarray) -> float:
    d2y = np.abs(np.diff(grid, 2, axis=0))
    d2x = np.abs(np.diff(grid, 2, axis=1))
    return float(np.nanmean(d2y) + np.nanmean(d2x))


def _densest_window(valid: np.ndarray, size: int) -> tuple[slice, slice]:
    h, w = valid.shape
    if h <= size or w <= size:
        return slice(None), slice(None)
    integral = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, where = -1, (0, 0)
    for r in range(0, h - size + 1, 32):
        for c in range(0, w - size + 1, 32):
            s = integral[r + size, c + size] - integral[r, c + size] - integral[r + size, c] + integral[r, c]
            if s > best:
                best, where = s, (r, c)
    r, c = where
    return slice(r, r + size), slice(c, c + size)


def _estimate_offset(x, y, z) -> tuple[float, float, float]:
    def score(ox, oy):
        return _roughness(_fill_gaps(_scatter(x, y, z, ox, oy)[0]))

    coarse = [(score(ox, oy), ox, oy) for ox in np.arange(0, 1, COARSE_STEP) for oy in np.arange(0, 1, COARSE_STEP)]
    _, bx, by = min(coarse)
    deltas = np.arange(-COARSE_STEP, COARSE_STEP + FINE_STEP / 2, FINE_STEP)
    fine = [(score((bx + dx) % 1, (by + dy) % 1), (bx + dx) % 1, (by + dy) % 1) for dx in deltas for dy in deltas]
    rough, ox, oy = min(fine)
    return round(float(ox), 3), round(float(oy), 3), rough


def restore_file(src: Path, dest: Path, work_crs: str) -> dict:
    """Restaura ``src`` sulla sua griglia UTM originale in ``dest`` (+ ``dest.json`` con la diagnostica)."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as ds:
        arr = ds.read(1, masked=True)
        t = ds.transform
        src_crs = ds.crs
    z = arr.astype("float64").filled(np.nan)
    info: dict = {"source": str(src), "work_crs": work_crs, "nn_fraction": nn_fraction(arr), "restored": False}
    if info["nn_fraction"] < NN_MIN_FRACTION or np.isnan(z).all():
        dest.unlink(missing_ok=True)
    else:
        rows, cols = np.mgrid[0 : z.shape[0], 0 : z.shape[1]] + 0.5
        to_work = Transformer.from_crs(src_crs, work_crs, always_xy=True)
        x, y = to_work.transform(t.c + cols * t.a + rows * t.b, t.f + cols * t.d + rows * t.e)
        win = _densest_window(~np.isnan(z), WINDOW_PX)
        ox, oy, rough = _estimate_offset(x[win], y[win], z[win])
        grid, conflicts, (x_first, y_top) = _scatter(x, y, z, ox, oy)
        empty = np.isnan(grid)
        grid = _fill_gaps(grid)
        part = dest.with_name(dest.name + ".part")
        with rasterio.open(
            part,
            "w",
            driver="GTiff",
            width=grid.shape[1],
            height=grid.shape[0],
            count=1,
            dtype="float32",
            crs=work_crs,
            transform=from_origin(x_first - 0.5, y_top + 0.5, 1.0, 1.0),
            nodata=float("nan"),
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as out:
            out.write(grid.astype("float32"), 1)
        os.replace(part, dest)
        valid = ~np.isnan(grid)
        info.update(
            restored=True,
            offset=[ox, oy],
            roughness=rough,
            conflicts=conflicts,
            filled_fraction=float(np.mean((empty & valid)[valid])) if valid.any() else 0.0,
        )
    dest.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("restauro LiDAR", extra={k: v for k, v in info.items() if k != "source"} | {"file": src.name})
    return info


def restored_source(src: Path, work_crs: str) -> Path:
    """Percorso da usare per ``src``: la versione restaurata (in cache) o ``src`` stesso se non serve."""
    src = Path(src)
    dest = interim_dir("pst_restored") / src.parent.name / src.name
    sidecar = dest.with_suffix(".json")
    if sidecar.exists() and sidecar.stat().st_mtime >= src.stat().st_mtime:
        info = json.loads(sidecar.read_text())
        if info.get("work_crs") == work_crs and (not info["restored"] or dest.exists()):
            return dest if info["restored"] else src
    info = restore_file(src, dest, work_crs)
    return dest if info["restored"] else src
