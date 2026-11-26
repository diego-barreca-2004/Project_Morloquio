"""Stadio A — punteggi di "percorribilità lineare" per pixel, senza etichette (alto recall).

Segnali:
- ``bench``: gradino di versante (striscia piana con taglio a monte e riporto a valle) dal DTM;
- ``lrm_lin``: linearità (Hessiano) del Local Relief Model: solchi e argini lineari;
- ``chm_gap``: corridoi lineari senza chioma dentro il bosco;
- ``ortho_lin``: linee chiare o scure nell'ortofoto (se disponibile);
- ``gps``: densità delle tracce GPS pubbliche.
Ogni segnale è normalizzato in 0–1 sulla tile, poi fuso con una media pesata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import sato

from morloquio.common.cache import params_hash
from morloquio.common.paths import processed_dir, runs_dir
from morloquio.common.runlog import get_logger
from morloquio.terrain.derivatives import _nan_gaussian

log = get_logger("candidates.linearity")


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """z robusto (mediana/MAD) sui pixel validi, poi logistica centrata a z=2: valori in 0–1, NaN preservati."""
    x = np.asarray(x, dtype="float64")
    valid = ~np.isnan(x)
    out = np.full(x.shape, np.nan, dtype="float32")
    if not valid.any():
        return out
    v = x[valid]
    med = np.median(v)
    scale = 1.4826 * np.median(np.abs(v - med))
    if scale <= 1e-12:
        scale = float(np.std(v))
    if scale <= 1e-12:
        out[valid] = 0.0
        return out
    z = (v - med) / scale
    out[valid] = (1.0 / (1.0 + np.exp(-(z - 2.0)))).astype("float32")
    return out


def _fill_nan(img: np.ndarray, sigma_px: float = 3.0) -> np.ndarray:
    if not np.isnan(img).any():
        return img.astype("float64")
    smooth = _nan_gaussian(img, sigma_px)
    filled = np.where(np.isnan(img), smooth, img)
    return np.where(np.isnan(filled), np.nanmean(img) if (~np.isnan(img)).any() else 0.0, filled).astype("float64")


def hessian_linearity(
    img: np.ndarray,
    res: float,
    widths_m=(1, 2, 3, 4, 6),
    polarity: Literal["valley", "ridge", "both"] = "both",
) -> np.ndarray:
    """Risposta di Sato (strutture tubolari) alle larghezze indicate, normalizzata 0–1."""
    nan = np.isnan(img)
    work = _fill_nan(img)
    sigmas = [w / (2.0 * res) for w in widths_m]
    responses = []
    if polarity in ("valley", "both"):
        responses.append(sato(work, sigmas=sigmas, black_ridges=True, mode="reflect"))
    if polarity in ("ridge", "both"):
        responses.append(sato(work, sigmas=sigmas, black_ridges=False, mode="reflect"))
    resp = np.max(responses, axis=0).astype("float64")
    resp[nan] = np.nan
    return robust_normalize(resp)


def bench_score(
    dtm: np.ndarray,
    res: float,
    half_widths_m=(1.0, 1.5, 2.5),
    offset_m: float = 3.0,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Gradino di versante: piano al centro, pendenza sia a monte sia a valle (vedi piano, Task 13)."""
    nan = np.isnan(dtm)
    z = _fill_nan(dtm)
    smooth = gaussian_filter(z, 2.0 / res, mode="nearest")
    g_rows, g_cols = np.gradient(smooth)
    norm = np.hypot(g_rows, g_cols)
    sloped = norm > 1e-6
    d_rows = np.where(sloped, g_rows / np.where(sloped, norm, 1), 0.0)
    d_cols = np.where(sloped, g_cols / np.where(sloped, norm, 1), 0.0)
    rows, cols = np.mgrid[0 : z.shape[0], 0 : z.shape[1]].astype("float64")

    def sample(t_m: float) -> np.ndarray:
        t = t_m / res
        return map_coordinates(z, [rows + t * d_rows, cols + t * d_cols], order=1, mode="nearest")

    best = np.zeros(z.shape, dtype="float64")
    for w in half_widths_m:
        s_pw, s_mw = sample(w), sample(-w)
        s_po, s_mo = sample(w + offset_m), sample(-w - offset_m)
        center = np.abs(s_pw - s_mw) / (2 * w)
        up = np.abs(s_po - s_pw) / offset_m
        down = np.abs(s_mw - s_mo) / offset_m
        outer = np.minimum(up, down)
        score = np.clip(outer - center, 0, None) * (center < 0.5 * outer)
        best = np.maximum(best, score)
    best[~sloped] = 0.0
    best[nan] = np.nan
    return robust_normalize(best) if normalize else best.astype("float32")


def chm_gap_linearity(chm: np.ndarray, res: float, min_canopy_m: float = 3.0) -> np.ndarray:
    """Corridoi lineari a chioma bassa dentro un intorno boscato (sentieri e piste sotto bosco largo)."""
    lin = hessian_linearity(chm, res, widths_m=(2, 3, 4, 6, 8), polarity="valley")
    context = _nan_gaussian(chm, 8.0 / res)
    return np.where(context > min_canopy_m, lin, 0.0 * lin).astype("float32")


def ortho_linearity(rgb: np.ndarray, res: float) -> np.ndarray:
    gray = rgb.astype("float32").mean(axis=0) / 255.0
    return hessian_linearity(gray, res, widths_m=(1, 2, 3, 4, 6), polarity="both")


def gps_density(points_xy: np.ndarray, transform: Affine, shape, sigma_m: float = 5.0) -> np.ndarray:
    """Densità dei punti GPS (coordinate nel CRS della griglia), normalizzata a massimo 1."""
    dens = np.zeros(shape, dtype="float64")
    if len(points_xy):
        r, c = rowcol(transform, points_xy[:, 0], points_xy[:, 1], op=np.floor)
        r, c = np.asarray(r, dtype=int), np.asarray(c, dtype=int)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        np.add.at(dens, (r[ok], c[ok]), 1.0)
        dens = gaussian_filter(dens, sigma_m / abs(transform.a))
    peak = dens.max()
    return (dens / peak if peak > 0 else dens).astype("float32")


def fuse(layers: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Media pesata dei layer disponibili; dove un layer è NaN i pesi degli altri si rinormalizzano."""
    num = None
    den = None
    for name, arr in layers.items():
        w = float(weights.get(name, 0.0))
        if w <= 0 or arr is None:
            continue
        valid = ~np.isnan(arr)
        contrib = np.where(valid, arr, 0.0) * w
        weight = valid * w
        num = contrib if num is None else num + contrib
        den = weight if den is None else den + weight
    if num is None:
        raise ValueError("nessun layer con peso positivo da fondere")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out.astype("float32")


def _read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        return ds.read(), ds.profile


def score_tile(tile_id: str, params: dict, run_id: str, gps_xy: np.ndarray | None = None) -> Path:
    """Calcola i punteggi dello Stadio A per la tile. ``params`` = stage_a.yaml + chiave ``terrain``."""
    cube = processed_dir("cube") / tile_id
    terrain = processed_dir("terrain") / params_hash(params["terrain"]) / tile_id
    out = runs_dir() / run_id / "stage_a" / tile_id
    out.mkdir(parents=True, exist_ok=True)
    dtm_b, profile = _read(cube / "dtm.tif")
    dtm = dtm_b[0]
    res = abs(profile["transform"].a)
    lrm = _read(terrain / "lrm.tif")[0][0]
    layers: dict[str, np.ndarray] = {
        "bench": bench_score(dtm, res),
        "lrm_lin": hessian_linearity(lrm, res, polarity="both"),
    }
    chm = _read(terrain / "chm.tif")[0][0]
    if not np.isnan(chm).all():
        layers["chm_gap"] = chm_gap_linearity(chm, res)
    if (cube / "ortho.tif").exists():
        rgb, o_prof = _read(cube / "ortho.tif")
        lin = ortho_linearity(rgb, abs(o_prof["transform"].a))
        # riporta la linearità dell'ortofoto sulla griglia a 1 m (media 2x2)
        h, w = dtm.shape
        layers["ortho_lin"] = lin[: 2 * h, : 2 * w].reshape(h, 2, w, 2).mean(axis=(1, 3))
    if gps_xy is not None and len(gps_xy):
        layers["gps"] = gps_density(gps_xy, profile["transform"], dtm.shape)
    layers["score"] = fuse(layers, params["weights"])
    layers["score"][np.isnan(dtm)] = np.nan
    prof = dict(profile)
    prof.update(
        driver="GTiff",
        dtype="float32",
        nodata=float("nan"),
        count=1,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    for name, arr in layers.items():
        with rasterio.open(out / f"{name}.tif", "w", **prof) as ds:
            ds.write(arr.astype("float32"), 1)
    log.info("Stadio A: punteggi calcolati", extra={"tile_id": tile_id, "layers": sorted(layers)})
    return out
