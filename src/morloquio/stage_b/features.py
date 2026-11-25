"""Canali per pixel dello Stadio B: derivati del LiDAR e segnali dello Stadio A, normalizzati in modo globale.

La normalizzazione è la stessa per tutte le tile (centro e scala robusti stimati una volta sull'area): una
normalizzazione per tile o per patch schiaccia i solchi di pochi decimetri quando nella stessa finestra ci sono
versanti alti (Remote Sensing 17(9):1539, §4). Nessun canale deriva da OSM: le etichette restano solo etichette.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio

from morloquio.candidates.linearity import bench_score, chm_gap_linearity, gps_density, hessian_linearity
from morloquio.common.cache import params_hash
from morloquio.common.paths import processed_dir
from morloquio.common.runlog import get_logger

CHANNELS = (
    "slope",
    "hillshade",
    "lrm",
    "curv_profile",
    "curv_plan",
    "chm",  # log(1 + DSM first − DTM)
    "veg_gap",  # log(1 + DSM first − DSM last): quanto la chioma lascia passare il laser
    "ndsm_last",  # DSM last − DTM, log con segno: sottobosco, muri, edifici
    "flowacc",  # log(1 + area drenata)
    "bench",
    "lrm_lin",
    "chm_gap",
    "gps",
    "valid",  # 1 dove il DTM è valido
)
_TERRAIN = ("slope", "hillshade", "lrm", "curv_profile", "curv_plan")
_EPS = 1e-6

log = get_logger("stage_b.features")


def _read1(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        return ds.read(1).astype("float32"), ds.profile


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _raw(tile_id: str, params: dict, gps_xy: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict]:
    cube = processed_dir("cube") / tile_id
    terrain = processed_dir("terrain") / params_hash(params["terrain"]) / tile_id
    dtm, profile = _read1(cube / "dtm.tif")
    res = abs(profile["transform"].a)
    valid = np.isfinite(dtm)
    nan = np.full_like(dtm, np.nan)

    def optional(name: str) -> np.ndarray:
        path = cube / f"{name}.tif"
        return _read1(path)[0] if path.exists() else nan

    out = {name: _read1(terrain / f"{name}.tif")[0] for name in _TERRAIN}
    chm = _read1(terrain / "chm.tif")[0]
    dsm_first, dsm_last = optional("dsm_first"), optional("dsm_last")
    out["chm"] = np.log1p(np.maximum(chm, 0.0))
    out["veg_gap"] = np.log1p(np.maximum(dsm_first - dsm_last, 0.0))
    out["ndsm_last"] = _signed_log(dsm_last - dtm)
    out["flowacc"] = np.log1p(np.maximum(_read1(terrain / "flowacc.tif")[0], 0.0))
    out["bench"] = bench_score(dtm, res)
    out["lrm_lin"] = hessian_linearity(out["lrm"], res, polarity="both")
    out["chm_gap"] = chm_gap_linearity(chm, res) if np.isfinite(chm).any() else np.zeros_like(dtm)
    if gps_xy is not None and len(gps_xy):
        out["gps"] = gps_density(gps_xy, profile["transform"], dtm.shape)
    else:
        out["gps"] = np.zeros_like(dtm)
    out = {k: np.where(valid, v, np.nan).astype("float32") for k, v in out.items()}
    out["valid"] = valid.astype("float32")
    return out, profile


def raw_channels(tile_id: str, params: dict, gps_xy: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Canali grezzi (unità fisiche o 0–1, NaN fuori DTM) della tile, prima della normalizzazione."""
    return _raw(tile_id, params, gps_xy)[0]


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    center = float(np.median(x))
    scale = 1.4826 * float(np.median(np.abs(x - center)))
    if scale < _EPS:  # canale sparso (es. tracce GPS quasi sempre 0): scala dai valori non nulli
        nz = np.abs(x[np.abs(x - center) > _EPS] - center)
        scale = float(np.median(nz)) if nz.size else 1.0
    return center, max(scale, _EPS)


def fit_normalization(
    tile_ids: list[str], params: dict, gps_xy: np.ndarray | None = None, sample_px: int = 200_000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Centro (mediana) e scala (MAD) di ogni canale su un campione di pixel validi di tutte le tile."""
    rng = np.random.default_rng(seed)
    per_tile = max(1, sample_px // max(1, len(tile_ids)))
    samples: dict[str, list[np.ndarray]] = {c: [] for c in CHANNELS if c != "valid"}
    for tile_id in tile_ids:
        raw = raw_channels(tile_id, params, gps_xy)
        idx = np.flatnonzero(raw["valid"] > 0)
        if not idx.size:
            continue
        pick = rng.choice(idx, size=min(per_tile, idx.size), replace=False)
        for c in samples:
            v = raw[c].ravel()[pick]
            samples[c].append(v[np.isfinite(v)])
    norm = {}
    for c, parts in samples.items():
        x = np.concatenate(parts) if parts else np.zeros(1, "float32")
        norm[c] = _robust_scale(x if x.size else np.zeros(1, "float32"))
    return norm


def features_dir(params: dict, norm: dict) -> Path:
    key = {"terrain": params["terrain"], "features": params.get("features", {}), "norm": norm}
    return processed_dir("stage_b") / params_hash(key)


def build_features(tile_id: str, params: dict, norm: dict, gps_xy: np.ndarray | None = None) -> Path:
    """Scrive ``features.npy`` (float16, C×H×W, valori in [−1, 1], NaN → 0) e ``meta.json`` della tile."""
    raw, profile = _raw(tile_id, params, gps_xy)
    div = float(params.get("features", {}).get("tanh_div", 3.0))
    layers = [np.tanh((raw[c] - norm[c][0]) / (div * norm[c][1])) for c in CHANNELS if c != "valid"]
    x = np.nan_to_num(np.stack(layers + [raw["valid"]]), nan=0.0).astype("float16")
    out = features_dir(params, norm) / tile_id
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"features.{os.getpid()}.npy"
    np.save(part, x)
    os.replace(part, out / "features.npy")
    meta = {
        "tile_id": tile_id,
        "channels": list(CHANNELS),
        "crs": profile["crs"].to_string(),
        "transform": list(profile["transform"])[:6],
        "shape": list(x.shape[1:]),
        "norm": {c: list(v) for c, v in norm.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("canali Stadio B", extra={"tile_id": tile_id, "dir": out.parent.name})
    return out
