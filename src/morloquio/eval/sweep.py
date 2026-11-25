"""Curva completezza/correttezza al variare della soglia sulla probabilità per pixel (fuori fold).

Serve a scegliere la soglia per il vincolo di completezza invece di fissarla a 0,5 (Virro 2025: in bosco una soglia
bassa è molto più completa). È un'approssimazione per pixel della valutazione vettoriale di ``osm_recall``:
- completezza: frazione dei punti delle vie OSM (uno ogni ``step_m``) entro ``buffer_m`` da un pixel ≥ soglia;
- correttezza: frazione dei pixel dello scheletro della maschera ≥ soglia entro ``buffer_m`` da una via OSM
  (limite inferiore: OSM è incompleto).
La soglia scelta va poi confermata con la valutazione vettoriale vera.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt, maximum_filter
from shapely.geometry.base import BaseGeometry
from skimage.morphology import skeletonize


def _samples(lines: gpd.GeoDataFrame, step_m: float) -> np.ndarray:
    pts = []
    for geom in lines.geometry:
        for part in getattr(geom, "geoms", [geom]):
            if part.is_empty or part.length == 0 or part.geom_type != "LineString":
                continue
            d = np.arange(step_m / 2, part.length, step_m)
            pts.append(shapely.get_coordinates(shapely.line_interpolate_point(part, d)))
    return np.concatenate(pts) if pts else np.empty((0, 2))


def threshold_curve(
    prob_paths: dict[str, Path],
    reference: gpd.GeoDataFrame,
    cores: dict[str, BaseGeometry],
    thresholds: list[float],
    buffer_m: float = 5.0,
    step_m: float = 1.0,
) -> pd.DataFrame:
    thresholds = sorted(float(t) for t in thresholds)
    hits = np.zeros(len(thresholds))
    skel = np.zeros(len(thresholds))
    matched = np.zeros(len(thresholds))
    ref_total = 0
    for tile, path in sorted(prob_paths.items()):
        with rasterio.open(path) as ds:
            prob = ds.read(1)
            transform = ds.transform
        res = abs(transform.a)
        valid = np.isfinite(prob)
        p = np.where(valid, prob, 0.0)
        rad = buffer_m / res
        r = int(math.ceil(rad))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        near = maximum_filter(p, footprint=(xx**2 + yy**2) <= rad**2)
        core = cores[tile]
        local = reference[reference.intersects(core)].clip(core)
        pts = _samples(local, step_m)
        if len(pts):
            rows, cols = (np.asarray(a) for a in rowcol(transform, pts[:, 0], pts[:, 1]))
            inside = (rows >= 0) & (rows < p.shape[0]) & (cols >= 0) & (cols < p.shape[1])
            rows, cols = rows[inside], cols[inside]
            keep = valid[rows, cols]
            vals = near[rows[keep], cols[keep]]
            ref_total += len(vals)
            hits += np.array([(vals >= t).sum() for t in thresholds])
        in_core = rasterize([(core, 1)], out_shape=p.shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        lines = [g for g in local.geometry if g is not None and not g.is_empty]
        if lines:
            osm = rasterize([(g, 1) for g in lines], out_shape=p.shape, transform=transform, all_touched=True)
            dist = distance_transform_edt(osm == 0) * res
        else:
            dist = np.full(p.shape, np.inf)
        for i, t in enumerate(thresholds):
            s = skeletonize((p >= t) & in_core & valid)
            skel[i] += s.sum()
            matched[i] += (s & (dist <= buffer_m)).sum()
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "completeness": hits / max(ref_total, 1),
            "correctness": matched / np.maximum(skel, 1),
            "ref_km": ref_total * step_m / 1000,
            "skeleton_km": skel * 1.0 / 1000,
        }
    )


def sweep_run(run_id: str, aoi, stage: str = "stage_b", thresholds=None, buffer_m: float = 5.0) -> Path:
    """Curva per tutte le tile del run con ``<stage>/<tile>/score.tif``; scrive ``eval/<stage>_sweep.csv``."""
    from morloquio.candidates.vectorize import core_polygon
    from morloquio.common.paths import interim_dir, runs_dir
    from morloquio.eval.osm_recall import reference_ways

    thresholds = thresholds or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    root = runs_dir() / run_id / stage
    probs = {p.parent.name: p for p in sorted(root.glob("*/score.tif"))}
    highways = gpd.read_file(interim_dir("osm") / f"{aoi.name}.gpkg", layer="highways")
    reference = reference_ways(highways).to_crs(aoi.work_crs)
    cores = {t: core_polygon(t, aoi.work_crs) for t in probs}
    curve = threshold_curve(probs, reference, cores, thresholds, buffer_m)
    out = runs_dir() / run_id / "eval" / f"{stage}_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    return out
