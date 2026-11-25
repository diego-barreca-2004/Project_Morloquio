"""Valutazione dello Stadio A contro OSM: completezza (recall) e correttezza (precisione) a buffer.

- completezza = lunghezza del riferimento OSM entro ``buffer_m`` da un candidato / lunghezza del riferimento;
- correttezza = lunghezza dei candidati entro ``buffer_m`` dal riferimento / lunghezza dei candidati.
La correttezza contro OSM è una stima per difetto: OSM è incompleto, ed è proprio quello che vogliamo superare.
Il riferimento esclude ciò che il terreno non può mostrare (ponti, gallerie, interni, livelli sotterranei).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import shapes
from shapely.geometry import shape

from morloquio.candidates.vectorize import core_polygon
from morloquio.common.aoi import AOI
from morloquio.common.grid import tiles_in_bbox
from morloquio.common.paths import interim_dir, processed_dir, runs_dir
from morloquio.common.runlog import get_logger

EXCLUDED_HIGHWAY = {"elevator", "corridor", "platform", "proposed", "construction", "abandoned", "razed"}
EXIT_THRESHOLD = 0.98

log = get_logger("eval.osm_recall")


def _is_set(value) -> bool:
    """Tag OSM presente e diverso da "no" (``bridge=yes``, ``tunnel=culvert``, ...)."""
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).lower() != "no"


def _layer(value) -> float:
    """``layer`` OSM come numero; mancante o non numerico = 0 (livello del suolo)."""
    try:
        layer = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(layer) else layer


def reference_ways(highways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = ~highways["highway"].isin(EXCLUDED_HIGHWAY)
    keep &= ~highways["bridge"].map(_is_set)
    keep &= ~highways["tunnel"].map(_is_set)
    keep &= highways["indoor"].map(lambda v: str(v).lower() != "yes")
    keep &= highways["layer"].map(_layer) >= 0
    return highways[keep].copy()


def _clip(gdf: gpd.GeoDataFrame, mask) -> gpd.GeoSeries:
    geoms = gdf.geometry
    if mask is not None:
        geoms = geoms.intersection(mask)
    return geoms[~geoms.is_empty]


def _matched_ratio(subject: gpd.GeoDataFrame, other: gpd.GeoDataFrame, buffer_m: float, mask) -> float:
    if subject.crs is not None and other.crs is not None and other.crs != subject.crs:
        other = other.to_crs(subject.crs)
    geoms = _clip(subject, mask)
    total = float(geoms.length.sum())
    if total <= 0:
        return float("nan")
    if other.empty:
        return 0.0
    zone = shapely.union_all(other.geometry.buffer(buffer_m).values)
    return float(geoms.intersection(zone).length.sum()) / total


def completeness(reference, candidates, buffer_m: float = 5.0, mask=None) -> float:
    return _matched_ratio(reference, candidates, buffer_m, mask)


def correctness(candidates, reference, buffer_m: float = 5.0, mask=None) -> float:
    return _matched_ratio(candidates, reference, buffer_m, mask)


def report_by_class(reference, candidates, buffer_m: float, mask) -> pd.DataFrame:
    zone = shapely.union_all(candidates.geometry.buffer(buffer_m).values) if not candidates.empty else None
    rows = []
    for highway, group in reference.groupby("highway"):
        geoms = _clip(group, mask)
        ref_m = float(geoms.length.sum())
        matched_m = float(geoms.intersection(zone).length.sum()) if zone is not None else 0.0
        rows.append((highway, ref_m / 1000, matched_m / 1000, matched_m / ref_m if ref_m > 0 else float("nan")))
    rep = pd.DataFrame(rows, columns=["highway", "ref_km", "matched_km", "completeness"])
    return rep.sort_values("ref_km", ascending=False).reset_index(drop=True)


def lidar_mask(aoi: AOI):
    """Area dove il DTM LiDAR è valido: pixel validi del ``dtm.tif`` delle tile T1, ritagliati ai nuclei."""
    parts = []
    for tile_id in tiles_in_bbox(aoi.bbox):
        d = processed_dir("cube") / tile_id
        meta_path = d / "meta.json"
        if not meta_path.exists() or json.loads(meta_path.read_text())["tier"] != "T1":
            continue
        with rasterio.open(d / "dtm.tif") as ds:
            valid = (~np.isnan(ds.read(1))).astype("uint8")
            polys = [shape(g) for g, v in shapes(valid, mask=valid.astype(bool), transform=ds.transform) if v == 1]
        if polys:
            parts.append(shapely.union_all(polys).intersection(core_polygon(tile_id, aoi.work_crs)))
    return shapely.union_all(parts) if parts else shapely.Polygon()


def evaluate_run(run_id: str, aoi: AOI, buffer_m: float = 5.0) -> Path:
    highways = gpd.read_file(interim_dir("osm") / f"{aoi.name}.gpkg", layer="highways")
    reference = reference_ways(highways).to_crs(aoi.work_crs)
    cand_path = runs_dir() / run_id / "stage_a" / "candidates_merged.gpkg"
    candidates = gpd.read_file(cand_path).to_crs(aoi.work_crs)
    mask = lidar_mask(aoi)
    out = runs_dir() / run_id / "eval"
    out.mkdir(parents=True, exist_ok=True)
    comp = completeness(reference, candidates, buffer_m, mask)
    report = {
        "run_id": run_id,
        "aoi": aoi.name,
        "buffer_m": buffer_m,
        "mask_km2": mask.area / 1e6,
        "ref_km": float(_clip(reference, mask).length.sum()) / 1000,
        "cand_km": float(_clip(candidates, mask).length.sum()) / 1000,
        "completeness": comp,
        "correctness": correctness(candidates, reference, buffer_m, mask),
        "exit_threshold": EXIT_THRESHOLD,
        "passed": bool(comp >= EXIT_THRESHOLD),
    }
    (out / "stage_a.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_by_class(reference, candidates, buffer_m, mask).to_csv(out / "stage_a_by_class.csv", index=False)
    log.info("valutazione Stadio A", extra={k: v for k, v in report.items() if k != "run_id"})
    return out / "stage_a.json"
