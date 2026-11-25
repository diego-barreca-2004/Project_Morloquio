"""Stadio A — dal raster dei punteggi alle linee candidate, e cucitura delle linee tra tile adiacenti.

Pipeline per tile: soglia → chiusura morfologica (raggio 1 px) → rimozione dei frammenti piccoli → scheletro
(skimage) → grafo dello scheletro (skan) → potatura degli speroni corti → linee in coordinate mappa → unione dei
tratti ai nodi di grado 2 → semplificazione (tolleranza = 1 pixel). Ogni tile viene poi ritagliata al proprio
nucleo (senza alone): i nuclei sono disgiunti, quindi la cucitura si riduce a unire gli estremi vicini.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from scipy.ndimage import binary_closing, label
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import linemerge
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from morloquio.common.grid import tile_bounds
from morloquio.common.paths import runs_dir
from morloquio.common.runlog import get_logger

COLUMNS = ["length_m", "mean_score", "max_score"]
LAYER = "candidates"
SPUR_M = 3.0  # speroni (ramo nodo-estremo) più corti di così sono rumore dello scheletro

log = get_logger("candidates.vectorize")


def _empty(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, geometry=gpd.GeoSeries([], crs=crs))


def _snap_endpoints(lines: list[LineString], tol: float) -> list[LineString]:
    """Porta allo stesso punto (centroide) gli estremi che distano meno di ``tol``."""
    if not lines or tol <= 0:
        return lines
    ends = np.array([c for g in lines for c in (g.coords[0], g.coords[-1])], dtype="float64")
    parent = np.arange(len(ends))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(ends).query_pairs(tol):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(ends))])
    centroid = {r: ends[roots == r].mean(axis=0) for r in np.unique(roots)}
    out = []
    for k, g in enumerate(lines):
        coords = np.asarray(g.coords, dtype="float64")
        coords[0] = centroid[roots[2 * k]]
        coords[-1] = centroid[roots[2 * k + 1]]
        degenerate = len(coords) == 2 and np.allclose(coords[0], coords[-1])  # tratto collassato in un punto
        if not degenerate:
            out.append(LineString(coords))
    return out


def _merge_lines(lines: list[LineString]) -> list[LineString]:
    if not lines:
        return []
    merged = linemerge(MultiLineString(lines))
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def _sample_scores(line: LineString, score: np.ndarray, transform) -> tuple[float, float]:
    res = abs(transform.a)
    n = max(2, int(line.length / res) + 1)
    pts = shapely.line_interpolate_point(line, np.linspace(0, line.length, n))
    r, c = rowcol(transform, shapely.get_x(pts), shapely.get_y(pts), op=np.floor)
    r = np.clip(np.asarray(r, dtype=int), 0, score.shape[0] - 1)
    c = np.clip(np.asarray(c, dtype=int), 0, score.shape[1] - 1)
    vals = score[r, c]
    if np.isnan(vals).all():
        return float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanmax(vals))


def _clean_mask(score: np.ndarray, threshold: float, min_px: int) -> np.ndarray:
    mask = np.where(np.isnan(score), False, score >= threshold)
    padded = np.pad(mask, 1)
    mask = binary_closing(padded, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))[1:-1, 1:-1]
    lab, n = label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def extract_candidates(
    score: np.ndarray, transform, crs: str, threshold: float, min_length_m: float
) -> gpd.GeoDataFrame:
    """Linee candidate dal raster dei punteggi (0–1). Colonne: ``length_m``, ``mean_score``, ``max_score``."""
    res = abs(transform.a)
    mask = _clean_mask(score, threshold, max(3, int(min_length_m / res)))
    skel = skeletonize(mask)
    # componenti dello scheletro di un solo pixel: nessun ramo, skan non le gestisce
    lab, _ = label(skel, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    skel &= sizes[lab] >= 2
    if not skel.any():
        return _empty(crs)
    sk = Skeleton(skel)
    summ = summarize(sk, separator="_")
    by_component: dict[int, list[LineString]] = {}
    for i, row in summ.iterrows():
        length = row["branch_distance"] * res
        if row["branch_type"] in (1, 2) and length < SPUR_M and (summ["skeleton_id"] == row["skeleton_id"]).sum() > 1:
            continue  # sperone corto o ponte interno a un nodo multiplo
        rc = sk.path_coordinates(i)
        if len(rc) < 2:
            continue
        xs, ys = xy(transform, rc[:, 0], rc[:, 1], offset="center")
        by_component.setdefault(int(row["skeleton_id"]), []).append(LineString(np.column_stack([xs, ys])))
    geoms, attrs = [], []
    for lines in by_component.values():
        lines = _merge_lines(_snap_endpoints(lines, 2 * res))
        if sum(g.length for g in lines) < min_length_m:
            continue
        for g in lines:
            g = g.simplify(res, preserve_topology=False)
            if g.length <= 0:
                continue
            mean, peak = _sample_scores(g, score, transform)
            geoms.append(g)
            attrs.append((g.length, mean, peak))
    if not geoms:
        return _empty(crs)
    return gpd.GeoDataFrame(pd.DataFrame(attrs, columns=COLUMNS), geometry=geoms, crs=crs)


def core_polygon(tile_id: str, work_crs: str) -> Polygon:
    """Nucleo della tile (senza alone) nel CRS di lavoro; i lati sono densificati, così tile vicine combaciano."""
    poly = shapely.segmentize(box(*tile_bounds(tile_id)), 0.0005)
    to_work = Transformer.from_crs("EPSG:4326", work_crs, always_xy=True)
    return shapely.transform(poly, to_work.transform, interleaved=False)


def crop_to_core(gdf: gpd.GeoDataFrame, tile_id: str, work_crs: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    core = core_polygon(tile_id, work_crs)
    cropped = gdf.copy()
    cropped["geometry"] = gdf.geometry.intersection(core)
    cropped = cropped.explode(index_parts=False)
    cropped = cropped[cropped.geometry.geom_type == "LineString"]
    cropped = cropped[cropped.geometry.length > 0].copy()
    cropped["length_m"] = cropped.geometry.length
    return cropped.reset_index(drop=True)


def merge_tiles(gdfs: list[gpd.GeoDataFrame], snap_m: float = 2.0) -> gpd.GeoDataFrame:
    """Unisce le linee di più tile: estremi entro ``snap_m`` coincidono, poi fusione ai nodi di grado 2."""
    crs = next((g.crs for g in gdfs if g.crs is not None), None)
    pieces = [g for g in gdfs if not g.empty]
    if not pieces:
        return _empty(crs)
    allp = pd.concat(pieces, ignore_index=True).explode(index_parts=False).reset_index(drop=True)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=crs)
    lines = _snap_endpoints(list(allp.geometry), snap_m)
    merged = gpd.GeoDataFrame(geometry=_merge_lines(lines), crs=crs)
    # attributi: media pesata sulla lunghezza dei pezzi originali che compongono ogni linea fusa
    mids = gpd.GeoDataFrame(
        {"w": allp.geometry.length, "mean_score": allp["mean_score"], "max_score": allp["max_score"]},
        geometry=allp.geometry.interpolate(0.5, normalized=True),
        crs=crs,
    )
    zone = gpd.GeoDataFrame(geometry=merged.buffer(max(snap_m, 0.01)), crs=crs)
    joined = gpd.sjoin(mids, zone, predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].copy()
    joined["ws"] = joined["mean_score"] * joined["w"]
    sums = joined.groupby("index_right")[["ws", "w"]].sum()
    merged["length_m"] = merged.geometry.length
    merged["mean_score"] = (sums["ws"] / sums["w"]).reindex(merged.index).astype("float64")
    merged["max_score"] = joined.groupby("index_right")["max_score"].max().reindex(merged.index).astype("float64")
    return merged[[*COLUMNS, "geometry"]]


def _write(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.unlink(missing_ok=True)
    gdf.to_file(path, layer=LAYER, engine="pyogrio", geometry_type="LineString")
    return path


def vectorize_tile(tile_id: str, params: dict, run_id: str, stage: str = "stage_a") -> Path:
    d = runs_dir() / run_id / stage / tile_id
    with rasterio.open(d / "score.tif") as ds:
        score = ds.read(1)
        transform = ds.transform
        crs = ds.crs.to_string()
    gdf = extract_candidates(score, transform, crs, float(params["threshold"]), float(params["min_length_m"]))
    gdf = crop_to_core(gdf, tile_id, crs)
    log.info(
        "candidati estratti", extra={"tile_id": tile_id, "count": len(gdf), "km": float(gdf.length_m.sum()) / 1000}
    )
    return _write(gdf, d / "candidates.gpkg")


def merge_run(run_id: str, stage: str = "stage_a") -> Path:
    root = runs_dir() / run_id / stage
    gdfs = [gpd.read_file(p, layer=LAYER) for p in sorted(root.glob("*/candidates.gpkg"))]
    merged = merge_tiles(gdfs)
    log.info("candidati cuciti", extra={"run_id": run_id, "tiles": len(gdfs), "count": len(merged)})
    return _write(merged, root / "candidates_merged.gpkg")
