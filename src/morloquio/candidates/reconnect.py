"""Ricucitura dei buchi nel grafo delle vie, dalle foglie (DeepRoadMapper, Máttyus et al. 2017).

Una via sotto il bosco o in un punto d'ombra del LiDAR esce dalla maschera a tratti: lo scheletro si spezza e la
completezza cala. Per ogni foglia (estremo di grado 1) si cercano linee entro ``search_m`` che nel grafo siano
lontane più di ``min_graph_m`` (o non raggiungibili); per la più promettente si traccia il percorso a costo minimo
sul raster ``1 − p`` e lo si accetta se la probabilità media lungo il percorso è almeno ``min_mean_score`` e il
percorso non gira troppo (``max_detour`` × distanza in linea d'aria). Tarata per accettare molto: per la completezza
un collegamento dubbio costa meno di un buco.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio.transform import rowcol, xy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from skimage.graph import route_through_array


def _key(pt) -> tuple[float, float]:
    return round(float(pt[0]), 2), round(float(pt[1]), 2)


def _graph(lines: list[LineString]):
    nodes: dict[tuple[float, float], int] = {}
    ends = []
    for g in lines:
        a = nodes.setdefault(_key(g.coords[0]), len(nodes))
        b = nodes.setdefault(_key(g.coords[-1]), len(nodes))
        ends.append((a, b))
    weight: dict[tuple[int, int], float] = {}
    degree = np.zeros(len(nodes), int)
    for (a, b), g in zip(ends, lines, strict=True):
        degree[a] += 1
        degree[b] += 1
        k = (min(a, b), max(a, b))
        weight[k] = min(weight.get(k, np.inf), max(g.length, 1e-6))
    rows = [a for a, _ in weight] + [b for _, b in weight]
    cols = [b for _, b in weight] + [a for a, _ in weight]
    vals = list(weight.values()) * 2
    graph = csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    coords = np.zeros((len(nodes), 2))
    for k, i in nodes.items():
        coords[i] = k
    return graph, ends, degree, coords


def _samples(lines: list[LineString], step: float) -> tuple[np.ndarray, np.ndarray]:
    pts, owner = [], []
    for i, g in enumerate(lines):
        d = np.append(np.arange(0.0, g.length, step), g.length)
        pts.append(shapely.get_coordinates(shapely.line_interpolate_point(g, d)))
        owner.append(np.full(len(d), i))
    return np.concatenate(pts), np.concatenate(owner)


def _route(score: np.ndarray, transform, start, end, margin: int):
    (r0, r1), (c0, c1) = (np.asarray(v) for v in rowcol(transform, [start[0], end[0]], [start[1], end[1]]))
    h, w = score.shape
    top, left = max(min(r0, r1) - margin, 0), max(min(c0, c1) - margin, 0)
    bottom, right = min(max(r0, r1) + margin + 1, h), min(max(c0, c1) + margin + 1, w)
    if not (0 <= r0 < h and 0 <= r1 < h and 0 <= c0 < w and 0 <= c1 < w):
        return None
    p = np.nan_to_num(score[top:bottom, left:right], nan=0.0).clip(0, 1)
    cost = np.maximum(1.0 - p, 0.01)
    path, _ = route_through_array(
        cost, (r0 - top, c0 - left), (r1 - top, c1 - left), fully_connected=True, geometric=True
    )
    path = np.asarray(path)
    values = p[path[:, 0], path[:, 1]]
    xs, ys = xy(transform, path[:, 0] + top, path[:, 1] + left, offset="center")
    return LineString(np.column_stack([xs, ys])), float(values.mean()), float(values.max())


def reconnect(gdf: gpd.GeoDataFrame, score: np.ndarray, transform, params: dict) -> gpd.GeoDataFrame:
    """Linee di ``gdf`` più i collegamenti accettati (stesse colonne: ``length_m``, ``mean_score``, ``max_score``)."""
    lines = [g for g in gdf.geometry if g is not None and not g.is_empty and g.geom_type == "LineString"]
    if not lines:
        return gdf
    search = float(params["search_m"])
    graph, ends, degree, coords = _graph(lines)
    leaves = np.flatnonzero(degree == 1)
    if not leaves.size:
        return gdf
    near = dijkstra(graph, directed=False, indices=leaves, limit=float(params["min_graph_m"]))
    pts, owner = _samples(lines, float(params["sample_m"]))
    tree = cKDTree(pts)
    margin = int(np.ceil(search / abs(transform.a))) // 2 + 3
    done: set[frozenset] = set()
    new_geoms, new_attrs = [], []
    for li, leaf in enumerate(leaves):
        start = coords[leaf]
        reachable = np.isfinite(near[li])
        close_lines = {i for i, (a, b) in enumerate(ends) if reachable[a] or reachable[b]}
        own = next(i for i, (a, b) in enumerate(ends) if leaf in (a, b))
        best_by_line: dict[int, int] = {}
        for c in tree.query_ball_point(start, search):
            line = int(owner[c])
            if line in close_lines or frozenset((own, line)) in done:
                continue
            if line not in best_by_line or np.hypot(*(pts[c] - start)) < np.hypot(*(pts[best_by_line[line]] - start)):
                best_by_line[line] = c
        best = None
        for line, c in best_by_line.items():
            routed = _route(score, transform, start, pts[c], margin)
            if routed is None:
                continue
            geom, mean, peak = routed
            euclid = max(float(np.hypot(*(pts[c] - start))), 1e-6)
            if mean < float(params["min_mean_score"]) or geom.length > float(params["max_detour"]) * euclid + 2:
                continue
            if best is None or mean > best[2]:
                best = (line, geom, mean, peak)
        if best is not None:
            line, geom, mean, peak = best
            done.add(frozenset((own, line)))
            new_geoms.append(geom)
            new_attrs.append((geom.length, mean, peak))
    if not new_geoms:
        return gdf
    extra = gpd.GeoDataFrame(
        pd.DataFrame(new_attrs, columns=["length_m", "mean_score", "max_score"]), geometry=new_geoms, crs=gdf.crs
    )
    return gpd.GeoDataFrame(pd.concat([gdf, extra], ignore_index=True), crs=gdf.crs)
