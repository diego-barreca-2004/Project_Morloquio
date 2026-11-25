"""Costruisce il visualizzatore locale di un run: ``runs/<run_id>/viewer/``.

Contenuto: ``index.html`` (MapLibre), ``layers.json`` (solo i layer che esistono davvero),
``xyz/<layer>/{z}/{x}/{y}.png`` e ``vector/{candidates,osm_highways}.geojson`` in EPSG:4326.
Si apre con ``morloquio view --run-id <id>``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd

from morloquio.candidates.vectorize import core_polygon
from morloquio.common.aoi import AOI
from morloquio.common.cache import params_hash
from morloquio.common.config import pipeline_config
from morloquio.common.grid import tiles_in_bbox
from morloquio.common.paths import interim_dir, processed_dir, runs_dir
from morloquio.common.runlog import get_logger
from morloquio.sources.catalog import load_catalog
from morloquio.viewer.tiles import render_xyz

STATIC = Path(__file__).parent / "static"
OSM_COLUMNS = ["osm_id", "highway", "surface", "tracktype", "access"]
RASTER_ZOOMS = range(12, 18)  # z17 ≈ 1 m/px alla latitudine del pilota: oltre, MapLibre ingrandisce
ORTHO_ZOOMS = range(12, 19)  # ortofoto a 0,5 m

log = get_logger("viewer.build")


def _raster_specs(run_id: str) -> list[tuple[str, str, str, range, callable]]:
    terrain = processed_dir("terrain") / params_hash(pipeline_config("terrain"))
    run = runs_dir() / run_id
    return [
        ("ortho", "Ortofoto", "rgb", ORTHO_ZOOMS, lambda t: processed_dir("cube") / t / "ortho.tif"),
        ("hillshade", "Ombreggiatura LiDAR", "gray", RASTER_ZOOMS, lambda t: terrain / t / "hillshade.tif"),
        ("lrm", "Rilievo locale (LRM)", "gray", RASTER_ZOOMS, lambda t: terrain / t / "lrm.tif"),
        ("score", "Punteggio Stadio A", "heat", RASTER_ZOOMS, lambda t: run / "stage_a" / t / "score.tif"),
    ]


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gdf.to_crs("EPSG:4326").to_json(drop_id=True), encoding="utf-8")
    return len(gdf)


def _attributions() -> list[str]:
    accepted = load_catalog().accepted
    wanted = ["osm_geofabrik_sud", "pst_lidar", "sentinel2_c1_l2a"]
    return [accepted[s].attribution for s in wanted if s in accepted]


def build_viewer(run_id: str, aoi: AOI) -> Path:
    run = runs_dir() / run_id
    out = run / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    tiles = tiles_in_bbox(aoi.bbox)

    raster = []
    for layer_id, name, mode, zooms, path_for in _raster_specs(run_id):
        found = [(t, path_for(t)) for t in tiles if path_for(t).exists()]
        target = out / "xyz" / layer_id
        shutil.rmtree(target, ignore_errors=True)  # niente tile vecchie di un run precedente
        if not found:
            continue
        cores = [core_polygon(t, aoi.work_crs) for t, _ in found]
        count = render_xyz([p for _, p in found], target, zooms=zooms, mode=mode, cores=cores)
        if count:
            raster.append(
                {
                    "id": layer_id,
                    "name": name,
                    "mode": mode,
                    "tiles": f"xyz/{layer_id}/{{z}}/{{x}}/{{y}}.png",
                    "minzoom": zooms.start,
                    "maxzoom": zooms.stop - 1,
                    "count": count,
                }
            )
    order = ["hillshade", "lrm", "score", "ortho"]  # ordine nel pannello: prima il terreno
    raster.sort(key=lambda r: order.index(r["id"]))

    vector = []
    cand = run / "stage_a" / "candidates_merged.gpkg"
    if cand.exists():
        n = _write_geojson(gpd.read_file(cand), out / "vector" / "candidates.geojson")
        vector.append(
            {"id": "candidates", "name": "Candidati Stadio A", "url": "vector/candidates.geojson", "count": n}
        )
    osm = interim_dir("osm") / f"{aoi.name}.gpkg"
    if osm.exists():
        hw = gpd.read_file(osm, layer="highways")
        hw = hw[[c for c in OSM_COLUMNS if c in hw.columns] + ["geometry"]]
        n = _write_geojson(hw, out / "vector" / "osm_highways.geojson")
        vector.append(
            {"id": "osm_highways", "name": "OSM attuale (vie)", "url": "vector/osm_highways.geojson", "count": n}
        )

    ev = run / "eval" / "stage_a.json"
    w, s, e, n_ = aoi.core_bbox
    layers = {
        "run_id": run_id,
        "aoi": aoi.name,
        "center": [round((w + e) / 2, 6), round((s + n_) / 2, 6)],
        "zoom": 15,
        "raster": raster,
        "vector": vector,
        "eval": json.loads(ev.read_text()) if ev.exists() else None,
        "attribution": _attributions(),
    }
    (out / "layers.json").write_text(json.dumps(layers, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(STATIC / "index.html", out / "index.html")
    log.info("visualizzatore costruito", extra={"run_id": run_id, "raster": [r["id"] for r in raster]})
    return out
