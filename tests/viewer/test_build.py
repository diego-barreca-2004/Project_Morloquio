import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from morloquio.common.aoi import AOI
from morloquio.common.cache import params_hash
from morloquio.common.config import pipeline_config
from morloquio.ingest.cube import tile_grid
from morloquio.viewer.build import build_viewer

CRS = "EPSG:32633"
TILE = "N3807E01567"
_T, _, _ = tile_grid(TILE, CRS, 1.0)
X0, Y0 = _T.c + 300, _T.f - 300  # raster finto dentro il nucleo della tile


def _tif(path, data, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        crs=CRS,
        transform=from_origin(X0, Y0, 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)


def test_build_viewer_writes_layers_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    aoi = AOI("t", (15.67, 38.07, 15.68, 38.08), (15.67, 38.07, 15.68, 38.08), CRS)
    terrain = tmp_path / "data" / "processed" / "terrain" / params_hash(pipeline_config("terrain")) / TILE
    _tif(terrain / "hillshade.tif", np.full((200, 200), 180, dtype="uint8"), 0)
    run = tmp_path / "runs" / "r1"
    _tif(run / "stage_a" / TILE / "score.tif", np.full((200, 200), 0.7, dtype="float32"), float("nan"))
    line = LineString([(X0 + 10, Y0 - 10), (X0 + 190, Y0 - 190)])
    gpd.GeoDataFrame(
        {"length_m": [line.length], "mean_score": [0.7], "max_score": [0.9]}, geometry=[line], crs=CRS
    ).to_file(run / "stage_a" / "candidates_merged.gpkg", layer="candidates", engine="pyogrio")
    osm = tmp_path / "data" / "interim" / "osm"
    osm.mkdir(parents=True)
    gpd.GeoDataFrame({"osm_id": [1], "highway": ["path"], "surface": [None]}, geometry=[line], crs=CRS).to_crs(
        "EPSG:4326"
    ).to_file(osm / "t.gpkg", layer="highways", engine="pyogrio")
    (run / "eval").mkdir()
    (run / "eval" / "stage_a.json").write_text(json.dumps({"completeness": 0.5, "correctness": 0.9}))

    out = build_viewer("r1", aoi)

    assert out == run / "viewer"
    layers = json.loads((out / "layers.json").read_text())
    assert [r["id"] for r in layers["raster"]] == ["hillshade", "score"]  # niente LRM né ortofoto: non esistono
    assert [v["id"] for v in layers["vector"]] == ["candidates", "osm_highways"]
    assert layers["center"] == [15.675, 38.075] and layers["eval"]["completeness"] == 0.5
    assert list((out / "xyz" / "hillshade").rglob("*.png")) and list((out / "xyz" / "score").rglob("*.png"))
    gj = json.loads((out / "vector" / "candidates.geojson").read_text())
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0]
    assert 15.6 < lon < 15.7 and 38.0 < lat < 38.1 and gj["features"][0]["properties"]["mean_score"] == 0.7
    html = (out / "index.html").read_text()
    assert "maplibre-gl@4.7.1" in html and "© OpenStreetMap contributors" in html
    assert "strava" not in html.lower() and "google" not in html.lower()


def test_build_viewer_without_eval_or_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    aoi = AOI("t", (15.67, 38.07, 15.68, 38.08), (15.67, 38.07, 15.68, 38.08), CRS)
    (tmp_path / "runs" / "r2").mkdir(parents=True)
    out = build_viewer("r2", aoi)
    layers = json.loads((out / "layers.json").read_text())
    assert layers["raster"] == [] and layers["vector"] == [] and layers["eval"] is None


def test_build_viewer_shows_stage_b(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    aoi = AOI("t", (15.67, 38.07, 15.68, 38.08), (15.67, 38.07, 15.68, 38.08), CRS)
    run = tmp_path / "runs" / "r3"
    _tif(run / "stage_a" / TILE / "score.tif", np.full((200, 200), 0.7, dtype="float32"), float("nan"))
    _tif(run / "stage_b" / TILE / "score.tif", np.full((200, 200), 0.9, dtype="float32"), float("nan"))
    line = LineString([(X0 + 10, Y0 - 10), (X0 + 190, Y0 - 190)])
    gpd.GeoDataFrame(
        {"length_m": [line.length], "mean_score": [0.9], "max_score": [0.95]}, geometry=[line], crs=CRS
    ).to_file(run / "stage_b" / "candidates_merged.gpkg", layer="candidates", engine="pyogrio")
    (run / "eval").mkdir()
    (run / "eval" / "stage_a.json").write_text(json.dumps({"stage": "stage_a", "completeness": 0.2}))
    (run / "eval" / "stage_b.json").write_text(json.dumps({"stage": "stage_b", "completeness": 0.8}))

    layers = json.loads((build_viewer("r3", aoi) / "layers.json").read_text())

    assert [r["id"] for r in layers["raster"]] == ["score", "score_b"]
    assert [v["id"] for v in layers["vector"]] == ["candidates_b"]
    assert layers["eval"]["stage"] == "stage_b"  # il pannello mostra lo stadio più avanzato
    assert set(layers["evals"]) == {"stage_a", "stage_b"}
