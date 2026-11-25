import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import LineString, box

from morloquio.common.aoi import AOI
from morloquio.eval.osm_recall import (
    EXCLUDED_HIGHWAY,
    completeness,
    correctness,
    evaluate_run,
    lidar_mask,
    reference_ways,
    report_by_class,
)
from morloquio.ingest.cube import tile_grid

CRS = "EPSG:32633"


def _gdf(lines, **cols):
    return gpd.GeoDataFrame(cols, geometry=lines, crs=CRS)


REF = _gdf([LineString([(0, 0), (100, 0)])], highway=["path"])


def test_full_coverage_is_one():
    cand = _gdf([LineString([(0, 1), (100, 1)])])
    assert completeness(REF, cand) == pytest.approx(1.0)
    assert correctness(cand, REF) == pytest.approx(1.0)


def test_half_coverage_is_half():
    cand = _gdf([LineString([(0, 0), (45, 0)])])  # con buffer 5 m copre il riferimento fino a x = 50
    assert completeness(REF, cand) == pytest.approx(0.5)
    assert correctness(cand, REF) == pytest.approx(1.0)


def test_offset_beyond_buffer_is_zero():
    cand = _gdf([LineString([(0, 7), (100, 7)])])
    assert completeness(REF, cand, buffer_m=5.0) == 0.0
    assert correctness(cand, REF, buffer_m=5.0) == 0.0


def test_empty_inputs():
    empty = _gdf([])
    assert completeness(REF, empty) == 0.0
    assert np.isnan(completeness(empty, REF)) and np.isnan(correctness(empty, REF))


def test_reference_excludes_bridges_tunnels_indoor():
    lines = [LineString([(i, 0), (i, 10)]) for i in range(8)]
    hw = gpd.GeoDataFrame(
        {
            "osm_id": range(8),
            "highway": ["path", "track", "footway", "service", "proposed", "path", "steps", "elevator"],
            "bridge": ["yes", None, None, None, None, "no", None, None],
            "tunnel": [None, "culvert", None, None, None, None, None, None],
            "indoor": [None, None, "yes", None, None, None, None, None],
            "layer": [None, None, None, "-1", None, None, "1", None],
        },
        geometry=lines,
        crs=CRS,
    )
    ref = reference_ways(hw)
    assert sorted(ref.osm_id) == [5, 6]
    assert {"elevator", "proposed", "construction"} <= EXCLUDED_HIGHWAY


def test_mask_limits_reference():
    mask = box(-10, -10, 50, 10)
    cand = _gdf([LineString([(0, 0), (45, 0)])])
    assert completeness(REF, cand, mask=mask) == pytest.approx(1.0)  # conta solo 0..50, tutta coperta
    outside = _gdf([LineString([(60, 0), (100, 0)])])
    assert correctness(outside, REF, mask=mask) != correctness(outside, REF)
    assert np.isnan(correctness(outside, REF, mask=mask))


def test_report_by_class():
    ref = _gdf(
        [LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)])],
        highway=["path", "track"],
    )
    cand = _gdf([LineString([(0, 0), (100, 0)])])
    rep = report_by_class(ref, cand, 5.0, None)
    assert list(rep.columns) == ["highway", "ref_km", "matched_km", "completeness"]
    rows = rep.set_index("highway")
    assert rows.loc["path", "completeness"] == pytest.approx(1.0)
    assert rows.loc["track", "completeness"] == pytest.approx(0.0)
    assert rows.loc["track", "ref_km"] == pytest.approx(0.1)


@pytest.fixture
def pilot(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    aoi = AOI("t", (15.67, 38.07, 15.69, 38.08), (15.67, 38.07, 15.68, 38.08), CRS)
    # tile T1 con DTM valido solo nella metà ovest; la tile est è T3
    t1 = "N3807E01567"
    transform, width, height = tile_grid(t1, CRS, 1.0)
    dtm = np.full((height, width), 100.0, dtype="float32")
    dtm[:, width // 2 :] = np.nan
    d = tmp_path / "data" / "processed" / "cube" / t1
    d.mkdir(parents=True)
    with rasterio.open(
        d / "dtm.tif",
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=float("nan"),
    ) as ds:
        ds.write(dtm, 1)
    (d / "meta.json").write_text(json.dumps({"tile_id": t1, "tier": "T1"}))
    d3 = tmp_path / "data" / "processed" / "cube" / "N3807E01568"
    d3.mkdir(parents=True)
    (d3 / "meta.json").write_text(json.dumps({"tile_id": "N3807E01568", "tier": "T3"}))
    return aoi, transform, width, height


def test_lidar_mask_is_valid_dtm_inside_t1_cores(pilot):
    aoi, transform, width, height = pilot
    mask = lidar_mask(aoi)
    from morloquio.candidates.vectorize import core_polygon

    core = core_polygon("N3807E01567", CRS)
    assert core.buffer(1).covers(mask)
    assert mask.area == pytest.approx(core.area / 2, rel=0.1)


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_evaluate_run_writes_reports(pilot, tmp_path, stage):
    aoi, transform, width, height = pilot
    x0 = transform.c + 200  # dentro la metà valida della tile T1
    y_mid = transform.f - height / 2
    ref = gpd.GeoDataFrame(
        {
            "osm_id": [1, 2],
            "highway": ["path", "track"],
            "bridge": [None, None],
            "tunnel": [None, None],
            "indoor": [None, None],
            "layer": [None, None],
        },
        geometry=[
            LineString([(x0, y_mid), (x0 + 100, y_mid)]),
            LineString([(x0, y_mid - 200), (x0 + 100, y_mid - 200)]),
        ],
        crs=CRS,
    ).to_crs("EPSG:4326")
    osm = tmp_path / "data" / "interim" / "osm"
    osm.mkdir(parents=True)
    ref.to_file(osm / "t.gpkg", layer="highways", engine="pyogrio")
    run = tmp_path / "runs" / "r1" / stage
    run.mkdir(parents=True)
    cand = gpd.GeoDataFrame(
        {"length_m": [100.0], "mean_score": [0.8], "max_score": [0.9]},
        geometry=[LineString([(x0, y_mid + 1), (x0 + 100, y_mid + 1)])],
        crs=CRS,
    )
    cand.to_file(run / "candidates_merged.gpkg", layer="candidates", engine="pyogrio")
    out = evaluate_run("r1", aoi, stage=stage)
    assert out == tmp_path / "runs" / "r1" / "eval" / f"{stage}.json"
    rep = json.loads(out.read_text())
    assert rep["completeness"] == pytest.approx(0.5, abs=0.01)
    assert rep["correctness"] == pytest.approx(1.0, abs=0.01)
    assert rep["exit_threshold"] == 0.98 and rep["passed"] is False
    by_class = pd.read_csv(out.parent / f"{stage}_by_class.csv")
    assert set(by_class.highway) == {"path", "track"}
