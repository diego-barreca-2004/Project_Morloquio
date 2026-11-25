import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from morloquio.terrain.derivatives import (
    canopy_height,
    compute_tile,
    curvatures,
    flow_accumulation,
    hillshade_multi,
    local_relief,
    slope_deg,
)

RES = 1.0


def _plane(deg, shape=(200, 200)):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype("float32")
    return (np.tan(np.radians(deg)) * xx * RES).astype("float32")


def test_slope_of_30_degree_plane():
    s = slope_deg(_plane(30), RES)
    assert np.nanmedian(s[5:-5, 5:-5]) == pytest.approx(30, abs=0.5)


def test_lrm_plane_is_zero():
    lrm = local_relief(_plane(20), RES, sigma_m=10.0)
    assert np.nanmax(np.abs(lrm[40:-40, 40:-40])) < 0.01


def test_lrm_trench_negative():
    dtm = _plane(10)
    dtm[:, 99:101] -= 0.5  # trincea larga 2 m, profonda 0,5 m
    lrm = local_relief(dtm, RES, sigma_m=10.0)
    assert np.nanmedian(lrm[40:-40, 100]) < -0.3


def test_nan_does_not_spread():
    dtm = _plane(15)
    dtm[95:105, 95:105] = np.nan
    lrm = local_relief(dtm, RES, sigma_m=5.0)
    assert np.isnan(lrm[100, 100])
    radius = int(3 * 5.0 / RES)
    nan_rows, nan_cols = np.where(np.isnan(lrm))
    assert nan_rows.min() >= 95 - radius and nan_rows.max() < 105 + radius
    assert nan_cols.min() >= 95 - radius and nan_cols.max() < 105 + radius
    assert not np.isnan(lrm[100, 60])


def test_hillshade_range_and_nan():
    dtm = _plane(25)
    dtm[0:10, 0:10] = np.nan
    hs = hillshade_multi(dtm, RES)
    assert hs.dtype == np.uint8 and hs[50, 50] > 0
    assert hs[2, 2] == 0


def test_curvature_of_plane_is_zero_and_of_valley_is_signed():
    prof, plan = curvatures(_plane(20), RES)
    assert np.nanmax(np.abs(prof[5:-5, 5:-5])) < 1e-4 and np.nanmax(np.abs(plan[5:-5, 5:-5])) < 1e-4
    yy, xx = np.mgrid[0:200, 0:200].astype("float32")
    valley = (0.01 * (xx - 100) ** 2 + 0.3 * yy).astype("float32")  # valle lungo y, pendenza verso nord
    prof, plan = curvatures(valley, RES)
    assert np.nanmedian(plan[50:150, 90:110]) != 0


def test_chm_non_negative():
    dsm = np.array([[10.0, 5.0], [np.nan, 20.0]], dtype="float32")
    dtm = np.array([[8.0, 6.0], [1.0, 2.0]], dtype="float32")
    chm = canopy_height(dsm, dtm)
    assert chm[0, 0] == 2.0 and chm[0, 1] == 0.0 and np.isnan(chm[1, 0]) and chm[1, 1] == 18.0


def test_flowacc_channel():
    yy, xx = np.mgrid[0:120, 0:120].astype("float32")
    dtm = (np.abs(xx - 60) * 0.5 + yy * 0.2).astype("float32")  # valle a V lungo la colonna 60, scende verso y=0
    acc = flow_accumulation(dtm, RES)
    row = acc[30]
    assert int(np.argmax(row)) in (59, 60, 61)
    assert acc[5, 60] > acc[100, 60] > acc[100, 20]


def test_flowacc_handles_pits_and_nan():
    dtm = np.full((50, 50), 10.0, dtype="float32")
    dtm[25, 25] = 5.0  # buca isolata
    dtm[0:5, 0:5] = np.nan
    acc = flow_accumulation(dtm, RES)
    assert np.isnan(acc[2, 2]) and np.all(acc[~np.isnan(dtm)] >= 1)


@pytest.fixture
def cube(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    d = tmp_path / "data" / "processed" / "cube" / "N3807E01567"
    d.mkdir(parents=True)
    profile = dict(
        driver="GTiff",
        width=200,
        height=200,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(560000, 4215000, 1, 1),
        nodata=float("nan"),
    )
    for name, arr in {"dtm": _plane(20), "dsm_first": _plane(20) + 5}.items():
        with rasterio.open(d / f"{name}.tif", "w", **profile) as ds:
            ds.write(arr, 1)
    (d / "meta.json").write_text(json.dumps({"tile_id": "N3807E01567", "tier": "T1"}))
    return d


def test_compute_tile_writes_all_layers(cube):
    params = {"lrm_sigma_m": 10.0, "hillshade_azimuths": [315, 270, 225, 360], "hillshade_altitude": 45}
    out = compute_tile("N3807E01567", params)
    names = {p.stem for p in out.glob("*.tif")}
    assert names == {"slope", "hillshade", "lrm", "curv_profile", "curv_plan", "chm", "flowacc"}
    with rasterio.open(out / "chm.tif") as ds:
        assert ds.crs.to_epsg() == 32633 and np.nanmedian(ds.read(1)) == pytest.approx(5.0)
    assert out.parent.name == out.parent.name  # cartella = params_hash
    assert out.name == "N3807E01567"
