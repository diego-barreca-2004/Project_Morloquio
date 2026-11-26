import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from morloquio.common.aoi import AOI
from morloquio.common.grid import tile_bounds
from morloquio.ingest.cube import HALO_M, build_cube, pst_sources_for, tier_for, tile_grid

TILE = "N3807E01567"
AOI_ = AOI("test", (15.62, 38.03, 15.75, 38.12), (15.66, 38.06, 15.69, 38.09), "EPSG:32633")


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "data"


def _plane_utm(x, y):
    """Superficie inclinata definita in coordinate UTM 33N: quota = 0.1*x + 0.05*y (metri)."""
    return 0.1 * (x - 560000) + 0.05 * (y - 4210000) + 100.0


def _write_geo_tile(path, tile_id, nodata=-9999.0, hole=None):
    """Tavola in EPSG:4326 a 1e-5° come quelle PST, con valori del piano UTM."""
    from pyproj import Transformer

    w, s, e, n = tile_bounds(tile_id)
    size = 1000
    res = (e - w) / size
    lon = w + (np.arange(size) + 0.5) * res
    lat = n - (np.arange(size) + 0.5) * res
    lon2, lat2 = np.meshgrid(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True).transform(lon2, lat2)
    data = _plane_utm(x, y).astype("float32")
    if hole is not None:
        r0, r1, c0, c1 = hole
        data[r0:r1, c0:c1] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(w, n, res, res),
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)
    return path


def _write_utm_tile(path, tile_id):
    w, s, e, n = transform_bounds("EPSG:4326", "EPSG:32633", *tile_bounds(tile_id))
    left, top = np.floor(w) - 100, np.ceil(n) + 100
    width, height = int(e - w) + 200, int(n - s) + 200
    xs = left + np.arange(width) + 0.5
    ys = top - np.arange(height) - 0.5
    x2, y2 = np.meshgrid(xs, ys)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(left, top, 1, 1),
        nodata=-9999.0,
    ) as ds:
        ds.write(_plane_utm(x2, y2).astype("float32"), 1)
    return path


def test_tile_grid_snapped_and_haloed():
    transform, width, height = tile_grid(TILE, "EPSG:32633", 1.0)
    assert transform.a == 1.0 and transform.e == -1.0
    assert transform.c == np.floor(transform.c) and transform.f == np.floor(transform.f)
    w, s, e, n = transform_bounds("EPSG:4326", "EPSG:32633", *tile_bounds(TILE), densify_pts=21)
    assert abs(width - ((e - w) + 2 * HALO_M)) <= 2
    assert abs(height - ((n - s) + 2 * HALO_M)) <= 2


def test_tier_for():
    assert tier_for(True, True) == "T1" and tier_for(True, False) == "T1"
    assert tier_for(False, True) == "T2" and tier_for(False, False) == "T3"


def test_reprojects_from_other_crs(tmp_path):
    geo = _write_geo_tile(tmp_path / "a" / "geo.tif", TILE)
    utm = _write_utm_tile(tmp_path / "b.tif", TILE)
    out_a = build_cube(TILE, AOI_, {"dtm_1x1": [geo]}, [])
    a = rasterio.open(out_a / "dtm.tif").read(1)
    out_b = build_cube(TILE, AOI_, {"dtm_1x1": [utm]}, [])
    b = rasterio.open(out_b / "dtm.tif").read(1)
    both = ~np.isnan(a) & ~np.isnan(b)
    assert both.sum() > 500_000
    assert np.median(np.abs(a[both] - b[both])) < 0.05


def test_neighbors_fill_halo(tmp_path):
    t1 = _write_geo_tile(tmp_path / "t1.tif", TILE)
    t2 = _write_geo_tile(tmp_path / "t2.tif", "N3807E01568")  # vicino a est
    out = build_cube(TILE, AOI_, {"dtm_1x1": [t1, t2]}, [])
    dtm = rasterio.open(out / "dtm.tif").read(1)
    mid = dtm.shape[0] // 2
    assert not np.isnan(dtm[mid, -HALO_M // 2])  # alone est riempito dal vicino
    assert np.isnan(dtm[mid, HALO_M // 4])  # alone ovest senza vicino resta vuoto


def test_nodata_propagates(tmp_path):
    t = _write_geo_tile(tmp_path / "t.tif", TILE, hole=(400, 600, 400, 600))
    out = build_cube(TILE, AOI_, {"dtm_1x1": [t]}, [])
    meta = json.loads((out / "meta.json").read_text())
    dtm = rasterio.open(out / "dtm.tif").read(1)
    assert np.isnan(dtm).any()
    assert 0.93 < meta["valid_fraction"]["dtm"] < 0.97  # il buco è il 4% della tavola
    assert not np.any(dtm < -1000)


def test_tile_without_lidar_is_t3():
    out = build_cube(TILE, AOI_, {"dtm_1x1": [], "dsm_first_1x1": [], "dsm_last_1x1": []}, [])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tier"] == "T3"
    assert not (out / "dtm.tif").exists()


def test_meta_contents(tmp_path):
    t = _write_geo_tile(tmp_path / "t.tif", TILE)
    out = build_cube(TILE, AOI_, {"dtm_1x1": [t], "dsm_first_1x1": [t]}, [])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tile_id"] == TILE and meta["tier"] == "T1" and meta["work_crs"] == "EPSG:32633"
    assert meta["halo_m"] == HALO_M and set(meta["sources"]) == {"dtm", "dsm_first"}
    with rasterio.open(out / "dtm.tif") as ds:
        assert ds.crs.to_epsg() == 32633 and ds.dtypes[0] == "float32"
        assert ds.profile.get("compress", "").lower() == "deflate"


def test_cube_undoes_nearest_neighbour_resampling(data_dir):
    from tests.ingest.test_restore import nn_source, terrain

    src = nn_source(data_dir / "raw" / "pst_lidar" / "dtm_1x1" / "D38071567_0101_c140.tif")
    out = build_cube(TILE, AOI_, {"dtm_1x1": [src]}, [])
    with rasterio.open(out / "dtm.tif") as ds:
        dtm = ds.read(1)
        t = ds.transform
    rows, cols = np.mgrid[0 : dtm.shape[0], 0 : dtm.shape[1]] + 0.5
    truth = terrain(t.c + cols * t.a, t.f + rows * t.e)
    valid = ~np.isnan(dtm)
    err = dtm[valid] - truth[valid]
    # riproiettare il nearest-neighbour lascia errori fino a pendenza × 0,5 m a righe regolari
    assert np.sqrt(np.mean(err**2)) < 0.03
    meta = json.loads((out / "meta.json").read_text())
    assert meta["restored"]["dtm"] == [True]


def test_zero_byte_sources_are_ignored(data_dir):
    root = data_dir / "raw" / "pst_lidar" / "dtm_1x1"
    root.mkdir(parents=True)
    (root / "D38071567_0101_c .tif").write_bytes(b"")
    assert pst_sources_for(TILE)["dtm_1x1"] == []


def test_pst_sources_for_picks_tile_and_neighbors(data_dir):
    root = data_dir / "raw" / "pst_lidar" / "dtm_1x1"
    root.mkdir(parents=True)
    for name in (
        "D38071567_0101_c140.tif",
        "D38071567_0101_c77.tif",
        "D38081568_0101_c140.tif",
        "D38101570_0101_c140.tif",
    ):
        (root / name).write_bytes(b"x")
    found = pst_sources_for(TILE)
    assert sorted(p.name for p in found["dtm_1x1"]) == [
        "D38071567_0101_c140.tif",
        "D38071567_0101_c77.tif",
        "D38081568_0101_c140.tif",
    ]
    assert found["dsm_first_1x1"] == []


def test_ortho_only_tile_is_t2(tmp_path):
    w, s, e, n = tile_bounds(TILE)
    size = 400
    res = (e - w) / size
    path = tmp_path / "ortho.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(w, n, res, res),
    ) as ds:
        ds.write(np.full((3, size, size), 120, dtype="uint8"))
    out = build_cube(TILE, AOI_, {}, [path])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tier"] == "T2" and meta["valid_fraction"]["ortho"] > 0.9
    with rasterio.open(out / "ortho.tif") as ds:
        assert ds.count == 3 and ds.res == (0.5, 0.5)
