import json

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from morloquio.common.grid import tile_bounds
from morloquio.ingest.restore import nn_fraction, restore_file, restored_source

TILE = "N3807E01567"
CRS = "EPSG:32633"
OFFSET = (0.05, 0.85)  # allineamento dei nodi originali osservato sui file PST del contratto 140


def terrain(x, y):
    """Terreno liscio con versanti fino a ~30°: quota in metri dalle coordinate UTM."""
    x0, y0 = 559000.0, 4214000.0
    return 300 + 0.35 * (x - x0) + 0.12 * (y - y0) + 8 * np.sin((x - x0) / 37.0) * np.cos((y - y0) / 53.0)


def _write_geo(path, data, nodata=-9999.0):
    w, s, e, n = tile_bounds(TILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(w, n, 1e-5, 1e-5),
        nodata=nodata,
    ) as ds:
        ds.write(data.astype("float32"), 1)
    return path


def _geo_centers(size=1000):
    w, s, e, n = tile_bounds(TILE)
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    return Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(w + cols * 1e-5, n - rows * 1e-5)


def nn_source(path, offset=OFFSET, hole=None):
    """Simula il prodotto MASE: ogni pixel a 1e-5° prende il valore del nodo UTM 1 m più vicino."""
    x, y = _geo_centers()
    ox, oy = offset
    nx = np.round(x - ox) + ox
    ny = np.round(y - oy) + oy
    data = terrain(nx, ny)
    if hole is not None:
        r0, r1, c0, c1 = hole
        data[r0:r1, c0:c1] = -9999.0
    return _write_geo(path, data)


def smooth_source(path):
    x, y = _geo_centers()
    return _write_geo(path, terrain(x, y))


def test_nn_fraction_detects_nearest_neighbour(tmp_path):
    with rasterio.open(nn_source(tmp_path / "nn.tif")) as ds:
        assert nn_fraction(ds.read(1, masked=True)) > 0.08  # ~12%: una colonna duplicata ogni ~8
    with rasterio.open(smooth_source(tmp_path / "smooth.tif")) as ds:
        assert nn_fraction(ds.read(1, masked=True)) < 0.001


def test_restore_finds_offset_and_true_nodes(tmp_path):
    src = nn_source(tmp_path / "nn.tif", hole=(400, 450, 0, 1000))
    info = restore_file(src, tmp_path / "out.tif", CRS)
    assert info["restored"] is True
    ox, oy = info["offset"]
    assert min(abs(ox - OFFSET[0]), 1 - abs(ox - OFFSET[0])) <= 0.03
    assert min(abs(oy - OFFSET[1]), 1 - abs(oy - OFFSET[1])) <= 0.03
    with rasterio.open(tmp_path / "out.tif") as ds:
        z = ds.read(1)
        t = ds.transform
        assert ds.crs.to_epsg() == 32633 and ds.res == (1.0, 1.0)
    rows, cols = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    truth = terrain(x, y)
    valid = ~np.isnan(z)
    err = np.abs(z[valid] - truth[valid])
    assert np.mean(err < 1e-3) > 0.85  # i nodi campionati tornano esatti
    assert np.sqrt(np.mean(err**2)) < 0.03  # le righe saltate sono interpolate bene
    assert np.isnan(z).any()  # il buco nei dati resta buco
    assert json.loads((tmp_path / "out.json").read_text())["offset"] == info["offset"]


def test_smooth_source_is_left_alone(tmp_path):
    src = smooth_source(tmp_path / "smooth.tif")
    info = restore_file(src, tmp_path / "out.tif", CRS)
    assert info["restored"] is False and not (tmp_path / "out.tif").exists()


def test_restored_source_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    src = nn_source(tmp_path / "data" / "raw" / "pst_lidar" / "dtm_1x1" / "D38071567_0101_c140.tif")
    out = restored_source(src, CRS)
    assert out == tmp_path / "data" / "interim" / "pst_restored" / "dtm_1x1" / "D38071567_0101_c140.tif"
    mtime = out.stat().st_mtime_ns
    assert restored_source(src, CRS) == out and out.stat().st_mtime_ns == mtime
    smooth = smooth_source(tmp_path / "data" / "raw" / "pst_lidar" / "dtm_1x1" / "D38071568_0101_c140.tif")
    assert restored_source(smooth, CRS) == smooth
    assert restored_source(smooth, CRS) == smooth  # la decisione "non serve" è anch'essa in cache


def test_restore_does_not_touch_other_writers_part_file(tmp_path):
    # un altro worker sta scrivendo la stessa tavola: il suo file temporaneo non va toccato
    dest = tmp_path / "out.tif"
    foreign = dest.with_name(dest.name + ".part")
    foreign.write_bytes(b"in scrittura da un altro processo")
    info = restore_file(nn_source(tmp_path / "nn.tif"), dest, CRS)
    assert info["restored"] is True and dest.exists()
    assert foreign.read_bytes() == b"in scrittura da un altro processo"
    assert not list(tmp_path.glob("out.tif.*.part"))


@pytest.mark.parametrize("offset", [(0.5, 0.5), (0.3, 0.1)])
def test_restore_other_alignments(tmp_path, offset):
    info = restore_file(nn_source(tmp_path / "nn.tif", offset=offset), tmp_path / "out.tif", CRS)
    ox, oy = info["offset"]
    assert min(abs(ox - offset[0]), 1 - abs(ox - offset[0])) <= 0.03
    assert min(abs(oy - offset[1]), 1 - abs(oy - offset[1])) <= 0.03
