import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from morloquio.terrain.derivatives import compute_tile

TERRAIN = {"lrm_sigma_m": 10.0, "hillshade_azimuths": [315, 270, 225, 360], "hillshade_altitude": 45}
TILES = ("N3807E01567", "N3807E01568")


def slope_with_trench(shape=(160, 160), deg=20.0, trench_col=80, seed=0):
    """Versante inclinato con rumore e una trincea N-S larga 3 m e profonda 0,5 m (una "via")."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype("float32")
    dtm = 300 + np.tan(np.radians(deg)) * xx + rng.normal(0, 0.03, shape).astype("float32")
    dtm[:, trench_col - 1 : trench_col + 2] -= 0.5
    return dtm


def write_cube(root, tile_id, dtm, origin=(560000.0, 4215000.0), canopy=8.0, last_gap=6.0):
    d = root / "data" / "processed" / "cube" / tile_id
    d.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=dtm.shape[1],
        height=dtm.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(*origin, 1, 1),
        nodata=float("nan"),
    )
    layers = {"dtm": dtm, "dsm_first": dtm + canopy, "dsm_last": dtm + canopy - last_gap}
    for name, arr in layers.items():
        with rasterio.open(d / f"{name}.tif", "w", **profile) as ds:
            ds.write(arr.astype("float32"), 1)
    (d / "meta.json").write_text(json.dumps({"tile_id": tile_id, "tier": "T1", "work_crs": "EPSG:32633"}))
    compute_tile(tile_id, TERRAIN)
    return d


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def two_cubes(data):
    a = slope_with_trench()
    a[:20, :20] = np.nan  # buco nel rilievo
    write_cube(data, TILES[0], a)
    write_cube(data, TILES[1], slope_with_trench(seed=1), origin=(560160.0, 4215000.0))
    return data


@pytest.fixture
def params():
    return {"terrain": TERRAIN, "features": {"tanh_div": 3.0}}
