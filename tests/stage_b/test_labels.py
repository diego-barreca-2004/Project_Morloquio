import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from morloquio.stage_b.labels import rasterize_labels

T = from_origin(0, 100, 1, 1)  # pixel (r, c) ha centro in (c + 0,5, 99,5 − r)
SHAPE = (100, 100)
PARAMS = {
    "half_width_m": {"track": 1.5, "primary": 4.0, "default": 1.5},
    "ignore_band_m": 3.0,
    "unlabeled_weight": 0.1,
}
CORE = box(0, 0, 100, 100)
VALID = np.ones(SHAPE, bool)


def _ways(highway, x=50.3):
    return gpd.GeoDataFrame({"highway": [highway]}, geometry=[LineString([(x, -5), (x, 105)])], crs="EPSG:32633")


def test_label_width_by_class():
    target, _ = rasterize_labels(_ways("track"), T, SHAPE, CORE, VALID, PARAMS)
    assert np.flatnonzero(target[50]).tolist() == [49, 50, 51]
    target, _ = rasterize_labels(_ways("primary"), T, SHAPE, CORE, VALID, PARAMS)
    assert np.flatnonzero(target[50]).tolist() == list(range(46, 54))
    target, _ = rasterize_labels(_ways("bus_guideway"), T, SHAPE, CORE, VALID, PARAMS)
    assert np.flatnonzero(target[50]).tolist() == [49, 50, 51]  # classe sconosciuta → default


def test_ignore_band_has_zero_weight():
    target, weight = rasterize_labels(_ways("track"), T, SHAPE, CORE, VALID, PARAMS)
    assert weight.dtype == np.float32 and target.dtype == np.uint8
    assert weight[50, 50] == 1.0  # positivo
    assert weight[50, 47] == 0.0 and weight[50, 46] == 0.0  # entro 4,5 m: incertezza di posizione
    assert weight[50, 45] == np.float32(0.1)  # oltre: non etichettato


def test_outside_core_and_invalid_have_zero_weight():
    valid = VALID.copy()
    valid[:10] = False
    _, weight = rasterize_labels(_ways("track"), T, SHAPE, box(0, 0, 60, 100), valid, PARAMS)
    assert weight[50, 70:].max() == 0.0  # fuori nucleo: potrebbe essere il nucleo di una tile di test
    assert weight[:10].max() == 0.0  # DTM assente
    assert weight[50, 50] == 1.0


def test_tile_without_osm_is_all_unlabeled():
    empty = gpd.GeoDataFrame({"highway": []}, geometry=[], crs="EPSG:32633")
    target, weight = rasterize_labels(empty, T, SHAPE, CORE, VALID, PARAMS)
    assert target.max() == 0 and np.allclose(weight, 0.1)
