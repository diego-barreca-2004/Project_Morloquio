import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

from morloquio.candidates.reconnect import reconnect

T = from_origin(0, 200, 1, 1)  # pixel (r, c) ha centro in (c + 0,5, 199,5 − r)
PARAMS = {"search_m": 50.0, "min_graph_m": 100.0, "min_mean_score": 0.15, "max_detour": 1.8, "sample_m": 2.0}


def _gdf(lines):
    n = len(lines)
    return gpd.GeoDataFrame(
        {"length_m": [g.length for g in lines], "mean_score": [0.8] * n, "max_score": [0.9] * n},
        geometry=lines,
        crs="EPSG:32633",
    )


def _score(gap_value):
    s = np.zeros((200, 200), "float32")
    s[99, 10:131] = 0.8  # riga y = 100,5
    s[99, 61:80] = gap_value  # il buco tra le due linee
    return s


LINES = [LineString([(10.5, 100.5), (60.5, 100.5)]), LineString([(80.5, 100.5), (130.5, 100.5)])]


def test_gap_with_weak_evidence_is_bridged():
    out = reconnect(_gdf(LINES), _score(0.3), T, PARAMS)
    assert len(out) == 3
    new = out.iloc[2]
    assert 18 <= new.geometry.length <= 24 and new.mean_score >= 0.25
    xs = [x for x, _ in new.geometry.coords]
    assert min(xs) <= 61 and max(xs) >= 80


def test_gap_without_evidence_stays_open():
    assert len(reconnect(_gdf(LINES), _score(0.0), T, PARAMS)) == 2


def test_lines_already_connected_nearby_are_not_bridged():
    corner = [LineString([(40.5, 100.5), (60.5, 100.5)]), LineString([(60.5, 100.5), (60.5, 130.5)])]
    assert len(reconnect(_gdf(corner), np.full((200, 200), 0.5, "float32"), T, PARAMS)) == 2


def test_empty_input_is_returned_unchanged():
    empty = _gdf([])
    assert reconnect(empty, _score(0.3), T, PARAMS).empty
