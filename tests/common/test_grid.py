import pytest

from morloquio.common.grid import (
    neighbors,
    pst_tavola_for,
    tile_bounds,
    tile_id_for,
    tile_id_from_tavola,
    tiles_in_bbox,
)


def test_tile_id_pilot_point():
    assert tile_id_for(38.0756, 15.6777) == "N3807E01567"


def test_tile_id_exact_boundary_is_stable():
    # 38.07 * 100 = 3806.9999… in virgola mobile: deve restare nella cella 3807
    assert tile_id_for(38.07, 15.67) == "N3807E01567"


def test_tile_id_southern_western_hemisphere():
    assert tile_id_for(-0.005, -0.005) == "S0001W00001"


def test_tile_bounds():
    assert tile_bounds("N3807E01567") == pytest.approx((15.67, 38.07, 15.68, 38.08))


def test_tile_bounds_roundtrip_negative():
    w, s, e, n = tile_bounds("S0001W00001")
    assert tile_id_for((s + n) / 2, (w + e) / 2) == "S0001W00001"


def test_pilot_bbox_has_117_cells():
    tiles = tiles_in_bbox((15.62, 38.03, 15.75, 38.12))
    assert len(tiles) == 117
    assert tiles == sorted(tiles)
    assert "N3807E01567" in tiles
    assert "N3812E01562" not in tiles  # bordo nord escluso


def test_neighbors():
    nb = neighbors("N3807E01567")
    assert len(nb) == 8
    assert "N3808E01568" in nb and "N3806E01566" in nb and "N3807E01567" not in nb


def test_pst_tavola_roundtrip():
    assert pst_tavola_for("N3807E01567") == "D38071567_0101"
    assert tile_id_from_tavola("D38071567_0101") == "N3807E01567"


def test_tile_id_from_bad_tavola():
    with pytest.raises(ValueError):
        tile_id_from_tavola("X1234")
