import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin, xy
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from morloquio.candidates.vectorize import (
    core_polygon,
    crop_to_core,
    extract_candidates,
    merge_run,
    merge_tiles,
    vectorize_tile,
)
from morloquio.ingest.cube import tile_grid

CRS = "EPSG:32633"
T0 = from_origin(560000, 4215000, 1, 1)
LOW, HIGH = 0.05, 0.9


def _empty(n=200):
    return np.full((n, n), LOW, dtype="float32")


def test_straight_line_one_segment():
    score = _empty()
    rr, cc = np.mgrid[0:200, 0:200]
    score[(np.abs(rr - cc) <= 1) & (rr >= 20) & (rr <= 180)] = HIGH
    gdf = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].geom_type == "LineString"
    assert gdf.length_m.iloc[0] == pytest.approx(160 * np.sqrt(2), rel=0.05)
    assert gdf.mean_score.iloc[0] == pytest.approx(HIGH, abs=0.05) and gdf.max_score.iloc[0] == pytest.approx(HIGH)
    assert gdf.crs.to_epsg() == 32633


def _two_close_roads():
    """Due vie parallele a 9 m: tra le due la probabilità cala ma resta sopra soglia (macchia unica a 0,35)."""
    rr, cc = np.mgrid[0:200, 0:200].astype("float32")
    a = np.exp(-(((cc - 95.5) / 2.5) ** 2))
    b = np.exp(-(((cc - 104.5) / 2.5) ** 2))
    score = np.maximum(LOW, 0.6 * np.maximum(a, b) + 0.4 * np.exp(-(((cc - 100) / 6) ** 2)))
    score[(rr < 20) | (rr > 180)] = LOW
    return score.astype("float32")


def test_ridge_centerline_separates_close_parallel_roads():
    score = _two_close_roads()
    assert score[100, 100] >= 0.35 and score[100, 100] < score[100, 95]  # una sola macchia, ma due creste
    by_mask = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0)
    by_ridge = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0, centerline="ridge")
    cut = LineString([(560000, 4215000 - 100.5), (560200, 4215000 - 100.5)])  # traversale a metà via

    def xs(gdf):
        hits = [gdf.geometry.union_all().intersection(cut)]
        pts = [p for h in hits for p in getattr(h, "geoms", [h]) if not p.is_empty]
        return sorted(p.x - 560000 for p in pts)

    assert len(xs(by_mask)) == 1 and abs(xs(by_mask)[0] - 100.5) < 1  # asse mediano: a 4–5 m da entrambe le vie
    left, right = xs(by_ridge)
    assert abs(left - 96.5) < 1.5 and abs(right - 104.5) < 1.5


def test_ridge_centerline_keeps_single_road_whole():
    score = _empty()
    rr, cc = np.mgrid[0:200, 0:200]
    score[(np.abs(rr - cc) <= 2) & (rr >= 20) & (rr <= 180)] = HIGH
    gdf = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0, centerline="ridge")
    assert len(gdf) == 1 and gdf.length_m.iloc[0] == pytest.approx(160 * np.sqrt(2), rel=0.08)


def test_short_blobs_removed():
    score = _empty()
    score[50:54, 50:54] = HIGH
    gdf = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0)
    assert len(gdf) == 0
    assert list(gdf.columns) == ["length_m", "mean_score", "max_score", "geometry"]


def test_nan_is_not_candidate():
    score = np.full((50, 50), np.nan, dtype="float32")
    assert len(extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0)) == 0


def test_t_junction_three_segments():
    score = _empty()
    score[99:102, 20:181] = HIGH
    score[100:181, 99:102] = HIGH
    gdf = extract_candidates(score, T0, CRS, threshold=0.35, min_length_m=10.0)
    assert len(gdf) == 3
    junction = Point(*xy(T0, 100, 100))
    assert all(g.distance(junction) < 1.5 for g in gdf.geometry)
    ends = [Point(g.coords[0]) for g in gdf.geometry] + [Point(g.coords[-1]) for g in gdf.geometry]
    near = [p for p in ends if p.distance(junction) < 1.5]
    assert len(near) == 3 and max(p.distance(near[0]) for p in near) < 1e-6  # stesso nodo
    assert sorted(round(v / 10) for v in gdf.length_m) == [8, 8, 8]


def _line_score(tile_id, line):
    transform, width, height = tile_grid(tile_id, CRS, 1.0)
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e
    (x0, y0), (x1, y1) = line.coords
    d = np.array([x1 - x0, y1 - y0]) / line.length
    dist = np.abs((xs - x0) * d[1] - (ys - y0) * d[0])
    return np.where(dist <= 1.5, HIGH, LOW).astype("float32"), transform


def _seam_line():
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    cx, cy = to_utm.transform(15.68, 38.075)  # punto sul bordo comune tra le due tile
    a = np.radians(20)
    return LineString([(cx - 2000 * np.cos(a), cy - 2000 * np.sin(a)), (cx + 2000 * np.cos(a), cy + 2000 * np.sin(a))])


def test_seam_continuity():
    line = _seam_line()
    parts = []
    for tile in ("N3807E01567", "N3807E01568"):
        score, transform = _line_score(tile, line)
        gdf = extract_candidates(score, transform, CRS, threshold=0.35, min_length_m=10.0)
        parts.append(crop_to_core(gdf, tile, CRS))
    merged = merge_tiles(parts, snap_m=2.0)
    assert len(merged) == 1 and merged.geometry.iloc[0].geom_type == "LineString"
    core = unary_union([core_polygon(t, CRS) for t in ("N3807E01567", "N3807E01568")])
    true_len = line.intersection(core).length
    assert merged.length_m.iloc[0] == pytest.approx(true_len, rel=0.02)
    assert merged.geometry.iloc[0].length == pytest.approx(merged.length_m.iloc[0])


def test_crop_to_core_removes_halo():
    tile = "N3807E01567"
    line = _seam_line()
    score, transform = _line_score(tile, line)
    gdf = extract_candidates(score, transform, CRS, threshold=0.35, min_length_m=10.0)
    cropped = crop_to_core(gdf, tile, CRS)
    core = core_polygon(tile, CRS)
    assert cropped.length_m.sum() < gdf.length_m.sum()
    assert all(core.buffer(0.01).covers(g) for g in cropped.geometry)


def test_vectorize_and_merge_run(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    line = _seam_line()
    for tile in ("N3807E01567", "N3807E01568"):
        score, transform = _line_score(tile, line)
        d = tmp_path / "runs" / "r1" / "stage_a" / tile
        d.mkdir(parents=True)
        with rasterio.open(
            d / "score.tif",
            "w",
            driver="GTiff",
            width=score.shape[1],
            height=score.shape[0],
            count=1,
            dtype="float32",
            crs=CRS,
            transform=transform,
        ) as ds:
            ds.write(score, 1)
        out = vectorize_tile(tile, {"threshold": 0.35, "min_length_m": 10.0}, "r1")
        assert out == d / "candidates.gpkg" and out.exists()
    merged = merge_run("r1")
    assert merged == tmp_path / "runs" / "r1" / "stage_a" / "candidates_merged.gpkg"
    import geopandas as gpd

    gdf = gpd.read_file(merged)
    assert len(gdf) == 1 and gdf.crs.to_epsg() == 32633


def test_vectorize_tile_reconnects_only_when_asked(monkeypatch, tmp_path):
    from morloquio.candidates import vectorize

    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    tile = "N3807E01567"
    score, transform = _line_score(tile, _seam_line())
    d = tmp_path / "runs" / "r1" / "stage_b" / tile
    d.mkdir(parents=True)
    profile = dict(driver="GTiff", width=score.shape[1], height=score.shape[0], count=1, dtype="float32")
    with rasterio.open(d / "score.tif", "w", crs=CRS, transform=transform, **profile) as ds:
        ds.write(score, 1)
    calls = []
    monkeypatch.setattr(vectorize, "reconnect", lambda gdf, s, t, p: calls.append(p) or gdf)
    base = {"threshold": 0.35, "min_length_m": 10.0}
    vectorize.vectorize_tile(tile, base, "r1", stage="stage_b")
    assert calls == []
    vectorize.vectorize_tile(tile, base | {"reconnect": {"search_m": 50.0}}, "r1", stage="stage_b")
    assert calls == [{"search_m": 50.0}]


def test_merge_without_snap_keeps_seam_gap():
    parts = []
    for tile in ("N3807E01567", "N3807E01568"):
        score, transform = _line_score(tile, _seam_line())
        gdf = extract_candidates(score, transform, CRS, threshold=0.35, min_length_m=10.0)
        parts.append(crop_to_core(gdf, tile, CRS))
    merged = merge_tiles(parts, snap_m=0.0)
    assert len(merged) == 2  # gli estremi sul bordo non coincidono al millimetro: senza snap restano separati
    assert merged.mean_score.notna().all()
