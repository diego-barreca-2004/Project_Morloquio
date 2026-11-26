import json

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from morloquio.candidates.linearity import (
    bench_score,
    chm_gap_linearity,
    fuse,
    gps_density,
    hessian_linearity,
    ortho_linearity,
    robust_normalize,
    score_tile,
)
from morloquio.common.paths import repo_root
from morloquio.terrain.derivatives import compute_tile

RES = 1.0
N = 200


def _slope_with_path(deg=30.0, xc=101.0, half=1.0, cut_deg=50.0, fill_deg=40.0):
    """Versante che sale verso est con sentiero a mezza costa lungo le isoipse: piano largo 2 m (xc ± 1 m,
    colonne 100–101), scarpata di taglio a monte (50°) e di riporto a valle (40°)."""
    x = (np.arange(N) + 0.5) * RES  # centri dei pixel
    t = np.tan(np.radians(deg))
    natural = t * x
    h0 = t * xc
    z = natural.copy()
    z[np.abs(x - xc) <= half] = h0
    up = x > xc + half
    z[up] = np.minimum(natural[up], h0 + np.tan(np.radians(cut_deg)) * (x[up] - xc - half))
    down = x < xc - half
    z[down] = np.maximum(natural[down], h0 - np.tan(np.radians(fill_deg)) * (xc - half - x[down]))
    return np.tile(z, (N, 1)).astype("float32")


def test_bench_detects_cut_path():
    dtm = _slope_with_path()
    score = bench_score(dtm, RES)
    on_path = np.nanmean(score[20:-20, 100:102])
    on_slope = np.nanmean(score[20:-20, 30:70])
    assert on_path > 5 * on_slope


def test_bench_ignores_plain_slope_and_flat():
    yy, xx = np.mgrid[0:N, 0:N].astype("float32")
    for dtm in (np.tan(np.radians(30)) * xx, np.zeros((N, N), dtype="float32")):
        raw = bench_score(dtm.astype("float32"), RES, normalize=False)
        assert np.nanmax(raw[10:-10, 10:-10]) < 1e-3


def test_hessian_finds_trench():
    img = np.zeros((N, N), dtype="float32")
    img[:, 99:101] = -0.5
    lin = hessian_linearity(img, RES, polarity="valley")
    row = lin[100, 80:120]
    assert 80 + int(np.argmax(row)) in (99, 100)
    assert lin[100, 100] > 0.8 and lin[100, 40] < 0.3


def test_chm_gap_corridor():
    chm = np.full((N, N), 15.0, dtype="float32")
    chm[:, 98:102] = 0.0  # corridoio largo 4 m nel bosco
    chm[:, 150:] = 0.0  # prato a est
    gap = chm_gap_linearity(chm, RES)
    assert np.nanmean(gap[20:-20, 98:102]) > 0.7
    assert np.nanmean(gap[20:-20, 40:70]) < 0.2  # bosco pieno
    assert np.nanmean(gap[20:-20, 170:190]) < 0.2  # prato


def test_ortho_linearity_finds_bright_track():
    rgb = np.full((3, N, N), 60, dtype="uint8")
    rgb[:, :, 99:103] = 200  # sterrata chiara larga 2 m a 0,5 m/px
    lin = ortho_linearity(rgb, 0.5)
    assert lin[100, 101] > 0.8 and lin[100, 40] < 0.3


def test_gps_density_peaks_at_points():
    transform = from_origin(0, N, RES, RES)
    pts = np.array([[50.5, N - 100.5]] * 10)
    dens = gps_density(pts, transform, (N, N), sigma_m=5.0)
    assert np.unravel_index(np.argmax(dens), dens.shape) == (100, 50)
    assert dens.max() == pytest.approx(1.0) and dens[10, 10] < 0.01


def test_gps_density_empty():
    dens = gps_density(np.empty((0, 2)), from_origin(0, N, 1, 1), (N, N))
    assert np.all(dens == 0)


def test_robust_normalize_range_and_nan():
    x = np.random.default_rng(0).normal(size=(50, 50)).astype("float32")
    x[0, 0] = np.nan
    x[10, 10] = 50
    y = robust_normalize(x)
    assert np.isnan(y[0, 0]) and y[10, 10] > 0.99 and np.nanmax(y) <= 1 and np.nanmin(y) >= 0


def test_fuse_renormalizes_missing_layers():
    a = np.full((4, 4), 0.8, dtype="float32")
    b = np.full((4, 4), 0.2, dtype="float32")
    missing = np.full((4, 4), np.nan, dtype="float32")
    w = {"a": 1.0, "b": 1.0, "c": 2.0}
    with_missing = fuse({"a": a, "b": b, "c": missing}, w)
    without = fuse({"a": a, "b": b}, w)
    np.testing.assert_allclose(with_missing, without)
    np.testing.assert_allclose(without, 0.5)


def test_fuse_all_nan_stays_nan():
    nan = np.full((2, 2), np.nan, dtype="float32")
    assert np.isnan(fuse({"a": nan}, {"a": 1.0})).all()


def test_score_tile_writes_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    tile = "N3807E01567"
    cube = tmp_path / "data" / "processed" / "cube" / tile
    cube.mkdir(parents=True)
    dtm = _slope_with_path()
    dtm[0:10, 0:10] = np.nan
    profile = dict(
        driver="GTiff",
        width=N,
        height=N,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(560000, 4215000, 1, 1),
        nodata=float("nan"),
    )
    for name, arr in {"dtm": dtm, "dsm_first": dtm + 12}.items():
        with rasterio.open(cube / f"{name}.tif", "w", **profile) as ds:
            ds.write(arr, 1)
    (cube / "meta.json").write_text(json.dumps({"tile_id": tile, "tier": "T1"}))
    terrain = yaml.safe_load((repo_root() / "configs/pipeline/terrain.yaml").read_text())
    params = yaml.safe_load((repo_root() / "configs/pipeline/stage_a.yaml").read_text()) | {"terrain": terrain}
    compute_tile(tile, terrain)
    gps = np.array([[560000 + 100.5, 4215000 - 100.5]] * 5)
    out = score_tile(tile, params, "run1", gps_xy=gps)
    assert out == tmp_path / "runs" / "run1" / "stage_a" / tile
    assert {p.stem for p in out.glob("*.tif")} == {"score", "bench", "lrm_lin", "chm_gap", "gps"}
    with rasterio.open(out / "score.tif") as ds:
        score = ds.read(1)
        assert ds.crs.to_epsg() == 32633 and score.shape == (N, N)
    assert np.isnan(score[5, 5]) and np.nanmax(score) <= 1 and np.nanmin(score) >= 0
    assert np.nanmean(score[20:-20, 100:102]) > np.nanmean(score[20:-20, 30:70])
