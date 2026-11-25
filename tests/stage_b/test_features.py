import json

import numpy as np

from morloquio.stage_b.features import CHANNELS, build_features, fit_normalization, raw_channels

from .conftest import TILES


def test_channels_shape_and_no_nan(two_cubes, params):
    norm = fit_normalization(list(TILES), params)
    out = build_features(TILES[0], params, norm)
    x = np.load(out / "features.npy")
    assert x.dtype == np.float16 and x.shape == (len(CHANNELS), 160, 160)
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    valid = x[CHANNELS.index("valid")]
    assert valid[:20, :20].max() == 0 and valid[40:, 40:].min() == 1
    meta = json.loads((out / "meta.json").read_text())
    assert meta["channels"] == list(CHANNELS) and meta["crs"] == "EPSG:32633" and len(meta["transform"]) == 6


def test_normalization_is_global_not_per_tile(two_cubes, params):
    norm = fit_normalization(list(TILES), params)
    a = np.load(build_features(TILES[0], params, norm) / "features.npy").astype("float32")
    b = np.load(build_features(TILES[1], params, norm) / "features.npy").astype("float32")
    s = CHANNELS.index("slope")
    # stessa pendenza nelle due tile → stesso valore normalizzato (con normalizzazione per tile sarebbe diverso)
    assert abs(np.median(a[s, 40:, 100:]) - np.median(b[s, 40:, 100:])) < 0.02
    assert set(norm) == set(CHANNELS) - {"valid"}


def test_trench_stands_out_in_relief_channels(two_cubes, params):
    raw = raw_channels(TILES[1], params)
    assert raw["lrm"][:, 80].mean() < raw["lrm"][:, 120].mean() - 0.2
    assert raw["lrm_lin"][:, 80].mean() > 3 * raw["lrm_lin"][:, 120].mean()


def test_missing_gps_gives_zero_channel(two_cubes, params):
    raw = raw_channels(TILES[0], params, gps_xy=None)
    assert np.nanmax(raw["gps"]) == 0
    pts = np.array([[560100.5, 4214900.5]] * 20)
    assert np.nanmax(raw_channels(TILES[0], params, gps_xy=pts)["gps"]) > 0
