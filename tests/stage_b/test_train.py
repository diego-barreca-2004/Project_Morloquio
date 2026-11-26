import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from shapely.geometry import LineString, box

from morloquio.common.paths import processed_dir
from morloquio.stage_b.dataset import PatchDataset
from morloquio.stage_b.features import CHANNELS, build_features, fit_normalization
from morloquio.stage_b.labels import build_labels
from morloquio.stage_b.predict import predict_array, predict_tile
from morloquio.stage_b.train import train_fold

from .conftest import TILES

LABELS = {"half_width_m": {"default": 1.5}, "ignore_band_m": 2.0, "unlabeled_weight": 0.1}
ORIGINS = (560000.0, 560160.0)


def _prepare(params):
    norm = fit_normalization(list(TILES), params)
    dirs = []
    for tile, x0 in zip(TILES, ORIGINS, strict=True):
        d = build_features(tile, params, norm)
        line = LineString([(x0 + 80.5, 4214830.0), (x0 + 80.5, 4215010.0)])  # lungo la trincea della fixture
        ways = gpd.GeoDataFrame({"highway": ["track"]}, geometry=[line], crs="EPSG:32633")
        build_labels(d, ways, LABELS, core=box(x0, 4214840.0, x0 + 160, 4215000.0))
        dirs.append(d)
    return dirs


@pytest.fixture
def tiny(two_cubes, params):
    p = params | {
        "model": {"base": 8, "depth": 2, "attention": True},
        "loss": {"focal_gamma": 2.0, "dice_weight": 1.0, "cldice_weight": 0.5, "cldice_iters": 3},
        "train": {
            "patch": 64,
            "batch": 4,
            "positive_fraction": 0.6,
            "patches_per_epoch": 16,
            "val_patches": 8,
            "epochs": 3,
            "lr": 0.003,
            "weight_decay": 0.0,
            "patience": 5,
            "channel_dropout": 0.2,
            "seed": 0,
            "device": "cpu",
            "workers": 0,
        },
        "predict": {"window": 96, "overlap": 32, "tta": True, "device": "cpu"},
    }
    return p, _prepare(p)


def test_labels_written_beside_features(tiny):
    _, dirs = tiny
    target = np.load(dirs[0] / "target.npy")
    weight = np.load(dirs[0] / "weight.npy")
    assert target.shape == weight.shape == (160, 160)
    assert target[:, 80].all() and weight[:20, :20].max() == 0  # buco nel DTM → peso 0
    assert json.loads((dirs[0] / "labels.json").read_text())["params"] == LABELS


def test_patch_sampler_hits_positives(tiny):
    _, dirs = tiny
    ds = PatchDataset(dirs, patch=64, length=12, positive_fraction=1.0, augment=True, channel_dropout=0.0, seed=1)
    for i in range(len(ds)):
        x, target, weight = ds[i]
        assert x.shape == (len(CHANNELS), 64, 64) and target.shape == weight.shape == (1, 64, 64)
        assert target.sum() > 0 and torch.isfinite(x).all()


def test_train_fold_tiny_runs_on_cpu(tiny, tmp_path):
    p, dirs = tiny
    model = train_fold(0, [dirs[0]], [dirs[1]], p, tmp_path / "fold_0")
    assert model.exists()
    hist = json.loads((tmp_path / "fold_0" / "history.json").read_text())
    assert len(hist["epochs"]) == 3 and all(np.isfinite(e["val_loss"]) for e in hist["epochs"])


def test_predict_matches_training_grid(tiny, tmp_path):
    p, dirs = tiny
    model = train_fold(0, [dirs[1]], [dirs[1]], p, tmp_path / "fold_0")
    out = predict_tile(dirs[0], model, p, tmp_path / "pred")
    cube = processed_dir("cube") / TILES[0] / "dtm.tif"
    with rasterio.open(out) as ds, rasterio.open(cube) as ref:
        score = ds.read(1)
        assert ds.transform == ref.transform and ds.crs == ref.crs and score.shape == (160, 160)
    assert np.isnan(score[:20, :20]).all()
    v = score[~np.isnan(score)]
    assert v.min() >= 0 and v.max() <= 1


class _FirstChannel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


def test_tta_is_equivariant():
    x = np.random.default_rng(0).normal(size=(3, 100, 70)).astype("float32")
    prob = predict_array(_FirstChannel(), x, window=48, overlap=16, tta=True, device="cpu")
    assert prob.shape == (100, 70)
    assert np.allclose(prob, 1 / (1 + np.exp(-x[0])), atol=1e-5)
