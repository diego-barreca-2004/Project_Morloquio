"""Predizione su una tile intera: finestre sovrapposte fuse con pesi gaussiani e TTA sulle 8 simmetrie del quadrato."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

from morloquio.common.runlog import get_logger
from morloquio.stage_b.model import UNet

DIHEDRAL = [(k, f) for f in (False, True) for k in range(4)]

log = get_logger("stage_b.predict")


def _gauss(n: int) -> np.ndarray:
    g = np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / (n / 6)) ** 2)
    return np.maximum(np.outer(g, g), 1e-3).astype("float32")


def _starts(size: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def _forward(model: torch.nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
    if not tta:
        return torch.sigmoid(model(x).float())[0, 0]
    batch = []
    for k, f in DIHEDRAL:
        t = torch.rot90(x[0], k, dims=(1, 2))
        batch.append(torch.flip(t, dims=(2,)) if f else t)
    prob = torch.sigmoid(model(torch.stack(batch)).float())[:, 0]
    out = []
    for p, (k, f) in zip(prob, DIHEDRAL, strict=True):
        p = torch.flip(p, dims=(1,)) if f else p
        out.append(torch.rot90(p, -k, dims=(0, 1)))
    return torch.stack(out).mean(0)


def predict_array(
    model: torch.nn.Module, x: np.ndarray, window: int, overlap: int, tta: bool, device: str
) -> np.ndarray:
    """Probabilità 0–1 per pixel di un array C×H×W."""
    c, h, w = x.shape
    hp, wp = max(h, window), max(w, window)
    xp = np.zeros((c, hp, wp), "float32")
    xp[:, :h, :w] = x
    acc = np.zeros((hp, wp), "float32")
    wsum = np.zeros((hp, wp), "float32")
    g = _gauss(window)
    stride = window - overlap
    use_amp = device.startswith("cuda")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for r in _starts(hp, window, stride):
            for col in _starts(wp, window, stride):
                tile = torch.from_numpy(xp[None, :, r : r + window, col : col + window]).to(device)
                prob = _forward(model, tile, tta).cpu().numpy()
                acc[r : r + window, col : col + window] += prob * g
                wsum[r : r + window, col : col + window] += g
    return (acc / wsum)[:h, :w]


def load_model(model_path: Path, device: str) -> UNet:
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = UNet(int(ckpt["in_ch"]), **ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def predict_tile(tile_dir: Path, model_path: Path, params: dict, out_dir: Path) -> Path:
    """Scrive ``out_dir/score.tif`` (float32 0–1, NaN fuori DTM) sulla griglia del cubo della tile."""
    from morloquio.stage_b.train import resolve_device

    pp = params["predict"]
    device = resolve_device(pp.get("device", "auto"))
    meta = json.loads((Path(tile_dir) / "meta.json").read_text())
    x = np.load(Path(tile_dir) / "features.npy").astype("float32")
    prob = predict_array(load_model(model_path, device), x, int(pp["window"]), int(pp["overlap"]), bool(pp["tta"]), device)
    prob[x[meta["channels"].index("valid")] < 0.5] = np.nan
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=prob.shape[1],
        height=prob.shape[0],
        count=1,
        dtype="float32",
        crs=meta["crs"],
        transform=Affine(*meta["transform"]),
        nodata=float("nan"),
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(out_dir / "score.tif", "w", **profile) as ds:
        ds.write(prob.astype("float32"), 1)
    log.info("Stadio B: probabilità", extra={"tile_id": meta["tile_id"], "model": str(model_path)})
    return out_dir / "score.tif"
