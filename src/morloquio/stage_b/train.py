"""Addestramento di un fold: AdamW + OneCycle, precisione mista bf16 su GPU, early stopping sulla loss di validazione."""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from morloquio.common.runlog import get_logger
from morloquio.stage_b.dataset import PatchDataset
from morloquio.stage_b.losses import total_loss
from morloquio.stage_b.model import UNet

log = get_logger("stage_b.train")


def resolve_device(name: str | None) -> str:
    if name in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("GPU non disponibile: lo Stadio B gira su CPU (molto più lento)", stacklevel=2)
        return "cpu"
    return name


def _evaluate(model, loader, loss_params, device, use_amp) -> dict[str, float]:
    model.eval()
    total, n, tp, fn, fp = 0.0, 0, 0.0, 0.0, 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for x, t, w in loader:
            x, t, w = x.to(device), t.to(device), w.to(device)
            logits = model(x)
            loss, _ = total_loss(logits, t, w, loss_params)
            total += float(loss) * len(x)
            n += len(x)
            pred = torch.sigmoid(logits.float()) > 0.5
            known = w > 0
            tp += float((pred & (t > 0) & known).sum())
            fn += float((~pred & (t > 0) & known).sum())
            fp += float((pred & (t == 0) & known).sum())
    return {
        "val_loss": total / max(n, 1),
        "val_recall_px": tp / max(tp + fn, 1.0),
        "val_precision_px": tp / max(tp + fp, 1.0),  # limite inferiore: i "falsi positivi" possono essere vie non in OSM
    }


def train_fold(fold: int, train_dirs: list[Path], val_dirs: list[Path], params: dict, out_dir: Path) -> Path:
    """Addestra il modello del fold e scrive ``model.pt`` (miglior epoca) e ``history.json`` in ``out_dir``."""
    tp = params["train"]
    torch.manual_seed(int(tp["seed"]) + fold)
    device = resolve_device(tp.get("device", "auto"))
    use_amp = device.startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = json.loads((Path(train_dirs[0]) / "meta.json").read_text())["channels"]
    model = UNet(len(channels), **params["model"]).to(device)
    common = dict(patch=int(tp["patch"]), seed=int(tp["seed"]) + 1000 * fold)
    train_ds = PatchDataset(
        train_dirs,
        length=int(tp["patches_per_epoch"]),
        positive_fraction=float(tp["positive_fraction"]),
        augment=True,
        channel_dropout=float(tp["channel_dropout"]),
        **common,
    )
    val_ds = PatchDataset(
        val_dirs,
        length=int(tp.get("val_patches", 256)),
        positive_fraction=float(tp["positive_fraction"]),
        augment=False,
        channel_dropout=0.0,
        **(common | {"seed": common["seed"] + 999}),
    )
    workers = int(tp.get("workers", min(8, os.cpu_count() or 1)))
    batch = int(tp["batch"])
    loader_kw = dict(batch_size=batch, num_workers=workers, pin_memory=use_amp, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    epochs = int(tp["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(tp["lr"]), weight_decay=float(tp["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(tp["lr"]), total_steps=epochs * math.ceil(len(train_ds) / batch)
    )
    best, bad, history = math.inf, 0, []
    model_path = out_dir / "model.pt"
    for epoch in range(epochs):
        t0 = time.time()
        train_ds.resample(epoch)
        model.train()
        losses = []
        for x, t, w in train_loader:
            x, t, w = x.to(device, non_blocking=True), t.to(device, non_blocking=True), w.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss, _ = total_loss(logits, t, w, params["loss"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **_evaluate(model, val_loader, params["loss"], device, use_amp)}
        row["seconds"] = round(time.time() - t0, 1)
        history.append(row)
        log.info("epoca", extra={"fold": fold, **row})
        if row["val_loss"] < best:
            best, bad = row["val_loss"], 0
            state = {
                "state_dict": model.state_dict(),
                "in_ch": len(channels),
                "model": dict(params["model"]),
                "channels": list(channels),
                "epoch": epoch,
            }
            torch.save(state, model_path)
        else:
            bad += 1
            if bad >= int(tp["patience"]):
                break
    summary = {"fold": fold, "device": device, "best_val_loss": best, "epochs": history}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model_path
