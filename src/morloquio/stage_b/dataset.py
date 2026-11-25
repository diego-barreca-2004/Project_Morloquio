"""Patch casuali per l'addestramento, estratte dai canali e dalle etichette scritte per tile.

Una quota ``positive_fraction`` di patch contiene un pixel OSM (spostato a caso fino a ¼ di patch dal centro), le
altre cadono su un pixel qualsiasi con peso > 0. Le coordinate e gli aumenti (rotazioni di 90°, ribaltamento,
spegnimento di canali) si estraggono una volta per epoca nel processo principale: con più worker il risultato è
identico e riproducibile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class _Tile:
    def __init__(self, d: Path):
        self.x = np.load(d / "features.npy", mmap_mode="r")
        self.target = np.load(d / "target.npy")
        self.weight = np.load(d / "weight.npy")
        self.positive = np.flatnonzero((self.target > 0) & (self.weight > 0))
        self.any = np.flatnonzero(self.weight > 0)
        self.channels = json.loads((d / "meta.json").read_text())["channels"]


def _crop(arr: np.ndarray, r0: int, c0: int, size: int) -> np.ndarray:
    out = arr[..., r0 : r0 + size, c0 : c0 + size]
    pad_r, pad_c = size - out.shape[-2], size - out.shape[-1]
    if pad_r or pad_c:  # tile più piccola della patch: zeri (peso 0)
        out = np.pad(out, [(0, 0)] * (out.ndim - 2) + [(0, pad_r), (0, pad_c)])
    return np.asarray(out, dtype="float32")


class PatchDataset(Dataset):
    def __init__(
        self,
        tile_dirs: list[Path],
        patch: int,
        length: int,
        positive_fraction: float,
        augment: bool,
        channel_dropout: float,
        seed: int,
        max_dropped: int = 3,
    ):
        self.tiles = [t for t in (_Tile(Path(d)) for d in tile_dirs) if t.any.size]
        if not self.tiles:
            raise ValueError("nessuna tile con pixel etichettati: impossibile estrarre patch")
        self.patch, self.length = patch, length
        self.positive_fraction, self.augment = positive_fraction, augment
        self.channel_dropout, self.max_dropped, self.seed = channel_dropout, max_dropped, seed
        self.droppable = [i for i, c in enumerate(self.tiles[0].channels) if c != "valid"]
        self.resample(0)

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng([self.seed, epoch])
        p, items = self.patch, []
        for _ in range(self.length):
            ti = int(rng.integers(len(self.tiles)))
            tile = self.tiles[ti]
            h, w = tile.weight.shape
            pool = tile.positive if tile.positive.size and rng.random() < self.positive_fraction else tile.any
            r, c = divmod(int(pool[rng.integers(pool.size)]), w)
            jr, jc = rng.integers(-(p // 4), p // 4 + 1, size=2)
            r0 = int(np.clip(r - p // 2 + jr, 0, max(h - p, 0)))
            c0 = int(np.clip(c - p // 2 + jc, 0, max(w - p, 0)))
            k = int(rng.integers(4)) if self.augment else 0
            flip = bool(rng.random() < 0.5) if self.augment else False
            drop = [i for i in self.droppable if rng.random() < self.channel_dropout][: self.max_dropped]
            items.append((ti, r0, c0, k, flip, drop))
        self.items = items

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        ti, r0, c0, k, flip, drop = self.items[i]
        tile = self.tiles[ti]
        x = _crop(tile.x, r0, c0, self.patch)
        t = _crop(tile.target[None], r0, c0, self.patch)
        w = _crop(tile.weight[None], r0, c0, self.patch)
        if drop:
            x[drop] = 0.0
        arrays = []
        for a in (x, t, w):
            a = np.rot90(a, k, axes=(1, 2))
            arrays.append(np.ascontiguousarray(a[..., ::-1] if flip else a))
        return tuple(torch.from_numpy(a) for a in arrays)
