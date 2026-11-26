"""Fold spaziali per la validazione incrociata a blocchi.

Tile vicine si somigliano (autocorrelazione spaziale): con tile casuali la metrica di test risulta gonfiata
(fino al 28% in Kattenborn et al. 2022). Le tile si raggruppano in blocchi di ``block_tiles`` × ``block_tiles``
celle allineati alla griglia globale (così restano stabili quando l'area cresce) e ogni blocco finisce intero in un
solo fold. I blocchi si distribuiscono ai fold bilanciando i km di vie OSM (prima i blocchi più carichi).
"""

from __future__ import annotations

import re
from collections import defaultdict

_TILE = re.compile(r"^N(\d{4})E(\d{5})$")


def block_of(tile_id: str, block_tiles: int) -> tuple[int, int]:
    m = _TILE.match(tile_id)
    if not m:
        raise ValueError(f"tile_id non valido: {tile_id!r}")
    return int(m.group(1)) // block_tiles, int(m.group(2)) // block_tiles


def assign_folds(tile_ids: list[str], k: int, block_tiles: int, km_by_tile: dict[str, float]) -> dict[str, int]:
    """Fold (0 … k−1) di ogni tile; deterministico, indipendente dall'ordine di ``tile_ids``."""
    blocks: dict[tuple[int, int], list[str]] = defaultdict(list)
    for tile_id in tile_ids:
        blocks[block_of(tile_id, block_tiles)].append(tile_id)
    if len(blocks) < k:
        raise ValueError(f"servono almeno {k} blocchi per {k} fold, ce ne sono {len(blocks)}")
    km = {key: sum(float(km_by_tile.get(t, 0.0)) for t in tiles) for key, tiles in blocks.items()}
    load, count = [0.0] * k, [0] * k
    folds: dict[str, int] = {}
    for key in sorted(blocks, key=lambda b: (-km[b], b)):
        f = min(range(k), key=lambda i: (load[i], count[i], i))
        load[f] += km[key]
        count[f] += 1
        for tile_id in blocks[key]:
            folds[tile_id] = f
    return folds
