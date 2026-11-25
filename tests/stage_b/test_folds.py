import random

import pytest

from morloquio.stage_b.folds import assign_folds


def _grid(n_lat, n_lon, lat0=3801, lon0=1560):  # blocchi allineati alla griglia globale (3801 e 1560 multipli di 3)
    return [f"N{lat0 + i:04d}E{lon0 + j:05d}" for i in range(n_lat) for j in range(n_lon)]


def test_blocks_never_split_across_folds():
    tiles = _grid(6, 6)
    folds = assign_folds(tiles, k=4, block_tiles=3, km_by_tile={t: 1.0 for t in tiles})
    for i in range(0, 6, 3):
        for j in range(0, 6, 3):
            block = {folds[f"N{3801 + i + a:04d}E{1560 + j + b:05d}"] for a in range(3) for b in range(3)}
            assert len(block) == 1
    assert sorted(set(folds.values())) == [0, 1, 2, 3]


def test_folds_balanced_and_deterministic():
    tiles = _grid(9, 9)
    rng = random.Random(3)
    km = {t: rng.uniform(0, 10) for t in tiles}
    a = assign_folds(tiles, k=3, block_tiles=3, km_by_tile=km)
    shuffled = tiles[:]
    rng.shuffle(shuffled)
    assert assign_folds(shuffled, k=3, block_tiles=3, km_by_tile=km) == a
    sums = [sum(km[t] for t in tiles if a[t] == f) for f in range(3)]
    assert max(sums) / min(sums) < 1.3


def test_too_few_blocks_is_an_error():
    tiles = _grid(3, 3)
    with pytest.raises(ValueError, match="blocchi"):
        assign_folds(tiles, k=2, block_tiles=3, km_by_tile={t: 1.0 for t in tiles})
