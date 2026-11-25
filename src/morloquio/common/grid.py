"""Griglia mondiale di tile da 0,01° allineata alla griglia LiDAR PST 1x1.

Identificatore di tile: ``N3807E01567`` = cella 38.07–38.08 N, 15.67–15.68 E.
L'indice di cella è il floor di coordinata×100; l'arrotondamento a 6 decimali evita che 38.07*100 = 3806.9999…
finisca nella cella sbagliata.
"""

from __future__ import annotations

import math
import re

TILE_DEG = 0.01
_SCALE = 100  # celle per grado
_TILE_RE = re.compile(r"^([NS])(\d{4})([EW])(\d{5})$")
_TAVOLA_RE = re.compile(r"^D(\d{4})(\d{4})_0101$")


def _index(coord: float) -> int:
    return math.floor(round(coord * _SCALE, 6))


def _format(i: int, j: int) -> str:
    ns = "S" if i < 0 else "N"
    ew = "W" if j < 0 else "E"
    return f"{ns}{abs(i):04d}{ew}{abs(j):05d}"


def _parse(tile_id: str) -> tuple[int, int]:
    m = _TILE_RE.match(tile_id)
    if not m:
        raise ValueError(f"tile_id non valido: {tile_id!r}")
    ns, i, ew, j = m.groups()
    # Per l'emisfero sud/ovest il numero è il valore assoluto dell'indice negativo
    return (-int(i) if ns == "S" else int(i)), (-int(j) if ew == "W" else int(j))


def tile_id_for(lat: float, lon: float) -> str:
    """Tile che contiene il punto (lat, lon) in EPSG:4326."""
    return _format(_index(lat), _index(lon))


def tile_bounds(tile_id: str) -> tuple[float, float, float, float]:
    """Limiti (west, south, east, north) della tile in EPSG:4326."""
    i, j = _parse(tile_id)
    return (j / _SCALE, i / _SCALE, (j + 1) / _SCALE, (i + 1) / _SCALE)


def tiles_in_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """Tile che intersecano l'interno del bbox (west, south, east, north); bordi est/nord esclusi."""
    west, south, east, north = bbox
    i0, i1 = _index(south), math.ceil(round(north * _SCALE, 6)) - 1
    j0, j1 = _index(west), math.ceil(round(east * _SCALE, 6)) - 1
    return sorted(_format(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))


def neighbors(tile_id: str) -> list[str]:
    """Le 8 tile adiacenti."""
    i, j = _parse(tile_id)
    return [_format(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def pst_tavola_for(tile_id: str) -> str:
    """Nome della tavola LiDAR PST 1x1 corrispondente (es. ``D38071567_0101``). Solo emisfero N/E."""
    i, j = _parse(tile_id)
    if i < 0 or j < 0:
        raise ValueError(f"nessuna tavola PST per {tile_id}")
    return f"D{i:04d}{j:04d}_0101"


def tile_id_from_tavola(tavola: str) -> str:
    """Inverso di :func:`pst_tavola_for`."""
    m = _TAVOLA_RE.match(tavola)
    if not m:
        raise ValueError(f"tavola PST non valida: {tavola!r}")
    return _format(int(m.group(1)), int(m.group(2)))
