"""Aree di interesse (AOI) definite in ``configs/aoi/<nome>.yaml``."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from morloquio.common.paths import repo_root

BBox = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)


@dataclass(frozen=True)
class AOI:
    name: str
    bbox: BBox
    core_bbox: BBox
    work_crs: str


def load_aoi(name: str) -> AOI:
    path = repo_root() / "configs" / "aoi" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"AOI {name!r} non trovata: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AOI(
        name=raw["name"],
        bbox=tuple(float(v) for v in raw["bbox"]),
        core_bbox=tuple(float(v) for v in raw["core_bbox"]),
        work_crs=raw["work_crs"],
    )
