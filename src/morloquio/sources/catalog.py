"""Catalogo delle fonti dati: un file YAML per fonte in ``configs/sources/``, filtrato dal license gate."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from morloquio.common.paths import repo_root
from morloquio.sources.licenses import LicenseRejected, check_data_license


@dataclass(frozen=True)
class SourceSpec:
    id: str
    name: str
    provider: str
    license: str
    license_url: str
    attribution: str
    resolution_m: float | None
    protocol: str
    adapter: str
    params: dict = field(hash=False)
    verified: str
    note: str = ""


_REQUIRED = [f.name for f in fields(SourceSpec) if f.name != "note"]


@dataclass
class Catalog:
    accepted: dict[str, SourceSpec]
    rejected: dict[str, str]


def default_sources_dir() -> Path:
    return repo_root() / "configs" / "sources"


def _parse(path: Path) -> SourceSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path.name}: campi mancanti {missing}")
    if raw["id"] != path.stem:
        raise ValueError(f"{path.name}: id {raw['id']!r} diverso dal nome del file")
    res = raw["resolution_m"]
    return SourceSpec(
        id=raw["id"],
        name=raw["name"],
        provider=raw["provider"],
        license=str(raw["license"]),
        license_url=raw["license_url"],
        attribution=raw["attribution"],
        resolution_m=None if res is None else float(res),
        protocol=raw["protocol"],
        adapter=raw["adapter"],
        params=dict(raw["params"] or {}),
        verified=str(raw["verified"]),
        note=raw.get("note", "") or "",
    )


def load_catalog(path: Path | None = None) -> Catalog:
    """Carica tutte le fonti; quelle con licenza non ammessa finiscono in ``rejected`` con il motivo."""
    directory = Path(path) if path else default_sources_dir()
    accepted: dict[str, SourceSpec] = {}
    rejected: dict[str, str] = {}
    for file in sorted(directory.glob("*.yaml")):
        spec = _parse(file)
        try:
            check_data_license(spec.license)
        except LicenseRejected as exc:
            rejected[spec.id] = str(exc)
            continue
        accepted[spec.id] = spec
    return Catalog(accepted=accepted, rejected=rejected)
