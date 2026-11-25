"""Percorsi del progetto. Dati e run stanno fuori da git e si spostano con variabili d'ambiente."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Radice del repository (la cartella che contiene ``pyproject.toml``)."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").exists():
        raise RuntimeError(f"radice del repo non trovata partendo da {__file__}")
    return root


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Cartella dati: ``$MORLOQUIO_DATA_DIR`` oppure ``<repo>/data``."""
    env = os.environ.get("MORLOQUIO_DATA_DIR")
    return Path(env) if env else repo_root() / "data"


def runs_dir() -> Path:
    """Cartella degli output delle esecuzioni: ``$MORLOQUIO_RUNS_DIR`` oppure ``<repo>/runs``."""
    env = os.environ.get("MORLOQUIO_RUNS_DIR")
    return _ensure(Path(env) if env else repo_root() / "runs")


def raw_dir(source_id: str) -> Path:
    """Download intatti di una fonte."""
    return _ensure(data_dir() / "raw" / source_id)


def interim_dir(name: str) -> Path:
    """Estratti e ritagli intermedi."""
    return _ensure(data_dir() / "interim" / name)


def processed_dir(name: str) -> Path:
    """Dati elaborati (cubo, derivati)."""
    return _ensure(data_dir() / "processed" / name)
