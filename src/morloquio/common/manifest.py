"""Manifest dei download: ``<cartella>/manifest.json`` con sha256, dimensione, URL e data di ogni file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from morloquio.common.http import FileRecord

MANIFEST_NAME = "manifest.json"


def read_manifest(directory: Path) -> dict[str, dict]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(directory: Path, rel: str, record: FileRecord, **extra) -> None:
    """Aggiunge o sostituisce la voce ``rel``; scrittura atomica del file manifest."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    manifest[rel] = {
        "sha256": record.sha256,
        "size": record.size,
        "url": record.url,
        "fetched_at": record.fetched_at,
        **extra,
    }
    tmp = directory / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / MANIFEST_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(directory: Path, rel: str) -> bool:
    """Vero solo se il file esiste e il suo sha256 coincide con quello registrato nel manifest."""
    entry = read_manifest(directory).get(rel)
    path = Path(directory) / rel
    if entry is None or not path.exists():
        return False
    return path.stat().st_size == entry["size"] and _sha256(path) == entry["sha256"]
