"""Adapter LiDAR PST (MASE): griglia delle tavole via ArcGIS REST, download via l'API pubblica dell'app.

Flusso per batch (massimo ``batch_max`` tavole): POST della sessione → attesa ``status=ready`` →
download degli zip → estrazione dei soli GeoTIFF, rinominati dalla tavola → POST di ack.
Gli zip sono dati non fidati: il nome di destinazione viene dalla tavola, mai dal percorso del membro.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx

from morloquio.common.grid import tile_id_from_tavola, tiles_in_bbox
from morloquio.common.http import FileRecord, PoliteClient
from morloquio.common.manifest import is_complete, update_manifest
from morloquio.common.paths import raw_dir
from morloquio.common.runlog import get_logger
from morloquio.sources.catalog import SourceSpec

SOURCE_ID = "pst_lidar"
OK_REASONS = {"OK", "METADATA_WARNING"}
_TAVOLA_IN_NAME = re.compile(r"(?:_(\d+)_)?(D\d{8}_0101)")
_PAGE = 2000

log = get_logger("sources.pst_lidar")


class PstError(RuntimeError):
    """Errore della API PST per una sessione o un batch."""


@dataclass(frozen=True)
class PstTile:
    fid: int
    tavola: str
    regione: str
    contratto: str
    date: str
    tile_id: str

    @property
    def filename(self) -> str:
        """Una tavola può essere coperta da più contratti (rilievi diversi): un file per contratto."""
        return f"{self.tavola}_c{self.contratto}.tif"


def list_tiles(client: PoliteClient, bbox, grid_url: str) -> list[PstTile]:
    """Tavole LiDAR 1x1 che cadono nelle celle del bbox (west, south, east, north)."""
    wanted = set(tiles_in_bbox(bbox))
    west, south, east, north = bbox
    tiles: dict[tuple[str, str], PstTile] = {}
    offset = 0
    while True:
        params = {
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "fid,tavola,regione,contratto,data",
            "returnGeometry": "false",
            "orderByFields": "fid",
            "resultOffset": str(offset),
            "resultRecordCount": str(_PAGE),
            "f": "json",
        }
        data = client.request("GET", grid_url, params=params).json()
        if "error" in data:
            raise PstError(f"errore della griglia PST: {data['error']}")
        features = data.get("features", [])
        for feat in features:
            a = feat["attributes"]
            tile = PstTile(
                fid=int(a["fid"]),
                tavola=a["tavola"],
                regione=a.get("regione") or "",
                contratto=str(a.get("contratto") or ""),
                date=str(a.get("data") or ""),
                tile_id=tile_id_from_tavola(a["tavola"]),
            )
            if not tile.contratto.strip():
                # cella segnaposto della griglia (contratto e data vuoti): lo zip contiene un TIFF da 0 byte
                log.info("tavola PST senza rilievo, saltata", extra={"tavola": tile.tavola, "fid": tile.fid})
                continue
            if tile.tile_id in wanted:
                tiles.setdefault((tile.tavola, tile.contratto), tile)
        if not data.get("exceededTransferLimit"):
            break
        offset += len(features)
    return sorted(tiles.values(), key=lambda t: (t.tile_id, t.contratto))


def request_session(
    client: PoliteClient,
    api_url: str,
    product: str,
    tiles: Sequence[PstTile],
    *,
    canale: str = "PST2",
    prodotto: str = "LIDAR_2014",
) -> str:
    payload = {
        "canale": canale,
        "prodotto": prodotto,
        "layer": product,
        "tileList": [
            {"fid": str(t.fid), "tavola": t.tavola, "regione": t.regione, "contratto": t.contratto} for t in tiles
        ],
        "metadataList": [],
    }
    data = client.request("POST", api_url, json=payload).json()
    reason = data.get("reason") or (data.get("upstream") or {}).get("reason")
    session_id = data.get("idSessione")
    if reason != "OK" or not session_id:
        raise PstError(f"sessione PST rifiutata (reason={reason!r})")
    return session_id


def public_file_url(api_url: str, raw_url: str) -> str:
    """Il backend restituisce URL del gateway interno: si riportano sull'origine pubblica dell'API."""
    if raw_url.startswith(api_url):
        return raw_url
    path = urlsplit(raw_url).path
    marker = "/api/v1/downloads"
    idx = path.find(marker)
    if idx < 0:
        raise PstError(f"URL di file PST inatteso: {raw_url}")
    return api_url + path[idx + len(marker) :]


def wait_ready(
    client: PoliteClient,
    api_url: str,
    session_id: str,
    timeout_s: float = 900,
    poll_s: float = 5,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    deadline = clock() + timeout_s
    while True:
        status = client.request("GET", f"{api_url}/{session_id}").json()
        state, reason = status.get("status"), status.get("reason")
        if state == "ready":
            if reason not in OK_REASONS:
                raise PstError(f"sessione {session_id} pronta con reason={reason!r}")
            return [public_file_url(api_url, u) for u in status.get("urls") or []]
        if state in {"error", "failed", "expired"}:
            raise PstError(f"sessione {session_id} in stato {state!r} (reason={reason!r})")
        if clock() > deadline:
            raise PstError(f"sessione {session_id} non pronta entro {timeout_s} s")
        sleep(poll_s)


def _match_member(name: str, wanted: dict[tuple[str, str], PstTile]) -> PstTile | None:
    """Tile del batch a cui appartiene il membro dello zip (es. ``calabria_140_D38071567_0101_DTM.tiff``)."""
    m = _TAVOLA_IN_NAME.search(name)
    if not m:
        return None
    contratto, tavola = m.group(1), m.group(2)
    if contratto and (tavola, contratto) in wanted:
        return wanted[(tavola, contratto)]
    same = [t for (tav, _), t in wanted.items() if tav == tavola]
    return same[0] if len(same) == 1 else None


def _extract_tifs(
    zip_path: Path, zip_url: str, root: Path, wanted: dict[tuple[str, str], PstTile], license: str
) -> list[str]:
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = PurePosixPath(member.filename).name
            if member.is_dir() or not name.lower().endswith((".tif", ".tiff")):
                continue
            tile = _match_member(name, wanted)
            if member.file_size == 0:
                log.warning("membro zip vuoto ignorato", extra={"member": member.filename, "zip": zip_url})
                continue
            if tile is None or ".." in PurePosixPath(member.filename).parts:
                log.warning("membro zip ignorato", extra={"member": member.filename, "zip": zip_url})
                continue
            rel = tile.filename
            part = root / (rel + ".part")
            digest = hashlib.sha256()
            size = 0
            with zf.open(member) as src, open(part, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            os.replace(part, root / rel)
            record = FileRecord(
                path=root / rel,
                sha256=digest.hexdigest(),
                size=size,
                url=f"{zip_url}#{member.filename}",
                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            update_manifest(
                root,
                rel,
                record,
                license=license,
                tile_id=tile.tile_id,
                fid=tile.fid,
                contratto=tile.contratto,
                survey_date=tile.date,
            )
            written.append(rel)
    return written


def _batches(tiles: Sequence[PstTile], batch_max: int) -> list[list[PstTile]]:
    """Batch di al massimo ``batch_max`` tavole, mai due volte la stessa tavola nello stesso batch."""
    batches: list[list[PstTile]] = []
    for tile in tiles:
        for batch in batches:
            if len(batch) < batch_max and all(t.tavola != tile.tavola for t in batch):
                batch.append(tile)
                break
        else:
            batches.append([tile])
    return batches


def fetch_product(
    client: PoliteClient,
    spec: SourceSpec,
    product: str,
    tiles: Sequence[PstTile],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Scarica il prodotto (``dtm_1x1``, ``dsm_first_1x1``, ``dsm_last_1x1``) per le tavole indicate."""
    p = spec.params
    if product not in p["products"]:
        raise ValueError(f"prodotto PST sconosciuto: {product}")
    root = raw_dir(SOURCE_ID) / product
    root.mkdir(parents=True, exist_ok=True)
    zips = root / ".zips"
    todo = [t for t in tiles if not is_complete(root, t.filename)]
    for batch in _batches(todo, int(p["batch_max"])):
        wanted = {(t.tavola, t.contratto): t for t in batch}
        try:
            session_id = request_session(
                client, p["api_url"], product, batch, canale=p["canale"], prodotto=p["prodotto"]
            )
            urls = wait_ready(client, p["api_url"], session_id, sleep=sleep)
        except PstError as exc:
            log.error("batch PST fallito", extra={"product": product, "tavole": list(wanted), "error": str(exc)})
            continue
        for i, url in enumerate(urls):
            zip_path = zips / f"{session_id}_{i}.zip"
            try:
                client.download(url, zip_path)
            except httpx.HTTPError as exc:
                # un batch non scaricabile non ferma gli altri: le sue tavole restano da fare al prossimo giro
                log.error("download zip PST fallito", extra={"url": url, "error": str(exc)})
                continue
            try:
                written = _extract_tifs(zip_path, url, root, wanted, spec.license)
            except zipfile.BadZipFile as exc:
                log.error("zip PST non valido", extra={"url": url, "error": str(exc)})
                written = []
            finally:
                zip_path.unlink(missing_ok=True)
            log.info("tavole estratte", extra={"product": product, "session": session_id, "count": len(written)})
        try:
            client.request("POST", f"{p['api_url']}/{session_id}/ack")
        except Exception as exc:  # l'ack è di cortesia: non blocca il download
            log.warning("ack PST fallito", extra={"session": session_id, "error": str(exc)})
    return [root / t.filename for t in tiles if is_complete(root, t.filename)]
