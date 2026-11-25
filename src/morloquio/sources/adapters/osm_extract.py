"""Estratto OpenStreetMap (Geofabrik) e ritaglio per AOI dei layer che servono alla pipeline.

Layer prodotti (GeoPackage, EPSG:4326):
- ``highways``: vie (riferimento e positivi);
- ``waterways``, ``barriers``, ``buildings``: negativi espliciti per lo Stadio B.
Gli edifici multipoligono (relazioni) non sono inclusi: bastano i contorni semplici come negativi.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import osmium
from shapely.geometry import LineString, MultiLineString, Polygon, box

from morloquio.common.http import FileRecord, PoliteClient
from morloquio.common.manifest import is_complete, read_manifest, update_manifest
from morloquio.common.paths import raw_dir
from morloquio.common.runlog import get_logger
from morloquio.sources.catalog import SourceSpec

SOURCE_ID = "osm_geofabrik_sud"
HIGHWAY_COLUMNS = [
    "osm_id",
    "highway",
    "surface",
    "tracktype",
    "access",
    "bridge",
    "tunnel",
    "layer",
    "indoor",
]
BARRIER_VALUES = {"wall", "retaining_wall", "fence", "city_wall"}
_SCHEMAS = {
    "highways": (HIGHWAY_COLUMNS, "MultiLineString"),
    "waterways": (["osm_id", "waterway"], "MultiLineString"),
    "barriers": (["osm_id", "barrier"], "MultiLineString"),
    "buildings": (["osm_id"], "MultiPolygon"),
}

log = get_logger("sources.osm_extract")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pbf(client: PoliteClient, spec: SourceSpec) -> FileRecord:
    """Scarica l'estratto PBF se è cambiato rispetto al manifest; verifica l'md5 pubblicato da Geofabrik."""
    root = raw_dir(SOURCE_ID)
    url = spec.params["pbf_url"]
    name = url.rsplit("/", 1)[-1]
    remote_md5 = client.request("GET", spec.params["md5_url"]).text.split()[0].lower()
    entry = read_manifest(root).get(name)
    if entry and entry.get("md5") == remote_md5 and is_complete(root, name):
        return FileRecord(root / name, entry["sha256"], entry["size"], entry["url"], entry["fetched_at"])
    record = client.download(url, root / name)
    if _md5(record.path) != remote_md5:
        record.path.unlink(missing_ok=True)
        raise ValueError(f"md5 diverso per {url}: atteso {remote_md5}")
    update_manifest(root, name, record, md5=remote_md5, license=spec.license)
    log.info("estratto OSM scaricato", extra={"url": url, "size": record.size})
    return record


def _clip_line(line: LineString, clip):
    """Parte lineare della via dentro il ritaglio; ``None`` se resta vuota o solo punti di contatto."""
    geom = line.intersection(clip)
    if geom.is_empty:
        return None
    if geom.geom_type in {"LineString", "MultiLineString"}:
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString" and not g.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiLineString(parts)


def _tag(tags, key: str):
    value = tags.get(key)
    return value if value is not None else None


def extract_layers(pbf: Path, bbox, out_gpkg: Path) -> dict[str, int]:
    """Ritaglia sul bbox (west, south, east, north) e scrive i layer nel GeoPackage; ritorna i conteggi."""
    clip = box(*bbox)
    rows: dict[str, list[dict]] = {k: [] for k in _SCHEMAS}
    way_filter = osmium.filter.KeyFilter("highway", "waterway", "barrier", "building").enable_for(
        osmium.osm.WAY
    )
    processor = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for obj in processor.with_filter(way_filter):
        if not obj.is_way():
            continue
        coords = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
        if len(coords) < 2 or len(coords) != len(obj.nodes):
            continue
        tags = obj.tags
        if "building" in tags:
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geom = Polygon(coords)
                if geom.is_valid and geom.intersects(clip):
                    rows["buildings"].append({"osm_id": obj.id, "geometry": geom.intersection(clip)})
            continue
        geom = _clip_line(LineString(coords), clip)
        if geom is None:
            continue
        if "highway" in tags:
            rec = {c: _tag(tags, c) for c in HIGHWAY_COLUMNS if c != "osm_id"}
            rows["highways"].append({"osm_id": obj.id, **rec, "geometry": geom})
        elif "waterway" in tags:
            rows["waterways"].append({"osm_id": obj.id, "waterway": tags.get("waterway"), "geometry": geom})
        elif tags.get("barrier") in BARRIER_VALUES:
            rows["barriers"].append({"osm_id": obj.id, "barrier": tags.get("barrier"), "geometry": geom})
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out_gpkg.unlink(missing_ok=True)
    counts = {}
    for layer, (columns, geom_type) in _SCHEMAS.items():
        gdf = gpd.GeoDataFrame(
            rows[layer], columns=[*columns, "geometry"], geometry="geometry", crs="EPSG:4326"
        )
        gdf.to_file(
            out_gpkg,
            layer=layer,
            driver="GPKG",
            engine="pyogrio",
            geometry_type=geom_type,
            promote_to_multi=True,
        )
        counts[layer] = len(gdf)
    log.info("layer OSM estratti", extra={"bbox": list(bbox), **counts})
    return counts
