from pathlib import Path

import geopandas as gpd
import pytest

from morloquio.sources.adapters.osm_extract import extract_layers

FIXTURE = Path(__file__).parents[1] / "fixtures" / "osm" / "mini.osm"
BBOX = (15.66, 38.06, 15.69, 38.09)


@pytest.fixture
def out(tmp_path):
    return tmp_path / "osm.gpkg"


def test_extract_counts(out):
    counts = extract_layers(FIXTURE, BBOX, out)
    assert counts == {"highways": 3, "waterways": 1, "barriers": 1, "buildings": 1}


def test_highway_columns(out):
    extract_layers(FIXTURE, BBOX, out)
    hw = gpd.read_file(out, layer="highways")
    assert hw.crs.to_epsg() == 4326
    for col in ("osm_id", "highway", "surface", "tracktype", "access", "bridge", "tunnel", "layer", "indoor"):
        assert col in hw.columns
    track = hw[hw.highway == "track"].iloc[0]
    assert track.bridge == "yes" and track.tracktype == "grade3" and track.osm_id == 101


def test_way_leaving_and_reentering_bbox_is_kept(out):
    import pyogrio

    extract_layers(FIXTURE, BBOX, out)
    # layer conforme alla specifica GeoPackage: tipo dichiarato multi, coerente con le geometrie ritagliate
    assert pyogrio.read_info(out, layer="highways")["geometry_type"] == "MultiLineString"
    assert pyogrio.read_info(out, layer="buildings")["geometry_type"] == "MultiPolygon"
    hw = gpd.read_file(out, layer="highways")
    way = hw[hw.osm_id == 107].geometry.iloc[0]
    assert way.geom_type == "MultiLineString" and len(way.geoms) == 2


def test_buildings_are_polygons(out):
    extract_layers(FIXTURE, BBOX, out)
    b = gpd.read_file(out, layer="buildings")
    assert b.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}


def test_fetch_pbf_verifies_md5_and_skips_when_unchanged(monkeypatch, tmp_path):
    import hashlib

    import httpx

    from morloquio.common.http import PoliteClient
    from morloquio.sources.adapters.osm_extract import fetch_pbf
    from morloquio.sources.catalog import load_catalog

    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    body = b"PBF-CONTENT"
    md5 = hashlib.md5(body).hexdigest()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if str(request.url).endswith(".md5"):
            return httpx.Response(200, text=f"{md5}  sud-latest.osm.pbf\n")
        return httpx.Response(200, content=body)

    spec = load_catalog().accepted["osm_geofabrik_sud"]
    client = PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, min_interval_s=0)
    rec = fetch_pbf(client, spec)
    assert rec.path.read_bytes() == body
    n = len(calls)
    fetch_pbf(client, spec)
    assert len(calls) == n + 1  # solo il .md5, nessun nuovo download


def test_fetch_pbf_md5_mismatch(monkeypatch, tmp_path):
    import httpx

    from morloquio.common.http import PoliteClient
    from morloquio.sources.adapters.osm_extract import fetch_pbf
    from morloquio.sources.catalog import load_catalog

    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))

    def handler(request):
        if str(request.url).endswith(".md5"):
            return httpx.Response(200, text="0" * 32 + "  sud-latest.osm.pbf\n")
        return httpx.Response(200, content=b"x")

    client = PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, min_interval_s=0)
    with pytest.raises(ValueError):
        fetch_pbf(client, load_catalog().accepted["osm_geofabrik_sud"])


def test_clip_drops_point_touches():
    from shapely.geometry import LineString, box

    from morloquio.sources.adapters.osm_extract import _clip_line

    clip = box(0, 0, 1, 1)
    assert _clip_line(LineString([(1, 1), (2, 2)]), clip) is None  # tocca solo l'angolo
    assert _clip_line(LineString([(0.5, 0.5), (2, 0.5)]), clip).length == 0.5
