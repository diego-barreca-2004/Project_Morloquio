import io
import json
import zipfile

import httpx
import pytest

from morloquio.common.http import PoliteClient
from morloquio.common.manifest import is_complete
from morloquio.sources.adapters import pst_lidar
from morloquio.sources.adapters.pst_lidar import PstTile, fetch_product, list_tiles, public_file_url
from morloquio.sources.catalog import load_catalog

API = "https://sim.mase.gov.it/core/api/esri/api/v1/downloads"
GRID = "https://gis.example/FeatureServer/2/query"


def _feature(fid, lat_i, lon_i, contratto="140"):
    return {
        "attributes": {
            "fid": fid,
            "tavola": f"D{lat_i:04d}{lon_i:04d}_0101",
            "regione": "Calabria",
            "contratto": contratto,
            "data": "090313",
        }
    }


def _tile(i):
    lat_i, lon_i = 3803 + i // 13, 1562 + i % 13
    return PstTile(
        fid=1000 + i,
        tavola=f"D{lat_i:04d}{lon_i:04d}_0101",
        regione="Calabria",
        contratto="140",
        date="090313",
        tile_id=f"N{lat_i:04d}E{lon_i:05d}",
    )


def _zip_for(tavolas, extra_members=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for tav in tavolas:
            zf.writestr(f"calabria_140_{tav}_DTM.tiff", f"TIFF-{tav}".encode())
        zf.writestr("area_selezionata/tile_selezionate.csv", "x")
        for name, data in extra_members:
            zf.writestr(name, data)
    return buf.getvalue()


class FakePst:
    """Simula griglia e API di download PST: una sessione per POST, uno zip con le tavole richieste."""

    def __init__(self, bad_batches=(), reason="OK", extra_members=()):
        self.posts = []
        self.acks = []
        self.bad_batches = set(bad_batches)
        self.reason = reason
        self.extra_members = extra_members
        self.sessions = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == API:
            body = json.loads(request.content)
            self.posts.append(body)
            n = len(self.posts)
            if n in self.bad_batches:
                return httpx.Response(202, json={"idSessione": None, "reason": "KO"})
            sid = f"s{n}"
            self.sessions[sid] = [t["tavola"] for t in body["tileList"]]
            return httpx.Response(202, json={"idSessione": sid, "reason": "OK", "numberOfUrl": 1})
        if request.method == "POST" and url.endswith("/ack"):
            self.acks.append(url.split("/")[-2])
            return httpx.Response(200)
        if request.method == "GET" and "/file/" in url:
            sid = url.split("/")[-3]
            return httpx.Response(200, content=_zip_for(self.sessions[sid], self.extra_members))
        if request.method == "GET" and url.startswith(API + "/"):
            sid = url.rsplit("/", 1)[-1]
            raw = f"http://mase-api-gateway.apps.dxap-prod.ocp.mase.priv/api/v1/downloads/{sid}/file/9"
            return httpx.Response(
                200,
                json={
                    "idSessione": sid,
                    "status": "ready",
                    "reason": self.reason,
                    "urls": [raw],
                    "expected": 1,
                },
            )
        return httpx.Response(404)


@pytest.fixture
def spec():
    return load_catalog().accepted["pst_lidar"]


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "data"


def _client(handler):
    return PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, min_interval_s=0)


def test_list_tiles_parses_and_pages():
    pages = [
        {"features": [_feature(311806, 3807, 1567)], "exceededTransferLimit": True},
        {"features": [_feature(311807, 3807, 1568), _feature(400000, 3812, 1567)]},  # 3812 fuori dal bbox
    ]
    offsets = []

    def handler(request):
        offsets.append(int(request.url.params["resultOffset"]))
        return httpx.Response(200, json=pages[len(offsets) - 1])

    tiles = list_tiles(_client(handler), (15.62, 38.03, 15.75, 38.12), GRID)
    assert offsets == [0, 1]
    assert PstTile(311806, "D38071567_0101", "Calabria", "140", "090313", "N3807E01567") in tiles
    assert [t.tile_id for t in tiles] == ["N3807E01567", "N3807E01568"]


def test_public_file_url():
    raw = "http://mase-api-gateway.apps.dxap-prod.ocp.mase.priv/api/v1/downloads/abc/file/9"
    assert public_file_url(API, raw) == API + "/abc/file/9"
    assert public_file_url(API, API + "/abc/file/9") == API + "/abc/file/9"


def test_batches_of_20(spec, data_dir):
    fake = FakePst()
    tiles = [_tile(i) for i in range(45)]
    paths = fetch_product(_client(fake.handler), spec, "dtm_1x1", tiles)
    assert [len(p["tileList"]) for p in fake.posts] == [20, 20, 5]
    assert fake.posts[0]["layer"] == "dtm_1x1" and fake.posts[0]["prodotto"] == "LIDAR_2014"
    assert fake.posts[0]["canale"] == "PST2" and fake.posts[0]["tileList"][0]["fid"] == "1000"
    assert len(paths) == 45
    root = data_dir / "raw" / "pst_lidar" / "dtm_1x1"
    assert (root / f"{tiles[0].tavola}.tif").read_bytes() == f"TIFF-{tiles[0].tavola}".encode()
    assert is_complete(root, f"{tiles[44].tavola}.tif")


def test_metadata_warning_is_success(spec):
    fake = FakePst(reason="METADATA_WARNING")
    paths = fetch_product(_client(fake.handler), spec, "dtm_1x1", [_tile(0)])
    assert len(paths) == 1


def test_bad_reason_raises_only_that_batch(spec):
    fake = FakePst(bad_batches={2})
    tiles = [_tile(i) for i in range(45)]
    paths = fetch_product(_client(fake.handler), spec, "dtm_1x1", tiles)
    assert len(paths) == 25  # batch 1 (20) + batch 3 (5)
    assert len(fake.posts) == 3


def test_bad_ready_reason_raises(spec):
    fake = FakePst(reason="ERROR")
    paths = fetch_product(_client(fake.handler), spec, "dtm_1x1", [_tile(0)])
    assert paths == []


def test_zip_member_mapped_by_tavola(spec, data_dir):
    evil = [("../evil.tif", b"x"), ("nested/../../evil2.tiff", b"y"), ("random_name.tif", b"z")]
    fake = FakePst(extra_members=evil)
    fetch_product(_client(fake.handler), spec, "dtm_1x1", [_tile(0)])
    root = data_dir / "raw" / "pst_lidar" / "dtm_1x1"
    tifs = sorted(p.name for p in root.glob("*.tif"))
    assert tifs == [f"{_tile(0).tavola}.tif"]
    assert not list(data_dir.rglob("evil*"))
    assert not list(root.parent.glob("*.tif"))


def test_skip_already_complete(spec):
    fake = FakePst()
    client = _client(fake.handler)
    fetch_product(client, spec, "dtm_1x1", [_tile(0)])
    fetch_product(client, spec, "dtm_1x1", [_tile(0)])
    assert len(fake.posts) == 1


def test_ack_after_download(spec):
    fake = FakePst()
    fetch_product(_client(fake.handler), spec, "dtm_1x1", [_tile(0), _tile(1)])
    assert fake.acks == ["s1"]


def test_wait_ready_times_out():
    def handler(request):
        return httpx.Response(200, json={"status": "pending", "reason": None, "urls": [], "expected": 1})

    clock = iter(range(0, 10_000, 10))
    with pytest.raises(pst_lidar.PstError):
        pst_lidar.wait_ready(
            _client(handler),
            API,
            "s1",
            timeout_s=30,
            poll_s=5,
            sleep=lambda s: None,
            clock=lambda: next(clock),
        )
