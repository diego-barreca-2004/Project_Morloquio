import httpx
import pytest

from morloquio.common.http import PoliteClient
from morloquio.sources.adapters.osm_gps import fetch_trackpoints, load_points, split_bbox
from morloquio.sources.catalog import load_catalog


def _gpx(n_points, tracks=1):
    pts_per = max(1, n_points // tracks)
    segs = []
    made = 0
    for t in range(tracks):
        k = pts_per if t < tracks - 1 else n_points - made
        pts = "".join(
            f'<trkpt lat="{38.07 + i * 1e-5:.6f}" lon="{15.67 + i * 1e-5:.6f}"/>'
            for i in range(made, made + k)
        )
        made += k
        segs.append(f"<trk><trkseg>{pts}</trkseg></trk>")
    return f'<?xml version="1.0"?><gpx version="1.0">{"".join(segs)}</gpx>'


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture
def spec():
    return load_catalog().accepted["osm_gps_traces"]


def test_pages_until_short_page(spec):
    sizes = [5000, 5000, 12]
    pages = []

    def handler(request):
        page = int(request.url.params["page"])
        pages.append(page)
        return httpx.Response(200, text=_gpx(sizes[page]))

    client = PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, min_interval_s=0)
    paths = fetch_trackpoints(client, spec, (15.66, 38.06, 15.69, 38.09))
    assert pages == [0, 1, 2]
    assert len(paths) == 3


def test_split_bbox_respects_limit():
    parts = split_bbox((15.0, 38.0, 16.0, 38.5), max_area_deg2=0.25)
    assert all((e - w) * (n - s) <= 0.25 + 1e-12 for w, s, e, n in parts)
    total = sum((e - w) * (n - s) for w, s, e, n in parts)
    assert total == pytest.approx(0.5)


def test_small_bbox_not_split():
    assert split_bbox((15.66, 38.06, 15.69, 38.09)) == [(15.66, 38.06, 15.69, 38.09)]


def test_load_points_counts(tmp_path):
    p = tmp_path / "page_0.gpx"
    p.write_text(_gpx(5, tracks=2))
    gdf = load_points([p])
    assert len(gdf) == 5
    assert gdf["track"].nunique() == 2
    assert gdf.crs.to_epsg() == 4326
