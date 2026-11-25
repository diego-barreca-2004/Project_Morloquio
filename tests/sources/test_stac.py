import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from morloquio.sources.adapters import stac
from morloquio.sources.adapters.stac import fetch_collection, fetch_window, select_items
from morloquio.sources.catalog import load_catalog


def _item(i, cloud, date):
    return {"id": f"it{i}", "properties": {"eo:cloud_cover": cloud, "datetime": date}, "assets": {}}


def test_select_items_orders_by_cloud_then_date():
    items = [
        _item(1, 5.0, "2024-01-01T10:00:00Z"),
        _item(2, 1.0, "2024-02-01T10:00:00Z"),
        _item(3, 1.0, "2025-03-01T10:00:00Z"),
        _item(4, 50.0, "2025-04-01T10:00:00Z"),
    ]
    sel = select_items(items, max_cloud=10, max_items=2)
    assert [i["id"] for i in sel] == ["it3", "it2"]


def test_select_items_without_cloud_field():
    items = [{"id": "dem", "properties": {"datetime": None}, "assets": {}}]
    assert [i["id"] for i in select_items(items, max_cloud=None, max_items=5)] == ["dem"]


@pytest.fixture
def cog(tmp_path):
    # raster 1000x1000 a 10 m in EPSG:32633 attorno al punto pilota
    path = tmp_path / "src.tif"
    transform = from_origin(560000, 4215000, 10, 10)
    data = np.arange(1000 * 1000, dtype="uint16").reshape(1000, 1000)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1000,
        height=1000,
        count=1,
        dtype="uint16",
        crs="EPSG:32633",
        transform=transform,
        tiled=True,
    ) as ds:
        ds.write(data, 1)
    return path


def test_fetch_window_crops_bbox(cog, tmp_path):
    with rasterio.open(cog) as ds:
        from rasterio.warp import transform_bounds

        w, s, e, n = transform_bounds("EPSG:32633", "EPSG:4326", 561000, 4211000, 563000, 4213000)
    out = fetch_window(str(cog), (w, s, e, n), tmp_path / "out.tif")
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 32633
        # la finestra contiene il riquadro di 2x2 km (200x200 px) e poco altro
        assert 200 <= ds.width <= 215 and 200 <= ds.height <= 215
        left, bottom, right, top = ds.bounds
        assert left <= 561000 and right >= 563000 and bottom <= 4211000 and top >= 4213000


def test_fetch_collection_writes_assets_and_skips_existing(monkeypatch, tmp_path, cog):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    spec = load_catalog().accepted["sentinel2_c1_l2a"]
    item = _item(7, 1.0, "2025-01-01T10:00:00Z")
    item["assets"] = {a: {"href": str(cog)} for a in spec.params["assets"]}
    calls = []
    monkeypatch.setattr(stac, "search_items", lambda *a, **k: [item])
    real = stac.fetch_window

    def counting(href, bbox, dest):
        calls.append(dest)
        return real(href, bbox, dest)

    monkeypatch.setattr(stac, "fetch_window", counting)
    bbox = (15.68, 38.06, 15.69, 38.07)
    paths = fetch_collection(spec, bbox)
    assert len(paths) == len(spec.params["assets"])
    assert all(p.exists() and p.parent.name == "it7" for p in paths)
    fetch_collection(spec, bbox)
    assert len(calls) == len(spec.params["assets"])
