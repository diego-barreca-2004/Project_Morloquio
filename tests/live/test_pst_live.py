import pytest
import rasterio

from morloquio.common.http import PoliteClient
from morloquio.sources.adapters.pst_lidar import fetch_product, list_tiles
from morloquio.sources.catalog import load_catalog

pytestmark = pytest.mark.live


def test_pilot_tile_dtm(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    spec = load_catalog().accepted["pst_lidar"]
    with PoliteClient() as client:
        tiles = [t for t in list_tiles(client, (15.67, 38.07, 15.68, 38.08), spec.params["grid_url"])]
        assert [t.tavola for t in tiles] == ["D38071567_0101"]
        paths = fetch_product(client, spec, "dtm_1x1", tiles)
    assert len(paths) == 1
    with rasterio.open(paths[0]) as ds:
        assert (ds.width, ds.height) == (1000, 1000)
        assert ds.dtypes[0] == "float32"
