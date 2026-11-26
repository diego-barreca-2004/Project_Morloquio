import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from morloquio.eval.sweep import threshold_curve

CRS = "EPSG:32633"
X0, Y0 = 560000.0, 4215000.0


def _prob(path):
    prob = np.zeros((200, 200), "float32")
    prob[20:181, 50] = 0.8  # via OSM a x = X0 + 50,5, lunga quanto la via
    prob[20:181, 150] = 0.3  # traccia debole senza OSM, stessa lunghezza
    prob[:10, :] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=200,
        height=200,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(X0, Y0, 1, 1),
        nodata=float("nan"),
    ) as ds:
        ds.write(prob, 1)
    return path


def test_threshold_curve_trades_completeness_for_correctness(tmp_path):
    ref = gpd.GeoDataFrame(
        {"highway": ["track"]}, geometry=[LineString([(X0 + 50.5, Y0 - 20), (X0 + 50.5, Y0 - 180)])], crs=CRS
    )
    core = box(X0, Y0 - 200, X0 + 200, Y0)
    curve = threshold_curve({"T": _prob(tmp_path / "p.tif")}, ref, {"T": core}, [0.2, 0.5, 0.9], buffer_m=5.0)
    rows = curve.set_index("threshold")
    assert rows.loc[0.2, "completeness"] == pytest.approx(1.0) and rows.loc[0.2, "correctness"] == pytest.approx(
        0.5, abs=0.05
    )
    assert rows.loc[0.5, "completeness"] == pytest.approx(1.0) and rows.loc[0.5, "correctness"] == pytest.approx(1.0)
    assert rows.loc[0.9, "completeness"] == pytest.approx(0.0)
    assert rows.loc[0.2, "ref_km"] == pytest.approx(0.16, abs=0.002)
