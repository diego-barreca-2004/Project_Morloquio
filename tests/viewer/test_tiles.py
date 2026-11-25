import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from morloquio.viewer.tiles import render_xyz


def _block():
    """Blocco 2x2 di tile z14 allineato (sta dentro una sola tile z13 e z12), vicino a Morloquio."""
    t = mercantile.tile(15.6777, 38.0756, 14)
    x0, y0 = t.x - t.x % 2, t.y - t.y % 2
    tl = mercantile.xy_bounds(mercantile.Tile(x0, y0, 14))
    br = mercantile.xy_bounds(mercantile.Tile(x0 + 1, y0 + 1, 14))
    return x0, y0, (tl.left, br.bottom, br.right, tl.top)


def _raster(path, bounds, data, crs="EPSG:3857", nodata=float("nan")):
    h, w = data.shape[-2:]
    count = 1 if data.ndim == 2 else data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=from_bounds(*bounds, w, h),
        nodata=nodata,
    ) as ds:
        ds.write(data if data.ndim == 3 else data[None])
    return path


def test_render_xyz_counts_and_alpha(tmp_path):
    x0, y0, bounds = _block()
    data = np.linspace(0, 1, 512 * 512, dtype="float32").reshape(512, 512)
    data[:64, :64] = np.nan
    src = _raster(tmp_path / "s.tif", bounds, data)
    n = render_xyz(src, tmp_path / "xyz", zooms=range(12, 15), mode="gray")
    assert n == 1 + 1 + 4
    assert len(list((tmp_path / "xyz").rglob("*.png"))) == 6
    img = np.asarray(Image.open(tmp_path / "xyz" / "14" / str(x0) / f"{y0}.png"))
    assert img.shape == (256, 256, 4)
    assert img[10, 10, 3] == 0 and img[200, 200, 3] == 255


def test_render_heat_is_transparent_on_low_scores(tmp_path):
    x0, y0, bounds = _block()
    data = np.zeros((512, 512), dtype="float32")
    data[:, 300:310] = 0.9
    src = _raster(tmp_path / "s.tif", bounds, data)
    render_xyz(src, tmp_path / "xyz", zooms=range(14, 15), mode="heat")
    img = np.asarray(Image.open(tmp_path / "xyz" / "14" / str(x0 + 1) / f"{y0}.png"))
    assert img[100, 10, 3] == 0  # punteggio 0: trasparente
    assert img[100, (300 - 256) * 1 + 4, 3] > 150  # punteggio alto: visibile


def test_render_mosaic_uses_cores(tmp_path):
    x0, y0, bounds = _block()
    w, s, e, n = bounds
    mid = (w + e) / 2
    a = _raster(tmp_path / "a.tif", (w, s, mid + 500, n), np.full((512, 300), 0.2, dtype="float32"))
    b = _raster(tmp_path / "b.tif", (mid - 500, s, e, n), np.full((512, 300), 0.8, dtype="float32"))
    from shapely.geometry import box

    cores = [box(w, s, mid, n), box(mid, s, e, n)]
    render_xyz([a, b], tmp_path / "xyz", zooms=range(14, 15), mode="heat", cores=cores)
    left = np.asarray(Image.open(tmp_path / "xyz" / "14" / str(x0) / f"{y0}.png"))
    right = np.asarray(Image.open(tmp_path / "xyz" / "14" / str(x0 + 1) / f"{y0}.png"))
    # 0.2 è quasi trasparente, 0.8 no: a destra del confine comanda b anche dove a ha l'alone
    assert left[128, 250, 3] < right[128, 5, 3]


def test_render_rgb_uses_mask(tmp_path):
    x0, y0, bounds = _block()
    rgb = np.full((3, 512, 512), 120, dtype="uint8")
    path = tmp_path / "o.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_bounds(*bounds, 512, 512),
    ) as ds:
        ds.write(rgb)
        mask = np.full((512, 512), 255, dtype="uint8")
        mask[:64, :64] = 0
        ds.write_mask(mask)
    render_xyz(path, tmp_path / "xyz", zooms=range(14, 15), mode="rgb")
    img = np.asarray(Image.open(tmp_path / "xyz" / "14" / str(x0) / f"{y0}.png"))
    assert img[10, 10, 3] == 0 and img[200, 200, 3] == 255 and abs(int(img[200, 200, 0]) - 120) <= 2
