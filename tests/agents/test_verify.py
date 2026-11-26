import json

import geopandas as gpd
import httpx
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import LineString

from morloquio.agents.llm import LLMConfig
from morloquio.agents.verify import ask_visible, render_crop, sample_segments, summarize

CRS = "EPSG:32633"
X0, Y0 = 560000.0, 4215000.0
CFG = LLMConfig(base_url="http://127.0.0.1:11434/v1", model="inkling-2-mini:26b-a5b", timeout_s=5)


def _gdf(lines):
    return gpd.GeoDataFrame(geometry=lines, crs=CRS)


def test_sample_extra_and_missed_segments():
    reference = _gdf(
        [LineString([(X0, Y0 - 50), (X0 + 200, Y0 - 50)]), LineString([(X0, Y0 - 300), (X0 + 80, Y0 - 300)])]
    )
    candidates = _gdf(
        [LineString([(X0, Y0 - 51), (X0 + 200, Y0 - 51)]), LineString([(X0, Y0 - 150), (X0 + 60, Y0 - 150)])]
    )
    extra = sample_segments(candidates, reference, buffer_m=5.0, n=10, seed=0, min_length_m=20.0)
    missed = sample_segments(reference, candidates, buffer_m=5.0, n=10, seed=0, min_length_m=20.0)
    assert len(extra) == 1 and abs(extra.geometry.iloc[0].centroid.y - (Y0 - 150)) < 1
    assert len(missed) == 1 and abs(missed.geometry.iloc[0].centroid.y - (Y0 - 300)) < 1


def test_render_crop_draws_segment(tmp_path):
    path = tmp_path / "hs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=400,
        height=400,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(X0, Y0, 1, 1),
    ) as ds:
        ds.write(np.full((400, 400), 128, "uint8"), 1)
    seg = LineString([(X0 + 150, Y0 - 200), (X0 + 250, Y0 - 200)])
    png = render_crop(seg, [path], tmp_path / "c.png", size_m=128, scale=4)
    img = np.asarray(Image.open(png).convert("RGB"))
    assert img.shape == (512, 512, 3)
    red = (img[..., 0] > 200) & (img[..., 1] < 80)
    assert red.any() and not red[:200].any()  # la linea sta al centro, non in alto


def _transport(text):
    return httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": text}}]}))


def test_ask_visible_parses_json_even_in_code_fence(tmp_path):
    png = tmp_path / "c.png"
    Image.new("RGB", (8, 8)).save(png)
    text = '```json\n{"via_visibile": true, "confidenza": 0.8, "motivo": "gradino lungo il versante"}\n```'
    ans = ask_visible(png, "extra", CFG, transport=_transport(text))
    assert ans["via_visibile"] is True and ans["confidenza"] == 0.8
    assert ask_visible(png, "missed", CFG, transport=_transport("non so"))["via_visibile"] is None


def test_summarize_fractions():
    answers = {
        "extra": [{"via_visibile": True}, {"via_visibile": False}, {"via_visibile": None}, {"via_visibile": True}],
        "missed": [{"via_visibile": False}],
    }
    s = summarize(answers)
    assert s["extra"] == {"n": 4, "risposte": 3, "visibile": 2 / 3}
    assert json.dumps(s)
