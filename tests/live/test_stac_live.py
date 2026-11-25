import pytest

from morloquio.sources.adapters.stac import search_items
from morloquio.sources.catalog import load_catalog

pytestmark = pytest.mark.live


def test_search_pilot_s2():
    spec = load_catalog().accepted["sentinel2_c1_l2a"]
    items = search_items(
        spec, "sentinel-2-c1-l2a", (15.62, 38.03, 15.75, 38.12), "2025-06-01/2025-09-30", 10, 3
    )
    assert 1 <= len(items) <= 3
    assert "red" in items[0]["assets"]
