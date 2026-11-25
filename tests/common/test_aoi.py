import pytest

from morloquio.common.aoi import load_aoi


def test_load_aoi():
    a = load_aoi("morloquio")
    assert a.name == "morloquio"
    assert a.work_crs == "EPSG:32633"
    assert a.bbox == (15.62, 38.03, 15.75, 38.12)
    assert a.core_bbox == (15.6663, 38.0666, 15.6891, 38.0846)


def test_unknown_aoi():
    with pytest.raises(FileNotFoundError):
        load_aoi("non-esiste")
