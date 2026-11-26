import pytest

from morloquio.sources.catalog import load_catalog

_VALID = """
id: {id}
name: Test
provider: Test
license: {license}
license_url: https://example.org/license
attribution: Test
resolution_m: 1.0
protocol: http
adapter: test
params: {{}}
verified: "2026-11-25"
"""


def test_real_catalog_loads():
    cat = load_catalog()
    for sid in (
        "pst_lidar",
        "osm_geofabrik_sud",
        "osm_gps_traces",
        "sentinel2_c1_l2a",
        "copernicus_dem_glo30",
    ):
        assert sid in cat.accepted, cat.rejected.get(sid)
    assert "agea_ortho_2012" in cat.rejected
    assert "unverified" in cat.rejected["agea_ortho_2012"]
    pst = cat.accepted["pst_lidar"]
    assert pst.license == "CC-BY-4.0" and pst.params["batch_max"] == 20
    assert pst.params["products"] == ["dtm_1x1", "dsm_first_1x1", "dsm_last_1x1"]


def test_nc_source_rejected(tmp_path):
    (tmp_path / "nc.yaml").write_text(_VALID.format(id="nc", license="CC-BY-NC-4.0"))
    cat = load_catalog(tmp_path)
    assert "nc" in cat.rejected and not cat.accepted


def test_missing_field_raises(tmp_path):
    (tmp_path / "broken.yaml").write_text("id: broken\nname: x\n")
    with pytest.raises(ValueError, match="broken.yaml"):
        load_catalog(tmp_path)


def test_id_must_match_filename(tmp_path):
    (tmp_path / "a.yaml").write_text(_VALID.format(id="b", license="CC-BY-4.0"))
    with pytest.raises(ValueError, match="a.yaml"):
        load_catalog(tmp_path)
