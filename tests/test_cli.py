import json
import threading
import urllib.request

import pytest

from morloquio import cli
from morloquio.cli import Stage, main


@pytest.fixture(autouse=True)
def dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _marker(ctx, tile):
    return ctx.run_dir / "fake" / f"{tile}.done"


def _fake_run(ctx, tile):
    if tile == "bad":
        raise RuntimeError("tile rotta")
    _marker(ctx, tile).parent.mkdir(parents=True, exist_ok=True)
    _marker(ctx, tile).write_text("ok")


def _fake_done(ctx, tile):
    return _marker(ctx, tile).exists()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setitem(cli.STAGES, "fake", Stage("fake", per_tile=True, run=_fake_run, done=_fake_done))


def _runjson(dirs, run_id="r1"):
    return json.loads((dirs / "runs" / run_id / "run.json").read_text())


def test_run_records_per_tile_status(fake, dirs):
    code = main(["run", "fake", "--aoi", "morloquio", "--tiles", "a,bad,c", "--run-id", "r1"])
    assert code == 1
    rj = _runjson(dirs)
    assert rj["tiles"]["a"]["fake"] == "ok" and rj["tiles"]["c"]["fake"] == "ok"
    assert rj["tiles"]["bad"]["fake"].startswith("error: RuntimeError: tile rotta")
    assert rj["aoi"] == "morloquio" and rj["stages"] == ["fake"] and "fake" in rj["params_hash"]
    assert "started_at" in rj and "git_commit" in rj


def test_run_with_process_pool(fake, dirs):
    code = main(["run", "fake", "--aoi", "morloquio", "--tiles", "a,b,c,d", "--run-id", "r2", "--workers", "2"])
    assert code == 0
    assert {t: s["fake"] for t, s in _runjson(dirs, "r2")["tiles"].items()} == dict.fromkeys("abcd", "ok")


def test_idempotent_skip(fake, dirs):
    assert main(["run", "fake", "--aoi", "morloquio", "--tiles", "a,c", "--run-id", "r1"]) == 0
    assert main(["run", "fake", "--aoi", "morloquio", "--tiles", "a,c", "--run-id", "r1"]) == 0
    assert {t: s["fake"] for t, s in _runjson(dirs)["tiles"].items()} == {"a": "skipped", "c": "skipped"}
    assert main(["run", "fake", "--aoi", "morloquio", "--tiles", "a", "--run-id", "r1", "--force"]) == 0
    assert _runjson(dirs)["tiles"]["a"]["fake"] == "ok"


def test_missing_lidar_tile_continues(dirs):
    cube = dirs / "data" / "processed" / "cube" / "N3807E01567"
    cube.mkdir(parents=True)
    (cube / "meta.json").write_text(json.dumps({"tile_id": "N3807E01567", "tier": "T3", "sources": {}}))
    code = main(["run", "stage_a", "--aoi", "morloquio", "--tiles", "N3807E01567", "--run-id", "r1"])
    assert code == 0
    assert _runjson(dirs)["tiles"]["N3807E01567"]["stage_a"] == "skipped: tier T3"


def test_tile_without_cube_is_skipped_by_terrain(dirs):
    assert main(["run", "terrain", "--aoi", "morloquio", "--tiles", "N3807E01567", "--run-id", "r1"]) == 0
    assert _runjson(dirs)["tiles"]["N3807E01567"]["terrain"] == "skipped: nessun cubo"


def test_terrain_redone_when_cube_is_newer(dirs):
    import os

    from morloquio.common.cache import params_hash
    from morloquio.terrain.derivatives import LAYERS

    tile = "N3807E01567"
    cube = dirs / "data" / "processed" / "cube" / tile
    cube.mkdir(parents=True)
    (cube / "dtm.tif").write_bytes(b"x")
    ctx = cli.RunContext(aoi=cli.load_aoi("morloquio"), run_id="r1", params=cli.stage_params())
    terrain = dirs / "data" / "processed" / "terrain" / params_hash(ctx.params["terrain"]) / tile
    terrain.mkdir(parents=True)
    for name in LAYERS:
        (terrain / f"{name}.tif").write_bytes(b"x")
        os.utime(terrain / f"{name}.tif", (1_000_000, 1_000_000))
    assert cli.STAGES["terrain"].done(ctx, tile) is False  # cubo rifatto dopo i derivati
    os.utime(cube / "dtm.tif", (500_000, 500_000))
    assert cli.STAGES["terrain"].done(ctx, tile) is True


def test_sources_command_lists_rejected(capsys):
    assert main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "agea_ortho_2012" in out and "unverified" in out
    assert "pst_lidar" in out and "CC-BY-4.0" in out


def test_fetch_dispatches_by_adapter_and_skips_rejected(monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(cli.FETCHERS, "osm_gps", lambda client, spec, aoi: calls.append(spec.id))
    assert main(["fetch", "--aoi", "morloquio", "--sources", "osm_gps_traces"]) == 0
    assert calls == ["osm_gps_traces"]
    assert main(["fetch", "--aoi", "morloquio", "--sources", "agea_ortho_2012"]) == 1
    assert "agea_ortho_2012" in capsys.readouterr().out
    assert main(["fetch", "--aoi", "morloquio", "--sources", "nessuna"]) == 2


def test_view_server_binds_localhost(dirs):
    viewer = dirs / "runs" / "r1" / "viewer"
    viewer.mkdir(parents=True)
    (viewer / "index.html").write_text("<html>ok</html>")
    server = cli.make_server("r1", 0)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        threading.Thread(target=server.handle_request, daemon=True).start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=5) as resp:
            assert b"ok" in resp.read()
    finally:
        server.server_close()


def test_stage_b_stages_registered():
    assert cli.B_ORDER == ["b_norm", "b_features", "b_train", "b_predict", "b_merge", "b_eval", "viewer"]
    assert all(s in cli.STAGES for s in cli.B_ORDER)
    assert cli.build_parser().parse_args(["run", "b", "--aoi", "morloquio"]).stage == "b"
    assert "stage_b" in cli.stage_params()


def test_stage_can_report_its_own_skip(monkeypatch, dirs):
    monkeypatch.setitem(
        cli.STAGES, "maybe", Stage("maybe", per_tile=True, run=lambda c, t: "skipped: niente da fare", done=_fake_done)
    )
    ctx = cli.RunContext(aoi=cli.load_aoi("morloquio"), run_id="r1")
    assert cli.run_stages(["maybe"], ctx, ["t1"]) == 0
    assert _runjson(dirs)["tiles"]["t1"]["maybe"] == "skipped: niente da fare"


def test_b_predict_skips_without_model(dirs):
    ctx = cli.RunContext(aoi=cli.load_aoi("morloquio"), run_id="r1", params=cli.stage_params())
    (ctx.run_dir / "stage_b").mkdir(parents=True)
    (ctx.run_dir / "stage_b" / "folds.json").write_text(json.dumps({"N3807E01567": 0}))
    cli._b_predict_run(ctx, None)
    report = json.loads((ctx.run_dir / "stage_b" / "predict.json").read_text())
    assert report["N3807E01567"].startswith("skipped: nessun modello")
