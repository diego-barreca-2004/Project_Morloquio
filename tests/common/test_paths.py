from morloquio.common.paths import data_dir, interim_dir, processed_dir, raw_dir, repo_root, runs_dir


def test_repo_root_has_pyproject():
    assert (repo_root() / "pyproject.toml").exists()


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert raw_dir("x") == tmp_path / "raw" / "x"
    assert raw_dir("x").is_dir()
    assert interim_dir("osm") == tmp_path / "interim" / "osm"
    assert processed_dir("cube") == tmp_path / "processed" / "cube"


def test_data_dir_default(monkeypatch):
    monkeypatch.delenv("MORLOQUIO_DATA_DIR", raising=False)
    assert data_dir() == repo_root() / "data"


def test_runs_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path / "r"))
    assert runs_dir() == tmp_path / "r"
    assert runs_dir().is_dir()
