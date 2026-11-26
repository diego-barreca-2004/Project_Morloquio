import hashlib
import json

from morloquio.common.http import FileRecord
from morloquio.common.manifest import is_complete, read_manifest, update_manifest


def _record(path, data: bytes):
    path.write_bytes(data)
    return FileRecord(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        url="https://example.org/" + path.name,
        fetched_at="2026-11-25T00:00:00Z",
    )


def test_update_and_read(tmp_path):
    rec = _record(tmp_path / "a.tif", b"abc")
    update_manifest(tmp_path, "a.tif", rec, license="CC-BY-4.0")
    m = read_manifest(tmp_path)
    assert m["a.tif"]["sha256"] == rec.sha256 and m["a.tif"]["license"] == "CC-BY-4.0"
    assert json.loads((tmp_path / "manifest.json").read_text())["a.tif"]["size"] == 3


def test_read_missing_manifest_is_empty(tmp_path):
    assert read_manifest(tmp_path) == {}


def test_complete_when_sha_matches(tmp_path):
    rec = _record(tmp_path / "a.tif", b"abc")
    update_manifest(tmp_path, "a.tif", rec)
    assert is_complete(tmp_path, "a.tif")


def test_partial_file_not_complete(tmp_path):
    (tmp_path / "a.tif.part").write_bytes(b"ab")
    assert not is_complete(tmp_path, "a.tif")


def test_changed_file_not_complete(tmp_path):
    rec = _record(tmp_path / "a.tif", b"abc")
    update_manifest(tmp_path, "a.tif", rec)
    (tmp_path / "a.tif").write_bytes(b"abX")
    assert not is_complete(tmp_path, "a.tif")


def test_file_without_manifest_not_complete(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"abc")
    assert not is_complete(tmp_path, "a.tif")
