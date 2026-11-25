import json
import re
from datetime import UTC, datetime

from morloquio.common.runlog import get_logger, new_run_id


def test_run_id_format():
    rid = new_run_id(datetime(2026, 11, 25, 21, 30, 5, tzinfo=UTC))
    assert re.fullmatch(r"20261125T213005Z-[0-9a-f]{6}", rid)


def test_run_ids_unique():
    assert new_run_id() != new_run_id()


def test_log_line_is_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MORLOQUIO_RUNS_DIR", str(tmp_path))
    log = get_logger("test", run_id="r1", stage="fetch")
    log.info("ciao", extra={"tile_id": "N3807E01567"})
    lines = (tmp_path / "r1" / "log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["msg"] == "ciao" and rec["run_id"] == "r1" and rec["stage"] == "fetch"
    assert rec["tile_id"] == "N3807E01567" and rec["level"] == "INFO" and "ts" in rec
    err = capsys.readouterr().err.strip().splitlines()
    assert json.loads(err[-1])["msg"] == "ciao"
