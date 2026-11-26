"""Identificatori di esecuzione e log strutturati JSON (una riga per evento)."""

from __future__ import annotations

import json
import logging
import secrets
import sys
from datetime import UTC, datetime

from morloquio.common.paths import runs_dir

# Attributi standard di LogRecord: tutto il resto è contesto da serializzare
_STD_ATTRS = set(vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None))) | {"message", "asctime"}


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rec = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rec.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS})
        if record.exc_info:
            rec["exc"] = self.formatException(record.exc_info)
        return json.dumps(rec, ensure_ascii=False, default=str)


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


def get_logger(name: str, run_id: str | None = None, **context) -> logging.LoggerAdapter:
    """Logger JSON su stderr e, se c'è ``run_id``, anche su ``runs/<run_id>/log.jsonl``."""
    logger = logging.getLogger(f"morloquio.{name}" + (f".{run_id}" if run_id else ""))
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = JsonFormatter()
        err = logging.StreamHandler(sys.stderr)
        err.setFormatter(fmt)
        logger.addHandler(err)
        if run_id:
            path = runs_dir() / run_id / "log.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    ctx = {"run_id": run_id, **context} if run_id else dict(context)
    return _ContextAdapter(logger, ctx)
