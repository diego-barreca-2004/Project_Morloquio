"""Client HTTP cortese: User-Agent onesto, intervallo minimo per host, retry con backoff, download atomici.

Nessun aggiramento di protezioni: niente browser automatici, niente User-Agent falsi.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from morloquio import __version__

USER_AGENT = f"Morloquio/{__version__} (+https://github.com/diego-barreca-2004/Project_Morloquio)"
_RETRY_STATUS = {429} | set(range(500, 600))


@dataclass(frozen=True)
class FileRecord:
    path: Path
    sha256: str
    size: int
    url: str
    fetched_at: str


class PoliteClient:
    def __init__(
        self,
        min_interval_s: float = 1.0,
        max_attempts: int = 5,
        backoff_s: float = 2.0,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _wait_turn(self, url: str) -> None:
        host = urlsplit(url).netloc
        now = self._clock()
        last = self._last_request.get(host)
        if last is not None and self.min_interval_s > 0:
            remaining = self.min_interval_s - (now - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request[host] = self._clock()

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.strip().isdigit():
                return float(retry_after)
        return self.backoff_s * (2**attempt)

    def _send(self, method: str, url: str, *, stream: bool = False, **kw) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            self._wait_turn(url)
            response = None
            try:
                request = self._client.build_request(method, url, **kw)
                response = self._client.send(request, stream=stream)
                if response.status_code not in _RETRY_STATUS:
                    if response.status_code >= 400:
                        if stream:
                            response.read()
                        response.raise_for_status()
                    return response
                if stream:
                    response.close()
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code} su {url}", request=request, response=response
                )
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < self.max_attempts - 1:
                self._sleep(self._retry_delay(attempt, response))
        assert last_exc is not None
        raise last_exc

    def request(self, method: str, url: str, **kw) -> httpx.Response:
        return self._send(method, url, **kw)

    def download(self, url: str, dest: Path, *, expected_sha256: str | None = None) -> FileRecord:
        """Scarica via ``dest.part``: il file finale esiste solo se completo e verificato."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(self.max_attempts):
            digest = hashlib.sha256()
            size = 0
            response = self._send("GET", url, stream=True)
            try:
                with open(part, "wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
                break
            except httpx.TransportError:
                # trasferimento interrotto a metà (rete lenta o instabile): si ricomincia da capo
                part.unlink(missing_ok=True)
                if attempt == self.max_attempts - 1:
                    raise
                self._sleep(self._retry_delay(attempt, None))
            finally:
                response.close()
        sha = digest.hexdigest()
        if expected_sha256 and sha != expected_sha256.lower():
            part.unlink(missing_ok=True)
            raise ValueError(f"sha256 diverso per {url}: atteso {expected_sha256}, ottenuto {sha}")
        os.replace(part, dest)
        return FileRecord(
            path=dest,
            sha256=sha,
            size=size,
            url=url,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
