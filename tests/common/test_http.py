import hashlib

import httpx
import pytest

from morloquio.common.http import USER_AGENT, PoliteClient


class _Recorder:
    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


def _client(handler, **kw):
    rec = _Recorder()
    return PoliteClient(transport=httpx.MockTransport(handler), sleep=rec, min_interval_s=0, **kw), rec


def test_user_agent_sent():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, text="ok")

    client, _ = _client(handler)
    client.request("GET", "https://example.org/x")
    assert seen["ua"] == USER_AGENT
    assert "Morloquio/" in USER_AGENT and "github.com/diego-barreca-2004/Project_Morloquio" in USER_AGENT


def test_retry_on_503_then_ok():
    codes = iter([503, 503, 200])

    def handler(request):
        return httpx.Response(next(codes))

    client, rec = _client(handler, backoff_s=2.0)
    assert client.request("GET", "https://example.org/x").status_code == 200
    assert len(rec.sleeps) == 2 and rec.sleeps[1] > rec.sleeps[0]


def test_honors_retry_after():
    codes = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])

    def handler(request):
        return next(codes)

    client, rec = _client(handler)
    client.request("GET", "https://example.org/x")
    assert rec.sleeps == [7.0]


def test_gives_up_after_max_attempts():
    def handler(request):
        return httpx.Response(500)

    client, rec = _client(handler, max_attempts=3)
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.org/x")
    assert len(rec.sleeps) == 2


def test_client_error_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    client, _ = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.request("GET", "https://example.org/x")
    assert len(calls) == 1


def test_min_interval_between_requests_same_host():
    clock = iter([0.0, 0.0, 0.2, 0.2])
    rec = _Recorder()
    client = PoliteClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        sleep=rec,
        min_interval_s=1.0,
        clock=lambda: next(clock),
    )
    client.request("GET", "https://example.org/a")
    client.request("GET", "https://example.org/b")
    assert rec.sleeps and rec.sleeps[0] == pytest.approx(0.8)


def test_download_atomic(tmp_path):
    body = b"x" * 1000

    def handler(request):
        return httpx.Response(200, content=body)

    client, _ = _client(handler)
    dest = tmp_path / "a" / "f.bin"
    rec = client.download("https://example.org/f.bin", dest)
    assert dest.read_bytes() == body
    assert not dest.with_name("f.bin.part").exists()
    assert rec.sha256 == hashlib.sha256(body).hexdigest() and rec.size == 1000
    assert rec.url == "https://example.org/f.bin"


def test_sha_mismatch_raises(tmp_path):
    client, _ = _client(lambda r: httpx.Response(200, content=b"abc"))
    dest = tmp_path / "f.bin"
    with pytest.raises(ValueError):
        client.download("https://example.org/f.bin", dest, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not dest.with_name("f.bin.part").exists()
