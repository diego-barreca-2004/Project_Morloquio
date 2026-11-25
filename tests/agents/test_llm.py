import base64
import json

import httpx
import pytest

from morloquio.agents.llm import LLMConfig, chat, load_llm_config

CFG = LLMConfig(base_url="http://127.0.0.1:11434/v1", model="inkling-2-mini:26b-a5b", timeout_s=5)


def _transport(seen):
    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "un sentiero"}}]})

    return httpx.MockTransport(handler)


def test_load_llm_config():
    cfg = load_llm_config()
    assert cfg.base_url == "http://127.0.0.1:11434/v1"
    assert cfg.model == "inkling-2-mini:26b-a5b" and cfg.timeout_s == 600


def test_chat_sends_openai_payload():
    seen = {}
    out = chat([{"role": "user", "content": "ciao"}], config=CFG, transport=_transport(seen))
    assert out == "un sentiero"
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen["body"]["model"] == "inkling-2-mini:26b-a5b"
    assert seen["body"]["messages"] == [{"role": "user", "content": "ciao"}]


def test_chat_encodes_image(tmp_path):
    png = tmp_path / "h.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfinto")
    seen = {}
    chat([{"role": "user", "content": "cosa vedi?"}], images=[png], config=CFG, transport=_transport(seen))
    content = seen["body"]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "cosa vedi?"}
    url = content[1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def test_chat_error_is_raised():
    transport = httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(httpx.HTTPStatusError):
        chat([{"role": "user", "content": "x"}], config=CFG, transport=transport)
