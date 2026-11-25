"""Client del modello linguistico locale (Inkling 2 Mini servito da Ollama, API compatibile OpenAI).

Tutto gira sulla macchina: nessuna chiamata a servizi esterni. Le immagini vanno come ``image_url`` in base64.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from morloquio.common.paths import repo_root


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    timeout_s: float


def load_llm_config() -> LLMConfig:
    raw = yaml.safe_load((repo_root() / "configs" / "llm.yaml").read_text(encoding="utf-8"))
    return LLMConfig(base_url=raw["base_url"], model=raw["model"], timeout_s=float(raw["timeout_s"]))


def _image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def chat(
    messages: list[dict],
    images: list[Path] | None = None,
    config: LLMConfig | None = None,
    transport=None,
    **options,
) -> str:
    """Invia la conversazione e ritorna il testo della risposta. Le immagini si aggiungono all'ultimo messaggio."""
    cfg = config or load_llm_config()
    msgs = [dict(m) for m in messages]
    if images:
        last = msgs[-1]
        text = last["content"] if isinstance(last["content"], str) else ""
        last["content"] = [{"type": "text", "text": text}, *(_image_part(Path(p)) for p in images)]
    body = {"model": cfg.model, "messages": msgs, "stream": False, **options}
    with httpx.Client(timeout=cfg.timeout_s, transport=transport) as client:
        resp = client.post(f"{cfg.base_url.rstrip('/')}/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
