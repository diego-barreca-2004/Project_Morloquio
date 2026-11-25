"""Benchmark del modello locale (Task 18): velocità, memoria e risposta su un'immagine di hillshade del pilota.

Uso:  uv run python scripts/bench_llm.py --image runs/<run>/bench/hillshade.png [--model inkling-2-mini:26b-a5b]
      uv run python scripts/bench_llm.py --make-image <hillshade.tif> --out hillshade.png  (ritaglio 512 px)

Usa l'API nativa di Ollama (``/api/chat``, streaming) per misurare il tempo al primo token e i token/s di
generazione dichiarati dal server (``eval_count`` / ``eval_duration``). Stampa una tabella Markdown.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio
from PIL import Image

from morloquio.agents.llm import load_llm_config

SHORT = "In una frase: che cos'è un modello digitale del terreno (DTM) ottenuto da LiDAR?"
LONG = (
    "Sei un ingegnere geomatico. Spiega in circa 300 parole, in italiano, come si riconosce un sentiero sotto "
    "il bosco in un DTM LiDAR a 1 m: quali forme del terreno lascia (gradino di versante, scarpata di monte, "
    "riporto a valle, solco), con quali derivati del terreno si vedono meglio (hillshade multidirezionale, "
    "Local Relief Model, curvature, pendenza) e quali falsi positivi sono tipici (fossi, terrazzamenti agricoli, "
    "muri a secco, linee di impluvio). Chiudi con tre criteri pratici per distinguere un sentiero da un fosso."
)
IMAGE_Q = (
    "Questa immagine è l'ombreggiatura (hillshade) di un DTM LiDAR a 1 m di una zona collinare della Calabria, "
    "512 × 512 m, nord in alto. Descrivi in 3–4 frasi la morfologia e indica se vedi strade o sentieri "
    "(linee continue, gradini di versante) e dove si trovano nell'immagine."
)
TERRAIN_WORDS = ("versant", "valle", "crinal", "rilievo", "pendi", "collin", "impluv", "dors")
ROUTE_WORDS = ("strad", "sentier", "pista", "tracciat", "carrare", "mulattier")


def make_image(tif: Path, out: Path, size: int = 512) -> Path:
    with rasterio.open(tif) as ds:
        r0, c0 = (ds.height - size) // 2, (ds.width - size) // 2
        arr = ds.read(1, window=((r0, r0 + size), (c0, c0 + size)))
    Image.fromarray(np.asarray(arr, dtype="uint8"), "L").save(out)
    return out


def _gpu_mib() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        used, total = (int(v) for v in out.split(","))
        return f"{used}/{total} MiB"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "n/d"


def _ram_used_gib() -> float:
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    total = int(info["MemTotal"].split()[0])
    avail = int(info["MemAvailable"].split()[0])
    return (total - avail) / 1024**2


def _ollama_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def run_prompt(root: str, model: str, prompt: str, image: Path | None, think: str | bool | None) -> dict:
    msg = {"role": "user", "content": prompt}
    if image is not None:
        msg["images"] = [base64.b64encode(image.read_bytes()).decode("ascii")]
    body = {"model": model, "messages": [msg], "stream": True}
    if think is not None:
        body["think"] = think
    t0 = time.monotonic()
    first = None
    text, thinking, final = [], [], {}
    with httpx.Client(timeout=1800) as client, client.stream("POST", f"{root}/api/chat", json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            m = chunk.get("message", {})
            if first is None and (m.get("content") or m.get("thinking")):
                first = time.monotonic() - t0
            text.append(m.get("content") or "")
            thinking.append(m.get("thinking") or "")
            if chunk.get("done"):
                final = chunk
    eval_s = final.get("eval_duration", 0) / 1e9
    return {
        "ttft_s": first,
        "total_s": time.monotonic() - t0,
        "eval_tokens": final.get("eval_count", 0),
        "tok_s": final.get("eval_count", 0) / eval_s if eval_s else float("nan"),
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "answer": "".join(text).strip(),
        "thinking_chars": len("".join(thinking)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--think", default=None, help="livello di ragionamento (es. low, high); default del modello")
    ap.add_argument("--make-image", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path, help="salva i risultati grezzi")
    args = ap.parse_args()
    if args.make_image:
        print(make_image(args.make_image, args.out or Path("hillshade.png")))
        return 0
    cfg = load_llm_config()
    model = args.model or cfg.model
    root = _ollama_root(cfg.base_url)
    ram_before = _ram_used_gib()
    cases = [("testo breve", SHORT, None), ("testo lungo", LONG, None)]
    if args.image:
        cases.append(("immagine hillshade", IMAGE_Q, args.image))
    results = []
    for name, prompt, image in cases:
        r = run_prompt(root, model, prompt, image, args.think)
        r.update(case=name, gpu=_gpu_mib(), ram_gib=_ram_used_gib())
        if image is not None:
            low = r["answer"].lower()
            r["image_ok"] = any(w in low for w in TERRAIN_WORDS) and any(w in low for w in ROUTE_WORDS)
        results.append(r)
    ps = httpx.get(f"{root}/api/ps", timeout=30).json()
    print(f"modello: {model} · think: {args.think or 'default'} · RAM prima: {ram_before:.1f} GiB")
    for m in ps.get("models", []):
        print(f"caricato: {m['name']} · totale {m['size'] / 2**30:.1f} GiB · in VRAM {m['size_vram'] / 2**30:.1f} GiB")
    print("\n| prova | token/s | primo token (s) | totale (s) | token generati | VRAM | RAM usata (GiB) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['case']} | {r['tok_s']:.1f} | {r['ttft_s']:.1f} | {r['total_s']:.1f} | {r['eval_tokens']} "
            f"| {r['gpu']} | {r['ram_gib']:.1f} |"
        )
    for r in results:
        print(f"\n### {r['case']}\n\n{r['answer']}")
        if "image_ok" in r:
            print(f"\n(controllo automatico parole chiave terreno + percorsi: {'ok' if r['image_ok'] else 'NO'})")
    if args.json:
        args.json.write_text(json.dumps({"model": model, "think": args.think, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
