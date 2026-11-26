"""Verifica a campione con Inkling in locale: le vie trovate dalla rete e assenti in OSM ("extra") si vedono davvero
nel terreno? Le vie OSM non trovate ("missed") si vedono nel LiDAR?

Contro OSM incompleto la correttezza è solo un limite inferiore e la completezza mescola errori della rete con vie
OSM invisibili nel LiDAR (tracciate da GPS o da foto, spostate, sotto edifici). Un giudizio visivo indipendente su
un campione separa i due casi. Inkling non è un rilevatore affidabile (docs/benchmarks/inkling-2-mini.md): i numeri
sono una stima, da leggere insieme ai ritagli PNG salvati accanto.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import rasterio
import shapely
from PIL import Image, ImageDraw
from rasterio.windows import from_bounds
from shapely.geometry import LineString
from shapely.ops import substring

from morloquio.agents.llm import LLMConfig, chat
from morloquio.common.runlog import get_logger

log = get_logger("agents.verify")

_COMMON = (
    "Immagine: ombreggiatura del terreno ricavata dal LiDAR (modello del terreno a 1 m, sotto la vegetazione), "
    "copre {size} × {size} m, nord in alto. "
)
PROMPTS = {
    "extra": _COMMON
    + "La linea rossa è un tracciato proposto da un modello automatico. Nel terreno sotto la linea rossa si vede una "
    "via percorribile (strada, pista, mulattiera o sentiero: un gradino piano sul versante, un solco, una fascia "
    "liscia e continua)? ",
    "missed": _COMMON
    + "La linea rossa è una via presente in OpenStreetMap, forse spostata di qualche metro. Nel terreno sotto o "
    "vicino alla linea rossa (entro circa 10 m) si vede una via percorribile? ",
}
_ANSWER = (
    'Rispondi solo con JSON: {"via_visibile": true oppure false, "confidenza": numero da 0 a 1, '
    '"motivo": "al massimo 15 parole"}.'
)


def sample_segments(
    subject: gpd.GeoDataFrame,
    other: gpd.GeoDataFrame,
    buffer_m: float,
    n: int,
    seed: int,
    min_length_m: float = 20.0,
    max_length_m: float = 60.0,
) -> gpd.GeoDataFrame:
    """Pezzi di ``subject`` lontani più di ``buffer_m`` da ``other`` (lunghi ≥ ``min_length_m``), campione di ``n``."""
    near = shapely.union_all(other.geometry.buffer(buffer_m).values) if len(other) else shapely.Polygon()
    pieces = []
    for geom in subject.geometry:
        rest = geom.difference(near)
        for part in getattr(rest, "geoms", [rest]):
            if part.is_empty or part.geom_type != "LineString" or part.length < min_length_m:
                continue
            if (
                part.length > max_length_m
            ):  # pezzo centrale: abbastanza lungo da vedersi, abbastanza corto da stare nel ritaglio
                start = (part.length - max_length_m) / 2
                part = substring(part, start, start + max_length_m)
            pieces.append(part)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pieces), size=min(n, len(pieces)), replace=False)) if pieces else []
    return gpd.GeoDataFrame(geometry=[pieces[i] for i in pick], crs=subject.crs)


def render_crop(segment: LineString, rasters: list[Path], out_png: Path, size_m: float = 128, scale: int = 4) -> Path:
    """Ritaglio di ombreggiatura centrato sul pezzo, ingrandito ``scale`` volte, con il pezzo disegnato in rosso."""
    c = segment.centroid
    half = size_m / 2
    for path in rasters:
        with rasterio.open(path) as ds:
            b = ds.bounds
            if not (b.left <= c.x <= b.right and b.bottom <= c.y <= b.top):
                continue
            res = abs(ds.transform.a)
            win = from_bounds(c.x - half, c.y - half, c.x + half, c.y + half, ds.transform)
            px = int(round(size_m / res))
            arr = ds.read(1, window=win, boundless=True, fill_value=0, out_shape=(px, px)).astype("float32")
        break
    else:
        raise ValueError(f"nessun raster contiene il punto {c.x:.1f}, {c.y:.1f}")
    arr = np.nan_to_num(arr, nan=0.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr).convert("RGB").resize((px * scale, px * scale), Image.Resampling.BILINEAR)
    k = scale / res
    pts = [((x - (c.x - half)) * k, ((c.y + half) - y) * k) for x, y in segment.coords]
    ImageDraw.Draw(img).line(pts, fill=(255, 0, 0), width=max(2, scale // 2))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def ask_visible(png: Path, kind: str, config: LLMConfig | None = None, transport=None, size_m: float = 128) -> dict:
    """Chiede a Inkling se la via è visibile; ``via_visibile`` è None se la risposta non è un JSON valido."""
    prompt = PROMPTS[kind].format(size=int(size_m)) + _ANSWER
    try:
        text = chat(
            [{"role": "user", "content": prompt}],
            images=[png],
            config=config,
            transport=transport,
            temperature=0,
            reasoning_effort="low",
        )
    except httpx.HTTPError as exc:
        return {"via_visibile": None, "errore": f"{type(exc).__name__}: {exc}"}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    visible = data.get("via_visibile")
    return {
        "via_visibile": visible if isinstance(visible, bool) else None,
        "confidenza": data.get("confidenza"),
        "motivo": data.get("motivo"),
        "risposta": text[:300],
    }


def summarize(answers: dict[str, list[dict]]) -> dict:
    out = {}
    for kind, rows in answers.items():
        known = [r["via_visibile"] for r in rows if r.get("via_visibile") is not None]
        out[kind] = {"n": len(rows), "risposte": len(known), "visibile": (sum(known) / len(known)) if known else None}
    return out


def verify_run(run_id: str, aoi, n: int = 40, seed: int = 0, buffer_m: float = 5.0, stage: str = "stage_b") -> Path:
    """Campiona ``n`` pezzi extra e ``n`` mancati nell'area LiDAR, li fa giudicare a Inkling, scrive il riassunto."""
    from morloquio.common.cache import params_hash
    from morloquio.common.config import pipeline_config
    from morloquio.common.grid import tiles_in_bbox
    from morloquio.common.paths import interim_dir, processed_dir, runs_dir
    from morloquio.eval.osm_recall import lidar_mask, reference_ways

    run = runs_dir() / run_id
    mask = lidar_mask(aoi)
    cand = gpd.read_file(run / stage / "candidates_merged.gpkg").to_crs(aoi.work_crs).clip(mask)
    highways = gpd.read_file(interim_dir("osm") / f"{aoi.name}.gpkg", layer="highways")
    ref = reference_ways(highways).to_crs(aoi.work_crs).clip(mask)
    terrain = processed_dir("terrain") / params_hash(pipeline_config("terrain"))
    rasters = [p for t in tiles_in_bbox(aoi.bbox) if (p := terrain / t / "hillshade.tif").exists()]
    out_dir = run / "eval" / f"{stage}_inkling"
    answers: dict[str, list[dict]] = {}
    for kind, (subject, other) in {"extra": (cand, ref), "missed": (ref, cand)}.items():
        rows = []
        for i, seg in enumerate(sample_segments(subject, other, buffer_m, n, seed).geometry):
            png = render_crop(seg, rasters, out_dir / f"{kind}_{i:03d}.png")
            ans = ask_visible(png, kind) | {"png": png.name, "x": seg.centroid.x, "y": seg.centroid.y}
            log.info("verifica Inkling", extra={"kind": kind, "i": i, "via_visibile": ans["via_visibile"]})
            rows.append(ans)
        answers[kind] = rows
    report = {"run_id": run_id, "stage": stage, "buffer_m": buffer_m, "summary": summarize(answers), "answers": answers}
    path = run / "eval" / f"{stage}_inkling.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
