"""CLI ``morloquio``: download delle fonti e stadi della pipeline, idempotenti per tile (vedi ADR 0001).

    morloquio sources
    morloquio fetch --aoi morloquio [--sources pst_lidar,osm_gps_traces]
    morloquio run all --aoi morloquio [--run-id ID] [--tiles N3807E01567,...] [--force] [--workers 4]
    morloquio view --run-id ID [--port 8765]

Ogni stadio per tile salta le tile con output già presente (salvo ``--force``) e scrive lo stato di ogni tile in
``runs/<run_id>/run.json``; un errore su una tile non ferma le altre (exit code 1 alla fine).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

from morloquio.common.aoi import AOI, load_aoi
from morloquio.common.cache import params_hash
from morloquio.common.grid import tiles_in_bbox
from morloquio.common.paths import interim_dir, processed_dir, raw_dir, repo_root, runs_dir
from morloquio.common.runlog import get_logger, new_run_id
from morloquio.sources.catalog import SourceSpec, load_catalog

# ---------------------------------------------------------------------------------------------- contesto e stadi


@dataclass
class RunContext:
    aoi: AOI
    run_id: str
    force: bool = False
    params: dict[str, dict] = field(default_factory=dict)
    gps_xy: np.ndarray | None = None

    @property
    def run_dir(self) -> Path:
        return runs_dir() / self.run_id


@dataclass(frozen=True)
class Stage:
    name: str
    per_tile: bool
    run: Callable[[RunContext, str | None], object]
    done: Callable[[RunContext, str | None], bool]
    requires_t1: bool = False


def _pipeline_config(name: str) -> dict:
    return yaml.safe_load((repo_root() / "configs" / "pipeline" / f"{name}.yaml").read_text(encoding="utf-8"))


def _tier(tile_id: str) -> str | None:
    meta = processed_dir("cube") / tile_id / "meta.json"
    return json.loads(meta.read_text())["tier"] if meta.exists() else None


def _fresh(out: Path, inputs: list[Path]) -> bool:
    """L'output esiste ed è più recente di tutti gli input."""
    if not out.exists():
        return False
    stamps = [p.stat().st_mtime for p in inputs if p.exists()]
    return not stamps or out.stat().st_mtime >= max(stamps)


def _cube_run(ctx, tile):
    from morloquio.ingest.cube import build_cube, pst_sources_for

    build_cube(tile, ctx.aoi, pst_sources_for(tile), [])


def _cube_done(ctx, tile):
    from morloquio.ingest.cube import LIDAR_LAYERS, pst_sources_for

    meta = processed_dir("cube") / tile / "meta.json"
    if not meta.exists():
        return False
    recorded = json.loads(meta.read_text()).get("sources", {})
    current = pst_sources_for(tile)
    # nuove tavole scaricate dopo il cubo = cubo da rifare
    return all(
        sorted(recorded.get(layer, [])) == sorted(str(p) for p in current[product])
        for product, layer in LIDAR_LAYERS.items()
    )


def _terrain_run(ctx, tile):
    from morloquio.terrain.derivatives import compute_tile

    compute_tile(tile, ctx.params["terrain"])


def _terrain_done(ctx, tile):
    from morloquio.terrain.derivatives import LAYERS

    d = processed_dir("terrain") / params_hash(ctx.params["terrain"]) / tile
    return all((d / f"{name}.tif").exists() for name in LAYERS)


def _stage_a_run(ctx, tile):
    from morloquio.candidates.linearity import score_tile
    from morloquio.candidates.vectorize import vectorize_tile

    score_tile(tile, ctx.params["stage_a"], ctx.run_id, gps_xy=ctx.gps_xy)
    vectorize_tile(tile, ctx.params["stage_a"], ctx.run_id)


def _stage_a_done(ctx, tile):
    return (ctx.run_dir / "stage_a" / tile / "candidates.gpkg").exists()


def _merge_run(ctx, _):
    from morloquio.candidates.vectorize import merge_run

    merge_run(ctx.run_id)


def _merge_done(ctx, _):
    root = ctx.run_dir / "stage_a"
    return _fresh(root / "candidates_merged.gpkg", sorted(root.glob("*/candidates.gpkg")))


def _eval_run(ctx, _):
    from morloquio.eval.osm_recall import evaluate_run

    evaluate_run(ctx.run_id, ctx.aoi, float(ctx.params["eval"]["buffer_m"]))


def _eval_done(ctx, _):
    inputs = [ctx.run_dir / "stage_a" / "candidates_merged.gpkg", interim_dir("osm") / f"{ctx.aoi.name}.gpkg"]
    return _fresh(ctx.run_dir / "eval" / "stage_a.json", inputs)


def _viewer_run(ctx, _):
    from morloquio.viewer.build import build_viewer

    build_viewer(ctx.run_id, ctx.aoi)


def _viewer_done(ctx, _):
    inputs = [ctx.run_dir / "stage_a" / "candidates_merged.gpkg", ctx.run_dir / "eval" / "stage_a.json"]
    return _fresh(ctx.run_dir / "viewer" / "layers.json", inputs)


STAGES: dict[str, Stage] = {
    "cube": Stage("cube", per_tile=True, run=_cube_run, done=_cube_done),
    "terrain": Stage("terrain", per_tile=True, run=_terrain_run, done=_terrain_done, requires_t1=True),
    "stage_a": Stage("stage_a", per_tile=True, run=_stage_a_run, done=_stage_a_done, requires_t1=True),
    "merge": Stage("merge", per_tile=False, run=_merge_run, done=_merge_done),
    "eval": Stage("eval", per_tile=False, run=_eval_run, done=_eval_done),
    "viewer": Stage("viewer", per_tile=False, run=_viewer_run, done=_viewer_done),
}
ALL_ORDER = ["cube", "terrain", "stage_a", "merge", "eval", "viewer"]


def stage_params() -> dict[str, dict]:
    from morloquio.ingest.cube import HALO_M

    terrain = _pipeline_config("terrain")
    return {
        "cube": {"halo_m": HALO_M},
        "terrain": terrain,
        "stage_a": _pipeline_config("stage_a") | {"terrain": terrain},
        "merge": {"snap_m": 2.0},
        "eval": {"buffer_m": 5.0},
        "viewer": {},
    }


def _run_tile(stage_name: str, ctx: RunContext, tile: str | None) -> str:
    """Esegue uno stadio su una tile (o sull'intero run se ``tile`` è None) e ne restituisce lo stato."""
    stage = STAGES[stage_name]
    log = get_logger("cli", run_id=ctx.run_id, stage=stage_name, tile_id=tile)
    try:
        if stage.requires_t1:
            tier = _tier(tile)
            if tier is None:
                return "skipped: nessun cubo"
            if tier != "T1":
                return f"skipped: tier {tier}"
        if not ctx.force and stage.done(ctx, tile):
            return "skipped"
        stage.run(ctx, tile)
        return "ok"
    except Exception as exc:  # una tile rotta non ferma le altre: lo stato finisce in run.json
        log.exception("stadio fallito")
        return f"error: {type(exc).__name__}: {exc}"


def _load_gps(aoi: AOI) -> np.ndarray:
    from morloquio.sources.adapters.osm_gps import load_points

    pages = sorted(raw_dir("osm_gps_traces").glob("**/page_*.gpx"))
    if not pages:
        return np.empty((0, 2))
    pts = load_points(pages)
    w, s, e, n = aoi.bbox
    pts = pts.cx[w - 0.01 : e + 0.01, s - 0.01 : n + 0.01].to_crs(aoi.work_crs)
    return np.column_stack([pts.geometry.x, pts.geometry.y]) if len(pts) else np.empty((0, 2))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def run_stages(stages: list[str], ctx: RunContext, tiles: list[str], workers: int = 1) -> int:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.run_dir / "run.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(
        aoi=ctx.aoi.name,
        run_id=ctx.run_id,
        stages=stages,
        params_hash={s: params_hash(ctx.params.get(s, {})) for s in stages},
        git_commit=_git_commit(),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    record.setdefault("tiles", {})
    record.setdefault("run_stages", {})
    failed = False
    for name in stages:
        if STAGES[name].per_tile:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    statuses = list(pool.map(_run_tile, [name] * len(tiles), [ctx] * len(tiles), tiles))
            else:
                statuses = [_run_tile(name, ctx, t) for t in tiles]
            for tile, status in zip(tiles, statuses, strict=True):
                record["tiles"].setdefault(tile, {})[name] = status
                failed |= status.startswith("error")
        else:
            status = _run_tile(name, ctx, None)
            record["run_stages"][name] = status
            failed |= status.startswith("error")
        _write_json_atomic(path, record)  # dopo ogni stadio: un crash lascia lo stato fin lì
    record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json_atomic(path, record)
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------------- download


def _fetch_pst(client, spec: SourceSpec, aoi: AOI):
    from morloquio.sources.adapters import pst_lidar

    tiles = pst_lidar.list_tiles(client, aoi.bbox, spec.params["grid_url"])
    for product in spec.params["products"]:
        pst_lidar.fetch_product(client, spec, product, tiles)


def _fetch_osm_extract(client, spec: SourceSpec, aoi: AOI):
    from morloquio.sources.adapters import osm_extract

    rec = osm_extract.fetch_pbf(client, spec)
    osm_extract.extract_layers(rec.path, aoi.bbox, interim_dir("osm") / f"{aoi.name}.gpkg")


def _fetch_osm_gps(client, spec: SourceSpec, aoi: AOI):
    from morloquio.sources.adapters import osm_gps

    osm_gps.fetch_trackpoints(client, spec, aoi.bbox)


def _fetch_stac(client, spec: SourceSpec, aoi: AOI):
    from morloquio.sources.adapters import stac

    stac.fetch_collection(spec, aoi.bbox)


FETCHERS: dict[str, Callable] = {
    "pst_lidar": _fetch_pst,
    "osm_extract": _fetch_osm_extract,
    "osm_gps": _fetch_osm_gps,
    "stac": _fetch_stac,
}


def cmd_fetch(aoi: AOI, source_ids: list[str] | None) -> int:
    from morloquio.common.http import PoliteClient

    catalog = load_catalog()
    known = set(catalog.accepted) | set(catalog.rejected)
    wanted = source_ids or sorted(catalog.accepted)
    unknown = [s for s in wanted if s not in known]
    if unknown:
        print(f"fonti sconosciute: {', '.join(unknown)}")
        return 2
    log = get_logger("cli.fetch")
    code = 0
    with PoliteClient() as client:
        for sid in wanted:
            if sid in catalog.rejected:
                print(f"{sid}: rifiutata, non scaricata ({catalog.rejected[sid]})")
                code = 1
                continue
            spec = catalog.accepted[sid]
            fetcher = FETCHERS.get(spec.adapter)
            if fetcher is None:
                print(f"{sid}: nessun adapter '{spec.adapter}'")
                code = 1
                continue
            try:
                fetcher(client, spec, aoi)
                print(f"{sid}: ok")
            except Exception as exc:
                log.exception("download fallito", extra={"source": sid})
                print(f"{sid}: errore {type(exc).__name__}: {exc}")
                code = 1
    return code


def cmd_sources() -> int:
    catalog = load_catalog()
    print("Fonti accettate:")
    for sid, spec in sorted(catalog.accepted.items()):
        print(f"  {sid:24} {spec.license:16} {spec.name}")
    print("Fonti rifiutate:")
    for sid, reason in sorted(catalog.rejected.items()):
        print(f"  {sid:24} {reason}")
    return 0


def make_server(run_id: str, port: int) -> ThreadingHTTPServer:
    """Server HTTP del visualizzatore, solo su 127.0.0.1."""
    directory = runs_dir() / run_id / "viewer"
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def cmd_view(run_id: str, port: int) -> int:
    if not (runs_dir() / run_id / "viewer" / "index.html").exists():
        print(f"visualizzatore non trovato: esegui prima `morloquio run viewer --run-id {run_id}`")
        return 1
    server = make_server(run_id, port)
    print(f"visualizzatore su http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C per uscire)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# ---------------------------------------------------------------------------------------------------------- main


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morloquio", description="Pipeline Morloquio")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="elenca le fonti accettate e rifiutate")
    fetch = sub.add_parser("fetch", help="scarica le fonti per un'AOI")
    fetch.add_argument("--aoi", required=True)
    fetch.add_argument("--sources", type=_csv)
    run = sub.add_parser("run", help="esegue uno stadio (o tutti)")
    run.add_argument("stage", choices=[*STAGES, "all"])
    run.add_argument("--aoi", required=True)
    run.add_argument("--run-id")
    run.add_argument("--tiles", type=_csv)
    run.add_argument("--force", action="store_true")
    run.add_argument("--workers", type=int, default=1)
    view = sub.add_parser("view", help="serve il visualizzatore di un run su 127.0.0.1")
    view.add_argument("--run-id", required=True)
    view.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sources":
        return cmd_sources()
    if args.command == "fetch":
        return cmd_fetch(load_aoi(args.aoi), args.sources)
    if args.command == "view":
        return cmd_view(args.run_id, args.port)
    aoi = load_aoi(args.aoi)
    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    ctx = RunContext(aoi=aoi, run_id=args.run_id or new_run_id(), force=args.force)
    known = stage_params()
    ctx.params = {s: known.get(s, {}) for s in {*stages, "terrain", "stage_a", "eval"}}
    if "stage_a" in stages:
        ctx.gps_xy = _load_gps(aoi)
    tiles = args.tiles or tiles_in_bbox(aoi.bbox)
    code = run_stages(stages, ctx, tiles, workers=max(1, args.workers))
    print(f"run {ctx.run_id}: {'errori, vedi run.json' if code else 'ok'} ({ctx.run_dir / 'run.json'})")
    return code


if __name__ == "__main__":
    sys.exit(main())
