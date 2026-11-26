# Morloquio F0–F2 (fondamenta, dati pilota, Stadio A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repo pubblica funzionante che, con un comando, scarica in automatico i dati aperti dell'area pilota Morloquio, costruisce il cubo dati per tile, calcola i derivati LiDAR, estrae i candidati di via (Stadio A), li misura contro OSM e li mostra in un visualizzatore locale; Inkling 2 Mini installato e misurato.

**Architecture:** Pacchetto Python `morloquio` diviso per responsabilità (`common`, `sources`, `ingest`, `terrain`, `candidates`, `eval`, `viewer`, `agents`). Ogni stadio lavora per tile 0,01°, è idempotente e scrive artefatti ispezionabili: dati condivisi in `data/` (grezzi, intermedi, elaborati), output di ogni esecuzione in `runs/<run_id>/`. Una CLI (`morloquio`) orchestra gli stadi.

**Tech Stack:** Python 3.12, uv, numpy, scipy, rasterio, pyproj, shapely, geopandas, pyogrio, scikit-image, skan, httpx, PyYAML, osmium (pyosmium), pystac-client, mercantile, Pillow, whitebox (WhiteboxTools Open Core), pytest, ruff; MapLibre GL JS per il visualizzatore; Ollama per Inkling 2 Mini.

**Spec:** `docs/superpowers/specs/2026-11-25-morloquio-design.md`

## Global Constraints

- Tutto open source al 100%: ogni dipendenza Python passa il license gate (Task 3); i pacchetti `nvidia-*` sono strato di sistema (eccezione esplicita). Licenze GPL rifiutate di default (incompatibili con il codice Apache-2.0), ammesse solo con override documentato.
- Dati: solo licenze in allowlist (CC0, CC BY, CC BY-SA, ODbL, PDDL, IODL 2.0, Etalab, OGL, pubblico dominio). Licenza ambigua = fonte rifiutata.
- Vietato in modo assoluto: Google Maps/Earth, Strava (heatmap inclusa), Esri/Bing/Mapbox imagery, Microsoft Road Detections, qualunque fonte a pagamento. Nessun codice, test, config o commento li usa o li scarica.
- Nessun aggiramento di firewall/anti-bot: solo client HTTP semplici, User-Agent onesto `Morloquio/<versione> (+https://github.com/diego-barreca-2004/Project_Morloquio)`, richieste sequenziali per host, backoff su 429/5xx.
- Repo pubblica `diego-barreca-2004/Project_Morloquio`; codice Apache-2.0; identificatori in inglese, commenti e documenti in italiano.
- Area pilota: bbox `[15.62, 38.03, 15.75, 38.12]` (west, south, east, north, EPSG:4326); nucleo `[15.6663, 38.0666, 15.6891, 38.0846]`; CRS di lavoro `EPSG:32633`.
- Tile: celle di 0,01°; `tile_id` = `N3807E01567` (cella 38.07–38.08 N, 15.67–15.68 E); tavola PST corrispondente `D38071567_0101`.
- Cartelle dati: `MORLOQUIO_DATA_DIR` (default `<repo>/data`) e `MORLOQUIO_RUNS_DIR` (default `<repo>/runs`), entrambe ignorate da git. Durante lo sviluppo in worktree puntano a `<repo-principale>/data` e `<repo-principale>/runs`, così i download sopravvivono al worktree.
- Dipendenze e versioni: Python `>=3.12,<3.13`; ogni dipendenza aggiunta con `uv add`, lock in `uv.lock`.
- Git: lavoro su branch `f0-f2-pilot`, mai push su `main`; PR draft verso `main`.
- Cambio rispetto allo spec (§7, scelta dell'assistente): niente Snakemake in F0–F2; orchestrazione con la CLI interna (stadi idempotenti per tile). DVC e MLflow arrivano in F3 con i modelli. Registrato in `docs/decisions/0001-orchestrazione-cli.md`.

## Review Focus

1. Cella della griglia senza LiDAR PST (mare, buchi di copertura: 117 celle nel bbox, 97 con LiDAR) → la pipeline prosegue, la tile riceve tier `T3`, nessun crash. Test in Task 11 e Task 16.
2. Sessione PST con `reason` = `METADATA_WARNING` → successo (osservato dal vivo); `reason` diverso da `OK`/`METADATA_WARNING` → `PstError` per quel batch, gli altri batch proseguono. Test in Task 6.
3. Download interrotto (file `.part` rimasto, sha256 diverso) → riscaricato, mai considerato completo. Test in Task 4.
4. Vie OSM su ponti, in galleria o interne (`bridge=*`, `tunnel=*`, `indoor=*`, `layer<0`, `highway=elevator|corridor|platform|proposed|construction`) → escluse dal riferimento del recall: il DTM non può mostrarle. Test in Task 15.
5. Via che attraversa il bordo tra due tile → dopo il merge è una linea continua (nessun buco > 2 m, nessun doppione nella fascia di alone). Test in Task 14.

---

## File Structure

```
Project_Morloquio/
  README.md  LICENSE  DATA_LICENSES.md  CONTRIBUTING.md  Makefile  pyproject.toml  uv.lock  .gitignore
  .github/workflows/ci.yml
  configs/
    aoi/morloquio.yaml
    sources/{pst_lidar,osm_geofabrik_sud,osm_gps_traces,sentinel2_c1_l2a,copernicus_dem_glo30,agea_ortho_2012}.yaml
    pipeline/{terrain,stage_a}.yaml
    llm.yaml
    license_overrides.yaml
  src/morloquio/
    __init__.py  cli.py
    common/{paths,aoi,grid,runlog,cache,http,manifest}.py
    sources/{licenses,catalog}.py
    sources/adapters/{pst_lidar,osm_extract,osm_gps,stac,agea_wms}.py
    ingest/cube.py
    terrain/derivatives.py
    candidates/{linearity,vectorize}.py
    eval/osm_recall.py
    viewer/{tiles,build}.py  viewer/static/index.html
    agents/llm.py
  scripts/{check_licenses.py,bench_llm.py}
  tests/  (specchio di src/, + tests/fixtures/, + tests/live/ marcati @pytest.mark.live)
  docs/ specs/ plans/ decisions/ benchmarks/ runbook.md
  data/  README.md tracciato; raw/ interim/ processed/ cache/ ignorati
  runs/  ignorata
```

---

### Task 1: Bootstrap repo pubblica e scheletro del pacchetto

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `README.md` (sovrascrive quello generato), `DATA_LICENSES.md`, `CONTRIBUTING.md`, `data/README.md`, `src/morloquio/__init__.py`, `docs/decisions/0001-orchestrazione-cli.md`
- Move: `docs/superpowers/specs/…design.md`, `docs/superpowers/plans/…pilot.md` nel repo
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: `morloquio.__version__: str` (`"0.1.0"`); target `make check` (ruff + pytest -m "not live" + license gate).

- [ ] **Step 1:** `gh repo create diego-barreca-2004/Project_Morloquio --public --description "Mappa aperta e automatica di tutte le strade e i sentieri del mondo, da soli dati open (LiDAR, ortofoto, OSM)." --add-readme --license apache-2.0` → `main` creato lato server.
- [ ] **Step 2:** In `<repo-principale>`: `git init`, remote `origin`, `git fetch origin`, allinea `main` a `origin/main`; poi EnterWorktree `f0-f2-pilot` e copia lì `docs/`.
- [ ] **Step 3: Test che fallisce**

```python
import morloquio
def test_version():
    assert morloquio.__version__ == "0.1.0"
```

- [ ] **Step 4:** `uv init --package --name morloquio` adattato a layout `src/`; `uv add --dev pytest ruff`; ruff `line-length = 110`, regole `E,F,I,UP,B`; pytest marker `live` registrato, `addopts = "-m 'not live'"`.
- [ ] **Step 5:** `.gitignore`: `data/*` tranne `data/README.md`, `runs/`, `.venv/`, `__pycache__/`, `*.part`, `.env`. `data/README.md` descrive `raw/<source_id>/` (download intatti + `manifest.json`), `interim/` (estratti e ritagli), `processed/` (cubo, derivati), `cache/`.
- [ ] **Step 6:** `uv run pytest -q` → PASS; `make check` → verde (il license gate arriva in Task 3: per ora il target lo chiama solo se lo script esiste).
- [ ] **Step 7:** Commit `chore: bootstrap repo, spec e piano` e push del branch `f0-f2-pilot`; apri PR draft verso `main`.

### Task 2: Base comune — percorsi, AOI, griglia, log, hash

**Files:**
- Create: `src/morloquio/common/{paths,aoi,grid,runlog,cache}.py`, `configs/aoi/morloquio.yaml`
- Test: `tests/common/test_{paths,aoi,grid,runlog,cache}.py`

**Interfaces:**
- Produces:
  - `repo_root() -> Path`; `data_dir() -> Path`; `runs_dir() -> Path`; `raw_dir(source_id: str) -> Path`; `interim_dir(name: str) -> Path`; `processed_dir(name: str) -> Path` (creano la cartella).
  - `@dataclass(frozen=True) AOI(name: str, bbox: tuple[float,float,float,float], core_bbox: tuple[...], work_crs: str)`; `load_aoi(name: str) -> AOI`.
  - `TILE_DEG = 0.01`; `tile_id_for(lat: float, lon: float) -> str`; `tile_bounds(tile_id: str) -> tuple[float,float,float,float]` (W,S,E,N); `tiles_in_bbox(bbox) -> list[str]` (ordinati); `neighbors(tile_id) -> list[str]` (8); `pst_tavola_for(tile_id) -> str`; `tile_id_from_tavola(tavola: str) -> str`.
  - `new_run_id(now: datetime | None = None) -> str` (formato `%Y%m%dT%H%M%SZ-<6 hex>`); `get_logger(name: str, run_id: str | None = None, **context) -> logging.LoggerAdapter` (JSON per riga su stderr e su `runs_dir()/<run_id>/log.jsonl`, campi `ts, level, logger, msg, run_id` + contesto).
  - `params_hash(params: Mapping) -> str` (sha1 del JSON canonico, 8 caratteri).

- [ ] **Step 1: Test che falliscono**

```python
def test_tile_id_pilot_point():
    assert tile_id_for(38.0756, 15.6777) == "N3807E01567"
def test_tile_id_exact_boundary_is_stable():
    assert tile_id_for(38.07, 15.67) == "N3807E01567"   # 38.07*100 = 3806.9999…
def test_tile_bounds():
    assert tile_bounds("N3807E01567") == pytest.approx((15.67, 38.07, 15.68, 38.08))
def test_pilot_bbox_has_117_cells():
    assert len(tiles_in_bbox((15.62, 38.03, 15.75, 38.12))) == 117
def test_pst_tavola_roundtrip():
    assert pst_tavola_for("N3807E01567") == "D38071567_0101"
    assert tile_id_from_tavola("D38071567_0101") == "N3807E01567"
def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MORLOQUIO_DATA_DIR", str(tmp_path)); assert raw_dir("x") == tmp_path / "raw" / "x"
def test_load_aoi():
    a = load_aoi("morloquio"); assert a.work_crs == "EPSG:32633" and a.bbox == (15.62, 38.03, 15.75, 38.12)
def test_params_hash_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1}) and len(params_hash({})) == 8
def test_log_line_is_json(tmp_path, monkeypatch): …  # una riga, json.loads ok, contiene run_id
```

- [ ] **Step 2:** `uv run pytest tests/common -q` → FAIL (moduli assenti).
- [ ] **Step 3:** Implementa. Indici di cella con `math.floor(round(x * 100, 6))`; formato `f"{'N' if lat>=0 else 'S'}{abs(i):04d}{'E' if lon>=0 else 'W'}{abs(j):05d}"`. `tiles_in_bbox` include le celle che intersecano l'interno del bbox (bordi est/nord esclusi). `uv add pyyaml`.
- [ ] **Step 4:** `uv run pytest tests/common -q` → PASS.
- [ ] **Step 5:** Commit `feat(common): paths, AOI, griglia tile, log JSON, params hash`.

### Task 3: License gate (dipendenze, dati) e CI

**Files:**
- Create: `src/morloquio/sources/licenses.py`, `scripts/check_licenses.py`, `configs/license_overrides.yaml`, `.github/workflows/ci.yml`
- Modify: `Makefile` (target `licenses`, incluso in `check`)
- Test: `tests/sources/test_licenses.py`

**Interfaces:**
- Produces: `class LicenseRejected(ValueError)`; `normalize_license(raw: str) -> str | None`; `check_data_license(spdx: str) -> None`; `check_code_license(raw: str) -> str`; `@dataclass Violation(package: str, raw_license: str, reason: str)`; `audit_environment(overrides_path: Path) -> list[Violation]`.

- [ ] **Step 1: Test che falliscono**

```python
@pytest.mark.parametrize("ok", ["CC-BY-4.0", "ODbL-1.0", "CC0-1.0", "IODL-2.0", "etalab-2.0"])
def test_data_allowed(ok): check_data_license(ok)
@pytest.mark.parametrize("bad", ["CC-BY-NC-4.0", "CC-BY-NC-ND-3.0", "unverified", "proprietary", ""])
def test_data_rejected(bad):
    with pytest.raises(LicenseRejected): check_data_license(bad)
def test_normalize_common_strings():
    assert normalize_license("MIT License") == "MIT"
    assert normalize_license("Apache Software License") == "Apache-2.0"
    assert normalize_license("BSD 3-Clause") == "BSD-3-Clause"
def test_spdx_expressions():
    assert check_code_license("MIT OR Apache-2.0") == "MIT OR Apache-2.0"
    with pytest.raises(LicenseRejected): check_code_license("MIT AND LicenseRef-Proprietary")
def test_gpl_rejected_without_override():
    with pytest.raises(LicenseRejected): check_code_license("GPL-3.0-only")
def test_audit_skips_system_layer(fake_dists):  # fixture con distribuzione finta "nvidia-cublas-cu12", licenza "Other/Proprietary"
    assert audit_environment(overrides) == []
```

- [ ] **Step 2:** `uv run pytest tests/sources/test_licenses.py -q` → FAIL.
- [ ] **Step 3:** Implementa. `audit_environment` legge `importlib.metadata.distributions()`: prima `License-Expression`, poi classifier `License :: OSI Approved :: …`, poi campo `License` (prima riga). Override in YAML: `system_layer_prefixes: ["nvidia-"]` e `packages: {nome: {license: SPDX, source: URL, note: …}}`. Codice ammesso: Apache-2.0, MIT, MIT-CMU, BSD-2-Clause, BSD-3-Clause, ISC, PSF-2.0, MPL-2.0, LGPL-2.1-or-later, LGPL-3.0-or-later, HPND, Unlicense, Zlib, 0BSD, CC0-1.0.
- [ ] **Step 4:** `scripts/check_licenses.py` stampa le violazioni ed esce con 1 se ce ne sono; `make licenses` → verde sull'ambiente attuale (eventuali pacchetti con metadati vuoti vanno in override con URL della licenza verificata a mano).
- [ ] **Step 5:** CI GitHub Actions su push/PR: `uv sync --frozen`, `make check`.
- [ ] **Step 6:** `uv run pytest -q` → PASS. Commit `feat(sources): license gate per dipendenze e dati, CI`.

### Task 4: Client HTTP cortese e manifest dei download

**Files:**
- Create: `src/morloquio/common/{http,manifest}.py`
- Test: `tests/common/test_http.py`, `tests/common/test_manifest.py`

**Interfaces:**
- Produces:
  - `USER_AGENT: str`; `class PoliteClient(min_interval_s: float = 1.0, max_attempts: int = 5, backoff_s: float = 2.0, timeout_s: float = 120.0, transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep)` con `request(method: str, url: str, **kw) -> httpx.Response` e `download(url: str, dest: Path, *, expected_sha256: str | None = None) -> FileRecord`.
  - `@dataclass FileRecord(path: Path, sha256: str, size: int, url: str, fetched_at: str)`.
  - `read_manifest(dir: Path) -> dict[str, dict]`; `update_manifest(dir: Path, rel: str, record: FileRecord, **extra) -> None` (scrittura atomica di `manifest.json`); `is_complete(dir: Path, rel: str) -> bool` (file presente e sha256 uguale al manifest).

- [ ] **Step 1: Test che falliscono** (con `httpx.MockTransport` e `sleep` finto)
  - `test_user_agent_sent` — l'header `User-Agent` della richiesta è `USER_AGENT`.
  - `test_retry_on_503_then_ok` — risposte 503, 503, 200 → ritorna 200, `sleep` chiamato 2 volte con backoff crescente.
  - `test_honors_retry_after` — 429 con `Retry-After: 7` → `sleep(7)`.
  - `test_gives_up_after_max_attempts` — sempre 500 → `httpx.HTTPStatusError`.
  - `test_download_atomic` — dopo il successo esiste `dest`, non esiste `dest.part`, sha256 corretto.
  - `test_partial_file_not_complete` — `dest.part` rimasto + manifest assente → `is_complete` False; file presente con sha diverso dal manifest → False.
  - `test_sha_mismatch_raises` — `expected_sha256` sbagliato → `ValueError`, `dest` non creato.
- [ ] **Step 2:** `uv run pytest tests/common/test_http.py tests/common/test_manifest.py -q` → FAIL.
- [ ] **Step 3:** Implementa (`uv add httpx`). Retry su `httpx.TransportError`, 429, 500–599; intervallo minimo tra richieste allo stesso host; download in streaming su `.part`, `fsync`, `os.replace`.
- [ ] **Step 4:** PASS. Commit `feat(common): client HTTP cortese con retry e download atomici`.

### Task 5: Catalogo fonti

**Files:**
- Create: `src/morloquio/sources/catalog.py`, `configs/sources/{pst_lidar,osm_geofabrik_sud,osm_gps_traces,sentinel2_c1_l2a,copernicus_dem_glo30,agea_ortho_2012}.yaml`
- Test: `tests/sources/test_catalog.py`

**Interfaces:**
- Consumes: `check_data_license` (Task 3).
- Produces: `@dataclass(frozen=True) SourceSpec(id: str, name: str, provider: str, license: str, license_url: str, attribution: str, resolution_m: float | None, protocol: str, adapter: str, params: dict, verified: str)`; `@dataclass Catalog(accepted: dict[str, SourceSpec], rejected: dict[str, str])`; `load_catalog(path: Path | None = None) -> Catalog`.

Valori dei YAML (verificati il 2026-11-25):
- `pst_lidar`: licenza `CC-BY-4.0`, `license_url: https://gn.mase.gov.it/portale/distribuzione-dati-pst`, attribuzione "Ministero dell'Ambiente e della Sicurezza Energetica – Piano Straordinario di Telerilevamento", 1.0 m, `protocol: arcgis-rest`, params `grid_url`, `api_url: https://sim.mase.gov.it/core/api/esri/api/v1/downloads`, `products: [dtm_1x1, dsm_first_1x1, dsm_last_1x1]`, `prodotto: LIDAR_2014`, `canale: PST2`, `batch_max: 20`.
- `osm_geofabrik_sud`: `ODbL-1.0`, URL `https://download.geofabrik.de/europe/italy/sud-latest.osm.pbf` (segue redirect), attribuzione "© OpenStreetMap contributors".
- `osm_gps_traces`: `ODbL-1.0`, `https://api.openstreetmap.org/api/0.6/trackpoints`.
- `sentinel2_c1_l2a`: licenza Copernicus "free, full and open" → `CC-BY-4.0` come equivalente dichiarato nel campo `note`; STAC `https://earth-search.aws.element84.com/v1`, collezione `sentinel-2-c1-l2a`, 10 m.
- `copernicus_dem_glo30`: stessa base STAC, collezione `cop-dem-glo-30`, 30 m; licenza `CC-BY-4.0` con nota sulla licenza Copernicus DEM (verificare il testo prima del merge; se non equivalente → `unverified`).
- `agea_ortho_2012`: `license: unverified` fino al Task 10.

- [ ] **Step 1: Test che falliscono**
  - `test_real_catalog_loads` — `load_catalog()` accetta `pst_lidar`, `osm_geofabrik_sud`, `osm_gps_traces`; `agea_ortho_2012` è in `rejected` con motivo che contiene `"unverified"`.
  - `test_nc_source_rejected(tmp_path)` — YAML con `CC-BY-NC-4.0` → in `rejected`.
  - `test_missing_field_raises(tmp_path)` — YAML senza `license` → `ValueError` con il nome del file.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(sources): catalogo fonti con license gate`.

### Task 6: Adapter LiDAR PST

**Files:**
- Create: `src/morloquio/sources/adapters/pst_lidar.py`, `tests/fixtures/pst/{grid_query.json,post_ok.json,status_ready_warning.json,tile.zip}`
- Test: `tests/sources/test_pst_lidar.py`, `tests/live/test_pst_live.py`

**Interfaces:**
- Consumes: `PoliteClient`, `update_manifest`, `is_complete`, `raw_dir`, `tile_id_from_tavola`, `SourceSpec.params`.
- Produces: `@dataclass(frozen=True) PstTile(fid: int, tavola: str, regione: str, contratto: str, date: str, tile_id: str)`; `class PstError(RuntimeError)`; `list_tiles(client, bbox, grid_url) -> list[PstTile]`; `request_session(client, api_url, product: str, tiles: Sequence[PstTile]) -> str`; `wait_ready(client, api_url, session_id: str, timeout_s: float = 900, poll_s: float = 5) -> list[str]`; `public_file_url(api_url: str, raw_url: str) -> str`; `fetch_product(client, spec: SourceSpec, product: str, tiles: Sequence[PstTile]) -> list[Path]` (file in `raw_dir("pst_lidar")/<product>/<tavola>.tif`).

Contratto API (verificato dal vivo, vedi spec §5.2): griglia `…/Grigliato_lidar/FeatureServer/2/query` con `geometry=<W,S,E,N>&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=fid,tavola,regione,contratto,data&returnGeometry=false&f=json`, paginazione `resultOffset`/`resultRecordCount=2000` finché `exceededTransferLimit`; POST JSON `{"canale","prodotto","layer","tileList":[{"fid": str, "tavola","regione","contratto"}],"metadataList":[]}` → `idSessione`; GET stato `{status, reason, urls, expected}`; file su `<api_url>/<id>/file/<idFile>`; POST `<api_url>/<id>/ack`.

- [ ] **Step 1: Test che falliscono** (MockTransport con le fixture)
  - `test_list_tiles_parses_and_pages` — due pagine → `PstTile(fid=311806, tavola="D38071567_0101", regione="Calabria", contratto="140", date="090313", tile_id="N3807E01567")` presente.
  - `test_batches_of_20` — 45 tile → 3 POST con 20, 20, 5 elementi.
  - `test_metadata_warning_is_success` — stato `{"status":"ready","reason":"METADATA_WARNING","urls":[…]}` → URL restituiti.
  - `test_bad_reason_raises_only_that_batch` — un batch con `reason:"KO"` → loggato `PstError`, gli altri batch scaricati.
  - `test_public_file_url` — `http://mase-api-gateway.apps.dxap-prod.ocp.mase.priv/api/v1/downloads/abc/file/9` → `https://sim.mase.gov.it/core/api/esri/api/v1/downloads/abc/file/9`.
  - `test_zip_member_mapped_by_tavola` — membro `calabria_140_D38071567_0101_DTM.tiff` → `dtm_1x1/D38071567_0101.tif`; membro `../evil.tif` o senza tavola riconoscibile → ignorato, nulla scritto fuori dalla cartella.
  - `test_skip_already_complete` — tile già nel manifest con sha valido → nessuna POST.
  - `test_ack_after_download` — POST su `/ack` dopo lo scaricamento.
  - `tests/live/test_pst_live.py::test_pilot_tile_dtm` (`@pytest.mark.live`) — scarica `D38071567_0101` DTM, raster 1000×1000 float32.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa. Lo zip è dato non fidato: si leggono solo i membri `*.tif|*.tiff`, il nome di destinazione è costruito dalla tavola, mai dal percorso del membro.
- [ ] **Step 4:** PASS; poi `uv run pytest -m live tests/live/test_pst_live.py -q` → PASS.
- [ ] **Step 5:** `uv add rasterio`; con rasterio leggi il tile scaricato e registra in `configs/sources/pst_lidar.yaml` → `params.crs`, `params.nodata`, `params.pixel_size` osservati (non indovinare: valori letti dal file).
- [ ] **Step 6:** Commit `feat(sources): adapter LiDAR PST via API pubblica`.

### Task 7: Adapter estratto OSM (Geofabrik)

**Files:**
- Create: `src/morloquio/sources/adapters/osm_extract.py`, `tests/fixtures/osm/mini.osm`
- Test: `tests/sources/test_osm_extract.py`

**Interfaces:**
- Consumes: `PoliteClient.download`, `raw_dir`, `interim_dir`.
- Produces: `fetch_pbf(client, spec) -> FileRecord` (verifica con il file `.md5` di Geofabrik); `extract_layers(pbf: Path, bbox, out_gpkg: Path) -> dict[str, int]` → layer GeoPackage in EPSG:4326: `highways` (colonne `osm_id, highway, surface, tracktype, access, bridge, tunnel, layer, indoor`), `waterways` (`osm_id, waterway`), `barriers` (`osm_id, barrier` ∈ wall, retaining_wall, fence, city_wall), `buildings` (`osm_id`, poligoni).

- [ ] **Step 1: Test che falliscono** — `mini.osm` con 1 `highway=path`, 1 `highway=track` con `bridge=yes`, 1 `waterway=stream`, 1 `barrier=retaining_wall`, 1 edificio, 1 via fuori bbox: `extract_layers` → `{"highways": 2, "waterways": 1, "barriers": 1, "buildings": 1}`; la colonna `bridge` della track vale `"yes"`.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa (`uv add osmium geopandas pyogrio shapely pyproj`) con `osmium.FileProcessor(...).with_locations()` e filtri per chiave; ritaglio con shapely sul bbox. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(sources): estratto OSM per AOI`.

### Task 8: Adapter tracce GPS pubbliche OSM

**Files:**
- Create: `src/morloquio/sources/adapters/osm_gps.py`
- Test: `tests/sources/test_osm_gps.py`

**Interfaces:**
- Produces: `split_bbox(bbox, max_area_deg2: float = 0.25) -> list[tuple]`; `fetch_trackpoints(client, spec, bbox) -> list[Path]` (`raw_dir("osm_gps_traces")/<bbox_key>/page_<n>.gpx`); `load_points(paths) -> geopandas.GeoDataFrame` (colonne `track`, `seq`, geometria punto EPSG:4326).

- [ ] **Step 1: Test che falliscono** — `test_pages_until_short_page` (pagine da 5000, 5000, 12 punti → 3 richieste `page=0,1,2`); `test_split_bbox_respects_limit` (bbox 1°×0,5° → ogni parte ≤ 0,25 deg²); `test_load_points_counts` (GPX fixture con 2 tracce, 5 punti → 5 righe, 2 valori di `track`).
- [ ] **Step 2:** FAIL. **Step 3:** Implementa. **Step 4:** PASS. **Step 5:** Commit `feat(sources): tracce GPS pubbliche OSM`.

### Task 9: Adapter STAC (Sentinel-2 C1 L2A, Copernicus DEM GLO-30)

**Files:**
- Create: `src/morloquio/sources/adapters/stac.py`
- Test: `tests/sources/test_stac.py`, `tests/live/test_stac_live.py`

**Interfaces:**
- Produces: `select_items(items: list[dict], max_cloud: float, max_items: int) -> list[dict]` (meno nuvole prima, poi più recenti); `search_items(spec, collection: str, bbox, datetime: str | None, max_cloud: float | None, max_items: int) -> list[dict]`; `fetch_window(href: str, bbox, dest: Path) -> Path` (lettura a finestra del COG nel suo CRS nativo, scrittura GeoTIFF deflate).
- Parametri pilota: S2 `datetime="2024-01-01/2025-12-31"`, `max_cloud=10`, `max_items=40`, asset `red, green, blue, nir, swir16, swir22, scl`; DEM asset `data`. File in `raw_dir("sentinel2_c1_l2a")/<item_id>/<asset>.tif` e `raw_dir("copernicus_dem_glo30")/<item_id>.tif`.

- [ ] **Step 1: Test che falliscono** — `test_select_items_orders_by_cloud_then_date`; `test_fetch_window_crops_bbox` (COG sintetico locale in EPSG:32633 → dimensioni attese della finestra); live: `test_search_pilot_s2` trova ≥ 1 item.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa (`uv add pystac-client`). **Step 4:** PASS (live separato).
- [ ] **Step 5:** Commit `feat(sources): Sentinel-2 e Copernicus DEM via STAC`.

### Task 10: Ortofoto AGEA 2012 — verifica licenza, poi adapter solo se aperta

**Files:**
- Create: `docs/decisions/0002-licenza-ortofoto-agea-2012.md`; se aperta: `src/morloquio/sources/adapters/agea_wms.py`, `tests/sources/test_agea_wms.py`
- Modify: `configs/sources/agea_ortho_2012.yaml`

**Interfaces:**
- Produces (solo se licenza aperta): `fetch_ortho_tile(client, spec, tile_id: str, res_m: float = 0.5) -> Path` → `raw_dir("agea_ortho_2012")/<tile_id>.tif` (RGB uint8, EPSG:32633), GetMap a blocchi da 1024×1024 px, layer `OI.ORTOIMMAGINI.2012.33`, formato `image/jpeg`.

- [ ] **Step 1:** Cerca la licenza esplicita del dataset/servizio nel record di metadati RNDT/INSPIRE (CSW del Geoportale nazionale, campi `useLimitation`/`otherConstraints`) e nella pagina del servizio. Registra in ADR 0002 citazioni testuali e URL.
- [ ] **Step 2 (esito aperto):** imposta la licenza SPDX nel YAML; test `test_getmap_blocks_cover_tile` (tile pilota a 0,5 m → mosaico con transform corretto, nessuna fessura tra blocchi) e `test_rejected_source_not_fetched`; implementa; PASS.
- [ ] **Step 2 (esito non aperto o ambiguo):** il YAML resta `unverified`, nessun adapter; il pilota prosegue con il solo LiDAR (tier T1-LiDAR) e lo dichiara nel runbook.
- [ ] **Step 3:** Commit `docs(sources): licenza ortofoto AGEA 2012` (+ `feat(sources): adapter WMS ortofoto` se aperta).

### Task 11: Cubo dati per tile (Stadio 0)

**Files:**
- Create: `src/morloquio/ingest/cube.py`
- Test: `tests/ingest/test_cube.py`

**Interfaces:**
- Consumes: `tile_bounds`, `AOI`, `processed_dir`, file grezzi dei Task 6 e 10.
- Produces: `HALO_M = 64`; `tile_grid(tile_id: str, work_crs: str, res_m: float, halo_m: float = HALO_M) -> tuple[Affine, int, int]` (griglia agganciata a multipli di `res_m`, include l'alone); `tier_for(has_lidar: bool, has_ortho: bool) -> str` (`"T1"` se LiDAR, `"T2"` se solo ortofoto, altrimenti `"T3"`); `build_cube(tile_id: str, aoi: AOI, raw_lidar: dict[str, list[Path]], raw_ortho: list[Path]) -> Path` → `processed_dir("cube")/<tile_id>/` con `dtm.tif`, `dsm_first.tif`, `dsm_last.tif` (1 m, float32, nodata NaN, deflate+predictor 3, tiled 256), `ortho.tif` (0,5 m, se c'è), `meta.json` `{tile_id, tier, valid_fraction: {layer: float}, sources: {layer: [file]}, halo_m, work_crs}`.

- [ ] **Step 1: Test che falliscono**
  - `test_tile_grid_snapped_and_haloed` — origine multipla di 1 m; larghezza ≈ (larghezza della tile in metri + 128) ± 1 px.
  - `test_reprojects_from_other_crs` — raster sintetico in EPSG:4326 e uno in EPSG:32633 con la stessa superficie inclinata → nel cubo differenza mediana < 0,05 m.
  - `test_neighbors_fill_halo` — due raster adiacenti → nessun NaN nell'alone sul lato condiviso.
  - `test_nodata_propagates` — valori nodata del sorgente (letti dal file) → NaN nel cubo, `valid_fraction` corretta.
  - `test_tile_without_lidar_is_t3` (Review Focus 1) — nessun file LiDAR → nessun `dtm.tif`, `meta.json` con `"tier": "T3"`, nessuna eccezione.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa con `rasterio.warp.reproject` (bilineare per i modelli altimetrici, cubica per l'ortofoto), combinazione "primo valido". **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(ingest): cubo dati per tile con alone`.

### Task 12: Derivati del terreno

**Files:**
- Create: `src/morloquio/terrain/derivatives.py`, `configs/pipeline/terrain.yaml`
- Test: `tests/terrain/test_derivatives.py`

**Interfaces:**
- Consumes: cubo del Task 11, `params_hash`.
- Produces (array 2D float32, NaN = nodata, `res` in metri): `slope_deg(dtm, res) -> ndarray`; `hillshade_multi(dtm, res, azimuths=(315, 270, 225, 360), altitude=45.0) -> ndarray[uint8]`; `local_relief(dtm, res, sigma_m: float = 10.0) -> ndarray`; `curvatures(dtm, res) -> tuple[ndarray, ndarray]` (profilo, planare); `canopy_height(dsm_first, dtm) -> ndarray` (≥ 0); `flow_accumulation(dtm, res) -> ndarray` (celle a monte, D8 dopo riempimento depressioni, via WhiteboxTools); `compute_tile(tile_id: str, params: dict) -> Path` → `processed_dir("terrain")/<params_hash>/<tile_id>/{slope,hillshade,lrm,curv_profile,curv_plan,chm,flowacc}.tif`.
- `terrain.yaml`: `lrm_sigma_m: 10.0`, `hillshade_azimuths: [315, 270, 225, 360]`, `hillshade_altitude: 45`.

- [ ] **Step 1: Test che falliscono**
  - `test_slope_of_30_degree_plane` → mediana 30 ± 0,5.
  - `test_lrm_plane_is_zero` → |LRM| < 0,01 lontano dai bordi.
  - `test_lrm_trench_negative` — trincea larga 2 m, profonda 0,5 m → LRM al centro < −0,3.
  - `test_nan_does_not_spread` — buco NaN 10×10 → NaN solo entro il raggio del kernel (3σ) dal buco.
  - `test_chm_non_negative`.
  - `test_flowacc_channel` — valle a V sintetica → accumulo massimo sull'asse della valle.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa: filtri NaN-aware con convoluzione normalizzata; `uv add scipy whitebox`; WhiteboxTools lavora su file temporanei. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(terrain): derivati LiDAR per tile`.

### Task 13: Stadio A — punteggi di linearità e fusione

**Files:**
- Create: `src/morloquio/candidates/linearity.py`, `configs/pipeline/stage_a.yaml`
- Test: `tests/candidates/test_linearity.py`

**Interfaces:**
- Consumes: cubo (Task 11), derivati (Task 12), punti GPS (Task 8).
- Produces: `hessian_linearity(img, res, widths_m=(1, 2, 3, 4, 6), polarity: Literal["valley", "ridge", "both"] = "both") -> ndarray` (0–1); `bench_score(dtm, res, half_widths_m=(1.0, 1.5, 2.5), offset_m: float = 3.0) -> ndarray`; `chm_gap_linearity(chm, res, min_canopy_m: float = 3.0) -> ndarray`; `ortho_linearity(rgb, res) -> ndarray`; `gps_density(points_xy: ndarray, transform, shape, sigma_m: float = 5.0) -> ndarray`; `robust_normalize(x) -> ndarray`; `fuse(layers: dict[str, ndarray], weights: dict[str, float]) -> ndarray`; `score_tile(tile_id: str, params: dict, run_id: str) -> Path` → `runs/<run_id>/stage_a/<tile_id>/{score,bench,lrm_lin,chm_gap,ortho_lin,gps}.tif`.

Algoritmo `bench_score` (gradino di versante: striscia piana con pendenza a monte e a valle), per ogni semilarghezza `w`:
```
d = direzione del gradiente del DTM lisciato (σ = 2 m), unitaria; s(p, t) = DTM campionato in p + t·d (map_coordinates, ordine 1)
center = |s(+w) − s(−w)| / (2w)
up     = |s(+w+o) − s(+w)| / o
down   = |s(−w) − s(−w−o)| / o
score_w = clip(min(up, down) − center, 0, ∞) · [center < 0.5 · min(up, down)]
bench = max_w score_w, poi robust_normalize
```
`robust_normalize`: `z = (x − mediana) / (1.4826·MAD)` sui pixel validi, poi `1 / (1 + exp(−(z − 2)))`. `fuse`: media pesata dei layer disponibili, pesi rinormalizzati dove un layer è NaN.
- `stage_a.yaml`: `weights: {bench: 1.0, lrm_lin: 1.0, chm_gap: 0.7, ortho_lin: 0.7, gps: 0.5}`, `threshold: 0.35`, `min_length_m: 10.0`.

- [ ] **Step 1: Test che falliscono**
  - `test_bench_detects_cut_path` — versante a 30° con sentiero piano largo 2 m tagliato lungo le isoipse → punteggio medio sul sentiero > 5 × quello del versante.
  - `test_bench_ignores_plain_slope_and_flat` — piano inclinato puro e piano orizzontale → punteggio ~0.
  - `test_hessian_finds_trench` — trincea di 2 m → massimo sull'asse.
  - `test_chm_gap_corridor` — bosco a 15 m con corridoio largo 4 m senza alberi → punteggio alto nel corridoio, ~0 in bosco pieno e su prato.
  - `test_fuse_renormalizes_missing_layers` — `ortho_lin` tutto NaN → risultato uguale alla fusione senza quel layer.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa (`uv add scikit-image`); `hessian_linearity` usa `skimage.filters.sato` con `sigmas = [w / (2·res) for w in widths_m]`. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(candidates): punteggi di linearità e fusione dello Stadio A`.

### Task 14: Vettorizzazione e cucitura tra tile

**Files:**
- Create: `src/morloquio/candidates/vectorize.py`
- Test: `tests/candidates/test_vectorize.py`

**Interfaces:**
- Consumes: `score.tif` (Task 13), `tile_bounds`.
- Produces: `extract_candidates(score: ndarray, transform, crs: str, threshold: float, min_length_m: float) -> geopandas.GeoDataFrame` (LineString, colonne `length_m, mean_score, max_score`); `crop_to_core(gdf, tile_id: str, work_crs: str) -> GeoDataFrame` (toglie la parte nell'alone); `merge_tiles(gdfs: list[GeoDataFrame], snap_m: float = 2.0) -> GeoDataFrame`; `vectorize_tile(tile_id, params, run_id) -> Path` → `runs/<run_id>/stage_a/<tile_id>/candidates.gpkg`; `merge_run(run_id) -> Path` → `runs/<run_id>/stage_a/candidates_merged.gpkg`.

- [ ] **Step 1: Test che falliscono**
  - `test_straight_line_one_segment` — striscia diagonale larga 3 px → una LineString di lunghezza ±5%.
  - `test_short_blobs_removed` — macchia di 4 m → nessuna linea con `min_length_m=10`.
  - `test_t_junction_three_segments` — incrocio a T → 3 segmenti che si toccano nel nodo.
  - `test_seam_continuity` (Review Focus 5) — una linea che attraversa il bordo tra due tile adiacenti elaborate separatamente → dopo `crop_to_core` + `merge_tiles` una sola linea, nessun buco > 2 m, lunghezza totale ±2% del vero.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa (`uv add skan`): soglia → chiusura morfologica raggio 1 px → `remove_small_objects` → `skeletonize` → `skan.Skeleton` → percorsi in coordinate mappa → shapely. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(candidates): vettorizzazione e cucitura tra tile`.

### Task 15: Valutazione contro OSM

**Files:**
- Create: `src/morloquio/eval/osm_recall.py`
- Test: `tests/eval/test_osm_recall.py`

**Interfaces:**
- Consumes: layer `highways` (Task 7), `candidates_merged.gpkg` (Task 14), maschera LiDAR valida (unione delle `valid_fraction` dai `meta.json`: poligoni delle tile T1).
- Produces: `EXCLUDED_HIGHWAY = {"elevator", "corridor", "platform", "proposed", "construction", "abandoned", "razed"}`; `reference_ways(highways: GeoDataFrame) -> GeoDataFrame` (esclude `EXCLUDED_HIGHWAY`, `bridge` non nullo e ≠ "no", `tunnel` non nullo e ≠ "no", `indoor=yes`, `layer < 0`); `completeness(reference, candidates, buffer_m: float = 5.0, mask=None) -> float`; `correctness(candidates, reference, buffer_m: float = 5.0, mask=None) -> float`; `report_by_class(reference, candidates, buffer_m, mask) -> pandas.DataFrame` (colonne `highway, ref_km, matched_km, completeness`); `evaluate_run(run_id: str, aoi: AOI) -> Path` → `runs/<run_id>/eval/stage_a.json` + `stage_a_by_class.csv`.
- Criterio di uscita F2: `completeness ≥ 0.98` sul riferimento dentro la maschera LiDAR del pilota.

- [ ] **Step 1: Test che falliscono**
  - `test_full_coverage_is_one`; `test_half_coverage_is_half`; `test_offset_beyond_buffer_is_zero` (parallela a 7 m con buffer 5 m → 0).
  - `test_reference_excludes_bridges_tunnels_indoor` (Review Focus 4) — vie con `bridge=yes`, `tunnel=culvert`, `indoor=yes`, `layer=-1`, `highway=proposed` → escluse; `bridge=no` → inclusa.
  - `test_mask_limits_reference` — riferimento metà dentro e metà fuori dalla maschera → conta solo la metà dentro.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa in `work_crs`. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(eval): completezza e correttezza contro OSM`.

### Task 16: CLI e orchestrazione degli stadi

**Files:**
- Create: `src/morloquio/cli.py`, `docs/decisions/0001-orchestrazione-cli.md` (completa la bozza del Task 1)
- Modify: `pyproject.toml` (`[project.scripts] morloquio = "morloquio.cli:main"`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: tutti gli stadi.
- Produces: `main(argv: list[str] | None = None) -> int`; comandi `morloquio fetch --aoi NAME [--sources id,…]`, `morloquio run STAGE --aoi NAME [--run-id ID] [--tiles id,…] [--force] [--workers N]` con `STAGE ∈ {cube, terrain, stage_a, merge, eval, viewer, all}`, `morloquio view --run-id ID [--port 8765]`, `morloquio sources` (stampa accettate/rifiutate con motivo). Ogni stadio salta le tile con output già presenti, salvo `--force`; scrive `runs/<run_id>/run.json` (`aoi, stages, params_hash per stadio, git_commit, started_at, tiles: {tile_id: {stage: "ok"|"skipped"|"error: …"}}`).

- [ ] **Step 1: Test che falliscono**
  - `test_run_records_per_tile_status` — stadi finti registrati, 3 tile di cui una che solleva un'eccezione → `run.json` con 2 `"ok"` e 1 `"error: …"`, exit code 1, le altre tile completate.
  - `test_idempotent_skip` — seconda esecuzione → tutte `"skipped"`.
  - `test_missing_lidar_tile_continues` (Review Focus 1) — tile T3 → `stage_a` la salta con `"skipped: tier T3"`, exit code 0.
  - `test_sources_command_lists_rejected` — output contiene `agea_ortho_2012` e il motivo, se ancora rifiutata.
- [ ] **Step 2:** FAIL. **Step 3:** Implementa con `argparse` e `concurrent.futures.ProcessPoolExecutor`. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(cli): orchestrazione idempotente per tile`.

### Task 17: Visualizzatore locale

**Files:**
- Create: `src/morloquio/viewer/{tiles,build}.py`, `src/morloquio/viewer/static/index.html`
- Test: `tests/viewer/test_tiles.py`, `tests/viewer/test_build.py`

**Interfaces:**
- Consumes: mosaici di hillshade/LRM/score del run, `candidates_merged.gpkg`, layer OSM, ortofoto se presente.
- Produces: `render_xyz(raster: Path, out_dir: Path, zooms: range = range(12, 19), mode: Literal["gray", "heat", "rgb"] = "gray") -> int` (PNG 256 px in EPSG:3857, trasparenza per i nodata; ritorna il numero di tile); `build_viewer(run_id: str, aoi: AOI) -> Path` → `runs/<run_id>/viewer/` con `index.html`, `layers.json`, `xyz/<layer>/{z}/{x}/{y}.png`, `vector/{candidates,osm_highways}.geojson`.
- `index.html`: MapLibre GL JS `4.7.1` da unpkg, centro `[15.6777, 38.0756]` zoom 15; layer attivabili: hillshade, LRM, punteggio Stadio A, candidati (colore per `mean_score`), OSM attuale (vie), ortofoto (se c'è), sfondo OSM standard con attribuzione "© OpenStreetMap contributors"; pannello con conteggi e km dal file `eval/stage_a.json`. Nessun riferimento a Strava o ad altri servizi vietati.

- [ ] **Step 1: Test che falliscono** — `test_render_xyz_counts_and_alpha` (raster sintetico piccolo → numero di tile atteso per z12–14, pixel nodata con alpha 0); `test_build_viewer_writes_layers_json` (run finto → `layers.json` elenca solo i layer esistenti; i GeoJSON sono in EPSG:4326).
- [ ] **Step 2:** FAIL. **Step 3:** Implementa (`uv add mercantile pillow`); `morloquio view` serve la cartella su `127.0.0.1` con `http.server`. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(viewer): visualizzatore locale del run`.

### Task 18: Inkling 2 Mini — installazione, client, benchmark

**Files:**
- Create: `src/morloquio/agents/llm.py`, `configs/llm.yaml`, `scripts/bench_llm.py`, `docs/benchmarks/inkling-2-mini.md`
- Test: `tests/agents/test_llm.py`

**Interfaces:**
- Produces: `@dataclass LLMConfig(base_url: str, model: str, timeout_s: float)`; `load_llm_config() -> LLMConfig`; `chat(messages: list[dict], images: list[Path] | None = None, config: LLMConfig | None = None, transport=None) -> str` (API compatibile OpenAI `/v1/chat/completions`, immagini come `image_url` base64).
- `llm.yaml`: `base_url: http://127.0.0.1:11434/v1`, `model: inkling-2-mini:26b-a5b`, `timeout_s: 600`.

- [ ] **Step 1:** Installa Ollama (MIT) in spazio utente (`~/.local/ollama`, tarball ufficiale Linux amd64, nessun `sudo`), avvia `ollama serve`, `ollama pull inkling-2-mini:26b-a5b` (~18–20 GB; in background mentre si lavora agli altri task). Verifica la licenza del binario e del modello (Apache-2.0) in `docs/benchmarks/inkling-2-mini.md`.
- [ ] **Step 2: Test che falliscono** — `test_chat_sends_openai_payload` (MockTransport: corpo con `model` e `messages`, ritorna il testo); `test_chat_encodes_image` (PNG → `data:image/png;base64,…`).
- [ ] **Step 3:** FAIL → implementa → PASS.
- [ ] **Step 4:** `scripts/bench_llm.py` misura su 3 prompt fissi (testo breve, testo lungo, immagine di hillshade della tile pilota): token/s di generazione, tempo al primo token, VRAM e RAM usate (`nvidia-smi`, `/proc/meminfo`), correttezza della risposta sull'immagine. Prova `26b-a5b` e `26b-a5b-iq4_NL`; tieni la quantizzazione più precisa con ≥ 5 token/s e input immagini funzionante. Se nessuna regge: annota i numeri e chiedi al proprietario (Global Constraints: niente cambio modello senza chiedere).
- [ ] **Step 5:** Commit `feat(agents): client LLM locale e benchmark Inkling 2 Mini`.

### Task 19: Esecuzione completa del pilota, runbook, PR

**Files:**
- Create: `docs/runbook.md`
- Modify: `README.md` (stato del progetto, comandi), `DATA_LICENSES.md` (attribuzioni effettive delle fonti usate)

- [ ] **Step 1:** `MORLOQUIO_DATA_DIR=<repo-principale>/data MORLOQUIO_RUNS_DIR=<repo-principale>/runs uv run morloquio fetch --aoi morloquio` → nel log: tile PST elencate (attese 97), file DTM/DSM scaricati, OSM estratto, tracce GPS, item S2/DEM.
- [ ] **Step 2:** `uv run morloquio run all --aoi morloquio --workers 8` → `run.json` senza errori; tile senza LiDAR marcate T3.
- [ ] **Step 3:** Leggi `runs/<run_id>/eval/stage_a.json`: se `completeness < 0.98`, apri `stage_a_by_class.csv` e il visualizzatore sulle vie mancate, correggi pesi/soglie in `stage_a.yaml` (o il rilevatore), riesegui `run stage_a … --force` e ripeti finché il criterio è soddisfatto. Registra i valori finali e il motivo di ogni modifica nel runbook.
- [ ] **Step 4:** `uv run morloquio view --run-id <id>` → verifica che si apra e mostri tutti i layer; schermata salvata in `docs/benchmarks/stage-a-pilot.png`.
- [ ] **Step 5:** `make check` verde; runbook con: variabili d'ambiente, comandi, dove stanno i dati, come aprire una tile a ogni stadio, risultati del pilota.
- [ ] **Step 6:** Commit `docs: runbook e risultati Stadio A pilota`; push; aggiorna la descrizione della PR draft con i risultati.

---

## Addendum — Stadio B anticipato (decisione del proprietario, 2026-11-25 23:58)

Il proprietario ha chiesto di passare al ML perché lo Stadio A geometrico non può arrivare a 0,98 (pilot-002: 0,185).
Motivazione, scelte e fonti in `docs/decisions/0004-stadio-b-ml.md` e `docs/research/2026-11-26-ml-strade-da-lidar.md`.
Codice in `src/morloquio/stage_b/`; parametri in `configs/pipeline/stage_b.yaml`.

**Vincoli aggiuntivi:** nessun canale di ingresso derivato da OSM (la metrica sarebbe finta); valutazione solo
out-of-fold (ogni tile predetta da un modello che non l'ha vista); loss solo sul nucleo della tile (l'alone può
essere il nucleo di una tile di test); Inkling scaricato dalla VRAM (`keep_alive: 0`) prima di addestrare.

**Review Focus aggiuntivo:** tile con DTM parziale (bordo del rilievo) → pesi 0 dove manca il DTM, nessun NaN nel
tensore; tile senza vie OSM → solo esempi non etichettati, nessuna divisione per zero; fold senza tile di validazione
→ errore chiaro; modello assente per una tile da predire → `skipped`, non crash; GPU assente → CPU con avviso.

### Task 20: Canali per pixel dello Stadio B

**Files:** Create `src/morloquio/stage_b/{__init__,features}.py`, `configs/pipeline/stage_b.yaml`, `docs/decisions/0004-stadio-b-ml.md`; Test `tests/stage_b/test_features.py`

**Interfaces:**
- Consumes: cubo (Task 11), derivati (Task 12), funzioni dello Stadio A (Task 13), punti GPS (Task 8).
- Produces: `CHANNELS: tuple[str, ...]`; `raw_channels(tile_id, params, gps_xy=None) -> dict[str, ndarray]`;
  `fit_normalization(tile_ids, params, gps_xy=None, sample_px=200_000) -> dict[str, tuple[float, float]]` (centro, scala robusti);
  `build_features(tile_id, params, norm, gps_xy=None) -> Path` → `processed/stage_b/<hash>/<tile>/features.npy`
  (float16, C×H×W, tanh((x−c)/s), NaN→0, ultimo canale `valid`) + `meta.json` (canali, transform, crs, norm).

- [ ] Test: `test_channels_shape_and_no_nan` (cubo sintetico con buco NaN → array C×H×W senza NaN, `valid`=0 nel buco);
  `test_normalization_is_global_not_per_tile` (stessa quota/pendenza in due tile → stesso valore normalizzato);
  `test_missing_gps_gives_zero_channel`. FAIL → implementa → PASS → commit `feat(stage_b): canali per pixel`.

### Task 21: Etichette OSM positive-unlabeled e fold spaziali

**Files:** Create `src/morloquio/stage_b/{labels,folds}.py`; Test `tests/stage_b/test_labels.py`, `tests/stage_b/test_folds.py`

**Interfaces:**
- Produces: `rasterize_labels(ways: GeoDataFrame, transform, shape, core: Polygon, valid: ndarray, params) -> tuple[ndarray, ndarray]`
  (target uint8 0/1 entro la semilarghezza della classe; peso float32: 1 sui positivi, 0 nella fascia di incertezza,
  `unlabeled_weight` altrove, 0 fuori nucleo e dove manca il DTM);
  `assign_folds(tile_ids, k, block_tiles, km_by_tile) -> dict[str, int]` (blocchi `block_tiles`×`block_tiles` di tile
  contigue, assegnati ai fold bilanciando i km OSM, deterministico).
- [ ] Test: `test_label_width_by_class`, `test_ignore_band_has_zero_weight`, `test_outside_core_zero_weight`,
  `test_blocks_never_split_across_folds`, `test_folds_balanced_and_deterministic`. FAIL → implementa → PASS → commit.

### Task 22: Rete e loss

**Files:** Create `src/morloquio/stage_b/{model,losses}.py`; Test `tests/stage_b/test_model.py`, `tests/stage_b/test_losses.py`

**Interfaces:**
- Produces: `UNet(in_ch: int, base: int = 32, depth: int = 4, attention: bool = True)` (blocchi residui, attention gate
  sulle skip, uscita 1 logit); `soft_skeleton(p, iters)`, `masked_focal_bce(logits, target, weight, gamma)`,
  `masked_soft_dice(prob, target, weight)`, `soft_cldice(prob, target, weight, iters)`, `total_loss(logits, target, weight, params) -> (Tensor, dict)`.
- [ ] Test: `test_unet_shape_any_multiple_of_16`, `test_zero_weight_pixels_do_not_affect_loss`,
  `test_cldice_penalizes_broken_line_more_than_thick_line`, `test_overfits_single_patch` (50 passi → loss ↓ 80%). FAIL → implementa → PASS → commit.

### Task 23: Addestramento k-fold e predizione

**Files:** Create `src/morloquio/stage_b/{dataset,train,predict}.py`; Test `tests/stage_b/test_train.py`, `tests/stage_b/test_predict.py`

**Interfaces:**
- Produces: `PatchDataset(tiles, patch, positive_fraction, augment)`; `train_fold(fold, train_tiles, val_tiles, params, out_dir) -> Path`
  (`model.pt` + `history.json`, AMP bf16, AdamW, early stopping sulla completezza di validazione);
  `predict_tile(tile_id, model_path, params, out_dir) -> Path` (finestre sovrapposte con fusione gaussiana, TTA diedrale 8 →
  `runs/<run>/stage_b/<tile>/score.tif` float32 0–1); `unload_llm()` (Ollama `keep_alive: 0`).
- [ ] Test: `test_train_fold_tiny_runs_on_cpu` (2 tile sintetiche 128 px, 3 epoche, file scritti),
  `test_predict_matches_training_grid` (stessa transform del cubo, valori in [0,1], NaN fuori DTM),
  `test_tta_is_equivariant` (rete identità → stessa uscita). FAIL → implementa → PASS → commit.

### Task 24: Integrazione CLI, vettorizzazione e valutazione dello Stadio B

**Files:** Modify `src/morloquio/cli.py`, `src/morloquio/candidates/vectorize.py`, `src/morloquio/eval/osm_recall.py`, `src/morloquio/viewer/build.py`; Test `tests/test_cli.py`, `tests/eval/test_osm_recall.py`

**Interfaces:**
- `vectorize_tile(..., stage="stage_a")`, `merge_run(run_id, stage="stage_a")`, `evaluate_run(run_id, aoi, buffer_m, stage="stage_a")` → `eval/<stage>.json`.
- Stadi CLI: `b_features`, `b_train` (globale), `b_predict`, `b_merge`, `b_eval`; `run b` li esegue in ordine e poi `viewer`.
- [ ] Test: `test_stage_b_stages_registered`, `test_eval_writes_per_stage_report`, `test_b_predict_skips_without_model`. FAIL → implementa → PASS → commit.

### Task 25: Pilota con lo Stadio B, iterazione verso 0,98, runbook, PR

- [ ] `run b --run-id pilot-b-001`; leggi `eval/stage_b.json` (completezza e correttezza out-of-fold, per classe).
- [ ] Itera (canali, etichette, loss, soglia, post-elaborazione del grafo) finché completezza ≥ 0,98 o il limite è
  documentato con numeri; ogni modifica e il suo motivo nel runbook.
- [ ] Chiudi il Task 19 (runbook, README, DATA_LICENSES, schermata, `make check`, PR).
