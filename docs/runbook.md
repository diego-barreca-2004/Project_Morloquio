# Runbook

Come scaricare i dati, eseguire la pipeline e leggere i risultati. Tutti i comandi partono dalla radice della repo.

## Ambiente

```bash
uv sync                      # dipendenze (torch 2.14.1 con CUDA 13.0 dall'indice ufficiale PyTorch)
make check                   # ruff, test (senza rete), controllo licenze
```

| Variabile | Default | Contenuto |
|---|---|---|
| `MORLOQUIO_DATA_DIR` | `<repo>/data` | dati scaricati (`raw/`), intermedi (`interim/`), elaborati (`processed/`) |
| `MORLOQUIO_RUNS_DIR` | `<repo>/runs` | un sottocartella per run: `run.json`, output degli stadi, valutazioni, visualizzatore |

Entrambe sono ignorate da git. In un worktree conviene puntarle fuori dal worktree, così i download restano.

GPU: lo Stadio B usa la GPU NVIDIA se c'è (altrimenti CPU, molto più lento). Prima di addestrare, `b_train` scarica
Inkling dalla VRAM (`keep_alive: 0`): non serve fermare Ollama.

## Comandi

```bash
morloquio sources                                    # fonti accettate e rifiutate dal license gate
morloquio fetch --aoi morloquio                      # download automatico: LiDAR PST, OSM, tracce GPS, Sentinel-2
morloquio run all --aoi morloquio --run-id <id> --workers 8   # cubo, derivati, Stadio A, valutazione, visualizzatore
morloquio run b   --aoi morloquio --run-id <id> --workers 8   # Stadio B: canali, 10 modelli, predizione, valutazione
morloquio view --run-id <id>                         # visualizzatore su http://127.0.0.1:8765
```

Ogni stadio è idempotente: rilanciare salta ciò che è già fatto e aggiornato. `--force` rifà tutto; un singolo stadio
si lancia per nome (`morloquio run b_vectorize ...`). Lo stato di ogni tile e di ogni stadio è in `runs/<id>/run.json`.

| Stadio | Per tile | Scrive |
|---|---|---|
| `cube` | sì | `data/processed/cube/<tile>/{dtm,dsm_first,dsm_last}.tif`, `meta.json` (tier T1/T2/T3, `version`) |
| `terrain` | sì | `data/processed/terrain/<hash>/<tile>/*.tif` |
| `stage_a` | sì | `runs/<id>/stage_a/<tile>/{score,bench,lrm_lin,chm_gap,gps}.tif`, `candidates.gpkg` |
| `merge`, `eval` | no | `stage_a/candidates_merged.gpkg`, `eval/stage_a.json`, `eval/stage_a_by_class.csv` |
| `b_norm` | no | `stage_b/norm.json` (centro e scala dei canali) |
| `b_features` | sì | `data/processed/stage_b/<hash>/<tile>/{features,target,weight}.npy` |
| `b_train` | no | `stage_b/folds.json`, `stage_b/models/fold_<k>/{model.pt,history.json}` |
| `b_predict` | no | `stage_b/<tile>/score.tif` (probabilità dal modello che non ha visto la tile) |
| `b_vectorize` | sì | `stage_b/<tile>/candidates.gpkg` (soglia + ricucitura dei buchi) |
| `b_merge`, `b_eval` | no | `stage_b/candidates_merged.gpkg`, `eval/stage_b.json`, `eval/stage_b_by_class.csv` |
| `viewer` | no | `viewer/` (MapLibre, tile XYZ, GeoJSON) |

Soglia: `python -c "from morloquio.common.aoi import load_aoi; from morloquio.eval.sweep import sweep_run;
print(sweep_run('<id>', load_aoi('morloquio')))"` scrive `eval/stage_b_sweep.csv` (completezza e correttezza per
soglia, fuori fold).

## Come aprire una tile a ogni stadio

Ogni artefatto è un GeoTIFF o GeoPackage in EPSG:32633: si apre in QGIS. Per la tile `N3807E01567` (Morloquio):
`data/processed/cube/N3807E01567/dtm.tif`, `data/processed/terrain/<hash>/N3807E01567/hillshade.tif`,
`runs/<id>/stage_b/N3807E01567/score.tif`, `runs/<id>/stage_b/N3807E01567/candidates.gpkg`.

## Risultati del pilota

(in aggiornamento durante il Task 25)

| Run | Stadio | Completezza | Correttezza | Note |
|---|---|---|---|---|
| pilot-001 | A | 0,183 | 0,159 | LiDAR non restaurato, 33 cubi falliti (segnaposto PST da 0 byte) |
| pilot-002 | A | 0,185 | 0,166 | restauro attivo, ma 11 cubi ancora vecchi |
| pilot-003 | A | 0,186 | 0,166 | tutti i cubi restaurati: il limite è il metodo, non i dati |
