# Project Morloquio

Mappa aperta e **completamente automatica** di tutte le vie percorribili del mondo: strade, sterrate, carrarecce,
mulattiere e sentieri, anche sotto il bosco. Si parte da soli dati aperti (LiDAR, ortofoto, OpenStreetMap,
Sentinel) e da strumenti open source al 100%.

- Area pilota: Morloquio, vicino a Reggio Calabria (38.0756 N, 15.6777 E).
- Strategia: Italia prima (molto LiDAR aperto), poi il resto del mondo con modelli addestrati sull'Italia.
- Design: [`docs/superpowers/specs/2026-11-25-morloquio-design.md`](docs/superpowers/specs/2026-11-25-morloquio-design.md)
- Piano in corso: [`docs/superpowers/plans/2026-11-25-morloquio-f0-f2-pilot.md`](docs/superpowers/plans/2026-11-25-morloquio-f0-f2-pilot.md)

## Regole del progetto

- Solo software open source e dati con licenza aperta (CC0, CC BY, ODbL, …). Le dipendenze passano un controllo
  automatico delle licenze (`make licenses`).
- Nessun uso di Google Maps/Earth, Strava, immagini Esri/Bing/Mapbox o fonti a pagamento.

## Avvio rapido

```bash
uv sync
make check
morloquio fetch --aoi morloquio
morloquio run all --aoi morloquio --run-id pilot --workers 8   # Stadio A (geometrico) e visualizzatore
morloquio run b   --aoi morloquio --run-id pilot-b --workers 8 # Stadio B (rete neurale su GPU)
morloquio view --run-id pilot-b
```

Comandi, stadi, file prodotti e risultati del pilota: [`docs/runbook.md`](docs/runbook.md).

## Come funziona

1. **Dati**: LiDAR del Piano Straordinario di Telerilevamento (DTM e DSM 1 m, CC BY 4.0), OpenStreetMap (ODbL),
   tracce GPS pubbliche di OpenStreetMap, Sentinel-2. Tutto scaricato in automatico.
2. **Stadio A**: segnali geometrici dal terreno (gradini sul versante, solchi lineari, varchi nella chioma, tracce
   GPS) combinati in un punteggio per pixel.
3. **Stadio B**: una rete U-Net con attenzione impara dal LiDAR dove passano le vie, usando OSM come etichette
   incomplete (apprendimento positive-unlabeled). Ogni tile è predetta da un modello che non l'ha mai vista
   (validazione spaziale a blocchi, 10 fold). Decisione: [`docs/decisions/0004-stadio-b-ml.md`](docs/decisions/0004-stadio-b-ml.md).
4. **Vettori**: soglia, asse delle vie dalla cresta della probabilità, grafo, cucitura tra tile.
5. **Valutazione**: completezza e correttezza contro OSM con buffer 5 m, solo dove c'è LiDAR.

## Struttura

```
configs/      aree, fonti dati, parametri della pipeline
src/morloquio/ codice (common, sources, ingest, terrain, candidates, stage_b, eval, viewer, agents)
tests/        test (quelli con rete sono marcati `live` ed esclusi di default)
docs/         design, piani, decisioni, benchmark, runbook
data/         dati scaricati ed elaborati (ignorati da git, vedi data/README.md)
runs/         output di ogni esecuzione (ignorati da git)
```

## Licenze

Codice: Apache-2.0 (vedi `LICENSE`). Dati prodotti: ODbL 1.0. Fonti e attribuzioni: `DATA_LICENSES.md`.
