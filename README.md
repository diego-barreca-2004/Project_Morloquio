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
```

## Struttura

```
configs/      aree, fonti dati, parametri della pipeline
src/morloquio/ codice (common, sources, ingest, terrain, candidates, eval, viewer, agents)
tests/        test (quelli con rete sono marcati `live` ed esclusi di default)
docs/         design, piani, decisioni, benchmark, runbook
data/         dati scaricati ed elaborati (ignorati da git, vedi data/README.md)
runs/         output di ogni esecuzione (ignorati da git)
```

## Licenze

Codice: Apache-2.0 (vedi `LICENSE`). Dati prodotti: ODbL 1.0. Fonti e attribuzioni: `DATA_LICENSES.md`.
