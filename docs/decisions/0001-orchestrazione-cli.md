# 0001 — Orchestrazione con la CLI interna invece di Snakemake (F0–F2)

Data: 2026-11-25 · Stato: accettata

## Contesto

Lo spec (§7) indicava Snakemake per la pipeline e DVC/MLflow per dati e modelli, come scelta dell'assistente.
In F0–F2 non ci sono ancora modelli addestrati, e la lista delle tile si conosce solo dopo aver interrogato la
griglia LiDAR (servirebbero i checkpoint di Snakemake).

## Decisione

In F0–F2 gli stadi sono comandi della CLI `morloquio`, idempotenti per tile: uno stadio salta le tile con output
già presenti (salvo `--force`) e scrive lo stato di ogni tile in `runs/<run_id>/run.json`. DVC e MLflow si
introducono in F3, quando arrivano i modelli.

## Come funziona (`src/morloquio/cli.py`)

- Stadi, in ordine fisso per `run all`: `cube` → `terrain` → `stage_a` → `merge` → `eval` → `viewer`.
  `cube`, `terrain` e `stage_a` lavorano per tile; gli altri sull'intero run.
- Una tile è "fatta" se il suo output esiste: `processed/cube/<tile>/meta.json` con le stesse tavole PST di
  adesso (nuove tavole = cubo da rifare), `processed/terrain/<params_hash>/<tile>/` con tutti i layer,
  `runs/<run_id>/stage_a/<tile>/candidates.gpkg`. Gli stadi di run si rifanno se un input è più recente
  dell'output. `--force` rifà tutto.
- `terrain` e `stage_a` saltano le tile senza LiDAR (`skipped: tier T3`) e quelle senza cubo
  (`skipped: nessun cubo`): non sono errori.
- Un'eccezione su una tile diventa `error: <tipo>: <messaggio>` in `run.json`, le altre tile proseguono e il
  comando esce con codice 1. Il traceback completo è in `runs/<run_id>/log.jsonl`.
- `run.json` si riscrive (in modo atomico) dopo ogni stadio, quindi un crash lascia lo stato fino a quel punto.
- `--workers N` usa `ProcessPoolExecutor` per gli stadi per tile.

## Conseguenze

- Meno dipendenze da far passare al controllo licenze, debug più diretto (un comando, una tile).
- Niente grafo delle dipendenze automatico: l'ordine degli stadi è fisso nella CLI.
- Da rivalutare in F6 (scala Italia), quando le tile diventano centinaia di migliaia.
