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

## Conseguenze

- Meno dipendenze da far passare al controllo licenze, debug più diretto (un comando, una tile).
- Niente grafo delle dipendenze automatico: l'ordine degli stadi è fisso nella CLI.
- Da rivalutare in F6 (scala Italia), quando le tile diventano centinaia di migliaia.
