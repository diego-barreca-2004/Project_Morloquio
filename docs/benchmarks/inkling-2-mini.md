# Inkling 2 Mini in locale — installazione e benchmark

Data: 2026-11-25 · Macchina: RTX 5070 Ti Laptop (12 GB VRAM), WSL2 con 25 GB RAM, 24 thread.

## Installazione (spazio utente, nessun `sudo`)

- Ollama 0.40.18 (licenza MIT), tarball ufficiale Linux amd64 estratto in `~/.local/ollama`.
- Server: `OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama serve` (solo localhost).
- Modello: `ollama pull inkling-2-mini:26b-a5b` — 26,1 B parametri, quantizzazione IQ4_NL, proiettore visivo
  CLIP 318 M, contesto 1 M token. **Licenza del modello: Apache-2.0** (dal blob `license` del manifest).
- Varianti: il registro pubblica solo `26b-a5b` e `26b-a5b-iq4_NL`, con **gli stessi digest** (stesso file).
  Nessuna quantizzazione più precisa (q8_0, bf16, q6_K, q5_K_M, q4_K_M: 404). Si usa `26b-a5b`.

## Risultati (`scripts/bench_llm.py`, immagine = hillshade 512 × 512 m della tile N3807E01567)

Ragionamento predefinito (`high`):

| prova | token/s | primo token (s) | totale (s) | token generati | VRAM |
|---|---|---|---|---|---|
| testo breve (include il caricamento) | 9,7 | 102,9 | 133,9 | 307 | 10,3 / 12,2 GB |
| testo lungo | 12,4 | 2,8 | 150,4 | 1834 | 10,3 / 12,2 GB |
| immagine hillshade | 12,4 | 18,5 | 108,6 | 1122 | 10,4 / 12,2 GB |

Ragionamento `low`:

| prova | token/s | primo token (s) | totale (s) | token generati |
|---|---|---|---|---|
| testo breve | 12,6 | 3,3 | 17,4 | 182 |
| testo lungo | 12,4 | 2,4 | 92,5 | 1116 |
| immagine hillshade | 12,7 | 4,8 | 48,2 | 552 |

Modello caricato: 9,8 GiB, di cui 9,1 GiB in VRAM. RAM di sistema usata durante la prova: ~5 GiB.

## Valutazione

- Criterio del piano (≥ 5 token/s e input immagini funzionante): **superato** (~12 token/s).
- Risposta sull'immagine: riconosce la morfologia (versanti ripidi, terrazzamenti) e colloca le linee sinuose
  dei tornanti nella fascia centro-destra, dove sono davvero; italiano con qualche parola sbagliata
  ("disectée", "along"). Non è affidabile come rilevatore: serve come assistente di verifica e triage, non per
  disegnare le strade.
- Per lavori in serie conviene `think: low`: stessa velocità, risposte 2–3 volte più corte.
