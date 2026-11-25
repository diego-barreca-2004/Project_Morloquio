# ADR 0004 — Stadio B anticipato: segmentazione con rete neurale addestrata su OSM

Stato: accettato · Data: 2026-11-26 · Decisione del proprietario (2026-11-25 23:58): "se non è raggiungibile perché non
vai di ML?"

## Contesto

Lo Stadio A (punteggi geometrici dal LiDAR, senza apprendimento) arriva a completezza 0,185 e correttezza 0,166 contro
530 km di vie OSM nel pilota (pilot-002). L'obiettivo del piano è 0,98. Nessun rilevatore geometrico letto si avvicina;
i lavori con reti profonde arrivano a 0,78–0,96 a seconda dell'oggetto e della tolleranza
(`docs/research/2026-11-26-ml-strade-da-lidar.md`).

## Decisione

Lo Stadio B, previsto dopo il pilota, si anticipa:

- **Rete**: Attention Residual U-Net (PyTorch, BSD-3) su 14 canali per pixel a 1 m: derivati del DTM, chioma, segnali
  dello Stadio A, tracce GPS. Normalizzazione globale (mediana/MAD su 32 tile), mai per patch.
- **Etichette**: vie OSM come positivi in un corridoio per classe; fascia di incertezza a peso 0; il resto non
  etichettato a peso 0,1 (positive-unlabeled: OSM è incompleto, quindi "niente OSM" non vuol dire "niente strada").
- **Nessun canale di ingresso derivato da OSM**: la valutazione contro OSM sarebbe finta.
- **Validazione**: 10 fold a blocchi di 2 × 2 tile; ogni tile è predetta dal modello del suo fold, che non l'ha mai
  vista; la loss è calcolata solo nel nucleo della tile (l'alone è il nucleo di una tile vicina, forse di test).
- **Loss**: focal + Dice + clDice (α 0,5, k 10).
- **GPU**: torch 2.14.1 cu130 sulla RTX 5070 Ti. I pacchetti `nvidia-*` e `cuda-toolkit` sono strato di sistema
  (spec §3.6). Inkling viene scaricato dalla VRAM (`keep_alive: 0`) prima di addestrare.
- **Misura**: completezza **e** correttezza vettoriali con buffer 5 m sulle predizioni out-of-fold
  (`runs/<run>/eval/stage_b.json`), per classe.

## Alternative scartate

- Continuare a tarare lo Stadio A: tetto basso e già misurato.
- Usare OSM come canale di ingresso (come P2CNet o il proximity layer dell'Ontario): alza la metrica senza trovare vie nuove.
- Reti che producono direttamente il grafo (RNGDet++, Sat2Graph): recall 72–77% e falliscono sotto gli alberi; la
  ricucitura dalle foglie si fa dopo, sul grafo dello scheletro.
- Validazione su tile casuali: gonfia la F1 del 15–28% (Kattenborn 2022).

## Conseguenze

- Tempo GPU: 10 modelli per run; `b_train` e `b_predict` girano in sequenza sulla GPU.
- Lo 0,98 resta un obiettivo oltre lo stato dell'arte pubblicato; se non viene raggiunto, il limite si documenta con i
  numeri out-of-fold e l'analisi per classe.
- Parametri in `configs/pipeline/stage_b.yaml`, codice in `src/morloquio/stage_b/`, stadi CLI `run b`.
