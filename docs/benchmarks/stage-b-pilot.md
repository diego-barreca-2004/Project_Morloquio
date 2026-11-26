# Stadio B nel pilota di Morloquio — risultati

Data: 2026-11-26 · Area: `morloquio` (bbox del pilota, 117 celle, 92 tile con LiDAR T1) · GPU: RTX 5070 Ti Laptop
(12 GB) · Riferimento: OSM (`reference_ways`, 530 km dentro la maschera LiDAR di 78,5 km²).

## Come si misura

- **Completezza** = km di vie OSM entro il buffer dalle linee trovate / km di vie OSM. **Correttezza** = km di linee
  trovate entro il buffer da OSM / km di linee trovate. Solo dove il DTM LiDAR è valido (`lidar_mask`).
- **Buffer 5 m** è la misura del piano (soglia di uscita 0,98). Il buffer 10 m è riportato come sensibilità
  all'errore di posizione di OSM: nel pilota molte vie OSM stanno a 5–10 m dalla via visibile nel LiDAR.
- **Fuori fold**: ogni tile è predetta dal modello del suo fold, che non l'ha vista né in addestramento né in
  validazione (10 fold a blocchi 2×2 tile, griglia globale). Nessun numero qui viene da tile viste in addestramento.
- La correttezza contro OSM è un **limite inferiore**: OSM nel pilota è incompleto (sentieri e piste mancanti), quindi
  una parte delle linee "sbagliate" sono vie vere assenti da OSM. La verifica a campione con Inkling ne stima la quota.

## Risultati (vettoriali, fuori fold)

| Run | Metodo | Compl. 5 m | Corr. 5 m | Compl. 10 m | Corr. 10 m | km linee |
|---|---|---|---|---|---|---|
| pilot-003 | Stadio A geometrico | 0,186 | 0,166 | — | — | 494 |
| pilot-b-001 | U-Net 14 canali, soglia 0,5, asse mediano | 0,839 | 0,562 | — | — | 779 |
| pilot-b-001 | + ricucitura dalle foglie | 0,850 | 0,513 | 0,919 | 0,566 | 956 |
| pilot-b-001 | soglia 0,2, asse mediano | 0,833 | 0,439 | 0,934 | 0,498 | 1007 |
| pilot-b-001 | soglia 0,5, **asse dalla cresta** (default) | **0,864** | **0,563** | 0,918 | 0,607 | 808 |

Curva per pixel (maschera sopra soglia entro 5 m da OSM, `eval/stage_b_sweep.csv`): soglia 0,05 → 0,982 / 0,307;
0,1 → 0,973 / 0,369; 0,2 → 0,959 / 0,439; 0,5 → 0,920 / 0,560. La rete "vede" il 98% di OSM, ma solo con
moltissimi falsi positivi; la perdita vera sta tra la maschera e la linea.

Dove si perde (pilot-b-001, soglia 0,5, asse mediano; 85 km OSM mancati): 42,6 km con la rete sotto soglia; 44,2 km
con la rete sopra soglia entro 5 m ma linea a 5–10 m (35,6 km) o 10–20 m (6,2 km); 18 km dentro macchie larghe
≥ 8 m (vie vicine fuse). L'asse dalla cresta recupera una parte del secondo gruppo.

## Limite attuale

Lo 0,98 a 5 m **non** è raggiunto: 0,864 (correttezza 0,563 contro un OSM incompleto). In letteratura nessun lavoro
supera ~0,86 di recall per pixel su oggetti lineari stretti da LiDAR (`docs/research/2026-11-26-ml-strade-da-lidar.md`).

## In sospeso

- pilot-b-002: stesso modello con 16 canali (+ openness positiva e negativa 10 m), ablazione.
- soglia 0,3 con asse dalla cresta (`eval/stage_b_t03ridge.json` in pilot-b-001).
- Verifica a campione con Inkling delle linee extra e mancate (`morloquio.agents.verify.verify_run`).
- Correzioni della review finale (ledger): ricampionamento delle patch con worker persistenti, ripresa sicura di
  `b_train`, invalidazione dei modelli. Fino ad allora i numeri qui sopra valgono per un addestramento fatto da zero
  in un solo passaggio (così sono stati prodotti), ma con patch che non cambiano tra un'epoca e l'altra.
