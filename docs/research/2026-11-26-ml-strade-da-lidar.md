# Ricerca: ML per strade e sentieri da LiDAR con etichette OSM incomplete

Data: 2026-11-26 · Base dell'ADR 0004 (Stadio B anticipato). Pagine lette con Crawl4AI, PDF scaricati ed estratti con
pypdf. Le note complete, una sezione per articolo (dati, metodo, iperparametri, numeri esatti, limiti, cosa adottiamo),
sono in `papers/`:

| File | Argomento | Articoli |
|---|---|---|
| [`papers/lidar-strade-sentieri.md`](papers/lidar-strade-sentieri.md) | strade, sentieri, fossi, solchi da LiDAR | 9 |
| [`papers/archeologia-rilievo.md`](papers/archeologia-rilievo.md) | hollow ways, CarcassonNet, auto-supervisione su DTM, LRM | 7 |
| [`papers/etichette-incomplete.md`](papers/etichette-incomplete.md) | etichette OSM mancanti o spostate, PU learning | 8 |
| [`papers/topologia-validazione.md`](papers/topologia-validazione.md) | clDice, grafo, validazione spaziale, metriche | 11 |

## Risultato principale

Nessun lavoro letto arriva a una recall per pixel di 0,98 su oggetti lineari stretti, tanto meno sotto bosco: il
massimo con precisione accettabile è 0,86 (fossi, Lidberg 2023). Le recall vicine o sopra 0,96 compaiono solo per
segmenti con tolleranza (Abdi 2022: 0,959 su segmenti di ~30 m sotto chioma, da DSM) o con precisione bassissima
(Meyer-Heß 2022: 98,45% con precisione 1,06%; PLS: 0,896 con precisione 0,192). Quindi:

- l'obiettivo 0,98 si misura come **completezza vettoriale con buffer** contro OSM (Wiedemann 1998), su tile mai viste;
- va sempre riportata insieme alla **correttezza**, altrimenti "strade ovunque" lo soddisfa (contro OSM incompleto la
  correttezza è un limite inferiore);
- per arrivarci serve probabilmente fondere rete, tracce GPS e ricucitura del grafo, non solo la rete.

## Numeri chiave per articolo

| Lavoro | Cosa | Risultato | Valutazione |
|---|---|---|---|
| McDermid 2025 (Remote Sensing 17(9):1539) | sentieri boreali, U-Net su DTM 50 cm | P 77 ± 9%, R 78 ± 14% | aree di test separate |
| Ontario 2026 (ISPRS XLIX-B1) | strade forestali, Attention ResU-Net, 11 canali | centerline entro 3 m: 66,7% → 87,2% | trasferimento Québec → Ontario |
| Lidberg 2023 | fossi, HPMF del DEM | R px 0,86, P 0,71, MCC 0,78 | 11 tile di test |
| Abdi 2022 | corridoi di esbosco sotto chioma | R 0,959 (DSM), 0,908 (CHM), 0,157 (DEMmin) | 390 segmenti rilevati sul campo |
| Bhatnagar 2022 | solchi di ruote, ResNet50+UNet | F1 0,67 (0,45–0,83), R 66% | un sito lasciato fuori alla volta |
| DDCM-Net 2019 | strade da LiDAR, Norvegia | F1 0,696 (Road2: 0,282) | immagine separata |
| Trier 2022 | strade forestali da Sentinel-2 | R px 41% → 57% con fascia ignorata | tile di test |
| CarcassonNet 2021 / 2022 | hollow roads | R 0,651, MCC 0,466; trasferito R 0,14–0,42 | nessun hold-out spaziale |
| Kazimi 2021 | hollow ways, HRNet + SSL | IoU 26,34; SSL +1,22 mIoU | regioni scansionate |
| Henry 2020 | Soft-Bootstrapped Dice, β 0,7 | metà etichette mancanti: IoU 57,11 (BCE 0,00) | rumore simulato |
| PLS 2026 | loss in finestre attorno ai positivi OSM | IoU 0,469 → 0,573, R 0,539 → 0,746 | OSM reale (18,4% mancante) |
| Wu 2019 | etichette a 3 stati da centerline OSM | F1 0,8839 vs 0,8927 supervisione completa | OSM reale |
| clDice 2021 | loss sullo scheletro | clDice 86,83 → 88,17 strade; costo < +10% | dataset pubblici |
| DeepRoadMapper 2017 | ricucitura dalle foglie con A* | precisione 71,2 → 84,5, recall 86,2 → 83,4 | — |
| Kattenborn 2022 | validazione casuale vs a blocchi | F1 gonfiata del 14,6–27,9% | — |
| Stock 2025 | progetto della CV spaziale | conta la dimensione del blocco; default 10 fold | — |

## Ricetta per Morloquio (corretta dopo la lettura integrale)

1. **Canali** (1 m, normalizzazione globale, nessun canale da OSM): pendenza, hillshade, LRM σ 10 m (nel range RVT),
   curvature, CHM e DSM first − DSM last (sotto chioma il segnale sta nella chioma: Abdi), DSM last − DTM, area drenata
   (falsi positivi tipici: torrenti e impluvi), segnali dello Stadio A, tracce GPS. Da aggiungere in ablation:
   openness ± a 10 m (rvt_py, Apache-2.0), filtro passa-alto a mediana (HPMF, Lidberg), Sentinel-2 come ausiliario.
2. **Etichette**: corridoio stretto per classe attorno alla centerline OSM, fascia di incertezza a peso 0, resto non
   etichettato a peso 0,1 (Lidberg usa 0,1 sullo sfondo); peso 0 fuori dal nucleo della tile. In ablation: finestre
   PLS (128 px, K 16), nnPU, DICE_SB β 0,7.
3. **Rete e loss**: Attention Residual U-Net; focal + (1 − α)·Dice + α·clDice con α 0,3–0,5 e k 10.
4. **Validazione**: 10 fold a blocchi 2 × 2 tile (~1,8 × 2,2 km), test = 1 fold, validazione = 1 fold, addestramento
   = 8; dimensione del blocco da verificare con il correlogramma dei canali (Stock).
5. **Soglia**: scelta sulle tile di validazione per il vincolo di completezza, non fissata a 0,5 (Virro: 0,1 molto più
   completo in bosco).
6. **Grafo**: scheletro → skan → ricucitura dalle foglie verso nodi entro 50 m quando il cammino nel grafo è > 100 m,
   A* sul costo 1 − p (DeepRoadMapper), tarata per accettare molto. Nessuna apertura morfologica aggressiva (Bhatnagar).
7. **Auto-addestramento**: predizioni ad alta confidenza fuori OSM verificate con Inkling in locale → nuove etichette
   (CarcassonNet: +5,5% di etichette, MCC +0,2; McDermid: due cicli).
8. **Metriche**: completezza e correttezza con buffer 5 m (più curve a 2 e 10 m), interruzioni per km, APLS per blocco.

## Cosa non è stato letto

- Salberg, Trier, Kampffmeyer 2017 (SCIA): il DNS del repository UiT non risolve (anche da resolver pubblici), la
  copia NVA risponde Forbidden; Springer a pagamento.
- Hesse 2010: nessuna copia aperta; algoritmo e parametri ricostruiti da RVT e rvt_py.
- TOPO (Biagioni & Eriksson 2012): nessuna copia aperta; definizione ricostruita dai lavori che la usano.
- Capitolo CHNT 25 di CarcassonNet: bloccato da un proof-of-work anti-bot, letto come capitolo 6 della tesi di Leiden.
- Materiali supplementari di Henry, Hu, RNGDet++; versioni editoriali a pagamento di P2CNet, ScRoadExtractor, Bhatnagar
  (letti i preprint).
- Nessun aggiramento di paywall o anti-bot; Firecrawl non usato.
