# Ricerca: ML per strade e sentieri da LiDAR con etichette OSM incomplete

Data: 2026-11-26 · Base per l'ADR 0004 (Stadio B anticipato). Fonti lette con Crawl4AI salvo dove indicato.

## Cosa fa lo stato dell'arte

| Tema | Tecnica | Fonte |
|---|---|---|
| Rete | U-Net su raster DTM a 50 cm–1 m, patch 512 px; Attention Residual U-Net con 11 canali LiDAR | [1], [2] |
| Canali | hillshade multidirezionale, pendenza, rugosità, bordi Sobel, DSM, CHM, punti 0,5–3 m, densità dei punti suolo | [2] |
| Canali extra | sky-view factor, openness, local dominance (RVT, Apache-2.0); LRM per i solchi | [9], [12] |
| Normalizzazione | min-max per patch sul DTM grezzo schiaccia i solchi da 20 cm quando ci sono colline → usare derivati detrendizzati | [1] |
| Etichette | centerline rasterizzate a larghezza standard (scheletro + buffer) | [1], [2] |
| Etichette incomplete | loss solo in finestre attorno ai pixel positivi (PLS) o solo nel buffer della strada | [5], [2] |
| Etichette spostate/mancanti | Soft-Bootstrapped Dice, robusta a omissioni e spostamenti | [6] |
| Auto-addestramento | modello iniziale → predizioni corrette → nuove etichette → secondo addestramento | [1] |
| Squilibrio di classe | focal loss | [2] |
| Aumenti | rotazioni di 90°, ribaltamenti, channel dropout (p 0,2, max 3 canali); TTA su 6 trasformazioni | [2], [1] |
| Connettività | clDice (loss sullo scheletro, preserva la topologia) | [8] |
| Buchi nel grafo | superficie di conduttività + percorso a costo minimo tra estremi | [2] |
| Grafo end-to-end | RNGDet++, Sat2Graph, DeH4R (per la fase successiva) | [11] |
| Valutazione | blocchi spaziali: con hold-out casuale si sovrastima fino al 28%; conta la dimensione del blocco | [7] |

## Numeri pubblicati (per calibrare le attese)

- Sentieri e tracce nella foresta boreale, DTM 50 cm: precisione 77 ± 9%, recall 78 ± 14% [1].
- Strade forestali, Ontario: centerline entro 3 m dal riferimento dal 66,7% all'87,2%, errore medio 2,07 m [2].
- Piste di esbosco in boschi diradati, U-Net su DSM: IoU 0,889–0,963 [4].
- Nessuna fonte letta riporta recall ≥ 98%: l'obiettivo è oltre lo stato dell'arte pubblicato.

## Ricetta per Morloquio

1. Canali per pixel a 1 m: hillshade, LRM, pendenza, curvature, rugosità, nDSM (DSM first − DTM), DSM first − DSM last
   (penetrabilità della chioma, sostituto della densità dei punti che non abbiamo), RVT (SVF, openness),
   segnali dello Stadio A (`bench`, `lrm_lin`, `chm_gap`, `gps`), Sentinel-2.
2. Etichette: vie OSM rasterizzate con larghezza per classe; loss pesata solo nel buffer delle vie OSM e in finestre
   ancorate ai positivi; zone senza OSM a peso basso (positive-unlabeled), non negativi certi.
3. Loss: focal + Soft-Bootstrapped Dice + clDice.
4. Validazione: k-fold a blocchi di tile (≥ 1 km), metrica di uscita solo su blocchi mai visti.
5. Auto-addestramento: predizioni ad alta confidenza fuori OSM → verifica visiva con Inkling in locale → nuove etichette.
6. Grafo: scheletro + percorso a costo minimo sulla probabilità per chiudere i buchi tra estremi vicini.

## Fonti

- [1] Mapping Trails and Tracks in the Boreal Forest Using LiDAR and CNN, Remote Sensing 17(9):1539, 2025 — https://www.mdpi.com/2072-4292/17/9/1539
- [2] Updating Forestry Road Networks in Ontario Using Single Photon LiDAR and Deep Learning, ISPRS Archives XLIX-B1-2026 — https://isprs-archives.copernicus.org/articles/XLIX-B1-2026/421/2026/
- [4] Logging Trail Segmentation via a Novel U-Net CNN and High-Density Laser Scanning Data, Remote Sensing 14(2):349, 2022 — https://www.mdpi.com/2072-4292/14/2/349
- [5] Positive-Guided Local Supervision for Robust Road Extraction, Remote Sensing 18(10):1589, 2026 — https://www.mdpi.com/2072-4292/18/10/1589
- [6] Henry et al., Aerial Road Segmentation in the Presence of Topological Label Noise (DLR) — https://elib.dlr.de/136343/
- [7] Kattenborn et al., ISPRS Open Journal 2022 (solo estratto di ricerca) — https://doaj.org/article/23b9fa0bb95147349b20e4cdb872469e
- [8] Shit et al., clDice — https://arxiv.org/abs/2003.07311
- [9] Relief Visualization Toolbox (rvt_py, Apache-2.0) — https://github.com/EarthObservation/rvt_py
- [10] P2CNet, Road Extraction with Satellite Images and Partial Road Maps — https://arxiv.org/abs/2303.12394
- [11] RNGDet++ — https://arxiv.org/abs/2209.10150 ; DeH4R — https://arxiv.org/abs/2508.13669 (solo estratti di ricerca)
- [12] CarcassonNet / hollow roads, tesi Leiden (solo estratto di ricerca) — https://scholarlypublications.universiteitleiden.nl/access/item%3A3203911/download
- Non letto: preprint UiT su strade piccole da LiDAR (DNS non risolto).
