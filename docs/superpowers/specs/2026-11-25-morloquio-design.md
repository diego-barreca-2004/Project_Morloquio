# Project Morloquio — Documento di design

Data: 2026-11-25 · Stato: bozza da approvare · Autore della richiesta: proprietario del progetto (diego-barreca-2004)

## 1. Intento e perché

Morloquio vuole produrre, in modo **completamente automatico**, la mappa di **tutte** le vie percorribili del mondo:
strade asfaltate, sterrate, carrarecce, mulattiere, sentieri, anche sotto copertura boschiva.

Il motivo: OpenStreetMap, Microsoft Road Detections e Meta/RapiD lasciano grandi buchi, soprattutto su sentieri e
tracciati minori. Il proprietario ritiene che le tecniche degli ultimi anni (deep learning, modelli fondazionali,
LLM agentici a pesi aperti) permettano di colmarli partendo solo da risorse aperte.

La sfida è esplicitamente "fare il massimo con poche risorse aperte". Non è un compito semplice e non va semplificato:
costo e tempo di calcolo non sono un vincolo, la qualità del risultato sì.

Strategia generale scelta dal proprietario: **Italia prima** (molti dati aperti → primo grande dataset), poi
estensione ai paesi senza dati, con modelli diventati abbastanza forti da riconoscere le strade anche lì.

Area pilota: zona di Morloquio, punto di riferimento 38°04'32.2"N 15°40'39.7"E (38.0756 N, 15.6777 E), vicino a
Reggio Calabria. Estensione: 38.03–38.12 N, 15.62–15.75 E (~10×11 km, 97 tile LiDAR 1×1 km), con un nucleo di 2×2 km
attorno al punto per la revisione di dettaglio.

## 2. Obiettivi, non-obiettivi, anti-obiettivi

**Obiettivi**
- G1. Pilota Morloquio: rete viaria completa (incluse vie sotto bosco) che il proprietario giudichi, a occhio,
  nettamente superiore alla heatmap Strava nella stessa zona.
- G2. Italia: dataset nazionale aperto di vie percorribili, generato automaticamente, pubblicabile.
- G3. Mondo: pipeline che scala a ogni paese, scegliendo automaticamente le migliori fonti aperte disponibili.
- G4. Acquisizione dati completamente automatizzata, dalla scoperta delle fonti al download.
- G5. Repo e cartelle organizzate per il debugging: ogni stadio produce artefatti ispezionabili.

**Non-obiettivi (per ora)**
- Routing/navigazione, app per utenti finali, rendering cartografico di pregio.
- Attributi ricchi (nomi, sensi unici, limiti di velocità): solo geometria, tipo di via e superficie stimata.
- Edifici, idrografia, uso del suolo (usati solo come input/negativi, non come prodotto).

**Anti-obiettivi (fallimento anche se "funziona")**
- Qualunque uso di strumenti, modelli o dati non aperti al 100%: Google Maps/Earth, Strava (heatmap inclusa),
  Esri, Bing, Mapbox, immagini commerciali, API a pagamento. Il proprietario lo vieta in modo assoluto.
- Passi manuali nella catena di produzione (l'unico intervento umano ammesso è la valutazione visiva del
  proprietario e, se la fornisce, l'annotazione di verità a terra per la valutazione).
- Strade inventate presentate come rilevate: ogni via in output dichiara da quali dati è stata rilevata e con
  quale confidenza.
- Una pipeline che non si riesce a debuggare (scatole nere senza artefatti intermedi).

## 3. Vincoli assoluti

1. **Open source al 100%**: codice, librerie, modelli (pesi con licenza OSI, es. Apache-2.0/MIT/BSD), runtime.
   Modelli con licenze custom o non commerciali sono esclusi (es. pesi "research only", DINOv3).
2. **Dati aperti**: ammessi CC0, CC BY, ODbL, PDDL, IODL 2.0, Etalab/Licence Ouverte, OGL, pubblico dominio e
   simili. Esclusi NC, ND, licenze ambigue. I server pubblici da cui si scaricano dati aperti non sono "strumenti".
3. **Strava**: solo riferimento visivo del proprietario. Il sistema non la scarica, non la legge, non la cita.
4. **Repo GitHub pubblica** `Project_Morloquio`.
5. **LLM agentico**: Inkling 2 Mini (Thinking Machines, Apache-2.0), versione più grande che gira sulla macchina
   del proprietario (vedi §5.6).
6. **Strato di sistema**: Windows/WSL2, driver NVIDIA e runtime CUDA/cuDNN sono ammessi come strato di sistema
   (senza di essi la GPU non si usa); il license gate controlla solo dipendenze, modelli e dati del progetto
   (i pacchetti `nvidia-*` installati da PyTorch sono in una eccezione esplicita). Scelta dell'assistente, correggibile.
7. **Hardware attuale**: RTX 5070 Ti Laptop 12 GB VRAM, WSL2 con 26 GB RAM + 24 GB swap, 24 thread, ~683 GB liberi.
   Nessun limite su tempo di calcolo; il cloud non è previsto finché non serve davvero.

## 4. Approcci considerati

| # | Approccio | Pro | Contro |
|---|-----------|-----|--------|
| 1 | Solo geometria del terreno (filtri sul DTM LiDAR, nessun ML) | Nessuna etichetta, risultati in giorni, trasparente | Confonde terrazzamenti, fossi, fiumare, muri a secco; inutilizzabile senza LiDAR |
| 2 | Solo deep learning (segmentazione su ortofoto + derivati LiDAR, etichette OSM) | Tetto alto, impara le distinzioni fini | OSM incompleto = etichette scarse e rumorose proprio dove serve; difficile da debuggare |
| 3 | **Ibrido a stadi** + auto-addestramento + distillazione verso fonti globali | Alto recall dalla geometria, precisione dal ML, errori localizzabili per stadio, scala per livelli di dati | Più componenti da costruire e mantenere |

**Scelta: approccio 3**, nella versione completa (non semplificata). Il proprietario ha delegato il "come" chiedendo
di non semplificare e di puntare alla qualità massima; la scelta è dell'assistente.

Perché: lo stadio geometrico trova le vie senza bisogno di etichette (cruciale dove OSM manca), il ML scarta i falsi
positivi, il grafo ricostruisce la rete. Ogni stadio produce artefatti propri, quindi ogni errore si attribuisce a uno
stadio preciso. L'Italia, ricca di LiDAR aperto, fa da "insegnante" per modelli che lavorano con soli dati globali.

## 5. Architettura

### 5.1 Livelli di dati (tier)

La qualità ottenibile dipende fisicamente dai dati disponibili. Ogni tile del mondo viene assegnata a un tier:

| Tier | Dati | Cosa si può rilevare |
|------|------|----------------------|
| T1 | LiDAR ≤ 1 m (DTM/DSM) + ortofoto ≤ 0,5 m | Tutto, sentieri sotto bosco inclusi (solco/gradino nel DTM) |
| T2 | Ortofoto ≤ 1 m, niente LiDAR | Vie in aree aperte; sotto bosco solo vie larghe con varco nella chioma |
| T3 | Solo satelliti globali: Sentinel-2 10 m (serie temporali), Sentinel-1 SAR, Copernicus DEM 30 m, tracce GPS OSM | Strade ≥ ~3–4 m; sentieri sotto bosco **non rilevabili direttamente**, solo inferibili |

**Limite fisico da dichiarare**: in T3 un sentiero di 1 m sotto chioma non è contenuto in nessun dato aperto. Il
modello può solo inferirne la presenza probabile dal contesto. Il sistema separa sempre "rilevato" da "inferito"
(campo `evidence` + `confidence` su ogni via). Come trattare le vie inferite nel prodotto: domanda Q2 (§12).

### 5.2 Catalogo fonti e acquisizione automatica

- `configs/sources/*.yaml`: una voce per fonte con id, fornitore, licenza (verificata), risoluzione, data,
  copertura (poligono o indice tile), protocollo (WMS/WCS/WFS/STAC/HTTP/S3/API), adapter, checksum.
- **License gate**: una fonte entra nel catalogo solo se la licenza è nella allowlist (§3.2). Licenza ambigua →
  rifiutata di default. Stesso gate per le dipendenze Python e i pesi dei modelli (controllo in CI).
- Un adapter per protocollo, scaricamento idempotente, ripresa dopo interruzione, cache su disco, checksum,
  log strutturato. Tutto per tile.
- **Strategia di storage**: l'Italia T1 vale diversi TB grezzi. Elaborazione in streaming per tile:
  scarica → deriva → conserva solo derivati compressi (COG) e metadati → elimina il grezzo (riscaricabile, checksum
  nel catalogo). Volumi reali da misurare in F1; storage esterno solo se serve.

Fonti iniziali verificate per il pilota (dettagli in §11):
- LiDAR PST del Ministero dell'Ambiente (MASE): DTM, DSM first, DSM last, 1 m, GeoTIFF, **CC BY 4.0**. Indice
  tile via WFS e via FeatureServer `Grigliato_lidar` (layer 2 = griglia 1×1, campi `fid`, `tavola`, `regione`,
  `contratto`, `data`). **Download automatico verificato il 2026-11-25** tramite la stessa API REST pubblica usata
  dall'app ufficiale, senza login né token:
  1. `POST https://sim.mase.gov.it/core/api/esri/api/v1/downloads` con JSON
     `{"canale":"PST2","prodotto":"LIDAR_2014","layer":"dtm_1x1|dsm_first_1x1|dsm_last_1x1","tileList":[{"fid","tavola","regione","contratto"}],"metadataList":[]}`
     → `idSessione` (HTTP 202);
  2. eventi SSE su `.../downloads/{idSessione}/events` oppure stato su `.../downloads/{idSessione}`;
  3. file su `https://sim.mase.gov.it/core/api/esri/api/v1/downloads/{idSessione}/file/{idFile}` (zip con GeoTIFF
     1000×1000 float32, ~4 MB per tile e prodotto);
  4. `POST .../downloads/{idSessione}/ack` a download concluso.
  Uso cortese: User-Agent onesto, richieste sequenziali, massimo 20 tile per richiesta (limite dell'app), backoff.
  Nessun aggiramento di firewall o protezioni.
- Ortofoto AGEA 2012, 50 cm, WMS nazionale ("AccessConstraints: Nessuno"). Licenza del servizio WMS non esplicita:
  da confermare prima dell'uso; se non risulta aperta, la fonte è esclusa.
- OpenStreetMap (estratti Geofabrik, ODbL), tracce GPS pubbliche OSM (API trackpoints; nel pilota solo 29 punti in
  ~2×2 km).
- Microsoft Road Detections (ODbL) **escluso per ora**: è derivato da immagini Bing, che il proprietario vieta.
  OSM resta ammesso (fonte aperta indicata dal proprietario). Da rivalutare con il proprietario in F8.
- Sentinel-2 L2A e Copernicus DEM GLO-30 da cataloghi STAC aperti senza account (es. Earth Search su AWS).

### 5.3 Pipeline per tile

Unità di lavoro: tile 1×1 km allineate alla griglia LiDAR PST (0,01°) in Italia; griglia equivalente altrove.
Ogni stadio legge e scrive file versionati per `run_id`; nessuno stadio tiene stato in memoria tra tile.

**Stadio 0 — Cubo dati.** Riproiezione (EPSG:32633 per il pilota), allineamento a griglia 0,5 m (ortofoto) e 1 m
(LiDAR), maschere di validità, metadati di data/fonte per pixel.

**Stadio A — Candidati (alto recall, senza etichette).**
- Dal DTM: Local Relief Model, openness positiva/negativa multi-scala, pendenza, curvature, sky-view factor, filtri
  di linearità (Hessiano/Frangi) a larghezze 1–6 m, rilevatore di **gradini di versante** (striscia piana con taglio
  a monte e riporto a valle: firma tipica di sentieri e carrarecce).
- Da DSM first − DTM: modello di altezza chioma; varchi lineari nella chioma.
- Dall'ortofoto: filtri di linearità orientati, bordi, colore/tessitura.
- Dalle tracce GPS: raster di densità.
- Fusione in un punteggio di "percorribilità lineare" per pixel, soglia bassa (recall ≥ 98% sulle vie OSM note),
  scheletrizzazione, vettorizzazione in segmenti con attributi geometrici.

**Stadio B — Classificazione (precisione).**
- B1. Rete di segmentazione densa su stack multi-canale (ortofoto + derivati terreno + chioma + punteggio A).
  Encoder pre-addestrati con licenza aperta da confrontare in F3 (es. pesi timm/ImageNet, Clay, Prithvi-EO-2.0,
  DINOv2: verificare la licenza dei pesi prima dell'uso). Output: probabilità di via + classe
  (strada asfaltata / sterrata / traccia / sentiero).
- B2. Classificatore per segmento (gradient boosting o piccola rete su grafo) con feature geometriche: lunghezza,
  sinuosità, pendenza longitudinale e sua regolarità, profilo trasversale, larghezza, connettività, relazione con le
  linee di deflusso (calcolate dal DTM: i fossi seguono il deflusso, i sentieri di solito no), relazione con
  terrazzamenti ed edifici.
- **Etichette**: positivi da OSM `highway=*`; negativi espliciti da OSM `waterway`,
  `barrier=wall|retaining_wall`, bordi di edifici, linee di deflusso, bordi di terrazzamento. **L'assenza in OSM
  non è un negativo** (OSM è incompleto per definizione): addestramento positive-unlabeled e perdite robuste al
  rumore.
- **Auto-addestramento**: i segmenti nuovi ad alta confidenza diventano pseudo-etichette per il giro successivo.
  Le regioni di test non entrano mai nel ciclo.
- **Active learning**: i segmenti più incerti vanno in coda di revisione (agente QA e, se vuole, proprietario).

**Stadio C — Grafo.**
- Grafo da scheletro, potatura degli speroni, chiusura dei buchi con percorsi a costo minimo (costo da pendenza,
  chioma, probabilità B) tra estremi liberi vicini.
- Controlli topologici: una via termina su un'altra via, un insediamento, una cima, una sorgente o un confine di
  tile; estremi liberi sospetti vanno in revisione.
- Cucitura tra tile adiacenti.
- Conflazione con OSM: ogni via è marcata come `existing_osm`, `new`, `conflict`.
- Assegnazione tag in stile OSM (`highway=track|path|footway|unclassified|...`, `surface` stimata).

### 5.4 Distillazione: l'Italia come laboratorio

- **Insegnante**: pipeline completa T1 sull'Italia → mappa densa, molto oltre OSM.
- **Studenti**: modelli che vedono solo input T2 (ortofoto) o T3 (serie Sentinel-2, Sentinel-1, DEM 30 m, densità
  GPS), addestrati sulle uscite dell'insegnante.
- **Misura onesta del degrado**: in regioni italiane di test si nascondono LiDAR/ortofoto e si confronta lo studente
  con l'insegnante. Questo dice, prima di andare all'estero, quanto vale il modello nei paesi senza dati.
- **Spostamento di dominio**: paesaggi italiani ≠ tropici/deserti/steppe. Prima delle aree T3 si aggiungono altri
  paesi T1 con dati aperti (individuati dall'agente, §5.6) per dare varietà all'insegnante.

### 5.5 Output

- GeoPackage per tile e mosaico nazionale; esportazione `.osm`/`.osm.pbf` con tag in stile OSM.
- Ogni via: geometria, classe, superficie stimata, `confidence`, `evidence` (quali fonti l'hanno vista), tier,
  data dei dati sorgente, `osm_status`.
- Licenza dei dati in uscita: **ODbL 1.0** (obbligata dall'uso di etichette OSM) con attribuzioni CC BY delle fonti.

### 5.6 Agenti con Inkling 2 Mini

Modello: **Inkling 2 Mini** (26,4 B parametri totali, 3,4 B attivi per token, 5,4 B contando i passaggi in loop,
da cui il tag Ollama `a5b`; MoE con strati in loop, input testo + immagini,
contesto 1 M token, Apache-2.0). È il più grande della famiglia che può girare su questa macchina: il successivo
(Inkling 2 Small, 594 B) richiede centinaia di GB. Il checkpoint nativo è ~20 GB (NVFP4/MXFP8) e la scheda ne
raccomanda 32 GB di VRAM: qui si usa la versione GGUF (~18–20 GB, es. `inkling-2-mini:26b-a5b` o IQ4_NL) con
llama.cpp/Ollama (MIT), parte degli esperti in RAM. In F0 si misura quale quantizzazione dà la precisione più alta
con velocità accettabile, e se l'input immagini funziona nel runtime scelto. Il modello è esposto come API locale
compatibile OpenAI, così sostituirlo è solo configurazione. Inkling è una scelta del proprietario: se in F0 non regge
(velocità o immagini), si chiede al proprietario prima di cambiare modello.

GPU condivisa: LLM e training non girano mai insieme sulla GPU; uno scheduler semplice li alterna.

Compiti degli agenti:
1. **Scoperta fonti** (abilita l'automazione mondiale): interroga cataloghi aperti (data.europa.eu, INSPIRE CSW,
   CKAN nazionali, data.gov, OpenAerialMap, indice open-lidar-data), estrae licenza, risoluzione, copertura,
   protocollo; propone voci di catalogo. Un validatore deterministico prova l'endpoint e applica il license gate.
   Niente motori di ricerca web generici.
2. **QA visivo**: riceve immagini renderizzate (ortofoto, hillshade, previsione sovrapposta) dei segmenti incerti e
   dà un giudizio motivato. I giudizi sono etichette deboli a basso peso, mai verità a terra.
3. (Più avanti) **Generazione adapter** per nuove fonti, con test, in sandbox.

Sicurezza operativa: la scheda del modello avverte che negli usi agentici può compiere azioni distruttive. Gli agenti
girano con strumenti in whitelist, scrittura solo in una cartella sandbox, nessuna shell libera.

## 6. Valutazione e criteri di successo

- **Pilota, accettazione**: il proprietario confronta a occhio l'output con la heatmap Strava (fuori dal sistema).
  Per questo il progetto fornisce un **visualizzatore locale** (MapLibre GL JS + PMTiles, licenze BSD) con layer
  attivabili: ortofoto, hillshade LiDAR, output Morloquio per classe e confidenza, OSM attuale.
- **Metriche automatiche** (non sostituiscono il giudizio del proprietario):
  - recall sulle vie OSM esistenti nel pilota (deve essere ≈ 100%);
  - completeness/correctness con buffer (Wiedemann), APLS e TOPO su regioni di test;
  - **test di riscoperta**: si nascondono le vie OSM di una regione ben mappata e si misura quante ne ritrova;
  - per tier: degrado degli studenti rispetto all'insegnante (§5.4).
- Regioni di test italiane: zone con sentieristica OSM molto curata e LiDAR aperto, da scegliere in F3; mai usate
  in addestramento.
- Obiettivi numerici iniziali (proposta da tarare dopo F2): pilota recall ≥ 95%, correttezza ≥ 90% sulle regioni
  di test T1.

## 7. Infrastruttura, repo, debugging

- Python 3.12, ambiente con `uv`; librerie principali (tutte aperte): GDAL, rasterio, PDAL, WhiteboxTools,
  scikit-image, NumPy, PyTorch, TorchGeo, segmentation-models-pytorch, timm, LightGBM, networkx/igraph,
  shapely/geopandas, pystac-client, osmium.
- Orchestrazione: Snakemake (pipeline per tile con wildcard). Versionamento dati e modelli: DVC con remote locale.
  Esperimenti: MLflow locale. Log strutturati JSON con `run_id`, `tile_id`, `stage`.
- Qualità: ruff, pytest, pre-commit, CI GitHub Actions (lint, test, license gate).
- Repo pubblica `Project_Morloquio`; codice **Apache-2.0**; identificatori in inglese, documentazione e commenti in
  italiano.

Struttura:

```
Project_Morloquio/
  README.md  LICENSE  DATA_LICENSES.md  CONTRIBUTING.md  pyproject.toml  uv.lock  Snakefile
  configs/            aree (aoi/morloquio.yaml), fonti (sources/), modelli (models/), pipeline
  src/morloquio/
    sources/          catalogo, license gate, adapter (pst_lidar, agea_wms, osm, gps_traces, sentinel2, ...)
    ingest/           tiling, riproiezione, cubo dati (Stadio 0)
    terrain/          derivati LiDAR
    candidates/       Stadio A
    classify/         Stadio B (dataset, modelli, training, inferenza)
    graph/            Stadio C
    distill/          insegnante → studenti
    agents/           Inkling: scoperta fonti, QA visivo, sandbox
    eval/             metriche e report
    export/           GeoPackage, OSM PBF, PMTiles
    viewer/           visualizzatore locale
    common/           log, run_id, I/O, griglia tile
  tests/              unit + fixture minuscole (mini-tile reali ritagliate)
  docs/               architettura, fonti, decisioni (ADR), superpowers/specs, superpowers/plans
  notebooks/          solo esplorazione, numerati, mai codice di produzione
  data/   (ignorata)  raw/ interim/ processed/ cache/
  runs/   (ignorata)  <run_id>/<stage>/<tile_id>/ artefatti + log
```

Debugging: ogni stadio scrive per tile raster/vettori intermedi e un `report.json`; un comando apre un tile a
qualunque stadio nel visualizzatore; ogni run è riproducibile da `run_id` (config + versioni dati + commit).

## 8. Fasi

Il piano dettagliato copre F0–F2; le fasi successive si pianificano quando si arriva, con i dati misurati.

| Fase | Contenuto | Uscita |
|------|-----------|--------|
| F0 | Repo pubblica, ambiente, CI, license gate, log, griglia tile, Inkling 2 Mini installato e misurato | `make check` verde; benchmark Inkling salvato |
| F1 | Catalogo fonti + adapter automatici per il pilota (PST LiDAR, ortofoto, OSM, GPS, MS Roads, Sentinel-2, DEM) | Un comando costruisce il cubo dati del pilota; visualizzatore mostra i layer |
| F2 | Stadio A sul pilota | Layer candidati; recall sulle vie OSM note ≥ 98% |
| F3 | Etichette + Stadio B, regioni di test | Metriche su test; prima revisione visiva del proprietario |
| F4 | Stadio C + export OSM | Rete connessa; seconda revisione del proprietario (vs Strava, a occhio) |
| F5 | Agente QA + active learning | Ciclo di miglioramento automatico misurato |
| F6 | Scala Italia (PST + fonti regionali aperte trovate dall'agente) | Dataset Italia v0 pubblicato |
| F7 | Studenti T2/T3 + misura del degrado | Metriche per tier |
| F8 | Agente scoperta fonti mondiale + altri paesi T1 | Catalogo globale |
| F9 | Estensione mondiale per tier | Dataset mondiale |

## 9. Rischi noti

- API di download PST non documentata ufficialmente: può cambiare senza preavviso. Mitigazione: adapter isolato con
  test di contratto giornaliero su un tile noto; il firewall del sito blocca i browser automatici, quindi si usano solo
  client HTTP semplici sull'API pubblica. In Calabria non risultano altre fonti LiDAR aperte.
- Licenza WMS AGEA 2012 non confermata → se non è aperta, il pilota T1 usa solo LiDAR + Sentinel-2.
- Dati vecchi (LiDAR 2008–2016, ortofoto 2012) → vie recenti assenti; rilevazione dei cambiamenti con Sentinel-2.
- Terrazzamenti e fiumare nelle colline reggine → falsi positivi; li gestisce lo Stadio B con negativi espliciti.
- Inkling 2 Mini uscito da pochi giorni → possibili bug dei runtime; il livello agenti resta sostituibile.
- Una sola GPU laptop → training lento a scala Italia; accettato dal proprietario.
- Limite fisico T3 sotto bosco (§5.1).

## 10. Decisioni

**Prese dal proprietario**: open source e dati aperti al 100%; divieto assoluto di Google/Earth/Strava/fonti a
pagamento; Strava solo come riferimento visivo suo; repo pubblica; automazione completa incluso il download; Italia
come primo dataset, poi paesi senza dati; Inkling come LLM, versione massima eseguibile; nessuna semplificazione;
costo e tempo di calcolo non contano; reti neurali non obbligatorie.

**Prese dall'assistente (correggibili)**: approccio 3; tier T1/T2/T3 e separazione rilevato/inferito; tile 1×1 km
sulla griglia PST; licenze Apache-2.0 (codice) e ODbL (dati); Snakemake + DVC + MLflow; storage in streaming;
agenti limitati ai cataloghi aperti; identificatori in inglese, documenti in italiano; visualizzatore locale per la
valutazione a occhio; obiettivi numerici iniziali; area pilota 97 tile (~10×11 km) con nucleo 2×2 km;
Microsoft Road Detections escluso; strato di sistema (driver NVIDIA, CUDA, WSL) ammesso; nessun cambio di LLM senza
chiedere al proprietario.

**Lasciate al builder**: scelta finale degli encoder e delle architetture (per benchmark), quantizzazione di
Inkling (per misura), soglie dei filtri, scelta delle regioni di test, dettagli degli adapter.

## 11. Fonti verificate (2026-11-25)

- Distribuzione dati PST, CC BY 4.0: https://gn.mase.gov.it/portale/distribuzione-dati-pst · note legali:
  https://gn.mase.gov.it/portale/note-legali · app di download: https://sim.mase.gov.it/pst/
- Indice tile LiDAR (WFS): `http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/wfs/QU_Lidar.map`, layer
  `EL.QUADRO.UNIONE.LIDAR.0101` (tile 1×1 km; tile 38.07–38.08 N / 15.67–15.68 E presente; 97 tile in
  38.03–38.12 N, 15.62–15.75 E).
- Ortofoto AGEA 2012 (WMS): `http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/WMS_v1.3/raster/ortofoto_colore_12.map`,
  layer `OI.ORTOIMMAGINI.2012.33`.
- Collezione LiDAR nazionali aperti: https://registry.opendata.aws/open-lidar-data/
- Inkling 2 Mini: https://thinkingmachines.ai/news/inkling-2-mini/ · https://huggingface.co/thinkingmachines/Inkling-2-Mini
  · https://ollama.com/library/inkling-2-mini

## 12. Domande chiuse con il proprietario (2026-11-25)

- **Q1. Cosa conta come via** → accettata la proposta: si mappa tutto ciò che ha evidenza fisica, con attributo
  `lifecycle` (`active` / `likely_abandoned`) e `access` quando deducibile. Il confronto con Strava si fa sul layer
  `active`.
- **Q2. Vie inferite** → accettata la proposta: layer separato `inferred`, mai nell'export `.osm` principale; i buchi
  brevi con evidenza parziale restano nel principale con `evidence=gap_fill`. Il proprietario nota che partendo
  dall'Italia (tier T1) il problema è marginale; diventa rilevante da F7.
- **Q3. Download PST** → il proprietario chiede una via legale automatica. Trovata e verificata (§5.2): l'API REST
  pubblica dell'app ufficiale, senza login. Non serve alcuna richiesta al Ministero.
- **Q4. Verità a terra** → nel pilota la correttezza la verifica il proprietario a occhio. Nessuna annotazione
  richiesta per ora; un sistema automatico di verifica si valuterà dopo il pilota. In F3–F4 le metriche automatiche
  (§6) restano di supporto; il visualizzatore è lo strumento principale di accettazione.
