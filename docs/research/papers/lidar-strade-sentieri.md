# Strade, sentieri, fossi e solchi da LiDAR con reti profonde

Note di lettura integrale (2026-11-26), base dell'ADR 0004. Pagine lette con Crawl4AI, PDF scaricati ed estratti
con pypdf. Sintesi e ricetta in `../2026-11-26-ml-strade-da-lidar.md`.

---

### Salberg, Trier, Kampffmeyer 2017 — Strade minori in immagini lidar con FCN (SCIA 2017)

- **Riferimento**: A.-B. Salberg, Ø. D. Trier, M. C. Kampffmeyer, "Large-Scale Mapping of Small Roads in Lidar Images Using Deep Convolutional Neural Networks", SCIA 2017, LNCS, Springer, pp. 193–204, DOI 10.1007/978-3-319-59129-2_17. Copie cercate: Munin (hdl 10037/29126), NVA/Sikt, CORE, Wayback, NR.
- **Lettura**: **non letto (testo integrale non raggiungibile)**. `munin.uit.no` è un CNAME verso `wr3.uit.no`, che restituisce NXDOMAIN anche dai resolver pubblici (resolver DNS pubblici): il guasto è del DNS di UiT, non locale. Il mirror `ninum.uit.no` non risolve. NVA (`nva.sikt.no`) elenca `article.pdf` (3,08 MB, submitted version) come file aperto, ma l'API di download anonima risponde `Forbidden`; CORE è dietro Cloudflare; Wayback non ha snapshot (servizio anche offline al momento). Le informazioni seguenti vengono solo da abstract, pagina progetto NR e dalla descrizione dei dati nel lavoro DDCM-Net (Liu et al. 2019, che usa lo stesso dataset e cita questo articolo come [18]).
- **Cosa fanno**: rete fully convolutional end-to-end per segmentare strade forestali da dati laser aerei; addestrata su "a large set of image patches with corresponding road label information"; applicata al comune di Etnedal (Norvegia). Progetto NR "LasTrak" (2015–2016) con l'Autorità cartografica norvegese, ufficio di Hamar; obiettivi: trovare strade forestali non mappate, classificare strade forestali e "tractor roads" per pendenza, rilevare vegetazione che copre la strada.
- **Dati** (secondo Liu et al. 2019): 2 immagini 19200 × 12800 con 2 bande derivate dal DTM, **gradiente di quota + hillshade**; 4 classi (sfondo, Road1, Road2, Road3). Risoluzione, numero di patch e origine delle etichette: non verificabili.
- **Metodo**: FCN end-to-end (dettagli non verificabili). La pagina NR dice che il modulo di rilevamento è "experimental" e richiede GPU; codice IDL e C.
- **Risultati**: "overall accuracy of 97.2%" (abstract). Liu et al. 2019 confermano indirettamente: il loro >99,3% è "about 2.1% higher than the previous work".
- **Limiti**: l'accuratezza globale è dominata dallo sfondo e non dice nulla sulla completezza; la pagina NR definisce il metodo sperimentale.
- **Per Morloquio**: unica cosa utilizzabile è la scelta dei canali, **gradiente + hillshade dal DTM**, che conferma l'idea di dare alla rete derivate morfometriche del DTM e non la quota assoluta. Il 97,2% non è un riferimento utile per il target di recall 0,98. Da rileggere se Munin torna raggiungibile.

---

### Liu, Kampffmeyer, Jenssen, Salberg 2019 — DDCM-Net joint-task per strade in lidar (IGARSS 2019)

- **Riferimento**: Q. Liu, M. Kampffmeyer, R. Jenssen, A.-B. Salberg, "Road mapping in lidar images using a joint-task dense dilated convolutions merging network", IGARSS 2019, pp. 5041–5044, DOI 10.1109/IGARSS.2019.8900082; letto arXiv 1909.04588v1 (PDF, 4 pagine).
- **Lettura**: completa.
- **Cosa fanno**: rete leggera con blocchi di convoluzioni dilatate densamente concatenate (DDCM) su encoder ResNet50 troncato; apprendimento congiunto con due uscite ausiliarie (classificazione multi-etichetta dei tipi di strada presenti nella patch, segmentazione binaria strada/non strada) oltre alla segmentazione a 4 classi; pesi delle loss ausiliarie estratti a caso a ogni iterazione.
- **Dati**: dataset lidar di Salberg et al. 2017: 2 immagini 19200 × 12800, 2 bande (gradiente di quota, hillshade), 4 classi (BG, Road1, Road2, Road3; le classi non sono definite nel testo). Un'immagine divisa in training e validazione, **l'altra usata come test** (split spaziale per immagine). Risoluzione al suolo non riportata. Dati ALS forniti da Oppland County, vettori dall'ufficio di Hamar.
- **Metodo**:
  - Blocco DC: convoluzione dilatata con stride 2, PReLU, BatchNorm, upsampling bilineare alla dimensione di ingresso, concatenazione con l'ingresso; modulo di fusione finale con filtri 1 × 1 + BN + PReLU.
  - Encoder: ResNet50 pre-addestrata su ImageNet, solo i primi 3 bottleneck (tolto l'ultimo e i FC). Con meno di 3 bande, una conv 3 × 3 + BN + PReLU genera la terza banda che viene impilata alle 2 originali.
  - DDCM sulle feature di basso livello con dilatazioni [1, 2, 3, 5, 7, 9]; decoder con due DDCM, dilatazioni [1, 2, 3, 4] e [1].
  - Loss: `L_total = L_mce + w_r1·L_bce + w_r2·L_lovasz`, con L_mce = cross-entropy 2D con median frequency balancing (task principale), BCE per la classificazione multi-etichetta, Lovász-softmax per la segmentazione binaria; w_r1, w_r2 ~ U(0, 1) estratti a ogni iterazione.
  - Ottimizzatore Adam con AMSGrad; LR iniziale 0,00012, dimezzato (×0,5) alle epoche [5, 15, 25, 65, 100]; weight decay 0,00005; LR doppio per i bias.
  - Lidar: 1000 patch 1024 × 1024 campionate a caso per epoca; aumenti flip/mirror; normalizzazione in [0,0, 1,0], nessuna sottrazione di media/deviazione standard. Batch ed epoche totali: non riportati.
  - Inferenza: TTA con flip e mirror; finestra scorrevole 1024 × 1024 con circa 60% di sovrapposizione, media delle predizioni sovrapposte.
  - Costo: JT-DDCM-R50 9,30 M parametri e 4,44 GFLOPs su ingresso 2 × 256 × 256 (DDCM-R50: 9,99 M e 4,86 GFLOPs su 3 × 256 × 256).
- **Risultati** (test set lidar, metriche per pixel, nessun buffer di tolleranza):

  | Modello | mF1 | BG | Road1 | Road2 | Road3 | mIoU | IoU Road1 | IoU Road2 | IoU Road3 |
  |---|---|---|---|---|---|---|---|---|---|
  | DDCM-R50 | 0,650 | 0,995 | 0,619 | 0,284 | 0,702 | 0,536 | 0,448 | 0,165 | 0,541 |
  | JT-DDCM-R50 | 0,696 | 0,997 | 0,715 | 0,282 | 0,789 | 0,592 | 0,556 | 0,164 | 0,652 |

  Accuratezza globale "above 99.3%" per entrambi. Il joint-task dà mappe "less fragmented". Su ISPRS Potsdam RGB, DDCM-R50 mIoU 0,808; su Vaihingen IRRG mF1 0,898, OA 0,904.
- **Limiti**: dichiarati: Road2 confusa con Road1 (caratteristiche simili in gradiente e hillshade), servono più campioni di Road2. Miei: niente precision/recall separati né metriche di completezza vettoriale; un solo test (un'immagine) senza varianza; risoluzione e definizione delle classi non dichiarate; nessun dettaglio su numero di epoche e batch.
- **Per Morloquio**:
  - Adottare: ingresso a derivate del DTM (gradiente, hillshade) con eventuale conv 3 × 3 apprendibile per arrivare a 3 canali e usare un encoder pre-addestrato; patch grandi (1024 × 1024 = circa 1 km a 1 m) e dilatazioni fino a 9 per avere contesto lungo sui tornanti; testa ausiliaria binaria "strada sì/no" accanto al compito principale; Lovász sulla testa binaria per ottimizzare l'IoU; TTA con flip e finestra scorrevole con circa 60% di sovrapposizione; normalizzazione in [0, 1] senza standardizzazione.
  - Scartare per ora: la segmentazione multi-classe per tipo di strada (Road2 F1 0,282); prima un problema binario strada/sentiero contro sfondo, la classe tipologica dopo.
  - I loro F1 per pixel (0,7–0,8) senza buffer sono più severi di una recall vettoriale con tolleranza: non confrontarli direttamente con il target 0,98.
  - A 9,3 M parametri la rete sta comodamente nei 12 GB della RTX 5070 Ti anche con patch 1024², batch piccolo (stima mia, da verificare).

---

### Trier, Salberg, Larsen, Nyvoll 2022 — Strade forestali in Sentinel-2 con U-Net (NLDL 2022)

- **Riferimento**: Ø. D. Trier, A.-B. Salberg, R. Larsen, O. T. Nyvoll, "Detection of forest roads in Sentinel-2 images using U-Net", Proceedings of the Northern Lights Deep Learning Workshop vol. 3 (2022), DOI 10.7557/18.6246, CC BY 4.0; letto il PDF da `septentrio.uit.no/index.php/nldl/article/download/6246/6501`.
- **Lettura**: completa.
- **Cosa fanno**: U-Net su Sentinel-2 a 10 m per trovare strade nuove (costruite dopo l'1/1/2018) e aggiornare la mappa della natura indisturbata (aree a più di 1 km da interventi pesanti) della Norvegia; servizio pilota serale automatico; prova con una banda aggiuntiva di gradiente del DTM.
- **Dati**: Sentinel-2, 10 bande (quelle a 10 m e 20 m, le 20 m ricampionate a 10 m bilineare); tutte le 13 bande solo per le nuvole (s2cloudless). Tile 100 × 100 km UTM 32N: 38 tile, 34 di training, 2 di validazione (T32VNL Larvik, T32VNN Lillehammer, viste durante il training), **2 di test mai viste** (T32VMN Geilo, T32VNR Trondheim). Immagini senza nuvole 2018–2021, aprile–settembre, niente neve sulle strade; in servizio solo scene con meno del 10% di nuvole. Etichette: vettori di tutte le strade note (dalle autostrade alle strade forestali) rasterizzati a 10 m; maschera acqua da vettori di laghi e fiumi. DTM nazionale 10 m da lidar (2–10 punti/m²), gradiente di Sobel G scalato come `0.1·√G`.
- **Metodo**: U-Net modificata a 10 bande; blocchi di 3 conv + BN + ReLU; 4 blocchi con downsampling ×2 nell'encoder, 4 con upsampling bilineare ×2 nel decoder, skip connection, un blocco finale, softmax a 2 classi. Cross-entropy; Adam, LR 0,004; 5000 batch da 16 crop casuali 256 × 256 (da immagini casuali); validazione ogni 500 iterazioni, checkpoint con la migliore accuratezza di validazione. Variante "buffer": i pixel adiacenti alle strade sono ignorati in training e test. Post-elaborazione: rimozione dei pixel coperti dalla maschera acqua, analisi di buffer sulla mappa della natura indisturbata. Niente scheletrizzazione né vettorizzazione.
- **Risultati** (pixel, tile di test, tutte le date):

  | Configurazione | Vero positivo | Falso negativo | Falso positivo |
  |---|---|---|---|
  | strade senza tractor roads | 41% | 59% | 6% |
  | strade incluse tractor roads | 29% | 71% | 7% |
  | strade, buffer ignorato, 10 bande | 57% | 43% | 18% |
  | strade, buffer ignorato, 10 bande + gradiente DTM | 61% | 39% | 23% |

  Migliore accuratezza di validazione 70% (senza tractor roads) e 66% (con). La recall varia molto con la data: T32VMN dal 53% al 64%, T32VNR dal 48% al 68% (tabella 2, 10 bande); con gradiente il massimo per data è 72% (T32VNR 2020-06-19). Strade nuove: solo ispezione visiva (non c'è verità a terra); falsi positivi tipici: torrenti non presenti nella maschera acqua, creste di roccia nuda, frane di ghiaia lunghe e strette; il gradiente del DTM ne elimina una parte ma non tutti (non contati "per mancanza di tempo").
- **Limiti**: dichiarati: falsi negativi alti (43% senza gradiente, 39% con) su singola data; falsi positivi 18–23% che includono anche strade nuove vere; il ritardo nuvole/neve. Miei: la recall è per pixel a 10 m, quindi una strada forestale larga 3–4 m occupa un pixel misto; i tractor roads sono quasi invisibili; nessuna metrica per segmento vettoriale.
- **Proposte degli autori**: accumulare le predizioni su più date; maschera di drenaggio/reticolo idrografico dal DTM dettagliato (HAND); maschera di creste e di zone troppo ripide; scartare predizioni a più di 200 m dalla rete stradale esistente; pesi di classe nella cross-entropy, campionamento che garantisca segmenti di strada in ogni mini-batch, adaptive class weighting loss, clDice.
- **Per Morloquio**:
  - Sentinel-2 da solo non serve per i sentieri: recall 29% con i tractor roads già a 10 m. Usarlo al più come canale ausiliario a bassa risoluzione (tagli recenti, piste nuove in aree aperte), come composito multi-data (massimo delle probabilità su più date estive), mai come sorgente principale.
  - Adottare l'**ignore-buffer**: pixel vicini all'asse OSM esclusi dalla loss (ignore index). Gestisce l'errore di posizione e di larghezza delle etichette OSM; qui la recall è salita dal 41% al 57% (FP dal 6% al 18%), ma la fascia era esclusa anche dal test, quindi parte del guadagno viene dalla valutazione più tollerante.
  - Il gradiente del DTM aggiunto agli spettrali ha aiutato (+4 punti di recall e meno falsi positivi su torrenti e frane): conferma che le derivate del DTM sono il segnale principale.
  - In Calabria/Sicilia fiumare e torrenti saranno il falso positivo principale: serve un canale o una maschera di drenaggio (flow accumulation / HAND dal DTM 1 m) come **feature**, non come cancellazione rigida, per non perdere le piste che corrono in alveo.
  - Il filtro "più di 200 m dalla rete esistente" va usato solo come punteggio di ranking: con target di recall 0,98 una regola rigida taglierebbe sentieri isolati veri.
  - Campionamento con strade garantite in ogni batch e loss topologica (clDice) da tenere nell'elenco.

---

### Wang, Cornelissen, Kissling 2025 — Sentieri di ungulati da ALS nazionale (Frontiers in Remote Sensing)

- **Riferimento**: J. Wang, P. Cornelissen, W. D. Kissling, "A workflow for extracting ungulate trails in wetlands using 3D point clouds obtained from airborne laser scanning", Front. Remote Sens. 6:1599128 (2025), DOI 10.3389/frsen.2025.1599128, CC BY; letto il PDF del sito Frontiers. Dati: Zenodo 10.5281/zenodo.13889768.
- **Lettura**: completa (materiale supplementare non letto).
- **Cosa fanno**: workflow **geometrico, senza deep learning**, per estrarre i sentieri del cervo nobile nei canneti: filtraggio dei punti (quasi) a terra, DTM a 10 cm, residuo tra due iterazioni di smoothing laplaciano, soglia statistica, clustering e tensor voting 3D sparso per tenere le strutture lineari.
- **Dati**: AHN4 (Paesi Bassi), aprile 2020, circa 25 punti/m²; 7 tile da 5 × 6,25 km, 6.243.044.410 punti, 42,3 GB LAZ. Oostvaardersplassen: palude di 36 km² + zona di margine di 20 km². Dopo il clipping 490.234.336 punti, dopo l'outlier removal 470.279.992. Etichette di verifica: 100 plot casuali di 30 × 30 m (50 solo cervo, 50 cervo + oche), segmentati a mano in CloudCompare (trail = 1, non-trail = 0), riportati su griglia a 10 cm: 9.000.000 celle etichettate.
- **Metodo** (parametri della tabella 1):
  - Pre-elaborazione: retiling a 50 m; clip con poligono (winding number); SOR con k = 6, α = 2,0.
  - Filtraggio near-terrain iterativo su griglia 3D: cella iniziale 15 m dimezzata fino a 0,1 m, soglia di altezza 0,5 m, più un controllo di pendenza locale. Risultato: 261.971.994 punti near-terrain e 208.307.998 di vegetazione.
  - DTM IDW a 0,1 m con raggio 0,3 m.
  - Estrazione: 2 iterazioni di smoothing laplaciano con kernel di 49 punti; `ΔH = DTM(t) − DTM(t+1)`; trail se `ΔH ≤ μ_ΔH − κ·σ_ΔH` con κ = 0,7; SOR con σ = 1,3; clustering per connettività (KD-tree + union-find) con raggio 0,3 m; rapporto larghezza/lunghezza del box ≤ 0,4 (forma allungata); tensor voting con R_tensor = 1,0 m, minimo 8 punti, soglia di saliency 0,4.
  - Sensibilità: 12 parametri × 10 valori ciascuno, sulle stesse 50 parcelle "solo cervo".
- **Risultati**: accuratezza globale per plot `Acc = 0,93 ± 0,05` (range 0,75–0,98) nelle 50 parcelle solo cervo; `0,90 ± 0,06` (range 0,73–0,98) nelle 50 cervo + oche. Esempio di un plot: (71016 + 11598)/90000 = 0,92. Trail estratti 1,88 km², pari all'11,6% dei 16,16 km² di palude studiata. Parametri più influenti: soglia di altezza, numero di iterazioni (più di 2 peggiora), κ, R_clustering, R_tensor. Precision, recall e F1 **non riportati**.
- **Limiti**: dichiarati: trasferibilità non provata (un sito, un rilievo); parametri da riottimizzare altrove (minimum grid size, height threshold, kernel, R_tensor legato alla larghezza del sentiero); servono almeno 32 GB di RAM e 40 GB di disco; codice C++; terreno piatto; trail poco profondi o di larghezza variabile possono non superare la soglia. Miei: i parametri sono scelti massimizzando l'accuratezza sulle **stesse 50 parcelle usate per valutare**, quindi 0,93 è ottimistico; l'accuratezza globale con circa l'11% di classe positiva non misura la recall; nella formula (18) del testo il denominatore è scritto "TN + TN + FP + FN" (refuso).
- **Per Morloquio**:
  - Non applicabile così com'è: richiede la nuvola di punti (filtraggio a terra, SOR, tensor voting 3D) che non abbiamo, e lavora su terreno piatto a 10 cm.
  - Idea da portare sul raster DTM 1 m: canale "**residuo di smoothing**" (DTM meno DTM levigato, simile a un local relief model) con soglia relativa μ − κσ per tile. È economico e mette in evidenza i solchi lineari; in montagna evidenzierà però anche impluvi e fossi, quindi va dato alla rete come feature, non usato come classificatore.
  - Il filtro di forma (rapporto ≤ 0,4) e il tensor voting per collegare segmenti interrotti sono idee di post-elaborazione per ricucire i gap sotto chioma (in 2D: tensor voting o chiusura direzionale sullo scheletro).
  - Con sentieri larghi 1–2 m su DTM a 1 m il segnale di depressione è di 1–2 pixel: la sola geometria del DTM resterà debole, il che è coerente con la recall 0,185 dello stadio A.

---

### Bhatnagar et al. 2022 — Solchi di ruote nei tagli forestali con ResNet50+UNet (Forestry)

- **Riferimento**: S. Bhatnagar, S. Puliti, B. Talbot, J. Heppelmann, J. Breidenbach, R. Astrup, "Mapping wheel-ruts from timber harvesting operations using deep learning techniques in drone imagery", Forestry (2022), DOI 10.1093/forestry/cpac023. Letto il **preprint Zenodo** (`zenodo.org/records/5208232/files/wheel_rut_Zenodo.pdf`, 39 pagine con appendici); README del repository `SmartForest-no/wheelRuts_semanticSegmentation` e default di `keras_segmentation/train.py`.
- **Lettura**: completa per il preprint; la versione pubblicata su Forestry non è stata confrontata (può differire).
- **Cosa fanno**: segmentazione binaria dei solchi delle macchine forestali in ortomosaici RGB da drone subito dopo il taglio raso, validata in leave-one-site-out su 20 cantieri.
- **Dati**: 20 siti nel sud-est della Norvegia, 0,5–21,5 ha, voli 2016–2019. DJI Phantom 2 + GoPro Hero 4 (12 MP) a circa 50 m e DJI Phantom 4 Pro (20 MP) a 60–100 m; sovrapposizione 80/70; 5–7 GCP RTK; Agisoft Photoscan; DEM con RMSE tipico < 5 cm; GSD 1–3 cm tipico (tabella 1: da 1,0 a 7,1 cm). Più di 160 km di solchi digitalizzati a mano come polilinee, in 3 classi di severità (lieve, moderato, grave; il DEM è stato usato solo per aiutare l'annotazione). Etichetta binaria: **buffer di 3 m** attorno alle linee, poi rasterizzato; aree non forestali mascherate. Tile png 210 × 210 px: 1909 immagini, 12 scartate perché sfocate, quindi 1897 per il training. Lunghezze: lieve 63,6%, moderato 24,6%, grave 11,8% dei tracciati.
- **Metodo**: encoder ResNet50 con pesi ImageNet, decoder UNet da zero (transpose conv + skip), softmax; aumenti flip e rotazione; cross-entropy; Adam; regolarizzazione L2; LR iniziale 0,01 (scelto a tentativi); batch 20; 20 epoche (circa 50 ore per sito, circa 45 giorni per le 20 fold) su workstation Precision 5820, 16 GB. Solo RGB in ingresso. Post-elaborazione: apertura binaria con disco di raggio 2 m (toglie le regioni sotto circa 5 m²), poi erosione in scala di grigi con disco di 20 cm. Il repository (Keras, fork di `keras_segmentation`) usa tile di 20 m con buffer di 2 m in inferenza; modello di default `singleTrack_allData_49epochs` addestrato 50 epoche su tutti i dati; pesi su un servizio di archiviazione cloud; dataset su Zenodo record 5746878; nessuna licenza dichiarata nel repository (API GitHub: `license: None`).
- **Risultati** (leave-one-out su 20 siti, pixel, dopo post-elaborazione): F1 solchi da 0,45 a 0,83, media 0,67; OA di test media 73% (matrice complessiva 73,1%); matrice complessiva: **producer's accuracy (recall) solchi 66%**, user's accuracy (precision) 68%, F1 0,67; non-solchi PA 78%, UA 77%. Migliore sito O (F1 0,83), peggiore N (F1 0,45, GSD 7,1 cm, solo 20% dei solchi rilevati). Siti a circa 2 cm (7 su 20): 71% in media; siti a 1 cm (5 su 20): 66%. Per severità: circa 74% gravi, 67% moderati, 62% lievi. Modello solo p4pro su sito P: training accuracy 99,3% a 20 epoche e 99,8% a 50, F1 +0,04. La post-elaborazione alza l'OA di test in media dell'1,2% ma aumenta i FN.
- **Limiti**: dichiarati: lievi solchi spettralmente simili allo sfondo; rumore da residui, tronchi e ombre (siti I ed E: OA 66% e 54%); sensibilità alla risoluzione. Miei: la metrica per severità è chiamata "UA" ma, dalle matrici in appendice S.2, è predetto-solco su riferimento della classe, cioè una **recall per classe** (es. sito A lieve: 1207829/(1207829+466467) = 72,1%); il buffer di 3 m attorno a solchi larghi meno di 1 m rende le etichette molto più larghe dell'oggetto e penalizza sia la precision sia la recall per pixel; in tabella 1 le latitudini (82–86°N) sono evidentemente errate.
- **Per Morloquio**:
  - Adottare: **validazione leave-one-site-out** o a blocchi spaziali (per noi: tile intere tenute fuori, per zona); riportare la recall per classe di "visibilità" (strada, pista, sentiero), come qui per severità.
  - Lezione sulle etichette da linee: la larghezza del buffer deve avvicinarsi alla larghezza reale (sentiero 1–2 m, a 1 m/pixel = 1–3 pixel); meglio un buffer stretto più una zona ignorata attorno (vedi Trier 2022) che un buffer largo.
  - Evitare l'apertura morfologica aggressiva (disco 2 m): aumenta i FN, il contrario di quello che serve per recall 0,98. Filtrare i frammenti per lunghezza dopo la scheletrizzazione, non per area prima.
  - La caduta a GSD 7,1 cm mostra che addestrare e predire alla stessa risoluzione conta: tenere tutto a 1 m nativo PST, senza ricampionamenti misti.
  - Encoder ResNet50 ImageNet su dati piccoli funziona: ok come baseline anche per canali DTM/DSM (con adattamento del primo strato).

---

### Yamato et al. 2021 — Tracce di alimentazione dei dugonghi da UAV con U-Net (PLOS ONE)

- **Riferimento**: C. Yamato, K. Ichikawa, N. Arai, K. Tanaka, T. Nishiyama, K. Kittiwattanawong, "Deep neural networks based automated extraction of dugong feeding trails from UAV images in the intertidal seagrass beds", PLoS ONE 16(8): e0255586 (2021), DOI 10.1371/journal.pone.0255586, CC BY; letto il PDF PLOS (24 pagine). La pagina PMC8363023 era troncata a 40k caratteri, quindi ho usato il PDF.
- **Lettura**: completa.
- **Cosa fanno**: U-Net per estrarre le tracce lasciate dai dugonghi su praterie di fanerogame intertidali da ortofoto UAV; seconda U-Net su immagini differenza tra giorni per le tracce nuove; stima della direzione delle tracce.
- **Dati**: isola di Talibong (Trang, Thailandia), 12–27 settembre 2019; DJI Phantom 4 Pro V2; GSD 1,0 cm a circa 37,7 m (12–13 settembre) e 0,5 cm a 18,9 m (15–27 settembre); circa 750 foto al giorno; sovrapposizione 80/70; Metashape (SfM-MVS); tutto ricampionato a 0,47 cm/pixel. Area aerea circa 250 × 150 m (3,6 ha); due quadrati di verità a terra 30 × 30 m (A e B) misurati a ±0,5 cm. Training del Model_1: ortofoto del plot A di 5 giorni più bordi dell'area (tabella 1); un'ortofoto di 6400 × 6400 px divisa in 9313 blocchi 256 × 256 con **75% di sovrapposizione** (192 px). Il plot B è stato escluso perché la copertura di fanerogame era troppo bassa per riconoscere le tracce. Larghezza media delle tracce 13,00 cm (n = 1352).
- **Metodo**: U-Net con una coppia encoder/decoder in più; encoder con conv strided (stride 2) + BN + leaky ReLU; decoder con deconvoluzione stride 2 + BN + ReLU; uscita sigmoide; ingresso 256 × 256. Aumenti: rotazioni (−90, 0, 90, 180), flip, gamma 0,6–1,2, contrasto 0,8–1,4. Split 50% training / 50% validazione+test. Adam, LR iniziale 0,01. Loss, batch ed epoche non riportati. Inferenza: blocchi 256 con 50% di sovrapposizione (4 predizioni per pixel), integrazione e binarizzazione con soglia di luminanza 125, scelta a vista su una curva precision-recall (soglie da 100 a 180, passo 10) di un piccolo sottoinsieme (plot A, 15 settembre). Model_2 (tracce nuove): differenza tra l'uscita di Model_1 di due giorni dopo correzione di posizione con template matching (correlazione normalizzata, blocchi 3200 × 3200 e poi 256 × 256, spostamento massimo 20% del blocco, fino a 3 m); RMSE delle ortofoto 1,42 ± 0,29 m (n = 10); 18.000 immagini (3600 per coppia, 5 coppie); stessa rete; apertura morfologica finale.
- **Valutazione**: F1, precision e recall per pixel con **buffer di tolleranza**: tutti i pixel predetti entro la dilatazione dell'annotazione contano come TP. Larghezza del buffer scelta sul "ginocchio" della curva F1 contro larghezza: 7 pixel = 3,25 cm, il 25% della larghezza media.
- **Risultati**:

  | GSD | Precision | Recall | F1 |
  |---|---|---|---|
  | 1 cm/pixel | 96,8% | 83,2% | 89,5% |
  | 0,5 cm/pixel | 94,4% | 81,8% | 87,7% |

  Model_2 F1 61,9% (correzione manuale ancora necessaria); area media delle tracce nuove 187,8 m² al giorno (n = 9); 63,9% delle tracce con direzione tra 112,5° e 157,5°, 17,7% a 22,5°. Il volo a 1 cm copre 3,81 volte l'area di quello a 0,5 cm nello stesso tempo, a parità di prestazioni.
- **Limiti**: dichiarati: le tracce vecchie (oltre circa 10 giorni) sono difficili anche per l'annotatore; errori di Model_2 cumulati con quelli di Model_1 e delle ortofoto (GCP con GPS palmare). Miei: lo split 50/50 è casuale su blocchi con 75% di sovrapposizione presi dalle stesse ortofoto, quindi training e test condividono pixel (**leakage spaziale**) e i numeri sono ottimistici; la soglia è scelta su dati di training; il buffer conta come TP solo le predizioni che cadono nella fascia, quindi rilassa soprattutto la precision; un solo sito di 3,6 ha.
- **Per Morloquio**:
  - Adottare il metodo di scelta della tolleranza: curva della metrica in funzione della larghezza del buffer e scelta al ginocchio, motivata dall'errore di annotazione. Per noi l'errore di posizione di OSM e delle tracce GPS è di metri, quindi la tolleranza sarà di alcuni pixel a 1 m; va dichiarata e applicata in modo simmetrico (buffer sulla predizione per la recall, buffer sull'etichetta per la precision).
  - Adottare la scelta della soglia dalla curva precision-recall, ma calcolata in modo automatico su tile di validazione separate, puntando alla soglia minima che dà recall ≥ 0,98 (non a vista).
  - Inferenza con sovrapposizione del 50% e media di 4 predizioni per pixel: coerente con la finestra scorrevole di DDCM-Net.
  - Da non ripetere: split casuale su tile sovrapposte. Per il nostro target servono tile di test spazialmente separate (blocchi interi, con un margine di almeno una patch).

---

### Abdi, Uusitalo, Kivinen 2022 — Logging trail sotto chioma con U-Net e ALS ad alta densità (Remote Sensing) [aggiunto]

- **Riferimento**: O. Abdi, J. Uusitalo, V.-P. Kivinen, "Logging Trail Segmentation via a Novel U-Net Convolutional Neural Network and High-Density Laser Scanning Data", Remote Sensing 14(2):349 (2022), DOI 10.3390/rs14020349, CC BY 4.0. MDPI bloccato da Akamai anche con `--stealth`; letta la ristampa identica dal repository HELDA dell'Università di Helsinki (hdl 10138/341614, bitstream `remotesensing.pdf`).
- **Lettura**: completa (il PDF contiene due volte il testo, versione impaginata e versione di revisione; ho deduplicato le frasi).
- **Cosa fanno**: U-Net binaria per ritrovare i logging trail (corridoi di harvester/forwarder larghi 4–5 m, a 20–25 m l'uno dall'altro) sotto chioma in pinete gestite del sud della Finlandia; confronto di 4 ingressi monocanale derivati dall'ALS: CHM, DSM, DEMavg, DEMmin.
- **Dati**: ALS NLS Finlandia, almeno 5 punti/m² (spaziatura media circa 40 cm), errore altimetrico medio < 10 cm, planimetrico < 45 cm (licenza acquistata nel 2020). 44 tile da 1 × 1 km (Parkano e Ikaalinen). CHM a 0,5 m; DEMmin (minimo per cella), DEMavg (media), DSM (massimo) per binning; tutti normalizzati min–max 0–255 e filtrati con mediana 3 × 3. Etichette: 336 km di logging trail digitalizzati da ortofoto NIR, CHM e profili della nuvola; il DTM serviva per escludere fossi e strade; buffer di 2 m dall'asse (larghezza 4 m). 1888 patch 256 × 256 (75% training, 25% validazione); escluse le patch delle aree di test.
- **Metodo**: U-Net a 4 livelli, ciascuno con 2 conv 3 × 3 + ReLU + BN (padding same), max-pooling 2 × 2, filtri da 16 a 128, bottleneck 256, decoder con conv trasposte stride 2 e concatenazioni, uscita 1 × 1 sigmoide; ingresso (256, 256, 1). Iperparametri da Hyperband (Keras Tuner): Adam (β1 = 0,9, β2 = 0,999, ε = 1·10⁻⁷), LR 0,0008, batch 32, dropout 0,2–0,4, inizializzazione HeNormal; minimo 100 epoche con early stopping; cross-entropy. Keras/TensorFlow su Quadro RTX 4000 8 GB. Nessuna post-elaborazione.
- **Valutazione**: indipendente e sul campo: 21 percorsi, **390 campioni** (segmenti di trail di circa 30 m e "edge" tra due trail come negativi), rilevati con GNSS Trimble GeoXT corretto in post-elaborazione (< 50 cm). Matrice di confusione per campione, non per pixel.
- **Risultati** (tabella 1, tre foreste di test):

  | Ingresso | Kappa | OA | IoU | Recall |
  |---|---|---|---|---|
  | CHM | 0,734 | 0,867 | 0,782 | 0,908 |
  | **DSM** | **0,846** | **0,923** | **0,867** | **0,959** |
  | DEMavg | 0,528 | 0,736 | 0,587 | 0,649 |
  | DEMmin | 0,136 | 0,553 | 0,155 | 0,157 |

  Per stadio del popolamento (DSM): giovani dopo il 1° diradamento k = 0,953, IoU = 0,963; maturi dopo il 2°/3° k = 0,854, IoU = 0,889; maturi prima del 2° diradamento (trail chiusi da rinnovazione) k = 0,684, IoU = 0,686. CHM negli stessi stadi: 0,859/0,893, 0,834/0,876, 0,438/0,505. Le strade forestali sono correttamente "non trail"; falsi positivi: fossi di drenaggio poco profondi e corridoi stretti.
- **Limiti**: dichiarati: rendimento più basso nei popolamenti vecchi non diradati; fossi confusi con trail; dati solo di due comuni; larghezza standard di 4 m assunta senza misure. Miei: l'abstract riporta per DEMavg k = 0,542 e IoU = 0,667, ma questi sono i valori del solo stadio "maturi diradati"; in tabella 1 sono 0,528 e 0,587. La normalizzazione min–max per tile del DSM toglie la quota assoluta ma rende i valori dipendenti dal rilievo di ogni tile. Recall misurata su segmenti di 30 m bilanciati, non su tutta la rete.
- **Per Morloquio**:
  - Risultato chiave: per corridoi sotto chioma il segnale sta nella **chioma** (DSM, CHM: recall 0,959 e 0,908), non nel terreno (DEMmin: 0,157), perché i forwarder finlandesi lasciano poca traccia nel suolo. Per i sentieri calabresi sotto bosco valgono entrambi i casi: mulattiere incassate e terrazzate si vedono nel DTM, sentieri con varco nella chioma si vedono nel DSM. Usare **DTM, DSM first, DSM last e CHM = DSM first − DTM** come canali separati, non solo il DTM.
  - La differenza DSM last − DTM può catturare il sottobosco (ipotesi mia, da provare).
  - Normalizzazione: meglio derivate locali (CHM, rilievo locale, pendenza) che min–max del DSM assoluto per tile, soprattutto con 1000 m di dislivello in un tile montano.
  - Metrica per segmenti di circa 30 m: buon modello per la nostra recall vettoriale contro OSM (frazione di segmenti OSM coperti dalla predizione entro una tolleranza).
  - Anche qui i fossi sono falsi positivi: rinforza l'idea di un canale idrografico dal DTM.
  - Rete minuscola (filtri 16–256), batch 32 a 256², 8 GB: dimensionamento compatibile con la RTX 5070 Ti.

---

### Lidberg et al. 2023 — Fossi di drenaggio in bosco da ALS con deep learning (J. Irrig. Drain. Eng.) [bonus]

- **Riferimento**: W. Lidberg, S. S. Paul, F. Westphal, K. F. Richter, N. Lavesson, R. Melniks, J. Ivanovs, M. Ciesielski, A. Leinonen, A. M. Ågren, "Mapping Drainage Ditches in Forested Landscapes Using Deep Learning and Aerial Laser Scanning", Journal of Irrigation and Drainage Engineering 149(3) (2023), DOI 10.1061/JIDEDH.IRENG-9796, CC BY (open access ibrido). Letto il **PDF pubblicato** (impaginato ASCE, 04022051) dalla copia DiVA dell'Università di Umeå (`umu.diva-portal.org/smash/get/diva2:1721509/FULLTEXT02`) e, in parallelo, la pagina full-text ASCE con Crawl4AI `--stealth`. Letti anche README e codice del repository `williamlidberg/Mapping-drainage-ditches-in-forested-landscapes-using-deep-learning-and-aerial-laser-scanning` (licenza MIT): `train.py`, `utils/loss.py`, `utils/unet.py`. Dati: Mendeley 10.17632/zxkg43jsx8.1.
- **Lettura**: completa. Gli MCC per paese esistono solo come immagine nella figura 3 e non sono leggibili dal testo.
- **Cosa fanno**: segmentazione binaria dei fossi di drenaggio forestali su scala regionale con una rete encoder-decoder a blocchi Xception e un solo canale di ingresso, l'**high-pass median filter (HPMF)** del DEM; test su tile svedesi tenute fuori e su 4 aree di altri paesi del Baltico.
- **Dati**: Svezia, 10 regioni a prevalenza forestale; ALS Leica ALS80-HP-8236 da 2888–3000 m, 1–2 punti/m²; 55 tile da 2,5 × 2,5 km (344 km²). DEM a 0,5 m (TIN gridding, Whitebox 1.4.0); HPMF con kernel di 11 celle (cella centrale meno mediana del vicinato; negativo = depressione). Etichette: fossi digitalizzati come linee da esperti calibrati tra loro (513 ore di digitalizzazione in totale), su hillshade multidirezionale + HPMF, con ortofoto e mappe storiche per i casi dubbi; 1607 km di fossi. Larghezza media da inventario di campo (2188 fossi): 2 m, deviazione standard 1,3 m. **Etichetta "naturale"**: pixel entro 3 m dalla linea **e** con HPMF < −0,075, poi filtro di maggioranza 3 celle per togliere pixel isolati. Il 20% delle tile scelto a caso per il test (n = 11 tile; split per tile). Test aggiuntivi: Svezia (68 km², 20 punti/m²), Finlandia (70 km², 5 punti/m²), Lettonia (25 km², 4 punti/m²), Polonia (44 km², 4 punti/m²), digitalizzati da squadre diverse.
- **Metodo**:
  - Chip 512 × 512; scartati i chip con meno dello 0,1% di pixel fosso (risultano 2367 coppie).
  - Rete encoder-decoder TensorFlow 2.6 con blocchi Xception (convoluzioni separabili) e conv trasposte nel decoder; strato CRF-as-RNN (Zheng et al. 2015) che penalizza discontinuità; loss cross-entropy pesata.
  - Dal codice (default del repository): `XceptionUNet` con profondità 2 (filtri 128 e 256, primo conv 32 con stride 2), softmax a 2 classi; `categorical_crossentropy` con peso 0,1 sulla classe sfondo (`zero_class_weight=0.1`); Adam (LR di default di Keras); `ReduceLROnPlateau` su loss con patience 10 e `min_lr` 0,00001; batch 4; aumenti attivi; 100 epoche di default; checkpoint sulla loss migliore; split casuale 80/20 dei chip nel generatore.
  - Inferenza: probabilità per pixel; soglia non indicata nel testo; 1 km² in 8,6 s su una GeForce GTX 1080 Ti.
- **Risultati** (pixel, tile di test svedesi): TP 1.518.628, FP 616.484, FN 248.361, TN 127.902.095. **Recall 86%**, precision 0,71, accuratezza globale 99%, kappa di Cohen 0,78, MCC 0,78, errore di omissione 14%, di commissione 29%. Paesi esteri: MCC massimo in Polonia e minimo in Lettonia (valori solo in figura). Falsi positivi: soprattutto canali naturali (la rete non distingue torrenti da fossi), più depositi sedimentari; alcuni FP sono probabilmente fossi mancanti nelle etichette estere. Confronto con 13 studi 2008–2020: recall riportate altrove tra 31% e 98%.
- **Limiti**: dichiarati: confusione con torrenti naturali; FP su depositi sedimentari; le metriche estere vanno lette con cautela (digitalizzazione diversa). Miei: valutazione solo per pixel, nessuna metrica per segmento o lunghezza; le etichette con soglia HPMF dipendono dalla risoluzione e dal kernel; **README e articolo non coincidono** (README: DEM 1 m, kernel HPMF di 5 celle, 688 chip, "82%" di fossi e MCC 0,72, altrove "recall of 84%"; articolo: 0,5 m, 11 celle, 2367 chip, recall 86%, MCC 0,78), quindi il codice pubblicato può non riprodurre esattamente il modello dell'articolo; nel codice lo split di validazione è casuale per chip.
- **Per Morloquio** (il lavoro più trasferibile del gruppo):
  - Oggetto molto simile ai nostri sentieri: linee strette (2 ± 1,3 m) sotto chioma, visibili come piccole depressioni nel DTM, etichette vettoriali senza larghezza. Ricetta da copiare quasi intera.
  - **Canale HPMF** (Whitebox `HighPassMedianFilter`, open source) dal DTM 1 m PST, con kernel da tarare (5 celle a 1 m secondo il README). Aggiungere hillshade e pendenza come canali extra, più CHM/DSM (vedi Abdi 2022) per i sentieri che non incidono il terreno.
  - **Etichette "naturali" da OSM**: pixel entro N m dall'asse OSM **e** con HPMF sotto soglia (o con rottura di pendenza, per le mulattiere a mezza costa che non sono depressioni), più filtro di maggioranza. Riduce l'errore di posizione OSM senza allargare la classe positiva. Il resto del buffer diventa zona ignorata nella loss (vedi Trier 2022).
  - Il filtro dei chip con meno dello 0,1% di positivi va adattato: sotto-campionare quei chip invece di scartarli o di usarli come negativi certi, perché con OSM incompleto un chip "vuoto" può contenere sentieri non mappati.
  - Cross-entropy pesata (0,1 sullo sfondo) è una base semplice; per il nostro target va abbinata a una soglia scelta sulla recall.
  - La rete chiama "fosso" qualsiasi canale: noi avremo lo stesso problema con impluvi, fiumare e canali irrigui; serve un canale idrologico (flow accumulation) e magari una classe "canale d'acqua" separata, così la rete impara la differenza invece di sopprimere la classe.
  - Recall per pixel 86% con precision 0,71: anche su un compito più facile (fossi più larghi e diritti dei sentieri) il pixel-level non arriva vicino a 0,98. Il nostro 0,98 è realistico solo come recall **vettoriale con tolleranza** e con fusione di più fonti (tracce GPS pubbliche).
  - Licenza MIT e immagine Docker: si possono riusare codice e struttura del modello; i pesi addestrati su Svezia sono un possibile pre-training per transfer learning (come in Virro et al. 2025).

### Virro et al. 2025 — Fossi da DTM LiDAR con U-Net e transfer learning Svezia → Estonia (Big Earth Data) [bonus, collegato a Lidberg]

- **Riferimento**: H. Virro, A. Kmoch, W. Lidberg, M. Muru, W. T. Chan, D. M. Moges et al., "Detection of drainage ditches from LiDAR DTM using U-Net and transfer learning", Big Earth Data (2025), DOI 10.1080/20964471.2025.2491177 (open access). Letta la pagina full-text Taylor & Francis con Crawl4AI `--stealth`.
- **Lettura**: **parziale**: la pagina è stata troncata a 40k caratteri (dopo l'inizio delle conclusioni) e le tabelle HTML (1, 2, 3, A1, A2) non sono state convertite, quindi mancano precision e recall per uso del suolo (tabella 3). Il PDF di T&F restituisce una pagina HTML (anti-bot); la copia SLU (`pub.epsilon.slu.se/37386/...`) reindirizza al nuovo portale `research.slu.se`.
- **Cosa fanno**: U-Net PyTorch pre-addestrata sul dataset svedese di Lidberg et al. (1360 coppie HPMF-etichetta 500 × 500 da DTM 0,5 m, HPMF normalizzato 0–1) e rifinita su 18 km² di Estonia (due aree di 3 × 3 km; DTM 1 m dell'Estonian Land Board, circa 2,1 punti/m²; etichette con lo stesso protocollo: buffer di 3 m e pixel con "HPMF values below 0.075", soglia attribuita a Lidberg et al., che però scrivono −0,075; il segno o la scala normalizzata non è chiarito; 72 tile da 500 × 500 m).
- **Metodo**: split 80/20 delle coppie; aumenti: rotazioni di 90, 180, 270° e flip orizzontale (1088 → 5440 coppie in pre-training; 58 → 290 in fine-tuning); ricampionamento da 500 × 500 a 512 × 512; ReLU, max-pooling; Adam con LR 0,0001; batch 4; 250 epoche; `BCEWithLogitsLoss`; kernel 3 × 3, 5 × 5, 7 × 7 a confronto; pesi presi all'epoca con la loss di test minima; Tesla V100.
- **Risultati**: pre-training F1 0,786–0,788 con aumenti; loss minima tra 0,014 e 0,016, raggiunta entro le prime 100 epoche (poi overfitting). Fine-tuning: F1 da 0,741 a 0,766; migliore FT3A (kernel 3 × 3, con aumenti) **F1 0,766**; senza fine-tuning, sui dati estoni, F1 da 0,570 a 0,677. Per uso del suolo (soglia 0,5): torbiere F1 0,822; foresta 0,734 (sito A) e 0,752 (B); seminativi 0,719 (A) e 0,779 (B). Abbassando la soglia a 0,1 migliora molto la completezza in foresta, con pochi falsi positivi in più (risultato solo qualitativo). Kernel più grandi non aiutano; gli aumenti sì, ma il pre-training PT3A dura 20 ore, più di 5 volte PT3U.
- **Limiti**: dichiarati: area di fine-tuning piccola (rischio di overfitting); DTM a 1 m non disponibile ovunque; FP lungo le scarpate stradali e sulle tracce di esbosco. Miei: split casuale 80/20 tra tile adiacenti della stessa area, senza area di test indipendente; i pesi sono scelti sulla loss del test set stesso (selezione ottimistica); il "test" su tutte le 72 tile include le tile di training.
- **Per Morloquio**:
  - Conferma che il **transfer learning** da un dataset ricco funziona con pochi km² locali (F1 da circa 0,6 a 0,766). Per noi: pre-addestrare su aree con OSM più completo (o sul dataset svedese dei fossi) e rifinire su Calabria/Sicilia.
  - In bosco la soglia 0,5 è conservativa: per il target di recall serve una soglia bassa (0,1 o meno), da scegliere su tile tenute fuori.
  - Kernel 3 × 3 sufficienti, aumenti geometrici utili.
  - Le tracce di esbosco sono FP per un modello di fossi ma sono veri positivi per noi: un modello dei fossi pre-addestrato "vede" già i nostri oggetti.

---

## Indicazioni trasversali per Morloquio (dai lavori letti)

1. **Canali di ingresso**: DTM in forma di derivate locali (HPMF o residuo di smoothing, pendenza, hillshade, gradiente) **più** chioma (CHM = DSM first − DTM, DSM). Lidberg e DDCM mostrano che il DTM filtrato basta per oggetti incisi nel terreno; Abdi mostra che i corridoi sotto chioma si vedono quasi solo nel DSM/CHM (recall 0,959 contro 0,157 del DEMmin). Sentinel-2 al più come canale ausiliario.
2. **Etichette da OSM**: buffer stretto attorno all'asse intersecato con un criterio morfometrico (HPMF sotto soglia, come Lidberg), una fascia ignorata nella loss attorno (Trier: recall dal 41% al 57%, con FP dal 6% al 18%; la fascia era esclusa anche dal test, quindi parte del guadagno è di valutazione), e nessun chip "vuoto" usato come negativo certo, perché OSM è incompleto.
3. **Loss e campionamento**: cross-entropy pesata (sfondo 0,1) o con median frequency balancing, più un termine IoU/Lovász o clDice sulla testa binaria; patch con strade garantite in ogni batch.
4. **Valutazione**: tile o regioni intere tenute fuori (leave-one-site-out come Bhatnagar, tile di test come Trier e Lidberg). Mai split casuali tra tile sovrapposte (Yamato, Virro). Tolleranza di posizione scelta con la curva "metrica contro larghezza del buffer" (Yamato), recall misurata su segmenti di lunghezza fissa (Abdi, circa 30 m) contro OSM.
5. **Soglia**: scelta automaticamente sulla curva precision-recall delle tile di validazione per arrivare a recall ≥ 0,98; in bosco serve una soglia bassa (Virro: 0,1 molto più completo di 0,5).
6. **Post-elaborazione**: niente aperture morfologiche aggressive (Bhatnagar: aumentano i FN); ricucitura dei gap con un criterio di continuità lineare (tensor voting di Wang, CRF di Lidberg) e filtro per lunghezza dopo la scheletrizzazione.
7. **Falsi positivi attesi**: torrenti, fiumare, fossi, impluvi, creste e frane (Trier, Lidberg, Abdi). Rimedio: canale idrologico dal DTM e, se possibile, una classe "canale d'acqua" esplicita.
8. **Realismo del target**: nessun lavoro supera una recall per pixel di 0,86 con precision accettabile su oggetti lineari stretti sotto bosco; le recall vicine a 0,96 sono per segmenti o con tolleranza. Il target 0,98 contro OSM va definito come recall vettoriale con buffer e ottenuto probabilmente fondendo rete, tracce GPS e geometria.

## Articoli non letti o letti in parte

| Articolo | Stato | Motivo |
|---|---|---|
| Salberg, Trier, Kampffmeyer 2017 (SCIA) | **non letto** | `munin.uit.no` è un CNAME verso `wr3.uit.no`, NXDOMAIN anche da resolver DNS pubblici; mirror `ninum.uit.no` non risolve; NVA/Sikt espone `article.pdf` ma l'API di download anonima risponde `Forbidden`; CORE dietro Cloudflare; Wayback offline e senza snapshot; editore Springer a pagamento (non aggirato). Usati solo abstract, pagina progetto NR e dati descritti da Liu et al. 2019. |
| Bhatnagar et al. 2022 (Forestry) | letto il preprint Zenodo | Versione pubblicata su academic.oup.com bloccata da Cloudflare (JS challenge) anche con `--stealth`: non confrontata, può differire. |
| Lidberg et al. 2023 (JIDE) | completo | Letto il PDF pubblicato da DiVA (download lento, circa 8 KB/s, completato al terzo tentativo); gli MCC per paese sono solo nella figura 3 (immagine). |
| Virro et al. 2025 (Big Earth Data) | parziale | Pagina T&F troncata a 40k caratteri, tabelle non convertite (manca la tabella 3 con precision/recall); PDF bloccato da anti-bot; copia SLU spostata. |
| Wang et al. 2025 (Frontiers) | completo, senza supplementi | Materiale supplementare (figure A1–A5) non letto. |
| McDermid, Terentieva, Chan 2025, "Mapping Trails and Tracks in the Boreal Forest Using LiDAR and CNN", Remote Sensing 17(9):1539 | **non letto: bloccato** | MDPI (Akamai) bloccato anche con `--stealth`; preprint su preprints.org bloccato (Akamai "Access Denied"). Dall'abstract (OpenAlex): DTM 10 cm da drone contro 50 cm da aereo, F1 74 ± 6% e 77 ± 9%, 2829 km di tracce su 59 km², Canada boreale. Da recuperare: è il lavoro più vicino ai sentieri. |
| Abdi et al. 2022 (Remote Sensing) | completo | MDPI bloccato, letta la ristampa identica di HELDA. |
