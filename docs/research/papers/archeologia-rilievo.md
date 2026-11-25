# Hollow ways, CarcassonNet, auto-supervisione su DTM, Local Relief Model

Note di lettura integrale (2026-11-26), base dell'ADR 0004. Pagine lette con Crawl4AI, PDF scaricati ed estratti
con pypdf. Sintesi e ricetta in `../2026-11-26-ml-strade-da-lidar.md`.

Convenzione: i numeri tra virgolette o nelle tabelle sono copiati dal testo; quelli marcati "(calcolo mio)" o "(proposta)" non vengono dagli autori.

---

### Verschoof-van der Vaart & Landauer 2021 — CarcassonNet: rilevamento e tracciamento di hollow roads nel LiDAR olandese

- **Riferimento**: W.B. Verschoof-van der Vaart, J. Landauer, "Using CarcassonNet to automatically detect and trace hollow roads in LiDAR data from the Netherlands", *Journal of Cultural Heritage* 47 (2021) 143–154 (online 2 nov 2020; il repository di Leiden riporta "2020, Vol. 47"). DOI 10.1016/j.culher.2020.10.009, CC BY 4.0. Letto il PDF dell'editore da Leiden (hdl 1887/3203910, `/download/1887/3203910/1`) e, per confronto, il capitolo 5 della tesi (hdl 1887/3256824), che ha lo stesso testo.
- **Lettura**: completa (testo, tabelle e bibliografia). Le figure non le ho viste, ma tutte le didascalie sono lette.
- **Cosa fanno**: rilevano le hollow roads (carrarecce medievali e post-medievali incassate, spesso a fasci paralleli) nel DTM. La CNN non segmenta: classifica tasselli 64×64 ("hollow road" / "empty"). Poi i tasselli positivi diventano poligoni in GIS e, con la scheletrizzazione, linee. L'idea "Carcassonne": i tratti brevi di strada si somigliano molto più delle strade intere.
- **Dati**:
  - Veluwe (NL): tre sottoaree (Speuld, Terlet, De Ginkel), "in total 93.75 km2". Il capitolo 6 della tesi dà invece 97.25 km²: incoerenza interna.
  - LiDAR AHN-2, aprile 2010, RIEGL LMS-Q680i full-waveform, 600 m di quota di volo, 100 000 Hz, "6−10" pt/m², interpolazione "moving planes", DTM 0,5 m. Dati liberi da PDOK.
  - Etichette: un solo annotatore (il primo autore) ha disegnato in QGIS poligoni a mano su LRM (Hesse 2010) e Openness, calcolati con RVT 1.3.
  - Tasselli: 393 586 in totale, di cui 10 185 positivi (2,6%) e 383 401 negativi (97,4%).
- **Metodo**:
  - Pre-elaborazione: "Fill nodata" in QGIS. Due dataset alternativi: hillshade RVT 1.3 oppure DTM grezzo, entrambi in Float32.
  - Tasselli 64×64 px (32 m a 0,5 m), sovrapposizione di 32 px su tutti i lati. È "the smallest size that yielded useful results".
  - Normalizzazione: "subtraction of the central pixel value so that each snippet has pixel (or grayscale) values between 0 and 255". Come si arrivi all'intervallo 0–255 non è specificato.
  - Etichetta del tassello dalla percentuale di sovrapposizione con i poligoni: positivo se ≥95%, negativo se ≤5%, scartato se intermedio ("data pruning").
  - Bilanciamento: eliminato circa il 98% dei negativi, positivi duplicati a caso. Risultato: 16 000 positivi + 16 353 negativi = 32 353, divisi 80/20 in addestramento (circa 25 900) e validazione (circa 6470).
  - Rete: ResNet-34 (fast.ai su PyTorch), pre-addestrata su ImageNet. "Progressive resizing" in 3 fasi:
    - 5 epoche a 64×64, tutti i layer congelati tranne gli ultimi completamente connessi;
    - 4 epoche con input ingrandito a 128×128, ancora congelati;
    - 3 epoche a 128×128 con la rete scongelata.
  - Ottimizzazione: Adam, discriminative layer training. Aumenti: solo flip orizzontale e verticale. Learning rate e batch size non riportati. GPU GTX 1060.
  - Soglia di confidenza "empirically set to 0.80".
  - Location-Based Ranking (LBR): mappa a 2 livelli. Rango 2 = suoli plaggen, aree agricole moderne, aree disturbate, aree edificate e strade (NWB); rango 1 = il resto. Le detezioni in rango 2 vengono ignorate. Nella validazione su Speuld il 98,5% dell'area a hollow road (2 237 170 m² su 2 270 730) cade in rango 1.
  - Post-elaborazione: in QGIS raster→vettore (Tileindex, Dissolve, Multipart to singlepart), poi scheletro con `skimage` skeletonize, grafo con `sknw` (Python 3.7) e infine Buffer + Dissolve per ottenere le linee.
  - Feedback loop: i falsi positivi vengono rivisti a mano e quelli che sono strade vere non etichettate ("new positives") entrano nel training. Ripetuto 2 volte, finché non ne emergono di nuovi.
- **Risultati**: metriche ad area (m² di TP/FP/TN/FN), soglia 0,80, Tabella 5:

  | Dati | LBR | R | P | F1 | ACC | MCC |
  |---|---|---|---|---|---|---|
  | HS | no | 0.511 | 0.360 | 0.423 | 0.897 | 0.375 |
  | DTM | no | 0.655 | 0.376 | 0.478 | 0.895 | 0.444 |
  | HS | sì | 0.509 | 0.385 | 0.438 | 0.904 | 0.391 |
  | DTM | sì | 0.651 | 0.407 | 0.501 | 0.904 | 0.466 |

  - Il DTM batte l'hillshade di circa 6 punti; gli autori lo attribuiscono alla perdita d'informazione nella visualizzazione (Shannon). LBR aggiunge 1–2 punti.
  - Feedback loop: la superficie etichettata passa da 7 235 120 a 7 632 390 m², cioè +397 270 m² (+5,49%; per sottoarea 4,86 / 6,04 / 5,45%). Due iterazioni migliorano l'MCC "by circa 0.2".
  - Abbassare la soglia di sovrapposizione a 25/50/75% peggiora il modello e non aiuta le strade sottili.
  - Confronto con metodi per strade moderne (F1 da 0,67 a 0,96), tra cui Ferraz et al. 2016 (strade forestali da LiDAR, F1 0.849).
- **Limiti dichiarati**:
  - Le strade singole, molto più strette del tassello di 64 px, vengono perse o rilevate solo in parte.
  - Sui fasci larghi lo scheletro genera rami perpendicolari al tracciato.
  - Le metriche di classificazione non misurano la qualità della rete stradale (connessione).
  - Falsi positivi da "objects of confusion": forme geologiche, altri oggetti archeologici, opere forestali.
  - Sviluppi previsti: U-Net, segmentazione nella post-elaborazione.
- **Limiti che noto**:
  - Nessun hold-out spaziale: la valutazione è "on the original dataset from the Veluwe", che contiene anche i tasselli di addestramento. Anche la soglia 0,80 è scelta sugli stessi dati, quindi le metriche sono ottimistiche.
  - Un solo annotatore e nessuna misura di concordanza tra annotatori.
  - La formula dell'MCC è stampata senza parentesi.
  - Il valore negativo dominante (TN ≈ 88,8 km²) gonfia l'ACC.
  - Recall ad area 0,65: lontanissimo dalla soglia del progetto.
- **Per Morloquio**:
  - **Adottare** il feedback loop "new positives" come procedura standard contro l'incompletezza di OSM: rivedere i falsi positivi più confidenti rispetto a OSM e promuovere a etichetta quelli che sono strade vere. Due giri sono bastati a trovare il 5,5% di etichette mancanti.
  - **Adottare** come input canali quantitativi in float derivati dal DTM, normalizzati per patch, invece di rendering 8-bit tipo hillshade (+0,069 di MCC DTM contro HS a parità di resto). L'hillshade multidirezionale resta utile per la revisione umana. Come canale di rete è da confermare con un'ablazione su DTM normalizzato, LRM e openness.
  - **Adottare** scheletro + `sknw` per le centerline (è già lo standard), con potatura degli spuri. Proposta: rami terminali sotto 10–15 m da tarare sulle tile di validazione.
  - **Adattare** il "pruning" 5–95%: tradotto in segmentazione, significa una fascia "ignore" (fuori dalla loss) attorno alle linee OSM tra la larghezza nominale e qualche metro oltre, invece di trattarla come negativa (proposta).
  - **Scartare** la classificazione a tasselli 64×64 a 0,5 m: la granularità di 32 m e la perdita delle strade strette sono incompatibili con recall ≥ 0,98 su mulattiere larghe 2–4 m.
  - **Scartare** la LBR come esclusione di zone (strade, aree edificate, agricoltura): Morloquio deve mappare anche le strade moderne, e ogni esclusione abbassa il tetto della recall. Al massimo può essere un canale o un prior.
  - **Scartare** il bilanciamento con circa il 98% di negativi buttati via: meglio pesi nella loss e campionamento di tile con strade.
  - **Valutazione**: CarcassonNet mostra l'errore da evitare. Servono tile tenute fuori spazialmente, la soglia scelta su tile di validazione separate dal test, e una recall per lunghezza con tolleranza, non per area.

---

### Verschoof-van der Vaart & Landauer 2022 — Trasferibilità di CarcassonNet (Germania e Slovenia)

- **Riferimento**: W. Verschoof-van der Vaart, J. Landauer, "Testing the transferability of CarcassonNet: A case study to detect hollow roads in Germany and Slovenia", in Börner W. et al. (Hrsg.), *Artificial Intelligence: New Pathways Towards Cultural Heritage*, Proceedings CHNT 25 (Vienna 2020), Propylaeum, Heidelberg 2022, pp. 61–68. DOI 10.11588/propylaeum.1045.c14482. Licenza del capitolo CC BY-NC-ND 3.0, letta nella pagina del catalogo.
- **Lettura**: completa, ma sulla versione del capitolo 6 della tesi di dottorato (Leiden 2022, hdl 1887/3256824, pp. 109–117). La tesi dichiara: "This chapter has been published as: … CHNT25". Il PDF Propylaeum non l'ho letto: è bloccato da Anubis (proof-of-work), sia con Crawl4AI `--stealth` sia con curl. Possibili piccole differenze redazionali rispetto alla versione stampata.
- **Cosa fanno**: applicano senza ri-addestrarlo il modello addestrato sulla Veluwe a tre aree con rilievo, uso del suolo e LiDAR diversi. L'unico parametro ricalibrato è la soglia di confidenza ("threshold moving").
- **Dati** (Tab. 6.1):

  | Area | km² | Risoluzione | pt/m² | Terreno / uso |
  |---|---|---|---|---|
  | Veluwe (NL), addestramento | 97.25 | 0,5 m | 6–10 | creste e valli, brughiera |
  | Schurwald (DE) | 25 | 1,0 m | 1–4 | montagna, bosco |
  | Upper Carniola (SI) | 1.18 | 0,5 m | 3–10 | colline, bosco |
  | Nova Gorica (SI) | 1.32 | 1,0 m | 3–10 | carso, bosco; trincee militari simili alle hollow roads |

  Etichette manuali di entrambi gli autori.
- **Metodo**: lo stesso del capitolo 5: tasselli 64×64, soglie 95/5%, circa 32 000 tasselli bilanciati, ResNet-34 ImageNet, "twelve epochs" con progressive resizing (5+4+3), vettorializzazione QGIS + Van Etten 2019. Metriche ad area. Soglia ottimale trovata empiricamente per ogni test set, massimizzando l'MCC.
- **Risultati** (Tab. 6.2):

  | Test | Soglia | P | R | MCC |
  |---|---|---|---|---|
  | Veluwe | 0.8 | 0.41 | 0.65 | 0.47 |
  | Schurwald | 0.75 | 0.57 | 0.14 | 0.25 |
  | Upper Carniola | 0.8 | 0.66 | 0.35 | 0.38 |
  | Nova Gorica | 0.1 | 0.51 | 0.42 | 0.37 |

  - **Incoerenza**: il testo dice "Upper Carniola … 0.37 (confidence threshold of 0.1) and [Nova Gorica] 0.38 (… 0.8)", cioè l'opposto della tabella. La tabella è coerente con i TP/FP/FN riportati, verificato ricalcolando P e R (calcolo mio).
  - L'Eq. 6.1 omette il termine (TP+FN) al denominatore: refuso.
  - Interpretazione degli autori: la precision resta simile o migliore, mentre a crollare è la recall. Il fattore principale non sarebbe né il terreno né la risoluzione (0,5 e 1,0 m danno risultati simili tra le due aree slovene), ma la densità media di punti al suolo: a Schurwald, con 1–4 pt/m², le hollow roads sono "much less pronounced".
- **Limiti dichiarati**: le aree slovene sono piccole (2,5 km² in totale) contro i 25 km² di Schurwald. Proposte: addestrare su più dataset, super-risoluzione con GAN.
- **Limiti che noto**:
  - La soglia "ottimale" è scelta sugli stessi dati di test: l'MCC riportato è un massimo a posteriori, non una prestazione operativa.
  - La soglia ottimale varia da 0,1 a 0,8 tra regioni, quindi le probabilità non sono calibrate tra domini.
  - Non c'è una misura diretta della densità al suolo per pixel.
- **Per Morloquio**:
  - Il dato più rilevante è che con DTM a 1 m e bassa densità al suolo sotto bosco (Schurwald) la recall cade a 0,14. Il PST MASE è un DTM a 1 m e non abbiamo la nuvola di punti. Proposta: stratificare la valutazione per copertura arborea, usando come proxy l'altezza della chioma DSM first − DTM, e per pendenza, e riportare la recall per strato.
  - **Adottare** il "threshold moving", ma con la soglia scelta su tile di validazione della stessa regione e distinte dalle tile di test, con vincolo recall ≥ 0,98 invece del massimo MCC.
  - Prevedere una ricalibrazione per regione (Calabria contro Sicilia) e, se servono, campioni etichettati di ciascuna.
  - **Scartare** l'idea di trasferire senza dati locali: anche con P ≈ 0,5–0,66, la R di 0,14–0,42 è inaccettabile.

---

### Kazimi 2021 — Tesi DGK C 879: apprendimento auto-supervisionato per monumenti archeologici nei DTM

- **Riferimento**: Bashir Kazimi, *Self Supervised Learning for Detection of Archaeological Monuments in LiDAR Data*, Dissertation Leibniz Universität Hannover (relatrice M. Sester; esame il 07.07.2021). DGK Reihe C Heft 879, München 2021, ISBN 978-3-7696-5291-8. URL letto: https://dgk.badw.de/fileadmin/user_upload/Files/DGK/docs/c-879.pdf
- **Lettura**: completa per i capitoli pertinenti:
  - cap. 2.3.4 (raster derivati);
  - cap. 4 (dati);
  - cap. 5 (metodo);
  - cap. 6.1–6.4 e 6.6–6.7 (esperimenti);
  - cap. 7 (discussione).

  Il §6.5 (4 regioni di test con annotazioni di tre colleghi) l'ho letto solo in parte: riguarda soltanto il dataset areale. Ho saltato i capitoli di base di deep learning (2.4) e la bibliografia.
- **Cosa fanno**: pretext auto-supervisionato su DTM non etichettati: la rete impara a generare dal DTM le visualizzazioni del rilievo calcolate con RVT. I pesi ottenuti servono poi da inizializzazione per classificazione, segmentazione d'istanza e segmentazione semantica su pochi dati etichettati.
- **Dati**:
  - DTM della Bassa Sassonia, 0,5 m, 47 000 km², da TIN con interpolazione lineare.
  - Pretext: 200 000 patch 224×224 casuali. Target RVT: LD, SLRM, slope, SVF, POS, NEG, con raggio di ricerca 64 px e 16 direzioni. Patch e raster scalati a [0,1] con min-max per patch (Eq. 4.1).
  - Dataset lineare (Harz): linee in ArcGIS trasformate in poligoni con Buffer per classe:
    - fossi 1 m per lato (2 m di larghezza);
    - sentieri e hollow ways 2 m per lato (4 m);
    - strade di contea 4 m, statali 6 m, federali 8 m per lato.

    Conteggi: fossi 837 segmenti (1.69 km²), sentieri 1183 (5.28 km²), strade 174 (3.06 km²), hollow ways 756 (1.39 km²). La colonna "Avg. area" della Tab. 4.2 (es. 54 479 m² per le hollow ways) non è coerente con totale e conteggio: refuso.
  - Split su tre regioni disgiunte (Tab. 6.13; addestramento / validazione / test): fossi 364/35/40, sentieri 668/135/102, strade 82/5/4, hollow ways 366/59/23.
  - Dataset areale: crateri di bomba 617, carbonaie 2543, tumuli 1410, buche minerarie 2986 (con cerchi approssimati). Cave di pietra: 3082.
- **Metodo**:
  - Rete principale HRNet. RVNet: encoder-decoder con uscita a 6 canali, loss BCE. RVGan: Pix2Pix con generatore HRNet.
  - Pretext: Keras, 50 epoche, batch 20, Adam, split 80/10/10.
  - Downstream lineare:
    - segmentazione semantica HRNet (da zero, da RVNet o da RVGan) contro DeepLabV3+ con ResNet: input 128×128, 50 epoche, batch 20, cross-entropy categoriale;
    - Mask R-CNN (ResNet o HRNet): 128×128, batch 8, ottimo a 14–15 epoche.
  - Inferenza su grandi aree a finestre affiancate (Algoritmo 2, stride = dimensione della finestra). Valutazione con Spatial Join di ArcGIS su poligoni (conteggio d'istanze che si intersecano), più IoU per pixel.
- **Risultati** (dataset lineare, test):
  - Segmentazione semantica, IoU (Tab. 6.21):

    | Modello | MIOU | Fosso | Sentiero | Strada | Hollow way |
    |---|---|---|---|---|---|
    | HRNet, pesi casuali | 52.24 | 17.73 | 44.17 | 81.61 | 26.34 |
    | DeepLabV3+ | 44.15 | 14.22 | 30.91 | 73.12 | 13.05 |
    | HRNet + RVNet | 53.46 | 18.70 | 44.98 | 81.68 | 30.71 |
    | HRNet + RVGan | 53.04 | 17.52 | 47.78 | 81.91 | 26.57 |

  - Conteggi d'istanza (Tab. 6.22, % di oggetti GT intersecati). "NC" = non catturati da nessuna classe. Hollow ways corrette: 49.71 / 61.92 / 59.21% (casuale / RVNet / RVGan), con NC 22.11 / 18.35 / 16.54. Sentieri NC: 12.05 / 14.47 / 12.87. Strade NC: 0.00 / 0.00 / 0.05.
  - Quindi la recall per oggetto agnostica alla classe è 1 − NC: hollow ways 77.9 / 81.7 / 83.5%, sentieri 88.0 / 85.5 / 87.1% (calcolo mio; basta un'intersezione qualsiasi).
  - Sovrapredizione (OP, % delle istanze predette che non toccano il GT) per le hollow ways: 90.87 / 91.39 / 90.56%. Per area (Tab. 6.23): 93.22 / 94.78 / 94.49%.
  - Mask R-CNN lineare, MAP: HRNet 48.77, ResNet 49.45, HRNet+RVNet 50.73, HRNet+RVGan 54.95. Molta confusione verso la classe "path" (Tab. 6.15).
  - Dataset areale, input singoli (Tab. 6.17, MIOU): DTM 62.37, SVF 62.83, LD 61.82, SLOPE 61.46, POS 61.10, SLRM 59.82, NEG 54.93, combinato dei 6 RV 61.58. Con SSL (Tab. 6.18): 61.49 → 66.21 (RVGan).
- **Limiti dichiarati**:
  - Etichette imprecise: cerchi per gli oggetti areali, buffer a larghezza fissa per quelli lineari.
  - Strutture sottili (fossi, hollow ways) predette male: IoU 17.52 e 26.57 con RVGan.
  - Le bounding box non servono per oggetti lineari in diagonale.
  - Poligoni predetti frammentati per gli oggetti lineari: serve post-elaborazione per riconnetterli.
  - Prestazioni limitate dalla quantità e qualità delle etichette. Prospettive: tag OSM come input, foto aeree.
- **Limiti che noto**:
  - Le patch di test sono ritagliate attorno agli oggetti: sulle aree lineari manca una scansione esaustiva del background.
  - Le OP del 90% e oltre mescolano falsi positivi veri e strade non etichettate; non c'è un audit.
  - Test molto piccolo per le strade (4 oggetti).
  - Il guadagno del pretext sul lineare è modesto: +1,22 MIOU.
  - La descrizione di "SLRM" nel cap. 2 è in realtà quella del LRM completo di Hesse (con il DEM "purgato"), mentre RVT calcola la versione semplice.
- **Per Morloquio**:
  - **Adottare** la segmentazione semantica, non a istanze né a tasselli: l'autore lo conclude esplicitamente per le strutture lineari.
  - **Adottare** un'architettura ad alta risoluzione (HRNet o equivalente) invece di DeepLabV3+: per le hollow ways IoU 26.34 contro 13.05, per i sentieri 44.17 contro 30.91.
  - **Adottare** larghezze di buffer per classe nella rasterizzazione di OSM a 1 m. Riferimento di Kazimi: 4 m per sentieri e hollow ways, 8/12/16 m di larghezza totale per strade di contea / statali / federali. Proposta di corrispondenza: path/track/mulattiera 4 m, unclassified/tertiary 8 m, secondary 12 m, primary 16 m.
  - **Adottare (opzionale)** il pretext "DTM → visualizzazioni RVT" su tutte le tile PST, escluse quelle di test, prima del fine-tuning. Costa poco, accelera la convergenza e dà guadagni piccoli ma coerenti (IoU hollow way +4.37 con RVNet). Target sensati per noi: LRM, openness ±, SVF, slope, LD.
  - **Scartare** l'IoU come metrica principale per strade strette: penalizza l'errore di larghezza del buffer. Usare recall per lunghezza con tolleranza, misurata a parte.
  - Le OP alte con etichette incomplete confermano che serve un campione di "falsi positivi" verificato a mano prima di giudicare la precision.

---

### Kazimi, Thiemann, Sester 2019 — Segmentazione semantica di strutture antropiche nei DTM (DL4DTM)

- **Riferimento**: B. Kazimi, F. Thiemann, M. Sester, "Semantic segmentation of manmade landscape structures in digital terrain models", *ISPRS Annals* IV-2/W7 (2019) 87–94. DOI 10.5194/isprs-annals-IV-2-W7-87-2019, CC BY 4.0. PDF da isprs-annals.copernicus.org.
- **Lettura**: completa.
- **Cosa fanno**: DeepLabv3+ adattato ai DTM, con due modifiche:
  - normalizzazione per singola patch;
  - uscita grande metà dell'input: si predice solo il centro, con stride di scansione M − N, così ogni pixel predetto ha contesto su tutti i lati.
- **Dati**: Harz, DTM 0,5 m, circa 3000 km². 157 crateri di bomba (2–10 m) e 1044 carbonaie (3–19 m). Input 128×128 ritagliati casualmente da 256×256 centrati sull'oggetto. Split 80/20: 127+854 / 30+190.
- **Metodo**:
  - Normalizzazione Eq. 4: (Xᵢ − mean(Xᵢ)) / (max(Xᵢ) − min(Xᵢ)) per singolo esempio. La versione per dataset faceva divergere il modello.
  - Input a 1 canale, uscita 64×64, secondo upsampling ×2 invece di ×4.
  - Keras, 150 epoche, batch 20, Adam lr 0.001, sparse categorical cross-entropy. Aumenti: ritaglio casuale e rotazioni casuali.
- **Risultati**:
  - mIoU sulle 220 patch di validazione: DeepLabv3+ 0.77, DL4DTM 0.75. L'uscita più grande favorisce DeepLabv3+, perché c'è più background.
  - Su due regioni di test scansionate: regione 1 0.35 contro 0.37, regione 2 0.37 contro 0.47.
  - Tasso di detezione: crateri 44/51 (86%) contro 49/51 (96%); carbonaie 188/233 (81%) contro 228/233 (98%).
  - DL4DTM trova anche oggetti non etichettati.
- **Limiti**: solo oggetti areali; etichette circolari; validazione e test piccoli; nessuna metrica per linee.
- **Per Morloquio**:
  - **Adottare** la predizione del solo centro della finestra, o un overlap-tile con scarto dei bordi. Esempio (proposta): input 512, uscita centrale 256, stride 256. È l'accorgimento che porta la detezione delle carbonaie dall'81 al 98% nelle aree scansionate. Con LRM e openness, che hanno raggi di 10–30 m, i bordi delle tile sono comunque inaffidabili.
  - **Adottare** la normalizzazione per patch del canale DTM: sottrazione della media della patch, con una scala fissa in metri (proposta) invece del min-max, che in patch piatte amplifica il rumore.

---

### Kazimi & Sester 2023 — SSL per segmentazione semantica di monumenti nei DTM (JCAA)

- **Riferimento**: B. Kazimi, M. Sester, "Self-Supervised Learning for Semantic Segmentation of Archaeological Monuments in DTMs", *Journal of Computer Applications in Archaeology* (2023). DOI 10.5334/jcaa.110, CC BY 4.0. PDF dal file server dell'editore (file server dell'editore).
- **Lettura**: completa per metodo, esperimenti e conclusioni; bibliografia scorsa.
- **Cosa fanno**: versione PyTorch dello stesso schema della tesi, solo sul dataset areale (senza oggetti lineari). Confronto con il pre-addestramento su immagini naturali.
- **Metodo**:
  - RVNet: 180 000 / 10 000 / 10 000 patch 224×224, MSE, SGD (lr 0.001, momentum 0.9, weight decay 0.0001), batch 208, 200 epoche (migliore la 139). Aumenti: flip, rotazione di 90°, ritaglio 128 da 224.
  - RVGAN: λ = 100 con L1; Adam (lr 0.0002, β1 0.5, β2 0.999); batch 256; 100 epoche (migliore la 51); discriminatore PatchGAN 70×70.
  - Target: LD, Slope, SLRM, SVF.
  - Downstream: finestre 256 con ritaglio 128, 100 epoche, batch 96, migliore tra Adam, SGD e RMSProp. Inferenza a finestre 128 con "stride of 85 pixels (i.e., 85 pixels of overlap …)": formulazione ambigua.
- **Risultati**:
  - Ricostruzione: L1 0.0929 (RVNet) contro 0.0390 (RVGAN); L2 0.0170 contro 0.0042.
  - Input con pesi casuali (Tab. 6, mIoU): {LD, Slope, SLRM, SVF} 63.71; con DTM 63.44; DTM da solo 62.64; SLRM 62.18; LD 61.91; tutti 61.29; SVF 61.23; RGB hillshade 61.03; Slope 59.27.
  - Pesi (Tab. 8): DTM con RVGAN 63.18 (blocchi congelati fino al 2), RVNet 63.02. RGB hillshade con ImageNet 62.85, COCO-Stuff 62.79, ADE20K 62.64, Pascal VOC 62.63.
  - Codice e pesi su https://github.com/SSL-DTM (licenza non verificata).
- **Per Morloquio**:
  - I guadagni delle visualizzazioni e del pretext sono dell'ordine di 1 punto di mIoU; ImageNet su hillshade non batte il DTM da pesi casuali.
  - Conclusione pratica: priorità a etichette e loss. Le visualizzazioni RVT come canali vanno tenute solo se un'ablazione sulle nostre tile lo giustifica.
  - Non usare backbone ImageNet su rendering RGB del rilievo.

---

### Hesse 2010 — Local Relief Model (dalle fonti aperte) e parametri RVT / rvt_py

- **Riferimento**: R. Hesse, "LiDAR-derived Local Relief Models – a new tool for archaeological prospection", *Archaeological Prospection* 17(2) (2010) 67–72, DOI 10.1002/arp.374.
- **Lettura**: articolo originale **non letto**: non ho trovato copie ad accesso aperto, solo pagine d'editore o aggregatori a pagamento. Algoritmo e parametri ricostruiti da fonti aperte lette per intero nelle parti pertinenti:
  1. Kokalj & Hesse 2017, *Airborne Laser Scanning Raster Data Visualization: A Guide to Good Practice*, Založba ZRC (PDF omp.zrc-sazu.si/zalozba/catalog/download/824/3485/287): §2.4–2.8, tabelle 5–9 e 14, caso di studio 5.2 sulle hollow ways.
  2. Hesse 2016, "Visualisierung hochauflösender Digitaler Geländemodelle mit LiVT", Edition Topoi BSA 34 (edition-topoi.org/download_pdf/bsa_034_07.pdf): §4.1–4.4.
  3. Manuale RVT 2.2.1 (zrc-sazu.si/sites/default/files/rvt_2.2.1_0.pdf): completo.
  4. rvt_py: README su GitHub, documentazione readthedocs (esempi e pagine dei metodi), sorgente `rvt/vis.py` (firme e default) e file `LICENSE`.
  5. Slide di Hesse ad AARG 2010.
- **Algoritmo LRM di Hesse** (guida 2017, p. 21; Hesse 2016, §4.4):
  1. Filtro passa-basso del DEM. LiVT offre forma circolare o quadrata e raggio regolabile.
  2. Mappa di trend removal: differenza DEM − DEM liscio.
  3. Linee di livello zero della mappa del punto 2 (dove DEM e DEM liscio coincidono); a quei punti si assegnano le quote del DEM.
  4. Interpolazione da quei punti di un "purged DEM". LiVT permette di scegliere l'algoritmo d'interpolazione e la distanza massima dai punti d'appoggio.
  5. LRM = DEM − purged DEM.

  Scopo: il semplice trend removal sottostima progressivamente l'altezza o profondità relativa man mano che la struttura si allarga; il LRM la restituisce in modo meno distorto. Il kernel dovrebbe riflettere la dimensione delle forme, "a generally safe bet is a kernel size of about 25 m". Il metodo funziona meglio su pendii graduali: "produces artefacts such as artificial banks and ditches where relief is diverse and/or changing abruptly". Hesse 2016 aggiunge che i cigli naturali diventano anomalie positive e i piedi di versante anomalie negative.
- **Parametri raccomandati** (guida 2017):

  | Tecnica | Generale | Molto piatto | Ripido/complesso |
  |---|---|---|---|
  | Trend removal / LRM, raggio del filtro | 10 m (stretch ±1.0 m) | 5 m (±0.5 m) | 25 m (±2.0 m) |
  | SVF | raggio 10 m, 16 direzioni; stretch 0.65–1.0 | 10 m, 16 dir.; 0.9–1.0 | 10 m, 16 dir.; 0.55–1.0 |
  | Openness positiva | 10 m, 16 dir.; 65–95° | 85–91° | 55–95° |
  | Openness negativa (scala di grigi invertita) | 60–95° | 75–95° | 45–95° |
  | Local dominance | raggio 10–20 m, osservatore 1.7 m, stretch 0.5–1.8 | 10–20 m, stretch 0.5–3.0 | colonna illeggibile nel testo estratto |

  - La matrice di Tab. 14 suggerisce: LRM con raggio circa 20 m, LD 10–20 m, openness 10 m, SVF circa 10 m.
  - Proprietà utili alla detezione automatica, dal testo:
    - l'openness è "devoid of general topography", e le firme degli oggetti sono "the same irrespectively of their location on a plane or slope";
    - la positiva evidenzia "ridges between hollow ways", la negativa "the actual hollow ways";
    - LD funziona bene per "hollow ways".
  - Caso di studio sulle hollow ways a Žiški vrh: è servita una combinazione di SVF (0.65–1.0), openness positiva (70–93), slope (0–45°) e hillshade.
- **RVT 2.2.1 (manuale)**:
  - SLRM: raggio predefinito 20 px, ammessi 5–50, "radius should be a bit more than half the size of the features". Dalla versione 1.2 usa una media semplice invece di quella gaussiana. Per il LRM completo il manuale rimanda a Hesse 2010 e LiVT.
  - SVF: 8 direzioni di default (16 "optimal for most applications"), raggio 10 px (ammessi 1–100), rimozione del rumore low/medium/high (ignora il primo 10/20/40% dei pixel).
  - LD: raggio minimo 10 px, massimo 20 px.
  - Nei raster a 8 bit lo stretch è fisso: SVF 0.6375–1, openness 60–95°, LD 0.5–1.8, slope 0–51°.
  - I pixel di bordo sono errati per un'ampiezza pari al raggio.
  - Licenza della versione IDL: Apache License 2.0.
- **rvt_py** (sorgente letto):
  - `slrm(dem, radius_cell=20)` è un filtro di media a box di lato 2r+1 (immagine integrale), poi DEM − media. Rifiuta raggi fuori da 10–50 px. **Non** implementa il purged DEM di Hesse.
  - `sky_view_factor(..., svf_n_dir=16, svf_r_max=10, svf_noise=0)` calcola anche l'openness positiva; la negativa si ottiene passando −DEM.
  - `local_dominance(min_rad=10, max_rad=20, rad_inc=1, angular_res=15, observer_height=1.7)`.
  - Ci sono anche `msrm` e `mstp`.
  - **Licenza rvt_py: Apache License 2.0** (file LICENSE verificato), compatibile con il vincolo open source.
  - LiVT (Hesse) è sotto GNU GPL (Hesse 2016), in VB.NET.
- **Per Morloquio**:
  - Il nostro "LRM" con gaussiana σ = 10 m è un trend removal, non il LRM di Hesse, a meno che la pipeline non calcoli il purged DEM. Come scala equivale a un box di raggio r ≈ σ·√3 ≈ 17 m (calcolo mio: varianza del box r²/3). Sta quindi tra i 10 m "general" e i 25 m "steep or complex", vicino ai circa 20 m della Tab. 14. **Tenerlo.**
  - Rischio specifico del pilota: in montagna con tornanti e terrazzamenti, il trend removal crea falsi argini e fossi ai cambi di pendenza, cioè firme molto simili a una strada.
  - **Aggiungere** openness positiva e negativa con r = 10 m e 16 direzioni (`rvt.vis.sky_view_factor`, compute_opns, con DEM e −DEM), calcolate in float32. Sono le visualizzazioni consigliate per terreno variabile e per la detezione automatica, e la negativa marca il fondo delle mulattiere incassate.
  - **Valutare** LD con 10–20 px e SVF con 10 px, 16 direzioni, noise "low" se compaiono strisce. Le curvature restano complementari.
  - Proposta multi-scala: una seconda scala di trend removal più fine per carreggiate di 2–4 m, per esempio σ ≈ 3 m (regola "un po' più di metà della dimensione della forma"). `rvt.vis.slrm` non scende sotto 10 px: usare scipy.
  - Il purged LRM completo non porta vantaggi chiari per strutture strette ed è costoso (interpolazione da linee di livello zero): **non prioritario**.
  - Calcolare tutto su tile con un margine almeno pari al raggio massimo (circa 3σ = 30 m per la gaussiana a 10 m) e non usare mai gli 8-bit con stretch fisso come input della rete.

---

### Meyer-Heß, Pfeffer, Juergens 2022 — CNN su DTM: tumuli vicino agli hollow ways (Westfalia)

- **Riferimento**: M.F. Meyer-Heß, I. Pfeffer, C. Juergens, "Application of Convolutional Neural Networks on Digital Terrain Models for Analyzing Spatial Relations in Archaeology", *Remote Sensing* 14(11) (2022) 2535. DOI 10.3390/rs14112535, CC BY 4.0.
- **Lettura**: completa. HTML MDPI letto con Crawl4AI `--stealth --max-chars 200000`: 62 450 caratteri, non troncato, con testo, Tabella 1 e didascalie. Il PDF MDPI è bloccato da Akamai e la URL `/htm` pure. Figure non viste.
- **Cosa fanno**: la CNN **non** rileva hollow ways. Rileva tumuli (Faster R-CNN) entro 3 km da 43 hollow ways noti, presi dal database LWL, e studia la distribuzione delle distanze.
- **Dati**:
  - LiDAR NRW 2018–2019, ultimi impulsi filtrati su griglia regolare da 1 pt/m², cioè un DTM a 1 m senza interpolazione (licenza dl-de/by-2-0).
  - Area di addestramento di 168 km² a sud di Paderborn.
  - 194 tumuli di riferimento (punti centrati, verificati a vista) e 219 punti casuali "non-mound".
  - Selezione degli hollow ways: almeno il 90% di bosco "archeologicamente rilevante" (il "Positive Layer", da cui sono escluse le infrastrutture moderne) entro 500 m.
- **Metodo**:
  - DTM lisciato con un filtro circolare di raggio 1 m, poi riempimento dei sink.
  - Tre bande: aspect, slope e una difference map (trend removal con filtro circolare di raggio 10 m, citando Hesse 2010).
  - Chip 30×30 px con stride 15 entro 12 m dai punti, rotazioni a passi di 45°.
  - Faster R-CNN con ResNet50 congelato, chip size 175, limite di 50 epoche (interrotto alla 25), in ArcGIS Pro 2.9. Hardware: GTX 1660 Super 6 GB; export 13 h 16 min, addestramento 2 h 18 min, detezione 1 h 46 min.
  - Post-classificazione:
    - rimozione delle detezioni "non-mound";
    - Positive Layer (centro della detezione dentro le aree ammesse);
    - soglia di confidenza;
    - filtro OBIA sulla difference map: media ≥0 e ≤0.15, minimo ≥−0.36 e ≤0.09. Soglie fissate a mano su 348 detezioni.
- **Risultati**:
  - Area di addestramento, soglia 50%: 17 993 risultati, 191/194 riferimenti trovati (recall 98.45%), precision 1.06%.
  - Dopo la rimozione dei non-mound: recall 97.94%, precision 2.64%.
  - Dopo il Positive Layer: 3549 risultati, 186 riferimenti trovati (recall 95.87%).
  - Soglia 99%: 161 risultati, recall 71.13%, precision 85.71%. Contando i "new positives" (tumuli erosi esclusi dal training) la precision sale a 90.68%.
  - Varianti peggiori:
    - meno dati: P 88.66%, R 44.33%;
    - altra dimensione dei chip: P 68.88%, R 69.59%;
    - Mask R-CNN alla soglia 90%: P 77.27%, R 26.29%.
  - Sottoinsieme fuori dall'area di addestramento (6 hollow ways): 12 detezioni dopo i filtri, precision ≥75%.
  - Intera area: 184 detezioni, 141 tumuli veri (precision 77%). Dei falsi positivi: 3 sono depressioni, 8 strutture moderne, 32 forme morfologicamente identiche (dune, cime).
  - Distribuzione: entro 500 m solo il 33% degli hollow ways ha tumuli; entro 1500 m il 72%, entro 2000 m l'84%, entro 3000 m il 91%. Massimo nella fascia 2000–2500 m (31%).
- **Limiti dichiarati**:
  - Il trasferimento fuori dall'area di addestramento richiede post-classificazione.
  - Il Positive Layer rimuove anche veri positivi.
  - Pochi campioni di addestramento.
  - Workflow non completamente automatizzato (ModelBuilder).
  - Il riempimento dei sink andrebbe forse fatto sulla difference map.
- **Limiti che noto**:
  - Recall misurata sugli stessi 194 tumuli usati per generare i chip: nessun hold-out.
  - Fuori dall'area d'addestramento la recall non è stimata, solo la precision.
  - Software proprietario (ArcGIS Pro).
- **Per Morloquio** (rilevanza bassa per il rilevamento di strade):
  - Lezione quantitativa: con un detector di oggetti su DTM a 1 m, una recall del 98% si ottiene solo con precision dell'1–3%. Il target ≥ 0,98 implica soglie basse e una post-elaborazione che filtri i falsi positivi senza toccare i veri. Ogni filtro va misurato per la recall che costa: qui il Positive Layer ha tolto 4 riferimenti su 190.
  - **Scartare** il riempimento dei sink: cancellerebbe proprio le mulattiere incassate.
  - **Scartare** il Positive Layer (esclusione delle infrastrutture moderne): incompatibile con l'obiettivo di mappare tutte le strade.
  - **Adattabile**: filtri a oggetti sui candidati lineari basati su statistiche del LRM o dell'openness lungo il segmento. Esempio (proposta): una mulattiera incassata deve avere LRM medio negativo al centro e positivo ai bordi. Va però tarato sulle tile di validazione per recall, non per precision.
  - Pista bibliografica dall'articolo, da leggere: Salberg, Trier, Kampffmeyer 2017, "Large-Scale Mapping of Small Roads in Lidar Images Using Deep CNNs". Da CarcassonNet: Ferraz, Mallet, Chehata 2016, ISPRS JPRS 112:23–36 (strade forestali in montagna da LiDAR, F1 0.849). Entrambe sono più vicine al problema di Morloquio di tutte le fonti di questo lotto.

---

### Sintesi trasversale per Morloquio

- Nessuna delle fonti si avvicina a recall 0,98 su strutture lineari:
  - CarcassonNet: R = 0,65 ad area, senza hold-out;
  - Kazimi: IoU hollow way 26–31, recall per oggetto agnostica alla classe circa 78–83% (calcolo mio);
  - trasferimento di CarcassonNet: R da 0,14 a 0,42.
- Il collo di bottiglia dichiarato da tutti sono le strutture strette e la bassa densità al suolo. Servono segmentazione ad alta risoluzione, uscita centrale o overlap-tile, recall per lunghezza con tolleranza e una soglia scelta su tile di validazione per vincolo di recall.
- Etichette: buffer per classe (Kazimi), fascia ignore attorno alle linee (dal pruning 5–95% di CarcassonNet), feedback loop "new positives" (CarcassonNet +5,5% di etichette e circa +0,2 di MCC).
- Input: DTM in float normalizzato per patch, trend removal (σ 10 m, valido) con eventuale seconda scala fine, openness ± a 10 m. L'aggiunta di molte visualizzazioni RVT vale circa 1 punto (Kazimi): da confermare con un'ablazione.
- Licenze: rvt_py Apache-2.0 (utilizzabile); LiVT GPL; RVT IDL Apache-2.0; tutti gli articoli letti sono CC BY, tranne il capitolo CHNT (CC BY-NC-ND 3.0) e la tesi DGK, che è a diritti riservati ma liberamente scaricabile.

---

## Articoli non letti o letti in parte

1. **Hesse 2010, *Archaeological Prospection* 17:67–72**: non letto. Nessuna copia aperta trovata (zendy.io con restrizioni regionali, Wiley a pagamento). Algoritmo ricostruito da Kokalj & Hesse 2017, Hesse 2016 (LiVT), manuale RVT 2.2.1, sorgente rvt_py e slide AARG 2010. Kernel, interpolazione e distanze del lavoro originale non sono verificati.
2. **Verschoof-van der Vaart & Landauer 2022, CHNT 25 (Propylaeum)**: PDF del capitolo non letto, bloccato da Anubis (proof-of-work) con Crawl4AI normale e `--stealth`, e anche con curl. Letta la versione integrale del capitolo 6 della tesi di Leiden, dichiarata identica alla pubblicazione.
3. **Kazimi 2021, DGK C 879**: letta in parte. Saltati i fondamenti di DL (§2.4), le tabelle del §6.5 (solo oggetti areali) e le appendici A.1–A.4 di esempi qualitativi. Tutti i capitoli su dati, metodo, esperimenti lineari e discussione sono letti per intero.
4. **Kazimi et al., "Semi Supervised Learning for Archaeological Object Detection in Digital Terrain Models"** (CHNT 24, Propylaeum, *Monumental Computations*): non letto, sullo stesso sito protetto da Anubis. Dagli snippet di ricerca tratta solo oggetti areali (crateri, carbonaie, tumuli).
5. **Meyer-Heß et al. 2022**: PDF MDPI bloccato da Akamai. Letto per intero l'HTML via Crawl4AI `--stealth` (non troncato); figure non viste.
6. **CarcassonNet JCH**: figure non viste (solo didascalie). Testo completo.
