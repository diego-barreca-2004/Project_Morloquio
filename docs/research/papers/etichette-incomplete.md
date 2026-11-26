# Etichette OSM incomplete o spostate: loss e supervisione debole

Note di lettura integrale (2026-11-26), base dell'ADR 0004. Pagine lette con Crawl4AI, PDF scaricati ed estratti
con pypdf. Sintesi e ricetta in `../2026-11-26-ml-strade-da-lidar.md`.

Contesto di giudizio: Morloquio, U-Net PyTorch su canali derivati da LiDAR DTM/DSM 1 m (più Sentinel-2), etichette OSM incomplete e spostate di qualche metro, zone senza OSM trattate come non etichettate (PU), obiettivo recall (completeness) ≥ 0,98 contro OSM su tile tenute fuori, GPU 12 GB.

Convenzioni: i numeri sono copiati dalle tabelle degli articoli. Dove una formula era illeggibile nel testo estratto l'ho verificata sulla pagina renderizzata del PDF (lo indico). I commenti marcati "[mio]" sono miei e non degli autori.

---

### Henry, Fraundorfer, Vig 2020 — Segmentazione stradale aerea con rumore topologico nelle etichette

- **Riferimento**: C. Henry (DLR), F. Fraundorfer (TU Graz), E. Vig (Amazon), "Aerial Road Segmentation in the Presence of Topological Label Noise", ICPR 2020 (25th Int. Conf. on Pattern Recognition, tenuta 10–15/01/2021), pp. 2336–2343, DOI 10.1109/ICPR48806.2021.9412054. Letto il PDF open access del DLR (elib.dlr.de/136343), testo già estratto in `crawla-ml-strade/pdf/henry.txt`. Non esiste una versione arXiv.
- **Lettura**: completa per l'articolo principale. Non letto il materiale supplementare, che gli autori citano per le tabelle complete (Precision, Recall, Correctness e Completeness per ogni esperimento) e per i dettagli delle metriche stradali: non l'ho trovato tra le fonti aperte.
- **Cosa fanno**: confrontano loss standard (BCE, Dice, Soft-IoU) e loss robuste al rumore, adattate dalla classificazione al caso per pixel (Unhinged, Sigmoid, Ramp, Savage, Hard/Soft-Bootstrapped BCE), e propongono una nuova loss, la Soft-Bootstrapped Dice. Corrompono poi le etichette in modo sintetico (spostamenti e omissioni) per misurare quanto ciascuna loss recupera.
- **Dati**:
  - Massachusetts Roads (MA): 1171 immagini 1500×1500 a 1 m/px. Le etichette sono centerline OSM rasterizzate e dilatate a linee costanti di 7 px. Split ufficiale 1108/14/49; per l'ablation usano un re-split 80/10/10. Gli autori notano che il 13% dei pixel del dataset cade in aree bianche (senza immagine) ma con strade annotate.
  - DeepGlobe (DG): 8570 immagini 1024×1024 a 0,5 m/px, strade a larghezza variabile annotate a mano. Ablation su un re-split del training ufficiale (4983 train / 1243 val).
  - Rumore reale: omissioni e registrazione già presenti in entrambi i dataset (Fig. 1). Rumore simulato, Sez. IV-F: Shift1 (piccole ROI duplicate con spostamento casuale), Shift2 (piccole ROI traslate con spostamento casuale), Shift3 (traslazione dell'intera immagine di 0–7 px, casuale), Shift4 (traslazione fissa di 7 px), Ablation1 (azzerate diverse piccole ROI), Ablation2 (azzerata una ROI pari a metà immagine), più le combinazioni Shift2+Ablation1 e Shift2+Ablation2.
  - Generalizzazione: CrackTree (206 immagini 800×600, centerline dilatate a 5 px) e ISBI'12 EM (membrane cellulari).
- **Metodo**:
  - Architettura: U-Net con encoder e decoder sostituiti da blocchi ResNet o DenseNet (Res-U-Net18/34/50/101, Dense-U-Net-121/169/201) ed encoder preaddestrato su ImageNet. Nel decoder: ConvT per l'upsampling, concatenazione con lo skip, conv 1×1 per ridurre i canali, poi il blocco denso o residuo. Modello scelto: Dense-U-Net-121.
  - Notazione: y_i ∈ {0,1} è l'etichetta, p_i ∈ [0,1] la probabilità predetta, N il numero di pixel.
  - Loss standard:
    - BCE(y,p) = −Σ_k Σ_i y_ik log(p_ik)
    - DICE(y,p) = 1 − (1 + Σ_i 2 y_i p_i) / (1 + Σ_i (y_i² + p_i²))
    - IoU(y,p) = 1 − (1 + Σ_i y_i p_i) / (1 + Σ_i (y_i + p_i − y_i p_i))
    - Il "+1" serve a dare loss nulla quando etichetta e predizione sono entrambe vuote.
  - Loss simmetriche:
    - Unhinged = (1/N) Σ (1 − y_i p_i)
    - Ramp = (1/N) Σ max(0, min(1, 1 − β y_i p_i))
    - Sigmoid = (1/N) Σ Sigmoid(−β y_i p_i)
    - Savage = (1/N) Σ 1/[1 + e^{2 y_i max(ε, min(1−ε, p_i))}]², con ε = 10⁻⁶
  - Loss bootstrapped (da Reed et al.):
    - BCE_HB = −Σ_k Σ_i (β y_ik + (1−β) p̂_ik) log(p_ik), dove p̂ è p sogliato a 0,5
    - BCE_SB = −Σ_k Σ_i [β y_ik + (1−β) p_ik] log(p_ik)
    - **DICE_SB** = 1 − (1 + Σ_i 2[β y_i + (1−β) p_i] p_i) / (1 + Σ_i ([β y_i + (1−β) p_i]² + p_i²))
    - L'articolo non dice se il termine (1−β)p del bersaglio sia staccato dal grafo (detach).
  - Iperparametri:
    - β validato in {0,4; 0,5; …; 0,9}. Per HB, SB e DICE_SB su MA e DG β = 0,7. Su CrackTree ed EM β = 0,4 o 0,5. Per Sigmoid, Ramp e Savage β = 1.
    - Le loss simmetriche si usano sempre sommate alla BCE: da sole non bastano. Le combinazioni sono somme non pesate.
    - Adam, lr fisso 10⁻⁴, nessun weight decay, 40 epoche, flip orizzontali e rotazioni di 90/180/270°.
    - Batch e dimensione delle patch non sono indicati nel testo principale.
    - TTA: 8 combinazioni di flip e rotazioni, somma delle probabilità (0–8) e soglia a 4,0.
  - Metriche: IoU, F1, P, R, più Correctness, Completeness e Quality di Wiedemann con buffer di 3 px (2 px su CrackTree).
- **Risultati**:
  - Architetture (Tab. I, BCE; IoU / F1 / Qual):

    | Modello | MA | DG |
    |---|---|---|
    | Dense-U-Net-121 | 57,12 / 73,03 / 70,06 | 65,13 / 79,19 / 72,43 |
    | Dense-U-Net-121 senza preaddestramento | 55,49 | 62,12 |
    | U-Net (senza preaddestramento) | 55,92 | 61,97 |
    | DeepLabv3+ (senza preaddestramento) | 52,95 | 59,65 |

  - Loss (Tab. II, Dense-U-Net-121, IoU):

    | Loss | MA | DG |
    |---|---|---|
    | BCE | 57,12 | 65,13 |
    | DICE | 57,50 | 65,18 |
    | BCE+IoU | 58,12 | 63,83 |
    | BCE+DICE+IoU | 57,99 | 65,57 |
    | BCE+Unhinged | 57,72 | 65,99 (migliore su DG) |
    | BCE_SB | 57,93 | 65,87 |
    | DICE_SB | 58,26 (migliore su MA) | 65,29 |

    Gli autori notano che i guadagni sono piccoli perché anche i test set sono rumorosi.
  - Rumore sintetico su MA (1 m, OSM: il caso più vicino a Morloquio), Tab. III, IoU:

    | Condizione | BCE senza correzione | BCE_SB | DICE_SB |
    |---|---|---|---|
    | Nessun rumore | 57,12 | 57,87 | 57,91 |
    | Shift1 | 57,19 | 57,54 | 56,85 |
    | Shift2 | 57,01 | 57,16 | 56,79 |
    | Shift3 (0–7 px casuale) | 47,26 | **53,80** | 41,63 |
    | Shift4 (7 px fisso) | 4,18 | 12,16 | 23,24 |
    | Ablation1 | 56,09 | 57,15 | 57,67 |
    | Ablation2 (metà immagine senza etichette) | **0,00** | 38,34 | **57,11** |
    | Shift2+Ablation1 | — | 56,93 | 57,42 |
    | Shift2+Ablation2 | — | 40,20 | 57,06 |

  - Su DG con DICE_SB: Shift1 68,03 (contro 65,13 senza rumore e senza correzione), Shift2 67,72, Shift4 42,58, Ablation2 64,41 (contro 64,88 senza rumore).
  - Conclusioni degli autori: la correzione del rumore aiuta sempre. Su MA, con spostamenti lievi, BCE_SB è leggermente meglio di DICE_SB; su DG DICE_SB vince ovunque; con rumore forte (Shift4, omissioni) DICE_SB è la scelta migliore. Piccoli jitter delle etichette più una loss robusta funzionano come data augmentation.
  - Split ufficiali:
    - MA (Dense-U-Net-121 + BCE_SB + IoU): IoU 65,16 senza TTA; IoU 66,61, F1 79,98, P 81,67, R 78,35 con TTA.
    - DG (+ BCE_SB + Ramp, con TTA): IoU 63,52. D-LinkNet: 63,29 con BCE, 64,36 con BCE_SB.
  - CrackTree: la Completeness più alta (93,79) la dà DICE_SB con β = 0,4.
- **Limiti**: test set rumorosi, quindi i guadagni misurati sono sottostimati. Il rumore sintetico è uniforme e casuale, non correlato con classe stradale o copertura arborea. Nessuna analisi di varianza tra seed. Un solo β per dataset. [mio] Nessun esperimento con omissione sistematica legata al tipo di strada, che è proprio il caso OSM-sentieri. Shift3 mostra che DICE_SB può peggiorare con offset casuali (41,63 contro 53,80 di BCE_SB su MA).
- **Per Morloquio**:
  - Adottare DICE_SB, β = 0,7 di partenza e validazione in {0,5…0,9}, per le omissioni: è l'unica loss che recupera l'Ablation2 (dimezzare le etichette) quasi senza perdite.
  - Per gli offset di qualche metro (a 1 m sono più pixel, come Shift3) BCE_SB è stata più robusta su MA. Proposta [mio]: BCE_SB + DICE_SB entrambi con β = 0,7. È una combinazione non testata dagli autori: va validata.
  - Il crollo con Shift4 (7 px fissi: IoU da 57 a 4 con BCE) dice che un offset sistematico va corretto o assorbito con una banda di tolleranza nelle etichette, non solo con la loss (vedi Wu 2019 e ScRoadExtractor).
  - Riprendere la metrica Completeness/Correctness di Wiedemann con buffer (3 px a 1 m) per il target di recall ≥ 0,98. Una recall per pixel senza tolleranza è irrealistica con etichette spostate.
  - Adottare la TTA a 8 trasformazioni con soglia sulla somma; abbassare la soglia per favorire la recall.
  - Scartare le loss simmetriche da sole (Unhinged, Savage e simili): servono comunque in somma alla BCE e danno guadagni marginali.

### He et al. 2026 — Positive-Guided Local Supervision (PLS)

- **Riferimento**: Hao He, Shuyang Wang, Lei Huang, Xiaohu Fan, Yongfei Li, Dongfang Yang, "Positive-Guided Local Supervision for Robust Road Extraction from Remote Sensing Imagery", Remote Sensing 2026, 18(10), 1589, DOI 10.3390/rs18101589. Letti la pagina MDPI (crawl stealth, troncata a 40k) e il PDF completo (mdpi-res.com, 27 pp.). Codice e annotazioni su github.com/hehao209/Positive-guided-Local-Supervision; le immagini immagini commerciali non sono ridistribuite.
- **Lettura**: completa (PDF). I valori del grafico per densità stradale (Fig. 6) non sono disponibili come numeri.
- **Cosa fanno**: il forward resta sull'intera immagine, ma la loss (BCE + Dice) si calcola solo dentro K finestre S×S centrate su pixel positivi dell'etichetta, cioè su strade annotate e quindi affidabili. Così il gradiente dei negativi falsi lontani dalle strade annotate (strade rurali mancanti) viene azzerato. All'inferenza non cambia nulla.
- **Dati**:
  - CH4P: 13.498 immagini 1000×1000 a circa 0,5 m/px (immagini commerciali, livello di zoom 17) da Shandong, Shanxi, Gansu e Guangxi. Le etichette sono OSM (highway, bridge, tunnel) rasterizzate con larghezza fissa di 6 px. Split 11.296 train / 2202 val. Densità media delle strade 3,94% (train, dev. std. 3,10%) e 4,15% (val), minimo 0,10%, massimo circa 25%.
  - Subset rifinito a mano: 150 immagini, due annotatori con consenso. 25 formano CH4P-mini-val e 125 CH4P-mini-test.
  - Qualità di OSM misurata su quelle 150 immagini: **il 18,4% dei pixel stradali reali manca in OSM** e **il 94,2% dei pixel marcati strada è corretto**. Gli errori dei positivi vengono da offset delle coordinate, larghezze sbagliate e mappe datate.
  - DeepGlobe: 6226 immagini 1024×1024 a 0,5 m, split casuale 5189 train / 1037 val (rumoroso). Inoltre 100 immagini del set senza etichette rifinite a mano: 20 formano DG-mini-val (iperparametri) e 80 DG-mini-test.
- **Metodo**:
  - Notazione: R⁺ sono i pixel etichettati strada, R⁻ = B ∪ M quelli etichettati sfondo, dove B è lo sfondo vero e M sono le strade non etichettate.
  - Loss classica: L_total = Σ_{p∈R⁺} ℓ(p̂_p,1) + Σ_{p∈B} ℓ(p̂_p,0) + Σ_{p∈M} ℓ(p̂_p,0) (Eq. 3). Il terzo termine spinge le strade mancanti verso lo sfondo.
  - Loss PLS: **L_PLS = (1/K) Σ_{i=1..K} (1/|P_i|) Σ_{p∈P_i} ℓ(p̂_p, y_p)** (Eq. 5).
  - Algoritmo 1:
    1. P̂ = Model(X);
    2. C = {p | Y_p = 1};
    3. campiona K centri a caso da C;
    4. per ogni centro ritaglia S×S da P̂ e Y e calcola la loss media della finestra;
    5. fai la media sulle K finestre.
  - Le finestre si possono sovrapporre. L'articolo non dice come vengono gestiti i bordi immagine né cosa succede con immagini senza positivi.
  - ℓ = BCE + Dice calcolate solo nelle finestre.
  - Backbone D-LinkNet34 e SegFormer, con la pipeline originale: batch 8; flip, rotazioni e color jitter; 100 epoche con early stopping sulla loss di validazione. La lr non è riportata (si rimanda alle implementazioni originali).
  - **Scelta finale S = 256, K = 16** (su immagini da 1000–1024 px a 0,5 m, quindi finestre di circa 128 m).
  - Analisi teorica (Sez. 4.3): con positivi rumorosi, le finestre non devono essere troppo piccole, perché servono molti negativi corretti per bilanciare i positivi sbagliati.
- **Risultati**:
  - Ablation su S con K = 16 (Tab. 1, mini-val, valori IoU / F1 / P / R):

    | S | DG | CH4P |
    |---|---|---|
    | 32 | 0,187 / 0,301 / 0,192 / **0,896** | 0,171 / 0,279 / 0,190 / 0,779 |
    | 64 | 0,366 / 0,522 / 0,394 / 0,861 | 0,330 / 0,475 / 0,377 / 0,751 |
    | 128 | 0,569 / 0,719 / 0,652 / 0,817 | 0,376 / 0,522 / 0,432 / 0,748 |
    | **256** | **0,679 / 0,803 / 0,842 / 0,776** | **0,499 / 0,634 / 0,593 / 0,746** |
    | 512 | 0,629 / 0,765 / 0,924 / 0,665 | 0,422 / 0,576 / 0,638 / 0,560 |
    | 1024 (baseline, K = 1) | 0,539 / 0,680 / 0,928 / 0,564 | 0,363 / 0,509 / 0,758 / 0,424 |

    Al crescere di S la precisione sale in modo monotono e la recall scende.
  - Ablation su K con S = 256 (Tab. 2, IoU): DG 0,687 (K=4), 0,693 (K=8), 0,679 (K=16), 0,631 (K=32); CH4P 0,446, 0,460, **0,499**, 0,449. La variazione è sotto il 6% di IoU. Un K troppo grande sovra-adatta localmente.
  - DG-mini-test (Tab. 3, F1 / IoU / P / R):

    | Metodo | F1 / IoU / P / R |
    |---|---|
    | D-LinkNet | 0,675 / 0,542 / 0,925 / 0,565 |
    | **D-LinkNet + PLS** | **0,799 / 0,675 / 0,892 / 0,736** |
    | SegFormer | 0,678 / 0,531 / 0,881 / 0,579 |
    | SegFormer + PLS | 0,754 / 0,613 / 0,870 / 0,675 |
    | RCFSNet | 0,691 / 0,548 / 0,863 / 0,610 |
    | UGD-DLinkNet | 0,635 / 0,482 / 0,854 / 0,527 |
    | D-LinkNet + GCE | 0,671 / 0,527 / 0,928 / 0,549 |
    | D-LinkNet + RLS (finestre casuali) | 0,706 / 0,562 / 0,854 / 0,626 |

  - CH4P-mini-test (Tab. 4, F1 / IoU / P / R):

    | Metodo | F1 / IoU / P / R |
    |---|---|
    | D-LinkNet | 0,617 / 0,469 / 0,822 / 0,539 |
    | **D-LinkNet + PLS** | **0,721 / 0,573 / 0,727 / 0,746** |
    | SegFormer | 0,598 / 0,434 / 0,790 / 0,498 |
    | SegFormer + PLS | 0,681 / 0,520 / 0,695 / 0,679 |
    | RCFSNet | 0,520 / 0,357 / 0,503 / 0,569 |
    | UGD-DLinkNet | 0,610 / 0,438 / 0,517 / 0,742 |
    | GCE | 0,422 / 0,315 / 0,757 / 0,361 |
    | RLS | 0,603 / 0,444 / 0,756 / 0,525 |

  - Trasferimento senza fine-tuning:
    - Da CH4P a DG-mini-test (Tab. 5): D-LinkNet IoU 0,212 (R 0,214); D-LinkNet + PLS 0,477 (R 0,599).
    - Da DG a CH4P-mini-test (Tab. 6): D-LinkNet 0,543; + PLS 0,550. Allenarsi su DG dà risultati migliori che allenarsi su CH4P, perché le etichette DG hanno larghezze vere.
  - Densità: il guadagno si concentra dove la densità stradale è sotto 0,10. Sopra 0,15 (urbano denso) PLS non aiuta e su DG è sotto il baseline: le finestre coprono quasi tutta l'immagine.
- **Limiti**:
  - Dichiarati: S, K e la forma delle finestre sono fissi, quindi le strade non etichettate dentro le finestre restano come negativi (il problema è mitigato, non eliminato). Sono previsti S e K adattivi. Il metodo non è indicato per le arterie urbane.
  - [mio] I test rifiniti sono piccoli (80 e 125 immagini) e non c'è varianza tra seed.
  - [mio] Nessuna supervisione negativa lontano dalle strade annotate: con S piccolo la precisione crolla (0,19 con S = 32).
  - [mio] Le metriche sono per pixel, senza tolleranza e senza topologia. Rifiniture e test sono degli stessi autori.
- **Per Morloquio**:
  - È il metodo più direttamente pertinente: OSM in Calabria e Sicilia sotto bosco è sparso, cioè nel regime di bassa densità dove PLS dà i guadagni maggiori (recall di D-LinkNet: +0,171 su DG, da 0,565 a 0,736; +0,207 su CH4P, da 0,539 a 0,746).
  - Da adottare come baseline o ablation contro nnPU. Alla stessa estensione al suolo, a 1 m corrisponde **S ≈ 128 px, K = 16** su tile da circa 512–1024 px. Va rifatto lo sweep S ∈ {64, 128, 256} perché la scala del LiDAR e la larghezza dei sentieri sono diverse.
  - Va bene per la GPU da 12 GB: costo identico al training normale.
  - Rischio per l'obiettivo recall ≥ 0,98 contro OSM: PLS alza la recall a spese della precisione e misurarla solo contro OSM premia qualsiasi sovra-predizione. Serve un controllo della precisione: alcune tile rifinite a mano, oppure la stima PU della precisione descritta in fondo alla sezione nnPU.
  - Variante [mio] da provare: dentro le finestre applicare la banda "ignore" attorno alle centerline (vedi Wu e ScRoadExtractor) e trattare il resto della finestra come U con nnPU invece che come negativo certo.
  - Il dato "OSM positivi 94,2% corretti, recall 81,6%" in Cina rurale è un riferimento utile per stimare il prior di nnPU (vedi sotto). Da misurare però sul nostro pilota.

### Xu et al. 2023 — P2CNet: estrazione da immagini e mappe stradali parziali

- **Riferimento**: Qianxiong Xu, Cheng Long, Liang Yu, Chen Zhang, "Road Extraction with Satellite Images and Partial Road Maps", IEEE TGRS 61, 4501214 (2023), DOI 10.1109/TGRS.2023.3261332. Letto il preprint arXiv 2303.12394v1 (PDF).
- **Lettura**: completa per il preprint. Non letta la versione TGRS (paywall). L'appendice sulle strategie di fusione, citata nel testo, non è nel PDF arXiv.
- **Cosa fanno**: usano la mappa parziale (OSM incompleta) come secondo input. Rete a due rami (satellite e mappa parziale) fusi con il GSAM (Gated Self-Attention Module), più una loss "Missing Part" che pesa di più i pixel stradali assenti dalla mappa parziale.
- **Dati**:
  - SpaceNet: 2549 immagini 1300×1300 (Las Vegas, Parigi, Shanghai, Khartoum), mappe complete.
  - "OSM dataset" (da RoadTracer): immagini 1024×1024 commerciali non aperte su 40 città, con le mappe OSM trattate come complete.
  - Le **mappe parziali sono sintetiche**: si tolgono segmenti interi finché resta meno di una frazione α dei pixel stradali, con α ∈ {0, 25, 50, 75%}. Il set "mix" campiona a caso da 25, 50 e 75%. Le mappe OSM reali non sono usate come parziali.
  - Split 80/10/10, crop 1024. Esempio citato: a Shanghai OSM copre il 54,19% delle strade presenti in SpaceNet.
- **Metodo**:
  - Ciascun ramo è un DeepLabV3+ con ResNet-34 preaddestrato su ImageNet. L'uscita è quella del ramo parziale.
  - GSAM, attenzione per canale: A_X = softmax(F_X F_Xᵀ) di dimensione C×C, con X ∈ {S,P}; A_fuse = A_S ⊙ A_P; F′_X^j = β_X Σ_i a_fuse^{ji} F_X^i + F_X^j, con β_X apprendibile e inizializzato a 0.
  - GSAM, gate: concat(F′_S, F′_P), due blocchi conv 3×3 + BN + ReLU, softmax lungo una nuova dimensione, per ottenere F′_C ∈ R^{2×C×H×W} con F′_C[0] + F′_C[1] = 1; F_fuse = F′_S ⊙ F′_C[0] + F′_P ⊙ F′_C[1]. Ci sono anche 4 GSAM dopo gli ultimi 4 stadi dell'encoder del ramo parziale.
  - Loss:
    - L_BCE_X = −(1/N) Σ [I_C log M_X + (1 − I_C) log(1 − M_X)]. Nel PDF l'Eq. 6 contiene il refuso "(1 − log M)".
    - L_Dice_X = 1 − 2|M_X ∩ I_C| / (|M_X| + |I_C|)
    - **I_MP = I_C − I_P; L_MP_S = −(1/|Ω_MP|) Σ_{i∈Ω_MP} I_MP^i log M_S^i**, dove Ω_MP sono i pixel stradali presenti solo nella mappa completa.
    - **L_total = L_BCE_S + L_Dice_S + λ L_MP_S + L_BCE_P + L_Dice_P**: la MP si applica solo al ramo satellite.
  - La MP è una BCE e non una Dice: con poche centinaia di pixel mancanti la Dice è instabile.
  - Training: Adam, lr 2×10⁻⁴, "momentum 0,5", batch 4. La lr si moltiplica per 0,2 se la mIoU di validazione non sale per 5 epoche; stop dopo 20 epoche senza miglioramento. Convergenza entro circa 40 epoche. Augmentation: flip, shift, rumore gaussiano, luminosità.
  - 61,04 M parametri e 28,04 GMAC a 224×224 (DeepLabV3+: 22,44 M).
- **Risultati**:
  - SpaceNet mix (Tab. I): P2CNet P 0,8424, R 0,8150, F1 0,8285, APLS 0,7489, IoU **0,7071**, mIoU 0,8426. Il migliore dei baseline (DLinkNet, input fusion) ha IoU 0,7019.
  - OSM mix (Tab. II): P 0,8788, R 0,8430, F1 0,8605, APLS **0,8028**, IoU **0,7552**. LinkNet ha IoU 0,7427, McGAN APLS 0,7843.
  - Senza mappa parziale (α = 0%): IoU 0,5297 su SpaceNet (DeepLabV3+ 0,5277) e 0,5243 su OSM (DLinkNet 0,5240). Le mappe parziali portano l'IoU su OSM da circa 0,52 a circa 0,75.
  - Il ramo parziale supera di 18,60 (SpaceNet) e 23,05 (OSM) punti di IoU il ramo satellite.
  - Fusione (Tab. IV, IoU mix / 0%): input fusion 0,6990 / 0,4876; features fusion (ramo parziale) 0,7019 / 0,5024. Con l'input fusion il modello impara a copiare la mappa parziale e generalizza peggio quando non c'è.
  - Strategie MP (Tab. V, SpaceNet mix):

    | Strategia, λ | R | IoU |
    |---|---|---|
    | Both, 1 | 0,8241 | 0,7001 |
    | Both, 30 | 0,9179 | 0,5814 |
    | Par, 30 | 0,9164 | 0,6041 |
    | Sat, 1 | 0,8105 | 0,7027 |
    | Sat, 30 | 0,8108 | 0,7037 |

    Pesare molto le parti mancanti sul ramo d'uscita alza la recall sopra 0,9 ma fa crollare la precisione.
  - Ablation (Tab. VI, IoU): base 0,7019; + MP 0,7037; + GSAM gate 0,7045; + gate e attenzione 0,7071.
- **Limiti**:
  - Le mappe "complete" usate come target sono a loro volta OSM e SpaceNet, considerate complete.
  - L'incompletezza è simulata togliendo segmenti a caso: non c'è la correlazione reale con tipo di strada e copertura, né l'offset posizionale.
  - Guadagni di GSAM e MP molto piccoli (decimi di punto di IoU), nessuna varianza tra seed.
- **Per Morloquio**:
  - La MP loss richiede la mappa completa, quindi non si applica direttamente: da noi la "completa" non esiste.
  - Idea riusabile [mio]: un "modello di completamento" addestrato togliendo segmenti OSM (α ∈ {25, 50, 75%}) dall'input e usando come target l'OSM intero, con la MP sui segmenti tolti (λ fino a 30 sul ramo non d'uscita è risultato stabile).
  - Due avvertenze per quell'idea:
    - Il target resta OSM, quindi i sentieri mancanti sono ancora negativi: va combinato con PU o PLS.
    - La valutazione su tile tenute fuori deve misurare la recall solo sui segmenti tolti dall'input, altrimenti la recall ≥ 0,98 contro OSM diventa banale (il modello copia l'input).
  - Scartare per ora l'architettura a due rami (61 M parametri): costo alto per +0,5 punti di IoU. L'input fusion (OSM come canale) è più semplice ma, secondo la Tab. IV, impara a copiare.

### Wu et al. 2019 — Estrazione stradale con centerline OSM come etichetta debole

- **Riferimento**: Songbing Wu, Chun Du, Hao Chen, Yingxiao Xu, Ning Guo, Ning Jing, "Road Extraction from Very High Resolution Images Using Weakly labeled OpenStreetMap Centerline", ISPRS Int. J. Geo-Inf. 2019, 8(11), 478, DOI 10.3390/ijgi8110478. Identificato dal record DOAJ f6836de0ca59477c9434bb6c9f54e3d4. Pagina MDPI letta con crawl stealth (troncata a 40k) e poi il PDF completo (mdpi-res.com).
- **Lettura**: completa (PDF, 19 pp.).
- **Cosa fanno**: dalla centerline costruiscono un'etichetta a tre stati (strada vicino all'asse, non-strada lontano, ignoto in mezzo). Addestrano con loss parziale (BCE + Dice solo sui pixel noti) più una loss normalized cut sui pixel ignoti, su una ResUNet con un "ponte" di convoluzioni dilatate in parallelo (MD-ResUNet).
- **Dati**:
  - dataset1: Cheng et al., immagini commerciali non aperte a 1 m, 224 train / 30 test, centerline ricavate dalla verità a terra manuale (quindi perfettamente allineate).
  - dataset2: 315 immagini aeree immagini commerciali non aperte a 1,2 m ("Seat", probabilmente Seattle), 285 train / 30 test, **centerline OSM reali**, maschere di test disegnate a mano.
  - Etichetta parziale: strada entro 7 m dall'asse, non-strada oltre 50 m, ignoto tra 7 e 50 m.
  - Confronti: "expand" (asse dilatato alla larghezza media delle strade del training) e "full mask" (supervisione completa).
- **Metodo**:
  - Loss (Eq. 1): L = Σ_{p∈Ω_l} −log S_p^{y_p} + [1 − 2|Pred ∩ GT_known| / (|Pred · GT_known| + |GT_known|)] + λ Σ_k S_k′ W (1 − S_k) / (d′ S_k). Ω_l sono i pixel noti; d = W1.
  - Affinità: W_ij = exp(−(i−j)²/(2σ²_rgb) − ‖f(i)−f(j)‖²/(2σ²_xy)). Le etichette dei σ nel testo appaiono invertite: il termine "(i−j)" è descritto come spaziale ma porta σ_rgb.
  - Gradiente della normalized cut: ∂E_NC/∂S_k = S_k′ W S_k d / (d′S_k)² − 2 W S_k / (d′S_k). Calcolo veloce con permutohedral lattice.
  - Training in due fasi: prima solo la loss parziale per "partialiteration" iterazioni, poi si aggiunge λ·NC.
  - Adam, lr 0,0002 divisa per 5 se la loss non scende per 3 volte, batch 2, 2× GTX1080Ti, σ_rgb = 15, σ_xy = 100, **λ = 0,01**. Encoder ResNet34 preaddestrato su ImageNet, input 1024×1024; ponte con dilatazioni 1, 2, 4, 8 più skip, in parallelo.
- **Risultati** (F1 / "mIoU" / P; la recall non è riportata):

  | Supervisione | dataset1 | dataset2 (OSM reale) |
  |---|---|---|
  | Parziale + NC (MD-ResUNet) | **0,9194 / 0,8389 / 0,9651** | **0,8839 / 0,7678 / 0,8793** |
  | Solo parziale | 0,8901 / 0,7803 / 0,9287 | 0,8550 / 0,7100 / 0,8265 |
  | Expand | 0,8721 / 0,7441 / 0,9429 | 0,8329 / 0,6657 |
  | Full mask | 0,9298 / 0,8597 / 0,9794 | 0,8927 / 0,7854 / 0,9044 |

  - Sweep di λ (Tab. 7, dataset1, mIoU): 0,001 → 0,805; 0,005 → 0,815; **0,01 → 0,839**; 0,05 → 0,796; 0,1 → 0,792; 0,5 → 0,798; 1 → 0,801; 5 → 0,791.
  - Tab. 8, dataset2: λ = 0 → 0,6999; 1e-5 → 0,7298; 1e-4 → 0,7283; 1e-3 → 0,7139; **0,01 → 0,7329**; 0,1 → 0,7133; 0,5 → 0,7139. Questi valori non coincidono con lo 0,7678 della Tab. 6, presumibilmente per la diversa inizializzazione (modello già preaddestrato con la loss parziale).
- **Limiti**:
  - Dichiarati: meno accurato dove il bordo strada non è evidente.
  - [mio] La "mIoU" riportata vale esattamente 2·F1 − 1 in ogni riga (es. 2·0,87653086 − 1 = 0,75306173). Non è un'IoU indipendente, quindi non aggiunge informazione e non si confronta con le IoU di altri lavori.
  - [mio] Test di 30 immagini per dataset. Manca la recall. Una sola città con OSM reale.
  - [mio] Anche "non-strada oltre 50 m" è un'ipotesi sbagliata per OSM incompleto: i sentieri non mappati diventano negativi certi.
- **Per Morloquio**:
  - Adottare l'etichetta a tre stati con loss parziale: è il modo più semplice di assorbire l'offset di OSM.
  - A 1 m: positivo entro a1 ≈ 1–2 m dall'asse OSM (larghezza minima di un sentiero, meno l'offset atteso: per offset di 3–5 m conviene un nucleo stretto). Ignorare la fascia fino ad a2 ≈ offset massimo + larghezza massima (circa 10–15 m per le piste). **Oltre a2 i pixel vanno in U per nnPU, non a 0.**
  - La normalized cut o la CRF loss con kernel RGBXY (λ = 0,01) andrebbe adattata ai canali LiDAR (hillshade, pendenza, local relief al posto del colore). È plausibile, ma non testato da nessuno su LiDAR.
  - Non riusare la "mIoU" di questo articolo come riferimento.

### Meng, Di, Yang, Wang 2023 — Preaddestramento su larga scala con OSM come etichetta debole

- **Riferimento**: Shiqiao Meng, Zonglin Di, Siwei Yang, Yin Wang (Tongji University, UC San Diego), "Large-scale Weakly Supervised Learning for Road Extraction from Satellite Imagery", arXiv 2309.07823v1 (14/09/2023). La sede di pubblicazione non è verificata.
- **Lettura**: completa. I risultati sono quasi tutti in grafici (Fig. 7–10): ho letto i valori dalle etichette numeriche dei grafici sulle pagine renderizzate. Le etichette piccole della Fig. 8 sono approssimate a ±0,1.
- **Cosa fanno**: preaddestrano D-LinkNet e U-Net su un dataset fino a 100 volte DeepGlobe, con immagini zoom 18 (0,5 m) da oltre 200 paesi ed etichette OSM rasterizzate. Poi fanno fine-tuning su DeepGlobe e Massachusetts. Studiano l'effetto della scala, della densità stradale, del tipo di fine-tuning e della capacità del modello.
- **Dati**:
  - Tile 256×256 cucite in mosaici 4096×4096, crop casuali 512×512.
  - Solo strade pavimentate, larghezze di rendering per classe: 15 px (motorway, trunk e link), 10 px (primary, secondary, tertiary e link), 5 px per le altre.
  - Difetti di OSM documentati: offset, strade mancanti, larghezza insufficiente, etichette che attraversano edifici.
  - Densità: 100× ha 6,529×10¹¹ pixel con densità media 6,566%; l'intero raccolto ha 1,244×10¹³ pixel con densità 1,015%.
  - Target: DeepGlobe 5603 train / 623 val (densità 4,644% nel testo, 4,664% nella Tab. 1); Massachusetts 1107 / 49 (6,764%).
- **Metodo**:
  - Dice soft: 1 − (2|Y_pred ∩ Y_gt| + smooth) / (|Y_pred| + |Y_gt| + smooth), con smooth = 1.
  - Adam, input 512, batch 6 per GPU, lr scalata linearmente col batch: massima 1,6×10⁻⁴ su 8 GPU, minima 4×10⁻⁵ su 2 GPU. Divisione per 10 se la mIoU non sale per 6 epoche; stop sotto 10⁻⁷. Stesse impostazioni per preaddestramento e fine-tuning.
  - Modelli: U-Net profonda (39,50 M parametri); D-LinkNet con ResNet-18 (20,99 M), ResNet-34 (31,10 M) e ResNet-50 (217,65 M).
- **Risultati** (mIoU in %):
  - Scala, D-LinkNet34 con densità 6,563% (Fig. 7):

    | Preaddestramento | DeepGlobe | Mass |
    |---|---|---|
    | Baseline | 61,2 | 59,8 |
    | 1× | 61,8 | 59,9 |
    | 2× | 62,5 | 60,5 |
    | 5× | 63,5 | 61,1 |
    | 10× | 64,8 | 62,5 |
    | 20× | 65,7 | 63,9 |
    | 50× | 66,5 | 65,4 |
    | 100× | **67,2** | **66,6** |

  - Densità del preaddestramento (Fig. 8, assi a 4,2 / 6,0 / 11,2%; la didascalia riporta per errore 6,563%):
    - DeepGlobe da baseline a 10×: 61,2 → 62,4 (densità 4,2%), 63,1 (6,0%), 65,4 (11,2%).
    - Mass: 59,8 → 60,3, 60,6, 64,9.
    - Una densità di preaddestramento non superiore a quella del target aiuta poco.
  - Fine-tuning (Fig. 9, 1× → 100×):
    - Rete intera: DG 61,5 → 67,2; Mass 59,9 → 66,6.
    - Solo decoder: DG 51,0 → 64,5; Mass 57,1 → 62,1.
    - Solo gli ultimi 3 strati: DG 43,0 → 52,2, sempre sotto il baseline.
  - Capacità (Fig. 10, baseline → 100×):

    | Modello | DG | Mass |
    |---|---|---|
    | U-Net | 50,8 → 62,0 | 52,0 → 63,2 |
    | D-LinkNet18 | 59,8 → 65,6 | 58,3 → 65,1 |
    | D-LinkNet34 | 61,2 → 67,2 | 59,8 → 66,6 |
    | D-LinkNet50 | 62,3 → **68,5** | 60,7 → 65,8 (overfitting) |

  - Tab. 2: 68,5 contro 68,3 di HG-Orientation+Attn e 67,2 di Orientation.
- **Limiti**:
  - Valutano sulla validazione interna, non sul test ufficiale di DeepGlobe. La "mIoU" non è definita: probabilmente è la media tra strada e sfondo, mentre la classifica DeepGlobe usa l'IoU della strada, quindi il confronto in Tab. 2 è dubbio [mio].
  - Una sola esecuzione, nessuna analisi del rumore OSM in sé.
  - Usano dati di grandi dimensioni a lr diverse a seconda del numero di GPU, quindi il confronto non è perfettamente controllato.
- **Per Morloquio**:
  - Conferma che l'OSM rumoroso in grande quantità è un buon preaddestramento e che il guadagno cresce con il volume (log-lineare fino a 100×).
  - Proposta [mio]: preaddestrare l'encoder su tutto il LiDAR italiano aperto disponibile con etichette OSM (non solo il pilota), poi rifinire sul pilota con la loss robusta o PU. Sempre fine-tuning dell'intera rete.
  - Risultato da tenere presente: un preaddestramento a densità bassa (zone boscose) aiuta poco. Conviene sovracampionare tile con OSM denso, ma con il rischio di sbilanciarsi verso le strade asfaltate.
  - Scartare il rendering con larghezze fisse per classe a 0,5 m: con il LiDAR a 1 m la larghezza va decisa sulla morfologia (DTM), o lasciata alla banda ignore.

### Kamalu, Choi 2020 — Mappatura stradale in ambienti poveri di dati con OSM

- **Riferimento**: John Kamalu, Benjamin Choi (Stanford), "Road Mapping in Low Data Environments with OpenStreetMap", arXiv 2006.07993v1 (14/06/2020). È un report di progetto con Natural Capital Project e The Nature Conservancy.
- **Lettura**: completa (6 pp.).
- **Cosa fanno**: classificano il tipo di strada (major, minor, two-track) in immagini DigitalGlobe a 30 cm centrate su una strada OSM, in Kenya e Perù. Non è un lavoro di segmentazione.
- **Dati**:
  - Immagini 1000×1000 (300×300 m) centrate su un punto della polilinea OSM. Circa 170K in Kenya (rapporto delle classi 1:20:12) e circa 97K in Perù (13:20:10).
  - Rimozione delle nuvole con soglia 150 sulla media per banda. Crop e ridimensionamento a 224×224.
  - Maschere da OSM: Bresenham più un raggio di 20 px.
- **Metodo**: ResNet50V2 e poi Xception preaddestrati su ImageNet, cross-entropia categoriale, Adam lr 0,0001, sottoinsiemi bilanciati.
- **Risultati**:
  - Xception: Kenya accuratezza 0,80 e F1 0,73; Perù 0,67 e 0,66.
  - Mascheramento: contesto occultato 0,77 / 0,72; strada occultata 0,79 / 0,71; maschera al posto di un canale RGB 0,81 / 0,74. La classe dipende molto dal contesto.
  - Binarizzazioni: major contro two-track 0,92; minor contro two-track 0,70.
  - Trasferimento: da Kenya a Perù 0,46 di accuratezza bilanciata; da Perù a Kenya 0,60.
  - D-LinkNet preaddestrato su DeepGlobe, su 150 immagini, **non trova nemmeno il 50% delle strade major** e ancora meno minor e two-track.
- **Limiti**: report preliminare. Nessuna segmentazione addestrata (OSM non dà poligoni), metriche solo di classificazione, nessuna analisi statistica.
- **Per Morloquio**: rilevanza bassa. Due avvertenze utili:
  - Un modello stradale preaddestrato altrove crolla per cambio di dominio, quindi va addestrato sul nostro dominio (LiDAR italiano), e la validazione deve essere spaziale tra Calabria e Sicilia.
  - La distinzione "minor" contro "two-track" è debole anche per un umano. Se un giorno si classificasse il tipo di tracciato, attendersi molta confusione tra pista e sentiero.
  - Niente da adottare tecnicamente.

### Wei, Ji 2021 — ScRoadExtractor: supervisione debole da scribble o centerline e propagazione delle etichette

- **Riferimento**: Yao Wei, Shunping Ji (Wuhan University), "Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images", IEEE TGRS 60 (2022, online 2021), documento IEEE 9372390. Letto il preprint arXiv 2010.13106v1 (versione sottomessa). Codice: github.com/weiyao1996/ScRoadExtractor.
- **Lettura**: completa del preprint. Le Eq. 1–6 sono state estratte male: le ho verificate sulle pagine 3 e 5 renderizzate. Non letta la versione TGRS finale.
- **Cosa fanno**: dalle centerline costruiscono una "proposal mask" a tre stati (strada, non-strada, ignoto), unendo un buffer attorno all'asse e un graph cut su superpixel. Addestrano una rete a due rami (segmentazione più bordi HED) con BCE parziale, CRF loss sui pixel e una loss MSE sui bordi.
- **Dati**:
  - Cheng: aereo a 1,2 m, 132 km², 300/49 tile.
  - Wuhan: satellite a 0,5 m, 200 km², 1944/648.
  - DeepGlobe: 0,5 m, 1333 km², 15000/5333.
  - Tile 512×512. **Le centerline sono ottenute per scheletrizzazione della verità a terra**, quindi sono complete e allineate: niente rumore OSM reale. Ci sono anche scribble simulati (erosione con kernel a croce di dimensione 7, ancora (3,6), più scheletrizzazione) su Wuhan.
- **Metodo**:
  - Propagazione delle etichette:
    1. Buffer: strada entro a1, non-strada oltre a2, ignoto in mezzo.
    2. Graph cut su superpixel SLIC (circa 400 per tile 512², compattezza 20, vicinato di Delaunay) con istogrammi HSV (20 bin per H e S): E(x) = Σ_i ψ_i(x_i|Hist, Sc) + Σ_{i,j} ψ_ij(x_i,x_j|Hist). ψ_i vale 0 se il superpixel tocca lo scribble con la stessa classe, KLDiv(Hist_i, Hist_r) se non tocca nessuno scribble, ∞ altrimenti. ψ_ij è la KL tra superpixel vicini.
    3. Fusione: dove il graph cut dice strada ma il buffer dice non-strada, il pixel diventa ignoto; altrove vale il buffer.
  - Rete DBNet: encoder ResNet-34 (ImageNet), ASPP con dilatazioni 1/2/4 più image pooling, decoder di segmentazione con ConvT e ramo bordi che riusa le feature. Il target dei bordi viene da HED preaddestrato su BSDS500, senza fine-tuning.
  - Loss (verificate sulla pagina renderizzata):
    - **L_seg = PBCE(Y_p, S_p) + α R(S)**
    - **PBCE = −Σ_{p∈Ω_k} [Y_p log S_p + (1 − Y_p) log(1 − S_p)]**, solo sui pixel noti Ω_k
    - **R(S) = S′ W (1 − S)** (CRF loss densa con kernel gaussiano su RGBXY, bilateral filtering veloce), con ∂R/∂S = −2WS
    - L_bound = (1/(wh)) Σ_ij (T_ij − B_ij)², cioè la MSE rispetto alla maschera HED
    - **L = L_seg + β L_bound**
  - Iperparametri: bande gaussiane RGB 15 e XY 100; **α = 0,5; β = 0,7**. Adam, lr 2e-4 divisa per 5 se la loss non scende per 3 epoche, batch 2, tile 512. TTA a 8 trasformazioni, soglia 0,5. GPU GTX1060 da 6 GB. Numero di epoche non indicato.
  - Buffer finali: a1 = 6 m (Cheng), 2 m (Wuhan e DeepGlobe); a2 = 15 m (Cheng e DeepGlobe), 29 m (Wuhan). Nei dettagli implementativi però si legge "a1 = 6 m e a2 = 18 m per Cheng": incoerenza interna.
- **Risultati**:
  - Tab. II (P / R / F1 / IoU):

    | Metodo | Cheng | Wuhan | DeepGlobe |
    |---|---|---|---|
    | **ScRoadExtractor** | 0,9033 / 0,8423 / 0,8657 / **0,7651** | 0,6963 / 0,6904 / 0,6580 / **0,5158** | 0,7954 / 0,7138 / 0,7132 / **0,5782** |
    | WeaklyOSM (= Wu 2019) | IoU 0,7170 | IoU 0,4805 | IoU 0,5239 |
    | ScribbleSup | R 0,9730, IoU 0,5190 | — | R 0,8813, IoU 0,2694 |

  - Scribble simulati su Wuhan (Tab. III): ScRoadExtractor IoU 0,5001 (R 0,5680); WeaklyOSM 0,4307 (R 0,4687).
  - Tipo di supervisione (Tab. IV, IoU):

    | Supervisione | Cheng | Wuhan | DeepGlobe |
    |---|---|---|---|
    | Graph-based | 0,4972 | 0,4689 | 0,2614 |
    | Expanded (asse dilatato a larghezza fissa, es. 10 m) | 0,7089 | 0,3912 | 0,4678 |
    | Buffer-based | 0,6769 | 0,4848 | 0,5224 |
    | **Proposal** | **0,7396** | **0,5028** | **0,5422** |
    | Full mask | 0,8597 | 0,5648 | 0,6408 |

  - Ramo bordi (Tab. V, IoU): solo segmentazione 0,6729 / 0,4944 / 0,5070; con il ramo BRN di BPG 0,6233 / 0,4948 / 0,5269; con il ramo proposto 0,7396 / 0,5028 / 0,5422.
  - a2 ottimale vicino alla larghezza massima delle strade (Fig. 8).
- **Limiti**:
  - Dichiarati: il divario con la supervisione piena resta (DeepGlobe 0,5422 contro 0,6408 a parità di rete).
  - [mio] Centerline perfette dalla verità a terra: nessun test con omissioni o offset reali. Proprio il nostro problema non è coperto.
  - [mio] Il graph cut su istogrammi HSV e l'HED dipendono dall'RGB.
  - [mio] Nella Tab. II, Wuhan F1 0,6580 contro P 0,6963 e R 0,6904 è incoerente se l'F1 si calcolasse sulle medie: probabilmente è una media per tile.
- **Per Morloquio**:
  - Adottare la PBCE sui soli pixel noti (identica alla loss parziale di Wu) con buffer a tre stati. È il mattone base per assorbire l'offset OSM.
  - La regolarizzazione CRF R(S) = S′W(1−S) con α = 0,5 (bande da ritarare: "colore" = canali LiDAR normalizzati, XY = 100 px a 1 m è forse troppo largo per i sentieri) può propagare l'etichetta su tutta la sede stradale nel DTM.
  - Il ramo bordi HED è da scartare: HED preaddestrato su foto RGB non ha senso su hillshade e pendenza. Al massimo si potrebbero usare come bordi le rotture di pendenza del DTM [mio, non testato].
  - Il graph cut su superpixel colore è da scartare per il LiDAR, o da sostituire con superpixel su canali morfometrici. Nella Tab. IV il graph-based da solo è il peggiore.
  - Nota utile: ScribbleSup ha recall altissima (0,973) con precisione 0,527. Anche qui una recall alta da sola non vuol dire nulla.

### Kiryo, Niu, du Plessis, Sugiyama 2017 — nnPU: PU learning con stimatore di rischio non negativo

- **Riferimento**: R. Kiryo, G. Niu, M. C. du Plessis, M. Sugiyama, "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NIPS 2017, arXiv 1703.00593v2. Codice Chainer: github.com/kiryor/nnPUlearning.
- **Lettura**: completa per il testo principale e l'appendice A (esperimenti aggiuntivi). Dimostrazioni dell'appendice B solo scorse: non servono all'applicazione.
- **Cosa fanno**: il rischio PU non distorto (uPU) va sotto zero con modelli flessibili e sovra-adatta. Lo correggono forzando a ≥ 0 la parte stimata del rischio dei negativi.
- **Formule**:
  - Notazione: π_p = p(y = +1) è il prior di classe, che si suppone noto o stimato. R_p^+(g) = E_p[ℓ(g(X),+1)], R_p^-(g) = E_p[ℓ(g(X),−1)], R_u^-(g) = E_{X∼p(x)}[ℓ(g(X),−1)].
  - Siccome π_n p_n(x) = p(x) − π_p p_p(x), vale π_n R_n^- = R_u^- − π_p R_p^-.
  - uPU (Eq. 2): **R̂_pu(g) = π_p R̂_p^+(g) − π_p R̂_p^-(g) + R̂_u^-(g)**
  - nnPU (Eq. 6): **R̃_pu(g) = π_p R̂_p^+(g) + max{0, R̂_u^-(g) − π_p R̂_p^-(g)}**
  - Algoritmo 1, per mini-batch: r_i = R̂_u^-(g; X_u^i) − π_p R̂_p^-(g; X_p^i). Se r_i ≥ −β si discende il gradiente di R̂_pu. Altrimenti si discende il gradiente di −r_i, cioè ∇(π_p R̂_p^- − R̂_u^-), con passo γη. Vincoli: 0 ≤ β ≤ π_p sup_t max_y ℓ(t,y) e 0 ≤ γ ≤ 1. **Negli esperimenti β = 0, γ = 1.**
  - Loss usata negli esperimenti: sigmoide ℓ_sig(t,y) = 1/(1 + exp(ty)), con gradiente mai nullo. Tab. 1: anche logistic, ramp, squared, double hinge e così via sono compatibili. La logistica, cioè la BCE sui logit, soddisfa la condizione lineare-dispari ℓ(t,+1) − ℓ(t,−1) = −t, ma non è stata quella usata negli esperimenti.
  - Garanzie: il bias decade esponenzialmente; consistenza in O_p(π_p/√n_p + 1/√n_u); riduzione dell'MSE rispetto a uPU sotto le condizioni del Teorema 3.
- **Dati e risultati**:
  - MNIST (π_p = 0,49), epsilon (0,50), 20News (0,44), CIFAR-10 (0,40); n_p = 1000, U = tutto il training (P e U dipendenti, ammesso).
  - Modelli: MLP a 6 strati, CNN a 13 strati con BatchNorm, regolarizzazione ℓ2, Adam (AdaGrad su 20News).
  - Risultati solo in grafici: uPU sovra-adatta (il rischio di training va negativo e quello di test sale), nnPU no. nnPU batte PN con pochi N su MNIST, epsilon e CIFAR-10 ed è pari su 20News.
  - **Prior sbagliato** con π′ ∈ {0,8…1,2}π_p: sottostimare π fa più danni che sovrastimarlo. Il migliore è stato talvolta 1,1π (epsilon, 20News) o 1,2π (CIFAR-10). Gli autori lo spiegano con β = 0: un π più grande allontana nnPU da uPU.
- **Limiti**:
  - Prior supposto noto.
  - Ipotesi implicita che i positivi etichettati siano campionati da p_p(x), cioè rappresentativi di tutti i positivi (SCAR) [mio: in OSM non vale, perché le strade principali sono mappate molto più dei sentieri sotto bosco].
  - Esempi i.i.d.; nessun esperimento di segmentazione.
- **Per Morloquio, applicazione per pixel** [mio, non dall'articolo]:
  - g(x) è il logit della U-Net per pixel.
  - P = pixel nel nucleo stretto della centerline OSM (a1 ≈ 1–2 m).
  - U = tutti i pixel validi della tile, inclusi i P (formulazione case-control come negli esperimenti, P e U dipendenti). In alternativa U = solo i pixel non etichettati, ma allora va usato π_u = (π − π_L)/(1 − π_L), con π_L la frazione di pixel etichettati.
  - Banda ignore tra a1 e a2: va esclusa sia da P sia da U, oppure lasciata in U, da decidere con un'ablation.
  - Per batch: Rp_plus = mean_{P} softplus(−g); Rp_minus = mean_{P} softplus(g); Ru_minus = mean_{U} softplus(g). In alternativa la sigmoide come nel paper: ℓ(t,+1) = sigmoid(−t), ℓ(t,−1) = sigmoid(t).
  - neg = Ru_minus − π·Rp_minus. Loss = π·Rp_plus + neg se neg ≥ −β; altrimenti si ottimizza −neg (con γ = 1). Partire con β = 0.
  - Le medie vanno fatte sul batch intero, non per tile, per avere un r_i stabile con π piccolo.
  - Prior π: è la frazione di pixel di strada vera.
    - Stima pratica: π ≈ π_OSM / c, con π_OSM la densità dei pixel P e c la completezza di OSM.
    - In Cina rurale c = 0,816 (He et al. 2026). Sotto bosco in Calabria c sarà più bassa: va misurata su poche tile completate a mano.
    - π varia molto tra tile (bosco, coltivi, urbano): meglio un π per strato di copertura, oppure un π globale leggermente sovrastimato (1,1–1,2× come suggerisce il paper).
  - Valutazione:
    - Lo stesso stimatore nnPU, con loss zero-uno, si può usare per scegliere il modello su tile PU tenute fuori: lo dicono gli autori, che indicano l'uso per la valutazione del rischio con dati di validazione PU.
    - Identità standard PU sotto SCAR [mio]: precisione ≈ recall_OSM × π / P(ŷ = 1). Così la recall ≥ 0,98 contro OSM si accompagna a una precisione stimata e si evita che il modello raggiunga l'obiettivo predicendo strade ovunque.
    - Siccome SCAR è violata, la recall contro OSM sovrastima la recall sui sentieri forestali: va riportata stratificata per tag highway (track, path, footway e simili) e per copertura arborea (nDSM).
  - Raccomandazione d'insieme [mio]: nnPU sul nucleo P con U sul resto, più la banda ignore per l'offset (Wu e ScRoadExtractor), con DICE_SB o BCE_SB (Henry) sui soli pixel P come termine ausiliario, da confrontare in ablation con PLS (He et al.: S ≈ 128 px a 1 m, K = 16) sullo stesso split spaziale.

---

## Lettura incompleta o non riuscita

| Articolo | Stato | Motivo |
|---|---|---|
| Henry et al. 2020 | Articolo completo | Materiale supplementare (tabelle complete con P/R/Completeness per ogni esperimento, batch e dimensione delle patch) non trovato in open access, quindi non letto. Nessuna versione arXiv. |
| He et al. 2026 | Completo (PDF) | Il crawl MDPI era troncato a 40k e privo delle formule (MathML perso): integrato con il PDF mdpi-res.com. Valori della Fig. 6 (densità) non leggibili come numeri. |
| Xu et al. 2023 (P2CNet) | Preprint completo | Versione TGRS (DOI 10.1109/TGRS.2023.3261332) non letta: paywall. Appendice sulle strategie di fusione assente nel PDF arXiv. |
| Wu et al. 2019 | Completo (PDF) | Il crawl stealth MDPI funzionava ma era troncato a 40k: usato il PDF. Nessun blocco anti-bot. |
| Meng et al. 2023 | Completo | Valori numerici letti dai grafici renderizzati (Fig. 7–10). Quelli della Fig. 8 sono approssimati. Sede di pubblicazione non verificata. |
| Kamalu, Choi 2020 | Completo | Rilevanza bassa (classificazione, non segmentazione). |
| Wei, Ji (ScRoadExtractor) | Preprint completo | Formule verificate su pagina renderizzata. ar5iv non ha la versione HTML (reindirizza all'abstract). Versione TGRS non letta. |
| Kiryo et al. 2017 (nnPU) | Completo | Dimostrazioni (appendice B) solo scorse. |

Nessun sito è risultato bloccato da anti-bot. Nessuno strumento Firecrawl è stato usato.
