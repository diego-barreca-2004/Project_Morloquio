# 0003 — Licenze Copernicus (Sentinel-2, Copernicus DEM) ammesse come `Copernicus-open`

Data: 2026-11-25 · Stato: accettata

## Contesto

I cataloghi STAC marcano Sentinel-2 e Copernicus DEM come `license: proprietary`, perché non usano una licenza
Creative Commons ma licenze proprie dell'UE. Il progetto ammette solo dati aperti.

## Verifica

- **Sentinel**: la *Sentinel Data Legal Notice* (e la *Copernicus Sentinel data licence rev. 1*) prevede accesso
  "free, full and open" e consente uso lecito, riproduzione, distribuzione, comunicazione al pubblico, adattamento,
  modifica e combinazione con altri dati, per qualunque scopo, commerciale incluso. Obbligo: attribuzione
  "Copernicus Sentinel data [anno]" o, per dati modificati, "Contains modified Copernicus Sentinel data [anno]".
  Fonti: https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice ,
  https://ads.atmosphere.copernicus.eu/licences/ec-sentinel
- **Copernicus DEM GLO-30**: licenza gratuita; chi distribuisce deve citare la fonte con la dicitura prescritta; i
  dati adattati o modificati richiedono la dicitura "produced using Copernicus WorldDEM-30 © DLR e.V. …"; nessun
  diritto sui marchi. Fonte:
  https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf
  Il testo completo dell'articolo sui diritti concessi non è stato letto direttamente (download del PDF fallito
  il 2026-11-25): da rileggere prima di pubblicare dati derivati dal DEM.

## Decisione

Nuovo identificatore `Copernicus-open` nell'allowlist dei dati. Il DEM a 30 m non entra nei dati pubblicati in
F0–F2 (serve solo come contesto); prima del primo rilascio pubblico si rilegge il PDF della licenza DEM.
