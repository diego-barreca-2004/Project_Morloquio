# 0002 — Ortofoto AGEA 2012: licenza non esplicita, fonte esclusa

Data: 2026-11-25 · Stato: accettata (in attesa di decisione del proprietario)

## Verifica

Servizio WMS `http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/WMS_v1.3/raster/ortofoto_colore_12.map`
(layer `OI.ORTOIMMAGINI.2012.33`, pixel 50 cm):

- GetCapabilities: `Fees: Nessuna condizione applicata`, `AccessConstraints: Nessuno`.
- Record di metadati RNDT/INSPIRE (CSW `http://www.pcn.minambiente.it/geoportal/csw`), id
  `m_amte:299FN3:eda6494c-b619-441a-a2d7-4097f8cde540` ("Ortofoto a colori AGEA periodo 2009-2012 - Regioni zona
  WGS84-UTM33"): `accessConstraints = otherRestrictions`, `useConstraints = otherRestrictions`,
  `otherConstraints = "Nessuna Limitazione al Pubblico Accesso"` e
  `"Questo servizio è ad accesso pubblico. Citare sempre la fonte del servizio."`
- Le note legali del Geoportale dichiarano CC BY 4.0 per i dati del *servizio di Download*; il WMS è un servizio di
  *visualizzazione* e non è coperto da quella frase.

## Decisione

Il testo garantisce l'accesso pubblico al servizio con citazione della fonte, ma non concede in modo esplicito
riuso, modifica e ridistribuzione dei dati derivati. Per la regola del progetto (licenza ambigua = fonte rifiutata)
l'ortofoto AGEA 2012 resta `unverified` e non viene scaricata. Il pilota F0–F2 lavora con il solo LiDAR (tier T1
senza ortofoto) più Sentinel-2.

## Da decidere

Il proprietario può decidere diversamente (è una sua scelta sul confine di "aperto"); in quel caso basta cambiare
`license` in `configs/sources/agea_ortho_2012.yaml` e implementare l'adapter WMS previsto dal piano (Task 10).
