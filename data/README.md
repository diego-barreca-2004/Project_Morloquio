# Cartella dati

Tutto ciò che sta qui è ignorato da git, tranne questo file. La posizione si può cambiare con la variabile
d'ambiente `MORLOQUIO_DATA_DIR` (default: questa cartella). Gli output delle esecuzioni stanno in `runs/`
(variabile `MORLOQUIO_RUNS_DIR`).

| Cartella | Contenuto | Si può cancellare? |
|----------|-----------|--------------------|
| `raw/<source_id>/` | Download intatti di ogni fonte, più `manifest.json` (URL, sha256, dimensione, data, licenza) | Sì: si riscarica con `morloquio fetch` |
| `interim/<nome>/` | Estratti e ritagli (es. vie OSM dell'area pilota in GeoPackage) | Sì: si rigenera dai grezzi |
| `processed/cube/<tile_id>/` | Cubo dati per tile: DTM, DSM, ortofoto allineati, `meta.json` | Sì: si rigenera |
| `processed/terrain/<params_hash>/<tile_id>/` | Derivati del terreno per tile | Sì: si rigenera |
| `cache/` | Cache temporanee | Sì |

`<source_id>` è l'id della fonte in `configs/sources/` (es. `pst_lidar`, `osm_geofabrik_sud`).
`<tile_id>` è una cella di 0,01° come `N3807E01567` (38.07–38.08 N, 15.67–15.68 E).
