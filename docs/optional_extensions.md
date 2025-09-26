# Sprint 4 — Extensiones opcionales

Este sprint activa los componentes opcionales previstos en el plan técnico
sin almacenar binarios en la repo. Todos los artefactos se generan bajo
`data/models/` y se consumen a través de scripts reproducibles.

## Modelos XGBoost por target

* El pipeline de `app/modules/model_training.py` entrena un `XGBRegressor`
  independiente por target cuando `xgboost` está instalado.
* Los resultados se guardan en `data/models/rexai_xgboost.joblib` y su
  resumen de métricas queda accesible desde `data/models/metadata.json`
  (sección `artifacts.xgboost`).
* Para compararlo con el RandomForest basta con ejecutar:

  ```bash
  python -m app.modules.model_training --samples 512 --seed 7
  ```

  El log mostrará MAE/RMSE por target y la metadata consolida ambos
  resultados para dashboards o reportes.

## Autoencoder tabular opcional

* El entrenamiento opcional crea `data/models/rexai_autoencoder.pt`
  (no versionado). Su configuración queda registrada en la metadata
  (`artifacts.autoencoder`), permitiendo reproducir dimensiones latentes.
* `LatentSpaceExplorer` (nuevo módulo) expone:
  * `detect_duplicates(df)`: identifica recetas casi idénticas en el espacio
    latente.
  * `propose_candidates(seed, objective)`: genera variantes alrededor de una
    receta semilla aplicando ruido gaussiano controlado.
* Las funciones degradan a resultados vacíos si el autoencoder no está
  disponible, de modo que la UI y los scripts existentes no se rompan.

## Optimización en espacio latente

* `scripts/optimize_latent.py` toma un dataset de candidatos (Parquet/CSV/JSON)
  y expande automáticamente recetas prometedoras según un objetivo lineal
  configurable. Ejemplo:

  ```bash
  python scripts/optimize_latent.py datasets/generated/candidates.parquet \
      --objective "rigidez:1.0,crew_min:-0.3,energy_kwh:-0.1" \
      --samples 128 --radius 0.4 --top 15
  ```

* El script también puede detectar duplicados latentes (`--duplicates-threshold`)
  para depurar listas antes de enviarlas a validación humana.

## Packaging de artefactos

* `scripts/package_model_bundle.py` sigue disponible para empaquetar modelos y
  metadata en un ZIP reproducible sin versionar binarios dentro del repositorio.
* El workflow esperado es: entrenar → `package_model_bundle.py` → subir a un
  artefacto remoto e indicar la URL mediante `MODEL_BUNDLE_URL`.

Con estas extensiones, el equipo puede ejecutar experimentos avanzados y
mantener los binarios fuera de la repo, cumpliendo con los criterios del
Sprint 4. Las pruebas unitarias nuevas (`tests/test_latent_optimizer.py`)
verifican que las utilidades se comportan de forma determinista y evitan
duplicados al generar candidatos.
