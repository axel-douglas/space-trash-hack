# Sprint 4 — Extensiones opcionales

Resumen de funcionalidades avanzadas que pueden activarse cuando se dispone de
más tiempo de cómputo o dependencias adicionales. Todas escriben artefactos en
`data/models/` y funcionan como capas opcionales sobre el pipeline base.

## 1. XGBoost por target

- Requisito: `pip install xgboost`.
- `app/modules/model_training.py` entrena un `XGBRegressor` por target y guarda el
  resultado en `data/models/rexai_xgboost.joblib`.
- Las métricas se integran en `data/models/metadata.json` (`artifacts.xgboost`).
- Ejemplo de ejecución:

```bash
python -m app.modules.model_training --samples 512 --seed 7
```

## 2. Autoencoder tabular

- Requisito: `pip install torch`.
- Genera `data/models/rexai_autoencoder.pt` y registra la configuración en la
  metadata (`artifacts.autoencoder`).
- `LatentSpaceExplorer` expone utilidades:
  - `detect_duplicates(df)` → detecta recetas casi idénticas en el espacio latente.
  - `propose_candidates(seed, objective)` → genera variantes alrededor de una
    receta semilla.
- Las funciones devuelven resultados vacíos si el autoencoder no está disponible,
  evitando errores en la UI.

## 3. Optimización en espacio latente

```bash
python scripts/optimize_latent.py datasets/generated/candidates.parquet \
    --objective "rigidez:1.0,crew_min:-0.3,energy_kwh:-0.1" \
    --samples 128 --radius 0.4 --top 15
```

- Amplía recetas prometedoras aplicando ruido gaussiano controlado.
- `--duplicates-threshold` ayuda a depurar candidatos similares antes de la
  validación humana.

## 4. Packaging de artefactos

- `python -m scripts.package_model_bundle --output dist/rexai_model_bundle_<tag>.zip`
  empaqueta modelos y metadata en un ZIP reproducible.
- Publicá el archivo generado como artefacto de release y configurá
  `MODEL_BUNDLE_URL` / `MODEL_BUNDLE_SHA256` para descargas automáticas.

Estas extensiones permiten experimentar sin versionar binarios en el repositorio
principal. Los tests (`tests/test_latent_optimizer.py`) garantizan que el flujo
sea determinista y sin duplicados.
