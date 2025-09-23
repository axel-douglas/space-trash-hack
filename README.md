# Space Trash Hack — Streamlit Demo (2025)

Demo ligera que muestra la lógica del "cerebro de reciclaje" para Marte:
1) Inventario de residuos (NASA non-metabolic waste, simplificado)
2) Diseño de objetivo (TargetSpec)
3) Generación de recetas (combinaciones + proceso)
4) Resultados y trade-offs (Pareto, Sankey, métricas)

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

### Fase 0 — Verificaciones antes de entrenar

1. **Validar ingestas y revisar errores**. Ejecuta las utilidades de `app.modules.data_pipeline`
   (o los tests asociados) para verificar que los CSV/Parquet crudos cumplen los modelos de
   datos. Cada incidencia se registra en `data/logs/ingestion.errors.jsonl`; inspecciona ese
   archivo antes de continuar para corregir filas inválidas o faltantes.
2. **Generar artefactos mínimos**. Una vez limpia la ingesta, corre
   `python -m app.modules.model_training` para producir el pipeline base y los metadatos
   indispensables en `data/models/` (por ejemplo `data/models/rexai_regressor.joblib` y
   `data/models/metadata.json`). El plan detallado de entrenamiento y variantes está descrito
   en [README_ML_GAMEPLAN.md](README_ML_GAMEPLAN.md).
3. **Confirmar el modo activo**. La app opera en modo heurístico cuando falta
   `data/models/rexai_regressor.joblib` y, por tanto, recurre a las reglas `heuristic_props`
   para estimar rigidez, estanqueidad, energía, agua y minutos de crew. En cuanto el modelo y
   sus metadatos existen en `data/models/`, `ModelRegistry.ready` habilita el modo IA y las
   predicciones pasan a provenir del ensemble entrenado (RandomForest + ensambles opcionales)
   con intervalos de confianza y explicabilidad.

## Entrenar y generar artefactos de IA

El pipeline de entrenamiento consume datasets físicos/químicos alineados con
los documentos de NASA y UCF:

- `datasets/raw/nasa_waste_inventory.csv`: taxonomía de residuos no-metabólicos
  (pouches, espumas, EVA/CTB, textiles, nitrilo, etc.).
- `datasets/raw/mgs1_composition.csv` / `mgs1_oxides.csv` / `mgs1_properties.csv`:
  composición mineralógica y propiedades de MGS-1 (regolito de referencia).
- `datasets/raw/nasa_trash_to_gas.csv`: rendimiento de procesos Trash-to-Gas y
  Trash-to-Supply-Gas para Gateway/Mars.
- `datasets/raw/logistics_to_living.csv`: eficiencia de reuso de CTB (Logistics-2-Living).

`app/modules/model_training.py` construye combinaciones realistas mezclando esos
datasets con el catálogo de procesos y genera:

- Dataset procesado en `datasets/processed/rexai_training_dataset.parquet`.
- Pipeline RandomForest multi-salida (`data/models/rexai_regressor.joblib`) con
  estimación de incertidumbre (desvío entre árboles + residuales de validación).
- Ensemble de modelos de "wow effect":
  - `data/models/rexai_xgboost.joblib` (boosting por target).
  - `data/models/rexai_tabtransformer.pt` (transformer tabular ligero, opcional si PyTorch está disponible).
- Autoencoder para embeddings latentes (`data/models/rexai_autoencoder.pt`, opcional PyTorch).
- Metadatos en `data/models/metadata.json` (features, targets, fecha, métricas,
  importancias de features, residuales, paths de artefactos).

Para regenerar todos los artefactos:

```bash
python -m app.modules.model_training
```

> Nota: la optimización bayesiana con Ax/BoTorch es opcional. El entorno Streamlit
> detecta automáticamente si `ax-platform` y `botorch` están instalados; en caso
> contrario utiliza el optimizador heurístico integrado. Para habilitarla basta con
> `pip install ax-platform botorch` antes de ejecutar la app.

Los binarios (`.joblib`, `.pt`, `.parquet`) permanecen ignorados por Git para
mantener el repo liviano. Cuando existen localmente, la app reemplaza las
predicciones heurísticas por las del modelo Rex-AI (RandomForest + XGBoost +
TabTransformer), expone bandas de confianza 95%, importancias promedio y el
vector latente entrenado sobre mezclas MGS-1 + residuos NASA.

## Distribución de artefactos ML

Los modelos entrenados se empaquetan automáticamente con:

```bash
python -m scripts.package_model_bundle
```

El script genera `dist/rexai-models-<timestamp>.zip` con todos los binarios
(`rexai_regressor.joblib`, clasificadores, ensambles opcionales) y las dos
versiones de `metadata.json`. El ZIP está listo para adjuntarse a un GitHub
Release o subirlo a un bucket S3/GCS. En el despliegue, basta con descomprimirlo
en `data/models/` para activar el modo IA.

## Verificación automática de readiness

Antes de publicar un release ejecutar:

```bash
python -m scripts.verify_model_ready
```

El chequeo carga `ModelRegistry`, valida que `ready == True`, que todas las
métricas/residuales estén presentes en `metadata.json` y reporta las rutas de
artefactos generados. Si falta algún binario o metadata crítica, el script sale
con error para evitar releases inconsistentes.

## Ejecutar la app

Con los artefactos generados, lanzar la demo con:

```bash
streamlit run app/Home.py
```

La UI detecta `ModelRegistry.ready` y, si los modelos están presentes, muestra
predicciones, bandas de confianza, importancia de features y comparaciones con
los ensambles opcionales. Ante ausencia de modelos utiliza los fallbacks
heurísticos originales.
