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
   `python -m app.modules.model_training --gold datasets/gold --append-logs "data/logs/feedback_*.parquet"`
   para producir el pipeline base y los metadatos indispensables en `data/models/`
   (por ejemplo `data/models/rexai_regressor.joblib` y `data/models/metadata.json`). El
   plan detallado de entrenamiento y variantes está descrito en
   [README_ML_GAMEPLAN.md](README_ML_GAMEPLAN.md).
3. **Confirmar el modo activo**. Si `data/models/rexai_regressor.joblib` no existe cuando se
   levanta la aplicación, `ModelRegistry` lanza un bootstrap automático que entrena un RandomForest
   sintético ligero y lo guarda en `data/models/`. Esto evita tener que versionar binarios para
   ejecutar la demo por primera vez. Mientras el bootstrap corre (o si fallara), la app recurre a
   las reglas `heuristic_props` para estimar rigidez, estanqueidad, energía, agua y minutos de crew.
   En cuanto el modelo y sus metadatos están presentes, `ModelRegistry.ready` habilita el modo IA y
   las predicciones pasan a provenir del ensemble entrenado (RandomForest + ensambles opcionales)
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

Para regenerar todos los artefactos (mezclando datasets simulados con el
corpus dorado y feedback humano cuando está disponible):

```bash
python -m app.modules.model_training --gold datasets/gold --append-logs "data/logs/feedback_*.parquet"
```

### Reentrenar desde feedback humano

Cada sesión de la tripulación registrada en el panel "Feedback & Impact"
genera archivos `data/logs/feedback_*.parquet` con las correcciones aplicadas
al proceso (rigidez, estanqueidad, penalizaciones de energía/agua/crew). El
comando dedicado `retrain_from_feedback` consume esos logs, convierte las
señales en targets supervisados y re-ejecuta el pipeline principal con
`--append-logs` ya configurado:

```bash
python -m app.modules.retrain_from_feedback
```

Opcionalmente podés especificar rutas alternativas:

```bash
python -m app.modules.retrain_from_feedback \
  --gold datasets/gold \
  --features datasets/gold/features.parquet \
  --logs "data/logs/custom_feedback_*.parquet"
```

Al finalizar, `data/models/metadata.json` y `metadata_gold.json` se actualizan
con la nueva fecha (`trained_at`) y el label de procedencia (`trained_on`, por
ejemplo `hil_v1` o `hybrid_v2`). La pantalla principal de la app refleja la
última fecha de reentrenamiento y la mezcla utilizada.

> Nota: la optimización bayesiana con Ax/BoTorch es opcional. El entorno Streamlit
> detecta automáticamente si `ax-platform` y `botorch` están instalados; en caso
> contrario utiliza el optimizador heurístico integrado. Para habilitarla basta con
> `pip install ax-platform botorch` antes de ejecutar la app.

Los binarios (`.joblib`, `.pt`, `.parquet`) permanecen ignorados por Git para
mantener el repo liviano. Cuando existen localmente, la app reemplaza las
predicciones heurísticas por las del modelo Rex-AI (RandomForest + XGBoost +
TabTransformer), expone bandas de confianza 95%, importancias promedio y el
vector latente entrenado sobre mezclas MGS-1 + residuos NASA. El bootstrap
automático genera un modelo sintético con la etiqueta `trained_on = "synthetic_v0_bootstrap"`,
que podés reemplazar en cualquier momento por artefactos reales empaquetados con
`python -m scripts.package_model_bundle --output dist/rexai_model_bundle_hybrid_v1.zip`.

## Distribución de artefactos ML

Los modelos entrenados se empaquetan automáticamente con:

```bash
python -m scripts.package_model_bundle --output dist/rexai_model_bundle_hybrid_v1.zip
```

El script genera un ZIP reproducible con todos los binarios
(`data/models/rexai_regressor.joblib`, clasificadores y ensambles opcionales) y
ambas versiones de metadata (`metadata.json` y `metadata_gold.json`). Ese
bundle debe subirse como Release Asset (o artifact de CI) bajo el nombre
`rexai_model_bundle_hybrid_v1.zip`.

Para reutilizarlo en otro entorno:

```bash
wget https://github.com/<org>/<repo>/releases/latest/download/rexai_model_bundle_hybrid_v1.zip
unzip rexai_model_bundle_hybrid_v1.zip -d /tmp/rexai-models
rsync -av /tmp/rexai-models/data/models/ data/models/
```

Reemplazá `<org>/<repo>` por la organización y el nombre reales del repositorio.

Colocar los archivos dentro de `data/models/` antes de ejecutar
`streamlit run app/Home.py` garantiza que la app arranque directamente en modo
IA (`ready=True`) sin depender del bootstrap sintético.

### Mantener el bundle actualizado

1. Reentrena con los datasets más recientes:

   ```bash
   python -m app.modules.model_training --gold datasets/gold --append-logs "data/logs/feedback_*.parquet"
   ```

   El pipeline actualizará `data/models/rexai_regressor.joblib`, clasificadores
   auxiliares, y escribirá `data/models/metadata.json` con un `trained_on`
   legible por `ModelRegistry.trained_label()` (por ejemplo `hybrid_v1`).

2. Empaqueta los artefactos con `python -m scripts.package_model_bundle --output dist/rexai_model_bundle_hybrid_v1.zip`,
   verifica con `python -m scripts.verify_model_ready` y publica el ZIP
   resultante.

3. Documenta la fecha y el label (`trained_on`) en el release/changelog para que
   los despliegues confirmen qué dataset alimentó el entrenamiento.

## Verificación automática de readiness

Antes de publicar un release ejecutar:

```bash
python -m scripts.verify_model_ready
```

El chequeo carga `ModelRegistry`, valida que `ready == True`, que todas las
métricas/residuales estén presentes en `metadata.json` y reporta las rutas de
artefactos generados. Si falta algún binario o metadata crítica, el script sale
con error para evitar releases inconsistentes.

## Benchmarks heurísticos vs IA

Para auditar la deriva entre las reglas `heuristic_props` y el modelo Rex-AI
podés ejecutar:

```bash
python scripts/run_benchmarks.py --format csv --with-ablation
```

El comando espera que `data/models/rexai_regressor.joblib` esté disponible y
genera tablas comparativas en `data/benchmarks/`. Además de los archivos
`scenario_predictions.csv`/`scenario_metrics.csv`, la flag `--with-ablation`
añade `ablation_predictions.csv` y `ablation_metrics.csv`, que documentan el
impacto de desactivar grupos de features (composición MGS-1, banderas NASA e
índices logísticos) durante la inferencia. Consulta [BENCHMARK.md](BENCHMARK.md)
para el resumen, la metodología y cómo interpretar los tres escenarios y los
resultados de ablation. En la corrida actual la IA supera a las heurísticas en
los tres casos de estudio, reduciendo el MAE global a 475 (vs reglas) y aportando
bandas de confianza medias de 630 unidades que permiten auditar la dispersión
de sus estimaciones.【F:data/benchmarks/scenario_metrics.csv†L2-L25】

Después de incorporar feedback humano, ejecuta el benchmark nuevamente apuntando
a un directorio distinto (`--output-dir data/benchmarks/post_feedback`) y
genera un comparativo con:

```bash
python scripts/plot_benchmark_deltas.py \
  --before data/benchmarks/pre_feedback/scenario_metrics.csv \
  --after data/benchmarks/post_feedback/scenario_metrics.csv
```

El HTML resultante (`data/benchmarks/feedback_deltas.html`) grafica cuánto se
redujeron (o empeoraron) MAE, RMSE y el ancho del CI95 frente al baseline
anterior, evidenciando el impacto directo del feedback en las métricas de la IA.

## Ejecutar la app

Con los artefactos generados, lanzar la demo con:

```bash
streamlit run app/Home.py
```

La UI detecta `ModelRegistry.ready` y, si los modelos están presentes, muestra
predicciones, bandas de confianza, importancia de features y comparaciones con
los ensambles opcionales. Ante ausencia de modelos utiliza los fallbacks
heurísticos originales.
