# Space Trash Hack — Streamlit Demo (2025)

Demo ligera que muestra la lógica del "cerebro de reciclaje" para Marte:
1) Inventario de residuos (NASA non-metabolic waste, simplificado)
2) Diseño de objetivo (TargetSpec)
3) Generación de recetas (combinaciones + proceso)
4) Resultados y trade-offs (Pareto, Sankey, métricas)

## Ejecución

La entrada recomendada para lanzar la demo es:

```bash
streamlit run app/Home.py
```

No se requieren variables de entorno adicionales para el arranque interactivo.
Sin embargo, podés personalizar dónde se buscan los artefactos de datos y
modelos exportando las siguientes variables antes de ejecutar cualquier
entrypoint:

- `REXAI_DATA_ROOT`: raíz alternativa que reemplaza a `data/` como punto de
  partida para datasets curados, logs y archivos "gold". Se admite tanto rutas
  absolutas como relativas o con `~`; siempre se normalizan a un path absoluto.
- `REXAI_MODELS_DIR`: ubicación explícita para los bundles de modelos. Si no se
  define, por defecto se usa `<DATA_ROOT>/models`, aprovechando el valor
  resultante de `REXAI_DATA_ROOT` cuando está presente.

> ⚙️ **Bootstrap obligatorio:** antes de cualquier `import app.*`, cada entrypoint
> de Streamlit (incluyendo los módulos dentro de `app/pages/`) debe llamar a
> `ensure_streamlit_entrypoint(__file__)`:
>
> ```python
> from app.bootstrap import ensure_streamlit_entrypoint
>
> PROJECT_ROOT = ensure_streamlit_entrypoint(__file__)
> ```
>
> Esto fuerza la inclusión de la raíz del repositorio en `sys.path` cuando se
> ejecutan archivos sueltos con `streamlit run` o `python app/...`, devuelve el
> directorio que contiene `app/__init__.py` y evita los `ModuleNotFoundError`
> al importar `app.*`.
>
> 🧪 **Scripts y tests**: para utilidades CLI o fixtures de Pytest, usa
> `ensure_project_root(__file__)` en vez de manipular `sys.path` a mano:
>
> ```python
> from app.bootstrap import ensure_project_root
>
> PROJECT_ROOT = ensure_project_root(__file__)
> ```
>
> Así mantenemos un único helper oficial para exponer el paquete `app` sin
> reintroducir hacks circulares.

El script `app/Home.py` centraliza la vista de *Mission Overview* y actúa como
único entrypoint interactivo, manteniendo alineada la pantalla principal con el
paso "Overview" de la navegación multipaso.

## Modo Demo 3D en el Mars Control Center

La pestaña **"Modo Demo"** del panel marciano ahora incorpora una escena 3D
interactiva sincronizada con la simulación logística:

- El modelo GLB de Marte (`app/static/models/24881_Mars_1_6792.glb`) se expone
  como asset estático y se renderiza mediante un `ScenegraphLayer` de PyDeck.
- Cada cápsula orbital adopta una órbita animada cuya posición depende de su
  `eta_minutes`; la escala se incrementa a medida que se aproxima al nodo de
  destino.
- Eventos críticos emitidos por `advance_timeline()` o por el guion demo se
  resaltan automáticamente en ámbar/rojo, con trayectorias más gruesas para
  facilitar las presentaciones.

Para activar la vista, ejecutá la simulación desde la pestaña **Flight Radar**
y luego abrí la subsección *"Vista orbital 3D"* dentro de **Modo Demo**. La
escena se mantiene en sincronía con los ticks manuales o automáticos del
simulador y comparte los mismos datos de telemetría que impulsan el radar 2D.

## Módulos principales

La refactorización de 2025 separó responsabilidades clave para mantener el
código testeable y comprensible:

| Módulo | Responsabilidades |
| --- | --- |
| `app/modules/data_sources.py` | Resuelve rutas dentro de `datasets/`, normaliza taxonomías NASA y expone el bundle cacheado de features oficiales. |
| `app/modules/generator.py` | Mezcla residuos, selecciona procesos y construye candidatos junto con sus features listos para inferencia. |
| `app/modules/logging_utils.py` | Construye payloads JSON serializables y mantiene escritores Parquet rotados para los logs de inferencia. |

Los tests unitarios apuntan a estos límites para garantizar que cada módulo
permanezca enfocado en su dominio (ingesta, generación y logging). El módulo
`generator` simplemente importa `official_features_bundle()` desde
`data_sources` y re-exporta `append_inference_log()` desde `logging_utils`, de
modo que las pruebas pueden interceptar la ingesta o el logging sin tocar la
arquitectura de ensamblado de candidatos.

### Semillas reproducibles del generador

El ensamblador de recetas ahora acepta una semilla explícita para repetir una
sesión completa de generación. Podés fijarla de tres maneras equivalentes:

- Cargar la app Streamlit y completar el campo **"Semilla (opcional)"** antes
  de presionar "Generar recomendaciones".
- Ejecutar el CLI `python scripts/generate_candidates.py --seed 1234` para
  persistir los candidatos deterministas en `data/candidates.json`.
- Definir la variable de entorno `REXAI_GENERATOR_SEED=1234` antes de invocar
  cualquier entrypoint. El parámetro explícito siempre tiene prioridad sobre la
  variable de entorno.

La semilla inicializa tanto el RNG global como los RNG por tarea, de modo que
los candidatos producidos (scores, combinaciones de residuos y desempates del
optimizador) se mantengan iguales entre ejecuciones.

> Nota: los helpers legacy `app/modules/branding.py`, `app/modules/charts.py`
> y `app/modules/embeddings.py` fueron eliminados. La app se apoya en
> `app/modules/ui_blocks.py` para el layout y en los modelos entrenados para
> producir incrustaciones latentes reales.

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`

### Fase 0 — Verificaciones antes de entrenar

1. **Validar ingestas y revisar errores**. Ejecuta las utilidades de `app.modules.data_pipeline`
   (o los tests asociados) para verificar que los CSV/Parquet crudos cumplen los modelos de
   datos. Cada incidencia se registra en `data/logs/ingestion.errors.jsonl`; inspecciona ese
   archivo antes de continuar para corregir filas inválidas o faltantes.
2. **Generar artefactos mínimos**. Ejecutá
   `python scripts/build_gold_dataset.py --output-dir data/gold` para refrescar el
   dataset curado y, luego, `python -m app.modules.model_training --gold data/gold --append-logs "data/logs/feedback_*.parquet"`
   para producir el pipeline base y los metadatos indispensables en `data/models/`
   (por ejemplo `data/models/rexai_regressor.joblib` y `data/models/metadata.json`). El
   plan detallado de entrenamiento y variantes está descrito en
   [README_ML_GAMEPLAN.md](README_ML_GAMEPLAN.md).
3. **Confirmar el modo activo**. Si `data/models/rexai_regressor.joblib` no existe cuando se
   levanta la aplicación, `ModelRegistry` lanza un bootstrap automático que entrena un RandomForest
   sintético ligero y lo guarda en `data/models/`. Esto evita tener que versionar binarios para
   ejecutar la demo por primera vez. Mientras el bootstrap corre (o si fallara), la app recurre a
   las reglas `heuristic_props` para estimar rigidez, estanqueidad, energía, agua y minutos de crew.
   En cuanto el modelo y sus metadatos (`metadata.json`, clasificadores y `trained_on = "gold_v1"`)
   están presentes en `data/models/`, `ModelRegistry.ready` arranca en `True` y las predicciones
   pasan a provenir del ensemble entrenado (RandomForest + ensambles opcionales) con intervalos de
   confianza y explicabilidad verificables vía `python -m scripts.verify_model_ready`.

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
- Corridas sintéticas utilizadas para depurar el pipeline en
  `datasets/processed/ml/synthetic_runs.parquet`.
- Dataset procesado en `data/processed/rexai_training_dataset.parquet`.
- Pipeline RandomForest multi-salida (`data/models/rexai_regressor.joblib`) con
  estimación de incertidumbre (desvío entre árboles + residuales de validación).
- Ensemble de modelos de "wow effect":
  - `data/models/rexai_xgboost.joblib` (boosting por target).
  - `data/models/rexai_tabtransformer.pt` (transformer tabular ligero, opcional si PyTorch está disponible).
- Autoencoder tabular opcional (`data/models/rexai_autoencoder.pt`, requiere PyTorch). Las
  incrustaciones latentes expuestas en la app provienen exclusivamente de este
  autoencoder entrenado; ya no existe ningún helper heurístico para fabricarlas
  manualmente.
- Metadatos en `data/models/metadata.json` (features, targets, fecha, métricas,
  importancias de features, residuales, paths de artefactos).

Para regenerar todos los artefactos (mezclando datasets simulados con el
corpus dorado y feedback humano cuando está disponible):

```bash
python -m app.modules.model_training --gold data/gold --append-logs "data/logs/feedback_*.parquet"
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
  --gold data/gold \
  --features data/gold/features.parquet \
  --logs "data/logs/custom_feedback_*.parquet"
```

Al finalizar, `data/models/metadata.json` y `metadata_gold.json` se actualizan
con la nueva fecha (`trained_at`) y el label de procedencia (`trained_on`, por
ejemplo `hil_v1` o `hybrid_v2`). La pantalla principal de la app refleja la
última fecha de reentrenamiento y la mezcla utilizada.

Como la instancia de `ModelRegistry` queda cacheada vía `st.cache_resource`,
tras copiar artefactos nuevos sin reiniciar la app recordá invalidar esa caché.
Podés exponer un botón admin en Streamlit que llame a
`app.modules.ml_models.get_model_registry().clear()` o ejecutar manualmente:

```bash
python -c "from app.modules.ml_models import get_model_registry; get_model_registry.clear()"
```

Luego de limpiar la caché, refrescá la página para que la UI vuelva a cargar la
metadata y los pipelines recién entrenados.

> Nota: la optimización bayesiana con Ax/BoTorch es opcional. El entorno Streamlit
> detecta automáticamente si `ax-platform` y `botorch` están instalados; en caso
> contrario utiliza el optimizador heurístico integrado. Para habilitarla basta con
> `pip install ax-platform botorch` antes de ejecutar la app.

### Logs generados en runtime

Los módulos que escriben métricas o telemetría crean sus rutas dinámicamente en
runtime. Por ejemplo, `app/modules/impact.py` y `app/modules/logging_utils.py`
invocan `LOGS_DIR.mkdir(parents=True, exist_ok=True)` antes de guardar archivos
Parquet/JSON. Gracias a esto no es necesario versionar `data/logs/`; la carpeta
se materializa automáticamente cuando se ejecutan los tests o la app y se
elimina limpiando los artefactos generados. Para verificar este flujo, ejecutá:

```bash
pytest tests/test_impact_logging.py
```

La prueba crea un directorio temporal, persiste entradas de impacto y feedback
utilizando los módulos anteriores y confirma que los Parquet aparecen sin
requerir un placeholder en Git.

Los binarios (`.joblib`, `.pt`, `.parquet`) permanecen ignorados por Git para
mantener el repo liviano. Cuando existen localmente, la app reemplaza las
predicciones heurísticas por las del modelo Rex-AI (RandomForest + XGBoost +
TabTransformer), expone bandas de confianza 95%, importancias promedio y el
vector latente entrenado sobre mezclas MGS-1 + residuos NASA. El bootstrap
automático genera un modelo sintético con la etiqueta `trained_on = "synthetic_v0_bootstrap"`,
que podés reemplazar en cualquier momento por artefactos reales empaquetados con
`python -m scripts.package_model_bundle --output dist/rexai_model_bundle_gold_v1.zip` para
distribuir un bundle reproducible en el que `ModelRegistry.ready` queda en `True` desde el arranque.

## Distribución de artefactos ML

Los modelos entrenados se empaquetan automáticamente con:

```bash
python -m scripts.package_model_bundle --output dist/rexai_model_bundle_gold_v1.zip
```

El script genera un ZIP reproducible con todos los binarios
(`data/models/rexai_regressor.joblib`, clasificadores y ensambles opcionales) y
ambas versiones de metadata (`metadata.json` y `metadata_gold.json`). Ese
bundle debe subirse como Release Asset (o artifact de CI) bajo el nombre
`rexai_model_bundle_gold_v1.zip`.

Para reutilizarlo en otro entorno:

```bash
wget https://github.com/<org>/<repo>/releases/latest/download/rexai_model_bundle_gold_v1.zip
unzip rexai_model_bundle_gold_v1.zip -d /tmp/rexai-models
rsync -av /tmp/rexai-models/data/models/ data/models/
```

Reemplazá `<org>/<repo>` por la organización y el nombre reales del repositorio.

Colocar los archivos dentro de `data/models/` **antes** de ejecutar
`streamlit run app/Home.py` garantiza que la app arranque directamente en modo
IA (`ready=True`) sin depender del bootstrap sintético. El comando anterior deja
los archivos `.joblib` y `metadata.json` en el lugar correcto; si ya tenés el
ZIP generado localmente podés simplemente descomprimirlo dentro del repositorio:

```bash
unzip dist/rexai_model_bundle_gold_v1.zip -d .
```

Esto recreará la estructura `data/models/` con los artefactos actualizados.
Antes de lanzar Streamlit ejecutá `python -m scripts.verify_model_ready` para
confirmar que `ModelRegistry.ready` devuelve `True` y que la app usará el
ensemble entrenado desde el inicio.

### Actualizar el bundle publicado

1. **Entrenar**: `python -m app.modules.model_training --gold data/gold --append-logs "data/logs/feedback_*.parquet"`
   genera los `.joblib` y `metadata*.json` bajo `data/models/`.
2. **Verificar**: ejecutá `python -m scripts.verify_model_ready` y conservá el
   JSON resultante como bitácora del reentrenamiento.
3. **Empaquetar**: `python -m scripts.package_model_bundle --output dist/rexai_model_bundle_<tag>.zip`
   produce el ZIP reproducible con todos los artefactos.
4. **Publicar**: subí el ZIP como release asset (o al almacenamiento acordado) y
   registrá la URL final en `MODEL_BUNDLE_URL` junto con el hash en
   `MODEL_BUNDLE_SHA256` dentro de los secrets del despliegue. Actualizá la
   referencia cada vez que cambie `<tag>` para que la descarga automática apunte
   al bundle nuevo.

### Descarga automática desde secrets

Para despliegues donde no queremos subir los binarios al repositorio, la app
puede descargar el ZIP desde un release público/privado antes de levantar el
registro de modelos. Configurá los siguientes valores como variables de entorno
o en `st.secrets`:

```toml
# .streamlit/secrets.toml
MODEL_BUNDLE_URL = "https://github.com/<org>/<repo>/releases/download/v1.0.0/rexai_model_bundle_gold_v1.zip"
MODEL_BUNDLE_SHA256 = "<hash calculado con sha256sum>"
```

Con esas claves presentes, `ModelRegistry` descarga el bundle a un directorio
temporal, valida opcionalmente el hash y lo extrae automáticamente sobre
`data/models/` antes de cargar `joblib`. Si los artefactos ya existen en el
directorio destino, la descarga se omite.

Flujo recomendado para publicar nuevos modelos:

1. Ejecutá `python -m scripts.package_model_bundle --output dist/<bundle>.zip`.
2. Calculá el hash con `sha256sum dist/<bundle>.zip` (o `shasum -a 256`).
3. Cargá el ZIP como Release Asset en GitHub y copiá la URL de descarga directa
   (`https://github.com/<org>/<repo>/releases/download/...`).
4. Actualizá `MODEL_BUNDLE_URL` y `MODEL_BUNDLE_SHA256` en el entorno/secrets
   del despliegue (Streamlit Cloud, Hugging Face, etc.).

Así cada reinicio de la app asegura que `data/models/` contenga la versión
publicada sin ejecutar el bootstrap sintético.

### Mantener el bundle actualizado

1. Reentrena con los datasets más recientes:

   ```bash
   python -m app.modules.model_training --gold data/gold --append-logs "data/logs/feedback_*.parquet"
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
índices logísticos) durante la inferencia.

Los errores se calculan tratando las heurísticas como baseline. El resumen por
escenario (promediado sobre las cinco métricas) incorpora también el ancho
medio de los intervalos de confianza al 95% (`ci95_width_mean`):

| Escenario | MAE medio | RMSE | CI95 (ancho medio) |
|-----------|----------:|-----:|-------------------:|
| CTB Reconfig | 16 016 | 33 654 | 33 210 |
| Espuma + MGS-1 + Sinter | 16 090 | 34 298 | 31 687 |
| Multicapa + Laminar | 16 281 | 34 214 | 32 596 |
| Global (los 3 escenarios) | 16 129 | 34 056 | 32 498 |

Los valores anteriores provienen de `data/benchmarks/scenario_metrics.csv` y se
actualizan automáticamente al rerunear el script.【F:data/benchmarks/scenario_metrics.csv†L18-L25】

### Observaciones

* **Gap frente a las heurísticas**: el ensemble entrenado sobre `gold_v1` reduce
  el error medio global a 16 129 (RMSE 34 056), casi dos órdenes de magnitud por
  debajo del bootstrap sintético previo y alineado con las escalas de cada
  target.【F:data/benchmarks/scenario_metrics.csv†L18-L25】
* **Consumo de crew**: sigue siendo el más sensible; el MAE por escenario ronda
  4 177 minutos, lo que equivale a una guardia extendida pero ya no a desvíos del
  orden de cientos de miles.【F:data/benchmarks/scenario_metrics.csv†L18-L22】
* **Energía y agua**: los errores promedio se estabilizan en ~76 032 kWh y
  435 litros agregados, consistentes con la variabilidad del dataset dorado.
  Las bandas de confianza capturan estos márgenes (≈146 058 kWh y 937 L).【F:data/benchmarks/scenario_metrics.csv†L21-L24】
* **Rigidez/estanqueidad**: las discrepancias siguen acotadas (≤0.54), lo que
  facilita auditar calibraciones mecánicas sin perder precisión perceptiva.【F:data/benchmarks/scenario_predictions.csv†L2-L23】
* **CI95**: las bandas medias caen a ~32 498 unidades globales, con escenarios
  entre 31 687 y 33 210 gracias al reentrenamiento con el corpus dorado.【F:data/benchmarks/scenario_metrics.csv†L18-L25】

Consulta [BENCHMARK.md](BENCHMARK.md) para el resumen detallado, la metodología
y cómo interpretar los tres escenarios y los resultados de ablation. Con los
artefactos `gold_v1` cargados, el ensemble alcanza un MAE global de 16 129, RMSE
34 056 y bandas CI95 promedio de 32 498, demostrando una mejora drástica frente
al bootstrap heurístico original.【F:data/benchmarks/scenario_metrics.csv†L18-L25】
Los barridos con `--with-ablation` confirman ajustes pendientes: quitar las
banderas NASA o los índices logísticos apenas modifica el MAE y puede ensanchar
las bandas de confianza, por lo que conviene priorizar calibraciones finas en
lugar de apagar features completos.【F:data/benchmarks/ablation_metrics.csv†L71-L73】

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
