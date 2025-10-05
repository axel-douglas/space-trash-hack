# Space Trash Hack — Consola Rex-AI (2025)

Demo interactiva de la plataforma Rex-AI para reciclar residuos orbitales y
marcianos. La aplicación combina inventarios NASA, catálogos de procesos y un
ensemble de modelos entrenados para sugerir recetas de reutilización, explicar
sus métricas y generar entregables operativos.

## Índice rápido

1. [Flujo interactivo de la app](#flujo-interactivo-de-la-app)
2. [Puesta en marcha](#puesta-en-marcha)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Datos y entrenamiento de IA](#datos-y-entrenamiento-de-ia)
5. [Feedback operativo y reentrenamiento](#feedback-operativo-y-reentrenamiento)
6. [Benchmarks y validaciones](#benchmarks-y-validaciones)
7. [Distribución de artefactos de modelo](#distribución-de-artefactos-de-modelo)
8. [Scripts útiles](#scripts-útiles)

---

## Flujo interactivo de la app

La consola guía a la tripulación a través de once pasos; cada pantalla mezcla
explicaciones para perfiles técnicos y operativos.

| Paso | Descripción | Página |
| --- | --- | --- |
| 1. **Home** | Resumen del inventario, salud del modelo y alertas críticas. | `app/Home.py` |
| 2. **Definir objetivo** | Configurá rigidez, estanqueidad y límites de recursos según el escenario (Residence Renovations, Cosmic Celebrations o Daring Discoveries). | `app/pages/2_Target_Designer.py` |
| 3. **Generador asistido** | Ejecutá la IA para proponer recetas, ajustá filtros y visualizá riesgo vs. recursos en tiempo real. | `app/pages/3_Generator.py` |
| 4. **Resultados y trade-offs** | Analizá la receta seleccionada con bandas de incertidumbre, comparativa heurística y trazabilidad NASA. | `app/pages/4_Results_and_Tradeoffs.py` |
| 5. **Compare & Explain** | Contrasta varias opciones, genera narrativa técnica y registra la decisión final. | `app/pages/5_Compare_and_Explain.py` |
| 6. **Pareto & Export** | Aplicá límites finales, identificá el frente Pareto y descargá planes JSON/CSV listos para operaciones. | `app/pages/6_Pareto_and_Export.py` |
| 7. **Scenario Playbooks** | Playbooks curados con filtros recomendados, métricas clave y checklist editable. | `app/pages/7_Scenario_Playbooks.py` |
| 8. **Feedback & Impact** | Capturá métricas reales, feedback cualitativo y prepará datos para reentrenamiento. | `app/pages/8_Feedback_and_Impact.py` |
| 9. **Spectral Mix Designer** | Calculá mezclas FTIR usando curvas de referencia NASA/UCF. | `app/pages/8_Spectral_Mix_Designer.py` |
| 10. **Capacity Simulator** | Proyectá producción por sol considerando turnos, downtime y límites diarios. | `app/pages/9_Capacity_Simulator.py` |
| 11. **Mars Control Center** | Consolida vuelos, inventario, decisiones IA y telemetría. | `app/pages/10_Mars_Control_Center.py` |
| 12. **Mission Planner** | Construí lotes, rutas logísticas y políticas de sustitución para la próxima misión. | `app/pages/11_Mission_Planner.py` |

Cada página comparte estilo y componentes reutilizables desde
`app/modules/ui_blocks.py`, de modo que la narrativa se mantenga consistente.

---

## Puesta en marcha

```bash
streamlit run app/Home.py
```

No se requieren variables de entorno para la demo, pero podés personalizar
rutas y artefactos con:

- `REXAI_DATA_ROOT`: reemplaza el directorio `data/` como raíz de datasets,
  logs y artefactos. Acepta rutas absolutas, relativas o con `~`.
- `REXAI_MODELS_DIR`: apunta explícitamente al directorio de modelos; por
  defecto usa `<DATA_ROOT>/models`.

### Bootstrap de entrypoints

Cada módulo de Streamlit debe invocar `ensure_streamlit_entrypoint(__file__)`
para inyectar la raíz del repositorio en `sys.path` y evitar errores al ejecutar
`streamlit run` directamente.

```python
from app.bootstrap import ensure_streamlit_entrypoint
PROJECT_ROOT = ensure_streamlit_entrypoint(__file__)
```

Para scripts y tests CLI utilizá `ensure_project_root(__file__)` como helper.

### Requisitos

- Python 3.10+
- `pip install -r requirements.txt`

---

## Estructura del proyecto

- `app/modules/data_sources.py`: resuelve rutas dentro de `datasets/`, normaliza
  taxonomías NASA y expone bundles cacheados.
- `app/modules/generator`: orquesta el generador de recetas (mezcla residuos,
  selecciona procesos y calcula features para inferencia).
- `app/modules/logging_utils.py`: serializa payloads y escribe logs Parquet.
- `app/modules/ml_models.py`: carga modelos entrenados, maneja el bootstrap
  sintético y expone `ModelRegistry` cacheado.
- `app/modules/navigation.py`: define el flujo multipaso utilizado por todas las
  páginas.
- `app/modules/ui_blocks.py`: layout y componentes visuales compartidos.

Tests unitarios se enfocan en estos límites para asegurar separación clara entre
ingesta, generación y logging.

### Semillas reproducibles

El generador acepta una semilla explícita para repetir sesiones completas:

- Campo **“Semilla (opcional)”** en la app Streamlit.
- CLI `python scripts/generate_candidates.py --seed 1234`.
- Variable `REXAI_GENERATOR_SEED=1234` antes de ejecutar cualquier entrypoint.

La semilla sincroniza RNG global y por tarea para que scores, combinaciones y
desempates del optimizador sean reproducibles.

---

## Datos y entrenamiento de IA

### Inventario de datasets

- `datasets/raw/nasa_waste_inventory.csv`: residuos no-metabólicos.
- `datasets/raw/mgs1_composition.csv`, `mgs1_oxides.csv`, `mgs1_properties.csv`:
  propiedades del regolito MGS-1.
- `datasets/raw/nasa_trash_to_gas.csv`: procesos Trash-to-Gas / Trash-to-Supply.
- `datasets/raw/logistics_to_living.csv`: eficiencia logística → habitabilidad.

### Pipeline base

1. **Validación de ingestas** — Ejecutá `python -m app.modules.data_pipeline`
   (o los tests asociados) para validar CSV/Parquet crudos. Los errores se
   registran en `data/logs/ingestion.errors.jsonl`.
2. **Construcción de dataset gold** — `python scripts/build_gold_dataset.py --output-dir data/gold`.
3. **Entrenamiento principal** — `python -m app.modules.model_training --gold data/gold --append-logs "data/logs/feedback_*.parquet"`.
4. **Modelos resultantes** — se guardan en `data/models/` (`rexai_regressor.joblib`,
   clasificadores auxiliares, ensembles opcionales) y metadata en `metadata.json`.

Si no hay modelos presentes, `ModelRegistry` entrena automáticamente un modelo
sintético liviano (`trained_on = "synthetic_v0_bootstrap"`) para evitar fallos en
la primera ejecución. Cuando se detectan artefactos reales (`metadata.json` con
`trained_on` válido) la app usa el ensemble entrenado y expone bandas de
incertidumbre, importancias de features y embeddings latentes.

### Artefactos generados

- `datasets/processed/rexai_training_dataset.parquet`
- `data/processed/rexai_training_dataset.parquet`
- `data/models/rexai_regressor.joblib`, `data/models/metadata.json`
- Ensambles opcionales: `rexai_xgboost.joblib`, `rexai_tabtransformer.pt`
- Autoencoder tabular opcional (`data/models/rexai_autoencoder.pt`)

---

## Feedback operativo y reentrenamiento

Cada sesión registrada en **Feedback & Impact** genera `data/logs/feedback_*.parquet`
con correcciones de rigidez, estanqueidad, energía, agua y crew. El comando
`python -m app.modules.retrain_from_feedback` convierte esas señales en targets
supervisados y reentrena el pipeline con `--append-logs` configurado.

Tras copiar nuevos artefactos recordá invalidar el cache de `ModelRegistry`:

```bash
python -c "from app.modules.ml_models import get_model_registry; get_model_registry.clear()"
```

Los módulos de impacto escriben automáticamente sus directorios (`LOGS_DIR.mkdir`)
por lo que no es necesario versionar `data/logs/`.

---

## Benchmarks y validaciones

El script `python scripts/run_benchmarks.py --format csv --with-ablation`
genera comparativas entre heurísticas y el modelo Rex-AI. Produce:

- `data/benchmarks/scenario_predictions.csv`
- `data/benchmarks/scenario_metrics.csv`
- Archivos de ablation (`ablation_predictions.csv`, `ablation_metrics.csv`)

Resumen de referencia (`data/benchmarks/scenario_metrics.csv`): MAE global
16 129, RMSE 34 056 y CI95 promedio 32 498 con `gold_v1`.

Para validar readiness antes de publicar ejecutá:

```bash
python -m scripts.verify_model_ready
```

El script asegura que `ModelRegistry.ready` sea `True` y que la metadata contenga
métricas, residuales y rutas consistentes.

---

## Distribución de artefactos de modelo

### Empaquetar y compartir

```bash
python -m scripts.package_model_bundle --output dist/rexai_model_bundle_gold_v1.zip
sha256sum dist/rexai_model_bundle_gold_v1.zip
```

El ZIP incluye todos los binarios (`joblib`, `pt`, `metadata.json`) y es ideal
para releases o artefactos de CI. Para reutilizarlo:

```bash
wget https://github.com/<org>/<repo>/releases/latest/download/rexai_model_bundle_gold_v1.zip
unzip rexai_model_bundle_gold_v1.zip -d /tmp/rexai-models
rsync -av /tmp/rexai-models/data/models/ data/models/
```

O simplemente `unzip dist/rexai_model_bundle_gold_v1.zip -d .` dentro del repo.
Luego corré `python -m scripts.verify_model_ready` para confirmar que la app
arrancará en modo IA.

### Descarga automática (secrets)

En despliegues remotos podés definir:

```toml
MODEL_BUNDLE_URL = "https://github.com/<org>/<repo>/releases/download/v1.0.0/rexai_model_bundle_gold_v1.zip"
MODEL_BUNDLE_SHA256 = "<hash sha256>"
```

Con esas claves `ModelRegistry` descargará, verificará y expandirá el bundle antes
de cargar los modelos.

### Mantener el bundle actualizado

1. Reentrená con los datasets más recientes.
2. Ejecutá `python -m scripts.verify_model_ready` y guardá el JSON de evidencia.
3. Empaquetá con `python -m scripts.package_model_bundle --output dist/rexai_model_bundle_<tag>.zip`.
4. Publicá el ZIP y actualizá `MODEL_BUNDLE_URL`/`MODEL_BUNDLE_SHA256`.

---

## Scripts útiles

- `python scripts/build_gold_dataset.py`: refresca el dataset curado.
- `python -m app.modules.model_training`: entrenamiento principal.
- `python -m scripts.generate_candidates --seed N`: genera candidatos determinísticos.
- `python scripts/run_benchmarks.py --with-ablation`: benchmarks completos.
- `python -m scripts.verify_model_ready`: chequeo de consistencia de modelos.
- `python -m scripts.package_model_bundle`: empaquetar artefactos entrenados.
- `python -m app.modules.retrain_from_feedback`: reentrenar incorporando feedback humano.

---

Para detalles adicionales consultá la documentación en `docs/` (onboarding,
setup de entorno, diseño de UX y guía de datos). Cada archivo fue actualizado
para que perfiles técnicos y no técnicos encuentren el contexto necesario.
